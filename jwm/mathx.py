"""Core math for JWM — every function here maps 1-1 to an equation in DESIGN.md
and has a property test in tests/test_jwm_math.py.

Conventions: tensors are torch, float32 unless stated; bbox = (cx, cy, w, h) in [0,1].
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. MRoPE — 3D rotary position embedding with float coordinates (DESIGN §4)
# ----------------------------------------------------------------------------

def mrope_inv_freq(sections: tuple[int, int, int], base: float = 10000.0) -> list[torch.Tensor]:
    """Per-axis inverse frequencies. sections = half-dims for (t, h, w).

    Each axis gets its own geometric frequency ladder indexed from 0, exactly as if
    it were a standalone RoPE of that (half-)size: theta_i = base^(-i/section).
    """
    out = []
    for sec in sections:
        i = torch.arange(sec, dtype=torch.float32)
        out.append(base ** (-i / max(sec, 1)))
    return out


def mrope_angles(
    coords: torch.Tensor,  # (..., seq, 3) float coords (t, h, w)
    sections: tuple[int, int, int],
    base: float = 10000.0,
) -> torch.Tensor:
    """angles (..., seq, sum(sections)): concat of coord_axis * inv_freq_axis."""
    inv = mrope_inv_freq(sections, base)
    parts = [coords[..., k : k + 1] * inv[k].to(coords.device) for k in range(3)]
    return torch.cat(parts, dim=-1)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels by angles (rotate-half convention).

    x: (..., seq, n_heads, head_dim) with head_dim = 2 * angles.shape[-1]
    angles: (..., seq, head_dim/2) — broadcast over heads.
    """
    half = x.shape[-1] // 2
    cos = torch.cos(angles).unsqueeze(-2)  # (..., seq, 1, half)
    sin = torch.sin(angles).unsqueeze(-2)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def text_coords(n: int, start: float = 0.0) -> torch.Tensor:
    """Text tokens: t = h = w = monotone position (reduces MRoPE to 1D RoPE)."""
    p = torch.arange(n, dtype=torch.float32) + start
    return torch.stack([p, p, p], dim=-1)


def grid_coords(t: float, rows: int, cols: int) -> torch.Tensor:
    """Vision tokens of one frame: shared t, (h, w) tile the spatial grid row-major."""
    r = torch.arange(rows, dtype=torch.float32).repeat_interleave(cols)
    c = torch.arange(cols, dtype=torch.float32).repeat(rows)
    tt = torch.full((rows * cols,), float(t))
    return torch.stack([tt, r, c], dim=-1)


def temporal_delta(tps: float, tps_base: float) -> float:
    """Absolute temporal modulation (Cosmos eq. 9): dt = TPS_base / TPS."""
    return tps_base / tps


# ----------------------------------------------------------------------------
# 2. Rectified flow (DESIGN §5.2, §6)
# ----------------------------------------------------------------------------

def rf_interpolate(x0: torch.Tensor, eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """x_sigma = sigma * eps + (1 - sigma) * x0. sigma broadcast to x0's shape."""
    while sigma.dim() < x0.dim():
        sigma = sigma.unsqueeze(-1)
    return sigma * eps + (1.0 - sigma) * x0


def rf_velocity_target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """v* = eps - x0 (constant along the straight path)."""
    return eps - x0


def rf_x0_from_v(x_sigma: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """One-step estimate: x0 = x_sigma - sigma * v (exact when v = v*)."""
    while sigma.dim() < x_sigma.dim():
        sigma = sigma.unsqueeze(-1)
    return x_sigma - sigma * v


def shift_sigma(t: torch.Tensor, s: float) -> torch.Tensor:
    """Timestep-shift reparameterization: sigma = s*tbar / (1 + (s-1)*tbar), tbar = 1-t.

    Endpoints sigma(t=0)=1, sigma(t=1)=0; monotone decreasing in t; s>1 biases
    the schedule toward high noise.
    """
    tbar = 1.0 - t
    return (s * tbar) / (1.0 + (s - 1.0) * tbar)


def sigma_schedule(n_steps: int, s: float, device=None) -> torch.Tensor:
    """Descending sigma grid of length n_steps+1 from 1 -> 0 via shift reparam."""
    t = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    return shift_sigma(t, s)


def logit_normal_sigma(shape, mean: float = 0.0, std: float = 1.0, device=None) -> torch.Tensor:
    """sigma = sigmoid(z), z ~ N(mean, std^2) — SD3 logit-normal timestep sampling."""
    z = torch.randn(shape, device=device) * std + mean
    return torch.sigmoid(z)


def mode_sigma(shape, s: float = 1.29, device=None) -> torch.Tensor:
    """SD3 'mode' sampling: t = 1 - u - s*(cos^2(pi*u/2) - 1 + u), u ~ U[0,1].

    Returns values in [0,1] for s in [0, ~1.73]; s>0 concentrates mass mid-schedule
    with heavy tails (used by Cosmos for video batches).
    """
    u = torch.rand(shape, device=device)
    t = 1.0 - u - s * (torch.cos(math.pi / 2.0 * u) ** 2 - 1.0 + u)
    return t.clamp(0.0, 1.0)


@torch.no_grad()
def euler_flow_sample(
    v_fn,
    x_init: torch.Tensor,
    n_steps: int,
    s: float = 1.0,
    guidance: float = 1.0,
    v_uncond_fn=None,
) -> torch.Tensor:
    """Euler integration of dx/dsigma = v from sigma=1 to sigma=0.

    x_{sigma_{k+1}} = x_{sigma_k} - (sigma_k - sigma_{k+1}) * v_hat(x, sigma_k)
    With CFG: v_hat = v_u + g * (v_c - v_u).
    """
    sig = sigma_schedule(n_steps, s, device=x_init.device)
    x = x_init
    for k in range(n_steps):
        sk = sig[k].expand(x.shape[0])
        v = v_fn(x, sk)
        if guidance != 1.0 and v_uncond_fn is not None:
            vu = v_uncond_fn(x, sk)
            v = vu + guidance * (v - vu)
        x = x - (sig[k] - sig[k + 1]) * v
    return x


# ----------------------------------------------------------------------------
# 3. 6D rotation representation (DESIGN §1.4 note; Zhou et al. 2019)
# ----------------------------------------------------------------------------

def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt. Columns are the orthonormal frame."""
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): first two columns flattened."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """Nearest rotation to arbitrary 3x3 M via SVD: R = U diag(1,1,det(UV^T)) V^T."""
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U @ Vh)
    S = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    return U @ S @ Vh


# ----------------------------------------------------------------------------
# 4. bbox utilities + metrics (DESIGN §5.3, §10)
# ----------------------------------------------------------------------------

def bbox_to_signed(b: torch.Tensor) -> torch.Tensor:
    """(cx,cy,w,h) in [0,1] -> [-1,1] for flow matching."""
    return b * 2.0 - 1.0


def bbox_from_signed(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1], clamped."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def bbox_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of (cx,cy,w,h) boxes in [0,1]. a, b: (..., 4) -> (...)."""
    ax1, ay1 = a[..., 0] - a[..., 2] / 2, a[..., 1] - a[..., 3] / 2
    ax2, ay2 = a[..., 0] + a[..., 2] / 2, a[..., 1] + a[..., 3] / 2
    bx1, by1 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
    bx2, by2 = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    ix = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0)
    iy = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(min=0)
    inter = ix * iy
    union = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0) + \
            (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0) - inter
    return inter / union.clamp(min=1e-8)


def expected_calibration_error(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) * |acc_b - conf_b| over equal-width confidence bins."""
    conf = conf.float().flatten()
    correct = correct.float().flatten()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = conf.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += (m.float().sum() / n) * (correct[m].mean() - conf[m].mean()).abs()
    return float(ece)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR in dB per-sample over trailing dims. a, b: (B, ...)."""
    mse = ((a - b) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(max_val ** 2 / mse.clamp(min=1e-12))


def sqrt_len_normalize(loss_sum: torch.Tensor, n_tokens: torch.Tensor) -> torch.Tensor:
    """Cosmos reasoner loss weighting: per-sample token-loss-sum / sqrt(n_tokens)."""
    return loss_sum / n_tokens.clamp(min=1).float().sqrt()


def char_error_rate(pred: str, ref: str) -> float:
    """CER = levenshtein(pred, ref) / len(ref) — the standard OCR metric.

    Computed on unicode characters (not bytes) so one wrong Vietnamese
    diacritic counts as one error, not multiple byte errors.
    """
    a, b = list(pred), list(ref)
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(b)
