"""JWM transformer layers — dual-tower MoT block with two-way flat attention
and AdaLN-zero sigma conditioning (DESIGN §3).

Hard invariant enforced by construction and verified by tests:
AR outputs NEVER depend on DM tokens (one-way AR -> DM information flow).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import JWMConfig
from .mathx import apply_rope, mrope_angles


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SigmaEmbedder(nn.Module):
    """sinusoidal(sigma * 1000) -> MLP. sigma in [0, 1]; clean tokens use sigma = 0."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:  # (..., ) -> (..., dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=sigma.device, dtype=torch.float32) / half
        )
        args = sigma.float().unsqueeze(-1) * 1000.0 * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class TowerAttention(nn.Module):
    """One tower's attention projections (separate params per tower — MoT)."""

    def __init__(self, cfg: JWMConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def qkv(self, x: torch.Tensor, angles: torch.Tensor):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)
        q = apply_rope(q, angles)
        k = apply_rope(k, angles)
        return q, k, v


def _sdpa(q, k, v, mask):
    """q,k,v: (B, S, H, Dh) -> (B, S, H*Dh). mask: bool (B, 1, Sq, Sk), True = attend."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, S, Dh)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class MoTBlock(nn.Module):
    """Dual-tower decoder block (DESIGN §3).

    Reasoner pathway : pre-norm attention + FFN, causal over AR tokens only.
    Generator pathway: AdaLN-zero modulated attention + FFN, bidirectional,
                       keys/values = [reasoner K,V of AR tokens ; generator K,V of DM tokens].
    """

    def __init__(self, cfg: JWMConfig, layer_idx: int = 0):
        super().__init__()
        d = cfg.d_model
        # reasoner tower — FFN is MoE when enabled (Inkling-mini), except the
        # first layer which stays dense (Inkling keeps early layers dense)
        self.r_norm1 = RMSNorm(d)
        self.r_attn = TowerAttention(cfg)
        self.r_norm2 = RMSNorm(d)
        use_moe = bool(getattr(cfg, "reasoner_moe", False)) and not (
            layer_idx == 0 and getattr(cfg, "moe_dense_first_layer", True))
        if use_moe:
            from .moe import MoEFFN
            eh = getattr(cfg, "moe_expert_hidden", 0) or d // 2
            self.r_ffn = MoEFFN(d, eh, cfg.moe_experts, cfg.moe_topk,
                                cfg.moe_shared, cfg.moe_aux_alpha)
        else:
            self.r_ffn = SwiGLU(d, cfg.ffn_hidden)
        # generator tower
        self.g_norm1 = RMSNorm(d)
        self.g_attn = TowerAttention(cfg)
        self.g_norm2 = RMSNorm(d)
        self.g_ffn = SwiGLU(d, cfg.ffn_hidden)
        # AdaLN-zero: per-token (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaln = nn.Linear(cfg.sigma_emb_dim, 6 * d)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward_ar(self, h_ar: torch.Tensor, ang_ar: torch.Tensor, ar_mask: torch.Tensor):
        """Reasoner pathway only. Returns (h_ar_next, k_ar, v_ar) — the detached,
        position-encoded K/V this layer exposes to the generator (cacheable at
        inference: computed ONCE, reused across every denoising step, Cosmos §5.3.1).

        detach(): generator gradients must not reshape reasoner representations,
        mirroring Cosmos' frozen-reasoner semantics; the reasoner learns only from
        the AR loss.
        """
        q, k, v = self.r_attn.qkv(self.r_norm1(h_ar), ang_ar)
        h_ar = h_ar + self.r_attn.wo(_sdpa(q, k, v, ar_mask))
        h_ar = h_ar + self.r_ffn(self.r_norm2(h_ar))
        return h_ar, k.detach(), v.detach()

    def forward_ar_step(self, h_new, ang_new, k_cache, v_cache, attn_mask):
        """Incremental reasoner step for KV-cached decoding (DESIGN §6 note).

        h_new: (B, 1, d) — the single new token. k_cache/v_cache: (B, S_past, H, Dh)
        already-roped K/V of every previous position. attn_mask: (B, 1, 1, S_past+1)
        bool over [past ; self]. Numerically identical to the last row of a full
        causal forward because earlier positions never attend to later ones.
        Returns (h_new_out, k_new, v_new) for cache append.
        """
        q, k, v = self.r_attn.qkv(self.r_norm1(h_new), ang_new)
        k_all = torch.cat([k_cache, k], dim=1)
        v_all = torch.cat([v_cache, v], dim=1)
        h_new = h_new + self.r_attn.wo(_sdpa(q, k_all, v_all, attn_mask))
        h_new = h_new + self.r_ffn(self.r_norm2(h_new))
        return h_new, k, v

    def forward_dm(self, h_dm, ka, va, ang_dm, sig_emb, dm_mask):
        """Generator pathway: AdaLN-zero modulated, joint attention over [AR; DM]."""
        sh1, sc1, g1, sh2, sc2, g2 = self.adaln(sig_emb).chunk(6, dim=-1)
        x = self.g_norm1(h_dm) * (1 + sc1) + sh1
        qd, kd, vd = self.g_attn.qkv(x, ang_dm)
        k_all = torch.cat([ka, kd], dim=1)
        v_all = torch.cat([va, vd], dim=1)
        h_dm = h_dm + g1 * self.g_attn.wo(_sdpa(qd, k_all, v_all, dm_mask))
        x2 = self.g_norm2(h_dm) * (1 + sc2) + sh2
        h_dm = h_dm + g2 * self.g_ffn(x2)
        return h_dm

    def forward(
        self,
        h_ar: torch.Tensor,          # (B, Sa, d)
        h_dm: torch.Tensor | None,   # (B, Sd, d) or None (pure language mode)
        ang_ar: torch.Tensor,        # (B, Sa, head_dim/2) MRoPE angles
        ang_dm: torch.Tensor | None,
        sig_emb: torch.Tensor | None,  # (B, Sd, sigma_emb_dim) per-token sigma embedding
        ar_mask: torch.Tensor,       # (B, 1, Sa, Sa) bool — causal & padding
        dm_mask: torch.Tensor | None,  # (B, 1, Sd, Sa+Sd) bool
    ):
        h_ar, ka, va = self.forward_ar(h_ar, ang_ar, ar_mask)
        if h_dm is None:
            return h_ar, None
        h_dm = self.forward_dm(h_dm, ka, va, ang_dm, sig_emb, dm_mask)
        return h_ar, h_dm


def build_ar_mask(valid: torch.Tensor) -> torch.Tensor:
    """Causal + padding mask for AR self-attention.

    valid: (B, Sa) bool. Returns (B, 1, Sa, Sa) bool, True = may attend.
    Padded QUERY rows keep the diagonal to avoid NaN softmax; their outputs are
    ignored by all losses.
    """
    B, S = valid.shape
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=valid.device))
    m = causal.unsqueeze(0) & valid.unsqueeze(1)          # (B, Sq, Sk) keys padded out
    eye = torch.eye(S, dtype=torch.bool, device=valid.device).unsqueeze(0)
    return (m | eye).unsqueeze(1)


def build_dm_mask(valid_ar: torch.Tensor, valid_dm: torch.Tensor) -> torch.Tensor:
    """Bidirectional mask for DM queries over [AR; DM] keys.

    valid_ar: (B, Sa), valid_dm: (B, Sd) -> (B, 1, Sd, Sa+Sd) bool.
    """
    B, Sd = valid_dm.shape
    keys = torch.cat([valid_ar, valid_dm], dim=1)          # (B, Sa+Sd)
    m = keys.unsqueeze(1).expand(B, Sd, keys.shape[1]).clone()
    # padded DM query rows: keep self position to avoid NaN
    Sa = valid_ar.shape[1]
    idx = torch.arange(Sd, device=valid_dm.device)
    m[:, idx, Sa + idx] = True
    return m.unsqueeze(1)
