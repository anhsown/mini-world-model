"""JWM model — dual-tower omnimodal world model (DESIGN.md).

Modes (token arrangement, Cosmos §2.2):
  QA     : [AR: BOS IMG... BOQ q BOA a EOS]                      -> autoregressive answer
  GROUND : [AR: BOS IMG... BOQ q BOG][DM: noisy bbox]            -> denoise bbox + confidence
  FD     : [AR: BOS IMG... BOQ motion BOG][DM: clean z_t, noisy z_{t+1}] -> denoise next latent
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import JWMConfig
from . import tokenizer as tok
from .layers import MoTBlock, RMSNorm, SigmaEmbedder, build_ar_mask, build_dm_mask
from .mathx import (
    bbox_from_signed,
    bbox_iou,
    bbox_to_signed,
    grid_coords,
    logit_normal_sigma,
    mrope_angles,
    rf_interpolate,
    rf_velocity_target,
    rf_x0_from_v,
    sigma_schedule,
    sqrt_len_normalize,
)


# ============================================================================
# Conv autoencoder — frozen "VAE" for the generation pathway (DESIGN §1.3)
# ============================================================================

class ConvAE(nn.Module):
    def __init__(self, z_ch: int = 8):
        super().__init__()
        def gn(c):
            return nn.GroupNorm(min(8, c), c)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), gn(32), nn.SiLU(),      # 64 -> 32
            nn.Conv2d(32, 64, 4, 2, 1), gn(64), nn.SiLU(),     # 32 -> 16
            nn.Conv2d(64, z_ch, 4, 2, 1),                       # 16 -> 8
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(z_ch, 64, 4, 2, 1), gn(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), gn(32), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )
        self.register_buffer("z_mean", torch.zeros(z_ch))
        self.register_buffer("z_std", torch.ones(z_ch))

    def encode(self, img: torch.Tensor) -> torch.Tensor:      # (B,3,64,64) -> (B,z,8,8)
        return self.enc(img)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def encode_std(self, img: torch.Tensor) -> torch.Tensor:
        z = self.encode(img)
        return (z - self.z_mean[None, :, None, None]) / self.z_std[None, :, None, None]

    def decode_std(self, z_std: torch.Tensor) -> torch.Tensor:
        z = z_std * self.z_std[None, :, None, None] + self.z_mean[None, :, None, None]
        return self.decode(z)

    @torch.no_grad()
    def fit_stats(self, imgs: torch.Tensor) -> None:
        """Standardize latents to ~N(0,1) per channel — required for flow matching."""
        z = self.encode(imgs)
        self.z_mean.copy_(z.mean(dim=(0, 2, 3)))
        self.z_std.copy_(z.std(dim=(0, 2, 3)).clamp(min=1e-4))


def merge_latent(z: torch.Tensor, m: int = 2) -> torch.Tensor:
    """(B,C,H,W) -> (B, (H/m)*(W/m), C*m*m) — 2x2 patch-merge into DM tokens."""
    B, C, H, W = z.shape
    z = z.view(B, C, H // m, m, W // m, m).permute(0, 2, 4, 1, 3, 5)
    return z.reshape(B, (H // m) * (W // m), C * m * m)


def unmerge_latent(t: torch.Tensor, C: int, grid: int, m: int = 2) -> torch.Tensor:
    """Inverse of merge_latent: (B, grid*grid, C*m*m) -> (B, C, grid*m, grid*m)."""
    B = t.shape[0]
    z = t.view(B, grid, grid, C, m, m).permute(0, 3, 1, 4, 2, 5)
    return z.reshape(B, C, grid * m, grid * m)


# ============================================================================
# JWM
# ============================================================================

class JWM(nn.Module):
    def __init__(self, cfg: JWMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # --- embeddings ---
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.patch_embed = nn.Linear(cfg.patch * cfg.patch * 3, d)
        self.e_img = nn.Parameter(torch.zeros(d))       # modality embeddings (Cosmos §2.1)
        self.e_lat = nn.Parameter(torch.zeros(d))
        self.e_act = nn.Parameter(torch.zeros(d))
        self.e_boundary = nn.Parameter(torch.zeros(d))  # improvement #1 option
        nn.init.normal_(self.e_img, std=0.02)
        nn.init.normal_(self.e_lat, std=0.02)
        nn.init.normal_(self.e_act, std=0.02)

        # --- DM projections (domain-aware in/out, Cosmos §2.1.3) ---
        self.lat_in = nn.Linear(cfg.lat_tok_dim, d)
        self.lat_v_head = nn.Linear(d, cfg.lat_tok_dim)
        self.act_in = nn.Linear(4, d)
        self.act_v_head = nn.Linear(d, 4)
        self.conf_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))

        # --- towers ---
        self.sigma_embedder = SigmaEmbedder(cfg.sigma_emb_dim)
        self.blocks = nn.ModuleList([MoTBlock(cfg, i) for i in range(cfg.n_layers)])
        self.r_final = RMSNorm(d)
        self.g_final = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight       # weight tying

        self.apply(self._init)
        # re-zero AdaLN after generic init (zero-init is a hard requirement)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaln.weight)
            nn.init.zeros_(blk.adaln.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def init_generator_from_reasoner(self) -> None:
        """Cosmos §4 initialization: the generator tower starts as a WEIGHT COPY
        of the (pre-trained) reasoner tower, transferring semantic knowledge into
        the synthesis pathway. AdaLN stays zero-init, so right after the copy the
        generator is a gated-closed identical twin."""
        for blk in self.blocks:
            blk.g_norm1.load_state_dict(blk.r_norm1.state_dict())
            blk.g_attn.load_state_dict(blk.r_attn.state_dict())
            blk.g_norm2.load_state_dict(blk.r_norm2.state_dict())
            blk.g_ffn.load_state_dict(blk.r_ffn.state_dict())
            nn.init.zeros_(blk.adaln.weight)
            nn.init.zeros_(blk.adaln.bias)
        self.g_final.load_state_dict(self.r_final.state_dict())

    # ------------------------------------------------------------------
    # AR sequence assembly (fixed layout, DESIGN §2)
    #   0: BOS | 1..64: IMG | 65: BOQ | 66..66+Q-1: q | 66+Q: BOA/BOG | then answer
    # ------------------------------------------------------------------

    def _img_tokens(self, img: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        B = img.shape[0]
        p = c.patch
        g = c.img_grid
        x = img.view(B, 3, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, 3 * p * p)
        return self.patch_embed(x) + self.e_img

    def _ar_coords(self, S_text_after_img: int, device) -> torch.Tensor:
        """Coordinates for [BOS][IMG grid][text slots...] per DESIGN §4 table."""
        c = self.cfg
        parts = [torch.tensor([[0.0, 0.0, 0.0]], device=device)]              # BOS p=0
        parts.append(grid_coords(1.0, c.img_grid, c.img_grid).to(device))     # IMG t=1
        p0 = 2.0
        p = torch.arange(S_text_after_img, dtype=torch.float32, device=device) + p0
        parts.append(torch.stack([p, p, p], dim=-1))
        return torch.cat(parts, dim=0)                                        # (Sa, 3)

    def _build_ar(
        self,
        img: torch.Tensor,                # (B,3,64,64) in [0,1]
        q_ids: torch.Tensor,              # (B, Qmax) padded with PAD
        q_valid: torch.Tensor,            # (B, Qmax) bool
        trigger: int,                     # tok.BOA or tok.BOG
        a_ids: torch.Tensor | None = None,   # (B, Amax) padded (answer + EOS)
        a_valid: torch.Tensor | None = None,
    ):
        c, dev = self.cfg, img.device
        B, Qmax = q_ids.shape
        pieces = [
            self.tok_emb(torch.full((B, 1), tok.BOS, device=dev)),
            self._img_tokens(img),
            self.tok_emb(torch.full((B, 1), tok.BOQ, device=dev)),
            self.tok_emb(q_ids.clamp(max=c.vocab_size - 1)),
            self.tok_emb(torch.full((B, 1), trigger, device=dev)),
        ]
        valid = [
            torch.ones(B, 1, dtype=torch.bool, device=dev),
            torch.ones(B, c.n_img_tokens, dtype=torch.bool, device=dev),
            torch.ones(B, 1, dtype=torch.bool, device=dev),
            q_valid,
            torch.ones(B, 1, dtype=torch.bool, device=dev),
        ]
        if a_ids is not None:
            pieces.append(self.tok_emb(a_ids.clamp(max=c.vocab_size - 1)))
            valid.append(a_valid)
        emb = torch.cat(pieces, dim=1)
        valid = torch.cat(valid, dim=1)
        S_text = emb.shape[1] - 1 - c.n_img_tokens
        coords = self._ar_coords(S_text, dev).unsqueeze(0).expand(B, -1, -1)
        return emb, coords, valid

    def _build_ar_text(self, q_ids: torch.Tensor, q_valid: torch.Tensor, trigger: int):
        """Text-only AR sequence (T2I mode): [BOS][BOQ] caption [BOG]."""
        dev = q_ids.device
        B, Qmax = q_ids.shape
        c = self.cfg
        pieces = [
            self.tok_emb(torch.full((B, 1), tok.BOS, device=dev)),
            self.tok_emb(torch.full((B, 1), tok.BOQ, device=dev)),
            self.tok_emb(q_ids.clamp(max=c.vocab_size - 1)),
            self.tok_emb(torch.full((B, 1), trigger, device=dev)),
        ]
        valid = [
            torch.ones(B, 1, dtype=torch.bool, device=dev),
            torch.ones(B, 1, dtype=torch.bool, device=dev),
            q_valid,
            torch.ones(B, 1, dtype=torch.bool, device=dev),
        ]
        emb = torch.cat(pieces, dim=1)
        valid = torch.cat(valid, dim=1)
        S = emb.shape[1]
        p = torch.arange(S, dtype=torch.float32, device=dev)
        coords = torch.stack([p, p, p], dim=-1).unsqueeze(0).expand(B, -1, -1)
        return emb, coords, valid

    def _dm_coords_lat(self, B, dev):
        """Latent grid alone in the DM subsequence (T2I mode), at t = G."""
        c = self.cfg
        g = 0.0 if c.use_boundary_embedding else c.ar_dm_gap
        return grid_coords(g, c.lat_grid, c.lat_grid).to(dev).unsqueeze(0).expand(B, -1, -1)

    # ------------------------------------------------------------------
    # tower runner
    # ------------------------------------------------------------------

    def _run(
        self,
        ar_emb, ar_coords, ar_valid,
        dm_emb=None, dm_coords=None, dm_valid=None, dm_sigma=None,
    ):
        c = self.cfg
        ang_ar = mrope_angles(ar_coords, c.rope_sections, c.rope_base)
        ar_mask = build_ar_mask(ar_valid)
        if dm_emb is None:
            h_ar = ar_emb
            for blk in self.blocks:
                h_ar, _ = blk(h_ar, None, ang_ar, None, None, ar_mask, None)
            return self.r_final(h_ar), None
        ang_dm = mrope_angles(dm_coords, c.rope_sections, c.rope_base)
        sig_emb = self.sigma_embedder(dm_sigma)                       # (B,Sd,emb)
        dm_mask = build_dm_mask(ar_valid, dm_valid)
        h_ar, h_dm = ar_emb, dm_emb
        for blk in self.blocks:
            h_ar, h_dm = blk(h_ar, h_dm, ang_ar, ang_dm, sig_emb, ar_mask, dm_mask)
        return self.r_final(h_ar), self.g_final(h_dm)

    @torch.no_grad()
    def _precompute_ar(self, ar_emb, ar_coords, ar_valid):
        """Run the reasoner tower ONCE, caching per-layer (K_AR, V_AR) for reuse
        across all denoising steps (Cosmos §5.3.1 reasoner caching, DESIGN §6)."""
        c = self.cfg
        ang_ar = mrope_angles(ar_coords, c.rope_sections, c.rope_base)
        ar_mask = build_ar_mask(ar_valid)
        h_ar = ar_emb
        kv = []
        for blk in self.blocks:
            h_ar, k, v = blk.forward_ar(h_ar, ang_ar, ar_mask)
            kv.append((k, v))
        return self.r_final(h_ar), kv

    def _run_dm_cached(self, kv, ar_valid, dm_emb, dm_coords, dm_valid, dm_sigma):
        """Generator pathway only, consuming cached reasoner K/V. Numerically
        identical to _run's DM output (verified by test_cached_sampler_equivalence)."""
        c = self.cfg
        ang_dm = mrope_angles(dm_coords, c.rope_sections, c.rope_base)
        sig_emb = self.sigma_embedder(dm_sigma)
        dm_mask = build_dm_mask(ar_valid, dm_valid)
        h_dm = dm_emb
        for blk, (k, v) in zip(self.blocks, kv):
            h_dm = blk.forward_dm(h_dm, k, v, ang_dm, sig_emb, dm_mask)
        return self.g_final(h_dm)

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------

    def loss_qa(self, img, q_ids, q_valid, a_ids, a_valid):
        """AR cross-entropy on answer slots with sqrt-length normalization."""
        emb, coords, valid = self._build_ar(img, q_ids, q_valid, tok.BOA, a_ids, a_valid)
        h, _ = self._run(emb, coords, valid)
        logits = self.lm_head(h)
        B, S, V = logits.shape
        Amax = a_ids.shape[1]
        # answer slots occupy the last Amax positions; predictor is previous slot
        lg = logits[:, S - Amax - 1 : S - 1, :]
        ce = F.cross_entropy(lg.reshape(-1, V), a_ids.reshape(-1), reduction="none").view(B, Amax)
        ce = ce * a_valid.float()
        loss = sqrt_len_normalize(ce.sum(dim=1), a_valid.sum(dim=1)).mean()
        # MoE load-balancing aux (reasoner tower only trains through this loss)
        aux = 0.0
        for blk in self.blocks:
            la = getattr(blk.r_ffn, "last_aux_loss", None)
            if la is not None:
                loss = loss + la
                aux = aux + float(la)
        with torch.no_grad():
            acc = ((lg.argmax(-1) == a_ids) & a_valid).sum() / a_valid.sum().clamp(min=1)
        m = {"qa_ce": float(loss), "qa_tok_acc": float(acc)}
        if aux:
            m["moe_aux"] = aux
        return loss, m

    def _dm_coords_ground(self, B, dev):
        """GROUND DM layout: [clean image latent grid (t=G)] + [bbox token (t=G, h=w=0)]."""
        c = self.cfg
        g = 0.0 if c.use_boundary_embedding else c.ar_dm_gap
        lat = grid_coords(g, c.lat_grid, c.lat_grid).to(dev)
        act = torch.tensor([[g, 0.0, 0.0]], device=dev)
        return torch.cat([lat, act], dim=0).unsqueeze(0).expand(B, -1, -1)

    def _dm_coords_fd(self, B, dev):
        c = self.cfg
        g = 0.0 if c.use_boundary_embedding else c.ar_dm_gap
        cur = grid_coords(g, c.lat_grid, c.lat_grid).to(dev)
        nxt = grid_coords(g + c.fd_delta_t, c.lat_grid, c.lat_grid).to(dev)
        return torch.cat([cur, nxt], dim=0).unsqueeze(0).expand(B, -1, -1)

    def loss_ground(self, img, q_ids, q_valid, bbox_gt, z_img_tok):
        """Rectified-flow loss on bbox action token + calibrated confidence BCE.

        DM subsequence = [clean image latent tokens][noisy bbox] — clean visual
        conditioning inside DM, exactly like Cosmos I2V/inverse-dynamics modes,
        so the generator tower has direct (non-detached) spatial access.
        """
        c, dev = self.cfg, img.device
        B, N = z_img_tok.shape[0], z_img_tok.shape[1]
        x0 = bbox_to_signed(bbox_gt)                                   # (B,4)
        eps = torch.randn_like(x0)
        sig = logit_normal_sigma((B,), c.logit_normal_mean, c.logit_normal_std, dev)
        x_sig = rf_interpolate(x0, eps, sig)
        v_star = rf_velocity_target(x0, eps)

        emb_ar, coords_ar, valid_ar = self._build_ar(img, q_ids, q_valid, tok.BOG)
        lat = self.lat_in(z_img_tok) + self.e_lat                      # clean, sigma=0
        act = self.act_in(x_sig).unsqueeze(1) + self.e_act             # noisy
        dm_emb = torch.cat([lat, act], dim=1)
        if c.use_boundary_embedding:
            dm_emb = dm_emb + self.e_boundary
        dm_coords = self._dm_coords_ground(B, dev)
        dm_valid = torch.ones(B, N + 1, dtype=torch.bool, device=dev)
        dm_sigma = torch.cat([torch.zeros(B, N, device=dev), sig.unsqueeze(1)], dim=1)
        _, h_dm = self._run(emb_ar, coords_ar, valid_ar, dm_emb, dm_coords, dm_valid, dm_sigma)
        v_hat = self.act_v_head(h_dm[:, -1])                           # (B,4) bbox token only
        flow = F.mse_loss(v_hat, v_star)

        # confidence: does the one-step x0 estimate land within IoU >= 0.5?
        with torch.no_grad():
            x0_hat = rf_x0_from_v(x_sig, v_hat, sig)
            iou = bbox_iou(bbox_from_signed(x0_hat), bbox_gt)
            y = (iou >= 0.5).float()
        p_logit = self.conf_head(h_dm[:, -1]).squeeze(-1)
        conf = F.binary_cross_entropy_with_logits(p_logit, y)

        loss = c.lambda_bbox * flow + c.lambda_conf * conf
        return loss, {"bbox_flow": float(flow), "conf_bce": float(conf),
                      "iou_1step": float(iou.mean())}

    def loss_fd(self, img, q_ids, q_valid, z_cur_tok, z_next_tok):
        """Rectified-flow loss on next-frame latent tokens (clean current as condition)."""
        c, dev = self.cfg, img.device
        B, N, D = z_next_tok.shape
        eps = torch.randn_like(z_next_tok)
        sig = logit_normal_sigma((B,), c.logit_normal_mean, c.logit_normal_std, dev)
        x_sig = rf_interpolate(z_next_tok, eps, sig.unsqueeze(-1))
        v_star = rf_velocity_target(z_next_tok, eps)

        emb_ar, coords_ar, valid_ar = self._build_ar(img, q_ids, q_valid, tok.BOG)
        cur = self.lat_in(z_cur_tok) + self.e_lat
        nxt = self.lat_in(x_sig) + self.e_lat
        dm_emb = torch.cat([cur, nxt], dim=1)
        if c.use_boundary_embedding:
            dm_emb = dm_emb + self.e_boundary
        dm_coords = self._dm_coords_fd(B, dev)
        dm_valid = torch.ones(B, 2 * N, dtype=torch.bool, device=dev)
        dm_sigma = torch.cat([torch.zeros(B, N, device=dev),
                              sig.unsqueeze(1).expand(B, N)], dim=1)
        _, h_dm = self._run(emb_ar, coords_ar, valid_ar, dm_emb, dm_coords, dm_valid, dm_sigma)
        v_hat = self.lat_v_head(h_dm[:, N:])                           # only noisy tokens
        flow = F.mse_loss(v_hat, v_star)                               # clean masked out of loss
        return c.lambda_latent * flow, {"fd_flow": float(flow)}

    def loss_t2i(self, q_ids, q_valid, z_target_tok):
        """Text-to-Image: denoise the full latent grid from a caption alone
        (no conditioning image anywhere — AR is text-only, DM is all-noisy)."""
        c = self.cfg
        dev = q_ids.device
        B, N, D = z_target_tok.shape
        eps = torch.randn_like(z_target_tok)
        sig = logit_normal_sigma((B,), c.logit_normal_mean, c.logit_normal_std, dev)
        x_sig = rf_interpolate(z_target_tok, eps, sig.unsqueeze(-1))
        v_star = rf_velocity_target(z_target_tok, eps)

        emb_ar, coords_ar, valid_ar = self._build_ar_text(q_ids, q_valid, tok.BOG)
        dm_emb = self.lat_in(x_sig) + self.e_lat
        if c.use_boundary_embedding:
            dm_emb = dm_emb + self.e_boundary
        dm_coords = self._dm_coords_lat(B, dev)
        dm_valid = torch.ones(B, N, dtype=torch.bool, device=dev)
        dm_sigma = sig.unsqueeze(1).expand(B, N)
        _, h_dm = self._run(emb_ar, coords_ar, valid_ar, dm_emb, dm_coords, dm_valid, dm_sigma)
        v_hat = self.lat_v_head(h_dm)
        flow = F.mse_loss(v_hat, v_star)
        return c.lambda_latent * flow, {"t2i_flow": float(flow)}

    @torch.no_grad()
    def sample_image(self, q_ids, q_valid, steps=None, shift=None, guidance: float = 1.0):
        """T2I sampling: caption -> latent tokens (decode with the ConvAE outside)."""
        c, dev = self.cfg, q_ids.device
        B = q_ids.shape[0]
        N, D = c.n_lat_tokens, c.lat_tok_dim
        steps = steps or c.sample_steps
        shift = shift or c.sample_shift
        emb_ar, coords_ar, valid_ar = self._build_ar_text(q_ids, q_valid, tok.BOG)
        _, kv_c = self._precompute_ar(emb_ar, coords_ar, valid_ar)
        if guidance != 1.0:
            q_pad = torch.full_like(q_ids, tok.PAD)
            q_nv = torch.zeros_like(q_valid)
            emb_u, coords_u, valid_u = self._build_ar_text(q_pad, q_nv, tok.BOG)
            _, kv_u = self._precompute_ar(emb_u, coords_u, valid_u)
        dm_coords = self._dm_coords_lat(B, dev)
        dm_valid = torch.ones(B, N, dtype=torch.bool, device=dev)

        def v_of(kv, avalid, x, s):
            dm_emb = self.lat_in(x) + self.e_lat
            if c.use_boundary_embedding:
                dm_emb = dm_emb + self.e_boundary
            h_dm = self._run_dm_cached(kv, avalid, dm_emb, dm_coords, dm_valid,
                                       s.unsqueeze(1).expand(B, N))
            return self.lat_v_head(h_dm)

        sig = sigma_schedule(steps, shift, device=dev)
        x = torch.randn(B, N, D, device=dev)
        for k in range(steps):
            s = sig[k].expand(B)
            v = v_of(kv_c, valid_ar, x, s)
            if guidance != 1.0:
                vu = v_of(kv_u, valid_u, x, s)
                v = vu + guidance * (v - vu)
            x = x - (sig[k] - sig[k + 1]) * v
        return x

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_answer(self, img, q_ids, q_valid, max_new: int | None = None) -> list[str]:
        c, dev = self.cfg, img.device
        B = img.shape[0]
        max_new = max_new or c.max_a_bytes
        out = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        a_ids = torch.full((B, 0), tok.PAD, dtype=torch.long, device=dev)
        for _ in range(max_new):
            a_valid = torch.ones_like(a_ids, dtype=torch.bool)
            emb, coords, valid = self._build_ar(img, q_ids, q_valid, tok.BOA,
                                                a_ids if a_ids.shape[1] else None,
                                                a_valid if a_ids.shape[1] else None)
            h, _ = self._run(emb, coords, valid)
            nxt = self.lm_head(h[:, -1]).argmax(-1)                    # (B,)
            nxt = torch.where(done, torch.full_like(nxt, tok.PAD), nxt)
            a_ids = torch.cat([a_ids, nxt.unsqueeze(1)], dim=1)
            for b in range(B):
                if not done[b] and int(nxt[b]) not in (tok.EOS, tok.PAD):
                    out[b].append(int(nxt[b]))
            done = done | (nxt == tok.EOS)
            if bool(done.all()):
                break
        return [tok.decode(o) for o in out]

    @torch.no_grad()
    def sample_bbox(self, img, q_ids, q_valid, z_img_tok, steps=None, shift=None, guidance=1.0):
        """Denoise bbox from pure noise. Returns (bbox in [0,1], calibrated confidence)."""
        c, dev = self.cfg, img.device
        B, N = z_img_tok.shape[0], z_img_tok.shape[1]
        steps = steps or c.sample_steps
        shift = shift or c.sample_shift
        emb_ar, coords_ar, valid_ar = self._build_ar(img, q_ids, q_valid, tok.BOG)
        if guidance != 1.0:
            q_pad = torch.full_like(q_ids, tok.PAD)
            q_nv = torch.zeros_like(q_valid)
            emb_u, coords_u, valid_u = self._build_ar(img, q_pad, q_nv, tok.BOG)

        dm_coords = self._dm_coords_ground(B, dev)
        dm_valid = torch.ones(B, N + 1, dtype=torch.bool, device=dev)
        lat = self.lat_in(z_img_tok) + self.e_lat

        # reasoner K/V computed ONCE, reused every denoising step (Cosmos §5.3.1)
        _, kv_c = self._precompute_ar(emb_ar, coords_ar, valid_ar)
        if guidance != 1.0:
            _, kv_u = self._precompute_ar(emb_u, coords_u, valid_u)

        def v_of(kv, avalid, x, s):
            act = self.act_in(x).unsqueeze(1) + self.e_act
            dm_emb = torch.cat([lat, act], dim=1)
            if c.use_boundary_embedding:
                dm_emb = dm_emb + self.e_boundary
            dm_sigma = torch.cat([torch.zeros(B, N, device=dev), s.unsqueeze(1)], dim=1)
            h_dm = self._run_dm_cached(kv, avalid, dm_emb, dm_coords, dm_valid, dm_sigma)
            return self.act_v_head(h_dm[:, -1]), h_dm

        sig = sigma_schedule(steps, shift, device=dev)
        x = torch.randn(B, 4, device=dev)
        for k in range(steps):
            s = sig[k].expand(B)
            v, _ = v_of(kv_c, valid_ar, x, s)
            if guidance != 1.0:
                vu, _ = v_of(kv_u, valid_u, x, s)
                v = vu + guidance * (v - vu)
            x = x - (sig[k] - sig[k + 1]) * v
        # confidence read-out at final (near-clean) state, sigma ~ 0
        _, h_fin = v_of(kv_c, valid_ar, x, torch.full((B,), 0.02, device=dev))
        conf = torch.sigmoid(self.conf_head(h_fin[:, -1]).squeeze(-1))
        return bbox_from_signed(x), conf

    @torch.no_grad()
    def sample_next_latent(self, img, q_ids, q_valid, z_cur_tok, steps=None, shift=None,
                           guidance: float = 1.0):
        """Denoise next-frame latent tokens. Returns (B, N, D) standardized-merged tokens.

        guidance != 1.0 enables classifier-free guidance on the motion text
        (text-dropout branch trained in ModeBatcher, DESIGN §5.2/§6)."""
        c, dev = self.cfg, img.device
        B, N, D = z_cur_tok.shape
        steps = steps or c.sample_steps
        shift = shift or c.sample_shift
        emb_ar, coords_ar, valid_ar = self._build_ar(img, q_ids, q_valid, tok.BOG)
        _, kv_c = self._precompute_ar(emb_ar, coords_ar, valid_ar)
        if guidance != 1.0:
            q_pad = torch.full_like(q_ids, tok.PAD)
            q_nv = torch.zeros_like(q_valid)
            emb_u, coords_u, valid_u = self._build_ar(img, q_pad, q_nv, tok.BOG)
            _, kv_u = self._precompute_ar(emb_u, coords_u, valid_u)
        dm_coords = self._dm_coords_fd(B, dev)
        dm_valid = torch.ones(B, 2 * N, dtype=torch.bool, device=dev)
        cur = self.lat_in(z_cur_tok) + self.e_lat

        def v_of(kv, avalid, x, s):
            nxt = self.lat_in(x) + self.e_lat
            dm_emb = torch.cat([cur, nxt], dim=1)
            if c.use_boundary_embedding:
                dm_emb = dm_emb + self.e_boundary
            dm_sigma = torch.cat([torch.zeros(B, N, device=dev),
                                  s.unsqueeze(1).expand(B, N)], dim=1)
            h_dm = self._run_dm_cached(kv, avalid, dm_emb, dm_coords, dm_valid, dm_sigma)
            return self.lat_v_head(h_dm[:, N:])

        sig = sigma_schedule(steps, shift, device=dev)
        x = torch.randn(B, N, D, device=dev)
        for k in range(steps):
            s = sig[k].expand(B)
            v = v_of(kv_c, valid_ar, x, s)
            if guidance != 1.0:
                vu = v_of(kv_u, valid_u, x, s)
                v = vu + guidance * (v - vu)
            x = x - (sig[k] - sig[k + 1]) * v
        return x
