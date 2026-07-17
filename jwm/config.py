"""JWM configuration (DESIGN §9). All dims chosen for GTX 1650 4GB."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JWMConfig:
    # transformer
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    head_dim: int = 32                      # d_model / n_heads
    ffn_hidden: int = 672                   # ~8/3 * d, multiple of 32
    vocab_size: int = 262
    dropout: float = 0.0

    # MRoPE (DESIGN §4): half-dim sections for (t, h, w); sum == head_dim // 2
    rope_sections: tuple = (8, 4, 4)
    rope_base: float = 10000.0
    ar_dm_gap: float = 64.0                 # G — positional gap between AR and DM (ablatable)
    use_boundary_embedding: bool = False    # improvement #1: learned boundary instead of gap
    tps_base: float = 5.0                   # JARVIS camera display fps
    fd_fps: float = 2.0                     # FD pair sampling rate -> delta_t = 2.5

    # vision
    image_size: int = 64
    patch: int = 8                          # -> 8x8 = 64 AR image tokens
    patch_merge: int = 1                    # NxN patch merge (Reader: 2 -> 4x fewer tokens)
    vision_mlp_layers: int = 1              # 1 = linear embed; 2+ = hierarchical MLP stem
    # conv-AE latent (frozen after pretrain)
    z_ch: int = 8                           # 8x8x8 latent
    lat_merge: int = 2                      # 2x2 merge -> 4x4 = 16 DM tokens, dim z_ch*4 = 32

    # Inkling-mini MoE for the REASONER tower (INKLING_MINI.md; generator stays dense)
    reasoner_moe: bool = False
    moe_experts: int = 32                   # Inkling: 256
    moe_topk: int = 4                       # Inkling: 6
    moe_shared: int = 1                     # Inkling: 2
    moe_expert_hidden: int = 0              # 0 = d_model // 2 (Inkling ratio: 3072 = 6144/2)
    moe_aux_alpha: float = 0.01
    moe_dense_first_layer: bool = True      # Inkling keeps early layer(s) dense

    # diffusion / flow
    sigma_emb_dim: int = 128
    text_dropout: float = 0.10
    lambda_bbox: float = 10.0               # Cosmos: action loss x10
    lambda_latent: float = 1.0
    lambda_conf: float = 0.05
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # sequence limits
    max_q_bytes: int = 64
    max_a_bytes: int = 40

    # sampling defaults
    sample_steps: int = 50
    sample_steps_fast: int = 4
    sample_shift: float = 3.0
    cfg_scale: float = 1.0                  # 1.0 = off

    @property
    def n_img_tokens(self) -> int:
        return self.img_grid ** 2

    @property
    def img_grid(self) -> int:
        return self.image_size // self.patch // self.patch_merge

    @property
    def img_tok_dim(self) -> int:
        return 3 * (self.patch * self.patch_merge) ** 2

    @property
    def n_lat_tokens(self) -> int:
        g = self.image_size // 8 // self.lat_merge   # AE downsamples 8x: 64->8; merge 2 -> 4
        return g * g

    @property
    def lat_grid(self) -> int:
        return self.image_size // 8 // self.lat_merge

    @property
    def lat_tok_dim(self) -> int:
        return self.z_ch * self.lat_merge ** 2

    @property
    def fd_delta_t(self) -> float:
        return self.tps_base / self.fd_fps
