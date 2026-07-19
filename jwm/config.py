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
    # Reader v3 may use a portrait native-aspect canvas. Zero keeps the legacy
    # square image_size behaviour used by existing checkpoints.
    image_height: int = 0
    image_width: int = 0
    patch: int = 8                          # -> 8x8 = 64 AR image tokens
    patch_merge: int = 1                    # NxN patch merge (Reader: 2 -> 4x fewer tokens)
    vision_mlp_layers: int = 1              # 1 = linear embed; 2+ = hierarchical MLP stem
    vision_stem: str = "mlp"                # "mlp" (legacy) | "local" (Reader v3)
    vision_local_layers: int = 0
    vision_local_heads: int = 8
    vision_window: int = 8
    vision_grad_checkpoint: bool = False
    # conv-AE latent (frozen after pretrain)
    z_ch: int = 8                           # 8x8x8 latent
    lat_merge: int = 2                      # 2x2 merge -> 4x4 = 16 DM tokens, dim z_ch*4 = 32

    # QA loss shaping — Day-4 fix: 1 EOS byte among ~200 answer slots is too
    # thin a stop signal; >1 upweights the EOS position in the answer CE
    eos_loss_weight: float = 1.0
    answer_input_corrupt: float = 0.0
    vision_contrast_alpha: float = 0.0
    vision_contrast_margin: float = 0.25
    tokenizer_mode: str = "byte"             # "byte" | "vi_char"

    # Reader-v3 auxiliary objectives. Dormant for all legacy configs.
    reader_ctc_weight: float = 0.0
    reader_box_weight: float = 0.0
    reader_coord_bins: int = 1001
    reader_decoder: str = "full_page_ctc"  # "full_page_ctc" | "line_roi_ctc"
    reader_roi_height: int = 4
    reader_roi_width: int = 96
    reader_roi_layers: int = 2
    reader_textness_weight: float = 0.0

    # Geometric Context Memory (LingBot-Map-inspired Eye Physical pathway).
    # Disabled by default so every existing checkpoint remains load-compatible.
    geometry_enabled: bool = False
    geometry_layers: int = 2
    geometry_register_tokens: int = 4
    geometry_anchor_frames: int = 2
    geometry_local_window: int = 8
    geometry_max_trajectory_frames: int = 512
    geometry_ffn_hidden: int = 0            # 0 -> ffn_hidden
    geometry_depth_weight: float = 1.0
    geometry_abs_pose_weight: float = 1.0
    geometry_rel_pose_weight: float = 0.5
    geometry_rel_translation_weight: float = 1.0

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
        return self.img_grid_h * self.img_grid_w

    @property
    def input_height(self) -> int:
        return self.image_height or self.image_size

    @property
    def input_width(self) -> int:
        return self.image_width or self.image_size

    @property
    def img_grid_h(self) -> int:
        return self.input_height // self.patch // self.patch_merge

    @property
    def img_grid_w(self) -> int:
        return self.input_width // self.patch // self.patch_merge

    @property
    def img_grid(self) -> int:
        # Compatibility alias for square image pipelines.
        return self.img_grid_h

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
