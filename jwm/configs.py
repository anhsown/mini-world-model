"""Canonical JWM scale definitions. The staged pipeline (jwm/stages/) uses SCALE_V3."""

from __future__ import annotations

from .config import JWMConfig
from . import tokenizer as tok


def scale_v1() -> JWMConfig:
    """10.7M — first proof of life (deprecated by v3 pipeline)."""
    return JWMConfig()


def scale_v2() -> JWMConfig:
    """28M — monolithic notebook run (superseded mid-training by v3 pipeline)."""
    return JWMConfig(d_model=384, n_layers=8, n_heads=12, ffn_hidden=1024)


def scale_v3() -> JWMConfig:
    """~70-80M — maximum safe from-scratch scale for a 4GB GTX 1650.

    d_model 512, 10 dual-tower layers, 16 heads x 32 head_dim,
    SwiGLU hidden 1408 (~8/3 * d, multiple of 32).

    Day-1 finding: at this scale the reasoner needs a substantially larger
    dedicated-QA budget than the GPU affords overnight (byte-exact long answers
    require val tok-acc ~0.98; 68M plateaued at 0.94 with 3200 steps while 28M
    reached ~0.98 with a comparable budget). Kept for the Day-2 budget study.
    """
    return JWMConfig(d_model=512, n_layers=10, n_heads=16, ffn_hidden=1408)


def pipeline_scale_moe() -> JWMConfig:
    """Inkling-mini: the proven 28M backbone with an MoE reasoner tower.

    ~86M TOTAL parameters (6.2x reasoner-FFN capacity) at ~31M ACTIVE per token
    — capacity of the failed 68M dense run at the step-cost of the proven 28M.
    Day-2 experiment; see jwm/INKLING_MINI.md.
    """
    return JWMConfig(d_model=384, n_layers=8, n_heads=12, ffn_hidden=1024,
                     reasoner_moe=True, moe_experts=32, moe_topk=4, moe_shared=1)


def reader_scale() -> JWMConfig:
    """JWM-Read — document/text reading on OUR architecture, sized for Kaggle T4 16GB.

    768px input, patch-16 + 2x2 merge -> 24x24 = 576 visual tokens, hierarchical
    MLP vision stem (Inkling-style), MoE reasoner, byte answers up to 192B.
    READ is just QA-mode with a bigger eye — no new mode machinery needed.
    """
    return JWMConfig(d_model=512, n_layers=10, n_heads=16, ffn_hidden=1408,
                     reasoner_moe=True, moe_experts=32, moe_topk=4, moe_shared=1,
                     image_size=768, patch=16, patch_merge=2, vision_mlp_layers=2,
                     max_q_bytes=96, max_a_bytes=224)


def reader_scale_v3() -> JWMConfig:
    """JWM-Read v3 for Kaggle T4x2.

    Portrait 1024x768 input. Patch-16 local reasoning happens on a 64x48 grid;
    learned 2x2 post-encoder merge yields 32x24 = 768 MoT visual tokens.
    The 16-expert top-2 router gives each expert more OCR updates than v2's
    32/top-4 setup while retaining sparse capacity.
    """
    return JWMConfig(
        d_model=512, n_layers=10, n_heads=16, ffn_hidden=1408,
        vocab_size=tok.VI_CHAR_VOCAB_SIZE, tokenizer_mode="vi_char",
        reasoner_moe=True, moe_experts=16, moe_topk=2, moe_shared=1,
        image_size=768, image_height=1024, image_width=768,
        patch=16, patch_merge=2, vision_stem="local", vision_mlp_layers=1,
        vision_local_layers=2, vision_local_heads=8, vision_window=8,
        vision_grad_checkpoint=True,
        max_q_bytes=112, max_a_bytes=176,
        eos_loss_weight=4.0, answer_input_corrupt=0.12,
        vision_contrast_alpha=0.15, vision_contrast_margin=0.20,
        reader_ctc_weight=1.0, reader_box_weight=0.35,
    )


def eye_physical_scale() -> JWMConfig:
    """Day-5 streaming physical eye for T4x2 training and local deployment.

    The camera may capture/display at 30 FPS, while geometry keyframes are
    selected adaptively. A 256px, 16x16 visual grid supplies dense local cues;
    GCM retains two full anchors, eight recent frames and compact trajectory
    summaries without making memory grow by a full image per frame.
    """
    return JWMConfig(
        d_model=384, n_layers=8, n_heads=12, ffn_hidden=1024,
        # Match jwm_v4's MoE shapes so its semantic reasoner can warm-start
        # this branch; only the new eye/geometry modules begin from scratch.
        reasoner_moe=True, moe_experts=32, moe_topk=4, moe_shared=1,
        image_size=256, patch=16, patch_merge=1,
        vision_stem="local", vision_local_layers=2,
        vision_local_heads=8, vision_window=4,
        vision_grad_checkpoint=True,
        max_q_bytes=96, max_a_bytes=128,
        geometry_enabled=True, geometry_layers=2,
        geometry_register_tokens=4, geometry_anchor_frames=2,
        geometry_local_window=8, geometry_max_trajectory_frames=256,
        geometry_depth_weight=1.0, geometry_abs_pose_weight=1.0,
        geometry_rel_pose_weight=0.5,
    )


def eye_physical_v2_scale() -> JWMConfig:
    """Pairwise metric Eye v2; semantic MoE/Generator remain frozen.

    The visual token and semantic dimensions remain compatible with JWM v4,
    while all geometry-v2 parameters are new.  Relative pose is inferred from
    frame-pair evidence and integrated from an identity first-camera frame.
    """
    return JWMConfig(
        d_model=384, n_layers=8, n_heads=12, ffn_hidden=1024,
        reasoner_moe=True, moe_experts=32, moe_topk=4, moe_shared=1,
        image_size=256, patch=16, patch_merge=1,
        vision_stem="local", vision_local_layers=2,
        vision_local_heads=8, vision_window=4,
        vision_grad_checkpoint=True,
        max_q_bytes=96, max_a_bytes=128,
        geometry_enabled=True, geometry_version="v2_pairwise",
        geometry_layers=2, geometry_register_tokens=4,
        geometry_anchor_frames=2, geometry_local_window=8,
        geometry_max_trajectory_frames=256,
        geometry_motion_radius=2,
        geometry_depth_weight=0.5,
        geometry_metric_depth_weight=1.0,
        geometry_abs_pose_weight=0.25,
        geometry_rel_pose_weight=1.0,
        geometry_cycle_weight=0.25,
        geometry_dynamic_weight=0.20,
        geometry_counterfactual_weight=0.25,
        geometry_counterfactual_margin=0.20,
        geometry_min_valid_fraction=0.20,
    )


def eye_physical_v2_ablation(arm: str) -> JWMConfig:
    """Controlled v2 arms; each arm adds exactly one mechanism."""
    arm = arm.upper()
    if arm not in {"A", "B", "C", "D"}:
        raise ValueError("Eye v2 ablation arm must be A, B, C or D")
    cfg = eye_physical_v2_scale()
    cfg.geometry_cycle_weight = 0.25 if arm in {"B", "C", "D"} else 0.0
    cfg.geometry_dynamic_weight = 0.20 if arm in {"C", "D"} else 0.0
    cfg.geometry_counterfactual_weight = 0.25 if arm == "D" else 0.0
    return cfg


def eye_physical_v3_scale() -> JWMConfig:
    """CTPG-Eye v3 for T4x2 training and bounded local deployment.

    The semantic MoT dimensions stay warm-start compatible with JWM-v4. The
    physical eye is camera-calibrated and trains independently at 256px; its
    sparse 96-channel path is small enough for two 16GB T4 workers.
    """
    return JWMConfig(
        d_model=384, n_layers=8, n_heads=12, ffn_hidden=1024,
        reasoner_moe=True, moe_experts=32, moe_topk=4, moe_shared=1,
        image_size=256, patch=16, patch_merge=1,
        vision_stem="local", vision_local_layers=2,
        vision_local_heads=8, vision_window=4,
        vision_grad_checkpoint=True,
        max_q_bytes=96, max_a_bytes=128,
        geometry_enabled=True, geometry_version="v3_ctpg",
        geometry_v3_width=96, geometry_track_points=64,
        geometry_track_radius=2, geometry_track_iterations=3,
        geometry_ba_iterations=2, geometry_memory_frames=32,
        geometry_depth_weight=1.0, geometry_rel_pose_weight=1.0,
        geometry_rel_translation_weight=1.0,
        geometry_track_weight=1.0, geometry_rigid_weight=0.25,
        geometry_dynamic_weight=0.20, geometry_ba_weight=0.20,
        geometry_counterfactual_weight=0.15,
    )


def eye_physical_v32_scale() -> JWMConfig:
    """JWM-Eye v3.2: ~350M-class causal physical world model for T4x2.

    Capacity is increased in both the semantic MoT and the supervised physical
    eye.  Unlike a cosmetic backbone-only scale-up, the new geometry graph adds
    a six-layer causal scene-register mixer, factorised depth/ray prediction,
    forward/backward track consistency and calibrated track confidence.
    """
    return JWMConfig(
        d_model=576, n_layers=16, n_heads=18, head_dim=32,
        ffn_hidden=1536, rope_sections=(8, 4, 4),
        reasoner_moe=True, moe_experts=32, moe_topk=4, moe_shared=1,
        moe_expert_hidden=288,
        image_size=256, patch=16, patch_merge=1,
        vision_stem="local", vision_local_layers=6,
        vision_local_heads=12, vision_window=4,
        vision_grad_checkpoint=True,
        max_q_bytes=128, max_a_bytes=192,
        geometry_enabled=True, geometry_version="v32_ctpg",
        geometry_v3_width=160, geometry_track_points=128,
        geometry_track_radius=3, geometry_track_iterations=6,
        geometry_ba_iterations=3, geometry_memory_frames=128,
        geometry_scene_registers=16, geometry_scene_layers=6,
        geometry_scene_width=384, geometry_scene_heads=12,
        geometry_pose_context=96, geometry_ray_residual=0.10,
        geometry_depth_weight=1.0, geometry_rel_pose_weight=1.0,
        geometry_rel_translation_weight=1.0,
        geometry_track_weight=0.75, geometry_rigid_weight=0.20,
        geometry_dynamic_weight=0.25, geometry_ba_weight=0.15,
        geometry_counterfactual_weight=0.20,
        geometry_ray_weight=0.20,
        geometry_track_cycle_weight=0.20,
        geometry_confidence_weight=0.15,
    )


def eye_physical_v32_smoke_scale() -> JWMConfig:
    """Graph-equivalent tiny profile for CPU unit and contract tests."""
    cfg = eye_physical_v32_scale()
    cfg.d_model = 48; cfg.n_layers = 2; cfg.n_heads = 6
    cfg.ffn_hidden = 96; cfg.moe_expert_hidden = 24
    cfg.moe_experts = 4; cfg.moe_topk = 2
    cfg.image_size = 32; cfg.patch = 8
    cfg.vision_local_layers = 1; cfg.vision_local_heads = 6
    cfg.geometry_v3_width = 32; cfg.geometry_track_points = 8
    cfg.geometry_track_radius = 1; cfg.geometry_track_iterations = 1
    cfg.geometry_ba_iterations = 1; cfg.geometry_memory_frames = 8
    cfg.geometry_scene_registers = 2; cfg.geometry_scene_layers = 1
    cfg.geometry_scene_width = 48; cfg.geometry_scene_heads = 6
    cfg.geometry_pose_context = 8
    cfg.rope_sections = (2, 1, 1); cfg.head_dim = 8
    return cfg


def reader_scale_v31() -> JWMConfig:
    """Corrective Reader warm-start: line ROI before 1D CTC decoding.

    It retains v3's visual stem/reasoner shape so those weights can be reused,
    while OCR/localization heads are newly initialized. Full-page flattened CTC
    is intentionally disabled.
    """
    cfg = reader_scale_v3()
    cfg.reader_decoder = "line_roi_ctc"
    cfg.reader_roi_height = 4
    cfg.reader_roi_width = 128
    cfg.reader_roi_layers = 2
    cfg.reader_textness_weight = 0.35
    cfg.reader_ctc_weight = 1.0
    cfg.reader_box_weight = 0.20
    return cfg


def pipeline_scale() -> JWMConfig:
    """ACTIVE scale for the staged pipeline.

    Currently the PROVEN 28M configuration (identical to v2, which reached
    56.4% QA / mIoU 0.20 on the same val/test) — chosen by the user on Day 1
    after the 68M trainability-budget diagnosis. Swap to scale_v3() once the
    Day-2 68M budget experiment lands.
    """
    return JWMConfig(d_model=384, n_layers=8, n_heads=12, ffn_hidden=1024)
