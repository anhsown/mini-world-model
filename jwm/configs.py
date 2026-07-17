"""Canonical JWM scale definitions. The staged pipeline (jwm/stages/) uses SCALE_V3."""

from __future__ import annotations

from .config import JWMConfig


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


def pipeline_scale() -> JWMConfig:
    """ACTIVE scale for the staged pipeline.

    Currently the PROVEN 28M configuration (identical to v2, which reached
    56.4% QA / mIoU 0.20 on the same val/test) — chosen by the user on Day 1
    after the 68M trainability-budget diagnosis. Swap to scale_v3() once the
    Day-2 68M budget experiment lands.
    """
    return JWMConfig(d_model=384, n_layers=8, n_heads=12, ffn_hidden=1024)
