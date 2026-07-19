import math

import torch

from jwm import JWM, JWMConfig
from jwm.geometry_data import render_geometry_sequence
from jwm.geometry_trainer import geometry_batch_metrics


def test_geometry_metrics_are_finite():
    cfg = JWMConfig(d_model=32, n_layers=1, n_heads=4, head_dim=8,
                    ffn_hidden=64, rope_sections=(2, 1, 1), image_size=16,
                    patch=8, geometry_enabled=True, geometry_layers=1,
                    geometry_register_tokens=2, geometry_anchor_frames=2,
                    geometry_local_window=2)
    model = JWM(cfg)
    sample = render_geometry_sequence(5, frames=3, height=16, width=16)
    batch = {"image": sample.image.unsqueeze(0),
             "depth": sample.depth.unsqueeze(0),
             "pose_c2w": sample.pose_c2w.unsqueeze(0)}
    metrics = geometry_batch_metrics(model, batch)
    assert all(math.isfinite(value) for value in metrics.values())
