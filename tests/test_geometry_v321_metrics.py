import torch

from jwm.geometry_v3_trainer import track_metrics
from jwm.geometry_math_v3 import resize_flow_with_valid


def test_confidence_contract_reports_pck1_and_pck3_separately():
    # A two-pixel error is wrong at PCK1 but correct at PCK3. Confidence=1
    # must therefore be calibrated for the BA/PCK3 contract, not PCK1.
    output = {
        "feature_hw": (4, 4),
        "track_source": torch.tensor([[[[0.0, 0.0]]]]),
        "track_target": torch.tensor([[[[2.0, 0.0]]]]),
        "track_confidence": torch.ones(1, 1, 1),
        "dynamic_probability": torch.zeros(1, 1, 1),
    }
    batch = {
        "rigid_flow": torch.zeros(1, 1, 4, 4, 2),
        "rigid_flow_valid": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        "dynamic_mask": torch.zeros(1, 2, 4, 4),
    }
    metrics = track_metrics(output, batch)
    assert metrics["track_pck1"] == 0.0
    assert metrics["track_pck3"] == 1.0
    assert metrics["track_ece"] == 0.0
    assert metrics["track_ece_pck1"] == 1.0


def test_invalid_flow_sentinel_cannot_bleed_into_valid_resize():
    flow = torch.ones(1, 1, 8, 8, 2)
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    flow[:, :, :, 4:] = 1e9
    valid[:, :, :, 4:] = False
    resized, mask = resize_flow_with_valid(flow, valid, (4, 4))
    assert torch.isfinite(resized).all()
    assert float(resized[mask[:, :, None].expand_as(resized)].max()) < 2.0
