"""Correctness tests for line-ROI OCR decoding."""

import torch

from jwm import JWM
from jwm.configs import reader_scale_v31
from jwm.reader_roi import LineROIRecognizer, line_roi_align


def test_line_roi_align_selects_requested_horizontal_region():
    # 4x4 grid where feature value equals x coordinate.
    x = torch.arange(4).view(1, 1, 4).expand(1, 4, 4).float()
    visual = x.reshape(1, 16, 1).requires_grad_()
    box = torch.tensor([[0.5, 0.0, 1.0, 1.0]])
    crop = line_roi_align(visual, box, 4, 4, 2, 4, padding=0.0)
    assert crop.mean() > 1.5
    crop.sum().backward()
    assert visual.grad is not None and visual.grad.abs().sum() > 0


def test_line_recognizer_shapes_and_text_targets():
    model = LineROIRecognizer(32, 4, 64, 2, 80, 4, 6, 3, 12)
    visual = torch.randn(2, 24, 32)
    boxes = torch.tensor([[0.1, 0.2, 0.8, 0.5],
                          [0.0, 0.0, 1.0, 1.0]])
    logits = model(visual, boxes)
    targets = model.textness_targets(boxes)
    assert logits.shape == (2, 12, 81)
    assert targets.shape == (2, 24)
    assert targets[1].sum() == 24


def test_v31_config_builds_new_decoder_without_enabling_geometry():
    cfg = reader_scale_v31()
    model = JWM(cfg)
    assert model.reader_roi is not None
    assert model.geometry is None
    assert cfg.reader_decoder == "line_roi_ctc"
