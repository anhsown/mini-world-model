"""Line-oriented OCR decoder for JWM-Read v3.1.

The failed v3 head interpreted a flattened 2D page as one CTC time axis. This
module first rectifies a labelled/predicted text region, pools it vertically,
then performs 1D contextual decoding from left to right.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU


def line_roi_align(visual: torch.Tensor, boxes: torch.Tensor, grid_h: int,
                   grid_w: int, out_h: int, out_w: int,
                   padding: float = 0.015) -> torch.Tensor:
    """Differentiable normalized xyxy crop from visual patch grids.

    Returns ``(B,out_h,out_w,D)``. Coordinates are compatible with JWM's
    normalized 0..1 synthetic and scene-text boxes.
    """
    b, n, d = visual.shape
    if n != grid_h * grid_w:
        raise ValueError(f"expected {grid_h * grid_w} visual tokens, got {n}")
    box = boxes.float().clamp(0.0, 1.0)
    x1 = (box[:, 0] - padding).clamp(0, 1)
    y1 = (box[:, 1] - padding).clamp(0, 1)
    x2 = (box[:, 2] + padding).clamp(0, 1)
    y2 = (box[:, 3] + padding).clamp(0, 1)
    x2 = torch.maximum(x2, x1 + 1.0 / grid_w)
    y2 = torch.maximum(y2, y1 + 1.0 / grid_h)
    u = torch.linspace(0, 1, out_w, device=visual.device,
                       dtype=visual.dtype).view(1, 1, out_w)
    v = torch.linspace(0, 1, out_h, device=visual.device,
                       dtype=visual.dtype).view(1, out_h, 1)
    gx = x1[:, None, None] * (1 - u) + x2[:, None, None] * u
    gy = y1[:, None, None] * (1 - v) + y2[:, None, None] * v
    gx = gx.expand(-1, out_h, -1) * 2 - 1
    gy = gy.expand(-1, -1, out_w) * 2 - 1
    grid = torch.stack((gx, gy), dim=-1)
    feature = visual.transpose(1, 2).reshape(b, d, grid_h, grid_w)
    crop = F.grid_sample(feature, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)
    return crop.permute(0, 2, 3, 1)


def _position(length: int, d: int, device, dtype):
    x = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = d // 2
    frequency = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=device) / max(1, half))
    p = torch.cat((torch.sin(x * frequency), torch.cos(x * frequency)), -1)
    return F.pad(p, (0, d - p.shape[-1])).to(dtype)


class _LineBlock(nn.Module):
    def __init__(self, d: int, heads: int, hidden: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, bias=False)
        self.ff = SwiGLU(d, hidden)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class LineROIRecognizer(nn.Module):
    def __init__(self, d: int, heads: int, hidden: int, layers: int,
                 vocab_size: int, grid_h: int, grid_w: int,
                 roi_h: int = 4, roi_w: int = 96):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.roi_h, self.roi_w = roi_h, roi_w
        self.textness = nn.Linear(d, 1)
        self.height_attention = nn.Linear(d, 1)
        self.blocks = nn.ModuleList([_LineBlock(d, heads, hidden)
                                     for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab_size + 1)

    def crop(self, visual: torch.Tensor, boxes: torch.Tensor):
        region = line_roi_align(visual, boxes, self.grid_h, self.grid_w,
                                self.roi_h, self.roi_w)
        weights = self.height_attention(region).softmax(dim=1)
        return (weights * region).sum(dim=1)

    def forward(self, visual: torch.Tensor, boxes: torch.Tensor):
        line = self.crop(visual, boxes)
        line = line + _position(line.shape[1], line.shape[2], line.device,
                                line.dtype).unsqueeze(0)
        for block in self.blocks:
            line = block(line)
        return self.head(self.norm(line))

    def textness_logits(self, visual: torch.Tensor):
        return self.textness(visual).squeeze(-1)

    def textness_targets(self, boxes: torch.Tensor) -> torch.Tensor:
        y, x = torch.meshgrid(
            (torch.arange(self.grid_h, device=boxes.device) + 0.5) / self.grid_h,
            (torch.arange(self.grid_w, device=boxes.device) + 0.5) / self.grid_w,
            indexing="ij")
        x, y = x.flatten()[None], y.flatten()[None]
        return ((x >= boxes[:, 0:1]) & (x <= boxes[:, 2:3]) &
                (y >= boxes[:, 1:2]) & (y <= boxes[:, 3:4])).float()

    @torch.no_grad()
    def predict_boxes(self, visual: torch.Tensor, threshold: float = 0.35):
        probability = self.textness_logits(visual).sigmoid()
        boxes = []
        for row in probability:
            mask = row.reshape(self.grid_h, self.grid_w) >= threshold
            if not bool(mask.any()):
                boxes.append(row.new_tensor([0.0, 0.0, 1.0, 1.0]))
                continue
            y, x = torch.where(mask)
            boxes.append(row.new_tensor([x.min() / self.grid_w,
                                         y.min() / self.grid_h,
                                         (x.max() + 1) / self.grid_w,
                                         (y.max() + 1) / self.grid_h]))
        return torch.stack(boxes)

