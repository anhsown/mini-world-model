"""Compact Geometric Context Memory for the JWM Eye Physical pathway.

This is a small, from-scratch adaptation of the architectural principle in
LingBot-Map's Geometric Context Attention, not a copy of its weights or model:

* anchor context keeps full tokens from the first A frames;
* a local pose-reference window keeps dense tokens from recent frames;
* evicted frames retain only camera/anchor/register summary tokens;
* old trajectory summaries are bounded by an EMA history token.

The module consumes visual tokens already produced by JWM's vision stem.  Its
outputs are shared world tokens plus depth, uncertainty and C2W camera pose,
so the same representation can condition both the AR reasoner and DM generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU
from .mathx import (anchor_depth_scale, relative_pose_c2w,
                    rot6d_to_matrix, rotation_geodesic,
                    uncertainty_depth_loss)


def temporal_sinusoid(index: int | torch.Tensor, d: int, device=None,
                       dtype=torch.float32) -> torch.Tensor:
    """Absolute video-time embedding with the usual sin/cos frequency ladder."""
    x = torch.as_tensor(index, dtype=torch.float32, device=device).reshape(-1, 1)
    half = d // 2
    freq = torch.exp(-math.log(10000.0) *
                     torch.arange(half, device=x.device, dtype=torch.float32) /
                     max(1, half))
    out = torch.cat((torch.sin(x * freq), torch.cos(x * freq)), dim=-1)
    if out.shape[-1] < d:
        out = F.pad(out, (0, d - out.shape[-1]))
    return out.to(dtype=dtype)


class _ResidualAttentionFFN(nn.Module):
    """Pre-norm attention + SwiGLU with small LayerScale residual gates."""

    def __init__(self, d: int, heads: int, hidden: int):
        super().__init__()
        self.norm_q = RMSNorm(d)
        self.norm_kv = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, bias=False)
        self.norm_ff = RMSNorm(d)
        self.ff = SwiGLU(d, hidden)
        self.attn_scale = nn.Parameter(torch.full((d,), 1e-2))
        self.ff_scale = nn.Parameter(torch.full((d,), 1e-2))

    def forward(self, query: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        kv = query if context is None else context
        delta, _ = self.attn(self.norm_q(query), self.norm_kv(kv),
                             self.norm_kv(kv), need_weights=False)
        h = query + self.attn_scale * delta
        return h + self.ff_scale * self.ff(self.norm_ff(h))


class _GeometricLayer(nn.Module):
    def __init__(self, d: int, heads: int, hidden: int):
        super().__init__()
        self.frame = _ResidualAttentionFFN(d, heads, hidden)
        self.context = _ResidualAttentionFFN(d, heads, hidden)


@dataclass
class _LayerMemory:
    anchors: list[torch.Tensor] = field(default_factory=list)
    local: list[torch.Tensor] = field(default_factory=list)
    trajectory: list[torch.Tensor] = field(default_factory=list)
    history: torch.Tensor | None = None


@dataclass
class GeometryMemoryState:
    """Runtime-only cache; intentionally excluded from model state_dict."""

    layers: list[_LayerMemory]
    batch_size: int
    frame_index: int = 0


class GeometricContextMemory(nn.Module):
    """Mini GCT operating on JWM visual tokens."""

    def __init__(self, d: int, heads: int, hidden: int, layers: int,
                 grid_h: int, grid_w: int, register_tokens: int = 4,
                 anchor_frames: int = 2, local_window: int = 8,
                 max_trajectory_frames: int = 512):
        super().__init__()
        if d % heads:
            raise ValueError("geometry d_model must be divisible by heads")
        if anchor_frames < 1 or local_window < 1:
            raise ValueError("anchor_frames and local_window must be positive")
        self.d = d
        self.grid_h, self.grid_w = grid_h, grid_w
        self.register_tokens = register_tokens
        self.num_special = 2 + register_tokens  # camera + anchor/stream + registers
        self.anchor_frames = anchor_frames
        self.local_window = local_window
        self.max_trajectory_frames = max_trajectory_frames

        self.camera_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.anchor_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.stream_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.register = nn.Parameter(torch.randn(1, register_tokens, d) * 0.02)
        self.layers = nn.ModuleList([
            _GeometricLayer(d, heads, hidden) for _ in range(layers)
        ])
        self.final_norm = RMSNorm(d)
        self.pose_head = nn.Linear(d, 9)       # rot6d + translation
        self.depth_head = nn.Linear(d, 2)     # depth preactivation + log uncertainty

    def new_state(self, batch_size: int = 1) -> GeometryMemoryState:
        return GeometryMemoryState([_LayerMemory() for _ in self.layers], batch_size)

    def _specials(self, batch: int, frame_index: int, anchor: bool,
                  device, dtype) -> torch.Tensor:
        marker = self.anchor_token if anchor else self.stream_token
        h = torch.cat((self.camera_token, marker, self.register), dim=1)
        h = h.expand(batch, -1, -1).to(device=device, dtype=dtype)
        t = temporal_sinusoid(frame_index, self.d, device=device, dtype=dtype)
        return h + t.view(1, 1, self.d)

    @staticmethod
    def _context(mem: _LayerMemory) -> torch.Tensor | None:
        # Fixed anchors establish coordinates; compact old summaries correct
        # drift; dense recent views provide local overlap.
        parts = list(mem.anchors)
        if mem.history is not None:
            parts.append(mem.history)
        parts.extend(mem.trajectory)
        parts.extend(mem.local)
        return torch.cat(parts, dim=1) if parts else None

    def _commit(self, mem: _LayerMemory, h: torch.Tensor, frame_index: int,
                detach: bool) -> None:
        x = h.detach() if detach else h
        if frame_index < self.anchor_frames:
            mem.anchors.append(x)
            return
        mem.local.append(x)
        if len(mem.local) > self.local_window:
            evicted = mem.local.pop(0)[:, :self.num_special]
            mem.trajectory.append(evicted)
        if len(mem.trajectory) > self.max_trajectory_frames:
            oldest = mem.trajectory.pop(0).mean(dim=1, keepdim=True)
            mem.history = oldest if mem.history is None else \
                0.99 * mem.history + 0.01 * oldest

    def _decode(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode ``(...,P,D)`` tokens while preserving all leading dims."""
        h = self.final_norm(h)
        camera = h[..., 0, :]
        visual = h[..., self.num_special:, :]
        pose_raw = self.pose_head(camera)
        identity6 = pose_raw.new_tensor([1., 0., 0., 0., 1., 0.])
        rot6d = pose_raw[..., :6] + identity6
        rotation = rot6d_to_matrix(rot6d)
        translation = pose_raw[..., 6:]
        pose = torch.zeros(*rotation.shape[:-2], 4, 4,
                           dtype=rotation.dtype, device=rotation.device)
        pose[..., :3, :3] = rotation
        pose[..., :3, 3] = translation
        pose[..., 3, 3] = 1.0
        depth_raw = self.depth_head(visual)
        depth = F.softplus(depth_raw[..., 0]) + 1e-4
        log_sigma = depth_raw[..., 1].clamp(-6.0, 6.0)
        # Visual grid first preserves JWM's existing MRoPE coordinate order;
        # compact state tokens follow at (h,w)=(0,0).
        world_tokens = torch.cat((visual, h[..., :self.num_special, :]), dim=-2)
        return {
            "pose_c2w": pose, "rotation": rotation,
            "translation": translation, "depth_tokens": depth,
            "depth_log_sigma": log_sigma,
            "context_tokens": h[..., :self.num_special, :],
            "world_tokens": world_tokens,
        }

    def bootstrap_anchors(self, visual: torch.Tensor,
                          state: GeometryMemoryState | None = None,
                          detach_state: bool = True):
        """Jointly initialize A anchor frames with bidirectional anchor attention.

        ``visual`` is ``(B,A,N,D)``.  This is the scale/coordinate bootstrap;
        later frames are strictly causal.
        """
        b, a, n, d = visual.shape
        if not (1 <= a <= self.anchor_frames):
            raise ValueError(f"expected 1..{self.anchor_frames} anchor frames, got {a}")
        state = state or self.new_state(b)
        if state.frame_index != 0 or state.batch_size != b:
            raise ValueError("anchor bootstrap requires a fresh matching state")
        specials = [self._specials(b, i, True, visual.device, visual.dtype)
                    for i in range(a)]
        h = torch.stack([torch.cat((specials[i], visual[:, i]), dim=1)
                         for i in range(a)], dim=1)
        p = h.shape[2]
        for li, layer in enumerate(self.layers):
            flat = layer.frame(h.reshape(b * a, p, d)).reshape(b, a, p, d)
            joined = flat.reshape(b, a * p, d)
            joined = layer.context(joined, joined)
            h = joined.reshape(b, a, p, d)
            for i in range(a):
                self._commit(state.layers[li], h[:, i], i, detach_state)
        state.frame_index = a
        return self._decode(h), state

    def stream_step(self, visual: torch.Tensor, state: GeometryMemoryState,
                    detach_state: bool = True):
        """Process one post-anchor frame causally and update structured memory."""
        b, _, d = visual.shape
        if state.batch_size != b:
            raise ValueError("state batch size does not match visual tokens")
        if state.frame_index < self.anchor_frames:
            raise ValueError("bootstrap anchor frames before stream_step")
        idx = state.frame_index
        h = torch.cat((self._specials(b, idx, False, visual.device, visual.dtype),
                       visual + temporal_sinusoid(idx, d, visual.device,
                                                  visual.dtype).view(1, 1, d)), dim=1)
        for li, layer in enumerate(self.layers):
            h = layer.frame(h)
            context = self._context(state.layers[li])
            kv = h if context is None else torch.cat((context, h), dim=1)
            h = layer.context(h, kv)
            self._commit(state.layers[li], h, idx, detach_state)
        state.frame_index += 1
        return self._decode(h), state

    def forward_sequence(self, visual: torch.Tensor,
                         detach_state: bool = False):
        """Causal sequence training/inference from ``(B,T,N,D)`` visual tokens."""
        b, t, _, _ = visual.shape
        if t < 1:
            raise ValueError("geometry sequence must contain at least one frame")
        a = min(t, self.anchor_frames)
        first, state = self.bootstrap_anchors(visual[:, :a],
                                              detach_state=detach_state)
        chunks = [{k: v for k, v in first.items()}]
        for i in range(a, t):
            out, state = self.stream_step(visual[:, i], state,
                                          detach_state=detach_state)
            chunks.append({k: v.unsqueeze(1) for k, v in out.items()})
        result = {k: torch.cat([c[k] for c in chunks], dim=1)
                  for k in chunks[0]}
        result["memory_state"] = state
        return result

    def loss(self, output: dict[str, torch.Tensor], depth_gt: torch.Tensor,
             pose_gt_c2w: torch.Tensor, depth_valid: torch.Tensor | None = None,
             depth_weight: float = 1.0, abs_pose_weight: float = 1.0,
             rel_pose_weight: float = 0.5,
             rel_translation_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
        """Depth + absolute C2W pose + local all-pairs relative-pose loss."""
        pred_depth = output["depth_tokens"]
        b, t, n = pred_depth.shape
        if depth_gt.ndim == 4:
            target = F.adaptive_avg_pool2d(
                depth_gt.reshape(b * t, 1, *depth_gt.shape[-2:]),
                (self.grid_h, self.grid_w)).reshape(b, t, n)
            valid = None
            if depth_valid is not None:
                valid = F.adaptive_avg_pool2d(
                    depth_valid.float().reshape(b * t, 1, *depth_valid.shape[-2:]),
                    (self.grid_h, self.grid_w)).reshape(b, t, n) > 0.99
        else:
            target, valid = depth_gt, depth_valid

        scale = anchor_depth_scale(target, valid, self.anchor_frames)
        target = target / scale.view(b, *([1] * (target.ndim - 1)))
        depth_loss = uncertainty_depth_loss(
            pred_depth, output["depth_log_sigma"], target, valid)

        gt = pose_gt_c2w.clone()
        gt[..., :3, 3] = gt[..., :3, 3] / scale.view(b, 1, 1)
        abs_rot = rotation_geodesic(output["rotation"], gt[..., :3, :3]).mean()
        abs_trans = F.smooth_l1_loss(output["translation"], gt[..., :3, 3])
        abs_pose = abs_rot + abs_trans

        rel_terms = []
        for i in range(t):
            for j in range(i + 1, min(t, i + self.local_window + 1)):
                pp = relative_pose_c2w(output["pose_c2w"][:, i],
                                       output["pose_c2w"][:, j])
                pg = relative_pose_c2w(gt[:, i], gt[:, j])
                r = rotation_geodesic(pp[..., :3, :3], pg[..., :3, :3])
                tr = (pp[..., :3, 3] - pg[..., :3, 3]).abs().mean(-1)
                rel_terms.append(r + rel_translation_weight * tr)
        rel_pose = torch.stack(rel_terms).mean() if rel_terms else abs_pose * 0.0
        total = (depth_weight * depth_loss + abs_pose_weight * abs_pose +
                 rel_pose_weight * rel_pose)
        metrics = {
            "geometry_depth": float(depth_loss.detach()),
            "geometry_abs_rotation_rad": float(abs_rot.detach()),
            "geometry_abs_translation": float(abs_trans.detach()),
            "geometry_relative_pose": float(rel_pose.detach()),
            "geometry_anchor_scale": float(scale.mean()),
        }
        return total, metrics
