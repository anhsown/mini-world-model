"""High-resolution local vision stem for JWM-Read v3.

Unlike the v2 MLP stem, raw 2x2 patches are not concatenated before any spatial
reasoning. A patch-16 convolution creates a high-resolution grid, local window
attention resolves glyph structure, and only then does learned 2x2 merging
compress tokens for the MoT reasoner.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """BHWD -> (B*n_windows, window*window, D)."""
    b, h, w, d = x.shape
    if h % window or w % window:
        raise ValueError(f"vision grid {h}x{w} must divide window={window}")
    return (x.view(b, h // window, window, w // window, window, d)
             .permute(0, 1, 3, 2, 4, 5)
             .reshape(-1, window * window, d))


def _unpartition(x: torch.Tensor, b: int, h: int, w: int, window: int) -> torch.Tensor:
    d = x.shape[-1]
    return (x.view(b, h // window, w // window, window, window, d)
             .permute(0, 1, 3, 2, 4, 5)
             .reshape(b, h, w, d))


class LocalVisionBlock(nn.Module):
    """Conv positional encoding + window attention + SwiGLU."""

    def __init__(self, d: int, heads: int, window: int, hidden: int):
        super().__init__()
        if d % heads:
            raise ValueError("vision hidden size must divide attention heads")
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.window = window
        self.pos = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=True)
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        p = self.pos(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + p
        win = _partition(self.norm1(x), self.window)
        n, s, _ = win.shape
        qkv = self.qkv(win).view(n, s, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)
        y = self.proj(y.transpose(1, 2).reshape(n, s, d))
        x = x + _unpartition(y, b, h, w, self.window)
        z = self.norm2(x)
        x = x + self.down(F.silu(self.gate(z)) * self.up(z))
        return x


class DocumentVisionStem(nn.Module):
    """Patchify -> local reasoning -> post-reasoning merge -> visual tokens."""

    def __init__(self, d: int, patch: int, merge: int, layers: int,
                 heads: int, window: int, ffn_hidden: int,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.patch = patch
        self.merge = merge
        self.window = window
        self.grad_checkpoint = grad_checkpoint
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch, stride=patch, bias=True)
        self.blocks = nn.ModuleList([
            LocalVisionBlock(d, heads, window, ffn_hidden) for _ in range(layers)
        ])
        merged_d = d * merge * merge
        self.merge_norm = nn.LayerNorm(merged_d, eps=1e-6)
        self.merge_proj = nn.Linear(merged_d, d, bias=False)
        # The box head consumes stem tokens before MoT/MRoPE.  Explicit absolute
        # 2D coordinates prevent its cross-attention from becoming permutation
        # invariant over otherwise unordered visual tokens.
        self.coord_proj = nn.Sequential(
            nn.Linear(2, d, bias=False), nn.SiLU(), nn.Linear(d, d, bias=False))
        self.final_norm = nn.LayerNorm(d, eps=1e-6)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(img).permute(0, 2, 3, 1)  # B,H,W,D
        for block in self.blocks:
            if self.grad_checkpoint and self.training and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        b, h, w, d = x.shape
        m = self.merge
        if h % m or w % m:
            raise ValueError(f"pre-merge grid {h}x{w} must divide merge={m}")
        gh, gw = h // m, w // m
        x = (x.view(b, gh, m, gw, m, d)
             .permute(0, 1, 3, 2, 4, 5)
             .reshape(b, gh * gw, d * m * m))
        x = self.merge_proj(self.merge_norm(x))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, gh, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, gw, device=x.device, dtype=x.dtype),
            indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, gh * gw, 2)
        return self.final_norm(x + self.coord_proj(coords))
