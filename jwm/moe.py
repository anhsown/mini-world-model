"""Inkling-mini MoE feed-forward (INKLING_MINI.md §3).

Faithful micro-scale port of Inkling's expert topology:
  * fine-grained SwiGLU experts with hidden = d/2 (Inkling: 3072 = 6144/2)
  * SIGMOID gating (not softmax), top-k selection, weights normalized to sum 1
  * shared expert(s) always active (Inkling: 2 shared; mini: 1)
  * Switch-style load-balancing auxiliary loss (open-standard stand-in for
    Inkling's internal balancing)

The module records `last_aux_loss` on every training forward so the caller can
add it to the task loss without changing layer signatures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    def __init__(self, d: int, expert_hidden: int, n_experts: int, top_k: int,
                 n_shared: int = 1, aux_alpha: float = 0.01):
        super().__init__()
        self.d = d
        self.h = expert_hidden
        self.E = n_experts
        self.k = top_k
        self.aux_alpha = aux_alpha
        # batched expert weights: (E, d, h) x2 gates/up, (E, h, d) down
        self.w_gate = nn.Parameter(torch.empty(n_experts, d, expert_hidden))
        self.w_up = nn.Parameter(torch.empty(n_experts, d, expert_hidden))
        self.w_down = nn.Parameter(torch.empty(n_experts, expert_hidden, d))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)
        # shared expert(s): one SwiGLU sized n_shared * expert_hidden
        sh = expert_hidden * max(1, n_shared)
        self.shared_gate = nn.Linear(d, sh, bias=False)
        self.shared_up = nn.Linear(d, sh, bias=False)
        self.shared_down = nn.Linear(sh, d, bias=False)
        self.router = nn.Linear(d, n_experts, bias=False)
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d) -> (B, S, d). Routed top-k experts + always-on shared."""
        B, S, d = x.shape
        flat = x.reshape(-1, d)                                  # (N, d)
        N = flat.shape[0]

        gates = torch.sigmoid(self.router(flat))                 # (N, E) sigmoid, not softmax
        top_w, top_i = torch.topk(gates, self.k, dim=-1)         # (N, k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        out = torch.zeros_like(flat)
        # expert-major dispatch: E small GEMM groups (fine at micro scale)
        for e in range(self.E):
            mask = (top_i == e)
            if not mask.any():
                continue
            tok_idx, slot_idx = mask.nonzero(as_tuple=True)
            xe = flat[tok_idx]                                   # (n_e, d)
            he = F.silu(xe @ self.w_gate[e]) * (xe @ self.w_up[e])
            ye = he @ self.w_down[e]                             # (n_e, d)
            out.index_add_(0, tok_idx, ye * top_w[tok_idx, slot_idx].unsqueeze(-1))

        # shared expert path (always active, like Inkling's shared experts)
        out = out + self.shared_down(F.silu(self.shared_gate(flat)) * self.shared_up(flat))

        # Switch-style load-balancing aux loss: E * sum_e f_e * P_e
        if self.training:
            with torch.no_grad():
                counts = torch.zeros(self.E, device=x.device)
                counts.scatter_add_(0, top_i.reshape(-1),
                                    torch.ones(N * self.k, device=x.device))
                f = counts / max(1, N * self.k)                  # fraction routed
            p = (gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-8)).mean(dim=0)
            self.last_aux_loss = self.aux_alpha * self.E * (f * p).sum()
        else:
            self.last_aux_loss = None
        return out.reshape(B, S, d)

    @torch.no_grad()
    def routing_stats(self, x: torch.Tensor) -> dict:
        """Router health: per-expert load fractions + entropy (dead-expert check)."""
        flat = x.reshape(-1, self.d)
        gates = torch.sigmoid(self.router(flat))
        _, top_i = torch.topk(gates, self.k, dim=-1)
        counts = torch.bincount(top_i.reshape(-1), minlength=self.E).float()
        load = counts / counts.sum().clamp(min=1)
        ent = -(load.clamp(min=1e-9) * load.clamp(min=1e-9).log()).sum()
        return {"load": load, "entropy": float(ent),
                "max_entropy": float(torch.log(torch.tensor(float(self.E)))),
                "dead_experts": int((counts == 0).sum())}
