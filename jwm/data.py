"""Batch preparation for JWM training — text padding, mode batching."""

from __future__ import annotations

import random

import torch

from . import tokenizer as tok
from .config import JWMConfig


def pad_text(texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode UTF-8 bytes, truncate/pad to max_len. Returns (ids, valid)."""
    ids = torch.full((len(texts), max_len), tok.PAD, dtype=torch.long)
    valid = torch.zeros(len(texts), max_len, dtype=torch.bool)
    for i, t in enumerate(texts):
        b = tok.encode(t)[:max_len]
        ids[i, : len(b)] = torch.tensor(b, dtype=torch.long)
        valid[i, : len(b)] = True
    return ids, valid


def pad_answers(texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Answers get an EOS appended (the model must learn to stop)."""
    ids = torch.full((len(texts), max_len), tok.PAD, dtype=torch.long)
    valid = torch.zeros(len(texts), max_len, dtype=torch.bool)
    for i, t in enumerate(texts):
        b = tok.encode(t)[: max_len - 1] + [tok.EOS]
        ids[i, : len(b)] = torch.tensor(b, dtype=torch.long)
        valid[i, : len(b)] = True
    return ids, valid


def imgs_to_float(u8: torch.Tensor, device) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [0,1]."""
    return u8.permute(0, 3, 1, 2).float().div(255.0).to(device)


class ModeBatcher:
    """Draws batches per mode from a built split.

    Step-synchronous mode selection (micro version of Cosmos' rank-synchronous
    stream selection): the trainer selects ONE mode per optimization step from a
    seeded schedule, so every sample in a step shares sequence shape and cost.
    """

    def __init__(self, split: dict, cfg: JWMConfig, seed: int = 0,
                 text_dropout: float | None = None):
        self.split = split
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.text_dropout = cfg.text_dropout if text_dropout is None else text_dropout

    def _maybe_drop_text(self, q_ids, q_valid):
        """10% question dropout -> unconditional branch for CFG (Cosmos §4.2.1)."""
        B = q_ids.shape[0]
        drop = torch.rand(B) < self.text_dropout
        q_ids = q_ids.clone()
        q_valid = q_valid.clone()
        q_ids[drop] = tok.PAD
        q_valid[drop] = False
        return q_ids, q_valid

    def pick_mode(self, probs: dict[str, float]) -> str:
        r = self.rng.random()
        acc = 0.0
        for m, p in probs.items():
            acc += p
            if r < acc:
                return m
        return list(probs)[-1]

    def batch_qa(self, n: int, device):
        d = self.split["qa"]
        idx = [self.rng.randrange(len(d["q"])) for _ in range(n)]
        img = imgs_to_float(d["img"][idx], device)
        q_ids, q_valid = pad_text([d["q"][i] for i in idx], self.cfg.max_q_bytes)
        a_ids, a_valid = pad_answers([d["a"][i] for i in idx], self.cfg.max_a_bytes)
        return img, q_ids.to(device), q_valid.to(device), a_ids.to(device), a_valid.to(device)

    def batch_ground(self, n: int, device):
        d = self.split["ground"]
        idx = [self.rng.randrange(len(d["q"])) for _ in range(n)]
        img = imgs_to_float(d["img"][idx], device)
        q_ids, q_valid = pad_text([d["q"][i] for i in idx], self.cfg.max_q_bytes)
        q_ids, q_valid = self._maybe_drop_text(q_ids, q_valid)
        bbox = d["bbox"][idx].to(device)
        return img, q_ids.to(device), q_valid.to(device), bbox

    def batch_fd(self, n: int, device):
        d = self.split["fd"]
        idx = [self.rng.randrange(len(d["q"])) for _ in range(n)]
        img = imgs_to_float(d["img"][idx], device)
        img1 = imgs_to_float(d["img1"][idx], device)
        q_ids, q_valid = pad_text([d["q"][i] for i in idx], self.cfg.max_q_bytes)
        q_ids, q_valid = self._maybe_drop_text(q_ids, q_valid)
        return img, img1, q_ids.to(device), q_valid.to(device)

    def batch_t2i(self, n: int, device):
        d = self.split["t2i"]
        idx = [self.rng.randrange(len(d["q"])) for _ in range(n)]
        img = imgs_to_float(d["img"][idx], device)         # target image -> AE outside
        q_ids, q_valid = pad_text([d["q"][i] for i in idx], self.cfg.max_q_bytes)
        q_ids, q_valid = self._maybe_drop_text(q_ids, q_valid)
        return img, q_ids.to(device), q_valid.to(device)
