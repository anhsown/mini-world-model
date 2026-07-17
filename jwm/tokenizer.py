"""Byte-level tokenizer (DESIGN §1.1) — Vietnamese-safe, zero training required."""

from __future__ import annotations

PAD, BOS, EOS, BOQ, BOA, BOG = 256, 257, 258, 259, 260, 261
VOCAB_SIZE = 262

_SPECIAL_NAMES = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", BOQ: "<boq>", BOA: "<boa>", BOG: "<bog>"}


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    data = bytes(i for i in ids if i < 256)
    return data.decode("utf-8", errors="replace")


def pretty(ids: list[int]) -> str:
    out = []
    buf: list[int] = []
    for i in ids:
        if i < 256:
            buf.append(i)
        else:
            if buf:
                out.append(decode(buf))
                buf = []
            out.append(_SPECIAL_NAMES.get(i, f"<{i}>"))
    if buf:
        out.append(decode(buf))
    return "".join(out)
