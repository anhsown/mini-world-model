"""Byte-level tokenizer (DESIGN §1.1) — Vietnamese-safe, zero training required."""

from __future__ import annotations

import unicodedata

PAD, BOS, EOS, BOQ, BOA, BOG = 256, 257, 258, 259, 260, 261
VOCAB_SIZE = 262

# Zero-training Vietnamese grapheme extension. ASCII and uncommon Unicode keep
# byte fallback, so this vocabulary is lossless and deterministic.
_VI_BASE = "aăâeêioôơuưyAĂÂEÊIOÔƠUƯYdđDĐ"
_VI_MARKS = ("áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
             "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
             "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
             "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ")
VI_CHARS = tuple(dict.fromkeys(_VI_BASE + _VI_MARKS))
VI_CHAR_TO_ID = {ch: VOCAB_SIZE + i for i, ch in enumerate(VI_CHARS)}
VI_ID_TO_CHAR = {i: ch for ch, i in VI_CHAR_TO_ID.items()}
VI_CHAR_VOCAB_SIZE = VOCAB_SIZE + len(VI_CHARS)

_SPECIAL_NAMES = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", BOQ: "<boq>", BOA: "<boa>", BOG: "<bog>"}


def vocab_size(mode: str = "byte") -> int:
    return VI_CHAR_VOCAB_SIZE if mode == "vi_char" else VOCAB_SIZE


def encode(text: str, mode: str = "byte") -> list[int]:
    text = unicodedata.normalize("NFC", text)
    if mode != "vi_char":
        return list(text.encode("utf-8"))
    out: list[int] = []
    for ch in text:
        tid = VI_CHAR_TO_ID.get(ch)
        if tid is not None:
            out.append(tid)
        else:
            out.extend(ch.encode("utf-8"))
    return out


def decode(ids: list[int], mode: str = "byte") -> str:
    if mode != "vi_char":
        data = bytes(i for i in ids if i < 256)
        return data.decode("utf-8", errors="replace")
    out: list[str] = []
    buf: list[int] = []
    for i in ids:
        if i < 256:
            buf.append(i)
            continue
        if buf:
            out.append(bytes(buf).decode("utf-8", errors="replace"))
            buf = []
        if i in VI_ID_TO_CHAR:
            out.append(VI_ID_TO_CHAR[i])
    if buf:
        out.append(bytes(buf).decode("utf-8", errors="replace"))
    return unicodedata.normalize("NFC", "".join(out))


def pretty(ids: list[int], mode: str = "byte") -> str:
    out = []
    buf: list[int] = []
    for i in ids:
        if i < 256:
            buf.append(i)
        else:
            if buf:
                out.append(decode(buf, mode=mode))
                buf = []
            out.append(VI_ID_TO_CHAR.get(i, _SPECIAL_NAMES.get(i, f"<{i}>")))
    if buf:
        out.append(decode(buf, mode=mode))
    return "".join(out)
