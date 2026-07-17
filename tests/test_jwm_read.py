"""Tests for JWM-Read: hierarchical vision stem (patch_merge + MLP), CER,
Vietnamese text rendering, and the lazy batcher. All CPU, tiny dims."""

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jwm.config import JWMConfig
from jwm.mathx import char_error_rate
from jwm.model import JWM
from jwm.read_data import (LazyReadBatcher, find_fonts, letterbox,
                           render_read_sample)
from PIL import Image


def tiny_reader_cfg(**kw):
    base = dict(d_model=64, n_layers=2, n_heads=2, ffn_hidden=96,
                image_size=64, patch=8, patch_merge=2, vision_mlp_layers=2,
                max_q_bytes=16, max_a_bytes=24)
    base.update(kw)
    return JWMConfig(**base)


# ---------------------------------------------------------------- vision stem

def test_img_grid_and_tok_dim_with_merge():
    c = tiny_reader_cfg()                      # 64px / (8*2) = 4x4 grid
    assert c.img_grid == 4
    assert c.img_tok_dim == 3 * 16 * 16        # merged patch is 16x16 px


def test_forward_shapes_with_merged_stem():
    c = tiny_reader_cfg()
    m = JWM(c)
    img = torch.rand(2, 3, 64, 64)
    toks = m._img_tokens(img)
    assert toks.shape == (2, 16, c.d_model)    # 4x4 = 16 visual tokens


def test_mlp_stem_depth():
    c = tiny_reader_cfg(vision_mlp_layers=2)
    m = JWM(c)
    linears = [l for l in m.patch_embed if isinstance(l, torch.nn.Linear)]
    assert len(linears) == 2
    c1 = tiny_reader_cfg(patch_merge=1, vision_mlp_layers=1)
    m1 = JWM(c1)
    assert isinstance(m1.patch_embed, torch.nn.Linear)


def test_merge_stem_qa_loss_backward():
    c = tiny_reader_cfg()
    m = JWM(c)
    img = torch.rand(2, 3, 64, 64)
    q = torch.randint(0, 200, (2, 8))
    qv = torch.ones(2, 8, dtype=torch.bool)
    a = torch.randint(0, 200, (2, 10))
    av = torch.ones(2, 10, dtype=torch.bool)
    out = m.loss_qa(img, q, qv, a, av)
    loss = out[0] if isinstance(out, tuple) else out
    loss.backward()
    assert torch.isfinite(loss)


# ------------------------------------------------------------------------ CER

def test_cer_exact_and_empty():
    assert char_error_rate("xin chào", "xin chào") == 0.0
    assert char_error_rate("", "") == 0.0
    assert char_error_rate("a", "") == 1.0


def test_cer_single_diacritic_is_one_error():
    # 'chao' vs 'chào': one substitution over 4 ref chars
    assert abs(char_error_rate("chao", "chào") - 0.25) < 1e-9


def test_cer_insert_delete():
    assert abs(char_error_rate("abcd", "abd") - 1 / 3) < 1e-9   # 1 insertion
    assert abs(char_error_rate("ab", "abc") - 1 / 3) < 1e-9     # 1 deletion


# ------------------------------------------------------------------ rendering

def test_render_all_levels():
    fonts = find_fonts()
    assert fonts, "no usable font found on this machine"
    rng = random.Random(0)
    for level in (1, 2, 3, 4):
        arr, text = render_read_sample(rng, level, fonts, [], size=256)
        assert arr.shape == (256, 256, 3) and arr.dtype == np.uint8
        assert text.strip()
        # text must actually darken pixels vs the near-white background
        assert arr.min() < 120


def test_render_deterministic_from_seed():
    fonts = find_fonts()
    a1, t1 = render_read_sample(random.Random(42), 2, fonts, [], size=128)
    a2, t2 = render_read_sample(random.Random(42), 2, fonts, [], size=128)
    assert t1 == t2 and np.array_equal(a1, a2)


def test_letterbox_preserves_aspect():
    im = letterbox(Image.new("RGB", (400, 200), (255, 0, 0)), 64)
    assert im.size == (64, 64)
    arr = np.asarray(im)
    assert (arr[0] == 128).all()               # top rows are gray padding
    assert (arr[32] == (255, 0, 0)).all(axis=-1).any()   # content in the middle


# -------------------------------------------------------------------- batcher

def test_lazy_batcher_shapes_synth_only():
    c = tiny_reader_cfg()
    b = LazyReadBatcher(c, doc_pairs=[], fonts=find_fonts(), corpus=[],
                        cam=None, synth_ratio=1.0, levels=(1, 2), seed=0)
    img, q_ids, q_valid, a_ids, a_valid = b.batch_qa(3, "cpu")
    assert img.shape == (3, 3, 64, 64) and img.dtype == torch.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert q_ids.shape == (3, c.max_q_bytes)
    assert a_ids.shape == (3, c.max_a_bytes)
    assert b.pick_mode({}) == "qa"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"(ok) {fn.__name__}")
    print(f"ALL {len(fns)} PASSED")
