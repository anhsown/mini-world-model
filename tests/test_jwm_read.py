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
from jwm.read_v3_data import (render_read_v3_sample, split_by_document,
                              ReadV3Batcher)
from jwm import tokenizer as tok
from PIL import Image


def tiny_reader_cfg(**kw):
    base = dict(d_model=64, n_layers=2, n_heads=2, ffn_hidden=96,
                image_size=64, patch=8, patch_merge=2, vision_mlp_layers=2,
                max_q_bytes=16, max_a_bytes=24)
    base.update(kw)
    return JWMConfig(**base)


def tiny_reader_v3_cfg(**kw):
    base = dict(
        d_model=64, n_layers=2, n_heads=2, ffn_hidden=96,
        vocab_size=tok.VI_CHAR_VOCAB_SIZE, tokenizer_mode="vi_char",
        image_size=48, image_height=64, image_width=48,
        patch=8, patch_merge=2, vision_stem="local",
        vision_local_layers=1, vision_local_heads=4, vision_window=2,
        max_q_bytes=24, max_a_bytes=20,
        reader_ctc_weight=0.5, reader_box_weight=0.25,
        vision_contrast_alpha=0.1,
    )
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


def test_v3_rectangular_local_stem_shape_and_backward():
    c = tiny_reader_v3_cfg()
    assert (c.img_grid_h, c.img_grid_w, c.n_img_tokens) == (4, 3, 12)
    m = JWM(c)
    img = torch.rand(2, 3, 64, 48, requires_grad=True)
    vis = m._img_tokens(img)
    assert vis.shape == (2, 12, 64)
    vis.square().mean().backward()
    assert m.vision_stem.patch_embed.weight.grad is not None
    assert m.vision_stem.coord_proj[0].weight.grad is not None


def test_v3_joint_qa_ctc_box_loss_backward():
    c = tiny_reader_v3_cfg(max_a_bytes=12)
    m = JWM(c)
    img = torch.rand(2, 3, 64, 48)
    q_ids, q_valid = __import__("jwm.data", fromlist=["pad_text"]).pad_text(
        ["Đọc chữ", "Đọc chữ"], c.max_q_bytes, c.tokenizer_mode)
    a_ids, a_valid = __import__("jwm.data", fromlist=["pad_answers"]).pad_answers(
        ["abc", "xyz"], c.max_a_bytes, c.tokenizer_mode)
    box = torch.tensor([[.1, .2, .8, .6], [.2, .1, .9, .7]])
    loss, metrics = m.loss_read_v3(
        img, q_ids, q_valid, a_ids, a_valid, box,
        torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert {"qa_ce", "ctc", "box_ce", "box_iou"} <= set(metrics)
    assert m.ocr_head.weight.grad is not None and m.coord_head.weight.grad is not None


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


def test_vi_grapheme_tokenizer_roundtrip_and_compression():
    text = "Tiếng Việt có dấu: Đặng Thị Hương."
    ids = tok.encode(text, mode="vi_char")
    assert tok.decode(ids, mode="vi_char") == text
    assert len(ids) < len(tok.encode(text, mode="byte"))


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


def test_v3_renderer_exact_box_and_batch_shapes():
    fonts = find_fonts()
    c = tiny_reader_v3_cfg(image_height=256, image_width=192,
                           image_size=192, patch=16, vision_window=4,
                           max_a_bytes=96)
    for level in (1, 2, 3, 4):
        arr, text, box, meta = render_read_v3_sample(
            random.Random(100 + level), level, fonts, [], c,
            random_text=True, degrade_prob=0.0)
        assert arr.shape == (256, 192, 3)
        assert np.all((box >= 0) & (box <= 1))
        assert box[2] > box[0] and box[3] > box[1]
        assert len(tok.encode(text, mode=c.tokenizer_mode)) < c.max_a_bytes
        assert meta["n_lines"] >= 1

    b = ReadV3Batcher(c, [], fonts, [], None, 1.0, (1,), 1.0, 9)
    batch = b.batch(2)
    assert batch[0].shape == (2, 3, 256, 192)
    assert batch[5].shape == (2, 4)
    assert batch[6].all() and batch[7].all()


def test_v3_document_split_has_no_page_leakage():
    rows = []
    for page in range(30):
        for turn in range(3):
            rows.append({"img": f"page_{page}.png", "q": str(turn), "a": "ok"})
    splits = split_by_document(rows, seed=7, val_pct=20, test_pct=20)
    image_sets = [{r["img"] for r in splits[k]} for k in ("train", "val", "test")]
    assert not (image_sets[0] & image_sets[1])
    assert not (image_sets[0] & image_sets[2])
    assert not (image_sets[1] & image_sets[2])
    assert sum(map(len, splits.values())) == len(rows)


# ------------------------------------------------------- Day-4: KV cache + EOS

def test_kv_cache_equivalence():
    """Cached greedy decoding must produce byte-identical output to the
    full-reforward reference path (causality guarantees identical states)."""
    torch.manual_seed(3)
    c = tiny_reader_cfg()
    m = JWM(c).eval()
    img = torch.rand(3, 3, 64, 64)
    q = torch.randint(0, 200, (3, 8))
    qv = torch.ones(3, 8, dtype=torch.bool)
    qv[1, 5:] = False                      # ragged padding must be respected
    fast = m.generate_answer(img, q, qv, max_new=12, use_cache=True)
    slow = m.generate_answer(img, q, qv, max_new=12, use_cache=False)
    assert fast == slow, f"{fast} != {slow}"


def test_kv_cache_equivalence_moe():
    torch.manual_seed(4)
    c = tiny_reader_cfg(reasoner_moe=True, moe_experts=4, moe_topk=2, moe_shared=1)
    m = JWM(c).eval()
    img = torch.rand(2, 3, 64, 64)
    q = torch.randint(0, 200, (2, 8))
    qv = torch.ones(2, 8, dtype=torch.bool)
    fast = m.generate_answer(img, q, qv, max_new=10, use_cache=True)
    slow = m.generate_answer(img, q, qv, max_new=10, use_cache=False)
    assert fast == slow


def test_eos_loss_weight_changes_loss():
    from jwm import tokenizer as tok
    torch.manual_seed(5)
    img = torch.rand(2, 3, 64, 64)
    q = torch.randint(0, 200, (2, 8))
    qv = torch.ones(2, 8, dtype=torch.bool)
    a = torch.randint(0, 200, (2, 10))
    a[:, 5] = tok.EOS
    av = torch.ones(2, 10, dtype=torch.bool)
    torch.manual_seed(6)
    m1 = JWM(tiny_reader_cfg())
    torch.manual_seed(6)
    m2 = JWM(tiny_reader_cfg(eos_loss_weight=5.0))
    l1 = m1.loss_qa(img, q, qv, a, av)[0]
    l2 = m2.loss_qa(img, q, qv, a, av)[0]
    assert float(l2) > float(l1)           # upweighted EOS -> strictly larger CE


# ------------------------------------------------------ Day-4: anti-shortcut text

def test_random_phrase_charset_and_shape():
    from jwm.read_data import random_phrase, _RAND_CHARS
    rng = random.Random(0)
    p = random_phrase(rng, 4)
    words = p.split(" ")
    assert len(words) == 4
    assert all(2 <= len(w) <= 6 for w in words)
    assert all(ch in _RAND_CHARS for w in words for ch in w)


def test_random_text_render_and_batcher():
    fonts = find_fonts()
    arr, text = render_read_sample(random.Random(1), 2, fonts, [], size=128,
                                   random_text=True)
    assert arr.shape == (128, 128, 3) and text.strip()
    c = tiny_reader_cfg()
    b = LazyReadBatcher(c, [], fonts, [], None, synth_ratio=1.0, levels=(1,),
                        seed=0, random_text_ratio=1.0)
    img, q_ids, q_valid, a_ids, a_valid = b.batch_qa(2, "cpu")
    assert img.shape == (2, 3, 64, 64)


def test_eval_set_includes_rand_kinds():
    from jwm.read_data import build_eval_set
    c = tiny_reader_cfg()
    es = build_eval_set(c, [], find_fonts(), [], None,
                        n_synth_per_level=1, n_doc=0, n_rand_per_level=2)
    kinds = {e["kind"] for e in es}
    assert "randL1" in kinds and "synthL4" in kinds
    assert sum(k.startswith("rand") for k in (e["kind"] for e in es)) == 8


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
