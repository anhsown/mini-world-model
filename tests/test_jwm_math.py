"""Property tests for every math function in jwm/ (DESIGN §10).

Each test names the mathematical property it certifies. All must pass before training.
"""

from __future__ import annotations

import math

import pytest
import torch

from jwm import JWM, JWMConfig, ConvAE, merge_latent, unmerge_latent
from jwm import tokenizer as tok
from jwm.layers import build_ar_mask, build_dm_mask
from jwm.mathx import (
    apply_rope,
    bbox_from_signed,
    bbox_iou,
    bbox_to_signed,
    euler_flow_sample,
    expected_calibration_error,
    grid_coords,
    logit_normal_sigma,
    matrix_to_rot6d,
    mode_sigma,
    mrope_angles,
    project_to_so3,
    psnr,
    rf_interpolate,
    rf_velocity_target,
    rf_x0_from_v,
    rot6d_to_matrix,
    shift_sigma,
    sigma_schedule,
    temporal_delta,
    text_coords,
)

torch.manual_seed(0)
SECTIONS = (8, 4, 4)  # head_dim 32 -> half 16


# ----------------------------------------------------------------- MRoPE ----

def test_rope_relative_invariance():
    """For text tokens (t=h=w=p): <RoPE(q,p), RoPE(k,p+D)> depends only on D."""
    q = torch.randn(1, 1, 2, 32)
    k = torch.randn(1, 1, 2, 32)
    def dot_at(p, delta):
        aq = mrope_angles(text_coords(1, start=p).unsqueeze(0), SECTIONS)
        ak = mrope_angles(text_coords(1, start=p + delta).unsqueeze(0), SECTIONS)
        return (apply_rope(q, aq) * apply_rope(k, ak)).sum()
    d1 = dot_at(3.0, 5.0)
    d2 = dot_at(41.0, 5.0)
    d3 = dot_at(3.0, 6.0)
    assert torch.allclose(d1, d2, atol=1e-4), "same delta must give same dot product"
    assert not torch.allclose(d1, d3, atol=1e-3), "different delta must differ"


def test_rope_norm_preservation():
    """Rotation must preserve vector norms exactly."""
    x = torch.randn(2, 7, 4, 32)
    ang = mrope_angles(torch.randn(2, 7, 3) * 100, SECTIONS)
    y = apply_rope(x, ang)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_mrope_allocation():
    """Grid coords tile (h,w) row-major with shared t; temporal modulation = base/tps."""
    g = grid_coords(7.0, 2, 3)
    assert g.shape == (6, 3)
    assert (g[:, 0] == 7.0).all()
    assert g[:, 1].tolist() == [0, 0, 0, 1, 1, 1]
    assert g[:, 2].tolist() == [0, 1, 2, 0, 1, 2]
    assert temporal_delta(tps=2.0, tps_base=5.0) == pytest.approx(2.5)
    assert temporal_delta(tps=5.0, tps_base=5.0) == pytest.approx(1.0)


def test_mrope_axis_independence():
    """Tokens differing only on w axis must produce identical t-section angles."""
    c1 = torch.tensor([[[4.0, 2.0, 0.0]]])
    c2 = torch.tensor([[[4.0, 2.0, 3.0]]])
    a1, a2 = mrope_angles(c1, SECTIONS), mrope_angles(c2, SECTIONS)
    assert torch.allclose(a1[..., :12], a2[..., :12])      # t (8) + h (4) unchanged
    assert not torch.allclose(a1[..., 12:], a2[..., 12:])  # w section changed


# -------------------------------------------------------- rectified flow ----

def test_rectified_flow_identities():
    """x_sigma endpoints; v* constant; one-step recovery; multi-step Euler with true v*."""
    x0, eps = torch.randn(16, 4), torch.randn(16, 4)
    assert torch.allclose(rf_interpolate(x0, eps, torch.zeros(16)), x0)
    assert torch.allclose(rf_interpolate(x0, eps, torch.ones(16)), eps)
    sig = torch.rand(16)
    x_sig = rf_interpolate(x0, eps, sig)
    v = rf_velocity_target(x0, eps)
    # one-step exact recovery: x0 = x_sigma - sigma * v*
    assert torch.allclose(rf_x0_from_v(x_sig, v, sig), x0, atol=1e-5)
    # Euler from sigma=1 with the exact (constant) velocity field recovers x0
    x = euler_flow_sample(lambda xt, s: v, eps.clone(), n_steps=13, s=1.0)
    assert torch.allclose(x, x0, atol=1e-4)


def test_shift_schedule():
    """sigma(t=0)=1, sigma(t=1)=0, strictly decreasing, s>1 biases high noise."""
    for s in (1.0, 3.0, 10.0):
        t = torch.linspace(0, 1, 101)
        sig = shift_sigma(t, s)
        assert sig[0] == pytest.approx(1.0) and sig[-1] == pytest.approx(0.0)
        assert (sig[1:] < sig[:-1]).all(), "must be strictly decreasing"
    t = torch.tensor([0.5])
    assert shift_sigma(t, 3.0) > shift_sigma(t, 1.0), "s>1 pushes sigma higher"
    grid = sigma_schedule(50, 3.0)
    assert grid.shape == (51,) and grid[0] == 1.0 and grid[-1] == 0.0


def test_sigma_samplers():
    """Both samplers land in [0,1]; logit-normal is symmetric around 0.5 for mean=0."""
    ln = logit_normal_sigma((20000,))
    assert 0.0 < ln.min() and ln.max() < 1.0
    assert abs(float(ln.mean()) - 0.5) < 0.02
    md = mode_sigma((20000,), s=1.29)
    assert 0.0 <= md.min() and md.max() <= 1.0
    # s=0 reduces mode sampling to uniform: mean ~ 0.5
    md0 = mode_sigma((20000,), s=0.0)
    assert abs(float(md0.mean()) - 0.5) < 0.02


# ------------------------------------------------------------- rotations ----

def test_rot6d_roundtrip():
    """R -> 6D -> R identity; Gram-Schmidt output orthonormal with det +1."""
    r6 = torch.randn(64, 6)
    R = rot6d_to_matrix(r6)
    eye = torch.eye(3).expand(64, 3, 3)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(64), atol=1e-5)
    R2 = rot6d_to_matrix(matrix_to_rot6d(R))
    assert torch.allclose(R, R2, atol=1e-5)


def test_svd_projection():
    """project_to_so3 returns a rotation; fixed point on true rotations."""
    R = rot6d_to_matrix(torch.randn(8, 6))
    M = R + 0.05 * torch.randn(8, 3, 3)
    Rp = project_to_so3(M)
    eye = torch.eye(3).expand(8, 3, 3)
    assert torch.allclose(Rp.transpose(-1, -2) @ Rp, eye, atol=1e-4)
    assert torch.allclose(torch.det(Rp), torch.ones(8), atol=1e-4)
    assert torch.allclose(project_to_so3(R), R, atol=1e-4)


# ------------------------------------------------------- bbox + metrics ----

def test_iou_cases():
    a = torch.tensor([0.5, 0.5, 0.4, 0.4])
    assert bbox_iou(a, a) == pytest.approx(1.0, abs=1e-6)
    b = torch.tensor([0.9, 0.9, 0.1, 0.1])
    assert bbox_iou(a, b) == pytest.approx(0.0, abs=1e-6)
    # half-overlap: boxes (0,0,2,2) and (1,0,2,2) in a 0..3 world -> IoU = 1/3
    c = torch.tensor([1 / 3, 1 / 3, 2 / 3, 2 / 3])
    d = torch.tensor([2 / 3, 1 / 3, 2 / 3, 2 / 3])
    assert bbox_iou(c, d) == pytest.approx(1 / 3, abs=1e-5)


def test_bbox_signed_roundtrip():
    b = torch.rand(32, 4)
    assert torch.allclose(bbox_from_signed(bbox_to_signed(b)), b, atol=1e-6)


def test_ece():
    """Perfectly calibrated predictions -> ECE ~ 0; anti-calibrated -> large."""
    conf = torch.rand(20000)
    correct = (torch.rand(20000) < conf).float()      # calibrated by construction
    assert expected_calibration_error(conf, correct) < 0.02
    assert expected_calibration_error(conf, 1 - correct) > 0.4


def test_psnr():
    a = torch.rand(2, 3, 8, 8)
    assert (psnr(a, a) > 100).all()
    b = a + 0.1
    expected = 10 * math.log10(1.0 / 0.01)
    assert torch.allclose(psnr(a, b), torch.full((2,), expected), atol=0.1)


# ------------------------------------------------- attention invariants ----

def _tiny_model():
    cfg = JWMConfig(d_model=64, n_layers=2, n_heads=2, head_dim=32, ffn_hidden=96,
                    sigma_emb_dim=32)
    return JWM(cfg), cfg


def _rand_batch(cfg, B=2, device="cpu"):
    img = torch.rand(B, 3, cfg.image_size, cfg.image_size)
    q = torch.randint(0, 256, (B, cfg.max_q_bytes))
    qv = torch.ones(B, cfg.max_q_bytes, dtype=torch.bool)
    return img, q, qv


def test_attention_isolation_ar_never_sees_dm():
    """Hard invariant: AR hidden states are identical whatever the DM tokens are."""
    torch.manual_seed(1)
    model, cfg = _tiny_model()
    model.eval()
    img, q, qv = _rand_batch(cfg)
    emb, coords, valid = model._build_ar(img, q, qv, tok.BOG)
    B = img.shape[0]
    dm_coords = torch.tensor([[cfg.ar_dm_gap, 0.0, 0.0]]).expand(B, 1, 3)
    dm_valid = torch.ones(B, 1, dtype=torch.bool)
    sig = torch.rand(B, 1)
    with torch.no_grad():
        x1 = model.act_in(torch.randn(B, 4)).unsqueeze(1)
        x2 = model.act_in(torch.randn(B, 4) * 50).unsqueeze(1)   # wildly different DM
        h1, _ = model._run(emb, coords, valid, x1, dm_coords, dm_valid, sig)
        h2, _ = model._run(emb, coords, valid, x2, dm_coords, dm_valid, sig)
        h3, _ = model._run(emb, coords, valid)                    # no DM at all
    assert torch.allclose(h1, h2, atol=1e-6), "AR must be independent of DM values"
    assert torch.allclose(h1, h3, atol=1e-6), "AR must be identical with/without DM"


def test_attention_causality_in_ar():
    """Changing a LATER AR token must not affect EARLIER AR hidden states."""
    torch.manual_seed(2)
    model, cfg = _tiny_model()
    model.eval()
    img, q, qv = _rand_batch(cfg, B=1)
    q2 = q.clone()
    q2[0, -1] = (q2[0, -1] + 7) % 256                              # change last q byte
    with torch.no_grad():
        e1, c1, v1 = model._build_ar(img, q, qv, tok.BOG)
        e2, c2, v2 = model._build_ar(img, q2, qv, tok.BOG)
        h1, _ = model._run(e1, c1, v1)
        h2, _ = model._run(e2, c2, v2)
    upto = e1.shape[1] - 2                                         # positions before change
    assert torch.allclose(h1[:, :upto], h2[:, :upto], atol=1e-6)
    assert not torch.allclose(h1[:, -1], h2[:, -1], atol=1e-4), "trigger token must change"


def _open_gates(model, d):
    """AdaLN-zero gates are 0 at init (by design). Open them so information flows,
    as training would; gate slots are chunks 3 and 6 of the 6*d adaln output."""
    for blk in model.blocks:
        torch.nn.init.normal_(blk.adaln.weight, std=0.05)
        with torch.no_grad():
            blk.adaln.bias[2 * d : 3 * d].fill_(1.0)   # gate1
            blk.adaln.bias[5 * d : 6 * d].fill_(1.0)   # gate2


def test_dm_depends_on_ar():
    """Generator output must change when the QUESTION changes — the question reaches
    the DM pathway ONLY through the reasoner K/V, so this proves AR->DM flow works."""
    torch.manual_seed(3)
    model, cfg = _tiny_model()
    _open_gates(model, cfg.d_model)
    model.eval()
    img, q, qv = _rand_batch(cfg, B=1)
    q2 = q.clone()
    q2[0, :8] = (q2[0, :8] + 31) % 256                             # different question
    B = 1
    x = torch.randn(B, 4)
    dm_coords = torch.tensor([[cfg.ar_dm_gap, 0.0, 0.0]]).expand(B, 1, 3)
    dm_valid = torch.ones(B, 1, dtype=torch.bool)
    sig = torch.full((B, 1), 0.5)
    with torch.no_grad():
        e1, c1, v1 = model._build_ar(img, q, qv, tok.BOG)
        e2, c2, v2 = model._build_ar(img, q2, qv, tok.BOG)
        _, hd1 = model._run(e1, c1, v1, model.act_in(x).unsqueeze(1), dm_coords, dm_valid, sig)
        _, hd2 = model._run(e2, c2, v2, model.act_in(x).unsqueeze(1), dm_coords, dm_valid, sig)
    assert not torch.allclose(hd1, hd2, atol=1e-4), "DM must read the AR condition"


def test_gradient_isolation():
    """Generator (flow) loss must produce ZERO gradient on reasoner-tower params
    and on AR-only embeddings, while updating generator-tower params."""
    torch.manual_seed(4)
    model, cfg = _tiny_model()
    _open_gates(model, cfg.d_model)          # so gradients actually flow in generator
    img, q, qv = _rand_batch(cfg, B=2)
    bbox = torch.rand(2, 4) * 0.5 + 0.25
    z_img = torch.randn(2, cfg.n_lat_tokens, cfg.lat_tok_dim)
    loss, _ = model.loss_ground(img, q, qv, bbox, z_img)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    r_keys = [n for n in grads if ".r_attn." in n or ".r_ffn." in n or ".r_norm" in n]
    assert r_keys and all(grads[n] is None or torch.all(grads[n] == 0) for n in r_keys), \
        "generator loss leaked gradients into reasoner tower"
    for n in ("patch_embed.weight", "tok_emb.weight"):
        g = grads.get(n)
        assert g is None or torch.all(g == 0), f"generator loss leaked into {n}"
    g_keys = [n for n in grads if ".g_attn." in n]
    assert any(grads[n] is not None and grads[n].abs().sum() > 0 for n in g_keys), \
        "generator attention received no gradient (gates stuck?)"
    for n in ("act_v_head.weight", "lat_in.weight"):
        assert grads[n] is not None and grads[n].abs().sum() > 0, f"{n} must learn"


def test_adaln_zero_init_identity():
    """At init (gates=0) the generator block must be an identity map."""
    torch.manual_seed(5)
    model, cfg = _tiny_model()
    model.eval()
    img, q, qv = _rand_batch(cfg, B=1)
    x = torch.randn(1, 4)
    dm_emb = model.act_in(x).unsqueeze(1) + model.e_act
    dm_coords = torch.tensor([[cfg.ar_dm_gap, 0.0, 0.0]]).expand(1, 1, 3)
    dm_valid = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        emb, coords, valid = model._build_ar(img, q, qv, tok.BOG)
        _, h_dm = model._run(emb, coords, valid, dm_emb, dm_coords, dm_valid,
                             torch.full((1, 1), 0.7))
    # g_final is RMSNorm; the pre-norm stream must equal the input embedding
    expect = model.g_final(dm_emb)
    assert torch.allclose(h_dm, expect, atol=1e-5)


def test_masks():
    va = torch.tensor([[True, True, False]])
    m = build_ar_mask(va)
    assert m.shape == (1, 1, 3, 3)
    assert m[0, 0, 2, 2], "padded query keeps its diagonal (NaN guard)"
    assert not m[0, 0, 0, 1], "causality: token 0 cannot see token 1"
    # interior pad: a VALID query must not attend a PADDED key
    va2 = torch.tensor([[True, False, True]])
    m2 = build_ar_mask(va2)
    assert not m2[0, 0, 2, 1], "valid query must not attend padded key"
    assert m2[0, 0, 1, 1], "padded query keeps diagonal"
    assert m2[0, 0, 2, 0] and m2[0, 0, 2, 2], "valid keys stay visible"
    vd = torch.tensor([[True, True]])
    dm = build_dm_mask(va, vd)
    assert dm.shape == (1, 1, 2, 5)
    assert dm[0, 0, 0, 4] and dm[0, 0, 1, 0], "DM attends bidirectionally over all valid"
    assert not dm[0, 0, 0, 2], "padded AR key must be masked for DM too"


# --------------------------------------------------------- end to end -------

def test_convae_and_merge():
    ae = ConvAE(z_ch=8)
    img = torch.rand(2, 3, 64, 64)
    z = ae.encode(img)
    assert z.shape == (2, 8, 8, 8)
    t = merge_latent(z)
    assert t.shape == (2, 16, 32)
    z2 = unmerge_latent(t, C=8, grid=4)
    assert torch.allclose(z, z2, atol=1e-6), "merge/unmerge must be exact inverses"
    rec = ae.decode(z)
    assert rec.shape == img.shape and rec.min() >= 0 and rec.max() <= 1


def test_end_to_end_all_modes():
    """All three training modes: finite losses, healthy backward, sane samplers."""
    torch.manual_seed(6)
    model, cfg = _tiny_model()
    ae = ConvAE(z_ch=8)
    img, q, qv = _rand_batch(cfg, B=2)
    a = torch.randint(0, 256, (2, 8))
    av = torch.ones(2, 8, dtype=torch.bool)

    z = merge_latent(ae.encode_std(img))
    l1, m1 = model.loss_qa(img, q, qv, a, av)
    l2, m2 = model.loss_ground(img, q, qv, torch.rand(2, 4) * 0.4 + 0.3, z)
    l3, m3 = model.loss_fd(img, q, qv, z, z)
    total = l1 + l2 + l3
    assert torch.isfinite(total)
    total.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"

    model.eval()
    ans = model.generate_answer(img, q, qv, max_new=4)
    assert len(ans) == 2
    bbox, conf = model.sample_bbox(img, q, qv, z, steps=4)
    assert bbox.shape == (2, 4) and (bbox >= 0).all() and (bbox <= 1).all()
    assert (conf >= 0).all() and (conf <= 1).all()
    z_next = model.sample_next_latent(img, q, qv, z, steps=4)
    assert z_next.shape == z.shape and torch.isfinite(z_next).all()


def test_cfg_combination_analytic():
    """CFG in euler_flow_sample: v = v_u + g*(v_c - v_u), integrated over sigma 1->0.

    With constant fields the sampler is exactly x_init - v_combined; g=1 must
    reduce to the conditional-only path. (Closes verified test-gap #1.)"""
    torch.manual_seed(7)
    x_init = torch.randn(5, 3)
    v_c, v_u = torch.randn(3), torch.randn(3)
    cond = lambda x, s: v_c.expand_as(x)
    unco = lambda x, s: v_u.expand_as(x)
    for g in (0.0, 1.0, 2.0, 3.5):
        out = euler_flow_sample(cond, x_init.clone(), n_steps=9, s=1.0,
                                guidance=g, v_uncond_fn=unco)
        expect = x_init - (v_u + g * (v_c - v_u))
        assert torch.allclose(out, expect, atol=1e-5), f"CFG wrong for g={g}"
    only_cond = euler_flow_sample(cond, x_init.clone(), n_steps=9)
    assert torch.allclose(only_cond, x_init - v_c, atol=1e-5)


def test_cached_sampler_equivalence():
    """_precompute_ar + _run_dm_cached must be numerically identical to _run.

    This certifies the reasoner-caching optimization (Cosmos §5.3.1 / DESIGN §6)
    changes cost, not results. (Closes verified spec-mismatch #1.)"""
    torch.manual_seed(8)
    model, cfg = _tiny_model()
    _open_gates(model, cfg.d_model)
    model.eval()
    img, q, qv = _rand_batch(cfg, B=2)
    z = torch.randn(2, cfg.n_lat_tokens, cfg.lat_tok_dim)
    x = torch.randn(2, 4)
    emb, coords, valid = model._build_ar(img, q, qv, tok.BOG)
    lat = model.lat_in(z) + model.e_lat
    act = model.act_in(x).unsqueeze(1) + model.e_act
    dm_emb = torch.cat([lat, act], dim=1)
    dm_coords = model._dm_coords_ground(2, img.device)
    dm_valid = torch.ones(2, cfg.n_lat_tokens + 1, dtype=torch.bool)
    sig = torch.cat([torch.zeros(2, cfg.n_lat_tokens), torch.full((2, 1), 0.4)], dim=1)
    with torch.no_grad():
        _, h_ref = model._run(emb, coords, valid, dm_emb, dm_coords, dm_valid, sig)
        _, kv = model._precompute_ar(emb, coords, valid)
        h_cached = model._run_dm_cached(kv, valid, dm_emb, dm_coords, dm_valid, sig)
    assert torch.allclose(h_ref, h_cached, atol=1e-6), \
        "cached DM pathway diverged from the reference joint forward"


def test_model_cfg_paths_run():
    """Model-level CFG branches (bbox + FD) execute and return sane values."""
    torch.manual_seed(9)
    model, cfg = _tiny_model()
    model.eval()
    img, q, qv = _rand_batch(cfg, B=2)
    z = torch.randn(2, cfg.n_lat_tokens, cfg.lat_tok_dim)
    bbox, conf = model.sample_bbox(img, q, qv, z, steps=3, guidance=2.0)
    assert bbox.shape == (2, 4) and torch.isfinite(bbox).all()
    assert (conf >= 0).all() and (conf <= 1).all()
    z_next = model.sample_next_latent(img, q, qv, z, steps=3, guidance=2.0)
    assert z_next.shape == z.shape and torch.isfinite(z_next).all()


def test_qa_alignment_by_overfit():
    """Pin the CE-slice alignment in loss_qa: overfitting ONE (img, q, a) sample
    must make generate_answer reproduce the answer bytes exactly. A shifted CE
    slice cannot pass this (mutation-tested gap from the adversarial audit)."""
    torch.manual_seed(10)
    model, cfg = _tiny_model()
    img, q, qv = _rand_batch(cfg, B=1)
    from jwm.data import pad_answers
    a_ids, a_valid = pad_answers(["do"], cfg.max_a_bytes)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    loss = None
    for it in range(400):
        loss, m = model.loss_qa(img, q, qv, a_ids, a_valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if float(loss) < 0.01:
            break
    model.eval()
    ans = model.generate_answer(img, q, qv, max_new=6)
    assert ans[0] == "do", f"expected exact answer 'do', got {ans[0]!r} (final CE={float(loss):.4f})"


def test_t2i_mode_runs():
    """T2I: text-only AR + all-noisy DM; loss finite, sampler shape-correct,
    and the AR-isolation invariant holds for the text-only builder too."""
    torch.manual_seed(11)
    model, cfg = _tiny_model()
    q = torch.randint(0, 256, (2, cfg.max_q_bytes))
    qv = torch.ones(2, cfg.max_q_bytes, dtype=torch.bool)
    z = torch.randn(2, cfg.n_lat_tokens, cfg.lat_tok_dim)
    loss, m = model.loss_t2i(q, qv, z)
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    z_hat = model.sample_image(q, qv, steps=3)
    assert z_hat.shape == z.shape and torch.isfinite(z_hat).all()
    z_hat_g = model.sample_image(q, qv, steps=3, guidance=2.0)
    assert torch.isfinite(z_hat_g).all()


def test_generator_init_from_reasoner():
    """After the Cosmos-style tower copy: g_* == r_* weights, AdaLN re-zeroed,
    and the generator behaves as an identity map (gates closed)."""
    torch.manual_seed(12)
    model, cfg = _tiny_model()
    _open_gates(model, cfg.d_model)              # dirty the gates first
    model.init_generator_from_reasoner()
    for blk in model.blocks:
        assert torch.equal(blk.g_attn.wq.weight, blk.r_attn.wq.weight)
        assert torch.equal(blk.g_ffn.w_gate.weight, blk.r_ffn.w_gate.weight)
        assert torch.all(blk.adaln.weight == 0) and torch.all(blk.adaln.bias == 0)
    model.eval()
    img, q, qv = _rand_batch(cfg, B=1)
    x = torch.randn(1, 4)
    dm_emb = model.act_in(x).unsqueeze(1) + model.e_act
    dm_coords = torch.tensor([[cfg.ar_dm_gap, 0.0, 0.0]]).expand(1, 1, 3)
    dm_valid = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        emb, coords, valid = model._build_ar(img, q, qv, tok.BOG)
        _, h_dm = model._run(emb, coords, valid, dm_emb, dm_coords, dm_valid,
                             torch.full((1, 1), 0.6))
    assert torch.allclose(h_dm, model.g_final(dm_emb), atol=1e-5)


def test_moe_ffn():
    """Inkling-mini MoE: shapes, top-k sparsity, aux loss, router health, grads."""
    from jwm.moe import MoEFFN

    torch.manual_seed(13)
    m = MoEFFN(d=64, expert_hidden=32, n_experts=8, top_k=2, n_shared=1)
    x = torch.randn(3, 11, 64, requires_grad=True)
    m.train()
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert m.last_aux_loss is not None and float(m.last_aux_loss) > 0
    (y.sum() + m.last_aux_loss).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.router.weight.grad is not None
    m.eval()
    stats = m.routing_stats(torch.randn(4, 50, 64))
    assert stats["load"].shape == (8,)
    assert abs(float(stats["load"].sum()) - 1.0) < 1e-5
    assert 0 <= stats["dead_experts"] <= 8


def test_moe_reasoner_model():
    """JWM with MoE reasoner: qa loss carries aux, generator stays dense,
    generator (flow) loss still leaks NO gradient into the MoE reasoner."""
    from jwm.moe import MoEFFN
    from jwm.layers import SwiGLU

    torch.manual_seed(14)
    cfg = JWMConfig(d_model=64, n_layers=3, n_heads=2, head_dim=32, ffn_hidden=96,
                    sigma_emb_dim=32, reasoner_moe=True, moe_experts=4, moe_topk=2)
    model = JWM(cfg)
    assert isinstance(model.blocks[0].r_ffn, SwiGLU), "first layer must stay dense"
    assert isinstance(model.blocks[1].r_ffn, MoEFFN)
    assert all(isinstance(b.g_ffn, SwiGLU) for b in model.blocks), "generator stays dense"

    img, q, qv = _rand_batch(cfg, B=2)
    a = torch.randint(0, 256, (2, 8))
    av = torch.ones(2, 8, dtype=torch.bool)
    loss, m = model.loss_qa(img, q, qv, a, av)
    assert "moe_aux" in m and m["moe_aux"] > 0
    loss.backward()
    assert model.blocks[1].r_ffn.router.weight.grad is not None

    model.zero_grad(set_to_none=True)
    z = torch.randn(2, cfg.n_lat_tokens, cfg.lat_tok_dim)
    gloss, _ = model.loss_ground(img, q, qv, torch.rand(2, 4) * 0.4 + 0.3, z)
    gloss.backward()
    moe_grads = [p.grad for n, p in model.named_parameters()
                 if ".r_ffn." in n and p.grad is not None]
    assert all(torch.all(g == 0) for g in moe_grads), \
        "generator loss leaked gradients into the MoE reasoner"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
