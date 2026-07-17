"""Training loops + evaluation metrics for JWM.

Metrics are chosen FOR this architecture (not blindly copied from classic ML):
  * QA        : exact-match accuracy per question kind + token accuracy + CE/PPL
  * GROUND    : IoU@0.5, mean IoU, ECE of the calibrated confidence head,
                fast-vs-slow sampler gap (4 vs 50 steps, mirrors Cosmos policy mode)
  * FD        : min-over-k PSNR (multimodal futures — Cosmos policy protocol) and
                the copy-last-frame baseline PSNR it must beat
  * latency   : ms per query on the deployment GPU
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .config import JWMConfig
from .data import ModeBatcher, imgs_to_float, pad_text
from .mathx import bbox_iou, expected_calibration_error, psnr
from .model import JWM, ConvAE, merge_latent, unmerge_latent


# ----------------------------------------------------------------------------
# ConvAE pretraining (stage 0 — then frozen, like Cosmos' frozen VAE)
# ----------------------------------------------------------------------------

def train_convae(ae: ConvAE, images_u8: torch.Tensor, device, steps=800, bs=64,
                 lr=2e-3, log_every=100, log=print):
    ae.to(device).train()
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)
    n = images_u8.shape[0]
    hist = []
    for it in range(steps):
        idx = torch.randint(0, n, (bs,))
        x = imgs_to_float(images_u8[idx], device)
        rec = ae.decode(ae.encode(x))
        loss = F.mse_loss(rec, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss))
        if it % log_every == 0 or it == steps - 1:
            log(f"  [convae {it:4d}/{steps}] recon_mse={loss:.5f}  psnr={float(psnr(rec, x).mean()):.2f}dB")
    ae.eval()
    with torch.no_grad():
        sample = imgs_to_float(images_u8[torch.randperm(n)[: min(512, n)]], device)
        ae.fit_stats(sample)
    for p in ae.parameters():
        p.requires_grad_(False)
    return hist


# ----------------------------------------------------------------------------
# main training stage
# ----------------------------------------------------------------------------

def train_stage(
    model: JWM, ae: ConvAE, split: dict, cfg: JWMConfig, device,
    steps: int, lr: float, batch_size: int = 32,
    mode_probs: dict | None = None, warmup: int = 100,
    seed: int = 0, log_every: int = 50, log=print, use_amp: bool = True,
    ckpt_fn=None, ckpt_every: int = 500,
):
    """One curriculum stage. Returns per-step history for plotting.

    ckpt_fn(steps_done): optional periodic-checkpoint callback (every ckpt_every
    steps and once at the end) — makes long runs survive machine shutdowns.
    """
    mode_probs = mode_probs or {"qa": 0.45, "ground": 0.35, "fd": 0.20}
    batcher = ModeBatcher(split, cfg, seed=seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)

    def lr_at(it):
        if it < warmup:
            return lr * (it + 1) / warmup
        p = (it - warmup) / max(1, steps - warmup)
        return lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = {"step": [], "mode": [], "loss": [], "lr": [], "metrics": []}
    t0 = time.perf_counter()
    for it in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        mode = batcher.pick_mode(mode_probs)          # step-synchronous mode selection
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if mode == "qa":
                img, q, qv, a, av = batcher.batch_qa(batch_size, device)
                loss, m = model.loss_qa(img, q, qv, a, av)
            elif mode == "ground":
                img, q, qv, bbox = batcher.batch_ground(batch_size, device)
                with torch.no_grad():
                    z = merge_latent(ae.encode_std(img))
                loss, m = model.loss_ground(img, q, qv, bbox, z)
            elif mode == "t2i":
                img, q, qv = batcher.batch_t2i(batch_size, device)
                with torch.no_grad():
                    z_tgt = merge_latent(ae.encode_std(img))
                loss, m = model.loss_t2i(q, qv, z_tgt)
            else:
                img, img1, q, qv = batcher.batch_fd(batch_size, device)
                with torch.no_grad():
                    z_cur = merge_latent(ae.encode_std(img))
                    z_nxt = merge_latent(ae.encode_std(img1))
                loss, m = model.loss_fd(img, q, qv, z_cur, z_nxt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        hist["step"].append(it)
        hist["mode"].append(mode)
        hist["loss"].append(float(loss))
        hist["lr"].append(lr_at(it))
        hist["metrics"].append(m)
        if it % log_every == 0 or it == steps - 1:
            rate = (it + 1) / (time.perf_counter() - t0)
            log(f"  [{it:5d}/{steps}] mode={mode:6s} loss={float(loss):.4f} "
                + " ".join(f"{k}={v:.4f}" for k, v in m.items())
                + f" | {rate:.1f} it/s")
        if ckpt_fn is not None and ((it + 1) % ckpt_every == 0 or it == steps - 1):
            try:
                ckpt_fn(it + 1)
            except Exception as exc:  # checkpointing must never kill training
                log(f"  (checkpoint save failed: {exc})")
    return hist


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def eval_qa(model: JWM, split: dict, cfg: JWMConfig, device, n: int = 200, log=print):
    model.eval()
    d = split["qa"]
    n = min(n, len(d["q"]))
    correct, per_kind = 0, {}
    preds = []
    for i0 in range(0, n, 32):
        sl = slice(i0, min(n, i0 + 32))
        img = imgs_to_float(d["img"][sl], device)
        q_ids, q_valid = pad_text(d["q"][sl], cfg.max_q_bytes)
        ans = model.generate_answer(img, q_ids.to(device), q_valid.to(device))
        for j, a_hat in enumerate(ans):
            i = i0 + j
            gt = d["a"][i].strip().lower()
            ok = a_hat.strip().lower() == gt
            kind = d["meta"][i]["kind"]
            per_kind.setdefault(kind, [0, 0])
            per_kind[kind][1] += 1
            per_kind[kind][0] += int(ok)
            correct += int(ok)
            preds.append({"q": d["q"][i], "gt": gt, "pred": a_hat.strip().lower(), "ok": ok})
    acc = correct / n
    kinds = {k: v[0] / v[1] for k, v in per_kind.items()}
    log(f"  QA exact-match: {acc:.3f} on {n} | per-kind: "
        + " ".join(f"{k}={v:.2f}" for k, v in kinds.items()))
    return {"acc": acc, "per_kind": kinds, "preds": preds}


@torch.no_grad()
def eval_ground(model: JWM, ae: ConvAE, split: dict, cfg: JWMConfig, device,
                n: int = 200, steps: int = 50, shift: float = 3.0, log=print):
    model.eval()
    d = split["ground"]
    n = min(n, len(d["q"]))
    ious, confs = [], []
    records = []
    for i0 in range(0, n, 32):
        sl = slice(i0, min(n, i0 + 32))
        img = imgs_to_float(d["img"][sl], device)
        q_ids, q_valid = pad_text(d["q"][sl], cfg.max_q_bytes)
        z = merge_latent(ae.encode_std(img))
        bbox, conf = model.sample_bbox(img, q_ids.to(device), q_valid.to(device), z,
                                       steps=steps, shift=shift)
        gt = d["bbox"][sl].to(device)
        iou = bbox_iou(bbox, gt)
        ious.append(iou.cpu())
        confs.append(conf.cpu())
        for j in range(bbox.shape[0]):
            records.append({"q": d["q"][i0 + j], "bbox": bbox[j].cpu().tolist(),
                            "gt": gt[j].cpu().tolist(), "iou": float(iou[j]),
                            "conf": float(conf[j])})
    iou = torch.cat(ious)
    conf = torch.cat(confs)
    res = {
        "iou_at_05": float((iou >= 0.5).float().mean()),
        "miou": float(iou.mean()),
        "ece": expected_calibration_error(conf, (iou >= 0.5)),
        "records": records,
        "iou_all": iou.tolist(),
        "conf_all": conf.tolist(),
    }
    log(f"  GROUND[{steps} steps]: IoU@0.5={res['iou_at_05']:.3f} mIoU={res['miou']:.3f} "
        f"ECE={res['ece']:.3f} (n={n})")
    return res


@torch.no_grad()
def eval_fd(model: JWM, ae: ConvAE, split: dict, cfg: JWMConfig, device,
            n: int = 64, k: int = 4, steps: int = 50, shift: float = 3.0, log=print):
    """min-over-k PSNR of sampled futures vs recorded future + copy baseline."""
    model.eval()
    d = split["fd"]
    n = min(n, len(d["q"]))
    best_psnrs, copy_psnrs = [], []
    samples_vis = []
    for i0 in range(0, n, 16):
        sl = slice(i0, min(n, i0 + 16))
        img = imgs_to_float(d["img"][sl], device)
        img1 = imgs_to_float(d["img1"][sl], device)
        q_ids, q_valid = pad_text(d["q"][sl], cfg.max_q_bytes)
        z_cur = merge_latent(ae.encode_std(img))
        B = img.shape[0]
        preds = []
        for _ in range(k):
            z_hat = model.sample_next_latent(img, q_ids.to(device), q_valid.to(device),
                                             z_cur, steps=steps, shift=shift)
            rec = ae.decode_std(unmerge_latent(z_hat, C=cfg.z_ch, grid=cfg.lat_grid))
            preds.append(rec)
        ps = torch.stack([psnr(p, img1) for p in preds])        # (k, B)
        best_psnrs.append(ps.max(dim=0).values.cpu())
        copy_psnrs.append(psnr(img, img1).cpu())                 # copy-last-frame baseline
        if i0 == 0:
            samples_vis = [img[:4].cpu(), img1[:4].cpu(), preds[0][:4].cpu()]
    best = torch.cat(best_psnrs)
    copy = torch.cat(copy_psnrs)
    res = {"fd_psnr_min_over_k": float(best.mean()), "fd_psnr_copy_baseline": float(copy.mean()),
           "beats_copy_frac": float((best > copy).float().mean()), "vis": samples_vis}
    log(f"  FD[k={k}]: PSNR(best-of-{k})={res['fd_psnr_min_over_k']:.2f}dB vs "
        f"copy-baseline={res['fd_psnr_copy_baseline']:.2f}dB | beats copy on "
        f"{res['beats_copy_frac']*100:.0f}% of pairs (n={n})")
    return res


@torch.no_grad()
def eval_t2i(model: JWM, ae: ConvAE, split: dict, cfg: JWMConfig, device,
             n: int = 48, steps: int = 50, shift: float = 3.0, log=print):
    """T2I self-consistency: generate an image from the caption, then let the
    model's OWN reasoner verify it — positive probe (object from caption must be
    'có') and negative probe (absent color must be 'không'). Plus latent-space
    MSE vs the paired reference (single-reference caveat: generation is
    stochastic, so MSE is secondary)."""
    from .sdg import PALETTE

    model.eval()
    d = split["t2i"]
    n = min(n, len(d["q"]))
    pos_ok, neg_ok, mses = 0, 0, []
    vis = []
    for i0 in range(0, n, 16):
        sl = slice(i0, min(n, i0 + 16))
        caps = d["q"][sl]
        q_ids, q_valid = pad_text(caps, cfg.max_q_bytes)
        z_hat = model.sample_image(q_ids.to(device), q_valid.to(device),
                                   steps=steps, shift=shift)
        rec = ae.decode_std(unmerge_latent(z_hat, C=cfg.z_ch, grid=cfg.lat_grid))
        z_ref = merge_latent(ae.encode_std(imgs_to_float(d["img"][sl], device)))
        mses.append(F.mse_loss(z_hat, z_ref).item())
        if i0 == 0:
            vis = [rec[:6].cpu(), imgs_to_float(d["img"][sl], "cpu")[:6]]
        # self-consistency probes on the GENERATED image
        for j, cap in enumerate(caps):
            color_in = next((c for c in PALETTE if f"màu {c}" in cap), None)
            color_out = next((c for c in PALETTE if f"màu {c}" not in cap), None)
            img_gen = rec[j : j + 1]
            if color_in:
                qi, qv = pad_text([f"có vật màu {color_in} nào không?"], cfg.max_q_bytes)
                ans = model.generate_answer(img_gen, qi.to(device), qv.to(device))[0]
                pos_ok += int("có" in ans.lower() or "co" in ans.lower())
            if color_out:
                qi, qv = pad_text([f"có vật màu {color_out} nào không?"], cfg.max_q_bytes)
                ans = model.generate_answer(img_gen, qi.to(device), qv.to(device))[0]
                neg_ok += int("không" in ans.lower() or "khong" in ans.lower())
    res = {"t2i_self_consistency_pos": pos_ok / n, "t2i_self_consistency_neg": neg_ok / n,
           "t2i_latent_mse": float(sum(mses) / len(mses)), "vis": vis}
    log(f"  T2I[{steps} steps]: self-consistency pos={res['t2i_self_consistency_pos']:.2f} "
        f"neg={res['t2i_self_consistency_neg']:.2f} latent-mse={res['t2i_latent_mse']:.3f} (n={n})")
    return res


@torch.no_grad()
def measure_latency(model: JWM, ae: ConvAE, split: dict, cfg: JWMConfig, device,
                    reps: int = 20, log=print):
    """Per-query wall-clock on the deployment GPU, batch=1 (robotics-style)."""
    model.eval()
    d = split["ground"]
    img = imgs_to_float(d["img"][:1], device)
    q_ids, q_valid = pad_text(d["q"][:1], cfg.max_q_bytes)
    q_ids, q_valid = q_ids.to(device), q_valid.to(device)
    z = merge_latent(ae.encode_std(img))
    out = {}
    for name, fn in {
        "qa_answer": lambda: model.generate_answer(img, q_ids, q_valid),
        "ground_50step": lambda: model.sample_bbox(img, q_ids, q_valid, z, steps=50),
        "ground_4step": lambda: model.sample_bbox(img, q_ids, q_valid, z, steps=4),
        "fd_50step": lambda: model.sample_next_latent(img, q_ids, q_valid, z, steps=50),
        "fd_4step": lambda: model.sample_next_latent(img, q_ids, q_valid, z, steps=4),
    }.items():
        fn()                                   # warmup
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if device != "cpu":
            torch.cuda.synchronize()
        out[name] = (time.perf_counter() - t0) / reps * 1000
        log(f"  latency {name}: {out[name]:.1f} ms")
    return out
