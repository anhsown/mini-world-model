"""Blind-image control for jwm_read_v1 + MTVQA-VI retry.

Blind test: teacher-forced token accuracy with CORRECT images vs SHUFFLED
images (batch rolled by 1). If accuracy barely drops, the model is not using
the image — its 0.78 train tok_acc was language modeling, not reading.
"""

import importlib.util as u
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = u.spec_from_file_location("b", ROOT / "scripts" / "bench_read_v1.py")
b = u.module_from_spec(spec)
spec.loader.exec_module(b)

from jwm.data import pad_answers, pad_text


@torch.no_grad()
def tf_acc(model, cfg, samples, shuffle_imgs):
    accs = []
    for i0 in range(0, len(samples), 8):
        chunk = samples[i0:i0 + 8]
        if len(chunk) < 2:
            break
        img = torch.from_numpy(np.stack([s[1] for s in chunk])) \
            .permute(0, 3, 1, 2).float().div(255.0).to(b.DEVICE)
        if shuffle_imgs:
            img = torch.roll(img, 1, dims=0)
        qs = [s[3] if len(s) > 3 else b.READ_QUESTIONS[0] for s in chunk]
        refs = [s[2] for s in chunk]
        q_ids, q_valid = pad_text(qs, cfg.max_q_bytes)
        a_ids, a_valid = pad_answers(refs, cfg.max_a_bytes)
        with torch.autocast("cuda", dtype=torch.float16, enabled=b.DEVICE == "cuda"):
            _, m = model.loss_qa(img, q_ids.to(b.DEVICE), q_valid.to(b.DEVICE),
                                 a_ids.to(b.DEVICE), a_valid.to(b.DEVICE))
        accs.append(m["qa_tok_acc"])
    return sum(accs) / len(accs)


def main():
    blob = torch.load(b.CKPT, map_location="cpu", weights_only=False)
    cfg = b.JWMConfig(**blob["cfg"])
    model = b.JWM(cfg)
    model.load_state_dict(blob["model"])
    model.eval().to(b.DEVICE)
    fonts = b.find_fonts()

    results = {}
    ladder = [s for s in b.ladder_samples(fonts, None) if s[0] in
              ("T1_word_80px", "T2_line_L2", "T3_para_L4")]
    docs = b.doc_samples(32)
    for name, samples in (("synth", ladder), ("doc", docs)):
        ok = tf_acc(model, cfg, samples, shuffle_imgs=False)
        bad = tf_acc(model, cfg, samples, shuffle_imgs=True)
        results[name] = {"tf_acc_correct_img": round(ok, 4),
                         "tf_acc_shuffled_img": round(bad, 4),
                         "vision_gain": round(ok - bad, 4)}
        print(f"{name}: correct={ok:.4f} shuffled={bad:.4f} gain={ok-bad:+.4f}")

    print("== MTVQA-VI retry ==")
    try:
        rows = b.run_tier(model, cfg, b.mtvqa_samples(50), max_new=64, log=print)
        results["mtvqa"] = b.summarize(rows)
        results["mtvqa_rows"] = rows[:10]
    except Exception as e:
        print("mtvqa failed:", e)

    out = ROOT / "data" / "bench_read_blind.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
