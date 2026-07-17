# -*- coding: utf-8 -*-
"""Live dashboard for the JWM v3 training pipeline.

    python scripts/pipeline_dashboard.py   ->  http://localhost:8877

Reads data/jwm_v3_run.log + stage checkpoints/reports; auto-refreshes every 5s.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "jwm_v3_run.log"
CKPT = ROOT / "jwm" / "checkpoints"
PORT = 8877

# Round-3 budgets: PROVEN 28M scale @ batch 48 (Day-1 decision), pinned v1 camera
STAGES = [
    ("r1_reasoner_pretrain", "R1 · Reasoner Pre-training (28M)", 3000, 0.55),
    ("r2_reasoner_sft", "R2 · Reasoner SFT", 800, 0.55),
    ("g1_generator_pretrain", "G1 · Generator Pre-training (copy tower)", 2200, 0.80),
    ("g2_generator_midtrain", "G2 · Generator Mid-training (action vào)", 2200, 0.75),
    ("g3_post_text2image", "G3 · Post: Text→Image", 400, 0.80),
    ("g4_post_image2video", "G4 · Post: Image→Video (FD)", 400, 0.80),
    ("g5_post_policy", "G5 · Post: Policy → jwm_v3.pt", 700, 0.85),
]


def parse_state() -> dict:
    text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    lines = text.splitlines()
    # current stage = last ">>> STAGE x <<<" marker in the file
    cur_stage, cur_idx = None, -1
    for i in range(len(lines) - 1, -1, -1):
        m = re.match(r">>> STAGE (\w+) <<<", lines[i])
        if m:
            cur_stage, cur_idx = m.group(1), i
            break
    # last training-step line after that marker
    step = total = None
    rate = None
    metrics_line = ""
    for ln in lines[cur_idx + 1:]:
        m = re.match(r"\s*\[\s*(\d+)/(\d+)\]\s+(.*?)\s*\|\s*([\d.]+) it/s", ln)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            metrics_line = m.group(3)
            rate = float(m.group(4))
    done_reports = {}
    for name, *_ in STAGES:
        rp = CKPT / f"stage_{name}.report.json"
        if rp.exists():
            try:
                done_reports[name] = json.load(rp.open(encoding="utf-8"))
            except Exception:
                pass
    partials = {p.name: datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")
                for p in CKPT.glob("stage_*.partial.pt")}
    final = (CKPT / "jwm_v3.pt").exists()
    return {"cur_stage": cur_stage, "step": step, "total": total, "rate": rate,
            "metrics": metrics_line, "reports": done_reports, "partials": partials,
            "tail": lines[-10:], "final": final,
            "log_mtime": (datetime.fromtimestamp(LOG.stat().st_mtime).strftime("%H:%M:%S")
                          if LOG.exists() else "?")}


def eta_minutes(state: dict) -> tuple[float, float]:
    """(overall_pct, remaining_min)."""
    done_steps = 0.0
    remain_min = 0.0
    total_steps = sum(s[2] for s in STAGES)
    passed_current = False
    for name, _, steps, est_rate in STAGES:
        if name in state["reports"]:
            done_steps += steps
        elif name == state["cur_stage"] and state["step"] is not None:
            done_steps += min(state["step"], steps)
            r = state["rate"] or est_rate
            remain_min += max(0, steps - state["step"]) / r / 60
            passed_current = True
        elif passed_current or name != state["cur_stage"]:
            if name not in state["reports"] and (passed_current or state["cur_stage"] is None
                                                 or name != state["cur_stage"]):
                remain_min += steps / est_rate / 60
    return 100.0 * done_steps / total_steps, remain_min


def render() -> str:
    s = parse_state()
    pct, remain = eta_minutes(s)
    rows = []
    reached_current = False
    for name, title, steps, _ in STAGES:
        if name in s["reports"]:
            rp = s["reports"][name]
            keys = {k: v for k, v in rp.items()
                    if k.startswith(("val_", "test_")) and isinstance(v, (int, float))}
            info = " · ".join(f"{k.replace('val_', '')}={v:.3f}" for k, v in list(keys.items())[:4])
            rows.append(f"<tr class='done'><td>✔</td><td>{title}</td>"
                        f"<td>{steps}</td><td>{info or 'xong'}</td></tr>")
        elif name == s["cur_stage"] and not s["final"]:
            reached_current = True
            if s["step"] is not None:
                p = 100 * s["step"] / max(1, s["total"] or steps)
                bar = (f"<div class='bar'><div class='fill' style='width:{p:.0f}%'></div></div>"
                       f"{s['step']}/{s['total']} ({p:.0f}%) · {s['rate'] or '?'} it/s")
            else:
                bar = "đang khởi động / eval..."
            rows.append(f"<tr class='run'><td>▶</td><td>{title}</td>"
                        f"<td>{steps}</td><td>{bar}<br><span class='m'>{s['metrics']}</span></td></tr>")
        else:
            rows.append(f"<tr class='wait'><td>·</td><td>{title}</td><td>{steps}</td><td>chờ</td></tr>")
    partials = " · ".join(f"{k} @ {v}" for k, v in s["partials"].items()) or "—"
    tail = "\n".join(ln.replace("<", "&lt;") for ln in s["tail"])
    final_banner = ("<div class='final'>🎉 jwm_v3.pt ĐÃ XUẤT — pipeline hoàn tất!</div>"
                    if s["final"] else "")
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='5'>
<title>JWM v3 Pipeline</title><style>
body{{background:#050810;color:#d9f6ff;font-family:'Segoe UI',monospace;padding:24px}}
h1{{color:#38d7ff;letter-spacing:.15em;font-size:20px}}
.top{{display:flex;gap:28px;margin:14px 0;font-size:14px}}
.card{{border:1px solid rgba(91,213,255,.28);border-radius:10px;padding:10px 16px;background:rgba(7,18,31,.84)}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td{{padding:8px 10px;border-bottom:1px solid rgba(91,213,255,.15);font-size:13px}}
tr.done td{{color:#7fe7c0}} tr.run td{{color:#ffe58a}} tr.wait td{{color:#5a7f95}}
.bar{{display:inline-block;width:220px;height:10px;border:1px solid #38d7ff;border-radius:5px;vertical-align:middle;margin-right:8px}}
.fill{{height:100%;background:linear-gradient(90deg,#087acb,#38d7ff);border-radius:5px}}
.m{{color:#69c7e8;font-size:11px}}
pre{{background:#02050a;border:1px solid rgba(91,213,255,.2);border-radius:8px;padding:10px;font-size:11px;color:#8fd8f0;overflow-x:auto}}
.final{{background:#0a3;color:#fff;padding:12px;border-radius:8px;font-size:16px;margin:10px 0}}
.obar{{width:100%;height:16px;border:1px solid #38d7ff;border-radius:8px;margin:6px 0}}
.ofill{{height:100%;background:linear-gradient(90deg,#087acb,#38d7ff);border-radius:8px}}
</style></head><body>
<h1>⚙ JWM v3 — PIPELINE 7 GIAI ĐOẠN (68.65M params)</h1>
{final_banner}
<div class='obar'><div class='ofill' style='width:{pct:.1f}%'></div></div>
<div class='top'>
<div class='card'>Tổng tiến độ: <b>{pct:.1f}%</b></div>
<div class='card'>ETA còn lại: <b>~{remain/60:.1f} giờ</b></div>
<div class='card'>Log cập nhật: {s['log_mtime']}</div>
<div class='card'>Shutdown-safe ckpt: {partials}</div>
</div>
<table>{''.join(rows)}</table>
<h3 style='color:#38d7ff;margin-top:18px'>console</h3>
<pre>{tail}</pre>
<div style='color:#5a7f95;font-size:11px'>tự refresh 5s · {time.strftime('%H:%M:%S')}</div>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"dashboard: http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
