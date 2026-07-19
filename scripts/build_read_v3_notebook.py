"""Generate the reproducible Kaggle T4x2 notebook for JWM-Read v3."""

from __future__ import annotations

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# JWM-Read v3 — Kaggle T4×2 training

This run trains **our JWM architecture**, not Qwen/QLoRA.  V3 adds a
pre-merge local vision encoder, Vietnamese grapheme tokens, CTC OCR,
parallel coordinate tokens, shuffled-image anti-shortcut loss, document-level
data splits and metric-gated curriculum stages.

Before running: select **GPU T4 ×2**, enable **Internet** and turn
**Persistence ON**. Expected base runtime is about **7–9 hours**; metric-gate
extensions can raise the worst case to roughly **11 hours**.
"""),
    code("""# 1) Verify the Kaggle runtime
import os, subprocess, sys, torch
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)
print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
assert torch.cuda.device_count() == 2, 'Select Accelerator: GPU T4 x2 before training.'
for i in range(2):
    print(i, torch.cuda.get_device_name(i),
          round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1), 'GiB')
"""),
    code("""# 2) Clone the public v3 source
from pathlib import Path
REPO = Path('/kaggle/working/mini-world-model')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/anhsown/mini-world-model.git', str(REPO)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from jwm.configs import reader_scale_v3
cfg = reader_scale_v3()
print('source:', REPO)
print('input:', cfg.input_height, 'x', cfg.input_width,
      '| visual tokens:', cfg.n_img_tokens, '| vocab:', cfg.vocab_size)
"""),
    code("""# 3) Download the licensed Vietnamese document dataset (~11 GB)
# Research use only; do not redistribute the dataset with model artifacts.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'huggingface_hub'], check=True)
from huggingface_hub import snapshot_download
DATA = Path('/kaggle/working/vdoc')
DATA.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id='trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning',
    repo_type='dataset', local_dir=str(DATA),
    allow_patterns=['data/vdoc.jsonl', 'shards/images-*.tar'],
    max_workers=4,
)
JSONL = DATA / 'data' / 'vdoc.jsonl'
assert JSONL.exists()
print('dataset downloaded:', DATA)
"""),
    code("""# 4) Extract image shards (resume-safe: each tar gets a .done marker)
from jwm.read_data import extract_tars
extract_tars(str(DATA / 'shards'), str(DATA), log=print)
first = next(DATA.glob('images/**/*.jpg'), None)
assert first is not None, 'No extracted image found.'
print('first image:', first)
"""),
    code("""# 5) Validate data hypotheses BEFORE spending GPU training time
from jwm.read_data import find_fonts, load_corpus_lines, load_doc_pairs
from jwm.read_v3_data import curate_doc_pairs, split_by_document, validate_read_v3_data
from jwm.sdg import CameraParams

fonts = find_fonts(); assert fonts
corpus = load_corpus_lines(str(JSONL), limit=10000)
probe_pairs = load_doc_pairs(str(JSONL), str(DATA), max_answer_bytes=300,
                             limit=30000, log=print)
probe_pairs = curate_doc_pairs(probe_pairs, cfg)
probe_splits = split_by_document(probe_pairs, seed=20260719, val_pct=3, test_pct=3)
cam = CameraParams(noise_std=4.0, blur_sigma=.55, jpeg_q=68,
                   contrast=1.04, wb_shift=4.0, vignette=.08)
validation = validate_read_v3_data(cfg, fonts, corpus, probe_splits, cam,
                                   n_synth=48, n_real=24)
print(__import__('json').dumps(validation, ensure_ascii=False, indent=2))
assert validation['valid'], 'Fix failed data hypothesis before training.'
"""),
    code("""# 6) Train with DDP on both T4s; rerunning this cell resumes atomically
OUT = Path('/kaggle/working/jwm_read_v3')
OUT.mkdir(parents=True, exist_ok=True)
cmd = [
    'torchrun', '--standalone', '--nproc_per_node=2',
    'scripts/train_read_v3_ddp.py',
    '--jsonl', str(JSONL), '--images', str(DATA), '--output', str(OUT),
    '--per-gpu-batch', '3', '--grad-accum', '2',
    '--checkpoint-every', '250', '--log-every', '25',
]
print(' '.join(cmd), flush=True)
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'; env['OMP_NUM_THREADS'] = '2'
result = subprocess.run(cmd, cwd=REPO, env=env)
print('torchrun return code:', result.returncode)
assert result.returncode == 0, 'Runtime error; inspect the last log. Resume is preserved.'
"""),
    code("""# 7) Inspect completion or the exact metric gate that blocked promotion
import json
status_path = OUT / 'training_status_v3.json'
assert status_path.exists(), 'Training did not reach an artifact boundary.'
status = json.loads(status_path.read_text(encoding='utf-8'))
print(json.dumps(status, ensure_ascii=False, indent=2))
metrics_path = OUT / 'metrics_read_v3.json'
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    print('\\nFINAL METRICS')
    print(json.dumps({
        'overall': metrics['read']['overall'],
        'synthetic': metrics['read']['synthetic'],
        'documents': metrics['read']['documents'],
        'vision_control': metrics['vision_control'],
    }, ensure_ascii=False, indent=2))
else:
    print('Model stopped safely at a failed metric gate. Download blocked checkpoint + stage metrics.')
"""),
    code("""# 8) Plot learning curves from executed logs
import matplotlib.pyplot as plt
history = json.loads((OUT / 'history_read_v3.json').read_text(encoding='utf-8'))
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for stage in sorted({r['stage'] for r in history}):
    rows = [r for r in history if r['stage'] == stage]
    x = [r['global_step'] for r in rows]
    ax[0].plot(x, [r.get('loss', float('nan')) for r in rows], label=stage)
    ax[1].plot(x, [r.get('qa_tok_acc', float('nan')) for r in rows], label=stage)
    ax[2].plot(x, [r.get('ctc', float('nan')) for r in rows], label=stage)
ax[0].set_title('Joint loss'); ax[1].set_title('Teacher-forced token accuracy')
ax[2].set_title('CTC loss')
for a in ax: a.grid(alpha=.25); a.legend(fontsize=7); a.set_xlabel('global optimizer step')
plt.tight_layout(); plt.show()
"""),
    code("""# 9) Collect the files to download from Kaggle Output
import shutil
BUNDLE = Path('/kaggle/working/jwm_read_v3_bundle')
BUNDLE.mkdir(exist_ok=True)
names = [
    'jwm_read_v3.pt', 'jwm_read_v3_blocked.pt',
    'metrics_read_v3.json', 'history_read_v3.json',
    'dataset_validation_v3.json', 'training_status_v3.json',
]
for name in names:
    src = OUT / name
    if src.exists():
        shutil.copy2(src, BUNDLE / name)
print('Download this folder:', BUNDLE)
for p in sorted(BUNDLE.iterdir()):
    print(f'{p.name:36s} {p.stat().st_size / 2**20:9.1f} MiB')
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "kaggle": {"accelerator": "gpu"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).resolve().parents[1] / "jwm" / "kaggle" / "jwm_read_t4x2_v3.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(target)
