"""Generate the gated real-anchored full Eye training notebook."""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    md("""# JWM Eye full training — real-anchored v1 (Kaggle T4x2)

This run consumes only synthetic data that passed the real-heldout A/B gate.
Checkpoint selection uses balanced real TUM/Bonn validation; the real test set
is evaluated once at the end. Enable **GPU T4 x2**, Internet and Persistence
**Files only**.
"""),
    code("""import torch
print('torch',torch.__version__,'cuda',torch.cuda.is_available())
print('gpu_count',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i,torch.cuda.get_device_name(i),round(p.total_memory/2**30,2),'GiB')
assert torch.cuda.device_count()==2, 'Select GPU T4 x2'
torch.set_float32_matmul_precision('high')
"""),
    code("""from pathlib import Path
import subprocess, os
os.chdir('/kaggle/working')
repo=Path('/kaggle/working/mini-world-model')
if not (repo/'.git').exists():
    subprocess.run(['git','clone','https://github.com/anhsown/mini-world-model.git',str(repo)],check=True)
else:
    subprocess.run(['git','-C',str(repo),'pull','--ff-only'],check=True)
os.chdir(repo)
subprocess.run(['pip','install','-q','-r',str(repo/'requirements.txt')],check=True)
subprocess.run(['python','-m','pytest','tests/test_real_anchored_data.py',
                'tests/test_geometry_v3_data.py','tests/test_adaptive_training.py','-q'],check=True)
"""),
    md("""## Required inputs

Attach the private Kaggle Dataset containing `jwm_v4.pt`. The A/B verdict can
either remain in `/kaggle/working` from the preceding notebook or be attached
as another private Kaggle Dataset.
"""),
    code("""from pathlib import Path
warmstarts=list(Path('/kaggle/input').rglob('jwm_v4.pt'))
assert warmstarts, 'Attach the private Kaggle Dataset containing jwm_v4.pt'
warmstart=str(warmstarts[0])
verdict_candidates=[Path('/kaggle/working/synthetic_ablation_verdict.json')]
verdict_candidates += list(Path('/kaggle/input').rglob('synthetic_ablation_verdict.json'))
verdict_candidates=[p for p in verdict_candidates if p.exists()]
assert verdict_candidates, 'Attach/upload synthetic_ablation_verdict.json from the A/B run'
admission=str(verdict_candidates[0])
print('warmstart',warmstart)
print('synthetic admission',admission)
"""),
    code("""import subprocess
subprocess.run(['python','scripts/prepare_real_anchor_data.py','--tier','starter',
 '--branch','eye','--reserve-free-gb','5','--delete-archives-after-extract'],check=True)
subprocess.run(['python','scripts/validate_real_anchor_geometry.py'],check=True)
subprocess.run(['python','scripts/build_real_anchored_synthetic.py',
 '--samples','250000','--profile-windows','96'],check=True)
subprocess.run(['python','scripts/report_dataset_pack.py'],check=True)
"""),
    md("""## Mechanism probes

These are graph/mechanism checks, separate from the completed data A/B test.
They must pass before the expensive adaptive run starts.
"""),
    code("""probe='/kaggle/working/eye_v3_probe_report.json'
subprocess.run(['python','scripts/probe_eye_v3.py','--output',probe,
                '--track-steps','120'],check=True)
"""),
    md("""## Adaptive full training

Rerunning this cell resumes atomically from `resume.pt`. Stages have minimum
evidence budgets and hard caps; real heldout causal/OOD metrics—not train
loss—decide advance, LR reduction, convergence or blocked stop.
"""),
    code("""out='/kaggle/working/jwm_eye_real_anchored_v1'
cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/train_eye_v3_ddp.py',
 '--output',out,'--warmstart',warmstart,'--probe-report',probe,
 '--anchor-root','data/real_anchor_v1/raw',
 '--synthetic-profile','data/real_anchor_v1/derived/eye_real_anchor_profile_v1.json',
 '--synthetic-admission',admission,'--synthetic-train-samples','250000',
 '--allow-partial-data','--per-gpu-batch','1','--grad-accum','2',
 '--checkpoint-every','200','--log-every','25','--eval-windows','16']
subprocess.run(cmd,check=True)
"""),
    code("""import json
out_path=Path(out)
test=json.loads((out_path/'final_real_test_metrics.json').read_text())
print('FINAL TEST VALID',test['valid'])
print(json.dumps(test['summary'],indent=2))
print(json.dumps(test['gates'],indent=2))
"""),
    code("""import shutil, hashlib
release=Path('/kaggle/working/jwm_eye_real_anchored_release')
if release.exists(): shutil.rmtree(release)
release.mkdir()
models=sorted(Path(out).glob('jwm_eye_v31*.pt'))
assert models, 'No final model artifact found'
for source in [models[-1],Path(out)/'metrics_v31.json',
               Path(out)/'final_real_test_metrics.json',
               Path(out)/'synthetic_admission_used.json',
               Path(out)/'dataset_validation_v31.json',
               Path(out)/'metric_catalog_v2.json',Path(probe)]:
    if source.exists(): shutil.copy2(source,release/source.name)
archive=shutil.make_archive('/kaggle/working/jwm_eye_real_anchored_artifacts','zip',release)
for model in release.glob('*.pt'):
    print(model.name,'sha256',hashlib.sha256(model.read_bytes()).hexdigest())
print('PORTABLE OUTPUT',archive,round(Path(archive).stat().st_size/2**20,2),'MiB')
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = (Path(__file__).resolve().parents[1] / "jwm/kaggle" /
          "jwm_eye_full_real_anchored_t4x2.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print(output)
