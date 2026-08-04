"""Build the resumable JWM-Eye v3.2.2 Kaggle T4x2 notebook."""

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
    md("""# JWM-Eye v3.2.2 — stage-safe factored geometry (Kaggle T4×2)

Select **GPU T4 x2**, enable Internet, and set Persistence to **Files only**.
Attach private inputs containing the latest `jwm_eye_v321_blocked.pt` (preferred;
its good tracker is reused selectively), `jwm_v4.pt` as fallback, and
`synthetic_ablation_verdict.json`. Checkpoints are atomic and rerunning the
training cell resumes from `resume.pt`.
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
requirements=repo/'requirements.txt'
if requirements.exists():
    subprocess.run(['pip','install','-q','-r',str(requirements)],check=True)
subprocess.run(['python','-m','pytest','tests/test_geometric_eye_v32.py',
                'tests/test_geometric_eye_v3.py','tests/test_geometry_v3_data.py',
                'tests/test_geometry_v321_metrics.py',
                'tests/test_adaptive_training.py','-q'],check=True)
"""),
    code("""from pathlib import Path
corrective=list(Path('/kaggle/input').rglob('jwm_eye_v321_blocked.pt'))
fallback=list(Path('/kaggle/input').rglob('jwm_v4.pt'))
assert corrective or fallback, 'Attach jwm_eye_v321_blocked.pt or jwm_v4.pt'
warmstart=str((corrective or fallback)[0])
verdicts=list(Path('/kaggle/input').rglob('synthetic_ablation_verdict.json'))
verdicts += list(Path('/kaggle/working').glob('synthetic_ablation_verdict.json'))
assert verdicts, 'Attach the admitted synthetic_ablation_verdict.json'
admission=str(verdicts[0])
print('warmstart',warmstart)
print('synthetic admission',admission)
"""),
    md("""## Build and validate the real-anchored data pack

Every dataset hypothesis is checked before any expensive training begins.
Existing extracted assets are reused safely after a resumed session.
"""),
    code("""import subprocess
subprocess.run(['python','scripts/prepare_real_anchor_data.py','--tier','starter',
 '--branch','eye','--reserve-free-gb','5','--delete-archives-after-extract'],check=True)
subprocess.run(['python','scripts/validate_real_anchor_geometry.py'],check=True)
subprocess.run(['python','scripts/build_real_anchored_synthetic.py',
 '--samples','250000','--profile-windows','96'],check=True)
subprocess.run(['python','scripts/report_dataset_pack.py'],check=True)
"""),
    code("""probe='/kaggle/working/eye_v322_probe_report.json'
subprocess.run(['python','scripts/probe_eye_v3.py','--output',probe,
                '--track-steps','120'],check=True)
"""),
    md("""## Exact v3.2.2 graph canary

This runs the same 381M architecture and physical loss graph as full training.
It blocks the long run on non-finite gradients or unsafe T4 memory use.
"""),
    code("""profile='/kaggle/working/eye_v322_profile.json'
subprocess.run(['torchrun','--standalone','--nproc_per_node=2',
 'scripts/profile_eye_v3_ddp.py','--architecture','v322',
 '--warmstart',warmstart,'--output',profile,'--steps','100',
 '--per-gpu-batch','1'],check=True)
"""),
    md("""## Adaptive full training

Held-out real causal/OOD gates—not training loss—control LR reductions, stage
advancement and stopping. Intermediate overfit rolls back to the stage-best
geometry and advances; only the final 7/7 contract can promote the model.
"""),
    code("""out='/kaggle/working/jwm_eye_v322'
cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/train_eye_v3_ddp.py',
 '--architecture','v322','--output',out,'--warmstart',warmstart,
 '--probe-report',probe,'--anchor-root','data/real_anchor_v1/raw',
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
release=Path('/kaggle/working/jwm_eye_v322_release')
if release.exists(): shutil.rmtree(release)
release.mkdir()
models=sorted(Path(out).glob('jwm_eye_v322*.pt'))
assert models, 'No final v3.2.2 model artifact found'
for source in [models[-1],Path(out)/'metrics_v322.json',
               Path(out)/'final_real_test_metrics.json',
               Path(out)/'synthetic_admission_used.json',
               Path(out)/'dataset_validation_v31.json',
               Path(out)/'stage_mixture_contract.json',
               Path(out)/'warmstart_report.json',Path(profile),Path(probe)]:
    if source.exists(): shutil.copy2(source,release/source.name)
for source in Path(out).glob('stage_*_best.pt'):
    shutil.copy2(source,release/source.name)
archive=shutil.make_archive('/kaggle/working/jwm_eye_v322_artifacts','zip',release)
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
    "nbformat": 4, "nbformat_minor": 5,
}

output = (Path(__file__).resolve().parents[1] / "jwm/kaggle" /
          "jwm_eye_v322_stage_safe_factored_t4x2.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output)
