"""Generate the portable Kaggle T4x2 synthetic-admission notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# JWM real-anchored synthetic A/B admission (Kaggle T4x2)

Two equal-initialization arms are trained for the same optimizer budget.
Evaluation is real-only. Synthetic data is admitted only when it improves the
real capability score without hiding a source regression. Enable **GPU T4 x2**,
Internet, and Persistence **Files only**.
"""),
    code("""import torch, os
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
else:
    subprocess.run(['pip','install','-q','numpy','Pillow','pytest'],check=True)
subprocess.run(['python','-m','pytest','tests/test_real_anchored_data.py','tests/test_geometry_v3_data.py','-q'],check=True)
"""),
    code("""from pathlib import Path
matches=list(Path('/kaggle/input').rglob('jwm_v4.pt'))
assert matches, 'Attach a private Kaggle Dataset containing jwm_v4.pt'
warmstart=str(matches[0]); print('warmstart',warmstart)
"""),
    markdown("""## Download and validate real anchors

The registry downloads 14 official TUM/Bonn archives with resume, archive
validation and SHA-256 manifesting. No optimizer starts if camera geometry or
split hypotheses fail.
"""),
    code("""import subprocess
subprocess.run(['python','scripts/prepare_real_anchor_data.py','--tier','starter',
 '--branch','eye','--reserve-free-gb','5',
 '--delete-archives-after-extract'],check=True)
subprocess.run(['python','scripts/validate_real_anchor_geometry.py'],check=True)
subprocess.run(['python','scripts/build_real_anchored_synthetic.py','--samples','250000','--profile-windows','96'],check=True)
subprocess.run(['python','scripts/report_dataset_pack.py'],check=True)
"""),
    markdown("""## Arm A — real only

Rerunning this cell resumes from `resume.pt` every 100 optimizer steps.
"""),
    code("""import subprocess
real_out='/kaggle/working/jwm_ablation_real_only'
cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/train_synthetic_ablation_ddp.py',
     '--arm','real-only','--output',real_out,'--warmstart',warmstart,
     '--steps','600','--per-gpu-batch','1','--grad-accum','2',
     '--checkpoint-every','100','--log-every','25']
subprocess.run(cmd,check=True)
"""),
    markdown("""## Arm B — 50% real + 50% real-anchored synthetic

Same initialization, seed, optimizer, global batch and number of steps as Arm A.
"""),
    code("""mixed_out='/kaggle/working/jwm_ablation_real_plus_synthetic'
cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/train_synthetic_ablation_ddp.py',
     '--arm','real-plus-synthetic','--output',mixed_out,'--warmstart',warmstart,
     '--steps','600','--per-gpu-batch','1','--grad-accum','2',
     '--checkpoint-every','100','--log-every','25']
subprocess.run(cmd,check=True)
"""),
    markdown("""## Admission verdict

A failed verdict means quarantine, not a broken notebook. Download the report
so we can inspect which real capability regressed.
"""),
    code("""import json, subprocess
verdict='/kaggle/working/synthetic_ablation_verdict.json'
subprocess.run(['python','scripts/admit_synthetic_ablation.py',
 '--real-only',real_out+'/probe_metrics.json',
 '--real-plus-synthetic',mixed_out+'/probe_metrics.json',
 '--output',verdict])
report=json.loads(Path(verdict).read_text())
print(json.dumps(report,indent=2))
print('DECISION:',report['decision'].upper())
"""),
    code("""import shutil
release=Path('/kaggle/working/jwm_synthetic_ablation_release')
if release.exists(): shutil.rmtree(release)
release.mkdir()
named_sources = [
    (Path(verdict), 'synthetic_ablation_verdict.json'),
    (Path(real_out)/'probe_metrics.json', 'real_only_probe_metrics.json'),
    (Path(mixed_out)/'probe_metrics.json', 'mixed_probe_metrics.json'),
    (Path('data/real_anchor_v1/dataset_pack_report_v1.json'), 'dataset_pack_report_v1.json'),
    (Path('data/real_anchor_v1/geometry_admission_v1.json'), 'geometry_admission_v1.json'),
    (Path('data/real_anchor_v1/derived/synthetic_admission_v1.json'), 'synthetic_admission_v1.json'),
]
for source, name in named_sources:
    shutil.copy2(source,release/name)
archive=shutil.make_archive('/kaggle/working/jwm_synthetic_ablation_artifacts','zip',release)
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
          "jwm_synthetic_ablation_t4x2.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print(output)
