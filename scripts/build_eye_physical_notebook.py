"""Build the Kaggle T4x2 Day-5 Eye Physical notebook."""

from pathlib import Path
import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# JWM Eye Physical v1 — Day 5 (T4×2)

LingBot-Map-inspired Geometric Context Memory, clean-room JWM implementation.
The notebook validates every dataset before training and stops at failed metric gates.
Use **Save Version → Save & Run All** with **GPU T4×2** and Persistence = Files only.
"""),
    code("""import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
assert torch.cuda.device_count() == 2, 'Select Kaggle accelerator GPU T4 x2'
"""),
    code("""# Public source repository owned by anhsown
%cd /kaggle/working
!rm -rf mini-world-model
!git clone https://github.com/anhsown/mini-world-model.git
%cd /kaggle/working/mini-world-model
!pip install -q -r requirements.txt
!python -m pytest tests -q
"""),
    md("""## Real RGB-D data

Train and validation use disjoint official TUM sequences. RGB/depth/pose timestamps are associated by the JWM adapter.
"""),
    code("""from pathlib import Path
import tarfile, urllib.request
root=Path('/kaggle/working/tum'); root.mkdir(exist_ok=True)
urls={
 'rgbd_dataset_freiburg1_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz',
 'rgbd_dataset_freiburg1_desk':'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz',
 'rgbd_dataset_freiburg2_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz',
 'rgbd_dataset_freiburg3_long_office_household':'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz'}
for name,url in urls.items():
    folder=root/name
    if folder.exists():
        continue
    archive=root/f'{name}.tgz'
    print('download', name)
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    archive.unlink()
print([p.name for p in root.iterdir() if p.is_dir()])
"""),
    code("""# Add jwm_v4.pt as a private Kaggle Dataset for semantic warm-start.
from pathlib import Path
matches=list(Path('/kaggle/input').rglob('jwm_v4.pt'))
warm=str(matches[0]) if matches else ''
print('warmstart:', warm or 'NONE — geometry trains, semantic transfer disabled')
"""),
    code("""# Pre-training hypothesis validation: analytic geometry + real TUM.
import json
from jwm.geometry_data import validate_geometry_dataset
from jwm.tum_rgbd import TUMRGBDWindowDataset, validate_tum_dataset
g=validate_geometry_dataset(output='/kaggle/working/geometry_validation.json')
train_roots=[str(root/'rgbd_dataset_freiburg1_xyz'),
             str(root/'rgbd_dataset_freiburg1_desk'),
             str(root/'rgbd_dataset_freiburg2_xyz')]
val_roots=[str(root/'rgbd_dataset_freiburg3_long_office_household')]
tum=TUMRGBDWindowDataset(train_roots,frames=8,frame_stride=3,window_stride=24)
t=validate_tum_dataset(tum)
print(json.dumps({'analytic':g,'tum':t},indent=2))
assert g['valid'] and t['valid']
"""),
    md("""## Metric-gated training

G0: exact analytic RGB-D/pose. G1: 35% real TUM + 65% analytic replay. Atomic checkpoint every 250 steps.
"""),
    code("""import shlex, subprocess
out='/kaggle/working/jwm_eye_physical_v1'
cmd=['torchrun','--standalone','--nproc_per_node=2',
     'scripts/train_eye_physical_ddp.py','--output',out,
     '--per-gpu-batch','1','--grad-accum','2',
     '--checkpoint-every','250','--log-every','25',
     '--tum-train',*train_roots,'--tum-val',*val_roots]
if warm:
    cmd += ['--warmstart',warm]
print(' '.join(shlex.quote(x) for x in cmd),flush=True)
subprocess.run(cmd,check=True)
"""),
    code("""# Inspect gates and artifacts.
import json
from pathlib import Path
outp=Path('/kaggle/working/jwm_eye_physical_v1')
print(json.dumps(json.loads((outp/'metrics.json').read_text()),indent=2))
for p in sorted(outp.iterdir()):
    print(p.name, round(p.stat().st_size/2**20,2),'MiB')
"""),
    code("""# Portable artifact appears in Version Output after Save & Run All.
import shutil
archive=shutil.make_archive('/kaggle/working/jwm_eye_physical_v1_artifacts','zip',
                            '/kaggle/working/jwm_eye_physical_v1')
print(archive, round(Path(archive).stat().st_size/2**20,2),'MiB')
"""),
]

notebook = {"cells": cells, "metadata": {"accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}
path = Path(__file__).resolve().parents[1] / "jwm" / "kaggle" / "jwm_eye_physical_t4x2_day05.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
