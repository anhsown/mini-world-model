"""Build the reproducible Kaggle T4x2 Eye Physical v2 notebook."""

from pathlib import Path
import json


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


cells = [
    md("""# JWM Eye Physical v2 — Day 5 (Kaggle T4×2)

Corrective pairwise metric geometry training. The notebook first validates
dataset hypotheses, runs equal-initialization arms A–D, and admits full
training only if a pilot passes every causal OOD gate.

Run with **Save Version → Save & Run All**, accelerator **GPU T4×2**, Internet
enabled, and Persistence **Files only**. Cell output appears in the completed
version log; the portable ZIP appears under Version Output.
"""),
    code("""import os, torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available(), 'gpu_count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i),
          round(torch.cuda.get_device_properties(i).total_memory/2**30, 1), 'GiB')
assert torch.cuda.device_count() == 2, 'Select Kaggle accelerator GPU T4 x2'
torch.set_float32_matmul_precision('high')
"""),
    code("""# Source and deterministic test gate
%cd /kaggle/working
!rm -rf mini-world-model
!git clone https://github.com/anhsown/mini-world-model.git
%cd /kaggle/working/mini-world-model
!pip install -q -r requirements.txt huggingface_hub
!python -m pytest tests -q
"""),
    code("""# Semantic warm-start: attach a private Kaggle Dataset containing jwm_v4.pt.
from pathlib import Path
matches = list(Path('/kaggle/input').rglob('jwm_v4.pt'))
assert matches, 'Attach your private jwm_v4.pt Kaggle Dataset before running'
warmstart = str(matches[0])
print('warmstart:', warmstart)
"""),
    md("""## Official real RGB-D datasets

The train/validation/test sequences are disjoint. TUM supplies ordinary real
RGB-D camera motion; Bonn supplies highly dynamic people/object scenes in the
same timestamped format.
"""),
    code("""from pathlib import Path
import tarfile, urllib.request, zipfile

def download_extract(url, root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    archive = root / url.rsplit('/', 1)[-1]
    marker = root / (archive.name + '.done')
    if not marker.exists():
        if not archive.exists():
            print('download', url, flush=True)
            urllib.request.urlretrieve(url, archive)
        print('extract', archive.name, flush=True)
        if archive.suffix == '.zip':
            with zipfile.ZipFile(archive) as zf: zf.extractall(root)
        else:
            with tarfile.open(archive) as tf: tf.extractall(root, filter='data')
        archive.unlink(missing_ok=True); marker.write_text('ok')

tum_root = Path('/kaggle/working/eye_v2_data/tum')
tum_urls = {
 'rgbd_dataset_freiburg1_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz',
 'rgbd_dataset_freiburg1_desk':'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz',
 'rgbd_dataset_freiburg2_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz',
 'rgbd_dataset_freiburg3_long_office_household':'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz',
 'rgbd_dataset_freiburg3_walking_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz'}
for url in tum_urls.values(): download_extract(url, tum_root)
tum_train = [str(tum_root/k) for k in list(tum_urls)[:3]]
tum_val = [str(tum_root/'rgbd_dataset_freiburg3_long_office_household')]
tum_test = [str(tum_root/'rgbd_dataset_freiburg3_walking_xyz')]
print('TUM:', len(tum_train), len(tum_val), len(tum_test))
"""),
    code("""bonn_root = Path('/kaggle/working/eye_v2_data/bonn')
bonn_base = 'https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/'
bonn_names = ['rgbd_bonn_balloon', 'rgbd_bonn_moving_obstructing_box',
              'rgbd_bonn_person_tracking', 'rgbd_bonn_balloon_tracking',
              'rgbd_bonn_crowd3']
for name in bonn_names: download_extract(bonn_base + name + '.zip', bonn_root)
bonn_train = [str(bonn_root/name) for name in bonn_names[:3]]
bonn_val = [str(bonn_root/bonn_names[3])]
bonn_test = [str(bonn_root/bonn_names[4])]
for path in bonn_train + bonn_val + bonn_test:
    assert (Path(path)/'groundtruth.txt').exists(), path
print('Bonn:', len(bonn_train), len(bonn_val), len(bonn_test))
"""),
    md("""## TartanAir exact synthetic geometry

`OldTownFall` and `ArchVizTinyHouseDay` supply outdoor/indoor variation while
remaining compact (about 2.5 GB compressed). P000/P001 are train trajectories
and P002 is validation. The adapter decodes the official lossless
RGBA-float32 depth representation.
"""),
    code("""from huggingface_hub import hf_hub_download
import zipfile

tartan_root = Path('/kaggle/working/eye_v2_data/tartanair')
tartan_root.mkdir(parents=True, exist_ok=True)
tartan_files = []
for env in ['OldTownFall', 'ArchVizTinyHouseDay']:
    tartan_files += [f'{env}/Data_easy/image_lcam_front.zip',
                     f'{env}/Data_easy/depth_lcam_front.zip']
for filename in tartan_files:
    cached = hf_hub_download('theairlabcmu/tartanair2', filename,
                             repo_type='dataset')
    marker = tartan_root/(filename.replace('/', '__') + '.done')
    if not marker.exists():
        print('extract', filename, flush=True)
        with zipfile.ZipFile(cached) as zf: zf.extractall(tartan_root)
        marker.write_text('ok')

poses = sorted(tartan_root.rglob('pose_lcam_front.txt'))
assert len(poses) >= 3, f'Expected >=3 trajectories, found {len(poses)}'
by_env = {}
for pose in poses:
    trajectory = pose.parent
    by_env.setdefault(trajectory.parents[1].name, []).append(trajectory)
tartan_train, tartan_val = [], []
for env, trajectories in sorted(by_env.items()):
    trajectories = sorted(trajectories)
    assert len(trajectories) >= 2, f'{env} needs >=2 trajectories'
    tartan_train += [str(p) for p in trajectories[:-1]]
    tartan_val.append(str(trajectories[-1]))
print('Tartan trajectories:', [Path(p).name for p in tartan_train],
      [Path(p).name for p in tartan_val])
"""),
    md("""## Pre-training hypothesis admission

No sample is admitted merely because it downloaded successfully. Each source
must pass RGB/depth/pose/motion checks and all trajectory splits must be
scene-disjoint.
"""),
    code("""import json
from argparse import Namespace
from pathlib import Path
from scripts.train_eye_physical_v2_ddp import make_datasets, validate_datasets

data_args = Namespace(
    tartan_train=tartan_train, tartan_val=tartan_val,
    tum_train=tum_train, tum_val=tum_val, tum_test=tum_test,
    bonn_train=bonn_train, bonn_val=bonn_val, bonn_test=bonn_test)
datasets = make_datasets(data_args)
validation_path = Path('/kaggle/working/dataset_validation_v2.json')
admission = validate_datasets(datasets, validation_path)
print(json.dumps(admission, indent=2))
assert admission['valid'], 'DATA BLOCKED: inspect failed hypothesis before training'
"""),
    md("""## Equal-initialization causal ablation

A=pairwise metric base; B=+SE(3) cycle; C=+dynamic mask; D=+wrong-image
counterfactual. Atomic `resume.pt` is updated every 200 optimizer steps.
"""),
    code("""import shlex, subprocess
out = '/kaggle/working/jwm_eye_physical_v2'

def data_flags():
    flags=[]
    for name, values in [
        ('--tartan-train', tartan_train), ('--tartan-val', tartan_val),
        ('--tum-train', tum_train), ('--tum-val', tum_val), ('--tum-test', tum_test),
        ('--bonn-train', bonn_train), ('--bonn-val', bonn_val), ('--bonn-test', bonn_test)]:
        flags += [name, *values]
    return flags

cmd = ['torchrun','--standalone','--nproc_per_node=2',
       'scripts/train_eye_physical_v2_ddp.py', '--mode','ablation',
       '--output',out,'--warmstart',warmstart,'--arms','ABCD',
       '--pilot-steps','800','--per-gpu-batch','1','--grad-accum','2',
       '--checkpoint-every','200','--log-every','25','--eval-windows','18',
       *data_flags()]
print(' '.join(shlex.quote(x) for x in cmd), flush=True)
subprocess.run(cmd, check=True)
"""),
    code("""# Inspect ablation and visualize the causal verdict.
import json, matplotlib.pyplot as plt
summary = json.loads((Path(out)/'ablation_summary.json').read_text())
print(json.dumps(summary, indent=2))
arms = list(summary['arms'])
depth = [summary['arms'][a]['ood']['controls']['normal']['depth_abs_rel'] for a in arms]
ate = [summary['arms'][a]['ood']['controls']['normal']['ate_metric'] for a in arms]
gates = [sum(summary['arms'][a]['ood']['gates'].values()) for a in arms]
fig, ax = plt.subplots(1,3,figsize=(12,3.2))
ax[0].bar(arms,depth); ax[0].set_title('OOD Depth AbsRel ↓')
ax[1].bar(arms,ate); ax[1].set_title('OOD metric ATE ↓')
ax[2].bar(arms,gates); ax[2].axhline(6,color='r',ls='--'); ax[2].set_title('Causal gates / 6 ↑')
plt.tight_layout(); plt.savefig(Path(out)/'ablation.png',dpi=160); plt.show()
"""),
    md("""## Full curriculum

This cell does not override a failed ablation. It runs only when at least one
arm passes every gate; otherwise it records a scientifically useful blocked
result and preserves the pilot metrics.
"""),
    code("""if summary['full_training_admitted']:
    cmd = ['torchrun','--standalone','--nproc_per_node=2',
           'scripts/train_eye_physical_v2_ddp.py','--mode','full','--arm','auto',
           '--output',out,'--warmstart',warmstart,
           '--steps-e0','1200','--steps-e1','2200','--steps-e2','1200',
           '--per-gpu-batch','1','--grad-accum','2',
           '--checkpoint-every','200','--log-every','25','--eval-windows','24',
           *data_flags()]
    print(' '.join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)
else:
    (Path(out)/'FULL_TRAINING_BLOCKED.txt').write_text(
        'No ablation arm passed all six causal OOD gates. Do not deploy.\\n')
    print('FULL TRAINING BLOCKED — inspect ablation_summary.json')
"""),
    code("""# Compact release: model/metrics only, never raw datasets.
import hashlib, shutil
release = Path('/kaggle/working/jwm_eye_physical_v2_release')
if release.exists(): shutil.rmtree(release)
release.mkdir()
for name in ['ablation_summary.json','winner.txt','ablation.png',
             'dataset_validation_v2.json','FULL_TRAINING_BLOCKED.txt',
             'jwm_eye_physical_v2.pt','jwm_eye_physical_v2_blocked.pt']:
    candidates = [Path(out)/name, Path('/kaggle/working')/name]
    for source in candidates:
        if source.exists(): shutil.copy2(source, release/name); break
for folder in Path(out).glob('arm_*'):
    if (folder/'metrics.json').exists():
        shutil.copy2(folder/'metrics.json', release/f'{folder.name}_metrics.json')
for folder in Path(out).glob('full_arm_*'):
    if (folder/'metrics.json').exists():
        shutil.copy2(folder/'metrics.json', release/f'{folder.name}_metrics.json')
if not list(release.glob('*.pt')) and (Path(out)/'winner.txt').exists():
    winner = (Path(out)/'winner.txt').read_text().strip()
    pilot = Path(out)/f'arm_{winner}'/'final.pt'
    if pilot.exists():
        shutil.copy2(pilot, release/'jwm_eye_physical_v2_pilot_blocked.pt')
for model in release.glob('*.pt'):
    print(model.name, 'sha256=', hashlib.sha256(model.read_bytes()).hexdigest())
archive = shutil.make_archive('/kaggle/working/jwm_eye_physical_v2_artifacts',
                              'zip', release)
print('PORTABLE OUTPUT:', archive,
      round(Path(archive).stat().st_size/2**20, 2), 'MiB')
print('Download this ZIP from the completed Version Output.')
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 5}

path = (Path(__file__).resolve().parents[1] / "jwm" / "kaggle" /
        "jwm_eye_physical_v2_t4x2_day05.ipynb")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                encoding="utf-8")
print(path)
