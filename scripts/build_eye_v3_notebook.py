"""Build the gate-controlled Kaggle T4x2 CTPG-Eye v3 notebook."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def md(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}


def code(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": value.splitlines(True)}


cells = [
md("""# JWM CTPG-Eye v3 — Day 5 (Kaggle T4×2)

Camera-calibrated track/point geometry with differentiable BA and adaptive
training budgets. Use **Save Version → Save & Run All**, accelerator **GPU
T4×2**, Internet on, Persistence **Files only**. A failed data/probe/profile
gate blocks the expensive run; a failed causal OOD gate exports a blocked
checkpoint, never a deployable brain.
"""),
code("""import torch, os
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('gpu_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), round(p.total_memory/2**30,2),'GiB')
assert torch.cuda.device_count()==2, 'Select GPU T4 x2 before Save Version'
torch.set_float32_matmul_precision('high')
"""),
code("""%cd /kaggle/working
!rm -rf mini-world-model
!git clone https://github.com/anhsown/mini-world-model.git
%cd /kaggle/working/mini-world-model
!pip install -q -r requirements.txt huggingface_hub
!python -m pytest tests/test_geometry_math_v3.py tests/test_geometry_v3_data.py tests/test_geometric_eye_v3.py tests/test_adaptive_training.py -q
"""),
code("""from pathlib import Path
matches=list(Path('/kaggle/input').rglob('jwm_v4.pt'))
assert matches, 'Attach the private Kaggle Dataset containing jwm_v4.pt'
warmstart=str(matches[0]); print('warmstart',warmstart)
"""),
md("""## Disjoint calibrated RGB-D sources

TUM covers ordinary real motion, Bonn covers people/dynamic objects, and
TartanAir supplies exact metric indoor/outdoor geometry. Every source is
validated before any sample enters the optimizer.
"""),
code("""from pathlib import Path
import tarfile, urllib.request, zipfile
def download_extract(url, root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    archive=root/url.rsplit('/',1)[-1]; marker=root/(archive.name+'.done')
    if marker.exists(): return
    if not archive.exists():
        print('download',url,flush=True); urllib.request.urlretrieve(url,archive)
    print('extract',archive.name,flush=True)
    if archive.suffix=='.zip':
        with zipfile.ZipFile(archive) as zf: zf.extractall(root)
    else:
        with tarfile.open(archive) as tf: tf.extractall(root,filter='data')
    archive.unlink(missing_ok=True); marker.write_text('ok')

tum_root=Path('/kaggle/working/eye_v3_data/tum')
tum_urls={
'rgbd_dataset_freiburg1_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz',
'rgbd_dataset_freiburg1_desk':'https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz',
'rgbd_dataset_freiburg2_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz',
'rgbd_dataset_freiburg3_long_office_household':'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz',
'rgbd_dataset_freiburg3_walking_xyz':'https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz'}
for url in tum_urls.values(): download_extract(url,tum_root)
tum_train=[str(tum_root/k) for k in list(tum_urls)[:3]]
tum_val=[str(tum_root/'rgbd_dataset_freiburg3_long_office_household')]
tum_test=[str(tum_root/'rgbd_dataset_freiburg3_walking_xyz')]
print('TUM',len(tum_train),len(tum_val),len(tum_test))
"""),
code("""bonn_root=Path('/kaggle/working/eye_v3_data/bonn')
bonn_base='https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/'
bonn_names=['rgbd_bonn_balloon','rgbd_bonn_moving_obstructing_box',
            'rgbd_bonn_person_tracking','rgbd_bonn_balloon_tracking','rgbd_bonn_crowd3']
for name in bonn_names: download_extract(bonn_base+name+'.zip',bonn_root)
bonn_train=[str(bonn_root/x) for x in bonn_names[:3]]
bonn_val=[str(bonn_root/bonn_names[3])]; bonn_test=[str(bonn_root/bonn_names[4])]
for path in bonn_train+bonn_val+bonn_test:
    assert (Path(path)/'groundtruth.txt').exists(),path
print('Bonn',len(bonn_train),len(bonn_val),len(bonn_test))
"""),
code("""from huggingface_hub import hf_hub_download
tartan_root=Path('/kaggle/working/eye_v3_data/tartanair'); tartan_root.mkdir(parents=True,exist_ok=True)
for env in ['OldTownFall','ArchVizTinyHouseDay']:
    for kind in ['image_lcam_front.zip','depth_lcam_front.zip']:
        filename=f'{env}/Data_easy/{kind}'
        cached=hf_hub_download('theairlabcmu/tartanair2',filename,repo_type='dataset')
        marker=tartan_root/(filename.replace('/','__')+'.done')
        if not marker.exists():
            print('extract',filename,flush=True)
            with zipfile.ZipFile(cached) as zf: zf.extractall(tartan_root)
            marker.write_text('ok')
poses=sorted(tartan_root.rglob('pose_lcam_front.txt'))
assert len(poses)>=3,f'Expected >=3 trajectories, got {len(poses)}'
by_env={}
for pose in poses: by_env.setdefault(pose.parent.parents[1].name,[]).append(pose.parent)
tartan_train=[]; tartan_val=[]
for env,rows in sorted(by_env.items()):
    rows=sorted(rows); assert len(rows)>=2,env
    tartan_train += [str(p) for p in rows[:-1]]; tartan_val.append(str(rows[-1]))
print('Tartan',len(tartan_train),len(tartan_val))
"""),
md("""## Mathematical and mechanism gate

This is not a benchmark score. It proves that K changes rays, recurrent tracks
can learn motion, BA improves pose, dynamic outliers are rejected, and memory
is bounded before full training is authorized.
"""),
code("""import subprocess, json
probe='/kaggle/working/eye_v3_probe_report.json'
subprocess.run(['python','scripts/probe_eye_v3.py','--output',probe,'--track-steps','120'],check=True)
probe_report=json.loads(Path(probe).read_text()); assert probe_report['valid']
"""),
code("""from argparse import Namespace
from scripts.train_eye_v3_ddp import make_datasets
from jwm.geometry_v3_data import validate_geometry_v3_datasets
data_args=Namespace(tartan_train=tartan_train,tartan_val=tartan_val,tartan_test=[],
                    tum_train=tum_train,tum_val=tum_val,tum_test=tum_test,
                    bonn_train=bonn_train,bonn_val=bonn_val,bonn_test=bonn_test)
datasets=make_datasets(data_args)
admission=validate_geometry_v3_datasets(datasets,'/kaggle/working/dataset_validation_v3.json')
print(json.dumps(admission,indent=2))
assert admission['valid'],f"DATA BLOCKED: {admission['failures']}"
"""),
md("""## Exact graph 100-step T4×2 profile

This cell profiles the full 256px graph, not a miniature proxy. Peak memory
must remain below 88% per rank. The measured rate gives the run-time estimate.
"""),
code("""profile='/kaggle/working/eye_v3_profile.json'
cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/profile_eye_v3_ddp.py',
     '--warmstart',warmstart,'--output',profile,'--steps','100','--per-gpu-batch','1']
print(' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
profile_report=json.loads(Path(profile).read_text()); print(json.dumps(profile_report,indent=2))
assert profile_report['valid'],'PROFILE BLOCKED: unsafe memory or non-finite training'
"""),
md("""## Adaptive full curriculum

Each stage has a minimum evidence budget and a hard cap. Held-out causal/OOD
slope decides continue, LR reduction, stage advance, convergence, overfit stop
or blocked stop. Atomic `resume.pt` is written every 200 optimizer steps.
"""),
code("""import shlex
out='/kaggle/working/jwm_eye_v3'
flags=[]
for name,values in [('--tartan-train',tartan_train),('--tartan-val',tartan_val),
 ('--tum-train',tum_train),('--tum-val',tum_val),('--tum-test',tum_test),
 ('--bonn-train',bonn_train),('--bonn-val',bonn_val),('--bonn-test',bonn_test)]:
    flags += [name,*values]
cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/train_eye_v3_ddp.py',
     '--output',out,'--warmstart',warmstart,'--probe-report',probe,
     '--per-gpu-batch','1','--grad-accum','2','--checkpoint-every','200',
     '--log-every','25','--eval-windows','12',*flags]
print(' '.join(shlex.quote(x) for x in cmd),flush=True)
subprocess.run(cmd,check=True)
"""),
code("""import matplotlib.pyplot as plt
metrics_path=Path(out)/'metrics_v3.json'; history=json.loads(metrics_path.read_text())
print('evaluations',len(history),'final status checkpoint=',
      [x.name for x in Path(out).glob('jwm_eye_v3*.pt')])
steps=[x['global_step'] for x in history]
depth=[x['report']['controls']['normal']['depth_abs_rel'] for x in history]
ate=[x['report']['controls']['normal']['ate_metric'] for x in history]
gates=[sum(x['report']['gates'].values()) for x in history]
fig,ax=plt.subplots(1,3,figsize=(12,3.2))
ax[0].plot(steps,depth); ax[0].set_title('OOD Depth AbsRel ↓')
ax[1].plot(steps,ate); ax[1].set_title('OOD metric ATE ↓')
ax[2].plot(steps,gates); ax[2].axhline(7,color='r',ls='--'); ax[2].set_title('Causal gates / 7 ↑')
plt.tight_layout(); plt.savefig(Path(out)/'training_metrics.png',dpi=160); plt.show()
"""),
code("""import hashlib, shutil
release=Path('/kaggle/working/jwm_eye_v3_release')
if release.exists(): shutil.rmtree(release)
release.mkdir()
for source in [Path(out)/'jwm_eye_v3.pt',Path(out)/'jwm_eye_v3_blocked.pt',
               Path(out)/'metrics_v3.json',Path(out)/'dataset_validation_v3.json',
               Path(out)/'training_metrics.png',Path(probe),Path(profile)]:
    if source.exists(): shutil.copy2(source,release/source.name)
for model in release.glob('*.pt'):
    print(model.name,'sha256',hashlib.sha256(model.read_bytes()).hexdigest())
archive=shutil.make_archive('/kaggle/working/jwm_eye_v3_artifacts','zip',release)
print('PORTABLE OUTPUT',archive,round(Path(archive).stat().st_size/2**20,2),'MiB')
"""),
]

notebook={"cells":cells,"metadata":{"accelerator":"GPU",
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.12"}},
    "nbformat":4,"nbformat_minor":5}
path=ROOT/'jwm'/'kaggle'/'jwm_eye_physical_v3_t4x2_day05.ipynb'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print(path)
