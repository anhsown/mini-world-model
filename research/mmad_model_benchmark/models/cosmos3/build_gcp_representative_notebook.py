from __future__ import annotations

import nbformat as nbf
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Cosmos3_Nano_MMAD_Representative1400_Optimized_T4x2.ipynb"
OUTPUT = HERE / "Cosmos3_Nano_MMAD_Representative1400_Optimized_GCP.ipynb"


def main() -> None:
    nb = nbf.read(SOURCE, as_version=4)

    nb.cells[0].source = """# Cosmos 3 Nano — MMAD Representative-1400 Optimized (Google Cloud)

Recommended first run: a **2-hour pilot** on a Compute Engine `g2-standard-24`
VM (2× NVIDIA L4, 96 GB RAM) with a persistent disk. A single-L4
`g2-standard-12` is a cheaper fit test, but may use CPU offload.

This notebook runs the immutable 1,400-question representative MMAD subset,
keeps deterministic reasoning, stops after `</think>` plus A/B/C/D, and resumes
from the same GitHub checkpoint namespace as the Kaggle notebook.

Before running, expose `HF_TOKEN` and optionally `GITHUB_TOKEN` as environment
variables or enter them through the hidden prompts. Never paste tokens into a cell.
The working root defaults to `/home/jupyter/jwm-work`; set `JWM_WORK_ROOT` when
using another persistent disk mount.
"""

    nb.cells[1].source = """import sys, subprocess

packages = [
    'pillow==11.3.0', 'transformers>=5.14.0', 'accelerate',
    'bitsandbytes>=0.49.0', 'qwen-vl-utils', 'safetensors',
    'remotezip', 'requests', 'psutil'
]
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '-U', *packages
], check=True)
print('Dependencies installed. Restart the kernel once, then continue at the environment cell.')
"""

    nb.cells[2].source = """import os, sys, json, time, shutil, subprocess, random
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

default_root = '/home/jupyter/jwm-work' if Path('/home/jupyter').exists() else '/workspace/jwm-work'
WORK = Path(os.environ.get('JWM_WORK_ROOT', default_root)).expanduser().resolve()
WORK.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(WORK / 'hf_cache')

REPO = WORK / 'mini-world-model'
if REPO.exists():
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
else:
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/anhsown/mini-world-model.git', str(REPO)], check=True)

BASE = REPO / 'research/mmad_model_benchmark'
DATA = WORK / 'mmad_full_data'
CACHE = WORK / '.mmad_archive_cache'
OUT = WORK / 'cosmos3_mmad_rep1400_opt256'
SMOKE_OUT = OUT / 'smoke_5.jsonl'
FULL_OUT = OUT / 'predictions_rep1400.jsonl'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
print('persistent work root:', WORK)
print('free disk GiB:', round(shutil.disk_usage(WORK).free / 2**30, 2))
print('repo commit:', subprocess.check_output(
    ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
"""

    nb.cells[3].source = """import torch, transformers, PIL
from PIL import Image, ImageDraw, ImageFont

assert torch.cuda.is_available(), 'Create the VM with an NVIDIA GPU and install the driver.'
gpus = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    gpus.append({'index': i, 'name': p.name,
                 'vram_GiB': round(p.total_memory / 2**30, 2)})
print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__,
                  'pillow': PIL.__version__, 'gpus': gpus}, indent=2))
assert sum(x['vram_GiB'] for x in gpus) >= 24, f'Need at least 24 GiB aggregate VRAM, got {gpus}'
if not any(any(kind in x['name'] for kind in ('L4', 'A100', 'H100', 'H200')) for x in gpus):
    print('WARNING: supported experimentally, but L4/A100/H100/H200 is recommended.')
"""

    nb.cells[5].source = """# Read secrets from the VM environment; hidden prompts are only a fallback.
import getpass

hf_token = os.environ.get('HF_TOKEN', '').strip()
if not hf_token:
    hf_token = getpass.getpass('HF_TOKEN (hidden, Enter for anonymous): ').strip()
if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
print('HF auth:', 'token enabled' if hf_token else 'anonymous')

github_token = os.environ.get('GITHUB_TOKEN', '').strip()
if not github_token:
    github_token = getpass.getpass('GITHUB_TOKEN (hidden, Enter for read-only): ').strip()
if github_token:
    os.environ['GITHUB_TOKEN'] = github_token
print('GitHub checkpoint push:', 'enabled' if github_token else 'read-only')
"""

    nb.cells[6].source = nb.cells[6].source.replace(
        "# Load Cosmos 3 Nano Reasoner-only BNB8 across both T4 GPUs.",
        "# Load Cosmos 3 Nano Reasoner-only BNB8 across all visible Google Cloud GPUs.",
    )
    old = """model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map='auto',
    max_memory={0: '14GiB', 1: '14GiB', 'cpu': '24GiB'},
    low_cpu_mem_usage=True,
    offload_folder=str(WORK / 'cosmos_offload'),
    offload_state_dict=True,
    attn_implementation='sdpa',
    token=hf_token,
).eval()"""
    new = """import psutil

gpu_memory = {}
for i in range(torch.cuda.device_count()):
    total_gib = int(torch.cuda.get_device_properties(i).total_memory / 2**30)
    gpu_memory[i] = f'{max(1, total_gib - 2)}GiB'
cpu_gib = max(16, int(psutil.virtual_memory().total / 2**30) - 12)
max_memory = {**gpu_memory, 'cpu': f'{cpu_gib}GiB'}
print('model max_memory:', max_memory)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map='auto',
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    offload_folder=str(WORK / 'cosmos_offload'),
    offload_state_dict=True,
    attn_implementation='sdpa',
    token=hf_token,
).eval()"""
    if old not in nb.cells[6].source:
        raise RuntimeError('Model load block changed; update the GCP builder.')
    nb.cells[6].source = nb.cells[6].source.replace(old, new)

    nb.cells[7].source = nb.cells[7].source.replace(
        "'backend': 'Transformers/device_map=auto/T4x2/opt256/deterministic',",
        "'backend': 'Transformers/device_map=auto/GCP/opt256/deterministic',",
    )

    phase2 = nb.cells[10].source
    phase2 = phase2.replace(
        "'kaggle_t4x2_opt256_rep1400'",
        "'cosmos3_bnb8_opt256_rep1400_shared'",
    )
    phase2 = phase2.replace(
        "for checkpoint in [FULL_OUT, *Path('/kaggle/input').rglob('predictions_rep1400.jsonl')]:",
        "for checkpoint in [FULL_OUT]:",
    )
    phase2 = phase2.replace(
        "archive = shutil.make_archive('/kaggle/working/cosmos3_mmad_rep1400_opt256_artifacts', 'zip', OUT)",
        "archive = shutil.make_archive(str(WORK / 'cosmos3_mmad_rep1400_opt256_artifacts'), 'zip', OUT)",
    )
    nb.cells[10].source = phase2
    nb.cells[11].source = nb.cells[11].source.replace(
        "archive = shutil.make_archive('/kaggle/working/cosmos3_mmad_rep1400_opt256_artifacts', 'zip', OUT)",
        "archive = shutil.make_archive(str(WORK / 'cosmos3_mmad_rep1400_opt256_artifacts'), 'zip', OUT)",
    )

    nb.metadata['cloud_runtime'] = {
        'provider': 'Google Cloud',
        'recommended_machine': 'g2-standard-24',
        'pilot_hours': 2,
        'persistent_root_env': 'JWM_WORK_ROOT',
    }
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    main()
