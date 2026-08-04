"""Build the PIADE B5 notebook for Kaggle T4x2.

Schema is now known from the real files, so this no longer guesses column names.

raw_data.csv — 429,394 rows, 10 columns, 5 machines (s_1..s_5), 2020-01 to 2022-01:
    interval_start  ISO8601, MIXED format (some rows carry .%f, some do not)
    equipment_ID    s_1..s_5
    alarm           133 codes; A_000 means NO alarm (78.55% of rows)
    type            production / performance_loss / downtime / idle / scheduled_downtime
    start, end      epoch floats, redundant with interval_start and elapsed
    elapsed         MILLISECONDS
    pi, po          CUMULATIVE package counters, must be differenced
    speed           packages/hour

Facts that shaped the design, all verified against the real file:

* `alarm != 'A_000'` is identical to `type == 'downtime'` on all 429,394 rows.
* LABEL LEAK: the last window row's `elapsed` is the duration of an interval that
  has not finished at cutoff time, so it is future information. It is censored.
* THE DESIGN DECISION. Sampling windows uniformly gives a task that one
  categorical field solves: "last row is production" alone reaches AUPRC 0.4995
  against a 0.3180 floor, exactly matching the elapsed-based leak rule, so
  censoring elapsed changed nothing. Beating that would only prove the model can
  read one word. Episodes are therefore restricted to windows whose last row is
  `production` — "the machine is running, will it break down within H seconds?".
  That variant has a 0.5691 positive rate, kills the categorical shortcut, and
  leaves persistence at only +0.08 over the floor, so a model has to read the
  degradation trend to win.
* Positive rate by horizon on uniform sampling: 30s 0.2542, 60s 0.2651,
  120s 0.2962, 3600s 0.8155. The one-hour horizon is too imbalanced to be useful.
* Because the positive rate is now ~0.57, AUROC and Brier are the informative
  headline metrics; AUPRC is still reported but its floor is high.

Usage:
    python build_piade_notebook.py
    -> PIADE_B5_Cosmos3_Kaggle.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "PIADE_B5_Cosmos3_Kaggle.ipynb"

RECORD = "7071747"
MODEL_ID = "ThePyProgrammer/Cosmos3-Nano-reasoner-bnb8-vllm-und-only"
BASE_MODEL_ID = "nvidia/Cosmos3-Nano"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


# ==========================================================================
md(f"""
# PIADE B5 — Cosmos 3 Nano Reasoner trên Kaggle T4×2

**Dataset:** Packaging Industry Anomaly DEtection, Zenodo `{RECORD}`, CC-BY-4.0
**Model:** `{MODEL_ID}` (nhánh reasoner ~9B, BNB8)

## Bài toán

Máy **đang chạy** (`production`). Cho Cosmos xem **12 khoảng vận hành gần nhất**, hỏi:

1. **Chẩn đoán tự do** — log này cho thấy gì, máy đang ở tình trạng nào?
   *(đọc tay, không chấm điểm — mã cảnh báo đã ẩn danh nên không có ground truth)*
2. **Dự đoán** — trong H giây tới máy có dừng vì lỗi không? *(chấm bằng AUROC / Brier / AUPRC)*

Cùng bộ episode đó chạy qua **3 cách biểu diễn**: `text` / `plot` / `text+plot`.
Câu hỏi chính là **cách nào Cosmos xử lý tốt nhất**.

## Vì sao chỉ lấy cửa sổ đang `production`

Nếu lấy mẫu đều trên mọi cửa sổ, bài toán bị **một trường phân loại duy nhất**
giải mất: chỉ cần "dòng cuối là production" đã đạt **AUPRC 0.4995** trên sàn
0.3180 — **bằng đúng** luật rò rỉ dựa trên `elapsed`, nghĩa là che `elapsed`
chẳng thay đổi gì. Thắng được nó cũng chỉ chứng minh model đọc được một chữ.

Lọc còn cửa sổ đang chạy thì:

| | Lấy mẫu đều | **Chỉ `production`** |
|---|---|---|
| Tỉ lệ dương tính | 0.3180 | **0.5691** — cân bằng |
| Shortcut theo `type` | AUPRC 0.4995 | **bị triệt tiêu** |
| Luật rò rỉ `elapsed` | 0.4995 | AUROC 0.5000 — vô dụng |
| Persistence | 0.3738 | 0.6493 (lift chỉ +0.08) |

Muốn thắng, model phải đọc được **xu hướng suy giảm** — tốc độ tụt, sản lượng
giảm, cảnh báo gần đây — chứ không đọc một từ khoá.

## Ba điều đã kiểm trên dữ liệu thật

- `alarm != 'A_000'` **trùng khớp 100%** với `type == 'downtime'` (429,394 dòng).
- `elapsed` của dòng cuối là thông tin tương lai → **đã che**.
- Tỉ lệ dương tính ~0.57 nên **AUROC và Brier** mới là metric chính; AUPRC vẫn
  báo nhưng sàn của nó cao.

## Cấu hình Kaggle

**Accelerator: GPU T4 × 2** · **Internet: On**. Chạy hết PHASE 3 xem output rồi
mới vào PHASE 4. Lần chạy dài dùng **Save Version → Run All**.
""")

# ==========================================================================
md("## Cài đặt — chạy 1 lần rồi **restart session**, sau đó bỏ qua cell này")

code("""
import sys, sysconfig, shutil, subprocess
from pathlib import Path

purelib = Path(sysconfig.get_paths()['purelib'])
shutil.rmtree(purelib / 'PIL', ignore_errors=True)
for stale in list(purelib.glob('Pillow-*.dist-info')) + list(purelib.glob('pillow-*.dist-info')):
    shutil.rmtree(stale, ignore_errors=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
                '--force-reinstall', '--no-deps', 'pillow==11.3.0'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',
                'transformers>=5.14.0', 'accelerate', 'bitsandbytes>=0.49.0',
                'qwen-vl-utils', 'safetensors'], check=True)
print('Xong. BAY GIO restart session mot lan, va dung chay lai cell nay.')
""")

# ==========================================================================
md("## PHASE 0 — Tải và kiểm dữ liệu")

code(f"""
import hashlib, json, os, time, random
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

RECORD  = '{RECORD}'
WORK    = Path('/kaggle/working' if Path('/kaggle/working').exists() else '.')
DATA    = WORK / 'piade_raw'
OUT_DIR = WORK / 'piade_out'
ARMS    = OUT_DIR / 'arms'
for p in (DATA, OUT_DIR, ARMS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260801
random.seed(SEED); np.random.seed(SEED)

NO_ALARM = 'A_000'          # ma cho biet KHONG co canh bao
WINDOW_N = 12               # so khoang cho model xem
HORIZON  = int(os.environ.get('PIADE_HORIZON', 120))       # giay
N_EPISODES = int(os.environ.get('PIADE_N_EPISODES', 500))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

source_files = []
for name in ('raw_data.csv',):
    target = DATA / name
    if not target.exists():
        url = f'https://zenodo.org/api/records/{{RECORD}}/files/{{name}}/content'
        print('tai', name, '...', flush=True)
        urllib.request.urlretrieve(url, target)
    source_files.append({{'file': name, 'bytes': target.stat().st_size,
                         'sha256': sha256(target)}})
    print(json.dumps(source_files[-1]))

raw = pd.read_csv(DATA / 'raw_data.csv', low_memory=False)
# Dinh dang timestamp KHONG dong nhat (co dong co .%f, co dong khong).
# Khong ep format='ISO8601' thi pandas se am tham tra ve NaT.
raw['ts'] = pd.to_datetime(raw['interval_start'], format='ISO8601')
assert raw['ts'].isna().sum() == 0, 'Con timestamp khong parse duoc'

raw['has_alarm']   = raw['alarm'].ne(NO_ALARM)
raw['elapsed_s']   = raw['elapsed'] / 1000.0          # goc la MILLI-giay
raw = raw.sort_values(['equipment_ID', 'ts']).reset_index(drop=True)
raw['pi_d'] = raw.groupby('equipment_ID')['pi'].diff().fillna(0).clip(lower=0)
raw['po_d'] = raw.groupby('equipment_ID')['po'].diff().fillna(0).clip(lower=0)

print('\\nshape', raw.shape, '| may:', sorted(raw.equipment_ID.unique()))
print('thoi gian:', raw.ts.min(), '->', raw.ts.max())
print('ti le co canh bao:', round(raw.has_alarm.mean(), 4))
print('trung khop alarm!=A_000 <-> type==downtime:',
      (raw.has_alarm == (raw.type == 'downtime')).mean())
print('elapsed(giay) median:', round(raw.elapsed_s.median(), 2))
""")

# ==========================================================================
md("""
## PHASE 1 — Đóng băng episode

Cửa sổ 12 khoảng, cutoff tại **thời điểm bắt đầu** khoảng cuối. Nhãn = có bất kỳ
cảnh báo nào trong `HORIZON` giây sau cutoff hay không.

Chia tập **theo máy** để test không dùng chung máy với các baseline có huấn luyện.
""")

code("""
GROUPS = {m: g.reset_index(drop=True) for m, g in raw.groupby('equipment_ID')}
machines = sorted(GROUPS)
rng = np.random.RandomState(SEED); rng.shuffle(machines)
test_machines  = set(machines[:2])          # 2/5 may de test
train_machines = set(machines[2:])
print('test :', sorted(test_machines))
print('train:', sorted(train_machines))

REQUIRE_RUNNING = os.environ.get('PIADE_REQUIRE_RUNNING', '1') == '1'

# Chi giu cua so ma dong cuoi la `production`.
# Lay mau deu tren moi cua so thi bai toan bi mot truong duy nhat giai mat:
# 'dong cuoi la production' dat AUPRC 0.4995 tren san 0.3180 — bang dung luat
# ro ri dua tren elapsed. Loc lai thanh 'may dang chay, sap hong khong?' se
# triet tieu shortcut do va can bang lop (ti le duong tinh 0.5691).
def build_episodes(allowed, n_target, split):
    horizon_ns = np.int64(HORIZON) * 1_000_000_000
    out = []
    for machine in sorted(allowed):
        g = GROUPS[machine]
        if len(g) < WINDOW_N + 2:
            continue
        ts  = g['ts'].values.astype('datetime64[ns]').astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(g['has_alarm'].values.astype(np.int64))])
        types = g['type'].values
        for start in range(0, len(g) - WINDOW_N - 1, max(1, WINDOW_N // 2)):
            lo = start + WINDOW_N
            if REQUIRE_RUNNING and types[lo - 1] != 'production':
                continue
            hi = int(np.searchsorted(ts, ts[lo - 1] + horizon_ns, side='right'))
            if hi <= lo:
                continue
            out.append({
                'episode_id': f'piade_{split}_{machine}_{start:06d}',
                'split': split, 'machine': machine,
                'row_start': int(start), 'row_end': int(lo),
                'cutoff': pd.Timestamp(ts[lo - 1]).isoformat(),
                'horizon_seconds': HORIZON,
                'label_alarm_next': int(cum[hi] - cum[lo] > 0),
                'n_future_rows': int(hi - lo),
            })
    r = np.random.RandomState(SEED)
    if len(out) > n_target:
        out = [out[i] for i in sorted(r.choice(len(out), n_target, replace=False))]
    return out

t0 = time.perf_counter()
test_eps  = build_episodes(test_machines,  N_EPISODES, 'test')
train_eps = build_episodes(train_machines, 400,        'train')
pos = sum(e['label_alarm_next'] for e in test_eps) / max(len(test_eps), 1)
print(f'dung xong trong {time.perf_counter()-t0:.1f}s  '
      f'(chi lay cua so dang production: {REQUIRE_RUNNING})')
print(f'test={len(test_eps)}  train={len(train_eps)}  ti le duong tinh={pos:.4f}')
print('>> Ti le duong tinh nay CHINH LA san AUPRC. Ky vong ~0.57 khi loc production.')
assert all(GROUPS[e['machine']].iloc[e['row_end'] - 1].type == 'production'
           for e in test_eps[:50]) or not REQUIRE_RUNNING, 'Bo loc production khong an'
""")

code("""
episodes = {'version': 'v1', 'seed': SEED, 'horizon_seconds': HORIZON,
            'window_n': WINDOW_N, 'no_alarm_code': NO_ALARM,
            'train_machines': sorted(train_machines),
            'test_machines': sorted(test_machines),
            'source_files': source_files, 'positive_rate_test': pos,
            'test': test_eps, 'train': train_eps}
payload = json.dumps(episodes, indent=2, default=str)
(OUT_DIR / 'episodes_v1.json').write_text(payload, encoding='utf-8')
EPISODES_SHA = hashlib.sha256(payload.encode()).hexdigest()
print('episodes_v1.json sha256 =', EPISODES_SHA)
print('>> Moi ket qua ve sau phai mang dung hash nay.')
""")

# ==========================================================================
md("""
## PHASE 2 — Render 3 arm, có **che thông tin tương lai**

Dòng cuối cửa sổ bị che `elapsed` và `end`. Tại thời điểm cutoff khoảng đó
**chưa kết thúc**, nên thời lượng của nó là thông tin từ tương lai. Không che
thì một luật 2 dòng đạt AUPRC 0.4671 so với sàn 0.2962 mà không hiểu gì cả.
""")

code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATES = ['production', 'performance_loss', 'downtime', 'idle', 'scheduled_downtime']
STATE_Y = {s: i for i, s in enumerate(STATES)}

def window_rows(ep):
    return GROUPS[ep['machine']].iloc[ep['row_start']:ep['row_end']]

def render_text(ep):
    w = window_rows(ep)
    lines = ['step | state | alarm | duration_s | packages_in | packages_out | speed_pkg_h']
    n = len(w)
    for i, (_, r) in enumerate(w.iterrows()):
        last = (i == n - 1)
        # Dong cuoi: che duration vi khoang do CHUA ket thuc tai cutoff.
        dur = 'in_progress' if last else f'{r.elapsed_s:.1f}'
        lines.append(f'{i+1:4d} | {r.type} | {r.alarm} | {dur} | '
                     f'{int(r.pi_d)} | {int(r.po_d)} | {int(r.speed)}')
    return '\\n'.join(lines)

def render_plot(ep, path):
    w = window_rows(ep)
    x = np.arange(len(w))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].step(x, [STATE_Y[t] for t in w.type], where='post', marker='o', lw=1.4)
    axes[0].set_yticks(range(len(STATES))); axes[0].set_yticklabels(STATES, fontsize=7)
    axes[0].set_ylabel('state', fontsize=8)
    for i, a in enumerate(w.has_alarm.values):
        if a:
            axes[0].axvline(i, color='crimson', alpha=.35, lw=2)
    axes[1].plot(x, w.speed.values, marker='o', lw=1.4)
    axes[1].set_ylabel('speed pkg/h', fontsize=8)
    axes[2].plot(x, w.po_d.values, marker='o', lw=1.4, label='packages out')
    axes[2].plot(x, w.pi_d.values, marker='s', lw=1.0, alpha=.6, label='packages in')
    axes[2].set_ylabel('per interval', fontsize=8); axes[2].legend(fontsize=7)
    axes[2].set_xlabel('observation step (last = in progress)', fontsize=8)
    for a in axes:
        a.grid(alpha=.3); a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=80, bbox_inches='tight'); plt.close(fig)
    return path

t0 = time.perf_counter()
for ep in test_eps:
    ep['text'] = render_text(ep)
    ep['plot'] = str(render_plot(ep, ARMS / f"{ep['episode_id']}.png"))
print(f'render {len(test_eps)} episode trong {time.perf_counter()-t0:.1f}s')
print('\\n--- vi du arm text ---')
print(test_eps[0]['text'])
""")

# ==========================================================================
md("## PHASE 3 — Nạp Cosmos 3 (một GPU) + định nghĩa prompt")

code(f"""
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image

MODEL_ID      = '{MODEL_ID}'
BASE_MODEL_ID = '{BASE_MODEL_ID}'

assert torch.cuda.is_available(), 'Bat GPU T4 x2 trong Session options.'
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret('HF_TOKEN')
except Exception:
    hf_token = os.environ.get('HF_TOKEN')

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28, token=hf_token)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map={{'': 0}},
    low_cpu_mem_usage=True, attn_implementation='sdpa', token=hf_token).eval()
model.generation_config.do_sample = False       # deterministic, khac notebook MMAD

dm = getattr(model, 'hf_device_map', {{}})
assert not any(str(v) in ('cpu', 'disk') for v in dm.values()), f'Bi offload: {{dm}}'
print('VRAM GPU0:', round(torch.cuda.memory_allocated(0) / 2**30, 2), 'GiB')
""")

code("""
SYSTEM_PROMPT = (
    'You are an industrial process analyst reading operational logs from a '
    'packaging machine. Each row is one interval during which the machine held '
    'one state. Alarm code A_000 means no alarm was active.'
)
LABELS = ['NO', 'YES']
label_first_token = [processor.tokenizer(f' {l}', add_special_tokens=False).input_ids[0]
                     for l in LABELS]
assert len(set(label_first_token)) == len(LABELS), 'Token dau cua nhan bi trung'

Q_DIRECT = ('Will ANY alarm occur on this machine within the next {h} seconds '
            'after the last observation step?\\n'
            'Answer with exactly one word: YES or NO.\\nANSWER:')

# Che do reason: yeu cau CHAN DOAN tu do truoc, chua duoc tra loi.
Q_REASON = (
    'Two tasks.\\n'
    '1. DIAGNOSIS: describe what this log shows — the state sequence, whether '
    'throughput is stable or degrading, any alarms already present, and what '
    'condition the machine appears to be in right now.\\n'
    '2. RISK: state whether an alarm looks imminent and why.\\n'
    'Be concise, at most six sentences. Do NOT give a YES/NO answer yet.'
)
ANSWER_CUE = ('\\nNow answer: will ANY alarm occur within the next {h} seconds? '
              'One word, YES or NO.\\nANSWER:')

def build_inputs(ep, arm, mode):
    content, images = [], []
    if arm in ('plot', 'text+plot'):
        content.append({'type': 'image', 'image': ep['plot']})
        images.append(Image.open(ep['plot']).convert('RGB'))
    if arm in ('text', 'text+plot'):
        content.append({'type': 'text', 'text': 'Observation window:\\n' + ep['text']})
    content.append({'type': 'text',
                    'text': (Q_REASON if mode == 'reason'
                             else Q_DIRECT.format(h=ep['horizon_seconds']))})
    conv = [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': content}]
    chat = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return processor(text=[chat], images=images or None,
                     padding=True, return_tensors='pt').to(model.device)

@torch.inference_mode()
def predict(ep, arm, mode='direct', max_new_tokens=160):
    inputs = build_inputs(ep, arm, mode)
    vision_kwargs = {k: v for k, v in inputs.items()
                     if k not in ('input_ids', 'attention_mask')}
    t0 = time.perf_counter(); diagnosis = ''
    if mode == 'reason':
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        diagnosis = processor.decode(gen[0][inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()
        cue = processor.tokenizer(ANSWER_CUE.format(h=ep['horizon_seconds']),
                                  add_special_tokens=False).input_ids
        seq = torch.cat([gen, torch.tensor([cue], device=gen.device, dtype=gen.dtype)], 1)
        out = model(input_ids=seq, attention_mask=torch.ones_like(seq), **vision_kwargs)
    else:
        out = model(**inputs)
    probs = torch.softmax(out.logits[0, -1, :].float()[label_first_token], dim=-1)
    p_yes = float(probs[LABELS.index('YES')])
    return {'p_yes': p_yes, 'pred': int(p_yes >= 0.5), 'diagnosis': diagnosis[:3000],
            'latency_seconds': round(time.perf_counter() - t0, 3)}
""")

# ==========================================================================
md("## PHASE 3B — SMOKE: xem tận mắt trước khi tiêu compute")

code("""
for arm in ('text', 'plot', 'text+plot'):
    print('#' * 72); print('# ARM:', arm); print('#' * 72)
    for ep in test_eps[:3]:
        r = predict(ep, arm, 'direct')
        print(f"  {ep['episode_id']}  p_yes={r['p_yes']:.4f} truth={ep['label_alarm_next']}"
              f"  ({r['latency_seconds']}s)")

print('\\n' + '=' * 72)
print('CHE DO REASON — chan doan tu do')
print('=' * 72)
for arm in ('text', 'plot', 'text+plot'):
    ep = test_eps[0]
    r = predict(ep, arm, 'reason')
    print(f"\\n--- ARM {arm} | p_yes={r['p_yes']:.4f} truth={ep['label_alarm_next']}"
          f" ({r['latency_seconds']}s) ---")
    print(r['diagnosis'] or '(RONG — xem lai Q_REASON)')
    assert r['diagnosis'].strip(), 'Che do reason khong sinh van ban'
print('\\nOK.')
""")

# ==========================================================================
md("""
## PHASE 3C — Checkpoint dùng chung qua GitHub

`/kaggle/working` mất sạch khi hết session. Dùng lại `SharedCheckpointStore` đã
xây cho MMAD: shard JSONL bất biến, push lên repo, khởi động thì kéo về và bỏ
qua episode đã xong. Cần Kaggle Secret `GITHUB_TOKEN`; không có thì vẫn chạy,
chỉ là checkpoint nằm local.
""")

code("""
import subprocess

REPO = WORK / 'mini-world-model'
if REPO.exists():
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=False)
else:
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/anhsown/mini-world-model.git', str(REPO)], check=True)
sys.path.insert(0, str(REPO / 'research/mmad_model_benchmark'))
try:
    github_token = UserSecretsClient().get_secret('GITHUB_TOKEN')
except Exception:
    github_token = os.environ.get('GITHUB_TOKEN')
os.environ['GITHUB_TOKEN'] = github_token or ''
print('GitHub push:', 'bat' if github_token else 'TAT — checkpoint chi nam local')

from common.shared_checkpoint import SharedCheckpointStore

def make_store(arm, mode):
    return SharedCheckpointStore(
        REPO, EPISODES_SHA, f'kaggle_t4_{arm}_{mode}'.replace('+', 'plus'),
        push_every=50, token=github_token,
        relative_root='research/piade_b0_b5/checkpoints')

def load_jsonl(path):
    p = Path(path)
    return ([json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
            if p.exists() else [])

def append_jsonl(path, row):
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + '\\n')

def seed_from_kaggle_input(arm, mode):
    seeded = {}
    for p in Path('/kaggle/input').rglob(f'piade_b5_{arm}_{mode}.jsonl'):
        for row in load_jsonl(p):
            if row.get('status') == 'ok' and row.get('episodes_sha256') == EPISODES_SHA:
                seeded[row['episode_id']] = row
    return seeded

print('san sang')
""")

# ==========================================================================
md("""
## PHASE 4 — FULL run

Ba tầng resume: file local → shard GitHub → output Save Version trước.
`direct` ~1 giờ cho cả 3 arm; `reason` ~6 giờ.
""")

code("""
MODE     = os.environ.get('PIADE_MODE', 'direct')
DEADLINE = time.perf_counter() + float(os.environ.get('PIADE_HOURS', 10.5)) * 3600

for arm in ('text', 'plot', 'text+plot'):
    out_path = OUT_DIR / f'piade_b5_{arm}_{MODE}.jsonl'
    store = make_store(arm, MODE); store.sync_from_remote()
    done = {r['episode_id'] for r in load_jsonl(out_path) if r.get('status') == 'ok'}
    done |= {s.split('|')[-1] for s in store.completed_ids}
    for eid, row in seed_from_kaggle_input(arm, MODE).items():
        if eid not in done:
            append_jsonl(out_path, row); done.add(eid)
    pending = [e for e in test_eps if e['episode_id'] not in done]
    print(f'ARM {arm} [{MODE}]: xong={len(done)} con lai={len(pending)}', flush=True)

    t0 = time.perf_counter()
    for i, ep in enumerate(pending, 1):
        if time.perf_counter() >= DEADLINE:
            print('HET NGAN SACH THOI GIAN — dung an toan'); break
        try:
            r = predict(ep, arm, MODE); status, err = 'ok', None
        except Exception as exc:
            if isinstance(exc, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
            r = {'p_yes': None, 'pred': None, 'diagnosis': '', 'latency_seconds': 0.0}
            status, err = 'error', f'{type(exc).__name__}: {exc}'[:300]
        row = {'sample_id': f"{arm}|{MODE}|{ep['episode_id']}",
               'episode_id': ep['episode_id'], 'arm': arm, 'mode': MODE,
               'model': MODEL_ID, 'base_model': BASE_MODEL_ID,
               'precision': 'community BNB8 reasoner-only',
               'backend': 'Transformers/single-T4/do_sample=False',
               'manifest_sha256': EPISODES_SHA, 'episodes_sha256': EPISODES_SHA,
               'horizon_seconds': ep['horizon_seconds'], 'machine': ep['machine'],
               'cutoff': ep['cutoff'], 'truth': ep['label_alarm_next'],
               'status': status, 'error': err, **r}
        append_jsonl(out_path, row); store.record(row)
        if i % 50 == 0:
            el = time.perf_counter() - t0
            print(f'  [{i}/{len(pending)}] ETA {(len(pending)-i)*el/i/60:.1f} phut', flush=True)
    store.flush(push=True)
    print(f'ARM {arm} xong sau {(time.perf_counter()-t0)/60:.1f} phut\\n', flush=True)
""")

# ==========================================================================
md("""
## PHASE 4B — Vòng `reason` để lấy chẩn đoán cho bạn tự đánh giá

Chọn các ca **sai tự tin nhất** cộng vài ca đúng, chạy lại ở chế độ `reason`.
Xuất ra CSV để bạn chấm tay phần chẩn đoán.
""")

code("""
N_ERR, N_OK = 12, 4
by_id = {e['episode_id']: e for e in test_eps}

for arm in ('text', 'plot', 'text+plot'):
    rows = [r for r in load_jsonl(OUT_DIR / f'piade_b5_{arm}_direct.jsonl')
            if r.get('status') == 'ok' and r.get('p_yes') is not None]
    if not rows:
        print('bo qua', arm); continue
    for r in rows:
        r['_e'] = abs(r['p_yes'] - r['truth'])
    wrong = sorted([r for r in rows if r['pred'] != r['truth']], key=lambda r: -r['_e'])[:N_ERR]
    right = sorted([r for r in rows if r['pred'] == r['truth']], key=lambda r: r['_e'])[:N_OK]
    out_path = OUT_DIR / f'piade_b5_{arm}_reason_subset.jsonl'
    done = {r['episode_id'] for r in load_jsonl(out_path) if r.get('status') == 'ok'}
    print(f'ARM {arm}: {len(wrong)} sai + {len(right)} dung')
    for r0 in wrong + right:
        if r0['episode_id'] in done:
            continue
        ep = by_id[r0['episode_id']]
        try:
            r = predict(ep, arm, 'reason'); status, err = 'ok', None
        except Exception as exc:
            r = {'p_yes': None, 'pred': None, 'diagnosis': '', 'latency_seconds': 0.0}
            status, err = 'error', str(exc)[:300]
        append_jsonl(out_path, {
            'episode_id': ep['episode_id'], 'arm': arm, 'mode': 'reason',
            'selected_as': 'wrong' if r0 in wrong else 'correct',
            'p_yes_direct': r0['p_yes'], 'truth': ep['label_alarm_next'],
            'machine': ep['machine'], 'cutoff': ep['cutoff'],
            'window_text': ep['text'], 'episodes_sha256': EPISODES_SHA,
            'status': status, 'error': err, **r})
    print(f'  -> {out_path}')
""")

code("""
# Xuat CSV de danh gia tay phan chan doan.
review = []
for arm in ('text', 'plot', 'text+plot'):
    for r in load_jsonl(OUT_DIR / f'piade_b5_{arm}_reason_subset.jsonl'):
        if r.get('status') != 'ok':
            continue
        review.append({
            'episode_id': r['episode_id'], 'arm': r['arm'],
            'selected_as': r['selected_as'], 'machine': r['machine'],
            'cutoff': r['cutoff'],
            'truth': 'CO canh bao' if r['truth'] else 'KHONG co canh bao',
            'model_pred': 'CO canh bao' if r['pred'] else 'KHONG co canh bao',
            'p_yes': round(r['p_yes'], 4) if r['p_yes'] is not None else None,
            'model_diagnosis': r['diagnosis'],
            'window_text': r.get('window_text', ''),
            # cot de nguoi cham dien tay
            'human_doc_log_dung_khong': '', 'human_chan_doan_hop_ly': '',
            'human_loai_loi': '', 'human_ghi_chu': '',
        })
df_rev = pd.DataFrame(review)
path = OUT_DIR / 'piade_diagnosis_review.csv'
df_rev.to_csv(path, index=False, encoding='utf-8-sig')
print(f'da ghi {len(df_rev)} dong -> {path}')
print('Mo bang Excel, doc cot model_diagnosis va dien 4 cot human_*')
if len(df_rev):
    print('\\n--- vi du ---')
    r = df_rev.iloc[0]
    print(f"{r.episode_id} | arm={r.arm} | that={r.truth} | doan={r.model_pred}")
    print(r.model_diagnosis[:800])
""")

# ==========================================================================
md("""
## PHASE 5 — Metric + baseline

Bốn baseline trên **đúng bộ episode test**. Cosmos phải vượt **tất cả**, đặc biệt
là baseline rò rỉ — nó đạt AUPRC 0.4671 mà không hiểu gì.
""")

code("""
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

y = np.array([e['label_alarm_next'] for e in test_eps], dtype=float)
base_rate = y.mean()
wins = [window_rows(e) for e in test_eps]

def norm(v):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

s_persist = np.array([float(w.has_alarm.any()) for w in wins])
# Chung to viec che elapsed co tac dung: baseline nay phai chet (AUROC ~0.5).
s_leak    = np.array([float(w.iloc[-1].elapsed_s <= HORIZON) for w in wins])
# Tin hieu THAT ma model can doc duoc: san luong va toc do dang suy giam.
s_thru    = norm([-(w.po_d.iloc[-3:].mean() - w.po_d.iloc[:-3].mean()) for w in wins])
s_speed   = norm([-(w.speed.iloc[-3:].mean() - w.speed.iloc[:-3].mean()) for w in wins])
s_nalarm  = norm([w.has_alarm.sum() for w in wins])

print(f'{"baseline":44s} {"AUPRC":>8} {"AUROC":>8} {"Brier":>8}')
print('-' * 72)
print(f'{"san ngau nhien (= ti le duong tinh)":44s} {base_rate:8.4f} {0.5000:8.4f} '
      f'{brier_score_loss(y, np.full(len(y), base_rate)):8.4f}')
BASELINES = {}
for name, s in (('persistence (cua so da co canh bao)', s_persist),
                ('so canh bao trong cua so', s_nalarm),
                ('san luong dang giam', s_thru),
                ('toc do dang giam', s_speed),
                ('KIEM SOAT: elapsed<=H (phai ~0.50)', s_leak)):
    ap, au = average_precision_score(y, s), roc_auc_score(y, s)
    BASELINES[name] = {'AUPRC': float(ap), 'AUROC': float(au)}
    print(f'{name:44s} {ap:8.4f} {au:8.4f} {brier_score_loss(y, s):8.4f}')
BEST_BASELINE = max(BASELINES.values(), key=lambda d: d['AUROC'])['AUROC']
print(f'\\n>> Cosmos phai vuot AUROC {BEST_BASELINE:.4f} moi coi la co gia tri.')
print('>> Dong KIEM SOAT gan 0.50 xac nhan viec che elapsed da chan duoc ro ri.')
""")

code("""
results = []
for arm in ('text', 'plot', 'text+plot'):
    rows = [r for r in load_jsonl(OUT_DIR / f'piade_b5_{arm}_{MODE}.jsonl')
            if r.get('status') == 'ok' and r.get('p_yes') is not None]
    if not rows:
        continue
    yy = np.array([r['truth'] for r in rows], dtype=float)
    pp = np.array([r['p_yes'] for r in rows])
    results.append({'arm': arm, 'n': len(rows), 'positive_rate': float(yy.mean()),
                    'AUPRC': float(average_precision_score(yy, pp)),
                    'AUROC': float(roc_auc_score(yy, pp)) if 0 < yy.mean() < 1 else None,
                    'Brier': float(brier_score_loss(yy, pp)),
                    'mean_latency_s': float(np.mean([r['latency_seconds'] for r in rows]))})

if results:
    df = pd.DataFrame(results)
    df['AUROC_vs_baseline'] = df.AUROC - BEST_BASELINE
    print(df.to_string(index=False))
    best = df.loc[df.AUROC.idxmax()]
    print(f"\\n>> Arm tot nhat: {best['arm']}  AUROC {best['AUROC']:.4f}")
    verdict = ('THANG baseline' if best['AUROC'] > BEST_BASELINE
               else 'KHONG vuot duoc baseline don gian')
    print(f">> {verdict}  (baseline manh nhat AUROC {BEST_BASELINE:.4f})")
    (OUT_DIR / f'piade_b5_metrics_{MODE}.json').write_text(
        json.dumps({'mode': MODE, 'episodes_sha256': EPISODES_SHA,
                    'horizon_seconds': HORIZON,
                    'require_running': REQUIRE_RUNNING,
                    'positive_rate': float(base_rate),
                    'results': results, 'baselines': BASELINES,
                    'best_baseline_auroc': float(BEST_BASELINE),
                    'verdict': verdict}, indent=2), encoding='utf-8')
    print('\\nda ghi metrics.')
else:
    print('chua co ket qua — chay PHASE 4 truoc.')
""")


def build() -> None:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    OUT.write_text(json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Da tao: {OUT}  ({len(cells)} cell)")


if __name__ == "__main__":
    build()
