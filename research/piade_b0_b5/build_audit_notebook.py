"""Build the PIADE audit notebook for Kaggle (CPU session, no GPU quota used).

This covers step 1 of the required experimental design in
CONVERSATION_CONTEXT_HANDOFF.md section 13: audit raw timestamps, schema, units,
duplicates, missingness and prevalence — BEFORE any episode is frozen.

It deliberately does not build episodes yet. Episode design depends on what the
audit finds (timestamp resolution above all), so guessing the design first would
be backwards.

Usage:
    python build_audit_notebook.py
    -> PIADE_Audit_Kaggle.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "PIADE_Audit_Kaggle.ipynb"

RECORD = "7071747"
FILES = ["raw_data.csv", "sequences_1h_data.csv"]

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


# --------------------------------------------------------------------------
md(f"""
# PIADE — Audit dữ liệu (Bước 0)

**Dataset:** Packaging Industry Anomaly DEtection (PIADE), Zenodo record `{RECORD}`
**License:** CC-BY-4.0 — được dùng và chia sẻ lại, chỉ cần ghi nguồn.
**Nhiệm vụ:** B0 (schema/tag understanding) + B5 (next-state/event prediction).

## Notebook này làm gì

Đây là **bước 1** trong quy trình bắt buộc: audit timestamp, schema, đơn vị,
trùng lặp, thiếu dữ liệu và tỉ lệ nhãn — **trước khi** đóng băng episode.

Notebook này **chưa** dựng episode. Thiết kế episode phụ thuộc vào kết quả audit
(quan trọng nhất là độ phân giải timestamp), nên đoán trước là làm ngược.

## Câu hỏi quyết định cần trả lời

> Raw interval có đủ độ phân giải để làm B5 chuẩn với horizon **30/60/120 giây** không?

- **Có** → làm B5 đúng chuẩn.
- **Không**, chỉ có aggregate 1 giờ → phải báo cáo là *next-hour forecasting*,
  **không được** gọi là B5. Đây là ranh giới trung thực bắt buộc.

## Cấu hình Kaggle

- **Accelerator: None (CPU)** — audit không cần GPU, và session CPU **không tính
  vào quota 30 giờ GPU/tuần**.
- **Internet: ON** — bắt buộc, để tải từ Zenodo.
""")

# --------------------------------------------------------------------------
code(f"""
import hashlib, json, os, sys, time
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 80)

RECORD  = '{RECORD}'
FILES   = {FILES!r}
WORK    = Path('/kaggle/working' if Path('/kaggle/working').exists() else '.')
DATA    = WORK / 'piade_raw'
OUT_DIR = WORK / 'piade_audit'
DATA.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

download_log = []
for name in FILES:
    target = DATA / name
    if not target.exists():
        url = f'https://zenodo.org/api/records/{{RECORD}}/files/{{name}}/content'
        print('downloading', name, '...', flush=True)
        started = time.perf_counter()
        urllib.request.urlretrieve(url, target)
        print(f'  done in {{time.perf_counter()-started:.1f}}s')
    entry = {{'file': name,
              'bytes': target.stat().st_size,
              'sha256': sha256(target)}}
    download_log.append(entry)
    print(json.dumps(entry, indent=1))

# Freeze the exact bytes we audited, so later runs can prove they used the same data.
(OUT_DIR / 'source_files.json').write_text(
    json.dumps({{'zenodo_record': RECORD, 'license': 'CC-BY-4.0',
                'files': download_log}}, indent=2), encoding='utf-8')
""")

# --------------------------------------------------------------------------
md("""
## 1. Schema — cột, kiểu dữ liệu, đơn vị

Không hard-code tên cột. Đọc thẳng từ file rồi mới suy ra ý nghĩa — đây cũng
chính là dữ liệu đầu vào cho **B0 (schema and tag understanding)**.
""")

code("""
frames = {}
for name in FILES:
    df = pd.read_csv(DATA / name, low_memory=False)
    frames[name] = df
    print('=' * 78)
    print(f'{name}   shape={df.shape}')
    print('=' * 78)
    info = pd.DataFrame({
        'dtype':    df.dtypes.astype(str),
        'n_unique': df.nunique(dropna=True),
        'n_null':   df.isna().sum(),
        'pct_null': (df.isna().mean() * 100).round(3),
        'example':  [df[c].dropna().iloc[0] if df[c].notna().any() else None
                     for c in df.columns],
    })
    print(info.to_string())
    print()

raw = frames['raw_data.csv']
seq = frames['sequences_1h_data.csv']
""")

# --------------------------------------------------------------------------
code("""
# Value ranges for numeric columns, top categories for object columns.
# This is the raw material for the B0 tag-understanding task.
def profile(df, name):
    print('#' * 78)
    print('#', name)
    print('#' * 78)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            print(f'{col:28s} NUM   min={d["min"]:<14.4g} max={d["max"]:<14.4g} '
                  f'mean={d["mean"]:<14.4g} n_unique={s.nunique()}')
        else:
            top = s.value_counts(dropna=True).head(6)
            joined = ', '.join(f'{k}={v}' for k, v in top.items())
            print(f'{col:28s} CAT   n_unique={s.nunique():<8} top: {joined[:110]}')
    print()

profile(raw, 'raw_data.csv')
profile(seq, 'sequences_1h_data.csv')
""")

# --------------------------------------------------------------------------
md("""
## 2. Timestamp — câu hỏi quyết định B5 chuẩn hay không

Cần biết: cột nào là thời gian, độ phân giải thực tế, có timezone không, có sắp
xếp tăng dần theo từng máy không, khoảng trống ra sao, và **phân bố độ dài
interval** — vì horizon 30/60/120 giây chỉ có nghĩa nếu interval đủ ngắn.
""")

code("""
def find_time_columns(df):
    hits = []
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ('time', 'date', 'start', 'end', 'stamp', 'ts')):
            hits.append(col)
    return hits

time_cols = find_time_columns(raw)
print('cot nghi la thoi gian:', time_cols)

parsed = {}
for col in time_cols:
    try:
        s = pd.to_datetime(raw[col], errors='coerce', format='mixed')
    except Exception:
        s = pd.to_datetime(raw[col], errors='coerce')
    ok = s.notna().mean()
    print(f'  {col:24s} parse_ok={ok:.4f}', end='')
    if ok > 0.5:
        parsed[col] = s
        print(f'  min={s.min()}  max={s.max()}  tz={s.dt.tz}')
    else:
        print('  -> khong phai thoi gian')
""")

# --------------------------------------------------------------------------
code("""
# Timestamp resolution: how many distinct sub-second / sub-minute values exist?
for col, s in parsed.items():
    valid = s.dropna()
    if valid.empty:
        continue
    print(f'--- {col} ---')
    print(f'  n={len(valid)}  n_unique={valid.nunique()}')
    print(f'  co giay khac 0 : {(valid.dt.second != 0).mean():.4f}')
    print(f'  co micro khac 0: {(valid.dt.microsecond != 0).mean():.4f}')
    deltas = valid.sort_values().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if not deltas.empty:
        q = deltas.dt.total_seconds().quantile([.01, .25, .5, .75, .99])
        print('  khoang cach giua 2 ban ghi lien tiep (giay):')
        print('   ', {f'p{int(k*100)}': round(v, 3) for k, v in q.items()})
    print()
""")

# --------------------------------------------------------------------------
code("""
# Interval duration distribution — decides whether 30/60/120s horizons are meaningful.
dur_cols = [c for c in raw.columns
            if any(k in c.lower() for k in ('elapsed', 'duration', 'length'))]
print('cot thoi luong:', dur_cols)

for col in dur_cols:
    s = pd.to_numeric(raw[col], errors='coerce').dropna()
    if s.empty:
        continue
    q = s.quantile([.01, .05, .25, .5, .75, .95, .99])
    print(f'\\n--- {col} ---')
    print('  quantile:', {f'p{int(k*100)}': round(v, 3) for k, v in q.items()})
    for horizon in (30, 60, 120):
        # Interpreted as seconds; if the unit turns out to be minutes the audit
        # summary will say so and this line must be re-read accordingly.
        share = (s <= horizon).mean()
        print(f'  ti le interval <= {horizon:4d} (don vi goc): {share:.4f}')
""")

# --------------------------------------------------------------------------
md("""
## 3. Trùng lặp và thiếu dữ liệu
""")

code("""
for name, df in frames.items():
    print('=' * 78)
    print(name)
    exact = df.duplicated().sum()
    print(f'  dong trung hoan toan : {exact}  ({exact/len(df)*100:.4f}%)')
    if time_cols and name == 'raw_data.csv':
        id_cols = [c for c in df.columns
                   if any(k in c.lower() for k in ('equip', 'machine', 'id', 'line'))]
        key = [c for c in (id_cols[:1] + time_cols[:1]) if c in df.columns]
        if len(key) == 2:
            dup_key = df.duplicated(subset=key).sum()
            print(f'  trung theo {key}: {dup_key}')
    nulls = df.isna().mean().sort_values(ascending=False)
    nonzero = nulls[nulls > 0]
    if nonzero.empty:
        print('  khong co gia tri thieu')
    else:
        print('  cot thieu nhieu nhat:')
        print(nonzero.head(10).round(5).to_string())
    print()
""")

# --------------------------------------------------------------------------
md("""
## 4. Máy, trạng thái, và **tỉ lệ cảnh báo** — quyết định metric

Tỉ lệ dương tính (prevalence) là con số phải có trước khi chạy model. Không có
nó thì không diễn giải nổi AUPRC: AUPRC của một bộ phân loại ngẫu nhiên **bằng
đúng tỉ lệ dương tính**, nên đây chính là mốc sàn để so.
""")

code("""
id_cols = [c for c in raw.columns
           if any(k in c.lower() for k in ('equip', 'machine', 'line', 'asset'))]
state_cols = [c for c in raw.columns
              if any(k in c.lower() for k in ('state', 'status', 'mode', 'type'))]
alarm_cols = [c for c in raw.columns
              if any(k in c.lower() for k in ('alarm', 'alert', 'fault', 'error', 'code'))]
print('may     :', id_cols)
print('trang thai:', state_cols)
print('canh bao :', alarm_cols)
print()

for col in id_cols + state_cols:
    vc = raw[col].value_counts(dropna=False)
    print(f'--- {col} ({raw[col].nunique()} gia tri) ---')
    print((vc.head(12) / len(raw)).round(5).to_string())
    print()
""")

# --------------------------------------------------------------------------
code("""
# Alarm prevalence — the floor for AUPRC.
prevalence = {}
for col in alarm_cols:
    s = raw[col]
    n_unique = s.nunique(dropna=True)
    non_null = s.notna().mean()
    print(f'--- {col} ---')
    print(f'  n_unique={n_unique}   ti le co gia tri={non_null:.4f}')
    vc = s.value_counts(dropna=True)
    print('  top 10:')
    print((vc.head(10) / len(raw)).round(6).to_string())
    prevalence[col] = {
        'n_unique': int(n_unique),
        'non_null_rate': float(non_null),
        'top10_share': {str(k): float(v / len(raw)) for k, v in vc.head(10).items()},
    }
    # Tail: how many alarm types are rare enough that AUPRC will be unstable?
    share = vc / len(raw)
    for thr in (0.01, 0.001, 0.0001):
        print(f'  so loai canh bao co ti le < {thr}: {(share < thr).sum()} / {n_unique}')
    print()
""")

# --------------------------------------------------------------------------
md("""
## 5. Kết luận audit

Ghi ra file JSON để bước dựng episode dùng lại, và in ra phán quyết về việc
B5 chuẩn có làm được không.
""")

code("""
summary = {
    'zenodo_record': RECORD,
    'license': 'CC-BY-4.0',
    'source_files': download_log,
    'shapes': {name: list(df.shape) for name, df in frames.items()},
    'raw_columns': list(raw.columns),
    'seq_columns': list(seq.columns),
    'time_columns_parsed': list(parsed.keys()),
    'id_columns': id_cols,
    'state_columns': state_cols,
    'alarm_columns': alarm_cols,
    'alarm_prevalence': prevalence,
    'exact_duplicate_rows': {name: int(df.duplicated().sum())
                             for name, df in frames.items()},
    'null_rate_max': {name: float(df.isna().mean().max())
                      for name, df in frames.items()},
}

for col, s in parsed.items():
    valid = s.dropna()
    summary.setdefault('timestamp_detail', {})[col] = {
        'min': str(valid.min()), 'max': str(valid.max()),
        'n_unique': int(valid.nunique()),
        'nonzero_second_rate': float((valid.dt.second != 0).mean()),
        'nonzero_microsecond_rate': float((valid.dt.microsecond != 0).mean()),
    }

path = OUT_DIR / 'piade_audit_v1.json'
path.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
print('da ghi:', path)
print()
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ('alarm_prevalence', 'raw_columns', 'seq_columns')},
                 indent=2, default=str)[:2500])
""")

# --------------------------------------------------------------------------
code("""
# PHAN QUYET: B5 chuan hay next-hour forecasting?
sub_minute = any(v['nonzero_second_rate'] > 0.01
                 for v in summary.get('timestamp_detail', {}).values())

print('=' * 70)
if sub_minute:
    print('KET LUAN: timestamp CO do phan giai duoi phut.')
    print('  -> Co the lam B5 CHUAN voi horizon 30/60/120 giay,')
    print('     mien la phan bo do dai interval o muc 2 cung ung ho.')
else:
    print('KET LUAN: timestamp KHONG co do phan giai duoi phut.')
    print('  -> KHONG duoc goi la B5 chuan.')
    print('     Phai bao cao la "next-hour forecasting" theo dung handoff muc 13.4.')
print('=' * 70)
print()
print('Buoc tiep theo: dua ket qua audit nay ve de thiet ke episode,')
print('roi dong bang episodes_v1.json dung chung cho ca 3 arm text/plot/text+plot.')
""")


def build() -> None:
    cells = []
    for kind, source in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"Da tao: {OUT}  ({len(cells)} cell)")


if __name__ == "__main__":
    build()
