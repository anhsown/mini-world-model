from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "research" / "mmad_model_benchmark"
OLD_PATH = BENCH / "outputs" / "cosmos3_full" / "predictions.jsonl"
NEW_PATH = BENCH / "outputs" / "cosmos3_t4x2" / "predictions_from_230.jsonl"
NEW_SCORED = BENCH / "outputs" / "cosmos3_t4x2" / "predictions_scored.csv"
OUT_DIR = BENCH / "outputs" / "cosmos3_combined"
NOTEBOOK = BENCH / "analysis" / "cosmos3_mmad_combined_analysis.ipynb"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def question_number(row: dict) -> int:
    return int(row.get("question_number") or row["sample_id"].rsplit("_", 1)[1])


def is_answer(row: dict) -> bool:
    return row.get("prediction") in {"A", "B", "C", "D"}


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return value


def merge_results() -> tuple[list[dict], dict]:
    old_attempts = [r for r in read_jsonl(OLD_PATH) if question_number(r) <= 229]
    old_latest: dict[str, dict] = {}
    old_attempt_counts = Counter()
    for row in old_attempts:
        old_latest[row["sample_id"]] = row
        old_attempt_counts[row["sample_id"]] += 1

    scored = pd.read_csv(NEW_SCORED).set_index("sample_id").to_dict("index")
    new_latest: dict[str, dict] = {}
    for row in read_jsonl(NEW_PATH):
        sid = row["sample_id"]
        meta = scored.get(sid, {})
        row.setdefault("source_dataset", meta.get("source_dataset"))
        row.setdefault("category", meta.get("category"))
        row.setdefault("question_type", meta.get("question_type"))
        row.setdefault("is_normal", normalize_bool(meta.get("is_normal")))
        row["question_number"] = question_number(row)
        new_latest[sid] = row

    merged: list[dict] = []
    for row in old_latest.values():
        item = dict(row)
        item["question_number"] = question_number(item)
        item["segment"] = "NVIDIA Build UI (official endpoint)"
        item["precision"] = item.get("precision", "endpoint-managed / undisclosed")
        item["attempt_count"] = old_attempt_counts[item["sample_id"]]
        item["parse_valid"] = is_answer(item)
        item["correct"] = is_answer(item) and item["prediction"] == item.get("ground_truth")
        merged.append(item)

    for row in new_latest.values():
        item = dict(row)
        item["segment"] = "Kaggle T4x2 (community BNB8)"
        item["attempt_count"] = 1
        item["parse_valid"] = is_answer(item)
        item["correct"] = is_answer(item) and item["prediction"] == item.get("ground_truth")
        merged.append(item)

    merged.sort(key=lambda r: (r["question_number"], r["segment"]))
    hashes = sorted({r.get("manifest_sha256") for r in merged if r.get("manifest_sha256")})
    overlaps = set(old_latest).intersection(new_latest)
    assert len(old_latest) == 229, len(old_latest)
    assert not overlaps, f"Unexpected overlap: {sorted(overlaps)[:5]}"
    assert len(hashes) == 1, hashes

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_jsonl = OUT_DIR / "predictions_combined.jsonl"
    combined_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged),
        encoding="utf-8",
    )

    flat = pd.json_normalize(merged, sep=".")
    flat.to_csv(OUT_DIR / "predictions_combined.csv", index=False, encoding="utf-8-sig")

    segment_metrics = []
    for segment, group in flat.groupby("segment"):
        answered = int(group["parse_valid"].sum())
        correct = int(group["correct"].sum())
        segment_metrics.append(
            {
                "segment": segment,
                "records": len(group),
                "answered": answered,
                "coverage": answered / len(group),
                "correct": correct,
                "conditional_accuracy": correct / answered if answered else None,
                "effective_accuracy": correct / len(group),
                "mean_latency_seconds": float(group["latency_seconds"].mean()),
            }
        )

    answered = sum(is_answer(r) for r in merged)
    correct = sum(bool(r["correct"]) for r in merged)
    summary = {
        "benchmark": "MMAD",
        "evaluation_type": "mixed-backend operational merge; not a single-model leaderboard score",
        "manifest_sha256": hashes[0],
        "records": len(merged),
        "question_range": [min(r["question_number"] for r in merged), max(r["question_number"] for r in merged)],
        "unique_question_numbers": len({r["question_number"] for r in merged}),
        "answered": answered,
        "coverage": answered / len(merged),
        "correct": correct,
        "conditional_accuracy": correct / answered,
        "effective_accuracy": correct / len(merged),
        "segments": segment_metrics,
    }
    (OUT_DIR / "metrics_combined.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return merged, summary


def build_notebook() -> None:
    title = """# Cosmos 3 Nano Reasoner — MMAD Combined Analysis

This executed notebook combines:

1. **229 questions** from the official NVIDIA Build Experience UI.
2. **393 questions** from the community Reasoner-only BNB8 checkpoint on Kaggle T4×2.

Both segments use the same canonical MMAD manifest. They are kept as separate `segment` values because the serving backend and numerical precision differ. The aggregate is useful for operational inspection, but **must not be reported as a single homogeneous model leaderboard score**.
"""
    setup = r"""from pathlib import Path
import json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', 100)

ROOT = Path.cwd()
candidates = [ROOT / 'research/mmad_model_benchmark/outputs/cosmos3_combined/predictions_combined.jsonl']
candidates += list(ROOT.rglob('cosmos3_combined/predictions_combined.jsonl'))
PREDICTIONS = next(p for p in candidates if p.exists())
rows = [json.loads(x) for x in PREDICTIONS.read_text(encoding='utf-8').splitlines() if x.strip()]
df = pd.json_normalize(rows, sep='.')
df['has_answer'] = df.prediction.isin(list('ABCD'))
df['correct'] = df['correct'].fillna(False).astype(bool)
reasoning_fallback = df.reasoning.fillna('').str.len()
response_fallback = df.response.fillna('').str.len()
df['reasoning_chars'] = df.get(
    'reasoning_chars', pd.Series(index=df.index, dtype=float)
).fillna(reasoning_fallback)
df['response_chars'] = df.get(
    'response_chars', pd.Series(index=df.index, dtype=float)
).fillna(response_fallback)
df['is_normal'] = df['is_normal'].map(lambda x: {'True': True, 'False': False}.get(str(x), x))
print('Source:', PREDICTIONS)
print('Rows:', len(df), '| Manifest:', df.manifest_sha256.nunique(), 'unique hash')
display(df[['segment','question_number','sample_id','status','prediction','ground_truth','correct']].head())
"""
    helpers = r"""def wilson(k, n, z=1.96):
    if not n: return (np.nan, np.nan)
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return center-half, center+half

def summarize(frame, key=None):
    groups = [('All merged records', frame)] if key is None else frame.groupby(key, dropna=False)
    out=[]
    for name,g in groups:
        answered=int(g.has_answer.sum()); correct=int(g.correct.sum())
        lo,hi=wilson(correct, answered)
        out.append({
            key or 'scope': name, 'n':len(g), 'answered':answered,
            'coverage':answered/len(g), 'correct':correct,
            'conditional_accuracy':correct/answered if answered else np.nan,
            'accuracy_ci_low':lo, 'accuracy_ci_high':hi,
            'effective_accuracy':correct/len(g),
            'mean_latency_s':g.latency_seconds.mean(),
            'median_latency_s':g.latency_seconds.median(),
        })
    return pd.DataFrame(out)
"""
    integrity = r"""audit = pd.DataFrame({
    'item':['Rows','Unique sample IDs','Unique question numbers','Duplicate sample IDs',
            'Manifest hashes','Min question','Max question','Question-index gaps'],
    'value':[len(df),df.sample_id.nunique(),df.question_number.nunique(),
             len(df)-df.sample_id.nunique(),df.manifest_sha256.nunique(),
             int(df.question_number.min()),int(df.question_number.max()),
             int(df.question_number.max()-df.question_number.min()+1-df.question_number.nunique())]
})
display(audit)
display(df.groupby('segment').agg(
    records=('sample_id','size'), min_question=('question_number','min'),
    max_question=('question_number','max'), unique_questions=('question_number','nunique')
).reset_index())
assert df.sample_id.is_unique
assert df.manifest_sha256.nunique() == 1
"""
    summary = r"""overall = summarize(df)
by_segment = summarize(df, 'segment')
display(overall.style.format({c:'{:.1%}' for c in ['coverage','conditional_accuracy','accuracy_ci_low','accuracy_ci_high','effective_accuracy']}))
display(by_segment.style.format({c:'{:.1%}' for c in ['coverage','conditional_accuracy','accuracy_ci_low','accuracy_ci_high','effective_accuracy']}))

fig, axes = plt.subplots(1,2,figsize=(14,4.5))
plot=by_segment.melt(id_vars='segment',value_vars=['coverage','conditional_accuracy','effective_accuracy'],
                     var_name='metric',value_name='value')
sns.barplot(data=plot,x='segment',y='value',hue='metric',ax=axes[0])
axes[0].axhline(.5,color='black',ls='--',lw=1); axes[0].set_ylim(0,1)
axes[0].set_title('Coverage and accuracy by execution segment'); axes[0].tick_params(axis='x',rotation=12)
status=pd.crosstab(df.segment,df.status)
status.plot(kind='bar',stacked=True,ax=axes[1],colormap='Set2')
axes[1].set_title('Terminal status by execution segment'); axes[1].tick_params(axis='x',rotation=12)
plt.tight_layout(); plt.show()
"""
    tasks = r"""task_metrics=summarize(df,'question_type').sort_values('conditional_accuracy')
display(task_metrics.style.format({c:'{:.1%}' for c in ['coverage','conditional_accuracy','accuracy_ci_low','accuracy_ci_high','effective_accuracy']}))

fig,ax=plt.subplots(figsize=(11,5))
sns.barplot(data=task_metrics,x='conditional_accuracy',y='question_type',hue='n',palette='viridis',ax=ax)
ax.axvline(.5,color='red',ls='--',label='binary baseline'); ax.set_xlim(0,1)
ax.set_title('MMAD capability accuracy (mixed execution segments)')
plt.tight_layout(); plt.show()

source_metrics=summarize(df,'source_dataset').sort_values('conditional_accuracy')
category_metrics=summarize(df,'category').sort_values('conditional_accuracy')
display(source_metrics.style.format({c:'{:.1%}' for c in ['coverage','conditional_accuracy','effective_accuracy']}))
display(category_metrics.style.format({c:'{:.1%}' for c in ['coverage','conditional_accuracy','effective_accuracy']}))
"""
    anomaly = r"""anom=df[df.question_type.eq('Anomaly Detection')].copy()
state_metrics=summarize(anom,'is_normal')
display(state_metrics.style.format({c:'{:.1%}' for c in ['coverage','conditional_accuracy','effective_accuracy']}))

answered=anom[anom.has_answer]
cm=pd.crosstab(answered.ground_truth,answered.prediction).reindex(index=['A','B'],columns=['A','B'],fill_value=0)
display(cm)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,ax=axes[0])
axes[0].set_title('Anomaly-detection confusion matrix')
state_plot=state_metrics.copy()
state_plot['state']=state_plot.is_normal.map({True:'Normal',False:'Anomalous'})
sns.barplot(data=state_plot,x='state',y='conditional_accuracy',ax=axes[1],color='#F58518')
axes[1].axhline(.5,color='black',ls='--'); axes[1].set_ylim(0,1)
axes[1].set_title('Normal-vs-anomalous discrimination')
plt.tight_layout(); plt.show()
"""
    latency = r"""latency=df.groupby('segment').latency_seconds.agg(['count','mean','median','min','max']).reset_index()
display(latency)
fig,axes=plt.subplots(1,2,figsize=(14,4.5))
sns.histplot(data=df,x='latency_seconds',hue='segment',bins=30,element='step',ax=axes[0])
axes[0].set_title('End-to-end latency distribution')
sns.boxplot(data=df,x='segment',y='latency_seconds',showfliers=False,ax=axes[1])
axes[1].tick_params(axis='x',rotation=12); axes[1].set_title('Latency by backend (outliers hidden)')
plt.tight_layout(); plt.show()

reason=df[df.has_answer].copy()
reason['reasoning_present']=reason.reasoning_chars.gt(0)
capture=reason.groupby('segment').agg(
    answers=('sample_id','size'), reasoning_present=('reasoning_present','mean'),
    median_reasoning_chars=('reasoning_chars','median'),
    median_response_chars=('response_chars','median')
).reset_index()
display(capture)
"""
    sequence = r"""ordered=df.sort_values('question_number').copy()
ordered['rolling_accuracy_30']=ordered.correct.rolling(30,min_periods=10).mean()
ordered['rolling_coverage_30']=ordered.has_answer.rolling(30,min_periods=10).mean()
fig,ax=plt.subplots(figsize=(14,5))
for segment,g in ordered.groupby('segment'):
    ax.scatter(g.question_number,g.correct.astype(float),s=10,alpha=.22,label=f'{segment}: individual')
    ax.plot(g.question_number,g.rolling_accuracy_30,lw=2,label=f'{segment}: rolling accuracy')
ax.axhline(.5,color='black',ls='--',lw=1)
ax.set(title='Observed question coverage and rolling correctness',xlabel='Canonical MMAD question number',ylabel='Correct / rolling rate',ylim=(-.05,1.05))
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

display(ordered.groupby(['segment','question_type']).agg(
    n=('sample_id','size'), accuracy=('correct','mean'), coverage=('has_answer','mean')
).reset_index())
"""
    conclusion = r"""seg=by_segment.set_index('segment')
ui=seg.loc['NVIDIA Build UI (official endpoint)']
kg=seg.loc['Kaggle T4x2 (community BNB8)']
weak=task_metrics.iloc[0]
strong=task_metrics.iloc[-1]
anom_state=state_metrics.set_index('is_normal')

text=f'''### Executive conclusion

- The merged corpus contains **{len(df)} unique MMAD questions**: 229 from NVIDIA Build and 393 from Kaggle T4×2.
- NVIDIA Build achieved **{ui.conditional_accuracy:.1%} conditional accuracy** with **{ui.coverage:.1%} coverage**. The Kaggle BNB8 segment achieved **{kg.conditional_accuracy:.1%} conditional accuracy** with **{kg.coverage:.1%} coverage**.
- The weakest observed capability is **{weak.question_type}** at **{weak.conditional_accuracy:.1%}** (n={int(weak.n)}); the strongest is **{strong.question_type}** at **{strong.conditional_accuracy:.1%}** (n={int(strong.n)}).
- In anomaly detection, normal and anomalous samples remain highly asymmetric. This is the key failure mode, not merely answer parsing.
- The Kaggle records are non-contiguous and currently cover only selected image groups. The combined set is therefore **diagnostic, not a valid full-MMAD leaderboard result**.
- Because the two segments use different backends and precision, compare their rows separately. The aggregate is useful only as an operational evidence pool.
'''
display(Markdown(text))
"""
    export = r"""out=PREDICTIONS.parent
by_segment.to_csv(out/'analysis_by_segment.csv',index=False)
task_metrics.to_csv(out/'analysis_by_task.csv',index=False)
source_metrics.to_csv(out/'analysis_by_source.csv',index=False)
category_metrics.to_csv(out/'analysis_by_category.csv',index=False)

report={
    'records':int(len(df)),
    'manifest_sha256':str(df.manifest_sha256.iloc[0]),
    'mixed_backend_warning':True,
    'by_segment':by_segment.replace({np.nan:None}).to_dict('records'),
    'by_task':task_metrics.replace({np.nan:None}).to_dict('records'),
}
(out/'analysis_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('Saved analysis tables and summary to:',out)
"""

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    cells = [
        nbf.v4.new_markdown_cell(title),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell("## 1. Integrity and merge audit"),
        nbf.v4.new_code_cell(helpers),
        nbf.v4.new_code_cell(integrity),
        nbf.v4.new_markdown_cell("## 2. Backend-aware coverage and accuracy"),
        nbf.v4.new_code_cell(summary),
        nbf.v4.new_markdown_cell("## 3. Capability, source and category slices"),
        nbf.v4.new_code_cell(tasks),
        nbf.v4.new_markdown_cell("## 4. Anomaly-detection failure analysis"),
        nbf.v4.new_code_cell(anomaly),
        nbf.v4.new_markdown_cell("## 5. Latency and reasoning capture"),
        nbf.v4.new_code_cell(latency),
        nbf.v4.new_markdown_cell("## 6. Canonical question-index coverage"),
        nbf.v4.new_code_cell(sequence),
        nbf.v4.new_markdown_cell("## 7. Automatic conclusion"),
        nbf.v4.new_code_cell(conclusion),
        nbf.v4.new_markdown_cell("## 8. Export analysis artifacts"),
        nbf.v4.new_code_cell(export),
    ]
    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    merged, summary = merge_results()
    build_notebook()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Notebook: {NOTEBOOK}")
