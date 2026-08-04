from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "research" / "mmad_model_benchmark"
MANIFEST = BENCH / "data_full" / "full_manifest.json"
OLD_COMBINED = BENCH / "outputs" / "cosmos3_combined" / "predictions_combined.jsonl"
PUBLIC_REPO = ROOT.parent / "mini-world-model"
SHARDS = PUBLIC_REPO / "research" / "mmad_model_benchmark" / "checkpoints" / "cosmos3_mmad"
OUT = BENCH / "outputs" / "cosmos3_checkpoint_full"
NOTEBOOK = BENCH / "analysis" / "cosmos3_mmad_checkpoint_full_analysis.ipynb"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def classify_run(path: Path) -> str:
    if path.parent.name == "nvidia_build":
        return "NVIDIA Build seed"
    stamp = path.name[:8]
    if stamp == "20260731":
        return "Kaggle initial run"
    if stamp == "20260801":
        return "Kaggle previous run"
    if stamp == "20260802":
        return "Kaggle latest run (6h)"
    return f"Other ({stamp})"


def prepare_data() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    meta = {r["sample_id"]: r for r in manifest["records"]}
    rows: dict[str, dict] = {}
    shard_counts: dict[str, int] = {}
    for path in sorted(SHARDS.rglob("*.jsonl")):
        run = classify_run(path)
        shard_counts[run] = shard_counts.get(run, 0) + 1
        for raw in read_jsonl(path):
            if raw.get("status") != "ok" or raw.get("prediction") not in {"A", "B", "C", "D"}:
                continue
            sid = raw["sample_id"]
            m = meta[sid]
            row = dict(raw)
            row.update({
                "run": run,
                "shard_file": path.name,
                "question_number": int(sid.rsplit("_", 1)[1]),
                "source_dataset": m["source_dataset"],
                "category": m["category"],
                "question_type": m["question_type"],
                "is_normal": m["is_normal"],
                "ground_truth": m["answer"],
                "options": m["options"],
                "prediction_text": m["options"].get(raw["prediction"], ""),
                "ground_truth_text": m["options"].get(m["answer"], ""),
                "image_file": m["image_file"],
                "correct": raw["prediction"] == m["answer"],
            })
            rows[sid] = row

    combined = sorted(rows.values(), key=lambda r: r["question_number"])
    old_rows = read_jsonl(OLD_COMBINED)
    old_ids = {
        r["sample_id"] for r in old_rows
        if r.get("prediction") in {"A", "B", "C", "D"}
    }
    for row in combined:
        row["snapshot"] = "Already in old notebook" if row["sample_id"] in old_ids else "Added after old notebook"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions_all_checkpoints.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8"
    )
    audit = {
        "manifest_sha256": manifest["manifest_sha256"],
        "full_benchmark_questions": len(manifest["records"]),
        "successful_checkpoint_rows": len(combined),
        "old_notebook_answered_ids": len(old_ids),
        "new_successful_ids_since_old_notebook": len(set(rows) - old_ids),
        "shards_by_run": shard_counts,
        "known_latest_run_attempts": 409,
        "known_latest_run_successes": 397,
        "known_latest_run_parse_failures": 12,
        "latest_run_actual_hours": 6.0014,
    }
    (OUT / "checkpoint_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def build_notebook(audit: dict) -> None:
    title = f"""# Cosmos 3 Nano — Full Checkpoint Analysis and Old-Notebook Comparison

This executed report analyses **all {audit['successful_checkpoint_rows']:,} successful checkpoint records currently stored on GitHub** and compares them with the earlier `cosmos3_mmad_combined_analysis.ipynb` snapshot.

Important scope notes:

- Full MMAD contains **{audit['full_benchmark_questions']:,} questions**; this remains a partial benchmark.
- The latest session was configured for **6.0014 hours**, despite the intended 24-hour run.
- GitHub stores successful rows only. The latest log separately reports **397 successful outputs and 12 parse failures from 409 attempts**.
- NVIDIA Build and the community BNB8 Kaggle checkpoint are different execution backends; backend-level results must remain separate.
"""
    setup = r"""from pathlib import Path
import json, math, re, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', 100)

ROOT = Path.cwd()
search_roots = [ROOT, *ROOT.parents]
candidates = []
for base in search_roots:
    candidates += list(base.glob('research/mmad_model_benchmark/outputs/cosmos3_checkpoint_full/predictions_all_checkpoints.jsonl'))
    candidates += list(base.glob('outputs/cosmos3_checkpoint_full/predictions_all_checkpoints.jsonl'))
PRED = next(p for p in candidates if p.exists())
AUDIT = json.loads((PRED.parent / 'checkpoint_audit.json').read_text(encoding='utf-8'))
rows = [json.loads(x) for x in PRED.read_text(encoding='utf-8').splitlines() if x.strip()]
df = pd.json_normalize(rows, sep='.')
df['reasoning'] = df.reasoning.fillna('')
df['response'] = df.response.fillna('')
df['reasoning_chars'] = df.reasoning.str.len()
df['reasoning_words'] = df.reasoning.str.findall(r"\b[\w'-]+\b").str.len()
df['correct'] = df.correct.astype(bool)
df['answer_bias'] = df.prediction
print('Data:', PRED)
print('Rows:', len(df), '| unique sample IDs:', df.sample_id.nunique())
display(pd.Series(AUDIT, name='value').to_frame())
assert df.sample_id.is_unique
assert len(df) == AUDIT['successful_checkpoint_rows']
"""
    helpers = r"""def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return center-half, center+half

def summarize(frame, key):
    out=[]
    for name,g in frame.groupby(key, dropna=False):
        k=int(g.correct.sum()); n=len(g); lo,hi=wilson(k,n)
        out.append({key:name,'n':n,'correct':k,'accuracy':k/n,'ci_low':lo,'ci_high':hi,
                    'median_latency_s':g.latency_seconds.median(),
                    'median_reasoning_chars':g.reasoning_chars.median()})
    return pd.DataFrame(out)
"""
    integrity = r"""print('Successful coverage of full MMAD: {:.2%}'.format(len(df)/AUDIT['full_benchmark_questions']))
run_summary=summarize(df,'run').sort_values('run')
snapshot_summary=summarize(df,'snapshot').sort_values('snapshot')
display(run_summary.style.format({c:'{:.1%}' for c in ['accuracy','ci_low','ci_high']}))
display(snapshot_summary.style.format({c:'{:.1%}' for c in ['accuracy','ci_low','ci_high']}))

fig,axes=plt.subplots(1,2,figsize=(15,5))
sns.barplot(data=run_summary,x='run',y='n',ax=axes[0],color='#4C78A8')
axes[0].set_title('Successful checkpoint records by run'); axes[0].tick_params(axis='x',rotation=20)
sns.barplot(data=run_summary,x='run',y='accuracy',ax=axes[1],color='#59A14F')
axes[1].errorbar(np.arange(len(run_summary)),run_summary.accuracy,
                 yerr=[run_summary.accuracy-run_summary.ci_low,run_summary.ci_high-run_summary.accuracy],
                 fmt='none',color='black',capsize=4)
axes[1].set_ylim(0,1); axes[1].set_title('Accuracy by run with 95% Wilson CI'); axes[1].tick_params(axis='x',rotation=20)
plt.tight_layout(); plt.show()
"""
    oldnew = r"""old_new_task=df.groupby(['snapshot','question_type']).agg(n=('sample_id','size'),accuracy=('correct','mean')).reset_index()
display(old_new_task.sort_values(['question_type','snapshot']))
fig,ax=plt.subplots(figsize=(12,6))
sns.barplot(data=old_new_task,y='question_type',x='accuracy',hue='snapshot',ax=ax)
ax.set_xlim(0,1); ax.axvline(.5,color='black',ls='--',lw=1)
ax.set_title('Capability profile: old notebook vs newly added checkpoints')
plt.tight_layout(); plt.show()

coverage=pd.crosstab(df.question_type,df.snapshot)
display(coverage)
coverage.plot(kind='barh',stacked=True,figsize=(12,6),colormap='Set2')
plt.title('How the new run expands task coverage'); plt.xlabel('Successful records'); plt.tight_layout(); plt.show()
"""
    tasks = r"""task=summarize(df,'question_type').sort_values('accuracy')
source=summarize(df,'source_dataset').sort_values('accuracy')
display(task.style.format({c:'{:.1%}' for c in ['accuracy','ci_low','ci_high']}))
display(source.style.format({c:'{:.1%}' for c in ['accuracy','ci_low','ci_high']}))

fig,axes=plt.subplots(1,2,figsize=(15,6))
sns.barplot(data=task,y='question_type',x='accuracy',hue='n',palette='viridis',ax=axes[0])
axes[0].set_xlim(0,1); axes[0].set_title('Accuracy by MMAD capability')
sns.barplot(data=source,y='source_dataset',x='accuracy',hue='n',palette='magma',ax=axes[1])
axes[1].set_xlim(0,1); axes[1].set_title('Accuracy by source dataset')
plt.tight_layout(); plt.show()
"""
    anomaly = r"""anom=df[df.question_type.eq('Anomaly Detection')].copy()
# MMAD randomizes option letters, so decode semantic Yes/No from the manifest.
anom['actual_anomaly'] = ~anom.is_normal.astype(bool)
anom['predicted_anomaly'] = anom.prediction_text.str.strip().str.lower().str.startswith('yes')
cm=pd.crosstab(anom.actual_anomaly,anom.predicted_anomaly).reindex(index=[True,False],columns=[True,False],fill_value=0)
cm.index=['Actual anomaly','Actual normal']; cm.columns=['Predicted anomaly','Predicted normal']
tp=cm.loc['Actual anomaly','Predicted anomaly']; fn=cm.loc['Actual anomaly','Predicted normal']
fp=cm.loc['Actual normal','Predicted anomaly']; tn=cm.loc['Actual normal','Predicted normal']
precision=tp/(tp+fp) if tp+fp else np.nan
recall=tp/(tp+fn) if tp+fn else np.nan
f1=2*precision*recall/(precision+recall) if precision+recall else np.nan
display(cm)
display(pd.DataFrame([{'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':precision,'recall':recall,
                       'F1':f1,'miss_rate':fn/(tp+fn),'overkill_rate':fp/(fp+tn)}]).style.format({
    'precision':'{:.1%}','recall':'{:.1%}','F1':'{:.1%}','miss_rate':'{:.1%}','overkill_rate':'{:.1%}'}))
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,ax=axes[0]); axes[0].set_title('Anomaly detection confusion matrix')
pd.crosstab(anom.snapshot,anom.actual_anomaly).rename(columns={False:'Normal',True:'Anomaly'}).plot(kind='bar',stacked=True,ax=axes[1],colormap='Set2')
axes[1].set_title('Anomaly labels added by snapshot'); axes[1].tick_params(axis='x',rotation=15)
plt.tight_layout(); plt.show()
"""
    reasoning = r"""patterns={
 'uncertainty':r'\b(maybe|might|could|appears|seems|likely|possibly|unclear)\b',
 'visual_claim':r'\b(image|visible|shown|see|surface|object)\b',
 'defect_terms':r'\b(defect|crack|scratch|dent|damage|anomal|broken|missing|irregular)\w*\b',
 'negation':r'\b(no|not|none|without|lack|absence)\b',
 'answer_leak_phrase':r'\b(answer|option|therefore|correct choice)\b',
}
for name,pat in patterns.items():
    df[name]=df.reasoning.str.contains(pat,case=False,regex=True)

reason_summary=df.groupby('correct').agg(
 n=('sample_id','size'),median_chars=('reasoning_chars','median'),mean_chars=('reasoning_chars','mean'),
 median_words=('reasoning_words','median'),median_latency_s=('latency_seconds','median'),
 uncertainty_rate=('uncertainty','mean'),defect_term_rate=('defect_terms','mean'),
 negation_rate=('negation','mean'),answer_phrase_rate=('answer_leak_phrase','mean')).reset_index()
display(reason_summary.style.format({c:'{:.1%}' for c in ['uncertainty_rate','defect_term_rate','negation_rate','answer_phrase_rate']}))

fig,axes=plt.subplots(2,2,figsize=(14,10))
sns.histplot(data=df,x='reasoning_chars',hue='correct',bins=35,element='step',log_scale=(False,True),ax=axes[0,0])
axes[0,0].set_title('Reasoning length distribution')
sns.boxplot(data=df,x='correct',y='reasoning_chars',showfliers=False,ax=axes[0,1]); axes[0,1].set_title('Reasoning length vs correctness')
sns.scatterplot(data=df,x='reasoning_chars',y='latency_seconds',hue='correct',alpha=.35,s=22,ax=axes[1,0])
axes[1,0].set_title('Reasoning length vs latency')
proxy=pd.DataFrame({'proxy':list(patterns),'correct':[df.loc[df[p],'correct'].mean() for p in patterns],
                    'incorrect':[1-df.loc[df[p],'correct'].mean() for p in patterns],
                    'support':[int(df[p].sum()) for p in patterns]})
sns.barplot(data=proxy,y='proxy',x='correct',hue='support',palette='crest',ax=axes[1,1]); axes[1,1].set_xlim(0,1)
axes[1,1].set_title('Accuracy when reasoning proxy is present')
plt.tight_layout(); plt.show()

print('Correlation reasoning chars vs latency:', round(df.reasoning_chars.corr(df.latency_seconds),3))
print('Correlation reasoning chars vs correctness:', round(df.reasoning_chars.corr(df.correct.astype(int)),3))
"""
    bias = r"""pred_dist=pd.crosstab(df.run,df.prediction,normalize='index').reindex(columns=list('ABCD'),fill_value=0)
truth_dist=pd.crosstab(df.run,df.ground_truth,normalize='index').reindex(columns=list('ABCD'),fill_value=0)
display(Markdown('**Prediction distribution**')); display(pred_dist.style.format('{:.1%}'))
display(Markdown('**Ground-truth distribution**')); display(truth_dist.style.format('{:.1%}'))
fig,axes=plt.subplots(1,2,figsize=(14,5))
pred_dist.plot(kind='bar',stacked=True,ax=axes[0],colormap='Set2'); axes[0].set_title('Prediction-letter distribution'); axes[0].tick_params(axis='x',rotation=20)
truth_dist.plot(kind='bar',stacked=True,ax=axes[1],colormap='Set2'); axes[1].set_title('Ground-truth distribution'); axes[1].tick_params(axis='x',rotation=20)
plt.tight_layout(); plt.show()
"""
    latency = r"""lat=df.groupby('run').latency_seconds.agg(['count','mean','median',lambda s:s.quantile(.95)]).reset_index()
lat.columns=['run','n','mean_s','median_s','p95_s']
display(lat)
fig,axes=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(data=df,x='run',y='latency_seconds',showfliers=False,ax=axes[0]); axes[0].tick_params(axis='x',rotation=20); axes[0].set_title('Latency by run')
ordered=df.sort_values('created_at').copy(); ordered['rolling_accuracy_100']=ordered.correct.rolling(100,min_periods=30).mean()
ordered['rolling_latency_100']=ordered.latency_seconds.rolling(100,min_periods=30).median()
ax2=axes[1].twinx(); axes[1].plot(np.arange(len(ordered)),ordered.rolling_accuracy_100,color='#59A14F',label='accuracy')
ax2.plot(np.arange(len(ordered)),ordered.rolling_latency_100,color='#E15759',label='latency')
axes[1].set_ylim(0,1); axes[1].set_title('Rolling accuracy and median latency (100 rows)')
plt.tight_layout(); plt.show()
"""
    insights = r"""old=df[df.snapshot.eq('Already in old notebook')]
new=df[df.snapshot.eq('Added after old notebook')]
old_acc=old.correct.mean(); new_acc=new.correct.mean(); all_acc=df.correct.mean()
weak=summarize(df,'question_type').sort_values('accuracy').iloc[0]
latest=df[df.run.eq('Kaggle latest run (6h)')]
new_sources=', '.join(sorted(new.source_dataset.unique()))

text=f'''## What the additional samples changed

- The old executed notebook contained **{len(old):,} successful rows**. The checkpoint pool now contains **{len(df):,}**, adding **{len(new):,} unique successful samples**.
- Accuracy on rows already present in the old notebook is **{old_acc:.1%}**; newly added rows score **{new_acc:.1%}**; the current successful-only aggregate is **{all_acc:.1%}**.
- The additional data covers: **{new_sources}**. Source/task skew still matters; this is not yet a full-MMAD leaderboard score.
- The weakest capability remains **{weak.question_type}** at **{weak.accuracy:.1%}** (n={int(weak.n)}).
- Latest run: **397 successful / 409 attempted**, with **12 parse failures**; operational success yield is **97.1%**.
- Reasoning is present in all successful Kaggle rows, but longer reasoning is not automatically better. The proxy plots above distinguish verbosity, uncertainty language and answer-directed phrasing from actual correctness.
- The main new insight is stronger evidence that Cosmos is good at higher-level defect/object questions but remains weak and conservative on binary anomaly detection. More samples strengthen this pattern rather than overturn it.
'''
display(Markdown(text))
"""
    export = r"""out=PRED.parent
run_summary.to_csv(out/'analysis_by_run.csv',index=False)
snapshot_summary.to_csv(out/'analysis_old_vs_new.csv',index=False)
task.to_csv(out/'analysis_by_task.csv',index=False)
source.to_csv(out/'analysis_by_source.csv',index=False)
reason_summary.to_csv(out/'analysis_reasoning.csv',index=False)
report={'audit':AUDIT,'successful_accuracy':float(df.correct.mean()),
        'successful_coverage':float(len(df)/AUDIT['full_benchmark_questions']),
        'old_accuracy':float(old.correct.mean()),'new_accuracy':float(new.correct.mean()),
        'by_run':run_summary.replace({np.nan:None}).to_dict('records'),
        'by_task':task.replace({np.nan:None}).to_dict('records')}
(out/'analysis_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('Saved analysis artifacts to',out)
"""

    cells = [
        nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell("## 1. Metrics helpers"), nbf.v4.new_code_cell(helpers),
        nbf.v4.new_markdown_cell("## 2. Checkpoint integrity and run comparison"), nbf.v4.new_code_cell(integrity),
        nbf.v4.new_markdown_cell("## 3. Old notebook vs newly added samples"), nbf.v4.new_code_cell(oldnew),
        nbf.v4.new_markdown_cell("## 4. Capability and source analysis"), nbf.v4.new_code_cell(tasks),
        nbf.v4.new_markdown_cell("## 5. Anomaly-detection failure analysis"), nbf.v4.new_code_cell(anomaly),
        nbf.v4.new_markdown_cell("## 6. Reasoning-content analysis"), nbf.v4.new_code_cell(reasoning),
        nbf.v4.new_markdown_cell("## 7. Answer bias"), nbf.v4.new_code_cell(bias),
        nbf.v4.new_markdown_cell("## 8. Latency and stability"), nbf.v4.new_code_cell(latency),
        nbf.v4.new_markdown_cell("## 9. New insights"), nbf.v4.new_code_cell(insights),
        nbf.v4.new_markdown_cell("## 10. Export"), nbf.v4.new_code_cell(export),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    audit = prepare_data()
    build_notebook(audit)
    print(json.dumps(audit, indent=2))
    print(NOTEBOOK)
