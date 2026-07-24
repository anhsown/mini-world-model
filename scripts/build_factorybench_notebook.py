"""Build the Colab-ready FactoryBench audit and baseline notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "factorybench" / "factorybench_baseline_colab.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """# FactoryBench → Industrial JWM: Audit and Baseline

This notebook verifies FactoryBench before any training and evaluates a
context-blind prior as a leakage-control floor.

The target contract is:

```text
sensor history + control signals + machine context + question
                              ↓
                    grounded text response
```

**Important:** FactoryBench is CC BY-NC 4.0. This notebook supports research
evaluation, not automatic commercial admission.
"""
    ),
    code(
        """from pathlib import Path
import os, subprocess, sys

REPO_URL = 'https://github.com/anhsown/mini-world-model'
cwd = Path.cwd().resolve()
repo = next((path for path in (cwd, *cwd.parents)
             if (path / 'jwm').exists() and (path / 'scripts').exists()), None)
if repo is None:
    repo = Path('/content/mini-world-model')
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'huggingface-hub>=0.20', 'pandas>=2.0',
                'matplotlib>=3.7', 'seaborn>=0.12'], check=True)
print('workspace:', Path.cwd())
"""
    ),
    md("## 1. Full schema, count, license and split-leakage audit"),
    code(
        """subprocess.run([sys.executable, 'scripts/audit_factorybench.py',
                '--output', 'research/factorybench', '--max-samples', '64'],
               check=True)
"""
    ),
    code(
        """import json
from IPython.display import display, Markdown

audit = json.load(open('research/factorybench/factorybench_audit.json',
                       encoding='utf-8'))
print(json.dumps(audit['admission'], indent=2))
print(json.dumps(audit['split_leakage'], indent=2))
display(Markdown(Path('research/factorybench/FACTORYBENCH_AUDIT.md')
                 .read_text(encoding='utf-8')))
"""
    ),
    md("## 2. Visualize causal levels, context structures and answer families"),
    code(
        """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
counts = audit['records']['counts']
rows = []
for level in range(1, 5):
    for split in ('train', 'validation', 'test'):
        rows.append({'level': f'L{level}', 'split': split,
                     'records': counts[f'L{level}_{split}']})
count_df = pd.DataFrame(rows)
display(count_df)
plt.figure(figsize=(8, 4))
sns.barplot(data=count_df, x='level', y='records', hue='split')
plt.title('FactoryBench records by Pearl causal level')
plt.tight_layout()
plt.show()

structures = pd.Series(audit['distribution']['context_structures'])
display(structures.to_frame('records'))
structures.plot(kind='bar', figsize=(8, 4), color='#4c78a8')
plt.title('Evidence packaging in FactoryBench')
plt.ylabel('records')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

reuse = pd.DataFrame(
    audit['distribution']['target_reuse']['test_answer_seen_exactly_in_train']
).T
display(reuse)
reuse['fraction'].sort_values().plot(kind='barh', figsize=(8, 5),
                                    color='#e45756')
plt.xlim(0, 1)
plt.xlabel('Fraction of test targets seen verbatim in train')
plt.title('Template/target reuse — not episode leakage')
plt.tight_layout()
plt.show()
"""
    ),
    md("## 3. Inspect representative records and the canonical JWM contract"),
    code(
        """samples = json.load(open(
    'research/factorybench/representative_samples_l1_l4.json',
    encoding='utf-8'))
print('representative strata:', len(samples))
for level in range(1, 5):
    item = next(sample for sample in samples if sample['level'] == level)
    print('\\n' + '=' * 90)
    print('LEVEL', level, '|', item['template_type'], '|',
          item['canonical']['target']['answer_format'])
    print('QUESTION:', item['question'])
    print('ANSWER:', item['answer'])
    print('STREAMS:', list(item['canonical']['inputs']['streams']))
    for name, stream in item['canonical']['inputs']['streams'].items():
        print(name, {
            role: len(stream[role]['channels'])
            for role in ('sensor_history', 'control_signals', 'machine_context')
        })
"""
    ),
    md("## 4. Run the context-blind template-prior floor"),
    code(
        """subprocess.run([sys.executable, 'scripts/run_factorybench_baseline.py',
                '--output',
                'research/factorybench/context_blind_baseline_results.json'],
               check=True)
baseline = json.load(open(
    'research/factorybench/context_blind_baseline_results.json',
    encoding='utf-8'))
display(pd.DataFrame(baseline['by_level']).T)
"""
    ),
    code(
        """level_scores = pd.DataFrame(baseline['by_level']).T
plt.figure(figsize=(7, 4))
sns.barplot(x=level_scores.index,
            y=level_scores.micro_primary_score,
            color='#f58518')
plt.ylim(0, 1)
plt.ylabel('Task-aware primary score')
plt.xlabel('Causal level')
plt.title('Context-blind floor — a sensor-aware model must beat this')
plt.tight_layout()
plt.show()

family_df = pd.DataFrame(baseline['by_level_and_answer_family']).T
display(family_df.sort_index())
"""
    ),
    md(
        """## 5. Interpretation and go/no-go gates

- Passing the dataset audit means FactoryBench is structurally usable for
  **research evaluation**.
- It does not clear commercial use because the license is non-commercial.
- The current JWM byte-text path cannot ingest the telemetry: nearly every
  record exceeds its context budget.
- The next architecture experiment needs separate numerical sensor tokens,
  control/action tokens and static machine-context tokens.
- Report L1–L4 separately. A high aggregate score can hide failure on
  counterfactual or troubleshooting tasks.
- Require correct-context vs shuffled-context and zero-context controls before
  claiming physical understanding.
"""
    ),
    code(
        """summary = {
    'dataset_research_admitted': audit['admission']['research'],
    'commercial_training_admitted': audit['admission']['direct_commercial_training'],
    'current_jwm_ready': audit['admission']['current_jwm_without_new_encoder'],
    'records_verified': audit['records']['actual_total'],
    'split_leakage_valid': audit['split_leakage']['valid'],
    'context_blind_macro_floor':
        baseline['macro_primary_score_over_levels'],
}
print(json.dumps(summary, indent=2))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for index, cell in enumerate(cells):
    cell["id"] = f"factorybench-{index:02d}"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUT)
