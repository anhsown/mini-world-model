"""Build the Colab notebook that visualizes the Industrial AI dataset landscape."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cosmos3_vs_industrial_dataset_landscape_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


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
        """# Cosmos 3 vs Industrial AI Dataset Landscape

This notebook compares public Industrial AI datasets with the modalities and operational modes of Cosmos 3.

It answers four questions:

1. Which capability gap does each dataset close?
2. Which Cosmos 3 mode can use it?
3. Is access/license compatible with the intended use?
4. Which dataset mixture should be shortlisted for an Industrial AI pilot?
"""
    ),
    code(
        """from pathlib import Path
import io, os, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 130

LOCAL = Path('industrial_dataset_landscape.csv')
REPO_URL = 'https://raw.githubusercontent.com/celesnity/mini-world-model/main/research/industrial_dataset_landscape.csv'
if LOCAL.exists():
    csv_path = LOCAL
else:
    urllib.request.urlretrieve(REPO_URL, 'industrial_dataset_landscape.csv')
    csv_path = Path('industrial_dataset_landscape.csv')
df = pd.read_csv(csv_path)
print('rows:', len(df), '| families:', df.family.nunique(), '| sources:', df.source_url.notna().sum())
display(df[['dataset','family','origin','modalities','priority','license','evidence_status']])
"""
    ),
    md("## 1. Coverage by dataset family"),
    code(
        """family_counts = (df[df.family != 'Cosmos baseline']
                 .groupby(['family','origin']).size().unstack(fill_value=0))
ax = family_counts.plot(kind='barh', stacked=True, figsize=(10,6), color=['#46a3ff','#ff9f43','#7d5fff'])
ax.set(title='Industrial dataset landscape by family and origin', xlabel='Dataset count', ylabel='')
plt.tight_layout()
plt.show()
display(family_counts)
"""
    ),
    md("## 2. Modality coverage heatmap"),
    code(
        """modalities = {
    'RGB/video': r'RGB|video|image',
    'Depth/3D': r'depth|3D|point cloud|range',
    'Audio': r'audio|sound|microphone',
    'Vibration': r'vibration|accelerometer',
    'Force/torque': r'force|torque',
    'PLC/process': r'SCADA|sensor|process|setpoint|actuator|temperature',
    'Action/control': r'action|command|manipulated|robot',
    'Outcome/failure': r'failure|success|anomaly|fault|RUL',
}
text = (df.modalities.fillna('') + ' ' + df.labels_or_targets.fillna('') + ' ' + df.recommended_use.fillna(''))
coverage = pd.DataFrame({k: text.str.contains(v, case=False, regex=True).astype(int).to_numpy()
                         for k,v in modalities.items()}, index=df.dataset)
industrial = df.family.ne('Cosmos baseline')
industrial_names = df.loc[industrial, 'dataset'].tolist()
plt.figure(figsize=(11,10))
sns.heatmap(coverage.loc[industrial_names], cmap=['#edf2f7','#16a085'], cbar=False,
            linewidths=.4, linecolor='white')
plt.title('Which modalities/signals each industrial dataset contributes')
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()
"""
    ),
    md("## 3. Which Cosmos gaps are filled?"),
    code(
        """gap_families = {
    'Telemetry\\nG01': 'G01',
    'Control\\nG02': 'G02',
    'Outcome\\nG03': 'G03',
    'Degradation\\nG04': 'G04',
    'Real anchor\\nG05': 'G05',
    'Fault taxonomy\\nG06': 'G06',
    'Sync\\nG07': 'G07',
    'Diversity\\nG08': 'G08',
    'Causality\\nG09': 'G09',
    'Inspection\\nG10': 'G10',
    'Audio/vibration\\nG11': 'G11',
    'Procedure\\nG12': 'G12',
    'Sensor faults\\nG13': 'G13',
    'Normal coverage\\nG14': 'G14',
    'Geometry\\nG16': 'G16',
    'OOD/abstain\\nG21': 'G21',
}
gap = pd.DataFrame({name: df.cosmos_gaps_filled.fillna('').str.contains(gid).astype(int).to_numpy()
                    for name,gid in gap_families.items()}, index=df.dataset)
plt.figure(figsize=(14,10))
sns.heatmap(gap.loc[industrial_names], cmap=['#f5f6fa','#8e44ad'], cbar=False,
            linewidths=.4, linecolor='white')
plt.title('Dataset-to-Cosmos industrial gap coverage')
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show()
"""
    ),
    md("## 4. Dataset similarity to the Cosmos 3 data contract"),
    code(
        """# A transparent heuristic: breadth of useful signal types, not a claim of model quality.
contract_cols = ['RGB/video','Depth/3D','Audio','Force/torque','PLC/process','Action/control','Outcome/failure']
score = coverage[contract_cols].sum(axis=1)
rank = (df.assign(contract_breadth=score.values)
          .query("family != 'Cosmos baseline'")
          .sort_values(['contract_breadth','priority'], ascending=[False,True]))
display(rank[['dataset','family','contract_breadth','cosmos_alignment','main_limit','priority']].head(12))

plt.figure(figsize=(10,6))
top = rank.head(12).sort_values('contract_breadth')
plt.barh(top.dataset, top.contract_breadth, color='#2d98da')
plt.xlabel('Number of Cosmos-relevant signal groups present (heuristic)')
plt.title('Closest public datasets to an omnimodal world/action record')
plt.xlim(0, len(contract_cols))
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """**Interpretation:** REASSEMBLE and RH20T are closest to the Cosmos world-action concept because they synchronize several sensory streams with actions. Inspection, audio and PLC datasets remain essential, but each is a specialist rather than an omnimodal world model dataset.
"""
    ),
    md("## 5. Access and license risk"),
    code(
        """def license_bucket(x):
    x = str(x).lower()
    if 'cc0' in x or 'cc by 4.0' in x:
        return 'Lower friction'
    if 'sharealike' in x or 'by-sa' in x:
        return 'Share-alike review'
    if 'noncommercial' in x or 'by-nc' in x:
        return 'Research/noncommercial'
    return 'Terms/legal review'

lic = df.loc[industrial].copy()
lic['license_bucket'] = lic.license.map(license_bucket)
summary = pd.crosstab(lic.family, lic.license_bucket)
summary.plot(kind='barh', stacked=True, figsize=(10,6),
             color={'Lower friction':'#20bf6b','Share-alike review':'#f7b731',
                    'Research/noncommercial':'#eb3b5a','Terms/legal review':'#8854d0'})
plt.title('License/access review by dataset family')
plt.xlabel('Dataset count')
plt.ylabel('')
plt.tight_layout()
plt.show()
display(lic[['dataset','access','license','evidence_status']].sort_values('evidence_status'))
"""
    ),
    md("## 6. Interactive shortlist"),
    code(
        """from ipywidgets import interact, Dropdown

families = ['All'] + sorted(df.loc[industrial,'family'].unique().tolist())
priorities = ['All','P0','P1','P2']

@interact(family=Dropdown(options=families, value='All'),
          priority=Dropdown(options=priorities, value='All'))
def shortlist(family='All', priority='All'):
    q = df.loc[industrial].copy()
    if family != 'All':
        q = q[q.family.eq(family)]
    if priority != 'All':
        q = q[q.priority.eq(priority)]
    display(q[['dataset','domain','origin','modalities','recommended_use',
               'main_limit','access','license','source_url']])
"""
    ),
    md("## 7. Recommended compositional curriculum"),
    code(
        """curriculum = pd.DataFrame([
    ['A. Cosmos foundation', 'Cosmos reasoner/generator/robot data',
     'General language, spatial, temporal, physical and action priors'],
    ['B1. Visual specialist', 'VisA + Real-IAD; evaluate on MVTec AD 2',
     'Defect localization, multi-view and visual OOD'],
    ['B2. Machine-state specialist', 'MIMII + DCASE + Paderborn',
     'Audio, vibration, electrical condition and unseen machines'],
    ['C. Process dynamics', 'Tennessee Eastman + SWaT + HAI',
     'State-action forecasting, interventions and process faults'],
    ['D. Long horizon', 'XJTU-SY + C-MAPSS',
     'Degradation and remaining useful life'],
    ['E. Procedure/action', 'Assembly101 + REASSEMBLE/RH20T',
     'Steps, mistakes, contact and synchronized robot action'],
    ['F. Company alignment', 'Factory-native synchronized episodes',
     'Bind public capabilities to target machines, products and outcomes'],
], columns=['stage','data','purpose'])
display(curriculum)
"""
    ),
    md("## 8. Final comparison"),
    code(
        """comparison = pd.DataFrame({
    'Dimension':['Breadth','Physical control','Real defects','Hidden machine state',
                 'Long-horizon health','Factory outcomes','Best role'],
    'Cosmos 3':['Very high','High for robots','Limited industrial depth','Limited public coverage',
                'Weak','Weak','General world-model foundation'],
    'Industrial public datasets':['Narrow per dataset','Strong in PLC/robot subsets','Strong',
                                  'Strong in audio/vibration/telemetry','Strong in RUL subsets',
                                  'Partial','Capability specialists'],
    'Company factory data':['Target-specific','Target-specific','Target-specific','Target-specific',
                            'Potentially strong','Strong','Alignment and deployment evidence'],
})
display(comparison)

display(Markdown('''
### Decision

Do not choose between Cosmos 3 data and Industrial AI data. Use Cosmos as the shared representation foundation, public industrial datasets as specialists, and synchronized company data as the alignment layer. A dataset enters training only after provenance, leakage, OOD, synchronization and real-only vs real+synthetic admission checks pass.
'''))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
