# FactoryBench Metrics and Admission Gates

## Why exact match is insufficient

FactoryBench has leakage-safe source episodes, but many questions and answers
are template-reused across splits. In the audited revision:

- 99.3% of L1 test questions occur verbatim in train;
- 90.5% of L2 test questions occur verbatim in train;
- 11.2% of L3 test questions occur verbatim in train;
- 100% of L4 test questions and answers occur verbatim in train.

A model can therefore earn non-trivial text overlap without reading telemetry.
Every result must include a context-blind prior and a shuffled-context control.

## Task-aware metrics

| Answer family | Primary metric | Required secondary metrics |
|---|---|---|
| Single choice | Accuracy | Macro-F1, confidence ECE |
| TF multi-label | Label F1 | Exact-set accuracy, per-label precision/recall |
| Ranking | Pairwise ordering accuracy | Exact ranking |
| Numeric scalar | Within provided tolerance | MAE, normalized absolute error |
| Numeric vector | Mean dimensions within tolerance | Vector MAE, all-dim pass rate |
| L4 free form | Root-cause accuracy + evidence grounding | Protocol-step F1, unsupported-claim rate, token-F1 |

`token-F1` is diagnostic only for L4. It cannot be the sole primary metric.

## Mandatory breakdowns

Report:

1. L1, L2, L3 and L4 separately;
2. every answer family separately;
3. FactoryWave, KUKA, AURSAD and voraus-AD separately;
4. micro average and macro-over-level average;
5. worst-source score;
6. 95% bootstrap confidence intervals, resampled by source episode.

Do not bootstrap individual Q&A rows because multiple items share an episode.

## Causal controls

For the same trained checkpoint evaluate:

- **Correct context:** matching telemetry and question.
- **Shuffled context:** telemetry replaced by another episode with the same
  source/shape where possible.
- **Zero context:** sensor and control values masked, metadata preserved.
- **Control ablation:** setpoint/action channels removed.
- **Sensor ablation:** feedback/effort channels removed.

The model demonstrates physical evidence use only when correct context
outperforms the matched controls.

## Provisional seven admission gates

These are research gates and should be recalibrated after human/expert
baselines.

1. **Data integrity:** all schema, parsing and source-episode split checks pass.
2. **Output validity:** at least 99% of predictions satisfy the requested
   answer format.
3. **Above-prior:** each causal level beats the context-blind template prior
   outside its episode-bootstrap 95% confidence interval.
4. **Context sensitivity:** correct-context score is significantly higher than
   shuffled-context score on L1–L4.
5. **Action dependence:** L2/L3 performance decreases when control/action
   channels are removed; otherwise intervention claims are unsupported.
6. **Robustness/calibration:** every source beats its matching blind prior and
   categorical ECE is at most 0.10.
7. **Grounded decision quality:** L4 reports root-cause correctness,
   protocol-step coverage and unsupported-claim rate; token overlap alone
   cannot pass.

## Current score floor

The verified context-blind template-prior scores are:

| Level | Primary score |
|---|---:|
| L1 | 0.2210 |
| L2 | 0.2401 |
| L3 | 0.2622 |
| L4 | 0.5035 |

Macro over levels: **0.3067**.

L4 is artificially high because repeated remediation text receives token
credit. A future Industrial JWM should not be called successful merely for
exceeding the aggregate floor; it must pass all causal controls.
