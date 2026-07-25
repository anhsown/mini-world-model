# FactoryTraj-Bench Task Contracts v0.1

## Decision

The generic item/output schemas are necessary but not sufficient to define a
benchmark. B0-B10 now each have a research question, required input, hidden
ground truth, observable output, primary metric, mandatory controls, baseline
families and an admission gate. Provisional thresholds must be frozen after
the seed baselines and never tuned on the locked real-machine test set.

## Capability matrix

| ID | Capability | Required output | Primary metric | Key control or gate |
|---|---|---|---|---|
| B0 | Schema/tag understanding | Type, unit, range, role, relationships, confidence | B0 contract macro score | Report all five components; anonymize names |
| B1 | State estimation | Current state distribution and evidence | Macro-F1 | Target >= 0.90; blind/shuffled controls |
| B2 | Cycle/event segmentation | Event intervals and labels | Range/event F1 | Provisional F1 >= 0.85 |
| B3 | Anomaly/fault localization | Interval, subsystem, anomaly score | Event AUPRC | Include delay and false events/hour |
| B4 | Causal diagnosis | Ranked causes with evidence | Top-1 cause accuracy | Beat blind and shuffled controls |
| B5 | Next-state/event prediction | Horizon-specific state/event probabilities | Next-event AUPRC | Beat persistence and specialists |
| B6 | SOP grounding | SOP step, preconditions and citations | SOP-step accuracy | Zero critical contradiction |
| B7 | Workflow planning | Validated, parameterized action plan | Simulator success rate | Zero critical constraint violation |
| B8 | Recovery ranking | Ranked actions, utility, risk, evidence | NDCG@k | Recall@3 >= 0.80; filter unsafe actions |
| B9 | Action-conditioned outcome | Outcome distribution for each action | Outcome Brier score | Beat observation-only prediction |
| B10 | OOD and abstention | OOD score, confidence, abstention reason | Area under risk-coverage curve | No unsafe confident action |

The complete specification is `factorytraj_task_contracts_v0.1.json`.

## B0 operational definition

B0 is not ordinary question answering. Given tag identifiers, representative
samples, validity masks and partial documentation, the model must recover:

- data type;
- engineering unit;
- instrument or engineering range;
- semantic role (feedback, setpoint, command, actuator or context);
- typed relationships to other tags;
- calibrated confidence.

Following OPC UA Data Access, `instrument_range`, `eu_range` and
`engineering_unit` are separate. The range seen in a compact sample is stored
as `observed_range`; it is never treated as an authoritative physical range.

The B0 score averages available component scores: type accuracy, unit
accuracy, role macro-F1, `1 - clipped normalized range error`, and relationship
macro-F1. Label coverage is reported separately so missing metadata cannot
silently improve the score.

## Metric definitions

- **Macro-F1** weights each state/fault class equally.
- **Brier score** is `mean(sum((p-y)^2))` for probabilistic correctness.
- **Range/event F1** evaluates event existence and temporal overlap; reports
  also include onset/offset error, delay and false events/hour.
- **Forecasting** uses next-event AUPRC and Brier score, sequence edit distance,
  and quantile loss/normalized RMSE for continuous targets.
- **Action ranking** uses NDCG@k, Recall@3, regret and invalid-action rate.
- **B9** compares correct actions against observation-only and shuffled-action
  controls.
- **B10** reports the full risk-coverage curve, OOD AUROC/AUPRC, selective
  risk, ECE and unsafe high-confidence recommendation rate.

Point accuracy is prohibited as the sole anomaly metric. Safety errors cannot
be averaged away by high task accuracy.

## Mandatory protocol

1. Split complete trajectories/runs with a temporal boundary embargo.
2. Keep the final real-machine OOD set outside synthetic generation and prompt
   development.
3. Report per source, machine, condition and task before macro averaging.
4. Bootstrap confidence intervals by trajectory, not overlapping window.
5. Run correct-context, shuffled-context and zero-context controls.
6. Compare specialist time-series, retrieval/rule and generalist/hybrid
   baselines.
7. Store evidence pointers and observable outputs, not private chain-of-thought.

## Dataset coverage conclusion

- FactoryBench supports B1, B3-B5 and parts of B8-B9, but its B0 metadata and
  B6-B7 safety/workflow labels are incomplete.
- FactoryWave/FactoryNet are stronger candidates for raw tag semantics and
  telemetry encoder work.
- Tennessee Eastman and SWaT provide process and sensor/actuator transfer.
- MIMII provides audio anomaly/OOD coverage, not full action trajectories.
- Authoritative B6-B9 labels require SOPs, authorization and measured outcomes
  from a simulator, microfactory or customer machine.

## Research basis

- [FactoryBench](https://arxiv.org/abs/2605.07675)
- [OPC UA Part 8 Data Access](https://reference.opcfoundation.org/specs/OPC-10000-8/5)
- [TSRBench](https://arxiv.org/abs/2601.18744)
- [Precision and Recall for Time Series](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)
- [IndustryBench](https://arxiv.org/abs/2605.10267)
