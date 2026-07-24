# FactoryBench Context-Blind Baseline

This baseline sees only the causal level and template identity. It does **not**
read the question or telemetry and is therefore a leakage-control floor.

| Level | Test records | Task-aware primary score |
|---|---:|---:|
| L1 | 1,309 | 0.2210 |
| L2 | 3,487 | 0.2401 |
| L3 | 321 | 0.2622 |
| L4 | 1,232 | 0.5035 |

Macro score across L1–L4: **0.3067**

The relatively high L4 token-F1 is not evidence of machine understanding.
FactoryBench contains repeated, templated remediation language. Future models
must report root-cause accuracy, evidence grounding and answer novelty alongside
token overlap. A sensor-aware JWM should beat this floor on every causal level,
not only in aggregate.
