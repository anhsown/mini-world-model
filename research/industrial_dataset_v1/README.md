# IWM-Episodes v1 Definition Pack

This folder is the first specification draft for a public, action-aware Industrial AI dataset.

## Artifacts

| File | Purpose |
|---|---|
| `INDUSTRIAL_DATA_REQUIREMENTS_V1.md` | Frozen v1 scope, data requirements and seven release gates |
| `industrial_feature_dictionary.csv` | Data dictionary for 58 features |
| `industrial_event_taxonomy_v1.json` | Event, action, severity and outcome vocabulary |
| `episode.schema.json` | JSON Schema for one synchronized episode |
| `golden_samples_v1.json` | Ten fictitious records covering the schema |
| `golden_samples_validation_report.json` | Latest validation result |
| `build_golden_samples.py` | Deterministically rebuild the examples |
| `validate_golden_samples.py` | JSON Schema and cross-field invariant checks |

## Validate

```bash
pip install jsonschema
python build_golden_samples.py
python validate_golden_samples.py
```

A public release remains blocked until real data passes all seven gates: rights, privacy, synchronization, labels, leakage, coverage and benchmark reproducibility.

## Next review

Bill, Đạt and the factory stakeholders need to approve:

- pilot assets and business use case;
- public versus private feature boundary;
- OT connectors and read-only ingestion policy;
- fault and outcome taxonomy owners;
- public release license.
