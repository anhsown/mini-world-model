# FactoryTraj-B0 Seed Report v0.1

## Decision

**The B0 benchmark dataset is admitted (6/6 gates).** This does not mean the
current JWM has passed B0: JWM has not yet produced structured predictions for
this benchmark.

This is a useful negative result: reporting the full-name rule score alone
would incorrectly claim that B0 is solved.

## Seed composition

- 125 FactoryNet schema tags, 85 Tennessee Eastman variables and 34 official
  OPC UA Companion Specification analog variables.
- 244 total records: 82 train and 162 validation/test.
- 10 source families, including CNC, mining, machinery, commercial-kitchen,
  glass, metal-forming and PADIM NodeSets.
- Held-out Cartesian and auxiliary tag families.
- Published data types and S-E-F-C roles.
- Units only where the physical quantity is recognizable; these remain marked
  for source-owner review.
- Tennessee Eastman XMEAS units and XMV range 0-100 come directly from
  simulator source code; no FactoryNet range was fabricated.
- Mole-composition outputs use the physically bounded 0-100 mol-% domain.
- OPC UA ranges, units, descriptions and hierarchy come from official
  normative NodeSet values.

FactoryNet defines Setpoint, Effort, Feedback and Context prefixes and exposes
the normalized channel schema, but does not provide complete authoritative
range metadata in the dataset card.

## Baseline results

| Baseline | B0 macro | Type accuracy | Unit accuracy | Role macro-F1 | Range | Relationship F1 |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.168 | 0.784 | 0.000 | 0.055 | 0.000 | 0.000 |
| Rule, full tag name | 0.413 | 1.000 | 0.073 | 0.601 | 0.000 | 0.393 |
| Rule, anonymized tag | 0.202 | 1.000 | 0.000 | 0.010 | 0.000 | 0.000 |
| Rule, anonymized tag + partial docs | 0.335 | 1.000 | 0.000 | 0.675 | 0.000 | 0.000 |

The no-documentation control still exposes a severe tag-name shortcut.
However, after hiding names while retaining legitimate partial documentation,
role macro-F1 reaches 0.708. This passes the role shortcut-control gate without
leaking the target through the tag prefix.

## Label coverage

| Label | Coverage |
|---|---:|
| Data type | 100.0% |
| Engineering unit | 76.5% |
| Authoritative range | 54.9% (89 records) |
| Role | 100.0% |
| Relationships | 37.7% |

## Admission gates

| Gate | Result |
|---|---|
| At least 100 items | Pass |
| At least two source families | Pass |
| Unit coverage >= 50% | Pass |
| At least 50 authoritative ranges | Pass |
| Relationship coverage >= 20% | Pass |
| Full-name vs anonymized-with-docs role gap <= 0.20 | Pass |

Overall decision: `ready_to_freeze_threshold_and_evaluate_jwm`.

The model pass threshold is now frozen in `b0_pass_threshold_v0.1.json`.

## Why JWM was not scored

The current JWM checkpoint was not evaluated because it has no structured
tag-contract adapter/output head. Treating unsupported output as a model score
would confuse interface absence with measured capability.

## Required next data

1. Implement a structured B0 adapter that emits the frozen JSON contract.
2. Produce both full-name and anonymized predictions.
3. Run `scripts/adjudicate_factorytraj_b0.py` exactly once per checkpoint.
4. Keep real-machine OPC UA exports as the next external-validity extension.

## OPC UA acquisition probe

The reusable collector was tested against the public Prosys OPC UA Simulation
Server. It collected eight variables and representative values, but that demo
address space exposed 0% EngineeringUnits and 0% InstrumentRange/EURange for
the selected variables. It is therefore useful for validating connectivity,
not as authoritative B0 ground truth.

Use `scripts/collect_opcua_b0_metadata.py` against a machine endpoint, then
review/export the fields in `b0_opcua_export_template.csv`. The validator
rejects missing IDs, roles, relationship JSON and unreviewed metadata.
`scripts/import_b0_opcua_export.py` appends reviewed rows, and
`scripts/run_factorytraj_b0_seed.py --data <merged.json>` reruns admission.

The Tennessee Eastman disturbance spans were audited and rejected as range
labels: they encode disturbance amplitudes, not InstrumentRange or EURange.
One-sided shutdown thresholds were also kept separate from authoritative
two-sided engineering ranges.

## Sources

- [FactoryNet dataset card](https://huggingface.co/datasets/factorynet/factorynet/blob/main/README.md)
- [FactoryNet schema viewer](https://huggingface.co/datasets/Forgis/FactoryNet)
- [OPC UA AnalogItem semantics](https://reference.opcfoundation.org/specs/OPC-10000-8/5.3.2)
