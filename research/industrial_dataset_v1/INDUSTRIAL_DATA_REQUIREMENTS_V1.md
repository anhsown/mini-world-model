# Industrial World Model Episodes v1 — Data Requirements

Date: 2026-07-23  
Owner: Mai Anh Son — Dataset track  
Status: Draft for technical and business review  
Working name: **IWM-Episodes v1**

## 1. Goal

IWM-Episodes v1 is a synchronized, action-aware Industrial AI dataset for learning:

- what a machine and its environment look and sound like;
- what the machine state is;
- what event or anomaly is happening;
- which action or intervention occurred;
- how the state changed afterwards;
- what operational outcome resulted.

The primary record is an **episode**, not an isolated image or an unstructured sensor row.

```text
observation + machine_state + action/intervention
                         ↓
              future_state + outcome
```

This contract follows the useful part of the Cosmos 3 world/action formulation while adding factory-native telemetry, PLC commands, operational outcomes and governance.

## 2. Frozen v1 scope

### Included

- machine-condition monitoring;
- short-horizon anomaly and event understanding;
- visual quality and spatial grounding where available;
- synchronized RGB video, audio and telemetry;
- operator, PLC and maintenance interventions;
- state and outcome prediction;
- normal, anomalous, recovery and ambiguous/OOD episodes.

### Deferred to v2

- autonomous closed-loop machine control;
- multi-month remaining-useful-life histories;
- full production scheduling and supply-chain optimization;
- unrestricted natural-language maintenance agents;
- synthetic data at scale before real-data admission;
- claims of generalization to every factory or machine family.

## 3. Target users and tasks

| User | Primary task |
|---|---|
| Condition-monitoring engineer | detect and classify machine anomalies |
| Quality engineer | localize product/process defects and assess severity |
| Maintenance engineer | explain likely causes and evaluate interventions |
| Production engineer | predict short-horizon state and operational outcome |
| World-model researcher | train reasoner, forward dynamics and inverse dynamics |
| Data engineer | ingest and synchronize machine, sensor and media streams |

## 4. Unit of data

An episode is a bounded time interval containing:

1. immutable identity and provenance;
2. a shared physical time reference;
3. one or more observations;
4. machine state and operating context;
5. zero or more actions/interventions;
6. zero or more annotated events;
7. a verified outcome;
8. data-quality and governance metadata;
9. world-model training targets;
10. a leakage-safe split group.

An episode may be normal and contain no anomaly. Absence of an anomaly is a meaningful label and must not be inferred from missing annotation.

## 5. Minimum modality contract

### Required for every episode

- at least one RGB video stream;
- at least one telemetry stream;
- operating mode;
- outcome status;
- synchronization metadata;
- provenance, license and split group.

### Required for the preferred v1 release

- RGB video;
- machine audio;
- at least three telemetry channels;
- explicit action or a confirmed `no_action`;
- event interval or confirmed `normal`;
- future-state target.

### Optional

- depth, thermal or event-camera stream;
- vibration waveform;
- PLC/network traffic;
- force/torque;
- product-level inspection masks and 2D/3D boxes;
- maintenance work-order reference.

## 6. Time and synchronization

- All streams use UTC timestamps internally.
- Original local timezone is retained.
- Each stream records sampling rate, clock source and measured/estimated clock error.
- Missing samples are represented by validity masks, not silently interpolated.
- Resampling creates a derived artifact; raw streams remain immutable.
- Action and event timestamps are relative to episode start and must lie within episode duration.
- Preferred maximum inter-stream clock error is 20 ms for video/audio/action and 100 ms for low-rate process telemetry.

## 7. Identity and leakage policy

The split is assigned by group, never by individual frame or row.

```text
split_group = site_id / asset_id / product_batch / capture_day
```

Minimum public evaluation:

- IID validation;
- unseen asset;
- unseen operating condition;
- future-time split;
- OOD/ambiguous split.

No media segment, overlapping time window, product instance or machine cycle may appear in more than one split.

## 8. Action safety policy

- Initial ingestion is read-only.
- The dataset records PLC commands and operator interventions but does not authorize the model to execute them.
- `safety_authorized` indicates whether the recorded action was approved by the existing factory control process.
- Counterfactual actions are marked synthetic and may not be mixed with executed actions.
- Emergency and safety actions require human/PLC ground truth, not model-generated labels.

## 9. Label and taxonomy policy

Each event must include:

- taxonomy ID;
- start/end time;
- severity;
- label source;
- review status;
- confidence;
- `unknown` or `ambiguous` when evidence is insufficient.

Root cause must not be inferred from correlation alone. Use:

- `confirmed`: maintenance or instrumentation evidence;
- `probable`: SME-supported but not directly verified;
- `unknown`: insufficient evidence.

## 10. Outcome contract

Every episode ends with one of:

- `normal_completion`;
- `quality_pass`;
- `quality_fail`;
- `recovered`;
- `degraded_continuing`;
- `stopped`;
- `unknown`.

When available, include downtime, recovery time, product disposition and post-action state. Missing business outcomes are explicit nulls with a reason.

## 11. Governance

Every episode carries:

- data owner;
- source type;
- license;
- permitted use;
- privacy review;
- redaction status;
- retention class;
- whether redistribution is permitted;
- transformation/provenance chain.

Public v1 may contain only company-owned data with explicit release approval, or third-party data whose terms permit redistribution. External research datasets with incompatible licenses are referenced through adapters, not copied into the release.

## 12. Seven public-release admission gates

| Gate | Pass condition |
|---|---|
| G1 Rights | ownership, consent, license and redistribution rights documented |
| G2 Privacy | faces, badges, screens and sensitive process information reviewed/redacted |
| G3 Synchronization | timestamps valid; clock error within declared limits |
| G4 Labels | taxonomy valid; required SME review completed; no impossible intervals |
| G5 Leakage | no site/asset/batch/time overlap across splits |
| G6 Coverage | required normal, anomaly, recovery and OOD conditions represented |
| G7 Benchmark | loader, baseline, metrics and checksums reproduce on a clean environment |

The release decision is binary: **7/7 or blocked**.

## 13. Dataset tiers

| Version | Minimum deliverable |
|---|---|
| v0.1 schema preview | schema, taxonomy, validator and 10–50 golden samples |
| v0.5 pilot | 3 asset families, ≥1,000 episodes, complete benchmark |
| v1.0 public | 3–5 asset families, 10K–30K normal episodes, ≥1K anomaly/intervention episodes, public test protocol |

Numbers are targets, not permission to manufacture redundant samples. Coverage and held-out performance take priority over raw volume.

## 14. Review decisions required

Before collection begins, Bill/Đạt and the relevant factory stakeholders should approve:

1. the pilot asset families;
2. which telemetry and commands may leave the OT environment;
3. the public/private feature boundary;
4. fault taxonomy ownership;
5. outcome definitions;
6. release license;
7. safety and privacy review owners.

## 15. Companion artifacts

- `industrial_feature_dictionary.csv`
- `industrial_event_taxonomy_v1.json`
- `episode.schema.json`
- `golden_samples_v1.json`
- `validate_golden_samples.py`

