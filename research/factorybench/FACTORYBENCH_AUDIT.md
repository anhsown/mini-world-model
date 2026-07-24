# FactoryBench Audit for Industrial JWM

Generated from revision `e2ad55f2c4a66d3f3190170cfe06a50cc93a2e46`.

## Decision

- **Research/benchmark admission:** PASS
- **Direct commercial training:** BLOCKED by CC BY-NC 4.0
- **Direct ingestion through current JWM text path:** BLOCKED; a numerical
  time-series/action encoder is required.

## Verified inventory

| Level | Train | Validation | Test |
|---|---:|---:|---:|
| L1 | 12,674 | 1,338 | 1,309 |
| L2 | 33,311 | 3,428 | 3,487 |
| L3 | 2,353 | 265 | 321 |
| L4 | 9,947 | 1,250 | 1,232 |

The repository contains **70,915 parseable Q&A
records**, which is 3 fewer than
the 70,918 stated on the dataset card. The repository is
4.124 GiB across
22 files.

## Leakage audit

No record ID or `(source dataset, episode)` pair crosses
train/validation/test. Reusing an episode across multiple causal levels inside
the same split is allowed, but level scores must be reported separately.

Evidence is packaged as 61,707 single-stream,
7,350 paired-stream and
1,858 option-only records. An adapter
that reads only `context.time_series` silently drops the comparative items.

## Architecture implication

Median serialized question + time-series context is
**11,989 bytes** and 100.0%
of records exceed JWM's 96-byte question path. Stringifying telemetry into AR
text would therefore truncate almost all physical evidence. FactoryBench
should enter JWM through:

1. a numerical sensor-history encoder;
2. a distinct control/action projection;
3. machine-context tokens;
4. the AR reasoner for the natural-language question and response.

Static machine/task context is not consistently embedded in the Q&A telemetry
object. FactoryWave records require a provenance-safe join with
`episodes.parquet` and the knowledge graph. Hidden fault/answer fields must be
excluded from model input.

## Target-reuse warning

100.0% of L4 free-form test targets appear verbatim in the
training target set. This is not source-episode leakage, but it makes exact
match and token-F1 vulnerable to template memorization. L4 evaluation must also
report root-cause correctness and evidence grounding.

## Admission hypotheses

- [x] `H_public_and_ungated`
- [x] `H_license_recorded`
- [x] `H_json_parseable`
- [x] `H_required_fields_present`
- [x] `H_split_ids_disjoint`
- [x] `H_source_episodes_disjoint`
- [x] `H_sensor_channels_available`
- [x] `H_control_channels_available`
- [ ] `H_current_jwm_text_path_sufficient`
- [ ] `H_commercial_reuse_cleared`

## Shareable samples

`representative_samples_l1_l4.json` contains 46 compact,
provenance-preserving records stratified by level, template, source and answer
family. Time series are head/tail sampled only for visualization; benchmark
evaluation must use the complete source records.

## Source

https://huggingface.co/datasets/FactoryBench/FactoryBench
