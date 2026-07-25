# Task 1 Report — FactoryTraj-B0 Schema and Tag Understanding

## 1. Executive summary

Task 1 developed and validated **FactoryTraj-B0 — Schema and Tag
Understanding**, the first benchmark in Celesnity's Industrial Machine
Understanding programme.

B0 determines whether a model understands industrial machine tags or merely
guesses their meanings from variable names. For every tag, the model must
recover:

- data type;
- engineering unit;
- InstrumentRange or EURange when available;
- operational role;
- relationships with other tags or machine components;
- prediction confidence.

Final results:

| Item | Result |
|---|---:|
| Total records | 244 |
| Training records | 82 |
| Validation/test records | 162 |
| Source families | 10 |
| Engineering-unit coverage | 76.5% |
| Authoritative range records | 89 |
| Authoritative range coverage | 54.9% |
| Relationship coverage | 37.7% |
| Dataset-admission gates | 6/6 passed |
| Task-contract validation | 11/11 passed |
| Automated tests | 7/7 passed |

The B0 benchmark dataset is now admitted and its model-pass criteria have been
frozen. This does **not** mean that the current JWM has passed B0. JWM does not
yet expose the structured industrial-tag adapter required to produce outputs
under this contract.

---

## 2. Background and objective

Industrial machine understanding must begin with machine-schema
understanding. A sensor may have an explicit name:

```text
reactor_pressure
```

but a real production system may expose the same concept as:

```text
PT_101
AI_0042
ns=4;s=Machine1.Tag27
```

A model that relies on prefixes such as `setpoint_`, `feedback_`, `effort_`
or `ctx_` may score well on a familiar dataset while failing on a new PLC,
vendor or factory.

B0 addresses the following question:

> Can the model reconstruct tag semantics from data types, units, samples,
> documentation and topology, or is it simply memorizing tag names?

B0 provides the foundation for later capabilities:

```text
B0: understand schema and tags
  → B1: estimate machine state
  → B2: detect events
  → B3: predict future state
  → B4–B5: understand causes and effects
  → B6+: rank safe actions
```

If a model cannot distinguish a sensor from a setpoint or actuator command,
its anomaly, dynamics and intervention conclusions cannot be trusted.

---

## 3. Operational definition of B0

### 3.1 Inputs

Each B0 record may provide:

- a tag name or anonymized tag ID;
- declared data type;
- representative samples and a validity mask;
- engineering unit when available;
- InstrumentRange or EURange when available;
- partial tag documentation;
- topology or relationships with other tags.

### 3.2 Required output

The model must return a structured contract:

```json
{
  "tag_id": "PT_101",
  "data_type": "float64",
  "engineering_unit": "kPa",
  "range": {
    "low": 0,
    "high": 3000
  },
  "role": "sensor_feedback",
  "relationships": [
    {
      "relation": "component_of",
      "target_tag_id": "reactor"
    }
  ],
  "confidence": 0.92
}
```

### 3.3 Role taxonomy

| Role | Meaning |
|---|---|
| `sensor_feedback` | A sensor measurement or observed machine state |
| `control_setpoint` | A control command or target value |
| `actuator_effort` | Applied force, torque, current or voltage |
| `machine_context` | Mode, runtime state, contextual temperature or metadata |
| `identifier` | Machine, episode or source identifier |
| `time` | Timestamp or elapsed time |

B0 is not an ordinary question-answering task. The model must produce a
machine-readable contract that can be scored automatically and consumed by
later machine-understanding stages.

---

## 4. Data sources

### 4.1 FactoryNet

FactoryNet contributes 125 industrial-robot schema tags organized around the
S-E-F-C representation:

- **Setpoint:** commanded position, velocity, acceleration and target torque.
- **Effort:** applied current, voltage, force and torque.
- **Feedback:** measured position, velocity and machine state.
- **Context:** temperature, robot mode, safety mode, runtime state and anomaly
  context.

FactoryNet supports evaluation of:

- role classification;
- setpoint–feedback correspondence;
- robot-joint and Cartesian tags;
- data types and physical quantities.

Limitations:

- tag names contain explicit semantic prefixes;
- the dataset card does not provide complete authoritative ranges;
- some units are inferred from physical quantity names and still require
  source-owner review;
- FactoryNet alone cannot establish cross-machine generalization.

Sources:

- [FactoryNet dataset card](https://huggingface.co/datasets/factorynet/factorynet/blob/main/README.md)
- [FactoryNet schema viewer](https://huggingface.co/datasets/Forgis/FactoryNet)

### 4.2 Tennessee Eastman

Tennessee Eastman metadata is parsed directly from simulator source code. B0
uses:

- 73 `XMEAS` measurement variables;
- 12 `XMV` manipulated variables;
- variable descriptions and engineering units;
- the `[0,100]` domain for XMV control variables;
- the physical `[0,100] mol %` domain for composition outputs.

This source adds process variables outside the robot domain:

- flow;
- pressure;
- temperature;
- liquid level;
- composition;
- manipulated control variables.

During the source audit, several values were deliberately rejected as range
ground truth:

- `hspan`, `sspan` and `spspan` encode disturbance duration or amplitude;
- one-sided shutdown thresholds are safety conditions;
- observed sample minima and maxima are not physical instrument ranges.

Only domains with explicit source-code or physical semantics were admitted.

Source:

- [Tennessee Eastman dataset repository](https://github.com/mv-per/tennessee-eastman-dataset)

### 4.3 OPC Foundation UA NodeSets

Official OPC Foundation NodeSets provide normative industrial-schema metadata
across multiple domains:

- CNC;
- mining;
- machinery;
- commercial kitchen equipment;
- glass;
- metal forming;
- PADIM/process instrumentation;
- additive manufacturing.

The parser admits only analog variables whose range values are explicitly
encoded in the XML. Extracted fields include:

- `DataType`;
- `EngineeringUnits`;
- `InstrumentRange`;
- `EURange`;
- `Description`;
- parent/component hierarchy.

Examples:

```text
AsymmetryLoad
EngineeringUnits = %
EURange = [0, 100]
```

```text
CurrentPayload
EngineeringUnits = tonne
EURange = [0, 200]
```

Node hierarchy is represented as:

```json
{
  "relation": "component_of",
  "target_tag_id": "parent_node_id"
}
```

NodeSets provide normative schema metadata rather than live machine
trajectories. They are suitable for B0 construction but do not replace
real-machine validation.

Sources:

- [OPC Foundation UA NodeSets](https://github.com/OPCFoundation/UA-Nodeset)
- [OPC UA Data Access Part 8](https://reference.opcfoundation.org/Core/Part8/v104/docs/5)

---

## 5. Range semantics

### 5.1 InstrumentRange

The physical range that an instrument can return.

Example:

```text
Pressure transmitter: 0–10 bar
```

### 5.2 EURange

The engineering or operational range declared by an OPC UA server.

Example:

```text
Engineering range: 2–8 bar
```

### 5.3 Observed range

The minimum and maximum values observed in a sample window:

```text
Observed over 60 seconds: 4.1–4.8 bar
```

Observed range is stored separately and is never treated as authoritative. A
short observation window does not define the physical limits of an
instrument.

---

## 6. Evaluation metrics

### 6.1 Data-type exact accuracy

The proportion of exactly correct data-type predictions:

```text
float64
int64
uint8
string
```

### 6.2 Unit exact accuracy

The proportion of exactly correct engineering units:

```text
Cel
kPa
A
V
N.m
rad/s
%
```

### 6.3 Role macro-F1

Macro-F1 evaluates role classification while weighting frequent and rare
classes equally.

### 6.4 Range score

Normalized range error:

\[
NRE =
\frac{
|\hat{l}-l| + |\hat{h}-h|
}{
2(h-l)
}
\]

Range score:

\[
RangeScore = 1-\min(1,NRE)
\]

Where:

- \(l,h\) are the authoritative lower and upper bounds;
- \(\hat{l},\hat{h}\) are the predicted bounds.

### 6.5 Relationship macro-F1

This metric evaluates relations such as:

```text
commands
tracks
component_of
```

### 6.6 B0 contract macro score

The B0 macro score averages the components for which valid ground truth is
available. A model cannot pass through the macro score alone: each component
also has an independent threshold.

---

## 7. Detecting the name shortcut

The first seed benchmark relied primarily on FactoryNet. A transparent rule
baseline achieved:

- approximately `0.521` B0 macro with full tag names;
- approximately `0.202` after tag-name anonymization;
- a role macro-F1 drop from approximately `0.621` to `0.012`.

The cause was direct target leakage through prefixes:

```text
setpoint_ → control_setpoint
feedback_ → sensor_feedback
effort_   → actuator_effort
ctx_      → machine_context
```

Reporting only the full-name result would incorrectly suggest that schema
understanding had been solved.

---

## 8. Shortcut controls

### 8.1 Full-name control

The baseline can see the complete tag name. This provides an upper control for
deterministic name parsing.

### 8.2 Anonymized-name control

Tag names are replaced with identifiers such as:

```text
tag_0001
tag_0002
```

No documentation is supplied. This measures absolute dependence on tag names.

### 8.3 Anonymized plus partial documentation

Names remain hidden, while legitimate partial documentation is retained:

```text
Commanded setpoint for joint position.
Measured sensor feedback for joint velocity.
Current payload of the hauling machine.
```

Final role results:

- full-name role macro-F1: `0.601`;
- anonymized plus documentation role macro-F1: `0.675`;
- the gap remains below the `0.20` limit.

This demonstrates that roles can be recovered from documentation rather than
only from prefixes. The anonymized no-documentation control remains in the
benchmark to expose shortcut reliance.

---

## 9. Final baseline results

| Baseline | B0 macro | Type | Unit | Role F1 | Range | Relationship |
|---|---:|---:|---:|---:|---:|---:|
| Majority | 0.168 | 0.784 | 0.000 | 0.055 | 0.000 | 0.000 |
| Full-name rules | 0.413 | 1.000 | 0.073 | 0.601 | 0.000 | 0.393 |
| Anonymized rules | 0.202 | 1.000 | 0.000 | 0.010 | 0.000 | 0.000 |
| Anonymized plus docs | 0.335 | 1.000 | 0.000 | 0.675 | 0.000 | 0.000 |

These baselines:

- establish the majority floor;
- detect name-based shortcuts;
- demonstrate that deterministic rules cannot solve the complete contract;
- support threshold freezing before JWM evaluation.

---

## 10. Dataset-admission gates

The benchmark itself must pass admission before it can be used to judge a
model.

| Gate | Result |
|---|---|
| At least 100 records | Passed |
| At least two source families | Passed |
| Engineering-unit coverage ≥ 50% | Passed |
| At least 50 authoritative ranges | Passed |
| Relationship coverage ≥ 20% | Passed |
| Full-name shortcut gap ≤ 0.20 | Passed |

Final decision:

```text
dataset admission = 6/6 passed
decision = ready_to_freeze_threshold_and_evaluate_jwm
```

### Why the range gate was refined

The original gate required authoritative ranges for at least 50% of all tags.
This conflicted with the contract's “range when available” requirement:
identifiers, strings, modes and context variables do not necessarily have an
EURange.

A fixed 50% ratio can bias the benchmark toward analog variables and distort
the real distribution of PLC or SCADA schemas. The gate was therefore
redefined as:

> At least 50 authoritative range labels from multiple sources.

This change was made before JWM evaluation and was not selected from model
results. After adding properly sourced records, the benchmark still achieved
54.9% range coverage.

---

## 11. Frozen model-pass criteria

A model must pass every threshold:

| Metric | Threshold |
|---|---:|
| B0 contract macro | ≥ 0.60 |
| Data-type exact accuracy | ≥ 0.95 |
| Unit exact accuracy | ≥ 0.60 |
| Role macro-F1 | ≥ 0.70 |
| Range score | ≥ 0.60 |
| Relationship macro-F1 | ≥ 0.60 |
| Full-to-anonymized macro drop | ≤ 0.10 |

Pass rule:

> All primary and robustness thresholds must pass. A high aggregate score
> cannot hide a failed component.

For example, a model with a macro score of 0.70 and a relationship F1 of 0.30
still fails B0.

The frozen criteria are stored in:

```text
research/factorytraj_bench/b0_pass_threshold_v0.1.json
```

---

## 12. Technical pipeline

```text
FactoryNet + Tennessee Eastman + OPC UA NodeSets
                    ↓
       Parse metadata with provenance
                    ↓
      Validate units, ranges, roles and links
                    ↓
            Build the B0 benchmark
                    ↓
        Run dataset-admission gates
                    ↓
         Freeze model-pass thresholds
                    ↓
    Full-name + anonymized model inference
                    ↓
             Frozen adjudication
                    ↓
              B0 pass or fail
```

Main components:

- `jwm/factorytraj_b0.py`: metrics and transparent baselines.
- `jwm/opcua_nodeset_b0.py`: official OPC UA NodeSet parser.
- `scripts/build_factorytraj_b0_seed.py`: benchmark construction.
- `scripts/run_factorytraj_b0_seed.py`: baselines and admission.
- `scripts/collect_opcua_b0_metadata.py`: OPC UA endpoint collection.
- `scripts/validate_b0_opcua_export.py`: reviewed-export validation.
- `scripts/import_b0_opcua_export.py`: new-machine metadata import.
- `scripts/adjudicate_factorytraj_b0.py`: frozen model adjudication.
- `research/factorytraj_bench/b0_seed_v0.1.json`: benchmark records.
- `research/factorytraj_bench/b0_seed_results_v0.1.json`: baseline results.
- `research/factorytraj_bench/b0_pass_threshold_v0.1.json`: frozen gate.

---

## 13. Current JWM status

Two separate conclusions must be maintained.

### The B0 benchmark is admitted

This means:

- the dataset is sufficiently large and diverse;
- unit, range and relationship coverage meet admission requirements;
- shortcut controls are operational;
- thresholds are frozen;
- the benchmark is ready for model evaluation.

### JWM has not passed B0

The current JWM does not yet expose a structured industrial-tag adapter and
output head. It therefore has not produced predictions under the B0 JSON
contract and has not been adjudicated.

The correct conclusion is:

> The B0 benchmark is complete and admitted; JWM has not yet been shown to
> possess schema-and-tag-understanding capability.

Assigning a failure score to an unsupported interface would confuse “not yet
measured” with “measured and failed.”

---

## 14. Next architecture step

The next component is a structured B0 adapter:

```text
Tag ID/name
+ declared data type
+ representative samples
+ partial documentation
+ topology
        ↓
Industrial tag encoder / adapter
        ↓
Shared machine-state representation
        ↓
Structured decoder
        ↓
type + unit + range + role + relationships + confidence
```

Evaluation procedure:

1. Run inference with full tag names.
2. Run inference with anonymized IDs while preserving documentation and
   samples.
3. Store predictions under the frozen JSON contract.
4. Run frozen adjudication.
5. Report every component and failure category.
6. Do not modify thresholds based on checkpoint results.

The benchmark should subsequently be extended with reviewed real-machine OPC
UA exports to measure external validity beyond normative schemas and
simulator data.

---


