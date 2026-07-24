# FactoryBench → Cosmos 3 / JWM Mapping

## Target contract

```text
(sensor history, control signals, machine context, question)
                              ↓
                    grounded text response
```

This is an **understanding/reasoner** task. FactoryBench is not a substitute for
Cosmos 3 video/audio generation data and does not by itself train a robot policy.

## Token and encoder mapping

| FactoryBench evidence | Industrial role | Cosmos 3 analogue | JWM implementation |
|---|---|---|---|
| `feedback_*`, measured effort/current/force/temperature | Sensor history / observed state | Non-language observation tokens | Numerical time-series encoder |
| `setpoint_*`, target torque, command/output channels | Control/action signal | Action tokens between states | Domain-aware action projection |
| robot/safety/runtime mode, task phase, machine metadata | Machine context | Language/context conditioning | Join allowed fields from episode metadata/knowledge graph, then use context embeddings |
| Natural-language question | Instruction | AR language tokens | Reasoner input |
| Ground-truth answer | Understanding target | Next-token language target | AR response head |
| Alternative/intervention metadata | Counterfactual action | Clean action condition | Intervention token with provenance flag |

## Proposed Industrial Reasoner path

```text
raw multirate channels
   ├─ resample + validity masks
   ├─ robust per-channel normalization
   └─ local temporal encoder
             ↓
      sensor-state tokens ───────────────┐
                                         ├─ causal multimodal attention
control/setpoint vectors → action tokens ┤
machine/task metadata → context tokens ──┤
question → AR language tokens ───────────┘
                                         ↓
                              grounded text response
```

The sensor and control paths must remain distinguishable. Combining both into
one undifferentiated vector would prevent forward/inverse dynamics and weaken
intervention reasoning.

FactoryBench Q&A rows do not consistently embed static machine context inside
the telemetry object. FactoryWave items can be joined to `episodes.parquet` and
the knowledge graph by provenance episode ID. This join must obey `hides` and
must never expose `fault_id`, root cause or answer-derived metadata to the model.

## Training curriculum

1. **L1 state grounding:** identify state, robot/task and short-horizon numeric
   values.
2. **L2 intervention:** predict changes after faults or interventions.
3. **L3 counterfactual:** compare factual and alternative trajectories.
4. **L4 decision:** explain root cause and remediation, grounded by the
   knowledge graph.

Use a mixed replay curriculum after adding each level so L3/L4 training does
not erase L1 state recognition.

## Architectural blockers in the current JWM

- The current QA path accepts at most tens of text bytes, while FactoryBench
  contexts are thousands of bytes.
- Byte-tokenizing floating-point telemetry wastes sequence length and discards
  channel geometry.
- JWM has no factory-domain action projection for heterogeneous control vectors.
- There is no multirate timestamp/validity-mask encoder.
- Free-form L4 responses require evidence and root-cause metrics, not exact
  match alone.

Therefore the first experiment should benchmark and validate the adapter, not
fine-tune the existing text-only QA interface blindly.

## Safety boundary

FactoryBench answers are advisory text. They do not authorize machine control.
Recorded actions, proposed interventions and executable commands must use
different provenance flags, and any future deployment remains read-only until
an external safety controller approves actions.
