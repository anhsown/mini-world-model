# mini-world-model (JWM)

A **micro omnimodal world model** in the spirit of NVIDIA Cosmos 3 — built from
scratch, trained end-to-end on a single **GTX 1650 (4GB)**, and deployed as the
vision brain of a Vietnamese-speaking desktop assistant.

Dual-tower Mixture-of-Transformers: an autoregressive **reasoner** (causal) and a
rectified-flow **generator** (bidirectional, AdaLN-zero) share one token sequence
with strict one-way AR→DM information flow. One model, four modes selected purely
by token arrangement:

| Mode | Input → Output |
|---|---|
| **QA** | image + question (vi/en) → byte-level answer |
| **GROUND** | image + referring expression → bbox *action token* + **calibrated confidence** P(IoU≥0.5) |
| **FD** | image + motion text → next-frame latent (forward dynamics) |
| **T2I** | structured caption → scene image |

## Highlights

- **All math hand-written and property-tested** (29 tests): 3D MRoPE with absolute
  temporal modulation, rectified flow (analytic x₀-recovery tests), two-way flat
  attention with gradient-isolation proofs, AdaLN-zero identity-at-init,
  mutation-proof CE alignment.
- **Cosmos-3-faithful staged training** ([jwm/stages/](jwm/stages/)): reasoner
  pretrain → SFT, then generator pretrain (**reasoner→generator tower copy**) →
  mid-train (action modality enters) → three post-training stages → deployable
  checkpoint. Every stage is shutdown-safe and resumable; one command chains them.
- **Two-branch data builders** ([jwm/data_builders/](jwm/data_builders/)):
  procedural scenes composited on real camera frames, with a camera-degradation
  model **auto-tuned until 5 Wasserstein statistics match real frames**; 3-axis
  programmatic judge; strict-quantile post-training tiers.
- **Calibrated confidence** (learned head + Platt): test ECE **0.025** — the model
  *knows what it doesn't know* (100% precision when asserting in trials; always
  abstains on out-of-distribution scenes).
- **Inkling-mini MoE reasoner** ([jwm/moe.py](jwm/moe.py), experiment in flight):
  fine-grained experts (hidden = d/2), sigmoid top-k routing, shared expert —
  74M total / 31M active per token on the same 4GB card.
- Documented deviations & critique: [jwm/DESIGN.md](jwm/DESIGN.md),
  [jwm/COSMOS3_CRITIQUE.md](jwm/COSMOS3_CRITIQUE.md),
  [jwm/INKLING_MINI.md](jwm/INKLING_MINI.md), daily logs in [jwm/logs/](jwm/logs/).

## Results (31M "v3", test set, single GTX 1650)

| Metric | Value |
|---|---|
| QA exact-match (byte-level, vi/en) | **58.8%** |
| Grounding mIoU / IoU@0.5 (4-step sampler) | **0.285 / 0.268** |
| Calibrated ECE | **0.025** |
| T2I self-consistency (own-reasoner judge) | 0.48 pos / 0.75 neg |
| Ground latency (4 denoising steps) | **~104 ms** |
| Full 7-stage pipeline wall-clock | 224 min |

## Quickstart

```bash
pip install torch pillow numpy pytest matplotlib

# 1. verify the math (29 property tests)
pytest tests/test_jwm_math.py -q

# 2. build the 2-branch data corpus (needs reference frames; see data_builders/common.py)
python -m jwm.data_builders.build_all

# 3. run the unified 7-stage pipeline (shutdown-safe; resumes where it stopped)
python -m jwm.stages.run_pipeline

# 4. watch it live
python scripts/pipeline_dashboard.py   # -> http://localhost:8877
```

## Repository map

```
jwm/
  mathx.py          hand-written math: MRoPE, rectified flow, rot6d, IoU/ECE
  layers.py         dual-tower MoT block, two-way attention, AdaLN-zero
  moe.py            Inkling-mini MoE FFN (sigmoid router, fine-grained experts)
  model.py          JWM: sequence assembly, 4 modes, losses, samplers, conf head
  sdg.py            procedural scene generation + camera model + judge
  data_builders/    reasoner branch (2 types) | generator branch (3 types)
  stages/           r1, r2, g1..g5 + run_pipeline (checkpoint-chained)
  DESIGN.md         the specification — code maps 1-1 to these equations
  logs/             daily engineering logs (vi + en), including the full
                    "68M debugging saga" — a case study in isolation testing
integration/
  world_brain.py    drop-in brain module for the JARVIS assistant (shadow mode,
                    reflection pass, auto checkpoint selection)
scripts/            trials harness (9-field logging), Platt calibration,
                    diagnostics, MoE experiment, live dashboard
tests/              29 property tests
```

Not in this repo: trained checkpoints, generated datasets, and training notebooks
(they embed frames from a private camera). All are reproducible from source.
