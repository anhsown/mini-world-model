# JWM — Project Log · DAY 1 (2026-07-16)

> From the Cosmos 3 paper to a working world-model brain trained on a GTX 1650
> 4GB, integrated into JARVIS with full trial logging.
> *(Multi-day log series — see [README.md](README.md) for the index.)*

---

## Phase 0 — Knowledge Foundation

Deep study of the **NVIDIA Cosmos 3** paper (Omnimodal World Models for Physical AI)
across four sections: architecture (dual-tower MoT, MRoPE, rectified flow, action
tokens), data (two branches, multi-tier curation), training (multi-stage curriculum),
and infrastructure (SILA, packing, caching).

## Phase 1 — JWM Architecture (v1, 10.7M params)

**Package `jwm/`** — written from scratch; every equation maps 1-1 to the spec in
[DESIGN.md](DESIGN.md):

| File | Contents |
|---|---|
| `mathx.py` | 3D MRoPE (float coords, temporal modulation δt=TPS_base/TPS), rectified flow (xσ, v*, Euler sampler, shift schedule), logit-normal/mode σ-sampling, rot6d↔SO(3), IoU, ECE, PSNR, sqrt-length loss normalization |
| `layers.py` | Dual-tower MoT block (causal reasoner / bidirectional generator), two-way flat attention (2 SDPA calls), AdaLN-zero, detached + cacheable reasoner K/V |
| `model.py` | 4 modes: QA (AR) / GROUND (bbox action token with clean-latent conditioning) / FD (next-frame prediction) / T2I (added in v3); calibrated confidence head P(IoU≥0.5); tower-copy init |
| `tokenizer.py` | Byte-level, 262 vocab (Vietnamese-safe by construction) |

**Two verification rounds, as required:**
- **27 property tests** (`tests/test_jwm_math.py`): RoPE relative-position invariance,
  analytic x₀ recovery for rectified flow, AR *never* sees DM (perturbation + gradient
  isolation), AdaLN-zero = identity at init, CE alignment pinned by a mutation-proof
  overfit test, cached-sampler numerical equivalence, analytic CFG.
- **Adversarial audit workflow (30 agents)** reading the actual code: 25 findings →
  5 confirmed → all fixed (reasoner K/V caching ≈8× cheaper sampling, CFG for the FD
  sampler, 3 test gaps).

**Cosmos 3 critique** ([COSMOS3_CRITIQUE.md](COSMOS3_CRITIQUE.md)) — a 38-agent
workflow (web research + 4 critique lenses + adversarial judging): 29 candidate
weaknesses → **7 survived**. Most severe: the self-reported JSON confidence is
*architecturally blind* to the generator's own sample. **3 improvements implemented:**
1. Learned confidence head + Platt calibration, measured by ECE (replacing verbalized confidence)
2. Learned-boundary-embedding vs fixed-15000-gap ablation
3. Inference-time reflection pass (the reasoner re-verifies generator output at the pipeline level)

## Phase 2 — SDG Dataset (validated against real cases)

`sdg.py` — SDG-JarvisSim: procedural scenes (6 shapes × 6 colors, physics motion,
occlusion), backgrounds cropped from **real JARVIS camera frames**, a camera-degradation
model **auto-tuned until 5 Wasserstein statistics** (luminance, contrast, gradient
energy, sharpness, color entropy) **fall below thresholds vs real frames** — exactly
the requested "rebuild until valid" loop. 3-axis programmatic judge
(faithfulness/completeness/correctness), scene-hash dedup, Cosmos-style structured captions.

## Phase 3 — v1 Training via an Observable Notebook

`train_world_brain.ipynb` (21 cells) executed live in JupyterLab (Run-All driven via
`window.jupyterapp`), watched in real time. v1 results (10.7M, test set):

| Metric | v1 | Note |
|---|---|---|
| QA exact-match | **56.4%** (count 81%, exist 64%) | |
| Grounding mIoU / IoU@0.5 | 0.20 / 0.18 | saturated at 10.7M |
| **ECE after Platt** | **0.040** (from 0.414) | meets the critique's <0.05 target |
| FD PSNR (min-over-k) | 20.4 vs copy-baseline 21.4 | |
| Ground latency, 4-step | **95ms** | real-time capable |

Plus: Stage 2.5 injected into the live notebook; `calibrate_confidence.py`.

## Phase 4 — JARVIS Integration + 66 Logged Trials

- `core/world_brain.py`: `off/shadow/primary` modes (default **shadow** — runs beside
  Qwen3-VL, log-only); reflection pass; auto-selects newest checkpoint (v3>v2>v1).
- `_shadow_world_brain` hook in `vision.py` (never allowed to break a live vision turn).
- `scripts/run_world_brain_trials.py`: **every trial records all 9 fields** — audio,
  transcript, frame, predicted region, answer, confidence, latency, ground truth,
  failure category.
- **66 trials** (40 synthetic with exact GT + 26 real WAVs through faster-whisper over
  real frames with hand-annotated boxes): synthetic — the model asserted 5 times and
  was **right 5/5 (100% precision)**, correctly abstained 34/40; real OOD — confidence
  ≈ 0.00-0.02, **never hallucinates certainty**.

## Phase 5 — 30fps Vision Mode (user request)

Investigation: our processing costs ~3ms (not the bottleneck) → the **V380 camera is
hardware-capped at 21.5fps @720p** (MJPG already default; 480p falls back to YUY2 at
11.5fps — worse; exposure props ignored by the driver; only one camera present).
Solution in `vision.py`: split **capture thread** (camera-native rate) + **30Hz display
thread** with **cross-fade frame interpolation** (standard FRC), BGR-direct HUD path,
auto-degrading width, honest two-number telemetry. **Measured: 30.0fps push,
28.3 distinct frames/s, p95 4.3ms.** 46/46 tests pass.

## Phase 6 — v2 28M (superseded) + Shutdown Safety

Monolithic 28M scale-up; mid-run the user needed to shut down → controlled interrupt +
partial checkpoint saved from kernel RAM (lessons: two kernels can coexist — interrupt
the right kernel id; Jupyter cancels the whole queued run-all on interrupt). Since
then **all training checkpoints periodically** (`ckpt_fn` in the trainer). v2 was later
superseded by v3.

## Phase 7 — v3: Paper-Faithful Restructure + 68.65M

User request: one file per training stage + two-branch data + 0.5B.
- **0.5B is infeasible on 4GB** (optimizer state alone ~8GB) → AskUserQuestion → agreed
  on **~80M** (actual 68.65M: d512/L10/h16; VRAM peak 3.01GB; 0.66 it/s).
- **`jwm/data_builders/`** (2 branches, 5 types): reasoner_pretrain (40K) +
  reasoner_sft (12K) | generator_image T2I (8K) + generator_video FD (6K) +
  generator_action bbox (14K). The paper's Audio type is deliberately replaced by Video
  (JARVIS owns ASR/TTS elsewhere). Post-tiers = top-quantile by quality score (fixed
  from a meaningless 99.6%-pass fixed threshold).
- **`jwm/stages/`** (7 checkpoint-chained stages): r1→r2 (reasoner) → g1 (train+freeze
  ConvAE, **reasoner→generator tower copy** per Cosmos §4) → g2 (action enters) →
  g3 (T2I post) → g4 (I2V post) → g5 (policy 4-step + Platt → deployable `jwm_v3.pt`).
  `run_pipeline.py` unifies them; **every stage is shutdown-safe**.
- Model gains **T2I mode** + a self-consistency metric (the model's own reasoner
  verifies its generated images).
- **Live dashboard** `scripts/pipeline_dashboard.py` (localhost:8877, 5s auto-refresh).

## Phase 8 — The v3 Reasoner Debugging Saga (ongoing)

| Round | Action | Result | Lesson |
|---|---|---|---|
| 1 | r1 1800 steps, 14K data | val QA 35% (v1: 56%) | — |
| 1b | +800 r2 steps | 38.4% → **37.2%** | more steps alone did nothing |
| — | Per-kind analysis | count 67%, **what_held 23.5%** | **exact-match ≈ tok_acc^answer_length** (0.95²⁷≈25% ✓) |
| 2 | 3× data (52K), r1 3200 steps | **38%** | more data alone not enough |
| — | 4-number diagnostic + image dump | train/val gap = **0.0001**; train-exact also only 35%; images: **colors right, SHAPES systematically wrong** (circle→square, triangle→square) | not memorization; optimization stuck (3000-step plateau) + shape signal diluted in long answers |
| 3 (now) | Attribute-decomposed curriculum (shape-only / color-only questions, ~10-byte answers) + awaiting the LR verdict from r2 (1.2e-4) | running | — |

Side discovery: a stale `serve_brain.py` (old chat server) had been holding ~1GB VRAM
**through every training run** → killed.

## Overall Statistics

- **~40 new** code/test/doc files; 27 property tests; 2 adversarial workflows (68 agents)
- 66 logged trials; 3 model generations (v1 complete, v2 superseded, v3 in training)
- Infrastructure: shutdown-safe staged pipeline, live dashboard, background monitors, trial harness
- Vision: 5fps → true 30fps display on a 21.5fps camera

## EPILOGUE — Saga closed at 02:16 (early Day 2)

The debugging saga ended with a complete elimination chain: more steps ✗ → 3× data ✗ →
lower LR ✗ → scale/init ✗ (3-arm experiment: 28M/68M/68M-scaled all 0.92 @400 steps) →
labels ✗ (eyeballed) → old-data probe 0.92 @400 → **conclusion: no bug anywhere**. The
two-layer truth: (1) 68M needs a QA-polish budget beyond one GTX-1650 night (68M@3200
steps → tok-acc 0.94; 28M@~2500 → 0.98); (2) the round-3 camera autotune drifted to
noise 10.37 (+44% vs v1), making shapes genuinely harder.

**Final prescription (user-chosen via AskUserQuestion):** run the pipeline at the proven
28M scale (`pipeline_scale()`), pin the camera to v1 parameters, rebuild data with the
attribute-decomposed curriculum (shape-only/color-only questions), batch 48. 68M becomes
the Day-2 research topic.

**Result — the 7-stage pipeline completed in 224 minutes; `jwm_v3.pt` (31M) shipped:**

| Metric (test) | v1 (10.7M) | **v3 (31M)** |
|---|---|---|
| QA exact-match | 56.4% | **58.8%** (where 68%, exist 77%, count 81%) |
| Grounding mIoU / IoU@0.5 (4-step) | 0.201 / 0.184 | **0.285 / 0.268** (+42%) |
| Calibrated ECE (val) | 0.040 | **0.025** |
| T2I self-consistency (new mode) | — | 0.48 pos / 0.75 neg |
| Ground latency, 4-step | 95ms | 104ms |

**Trials (74):** synthetic ok-rate 12.5%→**20%**, mean IoU 0.205→**0.247**, precision when
asserting still **100%** (8/8); real OOD still always correctly abstains (conf 0.02-0.06).
The v3 brain is auto-selected by `world_brain` and runs in JARVIS shadow mode.

Stage-by-stage: r1 54.5% → r2 57.2% → g2 grounding 0.275 → g3 T2I 0.67 → g5 ECE 0.025.
Every stage shutdown-safe, observable via the localhost:8877 dashboard.

## Opening Day 2

1. **68M budget experiment**: how many QA-polish steps does d512/L10 need to reach tok-acc 0.98
2. Consider promoting WORLD_BRAIN_MODE shadow → primary for synthetic-domain scenes
3. FD still below copy-baseline (20.1 vs 21.4) — improvement paths: larger motion, more frames
