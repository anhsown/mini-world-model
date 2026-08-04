# JARVIS / World Model Research — Conversation Context Handoff

**Last updated:** 2026-08-01  
**Workspace:** `C:\Users\ASUS\OneDrive\Desktop\Jarvis-Vision`  
**Primary public repository:** `https://github.com/anhsown/mini-world-model`  
**Team repository:** `https://github.com/celesnity/mini-world-model`

> This document packages the working context of the project for continuation in a new chat or by another researcher. Secrets, API keys, passwords, personal credentials, and authentication tokens have intentionally been omitted.

## 1. Long-term objective

The original objective was to redesign JARVIS as a Physical-AI/world-model agent inspired by NVIDIA Cosmos 3. The intended system should eventually combine:

1. **Eyes:** continuous visual perception, document reading, scene understanding, object/state tracking, geometry and physical dynamics.
2. **Ears:** speech-to-text and environmental audio understanding.
3. **Brain:** fast multimodal reasoning, future-state prediction, action planning, calibrated confidence and continual learning.
4. **Action:** safe interaction with the physical world, with deterministic safety checks and human authority where necessary.

The design principle inherited from Cosmos 3 is to couple understanding, prediction/generation and action rather than maintain disconnected VLM, forward-dynamics and policy models.

## 2. Cosmos 3 knowledge adopted

The project studied the Cosmos 3 architecture, data curriculum, training stages and infrastructure in detail. Important concepts retained in the implementation include:

- Two conceptual branches/towers:
  - autoregressive reasoner;
  - diffusion/generative world model.
- Mixture-of-Transformers-style separation while sharing representations.
- Causal AR attention for reasoning and bidirectional diffusion attention for generation.
- One-way reasoner-to-generator conditioning.
- 3D MRoPE and temporal-position treatment.
- Rectified-flow training and shifted noise schedules.
- Generator initialization from the reasoner where compatible.
- Reasoner pretraining followed by supervised fine-tuning.
- Generator pretraining, mid-training, text-to-image post-training, image-to-video post-training and policy post-training.
- Separate data branches for reasoning and generation/action data.
- Action-conditioned future prediction and policy learning.
- Confidence calibration and abstention as first-class requirements.

Important boundary identified in the company research plan: released Cosmos 3 Reasoner accepts text and visual inputs through its public interface, but does not natively accept arbitrary PLC/SCADA numerical time-series tensors. Therefore Cosmos is a candidate component, not an assumed industrial telemetry backbone.

## 3. JWM architecture generations

### 3.1 Core implementation

Important files in the local implementation include:

- `jwm/config.py`: model configuration and tokenizer/layout settings.
- `jwm/configs.py`: model scales, including dense, MoE and Reader configurations.
- `jwm/mathx.py`: MRoPE, rectified flow, rotation math, IoU, ECE, PSNR and CER.
- `jwm/layers.py`: RMSNorm, SwiGLU, attention, AdaLN/sigma conditioning and MoT blocks.
- `jwm/moe.py`: fine-grained Mixture-of-Experts implementation.
- `jwm/model.py`: complete JWM model, reasoner/generator paths and inference.
- `jwm/data.py`: batching and modality preparation.
- `jwm/trainer.py`: training and evaluation utilities.
- `jwm/sdg.py`: synthetic visual-world data generation.
- `jwm/stages/`: individually separated training stages.
- `core/world_brain.py`: integration layer for JARVIS.
- `core/vision.py`: real-time visual stream and 30 FPS display pipeline.

### 3.2 Tokenization and sequence layout

The initial JWM uses a byte-level tokenizer with special tokens:

- PAD = 256
- BOS = 257
- EOS = 258
- BOQ = 259
- BOA = 260
- BOG = 261

Typical AR layout:

```text
[BOS] [IMAGE TOKENS] [BOQ] question [BOA/BOG] answer
```

### 3.3 Model versions

- **JWM v1:** initial small dense architecture.
- **JWM v3:** staged Cosmos-inspired pipeline; final selected dense configuration was about 31M parameters after larger configurations showed optimization problems.
- **JWM v4:** Inkling-inspired fine-grained MoE reasoner, approximately 73.94M total parameters.
- **JWM-Read:** higher-resolution document-reading architecture with a hierarchical MLP vision stem, patch merging and Vietnamese OCR/document reasoning.
- **Eye v3/v3.1/v3.2 experiments:** geometry, depth, tracking, motion and causal-gate experiments; several checkpoints were blocked because only a minority of causal/OOD gates passed.

### 3.4 Inkling-inspired MoE

The reasoner was redesigned after studying Thinking Machines' Inkling architecture:

- 32 experts.
- Expert hidden dimension approximately `d_model / 2`.
- Top-4 routed experts plus one shared expert.
- Sigmoid routing followed by top-k selection and normalization.
- Dense first transformer layer.
- Switch-style load-balancing auxiliary loss.

Controlled experiment result:

- Dense reasoner pretraining validation QA: about 0.545.
- MoE reasoner pretraining validation QA: about 0.580.
- Dense SFT validation QA: about 0.572.
- MoE SFT validation QA: about 0.608.

The MoE architecture was therefore retained as a meaningful improvement.

## 4. Original staged training pipeline

The pipeline was reorganized so each training stage is a separate file and checkpoint:

### Reasoner

1. `r1_reasoner_pretrain`
2. `r2_reasoner_sft`

### Generator

1. `g1_generator_pretrain`
2. `g2_generator_midtrain`
3. `g3_post_text2image`
4. `g4_post_image2video`
5. `g5_post_policy`

Each stage saves a full checkpoint and supports partial checkpoint/resume. The pipeline is designed to survive shutdowns and continue from the most recent stage.

Representative JWM v3 results:

- Reasoner pretrain QA accuracy: 0.545.
- Reasoner SFT QA accuracy: 0.572.
- Generator mid-training 4-step mIoU: approximately 0.275.
- Final test QA accuracy: approximately 0.588.
- Final 4-step test mIoU: approximately 0.285.
- Calibrated ECE: approximately 0.025.

Representative JWM v4 results:

- MoE reasoner SFT QA accuracy: approximately 0.608 in the controlled experiment.
- Generator mid-training 4-step mIoU: approximately 0.361.
- The generator/future-dynamics path remained weaker than a copy-frame baseline in several evaluations.

## 5. Key lessons from architecture debugging

- Increasing parameter count without increasing optimization budget degraded the reasoner.
- A 68M dense model plateaued around 35–38% exact QA, while the smaller 31M model reached above 54%.
- Long byte-level answers made exact-match collapse approximately according to:

```text
exact_match ≈ token_accuracy ^ answer_length
```

- Shape information was diluted in long answers, leading to a shape/color attribute curriculum.
- Camera-domain autotuning drift changed synthetic image statistics and harmed training; camera parameters were later pinned.
- A stale local serving process consumed about 1 GB VRAM during training and was removed.
- Generator future prediction frequently failed to beat the copy-frame baseline.
- Autoregressive answer generation still lacks a fully optimized KV cache and remains a latency bottleneck.
- The expert dispatch loop is Python-heavy and should eventually be replaced by grouped/batched expert GEMM.

## 6. Dataset work completed

### 6.1 JWM synthetic branches

Two high-level branches were implemented:

- **Reasoner data:** pretraining and SFT data.
- **Generator data:** image/T2I, video/future-dynamics and action/grounding data.

Synthetic data underwent hypothesis checks, deduplication, quality filtering, camera-domain matching and leakage checks before admission.

### 6.2 Vietnamese document reasoning

Downloaded dataset:

- `trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning`
- Approximately 544,795 image records in the published description.
- Local downloaded subset/package previously occupied roughly 11.2 GB with 64,516 page images.
- Research license; do not redistribute without verifying terms.

JWM-Read was redesigned to support:

- 768 px document inputs.
- Patch merging.
- Hierarchical MLP vision stem.
- Vietnamese synthetic-text curriculum.
- OCR/document QA.
- CER and exact-match evaluation.

The first Kaggle Reader training runs improved teacher-forced token accuracy but failed free-running CER gates, exposing a gap between token prediction and autoregressive reading capability.

### 6.3 Real-anchored Eye datasets

Dataset work included TUM RGB-D and Bonn dynamic-scene sources, plus synthetic data anchored to real distributions. Validation compared real-only and mixed-data training. A synthetic-admission ablation reported improvement ratios for depth, ATE, RPE and tracking metrics, but absolute tracking metrics and causal-gate performance remained problematic.

### 6.4 Industrial dataset schema research

A 58-feature industrial data contract was defined, grouped into:

- schema/version;
- identity;
- time and synchronization;
- observations;
- machine state;
- actions;
- events;
- outcomes;
- data quality;
- governance;
- dataset split/leakage controls;
- world-model targets.

The intended trajectory unit is:

```text
Context → Observation → State → Decision → Action → Outcome
```

## 7. FactoryBench and industrial machine understanding

FactoryNet/FactoryBench were identified as close to the desired industrial signal-to-text problem.

FactoryNet uses a structure similar to S-E-F-C:

- Setpoint: commanded position, velocity, acceleration and target torque.
- Effort: current, voltage, force and executed torque.
- Feedback: measured position, velocity and state.
- Context: temperature, operating mode, safety mode, task phase and anomaly labels.

FactoryBench adds question-answer tasks over industrial robot telemetry and supports multiple causal-reasoning levels. It is useful for research but its CC BY-NC 4.0 license must be checked before commercial use.

The company-level target was reframed as:

```text
(sensor history, control signals, machine context) → text response
```

The first output should be understanding/advisory text, not autonomous PLC control.

## 8. B0 benchmark work

A FactoryTraj-B0 Schema and Tag Understanding benchmark was created with:

- 244 total records.
- 82 training/reference records.
- 162 validation/test records.
- 10 source families.

B0 evaluates whether a model can infer:

- tag type;
- unit;
- range;
- role;
- relationships to other tags.

The benchmark and technical reports were pushed to the repositories and documented in Vietnamese and English.

## 9. HATREC + Cosmos 3 experiment

Dataset:

- HATREC video dataset.
- Local location used during testing: `research/hatrec_cosmos3` and a dataset folder containing 546 short assembly videos.
- Seven task labels:
  0. Assembling the spring
  1. Placing the white plastic part
  2. Screwing-1
  3. Inflating the valve
  4. Placing the black plastic part
  5. Screwing-2
  6. Fixing the cable

An automated NVIDIA Build UI runner was developed to:

- authenticate through a dedicated browser session;
- upload each video;
- submit a structured industrial-video prompt;
- capture reasoning and final answer;
- record timeout/no-output/partial cases;
- save Markdown/JSON outputs;
- evaluate predictions against labels.

The full 546-video run was analyzed for:

- task accuracy;
- confusion patterns;
- response completeness;
- reasoning quality;
- latency and timeouts;
- failure categories.

The research team also discovered major HATREC validity concerns:

- filenames can expose labels unless sanitized;
- cycles are visually similar across splits;
- static visual shortcuts may solve the task without temporal understanding;
- V-JEPA reached suspiciously high results under a leaked/shortcut-prone setup.

Future HATREC claims must use renamed neutral files, participant/cycle-aware splits and static-frame controls.

## 10. MMAD benchmark work

MMAD is an ICLR 2025 industrial visual anomaly benchmark:

- 39,670 multiple-choice questions in the local canonical manifest.
- 8,366 unique images.
- 38 industrial product categories.
- Seven major subtask families.
- Published reference includes GPT-4o around 74.9% and a human baseline.

### 10.1 Qwen2-VL

Qwen2-VL-2B was run on Kaggle T4×2 with checkpoint/resume. The complete manifest contains 39,670 questions. The run generated prediction JSONL, scored CSV, metrics JSON and analysis figures. One interrupted run had 26,152 records at the downloaded checkpoint before continuation.

Observed intermediate metrics from a partial run included:

- completion around 21% at that checkpoint;
- micro accuracy around 0.753;
- macro task accuracy around 0.770;
- task-level variation, with Defect Analysis much stronger than Defect Localization.

Final metrics must be read from the latest local `metrics.json`, not inferred from this handoff.

### 10.2 Cosmos 3 through NVIDIA Build

The UI automation initially completed about 221 MMAD questions before sustained no-output behaviour. It captured official endpoint reasoning in the form:

```text
<think>
reasoning
</think>

final answer
```

Official NVIDIA Build results usually contained reasoning, while the community BNB8 Kaggle output initially showed `reasoning_present = 0` because the local parser/model output format did not expose or preserve reasoning in the same way.

### 10.3 Cosmos 3 on Kaggle T4×2

Cosmos 3 Nano Reasoner was hosted with the community checkpoint:

```text
ThePyProgrammer/Cosmos3-Nano-reasoner-bnb8-vllm-und-only
```

It is loaded using `Qwen3VLForConditionalGeneration` because the Cosmos checkpoint declares a Qwen3-VL-compatible Transformers architecture. The weights remain Cosmos-derived; this does not mean Qwen was substituted as the tested model.

Important reporting label:

```text
Cosmos 3 Nano Reasoner, community BNB8 quantized checkpoint,
zero-shot inference on Kaggle T4×2.
```

A smoke test on the first five MMAD questions confirmed working visual reasoning and separated reasoning/final responses. A later run starting around question 230 produced approximately 380 Kaggle records before packaging.

The official UI and Kaggle results were merged for analysis. Checkpoint synchronization through GitHub was proposed so both backends could skip completed `sample_id` values.

## 11. V-JEPA 2 work

V-JEPA 2 ViT-L 300M was tested on HATREC using PyAV for video decoding. The notebook generated:

- full-clip embeddings;
- visible-frame embeddings;
- predictions;
- metrics;
- analysis figures.

V-JEPA is fundamentally an embedding/prediction encoder, not a language reasoner. It can classify labels after adding a trained probe, but it does not natively produce free-form explanations or multiple-choice language answers.

The apparently perfect HATREC result was considered unreliable because of train/test similarity, static shortcuts and label leakage risks.

## 12. Current company research plan — Phase 1

The current plan defines task suite B0–B10, but only tasks supported by available public data should be evaluated in Phase 1.

### Assigned tasks for Sơn

#### A. Cosmos3-Nano zero-shot on ALPI/PIADE

- Dataset: `https://zenodo.org/records/7071747`
- Tasks: **B0 and B5**.
- Mandatory experiment: use the exact same locked episodes and compare three representations:
  1. text;
  2. plot;
  3. text+plot.
- Goal: determine which representation Cosmos understands best.

#### B. Cosmos3-Nano and Qwen2-VL on MMAD subset

- Task: **B3**.
- Compare results directly with published GPT-4o and human references.

### Relevant task definitions

- **B0 — Schema and tag understanding**
  - Input: tag names, samples, units and partial documentation.
  - Output: type, unit, range, role and relationships.
  - Metric: exact/semantic accuracy.

- **B3 — Anomaly and fault localization**
  - Input: signals, alarms and context.
  - Output: anomaly interval/responsible subsystem.
  - Metric: Event-AUPRC and Top-k localization where applicable; MMAD should also retain its published scoring protocol.

- **B5 — Next-state/event prediction**
  - Input: observation window and production context.
  - Output: events/state in the next 30–120 seconds.
  - Metrics: AUPRC, Brier score and sequence edit distance.

## 13. PIADE/ALPI research findings

The Zenodo record is officially named **Packaging Industry Anomaly DEtection (PIADE)**. “ALPI/PIADE” appears to be the internal task alias until a separate ALPI artifact is verified.

PIADE contains data from five packaging machines over roughly 2020–2022. Public metadata describes approximately:

- 429,394 raw production intervals;
- 133 alarm types;
- machine states including idle, production, downtime, performance loss and scheduled downtime;
- start/end/elapsed values;
- input packages (`pi`);
- output packages (`po`);
- production speed in packages/hour;
- one-hour aggregate sequences with state percentages, transitions and alarm counters.

### Required experimental design

1. Audit raw timestamps, schema, units, duplicates, missingness and prevalence.
2. Create leakage-safe canonical episodes with fixed cutoffs.
3. Use raw intervals for the official B5 horizon of 30/60/120 seconds.
4. If only one-hour aggregates are used, report the task as next-hour forecasting rather than canonical B5.
5. Freeze the same episodes and labels for all representation arms.
6. Use text, plot and text+plot with identical task semantics.
7. Report AUPRC and Brier rather than plain accuracy for imbalanced alarm prediction.
8. Compare with no-alarm, persistence, frequency/Markov and available XGBoost/Chronos baselines.
9. Include paired confidence intervals and concrete failure cases.

## 14. Standard Phase 1 report structure

Every model/dataset report should contain:

1. Basic information: model, dataset, date and researcher.
2. Setup: hardware, checkpoint, zero-shot versus trained status and cost.
3. Leakage checks.
4. Main task-appropriate metrics plus a simple baseline on the same data.
5. Shortcut checks for image/video tasks.
6. At least 3–5 concrete failures with error categories.
7. One verdict per dataset:
   - clearly beats baseline;
   - equivalent to baseline;
   - below baseline.
8. Limitations and missing checks.

Do not report accuracy as the principal metric for imbalanced data unless it is also the official published protocol and is accompanied by suitable imbalance-aware metrics.

## 15. Immediate sprint plan

The next sprint should close Phase 1 rather than prematurely fine-tune or build a Phase 2 adapter.

Primary commitments:

- complete the PIADE audit;
- freeze B0/B5 episode and split specifications;
- generate text/plot/text+plot representations;
- host Cosmos 3 Nano on Kaggle T4×2;
- run a pilot and then at least 500 episodes per representation where compute permits;
- calculate B0 exact/semantic accuracy and B5 AUPRC/Brier/sequence metrics;
- compare with simple and team-provided baselines;
- complete the MMAD Cosmos/Qwen comparison;
- produce executed notebooks, failure analyses and standardized reports;
- push reproducible, license-compliant artifacts to GitHub and update Notion.

## 16. Operational practices

- Kaggle T4×2 is the main free training/inference environment.
- Use `Files only` persistence or Save Version/Run All when appropriate.
- Save atomic JSONL predictions after every sample or small batch.
- Save model checkpoints every few hundred optimizer steps.
- Upload critical checkpoints as private Kaggle datasets before a session expires.
- Long runs must support resume by stable `sample_id` rather than positional counters.
- Do not put secrets directly in source code, notebooks, Git history or this handoff.
- Do not redistribute restricted or non-commercial datasets.
- When pushing public repositories, exclude model checkpoints, user audio, camera frames, private data and credentials.

## 17. Repositories and authorship

Two repositories have been used:

- Public/personal: `anhsown/mini-world-model`.
- Team: `celesnity/mini-world-model`.

The requested visible Git author is **anhsown**. Previous Claude/co-author metadata was removed or intended to be removed from active history. Future commits should use the user’s configured Git identity and should not append unrelated co-author trailers.

## 18. Security note

Several credentials and tokens were pasted during the historical conversation. They are not reproduced here. Any credential ever pasted into chat, notebook output, terminal output or screenshots should be considered exposed and rotated. Store replacements only in GitHub/Kaggle/Hugging Face secret managers or environment variables.

## 19. Recommended continuation prompt

Use the following when starting a new task/chat:

```text
Read CONVERSATION_CONTEXT_HANDOFF.md completely. Continue the current Phase 1
research assignment for Sơn. The immediate priority is the mandatory Cosmos 3
Nano zero-shot ALPI/PIADE representation experiment for B0/B5 using identical
locked episodes represented as text, plot and text+plot. Preserve the existing
MMAD/HATREC/Qwen/Cosmos findings, use task-appropriate metrics, prevent leakage,
support Kaggle T4x2 checkpoint/resume and do not expose credentials.
```

