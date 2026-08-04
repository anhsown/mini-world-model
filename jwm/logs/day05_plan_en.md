# JWM — DAY 5 Plan · After JWM-Read v3 training

> Goal: independently establish whether v3 uses real document pixels, decide
> whether it can become the foveal encoder of JWM-Eye, and only then build the
> object-centric and future-latent foundation of JWM-Eye Physical.

## Non-negotiable gates

- Audit checkpoint/config/history and reject corrupt, non-finite or blocked artifacts.
- Benchmark v1/v2/v3 under one harness on synthetic random OCR, VietDocVQA,
  MTVQA-VI and a camera/OOD suite.
- Run correct, shuffled, blank, wrong-crop and correct-crop image controls.
- Promotion requires all internal v3 gates, real correct-image win rate above
  0.60 with a lower 95% confidence bound above chance, a clearly positive real
  vision gap, and at least 15% relative median-CER improvement on two real/OOD
  suites versus v2.
- A checkpoint file or low teacher-forced loss is not proof of vision learning.

## Decision branches

- **PASS:** archive v3 as the foveal checkpoint and proceed to Eye Physical.
- **PARTIAL:** vision use is real but transfer is weak; run a short targeted
  domain-adaptation ablation.
- **FAIL:** blind control remains near zero; stop promotion and localize the
  Reader architecture/data failure before adding temporal modules.

## JWM-Eye Physical v1 hypothesis

- A low-resolution peripheral path handles all 30 FPS frames for motion,
  change, saliency and collision cues.
- The accepted JWM-Read encoder handles high-resolution keyframes and selected
  crops at roughly 2–5 Hz.
- Persistent object slots hold identity, mask/box, depth, SE(3), velocity,
  visibility, uncertainty and timestamp.
- Ego-, world- and object-centric coordinate transforms separate camera motion
  from object motion.
- FLARE-style future tokens predict EMA/frozen future embeddings at multiple
  horizons; pixel-video generation is reserved for detailed rollouts and audit.
- A latent-action tokenizer is prepared from unlabeled frame transitions but is
  never treated as a real control command without an embodiment adapter and
  safety controller.

## Data contract

Use a balanced mixture of OCR retention, real temporal video, robot/action
trajectories, curated simulation and counterfactual controls. Before training,
validate licenses, decodability, timestamp/FPS integrity, sequence-level split,
track continuity, coordinate conventions, action-transition alignment,
real/sim sampling ratios, event diversity, anti-shortcut controls and privacy.
Acquire additional data by failure cluster rather than random scale alone.

## Curriculum

1. E0 temporal bootstrap;
2. E1 object state and tracking;
3. E2 spatial reference frames;
4. E3 future-latent alignment;
5. E4 latent-action discovery;
6. E5 controlled joint Reasoner–Generator/Policy adaptation.

The accepted Reader and Generator remain frozen or adapter-only through E0–E4.

## Required evaluation

- OCR CER/ANLS retention;
- grounding mIoU/mAP;
- HOTA/IDF1 and ID switches;
- depth AbsRel/δ1 and pose error when labels exist;
- correct-order versus shuffled/reversed temporal controls;
- future-latent retrieval and collapse checks;
- correct-action versus zero/shuffled-action controls;
- object permanence and trajectory error;
- 30 FPS capture/fast-path throughput, p50/p95 latency, VRAM and ECE;
- closed-loop task success and the full auditable trial schema.

## End-of-day outputs

If v3 passes: reproducible benchmark report, promotion manifest, Eye v1 spec,
prototype modules, validated compact data manifest, tests, 100-step profile and
a T4×2 E0/E1 notebook. If it does not pass: a failure-localization report and
one controlled Reader v3.1 experiment. Day 5 is not complete merely because a
notebook reaches its last cell.

