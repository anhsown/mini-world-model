# JWM — Project Log · DAY 3 (2026-07-18)

> JWM-Read: teaching JWM to read Vietnamese text, documents, and OOD scenes — **on
> our own architecture**, trained on Kaggle T4 after proving 4GB local can't carry it.
> *(Log series index: [README.md](README.md))*

---

## Opening — Day 2 legacy

`jwm_v4.pt` (73.94M MoE) is the active shadow brain. Backlog: 8.3s/trial latency
(no KV cache), ECE 0.084 needs recalibration, FD below copy-baseline. Today the
user brought a new task: a 64.5K-page Vietnamese document dataset — "clone it and
train our models."

## 1. The QLoRA detour — a failed experiment worth keeping

First plan: QLoRA Qwen3-VL-2B on the GTX 1650. Both smoke runs died:
- v1 OOM: peft upcasts the 151K-vocab lm_head to fp32 (~1.2GB); LoRA targets
  leaked into the vision tower
- v2 (LLM-only modules, manual kbit prep): no hard OOM, but Windows WDDM spilled
  into shared memory — "9.6GB VRAM" on a 4.3GB card → **112s/step** → 10K steps ≈ 13 days
- User settled two things: (a) "we train on OUR architecture" — no third-party
  model route; (b) "then use Kaggle T4"

Lesson: on 4GB + WDDM, a 2B model even at 4-bit is a speed illusion; and the
project's goal is a self-written architecture, not fine-tuning someone else's.

## 2. Re-architecting the eye — JWM-Read (`reader_scale`)

User's question: "can we re-architect JWM to read text, documents, OOD scenes?"
Answered with design, not a new mode — READ is just QA-mode with a bigger eye:
- **768px** input (up from 64px), patch-16 + **2×2 merge** (Inkling-style
  hierarchical MLP stem, 2 layers) → 24×24 = **576 visual tokens**, img_tok_dim 3072
- New config fields: `patch_merge`, `vision_mlp_layers`; d512/L10/MoE 32 experts
  keeps the Day-2 winning topology → **167.9M total / ~91M active**
- Local smoke: 768px forward+backward at batch 2 = **1.82GB VRAM** → T4 takes batch 16
- New metric: **CER** (Levenshtein over unicode chars — one wrong diacritic = one
  error, not 2-3 byte errors) + per-level `eval_read`

## 3. Data — lazy loading is mandatory, not optional

64,516 pages × 768² uint8 ≈ 50GB of tensors — cannot precompute → `read_data.py`:
- **Unlimited synthetic**: PIL-rendered Vietnamese text, 4 levels (L1 big words
  70-130px → L4 paragraphs 20-36px), paper backgrounds, pinned Day-1 camera degrade
- **Real docs**: 64.5K multi-turn JSONL → flattened to single turns; original
  answers are long (>90% exceed 224 bytes) → fallback to the **first standalone
  answer sentence** → kept **21,641 pairs** (vs ~7K with naive filtering — lesson:
  never truncate mid-sentence, truncated supervision teaches truncation)
- `LazyReadBatcher` renders/loads at batch time + `PrefetchBatcher` (background
  thread hides CPU rendering behind GPU steps); `train_stage` gained a `batcher=`
  injection parameter
- 11 new tests (40 total, all passing): merged-stem shapes, CER properties,
  seed-deterministic rendering, aspect-preserving letterbox

## 4. Kaggle T4 — first off-machine infrastructure

- Self-contained notebook `jwm/kaggle/jwm_read_t4.ipynb`: clone GitHub repo →
  HF dataset download → extract tars → 3-stage curriculum (1.5K/4K/6K steps,
  batch 16) → CER eval → save; **atomic checkpoint every 500 steps**, stage-level resume
- Full set of real-world bumps: GPU greyed out (unverified account → switched),
  private repo clone failure (pushed to public `anhsown/mini-world-model`),
  output files lost to expired session without Persistence (learned **Save & Run
  All (Commit)** — runs in cloud background, output persists forever)
- Notebook stays local, never pushed (privacy rule); source pushed to both
  remotes: celesnity (`32e658f`) + anhsown

## 5. Run complete — jwm_read_v1.pt (671MB)

- Stage 0: tok_acc 0→**0.62** over 1500 steps · Stage 1: 0.51→**0.71** over 4000 ·
  Stage 2 (50% real docs): →**~0.80** over 6000, 0.7 it/s, clean run
- `moe_aux` flat at ~0.090 = balanced router, no dead experts; the `float(la)`
  warning was audited — metric-only, aux gradients still flow (line 333)
- Kaggle eval: doc CER 0.78; synthetic CER **inverted vs difficulty** (L1 29.9 →
  L4 0.82) — the first hint of something bigger (see §6)

## 6. Benchmarks — the model does not read, caught red-handed

User: "run benchmarks, find the dead point." Research: MTVQA (ByteDance, public)
usable; ViTextVQA/ViOCRVQA/5CD-AI all gated; VinText needs detection. Built 3
benchmarks + 1 control (`scripts/bench_read_v1.py`, `bench_read_blind.py`):

**Synthetic ladder (108 samples, seed 2026):**

| Test | CER md | Exact | Contains | Stops (EOS) |
|---|---|---|---|---|
| T0 single char 200px | 52.0 | 0 | 0.17≈chance | 0 |
| T1 single word (120→28px) | 12-15 | 0 | **0** | 0 |
| T2 line / T3 paragraph | 2.3 / 0.82 | 0 | 0 | 0 |
| T4 word 80px + camera noise | 14.8 | 0 | 0 | 0 |

**VietDocVQA held-out (40 pages):** CER md 0.70, exact 0, stop rate **0.80**.
**MTVQA-VI (50 samples):** CER md 2.36, exact 0, contains 0 — answers match the
question *type* (brand question → "The brand of this product is...") with fully
hallucinated content.

**Blind-image control (decisive):** teacher-forced tok_acc with correct vs
SHUFFLED images: synth 0.6068 vs 0.6073, doc 0.7878 vs 0.7879 — **Δ ≈ 0.000,
the model never uses the image when generating text**. The run's 0.78 tok_acc
was 100% language modeling.

**Root cause — shortcut learning from a curriculum design mistake**: v3/v4 were
forced to look because color/shape/position answers are unpredictable from
language; JWM-Read trained on **real Vietnamese words** — highly predictable
from language alone → gradient took the easy path and sat in the parroting
minimum for all 11.5K steps. The only visual signal learned: image-*type*
classification (stops 80% on real docs vs 0% on synthetic). The "CER improves
with length" pattern (52→0.82) is a denominator illusion, not ability.

## 7. Side incident — colleagues' sessions appearing in the app

Shared company Claude account; colleagues enabled remote control → their sessions
appear on every machine on that account. Verified this project's session is
**purely local, no remote control** (no hover label, absent from claude.ai/code).
Recommendation sent: separate accounts per person.

## Day 4 agenda — fix the root cause, retake the same exam

1. **Anti-shortcut data**: core curriculum = RANDOM char/word sequences
   (language-unpredictable → scoring requires looking); real text returns later
2. **Stage gating by free-running CER** — tok_acc fooled us for an entire run
3. EOS: dense short answers + upweight the stop token
4. KV cache for generate_answer (108-sample benchmark took 40 min — need 10×)
5. Train v2 on Kaggle → rerun the exact 3 fixed-seed benchmarks, measure the delta
6. Standing backlog: batched expert GEMM, ECE recalibration, FD vs copy-baseline
