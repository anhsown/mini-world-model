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

## 5. The commit run in flight (as of writing)

- Stage 0: tok_acc 0→**0.62** over 1500 steps · Stage 1: 0.51→**0.71** over 4000 ·
  Stage 2 (50% real docs): at **0.75-0.80** by step 4250/6000, still climbing, 0.7 it/s
- `moe_aux` flat at ~0.090 = balanced router, no dead experts; the `float(la)`
  warning was audited — metric-only, aux gradients still flow (line 333)
- Results will be read via CER, not exact-match (Day-1 law:
  exact ≈ tok_acc^answer_byte_length — 50-200-byte answers make low exact physics)

## 6. Side incident — colleagues' sessions appearing in the app

Shared company Claude account; colleagues enabled remote control → their sessions
appear on every machine on that account. Verified this project's session is
**purely local, no remote control** (no hover label, absent from claude.ai/code).
Recommendation sent: separate accounts per person.

## Day 4 agenda

1. Download `jwm_read_v1.pt` + `metrics_read_v1.json` → read the per-level CER table
2. Wire the Reader into WorldBrain (checkpoint router: v4 for scenes, read_v1 for text?)
3. Real reading trials through the webcam (signs, book pages held up to camera)
4. Standing backlog: KV cache for generate_answer, batched expert GEMM, ECE
   recalibration, FD vs copy-baseline
