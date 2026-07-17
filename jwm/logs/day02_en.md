# JWM — Project Log · DAY 2 (2026-07-17)

> Inkling-mini: ported Thinking Machines' MoE topology to micro scale as the new
> reasoner → **jwm_v4.pt, record numbers on every understanding + grounding metric**.
> *(Series index: [README.md](README.md))*

---

## Opening — Day 1's inheritance

Day 1 closed with `jwm_v3.pt` (31M dense, QA 58.8%, mIoU 0.285) and an open
question: how to get 68M-class capacity when the step budget only affords 28M?

## 1. Reading Inkling (`thinkingmachines/Inkling`)

From the real config.json: 975B/41B active, 66 layers, **DeepSeek-style
fine-grained MoE** (256 experts with hidden = d/2, top-6 + 2 shared, sigmoid
gating, dense early layer), 55/66 local-attention layers (window 512), 8 MTP
layers, vision = hierarchical MLP with 40px patches. No public dataset ("None
public yet"). Cannot run locally: ~550GB at NVFP4 — quantization cannot bridge a
137× gap (worked through the math for the user).

## 2. Inkling-mini — answering Day 1's question

Grafted **exactly the valuable part** (the MoE topology) into JWM's reasoner
tower, keeping everything proven (one-variable-at-a-time):
- 32 experts hidden d/2=192, top-4 + 1 shared, sigmoid→top-k→normalize, dense layer 0
- Switch aux loss α=0.01; generator tower stays dense
- **73.94M total / 30.6M active** — more capacity than the failed 68M dense run
  at the step cost of the proven 28M (87.5% of dense speed)
- 29/29 tests (added: sparsity, aux, generator gradients never leak into the MoE)

## 3. Controlled A/B (one variable; same data/LR/batch/seeds)

| | dense (v3 baseline) | **MoE** |
|---|---|---|
| r1 (3000 steps) | 54.5% | **58.0%** |
| r2 (800 steps) | 57.2% | **60.8% → WIN** |

Router health perfect: entropy 3.28–3.40 of 3.47 max, **zero dead experts** in all 7 layers.

## 4. g1→g5 on the MoE reasoner → jwm_v4.pt (129.6 min)

Bug caught before launch: `init_generator_from_reasoner` copying an MoE FFN onto
a dense generator FFN would shape-crash → made MoE-aware (attention+norms only
when FFN types differ).

**Graduation table (test set):**

| Metric | v1 | v3 | **v4-MoE** |
|---|---|---|---|
| QA exact-match | 56.4% | 58.8% | **65.6%** |
| — what_held | 57% | 57% | **68.8%** |
| — where | 55% | 68% | **72.3%** |
| Grounding IoU@0.5 (4-step) | 0.184 | 0.268 | **0.360** |
| Grounding mIoU | 0.201 | 0.285 | **0.355** |
| FD beats-copy | 27% | 31% | **45.8%** |
| T2I neg-probe | — | 0.75 | 0.708 |

**Trials (74):** mean IoU 0.247 → **0.373 (+51%)**; IoU@0.5 0.275 → **0.375**.

## 5. Honest caveats (Day 3 agenda)

1. **Batch-1 inference latency**: 8.3s/trial (v3: 1.6s) — two culprits:
   (a) the 32-iteration expert-major loop is unoptimized for tiny batches,
   (b) `generate_answer` rebuilds the whole sequence per byte — **no KV cache**
   (fixing this helps dense AND MoE, est. ~10×).
2. **4-step ECE 0.084** (v3: 0.052) — calibration slightly degraded; wrong_abstain
   up (7 vs 2), first sub-100% precision-when-asserting (8/9). Needs per-generation
   Platt + threshold review.
3. FD still below the copy baseline (20.77 vs 21.39) despite the big beats-copy jump.

## 6. Also today

- **GitHub push**: https://github.com/celesnity/mini-world-model (44 files,
  allowlist staging — notebooks/data/checkpoints excluded for privacy + size).
- Drafted the user's English daily-summary message.

## Day 2 statistics

Inkling read + mini design + MoE implementation + 218-min A/B + 130-min pipeline
+ trials — all in one day, on one GTX 1650.
