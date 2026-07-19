# JWM-Read v3 — architecture and Kaggle T4×2 recipe

## Why v2 failed

V2 improved synthetic teacher-forced vision gain (`+0.136`) but remained blind
on real documents (`-0.0002`). It compressed each raw 2×2 patch group before
spatial reasoning, optimized only autoregressive QA cross-entropy, and used a
byte tokenizer whose long Vietnamese targets weakened exact visual supervision.
Its public benchmark still had 0% exact match on real VietDocVQA and MTVQA-VI.

## V3 model

- **Input:** aspect-preserving portrait canvas 1024×768.
- **Vision:** patch-16 convolution → two local window-attention blocks on the
  64×48 grid → learned 2×2 merge → 32×24 = 768 visual tokens. Glyph relations
  are therefore resolved before compression.
- **Reasoner:** d=512, 10 MoT layers, 16 heads; Inkling-mini sparse FFN with 16
  experts, top-2 routed + one shared expert. The generator tower remains present
  for JWM compatibility but is frozen during Reader training.
- **Tokenizer:** lossless hybrid Vietnamese grapheme vocabulary. Common
  precomposed Vietnamese characters use one token; unseen Unicode falls back to
  UTF-8 bytes. Special-token IDs and legacy byte checkpoints remain unchanged.
- **OCR head:** CTC over the 768 spatial tokens, with a dedicated blank token.
- **Region head:** four learned queries predict `(x1,y1,x2,y2)` independently as
  coordinate tokens in 1001 bins, following the useful localization principle
  from LocateAnything without cloning its 3B Qwen/MoonViT stack.

The stage objective is

`L = L_QA + λctc L_CTC + λbox Σ CE(coord_i) + λvis max(0,m+Lright-Lshuffle) + LMoE`.

Only synthetic samples receive transcript CTC and exact box labels. Real
document-QA samples receive QA + shuffled-image contrast; a document answer is
never mislabeled as the full-page transcript.

## Data validity contract

Before optimization, `validate_read_v3_data` must pass all hypotheses:

1. random anti-shortcut labels are unique;
2. every synthetic box is valid and inside the image;
3. no question/answer label is truncated;
4. rendered text is visibly distinguishable from paper;
5. train/validation/test are split by document page with zero leakage;
6. real pages are actually loadable.

The validator also reports synthetic-vs-real image statistics and their mean
absolute domain gap. This is a diagnostic, not a fabricated universal threshold.

## Metric-gated curriculum

| Stage | Mix | Primary purpose | Promotion evidence |
|---|---|---|---|
| s0 glyph bootstrap | 100% random L1–L2 | force pixel→glyph learning | CTC-CER + vision gap |
| s1 layout OCR | 100% random/natural L1–L4 | multiline OCR + region geometry | CTC-CER + box IoU + vision gap |
| s2 real adaptation | 55% synthetic / 45% document | bridge rendered and real pages | synthetic free CER + shuffled-image win rate |
| s3 reasoning/OOD | 35% synthetic / 65% document | document QA and complex scenes | final held-out battery |

A failed gate extends the same stage. If it still fails after its extension
budget, training stops safely and exports `jwm_read_v3_blocked.pt`; it does not
silently contaminate the next stage as v2 did.

## Infrastructure

- Launch: `torchrun --standalone --nproc_per_node=2`.
- Kaggle default: 3 samples/T4 × 2 GPUs × accumulation 2 = global batch 12.
- Mixed precision uses initial scale 1024; the usual 65536 overflowed the first
  CTC backward pass in the gradient audit.
- Atomic resume checkpoint every 250 optimizer steps.
- Rank-specific random streams, DDP gradient synchronization, gradient clipping,
  cosine decay and JSON histories.
- Base schedule: 9,500 optimizer steps; expected T4×2 runtime 7–9 hours, up to
  roughly 11 hours if metric extensions are used.

Primary files:

- `jwm/vision_v3.py`
- `jwm/read_v3_data.py`
- `jwm/read_v3_trainer.py`
- `scripts/train_read_v3_ddp.py`
- `jwm/kaggle/jwm_read_t4x2_v3.ipynb`

