# JWM Project Log · DAY 4 (2026-07-19)

## V2 verdict

V2 learned to use synthetic images (blind-control vision gain **+0.136**) but
remained blind on real documents (**−0.0002**). VietDocVQA median CER was 0.99
and MTVQA-VI median CER was 1.00, both at 0% exact match.

## JWM-Read v3

- 1024×768 input; patch-16 local attention on 64×48 before learned 2×2 merge.
- 768 visual tokens, d512/L10 reasoner, 16-expert top-2 MoE.
- 119.06M total / 82.77M trainable parameters with the generator frozen.
- Lossless Vietnamese grapheme tokenizer with byte fallback.
- Direct CTC OCR, four parallel 0–1000 coordinate heads, noisy teacher forcing
  and correct-vs-shuffled image contrast.
- Synthetic data receives exact QA+CTC+box supervision; real documents receive
  QA/visual supervision only.

The real-data probe passed all six pre-training hypotheses: unique anti-shortcut
labels, valid boxes, no truncation, visible text, zero page leakage and loadable
real pages. Measured synthetic–real statistics gap: 0.05003.

The Kaggle trainer now uses T4×2 DDP, global batch 12, FP16, atomic resume and
four metric-gated stages. A failed gate cannot silently advance. Gradient audit
also caught CTC overflow at AMP scale 65536; scale 1024 produced zero NaN/Inf
gradients. End-to-end smoke passed and 51/51 Reader/math tests pass.

Notebook: `jwm/kaggle/jwm_read_t4x2_v3.ipynb`. Expected base runtime: 7–9 hours;
worst case with stage extensions: about 11 hours.

## V3 training and benchmark verdict

Kaggle stopped safely at `s0_glyph_bootstrap`, step 3,200, with
`blocked_by_metric_gate`. The checkpoint is structurally valid and contains no
NaN/Inf tensors, but CTC-CER was **1.000** against a promotion threshold of
0.72.

The dimension-correct **JWM-EyeRead-v3** benchmark evaluated 186 samples at
1024×768. Exact match and containment were **0% on every tier**. AR CER was
0.916 on L2 lines, 0.977 on L4 paragraphs, 1.037 on 40 real VietDocVQA pages,
and 1.195 on 50 OOD MTVQA-VI samples. Direct CTC CER was 1.000 across every
synthetic tier; text-box IoU peaked at only 0.275 on large L4 paragraphs.

The blind-image control nevertheless measured a positive shuffled-minus-correct
loss gap of **+0.554**, with correct images winning **63.2%** of batches. Thus
the checkpoint encodes weak visual evidence, but its full-page 2D CTC head
collapses to blank (99.73% of positions) and cannot decode that evidence. Blank
logit calibration improved CER only to about 0.895 with 0% exact match.

Decision: preserve the visual stem and reasoner as a warm start, reinitialize
OCR/localization heads, replace full-page CTC with ROI/line-wise 1D decoding,
and only then add geometric streaming memory for Eye Physical.
