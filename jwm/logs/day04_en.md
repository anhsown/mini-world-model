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

