# V-JEPA 2 × HATREC: research verdict

- Dataset: 546 videos, 78 cycles, 7 balanced tasks. Split: 54 train / 12 validation / 12 test cycles.
- Primary result: 84/84 correct on held-out cycles; Macro-F1 = 100%. Wilson 95% accuracy interval = 95.63%–100.00%.
- 14-visible-frame ablation also reaches 100%; the additional frames did not improve this benchmark.
- Negative controls behave near chance: majority accuracy 14.29%; shuffled-label accuracy 9.52%.
- Procedural leakage checks pass: filenames neutralized and exact duplicates = 0.
- However, mean nearest train cosine is 0.9922 and p95 is 0.9959; the environment/tasks remain visually similar.
- Therefore this result validates linear separability of HATREC task representations, not language reasoning and not yet temporal motion understanding.

## Required next tests
1. Single middle frame repeated to 64 frames.
2. Temporal frame-order shuffle/reversal.
3. Object/tool masking or background intervention.
4. Hold out operator/workstation/product if metadata becomes available.
5. Save per-sample probabilities for reliability diagrams and abstention analysis.
