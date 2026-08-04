# JWM Eye v3.1 — Held-out Causal/OOD Benchmark

Checkpoint: `jwm_eye_v31_blocked.pt`  
SHA-256: `B084BA6630DA664F1DCD656073CC6909CFFAEE11C99D0F179B94A76752531BAF`

## Integrity and preflight

- Status: `blocked_by_causal_ood_gate`
- Parameters in checkpoint: 79,749,604
- Non-finite tensors: 0
- Dataset admission: PASS, no scene leakage
- Controlled mechanism probes: 5/5 PASS
- Actual-mixture 250-step profile: PASS, 0.835 GiB/GPU, 2.498 optimizer steps/s

## Final held-out metrics

| Metric | Result | Required | Verdict |
|---|---:|---:|---|
| Depth AbsRel | 0.4660 | lower | learned |
| Depth Delta1 | 0.4988 | higher | weak |
| Depth prior gain | 1.798x | >=1.20x | PASS |
| Metric ATE | 0.4499 m | lower | weak |
| Identity pose gain | 1.067x | >=1.20x | FAIL |
| RPE translation | 0.1567 m | lower | weak |
| RPE rotation | 2.3587 deg | lower | weak |
| BA residual reduction | 12.49% | >=15% | FAIL |
| Wrong-window depth ratio | 1.033x | >=1.25x | FAIL |
| Wrong-window pose ratio | 1.017x | >=1.25x | FAIL |
| Reverse-time RPE ratio | 1.038x | >=1.10x | FAIL |
| Wrong-intrinsics pose ratio | 1.015x | >=1.15x | FAIL |

Final causal gates: **1/7 PASS**.

## Independent local recheck

The checkpoint was loaded independently and evaluated on a disjoint procedural
scene (`seed=30000123`, 6 frames, 128px). It passed depth and pose versus simple
priors but only **2/7** causal gates. Normal metrics were Depth AbsRel `0.2057`,
ATE `0.1604 m`, RPE translation `0.0546 m`, RPE rotation `0.5689 deg`, and track
EPE `0.2807 px`. BA reduction remained `0%`; wrong-window pose ratio was
`1.013x`, reverse-time ratio `1.093x`, and wrong-intrinsics ratio `1.018x`.
Tracker confidence was only `2.94e-5`. This independently confirms confidence
collapse, inactive BA, and weak temporal/calibration sensitivity.

## Root-cause findings

1. Valid rigid flow was interpolated together with unbounded invalid flow and
   masked only afterwards. This contaminated nearby valid samples and produced
   an impossible `track_epe=483729.5` on a bounded 64×64 feature grid.
2. The raw pixel EPE dominated batches and produced the large loss/gradient
   spikes seen during training. Pose and BA therefore received noisy features.
3. Tracker confidence collapsed to `0.0012`; there is no explicit confidence
   calibration target, so BA effectively loses usable correspondence weight.
4. Stage 1 never learned better-than-identity odometry. Its best pose gain over
   the whole run was only 1.107x.
5. Reverse-time and wrong-window objectives are not directly trained. The
   wrong-intrinsics contrast is enabled only from stage 2, but the controller
   stopped the run in stage 1. This is a circular curriculum dependency.
6. Requiring shuffled windows to damage single-frame depth is conceptually
   incorrect: a good framewise depth head can remain accurate. That gate should
   be replaced by temporal-consistency or future-prediction sensitivity.
7. The controller records a best step but the trainer does not restore the
   best OOD checkpoint before LR decay, stage transition, or overfit stop.

## Eye v3.2 corrective plan

### Data contract

- Zero invalid rigid-flow vectors before resize; erode/area-average the valid
  mask and supervise only cells whose support is fully valid.
- Add admission gates for bounded valid flow, bounded downsampled targets,
  motion magnitude quantiles, and per-source nontrivial-motion coverage.
- Stratify pose batches by translation/rotation/flow magnitude instead of
  allowing identity-like windows to dominate.
- Generate validated normal/reverse/wrong-window/wrong-K paired samples from
  the same scene and reserve disjoint causal-control test scenes.

### Architecture and loss

- Use coarse-to-fine track correlation and explicit `dt`/intrinsics features in
  the pose pathway.
- Replace raw track EPE with normalized Charbonnier/Huber loss and forward/backward
  cycle consistency.
- Supervise confidence with correspondence correctness and prevent all-zero
  confidence collapse.
- Add direct temporal-order, wrong-window, and wrong-intrinsics ranking losses
  before stage 2; do not optimize incorrect outputs without a bounded margin.
- Retain framewise depth but evaluate temporal pointmap/reprojection consistency
  for wrong-window controls.

### Curriculum and controller

- Stage 0: require finite bounded tracking EPE plus calibrated confidence.
- Stage 1: motion-stratified pose/BA training with temporal counterfactuals mixed in.
- Stage 2: dynamic masks and stronger calibration controls.
- Stage 3: final disjoint causal/OOD gate only.
- Save and restore the best OOD checkpoint; an overfit stop must roll back to
  that checkpoint rather than export the final degraded weights.

Adding more steps to v3.1 is not justified. The supervision and curriculum must
be corrected first; otherwise extra steps amplify contaminated track targets.
