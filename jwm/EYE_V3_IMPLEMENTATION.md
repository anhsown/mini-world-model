# CTPG-Eye v3 — implementation and gate contract

## Outcome

Eye v3 is a camera-calibrated physical pathway:

`RGB + K + timestamps → ray-conditioned pyramid → sparse recurrent tracks →`
`static/dynamic split → metric pointmap → initial SE(3) → differentiable BA →`
`bounded world tokens`.

The Cosmos-style reasoner/generator towers remain intact. This upgrades the
physical reasoner input; it does not claim generator/world-action completion.

## Data contract

Every admitted window carries RGB, metric depth/validity, normalized C2W pose,
**per-frame intrinsics**, float64 timestamps, an explicit projection-axis
convention, dynamic-mask provenance, and exact static rigid flow. The v3
collator is mandatory and fixes the central v2 defect where dataset intrinsics
were dropped before the model received the batch.

Admission validates RGB variation, metric depth, SE(3), monotonic timestamps,
plausible K, finite rays, flow coverage, measured depth–pose–K reprojection,
counterfactual-K sensitivity, and scene-disjoint splits. Bonn pseudo dynamic
labels are derived from calibrated depth-motion violations and remain tagged.

## Architecture and mathematics

- 1/4–1/8–1/16 convolutional pyramid conditioned by physical camera rays.
- Top-K sparse points and RAFT-style recurrent local correlation.
- Confidence, dynamic probability and residual 3D scene-flow heads.
- Positive metric z-depth with heteroscedastic uncertainty.
- Stable SE(3) exponential and robust pose-only Gauss–Newton BA.
- FP32 BA under AMP, Huber weights, Levenberg damping, singular-system retry
  and pseudoinverse fallback.
- Fixed-size detached memory ring.

Losses cover metric depth NLL, geodesic rotation, motion-normalized
translation, track EPE, rigid consistency, dynamic BCE, BA monotonicity and
wrong-intrinsics BA contrast.

## Verified mechanism probes

| Probe | Result |
|---|---:|
| Wrong-K camera effect | 6.250 px EPE (normal 0.000) |
| Sparse track overfit | 1.000 → 0.003 px EPE |
| BA translation error | 0.0782 m → 0.00000036 m |
| Dynamic filtering | 0.0231 m → 0.00000013 m |
| Bounded memory | retained 4 of 19 frames |

These are mechanism proofs, not final benchmark scores.

## Adaptive step budget

Each stage has a minimum evidence budget and hard cap. Fixed held-out/OOD
metrics decide continue, LR reduction, stage advance, convergence, blocked
stop, overfit stop or unstable stop. The final gate requires depth and pose to
beat fixed priors, BA to reduce a non-trivial residual, and wrong-window,
reverse-time and wrong-intrinsics controls to degrade by declared margins.

## Verified execution

- New v3 tests: 16/16 passed.
- Full repository regression: 123/123 passed.
- Controlled probes: 5/5 passed.
- Local full-graph GTX 1650 smoke: 256px × 6 frames, batch 1, 0.825 GiB peak
  allocated and 1.22 s/step in the short two-step measurement.
- T4×2 default: per-GPU batch 1, gradient accumulation 2. The notebook runs an
  exact 100-step profile and blocks above 88% memory per rank.

## Files

- `jwm/geometry_math_v3.py` — camera/SE(3)/flow/BA mathematics.
- `jwm/geometry_v3_data.py` — calibrated contract and admission.
- `jwm/geometric_eye_v3.py` — CTPG physical eye.
- `jwm/geometry_v3_trainer.py` — metrics and causal controls.
- `jwm/adaptive_training.py` — adaptive evidence budget.
- `scripts/probe_eye_v3.py` — controlled gates.
- `scripts/profile_eye_v3_ddp.py` — exact graph pre-flight.
- `scripts/train_eye_v3_ddp.py` — adaptive DDP curriculum.
- `jwm/kaggle/jwm_eye_physical_v3_t4x2_day05.ipynb` — Kaggle runbook.

