# JWM — DAY 5 · Eye Physical v1: internal pass, OOD failure

## Training outcome

- Warm-started from `jwm_v4.pt`: 212 semantic tensors loaded; the vision stem and Geometric Context Memory were trained from scratch.
- `g0_exact_geometry`: 1,800 steps, `Depth AbsRel=0.0540`, `ATE=0.0169`, **passed**.
- `g1_real_rgbd_adapt`: 3,000 steps, `Depth AbsRel=0.3333`, `ATE=0.0702`, **passed** its internal gate.
- The final checkpoint contains 292 tensors with no NaN/Inf. SHA-256: `75E260BD43E46E75AA0649703B8C10EB3DB4836337039CECE1B67EA9FD5C71C2`.

## Independent benchmark

Benchmark: **JWM-Eye-Physical Independent Geometry + Blind Controls**.

| Dataset / control | Depth AbsRel ↓ | ATE ↓ | Abs. rotation ↓ | RPE trans. ↓ | RPE rotation ↓ |
|---|---:|---:|---:|---:|---:|
| Procedural test, correct image | **0.03894** | **0.01510** | 1.13669° | **0.00442** | 0.56676° |
| Procedural, wrong scene | 0.18626 | 0.01598 | 1.13695° | 0.00485 | 0.56679° |
| Procedural, constant/identity prior | 0.77238 | 0.03188 | **1.11483°** | 0.01318 | **0.55742°** |
| TUM fr3/walking_xyz OOD, correct image | 0.40807 | 0.03843 | 2.82815° | 0.00982 | 1.15841° |
| TUM OOD, black image | 0.32924 | 0.03859 | 2.83377° | 0.00996 | 1.15776° |
| TUM OOD, wrong window | 0.41607 | 0.03909 | 2.82986° | 0.01038 | 1.15769° |
| TUM OOD, constant/identity prior | **0.29444** | **0.03699** | **2.80012°** | **0.00921** | **1.15544°** |

### Interpretation

- On held-out procedural scenes, depth uses visual evidence: wrong scenes degrade AbsRel by `4.78×`, and black images by `12.13×`.
- Pose is not sufficiently image-conditioned: wrong scenes degrade ATE by only `1.058×`; rotation is effectively unchanged and loses to the identity prior.
- On dynamic real TUM OOD data, the model loses to the constant-depth/identity-pose prior on all six metrics; black images even produce better depth than correct images.
- Real adaptation improved procedural metrics over stage 0 (about 28% lower Depth AbsRel and 22% lower ATE), but did not transfer to dynamic real scenes.
- End-to-end throughput on the GTX 1650 while JARVIS was running was `20.44 FPS`, below the 30 FPS target.

## Decision

**BLOCKED — do not promote or attach this checkpoint to JARVIS.** It passed its training gates but failed the external OOD benchmark and the causal vision-dependence gate for pose.

Next-round hypotheses: procedural shortcuts, insufficiently diverse real RGB-D, near-identity motion bias, pose losses/gates that do not require beating an identity prior, and anchor-scaled depth metrics masking scale failure. Do not add more steps on the same mixture before controlled ablations.

## Artifacts

- `jwm/benchmarks/eye_physical_v1_full.json`
- `jwm/benchmarks/eye_physical_v1_full.md`
- `jwm/benchmarks/eye_physical_v1_tum_walking_xyz_controlled.json`
- `scripts/bench_eye_physical.py`

## Eye Physical v2 — corrective build (awaiting training)

- Replaced the absolute-pose shortcut with a local pairwise cost volume,
  relative SE(3) prediction, and trajectory integration from `T0 = I`.
- Split relative depth from metric scale and added valid-depth weighting,
  dynamic masking, forward/reverse cycle, and wrong-image counterfactual loss.
- The new curriculum combines exact procedural data, TartanAir, TUM RGB-D,
  and Bonn Dynamic. Every source must pass metric-scale, SO(3), motion, and
  scene-split hypotheses before admission.
- Arms A–D now share initialization and sample order; the winning pilot is
  chained into E0 → E1 → E2. Promotion requires beating fixed priors and all
  six causal controls; failures produce only a blocked checkpoint.
- Full scale is `86.77M` total / `12.91M` trainable. An 8-frame AMP smoke test
  passed on the GTX 1650 at `1606.3 MiB` peak allocation; all `107` tests pass.
- Kaggle notebook: `jwm/kaggle/jwm_eye_physical_v2_t4x2_day05.ipynb`.

## Eye Physical v2 — T4×2 pilot result

Dataset admission **passed** for every train/validation/test source: TUM, Bonn,
and TartanAir were valid and scene splits had no leakage. Arms A–D each ran for
800 optimizer steps with identical initialization and sample order.

| Arm | Depth AbsRel ↓ | Depth δ1 ↑ | Metric ATE ↓ | Gates / 6 |
|---|---:|---:|---:|---:|
| A — pairwise base | 0.3068 | 0.4871 | 0.1730 | 1 |
| **B — + SE(3) cycle** | **0.3051** | 0.4949 | 0.0748 | **1** |
| C — + dynamic mask | 0.3063 | 0.5055 | 0.0773 | 1 |
| D — + counterfactual | 0.3086 | **0.5163** | **0.0738** | 1 |

Arm B won the composite score, but no arm passed all causal gates, so full
E0→E1→E2 training was correctly blocked. On held-out real TUM+Bonn OOD data,
arm B achieved only the black-image depth gate: prior/model depth `1.126×`
(needs `1.20×`), prior/model ATE `0.898×` (needs `1.20×`), black/normal depth
`1.459×` (pass), wrong/normal depth `1.153×`, wrong/normal ATE `1.155×`, and
reverse/normal motion RPE `1.067×`.

Depth now uses visible evidence, but not the correct frame pairing strongly
enough. Pose still loses to the identity prior, and weak reverse-time
sensitivity shows that temporal direction and ego-motion are underlearned.
The pilot checkpoint is explicitly `blocked_by_ood_gate`, has 86,871,076
parameters with no NaN/Inf, and SHA-256
`423230C5CFA61F09B937F5507BFC5261A3B3664BC6DC71D7D0544C087D1AAFAE`.

**Decision: BLOCKED.** Do not attach this pilot to JARVIS or continue the full
curriculum from it. The next round must first strengthen temporal-direction,
hard-motion sampling, and wrong-window ranking objectives.

## Eye Physical v3 — build completed through gate 6

- Found the v2 camera defect: adapters emitted intrinsics but the collator
  dropped them. V3 requires per-frame K, float64 timestamps, projection
  convention, rigid flow, and dynamic-label provenance.
- Implemented CTPG-Eye: ray-conditioned pyramid, recurrent sparse tracks,
  static/dynamic separation, metric pointmaps, SE(3), robust differentiable BA,
  and bounded memory.
- All five mechanism probes and all 123 repository tests pass. The short local
  256px × 6-frame full-graph smoke allocated 0.825 GiB peak GPU memory.
- Added adaptive evidence budgeting: OOD slope chooses continue, LR reduction,
  stage advance, convergence or blocked stop; training loss cannot promote.
- T4×2 notebook: `jwm/kaggle/jwm_eye_physical_v3_t4x2_day05.ipynb`. Full-scale
  training still waits for real-source admission and Kaggle's exact 100-step profile.
