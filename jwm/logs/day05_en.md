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
