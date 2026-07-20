# JWM Eye v3 — Research and Architecture Decision

Date: 2026-07-20  
Status: design recommendation after the blocked Eye Physical v2 pilot

## 1. Decision

Eye v3 should be a **calibrated track–point geometry system**, not a larger
version of the v2 pairwise pose head. It remains an adapter underneath the
Cosmos-style Reasoner/Generator towers; it does not replace the two-branch
world-model architecture.

The recommended design combines four ideas:

1. camera-aware depth/ray prediction, inspired by
   [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3),
   [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth), and
   [MapAnything](https://github.com/facebookresearch/map-anything);
2. sparse multi-frame correspondence and recurrent refinement, inspired by
   [DPVO](https://github.com/princeton-vl/DPVO) and
   [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT);
3. differentiable sliding-window bundle adjustment, inspired by DPVO and
   [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM);
4. explicit decomposition of camera motion and independent object motion,
   inspired by [Dynamo-Depth](https://dynamo-depth.github.io/) and
   [SpatialTrackerV2](https://github.com/henry123-boy/SpaTrackerV2).

Large models are offline teachers and benchmarks. The deployed JWM module is
our own smaller student.

## 2. Why Eye v2 failed

The blocked pilot is not evidence that learned geometry is impossible. It is
evidence that the current objective admits easier non-geometric solutions.

| Implementation finding | Consequence observed in the pilot |
|---|---|
| `stack_rows()` drops camera intrinsics although the datasets provide them | the same optical displacement can correspond to different metric motion, so cross-camera metric pose is underdetermined |
| local correlation is computed only on a 16×16 token grid with radius 2 | large motion, thin geometry and sub-pixel motion are lost before pose prediction |
| pair motion is globally pooled into one pose vector | no persistent correspondences and no geometric evidence explaining the pose |
| pairwise poses are merely multiplied | drift accumulates; there is no bundle adjustment or pose-graph correction |
| translation uses default Smooth-L1 on small metric values | the identity/near-zero translation solution receives weak gradients |
| forward/reverse cycle is the main temporal regularizer | identity motion satisfies the cycle exactly; cycle consistency does not prove image use |
| dynamic masks only reweight global pooling | the model does not represent rigid flow and independent 3D object flow separately |
| counterfactual negatives are whole wrong windows with a weak energy hinge | no direct pressure to solve ordered correspondences, focal changes, freeze/repeat or time reversal |
| no flow, point tracks, epipolar, reprojection or point-map losses | depth and pose can improve independently without agreeing in 3D |

This explains the v2 pattern: reasonable depth, near-identity pose, and weak
degradation under wrong-window or reverse-time controls.

## 3. Research map

### 3.1 Geometry foundation models

- **Depth Anything 3 (DA3):** one transformer predicts a unified depth–ray
  representation and supports depth, intrinsics and pose. The official repo
  releases Apache-2.0 80M and 120M variants, making DA3-Small/Base suitable as
  teachers and ablation baselines. We should adopt the representation, not add
  all DA3 tasks to the JWM student.
- **MapAnything:** useful as the reference data contract. It accepts images,
  intrinsics or ray directions, depth and poses in multiple combinations and
  emits point maps, rays, depth, camera translation/quaternion and confidence.
  It also supplies a modular benchmark harness and an Apache-2.0 model option.
- **VGGT:** jointly predicts intrinsics, extrinsics, point maps, depth and 3D
  tracks, but the main checkpoint is 1B. It is a strong teacher/reference, not
  a 30-FPS GTX-1650 deployment target.
- **CUT3R:** demonstrates a recurrent persistent scene state and ray-map query.
  We should retain the bounded-state idea, but use explicit geometric tracks
  rather than asking one latent state to solve all geometry.
- **NVIDIA VGG-T³:** compresses long-view KV memory into a fixed-size MLP with
  test-time training. It is interesting for a later lifelong scene memory, but
  the official project reports roughly 10-FPS visual localization, so it is not
  the first Eye v3 fast-path component.

### 3.2 Visual odometry and correspondence

- **DPVO:** sparse patches, recurrent correspondence updates and differentiable
  bundle adjustment. This is the most transferable algorithmic skeleton for a
  small JWM pose pathway.
- **DROID-SLAM:** dense correlation plus recurrent updates and dense bundle
  adjustment. Its official implementation requires about 11 GB for inference
  and much more for training, so it is a teacher/reference rather than a direct
  dependency.
- **SEA-RAFT:** directly initializes flow, iteratively refines it, predicts
  uncertainty through a mixture-of-Laplace objective, and uses rigid-motion
  pretraining. This is a better correspondence loss/teacher than v2's single
  coarse local cost volume.
- **CoTracker3:** online/offline long-term point tracking and visibility. Its
  real-video pseudo-label curriculum is especially valuable: synthetic labels
  bootstrap exact tracking, then teacher agreement admits unlabeled real video.

### 3.3 Dynamic geometry

- **Dynamo-Depth:** jointly predicts depth, rigid flow, independent 3D flow and
  motion segmentation. Its decomposition directly addresses the ambiguity
  where a moving object can be explained as incorrect depth.
- **SpatialTrackerV2:** decomposes scene geometry, camera ego-motion and
  point-wise motion, then iteratively refines 2D/3D tracks, dynamic probability
  and camera poses. This is the right output contract for the long-term JWM eye.

### 3.4 NVIDIA paths worth adopting

- **ViPE:** estimates camera intrinsics, camera motion and near-metric depth
  from unconstrained pinhole, wide-angle and 360° video. Use it offline to
  create pseudo-labels and reject bad Internet/local clips; do not require it
  in Jarvis runtime.
- **Fast-FoundationStereo:** distillation, blockwise NAS and structured pruning
  produce 640×480 TensorRT runtimes around 14–23 ms with roughly 646–653 MB
  peak memory in the official measurements. If a stereo/RealSense camera is
  available, this is the most reliable production metric-depth route.
- **E3D-Bench:** evaluates sparse/video depth, reconstruction, multi-view pose,
  tracks and OOD behavior. Its main lesson for JWM is to decompose the hard
  problem when exact 3D supervision is limited, then join modules only after
  each passes its own causal gate.

## 4. Recommended architecture: CTPG-Eye

`CTPG` means **Calibrated Track–Point Geometry**.

```text
RGB stream + K/Camera ID
        │
        ├── multi-scale visual stem (1/4, 1/8, semantic 1/16)
        │       ├── depth/ray/normal/confidence head
        │       └── foveal features for Reasoner/OCR
        │
        ├── adaptive patch selector
        │       └── recurrent track/flow/visibility updater
        │                    │
        │                    ├── static weighted tracks ──> differentiable BA
        │                    └── residual 3D flow ────────> dynamic object state
        │
        └── bounded scene memory (4–8 keyframes + persistent object/point slots)
                             │
                             ├── camera SE(3), metric point map, uncertainty
                             ├── dynamic tracks and object motion
                             └── geometry tokens to Reasoner and Generator
```

### 4.1 Camera/ray representation

For pixel `p = [u,v,1]^T` and intrinsic matrix `K_t`, the camera ray is

`r_t(p) = normalize(K_t^-1 p)`.

The model predicts positive distance `d_t(p)` and obtains the camera-frame
point map

`P_t(p) = d_t(p) r_t(p)`.

Known intrinsics are always supplied. Unknown intrinsics are predicted by a
small camera head and then converted to ray maps. Random focal length,
principal point, crop, resize and distortion must update `K` analytically; an
augmentation is invalid if image and camera metadata diverge.

### 4.2 Reprojection and rigid flow

For relative camera transform `T_t->j`, the rigidly reprojected pixel is

`p_hat_j = pi(K_j T_t->j [P_t(p), 1]^T)`.

Observed flow is decomposed as

`f_obs = f_rigid(d, K, T) + q_dyn f_ind`,

where `q_dyn` is dynamic probability and `f_ind` is independently moving 3D
flow. Static tracks estimate camera motion; dynamic residuals update object
slots. A single mask that merely suppresses tokens is insufficient.

### 4.3 Differentiable sliding-window BA

For keyframes `i,j`, tracked patch `k`, visibility `v_ijk`, confidence
`w_ijk`, inverse depth `rho_ik`, and robust penalty `rho_h`, optimize

`min_{T,rho} sum_ijk v_ijk w_ijk rho_h(||pi(T_j T_i^-1 P_ik)-p_jk||^2)`.

Use 2–4 learned Gauss–Newton update iterations during training and deployment,
with a 4–8 keyframe window. The network predicts residual updates and weights;
the optimizer enforces the geometry.

### 4.4 Bounded temporal memory

- fast frame state: previous features, flow, tracks and uncertainty;
- keyframe state: 4–8 frames selected by parallax/uncertainty, not fixed time;
- persistent point/object slots: ID, 3D position, velocity, visibility,
  dynamic probability, confidence and timestamp;
- eviction: remove redundant/low-confidence keyframes, keep geometrically
  informative views.

Test-time weight updates are postponed. v3 updates state, not model weights,
so one corrupted clip cannot silently rewrite the eye.

### 4.5 Dual-rate 30-FPS runtime

- every frame at 30 FPS: low-resolution visual stem, initial flow, track update,
  event/change/looming and uncertainty;
- geometry keyframe at 7.5–10 Hz: dense depth/rays, BA and memory update;
- forced keyframe: high uncertainty, new scene, large parallax or track loss;
- foveal Reasoner/OCR at 2–5 Hz or on request;
- intermediate frames receive propagated depth/pose with uncertainty growth.

This preserves 30-FPS reactivity without pretending that the full 3D model can
run at 30 FPS on a 4-GB GTX 1650.

## 5. Objectives

The training loss should be staged, then combined with per-task valid-label
normalization:

`L = λray Lray + λd Ldepth + λn Lnormal + λf Lflow + λtrk Ltrack`

`  + λepi Lepipolar + λp Lpose + λrep Lreprojection + λba LBA`

`  + λdyn Ldynamic + λcf Lcounterfactual + λcal Lconfidence`.

Recommended terms:

- `Ldepth`: log-L1 + SILog + edge/gradient + uncertainty NLL;
- `Lflow`: mixture-of-Laplace NLL plus forward/backward visibility;
- `Ltrack`: robust 2D endpoint, 3D point error and visibility BCE;
- `Lepipolar`: Sampson distance on static correspondences;
- `Lpose`: SO(3) geodesic plus translation normalized by GT motion bucket;
- `Lreprojection`: photometric and teacher-feature reprojection, masked for
  occlusion and dynamics;
- `LBA`: residual after the final optimizer update, supervised trajectory loss,
  and monotonic residual-reduction loss;
- `Ldynamic`: focal/Dice mask plus independent 3D-flow error;
- `Lconfidence`: calibrated NLL/Brier loss for depth, tracks and pose risk.

The translation term must not use default Smooth-L1 on meter-scale values:

`L_trans = SmoothL1((t_pred-t_gt)/max(||t_gt||, τ), beta=0.05)`.

Training must balance static, small, medium and large translation/rotation
buckets. Otherwise static clips dominate and identity remains optimal.

### Direct causal negatives

For the same anchor frame, construct:

- correct ordered next frame;
- exact repeat/frozen frame;
- reversed pair/window;
- shuffled frame from the same scene;
- wrong scene;
- correct image with wrong intrinsics;
- incorrect timestamp/FPS.

Apply contrastive ranking directly to correspondence/reprojection energy:

`L_cf = max(0, m + E_correct - E_negative)`.

The negative must be hard but valid. Controls are evaluated separately and are
never mixed into the ordinary validation average.

## 6. Dataset plan

| Dataset/source | Useful labels | Eye v3 role | License/constraint | Priority |
|---|---|---|---|---:|
| TartanAir | RGB, depth, pose, flow | exact pose/BA bootstrap and motion buckets | verify current terms before redistribution | P0 |
| TUM RGB-D | real RGB-D, intrinsics, mocap pose | real indoor pose/depth benchmark | benchmark use | P0 |
| Bonn Dynamic RGB-D | dynamic people, depth, pose | held-out dynamic ego-motion | benchmark use | P0 |
| Kubric MOVi | metric depth, forward/back flow, masks, camera/object physics | exact unit tests and dynamic decomposition | verify asset-level terms; Kubric code is open | P0 |
| Spring + RobustSpring | stereo, disparity, forward/back flow, scene flow, 20 corruptions | correspondence and robustness | Spring assets are CC BY 4.0 | P0 |
| Dynamic Replica | stereo dynamic humans/animals, depth, flow, tracks, cameras | dynamic geometry validation | verify download terms | P0 |
| Hypersim | 77.4K images, metric geometry, complete cameras | indoor ray/depth pretraining | CC BY-SA 3.0 | P1 |
| PointOdyssey | 2D/3D tracks, visibility, depth, masks, cameras | long-track curriculum | CC BY-NC-SA 4.0; research only | P1 |
| ARKitScenes subset | real mobile RGB-D, per-frame intrinsics/pose, laser geometry | real camera diversity | Apple dataset license; 623 GB full, use shards | P1 |
| ScanNet++ subset | 1000+ scenes, laser/DSLR/iPhone RGB-D, poses/intrinsics | real OOD indoor benchmark | registration/custom terms | P1 |
| ViPE DynPose-100K subset | pseudo intrinsics, camera motion, near-metric depth on dynamic web video | real pseudo-label adaptation | teacher/data terms must be tracked | P1 |
| HOI4D subset | egocentric RGB-D, motion/panoptic, hand/object pose, cameras | later manipulation/object-motion stage | CC BY-NC 4.0 | P2 |
| local JARVIS camera | calibrated household clips, optional AprilTag/RealSense GT | final domain SFT and latency tests | private; never export to Git | P0 after consent |

Do not download everything. Start with compact scene-disjoint shards that
cover the required motion/camera/dynamic buckets, prove benefit, then scale.

### Dataset admission hypotheses

Every source must pass before entering training:

1. camera convention, metric units and handedness are explicitly converted;
2. `K` matches every resize/crop/flip and reconstructs known rays;
3. GT flow agrees with depth+pose rigid reprojection on static pixels;
4. forward/reverse transforms multiply to identity within tolerance;
5. timestamps increase and FPS/frame gaps match the training sample;
6. train/val/test are scene- and source-video-disjoint;
7. motion buckets contain enough non-identity samples;
8. dynamic masks/visibility are non-empty where claimed;
9. depth coverage/range and invalid-value semantics are documented;
10. pseudo-labels pass teacher agreement, confidence and reprojection gates;
11. synthetic/real statistics and metrics are reported separately;
12. licenses permit the intended use and no private media enters Git.

## 7. Training curriculum on Kaggle T4×2

| Stage | Trainable components | Main data | Required gate |
|---|---|---|---|
| G0 camera/ray unit | camera head, depth-ray head | MOVi, Hypersim, Tartan | ray reprojection and intrinsics tests pass |
| G1 correspondence | multi-scale stem, flow/track updater | Spring, MOVi, PointOdyssey subset | flow EPE, track AJ/visibility beat coarse v2 |
| G2 metric geometry | depth/ray/normal/confidence | Tartan, Hypersim, ARKit subset | depth beats fixed prior per camera/depth bucket |
| G3 odometry | adaptive patches, SE(3), differentiable BA | Tartan, TUM train, ARKit subset | pose beats identity and BA improves pre-BA residual |
| G4 dynamics | dynamic mask, independent 3D flow, object slots | MOVi, Dynamic Replica, Bonn train subset | dynamic mIoU and static/dynamic pose gates |
| G5 real adaptation | small adapters/student heads | confidence-filtered ViPE + local calibrated clips | teacher agreement and real OOD improve without exact-data regression |
| G6 JWM integration | geometry-token projector; Reasoner adapters only | balanced replay | Reasoner retention, latency and causal-use gates |

Freeze already-proven modules for the first half of each new stage, then open
them with a 5–10× lower learning rate. Keep at least 20% exact synthetic/GT
replay during pseudo-label adaptation.

## 8. Benchmark and promotion gates

Report every metric by dataset, camera/focal bucket, depth range, motion bucket,
static/dynamic region and corruption type.

- depth: AbsRel, RMSE, δ1, boundary F1, uncertainty ECE;
- pose: ATE, RPE translation, RPE rotation and drift per meter/second;
- flow: EPE, 1-pixel/3-pixel outliers, occlusion split;
- tracks: AJ, δavg, survival/visibility and 3D trajectory error;
- dynamics: mIoU/F1 and independent 3D-flow EPE;
- geometry: point-map accuracy/completeness and reprojection residual;
- deployment: capture FPS, fast-path FPS, keyframe Hz, p50/p95 latency,
  VRAM/RAM and dropped frames.

Promotion requires all of the following:

1. depth beats a train-only fixed-depth prior by at least 20%;
2. pose beats identity and constant-velocity priors by at least 20%;
3. final BA residual is lower than pre-BA residual on at least 90% of windows;
4. black/wrong/reversed/frozen/wrong-K controls degrade the matching metric;
5. model beats v2 on TUM+Bonn and does not regress on exact Tartan/MOVi;
6. no dataset has a catastrophic hidden aggregate (motion/camera buckets pass);
7. 30-FPS fast path and bounded memory pass on the target machine;
8. confidence supports safe abstention under blur, low texture and track loss.

## 9. What to reuse and what not to clone

| Project | Reuse | Do not copy directly |
|---|---|---|
| DA3 / MapAnything | ray/point-map contract, teacher predictions, benchmark wrappers | full large model as Jarvis runtime |
| DPVO | patch graph, recurrent update, differentiable BA formulation | CUDA stack/checkpoint dependency in core JWM |
| SEA-RAFT | initial-flow + refinement design, uncertainty loss, pseudo labels | full dense model at every 30-FPS frame |
| CoTracker3 | online visibility/tracking curriculum | non-commercial checkpoint in deployable product without license review |
| ViPE | offline annotation and quality filtering | runtime dependency or unfiltered pseudo labels |
| CUT3R / VGG-T³ | bounded persistent memory research | test-time parameter updates in v3 |
| Fast-FoundationStereo | optional calibrated stereo production backend | assume monocular webcam can provide the same metric certainty |

## 10. Immediate experiments before another long run

1. **K-only repair probe:** pass real intrinsics/ray maps through the current
   pipeline and normalize all camera conventions. If pose does not improve,
   do not scale v2.
2. **Track-head proof:** train only multi-scale flow/tracks on a small
   MOVi+Spring shard; require controls and uncertainty to work.
3. **BA proof:** feed GT/noisy tracks and depth to the new BA layer; verify it
   monotonically lowers reprojection and pose error before neural integration.
4. **Dynamic proof:** on MOVi/Dynamic Replica, show rigid-only degradation and
   recovery from the independent-flow branch.
5. **100-step T4×2 profile:** determine batch/window/resolution, wall time and
   memory before writing the full notebook schedule.

Only after these five experiments pass should we assemble and train the full
Eye v3 student.

## 11. Adaptive step-budget algorithm

Eye v3 does not choose a fixed number of steps in advance. Every stage defines
a **minimum evidence budget**, an absolute **hard cap**, and an evaluation
interval. At each interval, rank zero runs the same frozen ID, OOD and causal
control suites and feeds their metrics to `AdaptiveTrainingBudget` in
`jwm/adaptive_training.py`.

Raw metrics are normalized from their baseline `b_i` to their target `g_i`:

`progress_i = (metric_i-b_i)/(g_i-b_i)` for higher-is-better metrics,

with the signs reversed for lower-is-better metrics. The weighted OOD score is

`S_k = sum_i w_i progress_i / sum_i w_i`.

The controller uses a trailing median and the recent slope of `S_k`; training
loss is intentionally excluded from this score. It makes the following
sequential decision:

| Condition | Action |
|---|---|
| before `min_steps` and numerically healthy | continue to collect evidence |
| material positive OOD slope | continue |
| gates pass and OOD gain plateaus | advance stage, or stop final stage |
| gates fail and progress plateaus | reduce LR once/twice and re-evaluate |
| gates still fail after allowed LR decays | stop as architecture/data blocked |
| train loss improves while OOD score persistently regresses | stop overfit and restore best checkpoint |
| NaN/Inf, exploding or vanishing gradient | stop unstable |
| `max_steps` reached | hard stop; pass only if every mandatory gate passes |

The controller also estimates the steps required to reach normalized score
one from the recent learning-curve slope. This projection is advisory only:
causal gates and the hard cap always override it. The best checkpoint is the
one with the highest held-out OOD score, never necessarily the final step.

Recommended initial settings for a 100-step profile are then scaled in units
of optimizer steps, not examples:

- `eval_every`: enough to process 0.5–1.0 effective epoch or at least 100 steps;
- `min_steps`: at least five evaluations, so slope estimation is meaningful;
- `plateau_patience`: four evaluations;
- LR decay: ×0.3, at most two times;
- `max_steps`: 2–3 times the first projected target step, bounded by available
  compute and fixed before the long run.

The full controller state is checkpointed so a Kaggle restart preserves the
best step, LR-decay count and patience history. This prevents resume from
silently granting the model a fresh stopping budget.

## 12. Implementation status (Day 5)

Steps 1–6 are implemented in `EYE_V3_IMPLEMENTATION.md`: calibrated data
contract, CTPG-Eye architecture, mathematical tests, controlled mechanism
probes, full-graph local profile, adaptive DDP trainer and a gated Kaggle T4×2
notebook. Full-scale training remains intentionally unexecuted until Kaggle's
100-step exact-graph profile and real-source admission both pass.
