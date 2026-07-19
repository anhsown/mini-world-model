# JWM Eye Physical v2

Eye Physical v2 is the corrective visual-geometry pathway for JARVIS. It does
not replace the Cosmos-style two-branch design: the semantic Reasoner and the
diffusion Generator remain frozen while a new streaming eye learns metric
depth, camera motion, and persistent geometric context.

## Why v1 was rejected

Eye v1 passed its procedural and internal TUM gates, but on the independent
TUM `fr3/walking_xyz` control it lost to a constant-depth/identity-pose prior
on all six geometry metrics. A black image even improved depth. This exposed
three shortcuts: absolute per-frame pose regression, anchor-normalized depth,
and insufficient dynamic/real-world diversity.

## Architecture

For visual tokens `F_t(p)` at frame `t`, v2 constructs a local pairwise cost
volume rather than regressing pose from one frame:

`C_t(p, delta) = <normalize(F_t(p)), normalize(F_{t-1}(p+delta))> / sqrt(d)`

The pair-motion token is

`M_t = Phi([F_t, F_{t-1}, F_t-F_{t-1}, C_t])`.

A learned dynamic probability `q_t(p)` excludes independently moving regions
from the global ego-motion statistic:

`m_t = sum_p (1-q_t(p)) M_t(p) / sum_p (1-q_t(p))`.

The relative-pose head predicts an identity-centred 6D rotation and metric
translation. It cannot access an absolute frame index or target trajectory:

`Delta_T_t = SE3(Psi_pose(m_t))`, `T_0 = I`,
`T_t = T_{t-1} Delta_T_t`.

Forward and reverse predictions are made by the same pair encoder. Their
cycle must be identity: `Delta_T_forward Delta_T_reverse ~= I`.

Depth has separate shape and scale paths:

`d_relative = softplus(Psi_depth(H_t))`,
`d_metric = d_relative exp(Psi_scale(mean_p H_t(p)))`.

Missing sensor pixels are removed before pooling. Partially valid patches are
weighted by valid-pixel fraction instead of being discarded or averaged with
zero depth.

## Objective

`L = 0.5 L_relative_depth + 1.0 L_metric_depth`

`  + 0.25 L_absolute_pose + 1.0 L_relative_pose`

`  + lambda_cycle L_SE3_cycle + lambda_dynamic L_dynamic`

`  + lambda_cf max(0, margin + E(correct)-E(wrong))`.

The counterfactual term explicitly forces the correct image sequence to have
lower joint depth/pose energy than a wrong sequence. Ground-truth poses are
always normalized to the first camera, but metric translations and metric
depth are never target-normalized.

## Controlled ablation

All arms share the same seed and warm-start:

| Arm | Pairwise metric base | SE(3) cycle | Dynamic mask | Counterfactual |
|---|---:|---:|---:|---:|
| A | yes | no | no | no |
| B | yes | yes | no | no |
| C | yes | yes | yes | no |
| D | yes | yes | yes | yes |

Full training is admitted only when at least one pilot arm passes all causal
OOD gates. More steps are not used to hide a failed hypothesis.

## Dataset curriculum

- Procedural exact RGB-D/pose/dynamic masks: 10-15%, used as a geometry unit
  test and replay source, never as the main realism source.
- TartanAir: diverse synthetic RGB, exact float32 metric depth and camera pose;
  only static environments are admitted as zero-dynamic supervision.
- TUM RGB-D: real sensor appearance, metric depth and mocap camera pose.
- Bonn Dynamic RGB-D: real dynamic people/objects, metric depth and mocap pose.

Every source must pass finite RGB, valid metric depth, SO(3), non-trivial
motion, first-pose normalization, and scene-disjoint split hypotheses before
training. TartanAir v2 four-channel PNG depth is decoded as raw float32 bytes;
treating it as a 16-bit PNG is explicitly tested against.

## Promotion gates

Evaluation uses normal images plus black-image, reverse-time, wrong-window and
fixed-prior controls. The fixed depth value is estimated from training data
only. A deployable checkpoint must satisfy all six conditions:

- depth improves at least 20% over the fixed-depth prior;
- metric ATE improves at least 20% over identity pose;
- black images worsen depth by at least 25%;
- wrong windows worsen depth by at least 25%;
- wrong windows worsen pose by at least 25%;
- reversing time worsens motion RPE by at least 10%.

The output is named `jwm_eye_physical_v2.pt` only after passing. A failed run
is preserved as `jwm_eye_physical_v2_blocked.pt` for diagnosis and is never
attached to JARVIS.
