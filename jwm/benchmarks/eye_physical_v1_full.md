# JWM Eye Physical v1 benchmark

All metrics are lower-is-better.

| Evaluation | Depth AbsRel | Depth RMSE | ATE | Abs rotation (deg) | RPE translation | RPE rotation (deg) |
|---|---:|---:|---:|---:|---:|---:|
| procedural/normal | 0.03894 | 0.07550 | 0.01510 | 1.13669 | 0.00442 | 0.56676 |
| procedural/black | 0.47251 | 0.62042 | 0.03681 | 1.14809 | 0.01528 | 0.56506 |
| procedural/mean_only | 0.20576 | 0.37886 | 0.01671 | 1.13779 | 0.00509 | 0.56660 |
| procedural/frozen_first | 0.06048 | 0.15912 | 0.01518 | 1.13669 | 0.00447 | 0.56677 |
| procedural/reverse_time | 0.06580 | 0.16735 | 0.01540 | 1.13661 | 0.00450 | 0.56679 |
| procedural/wrong_scene | 0.18626 | 0.38467 | 0.01598 | 1.13695 | 0.00485 | 0.56679 |
| procedural/constant_identity_prior | 0.77238 | 0.60189 | 0.03188 | 1.11483 | 0.01318 | 0.55742 |
| procedural/stage0_normal | 0.05428 | 0.09594 | 0.01929 | 1.11964 | 0.00512 | 0.54778 |

## Image-dependence verdict

- Wrong-scene Depth AbsRel ratio: 4.783x
- Wrong-scene ATE ratio: 1.058x
- Black-image Depth AbsRel ratio: 12.133x
- Visual evidence gate: **FAIL**
