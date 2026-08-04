# JWM Eye Physical v1 benchmark

All metrics are lower-is-better.

| Evaluation | Depth AbsRel | Depth RMSE | ATE | Abs rotation (deg) | RPE translation | RPE rotation (deg) |
|---|---:|---:|---:|---:|---:|---:|
| procedural/normal | 0.03013 | 0.06286 | 0.01576 | 1.18949 | 0.00496 | 0.59357 |
| procedural/black | 0.48527 | 0.62145 | 0.03645 | 1.20298 | 0.01500 | 0.59338 |
| procedural/mean_only | 0.20362 | 0.36790 | 0.01648 | 1.19087 | 0.00525 | 0.59355 |
| procedural/frozen_first | 0.05008 | 0.13780 | 0.01590 | 1.18953 | 0.00503 | 0.59357 |
| procedural/reverse_time | 0.05427 | 0.14562 | 0.01636 | 1.18958 | 0.00507 | 0.59363 |
| procedural/wrong_scene | 0.15739 | 0.35929 | 0.01608 | 1.19042 | 0.00506 | 0.59374 |
| procedural/constant_identity_prior | 0.78630 | 0.59920 | 0.03151 | 1.18627 | 0.01285 | 0.59319 |
| tum/normal | 0.40807 | 0.53613 | 0.03843 | 2.82815 | 0.00982 | 1.15841 |
| tum/black | 0.32924 | 0.47862 | 0.03859 | 2.83377 | 0.00996 | 1.15776 |
| tum/frozen_first | 0.39934 | 0.52263 | 0.03793 | 2.82664 | 0.00960 | 1.15809 |
| tum/reverse_time | 0.39259 | 0.52168 | 0.03819 | 2.82751 | 0.00977 | 1.15730 |

## Image-dependence verdict

- Wrong-scene Depth AbsRel ratio: 5.223x
- Wrong-scene ATE ratio: 1.020x
- Black-image Depth AbsRel ratio: 16.105x
- Visual evidence gate: **FAIL**
