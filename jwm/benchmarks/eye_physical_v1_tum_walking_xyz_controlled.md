# JWM Eye Physical v1 benchmark

All metrics are lower-is-better.

| Evaluation | Depth AbsRel | Depth RMSE | ATE | Abs rotation (deg) | RPE translation | RPE rotation (deg) |
|---|---:|---:|---:|---:|---:|---:|
| procedural/normal | 0.02670 | 0.05722 | 0.01890 | 0.50797 | 0.00679 | 0.25076 |
| procedural/black | 0.47488 | 0.61141 | 0.03026 | 0.53959 | 0.01209 | 0.25436 |
| procedural/mean_only | 0.18983 | 0.35771 | 0.01480 | 0.51105 | 0.00548 | 0.25101 |
| procedural/frozen_first | 0.03580 | 0.09540 | 0.01897 | 0.50785 | 0.00681 | 0.25072 |
| procedural/reverse_time | 0.04302 | 0.10438 | 0.01916 | 0.50775 | 0.00685 | 0.25073 |
| procedural/wrong_scene | 0.17001 | 0.36186 | 0.01439 | 0.51102 | 0.00574 | 0.25123 |
| procedural/constant_identity_prior | 0.76929 | 0.58937 | 0.02589 | 0.54740 | 0.00973 | 0.27373 |
| tum/normal | 0.40807 | 0.53613 | 0.03843 | 2.82815 | 0.00982 | 1.15841 |
| tum/black | 0.32924 | 0.47862 | 0.03859 | 2.83377 | 0.00996 | 1.15776 |
| tum/frozen_first | 0.39934 | 0.52263 | 0.03793 | 2.82664 | 0.00960 | 1.15809 |
| tum/reverse_time | 0.39259 | 0.52168 | 0.03819 | 2.82751 | 0.00977 | 1.15730 |
| tum/wrong_window | 0.41607 | 0.53349 | 0.03909 | 2.82986 | 0.01038 | 1.15769 |
| tum/constant_identity_prior | 0.29444 | 0.37016 | 0.03699 | 2.80012 | 0.00921 | 1.15544 |

## Image-dependence verdict

- Wrong-scene Depth AbsRel ratio: 6.367x
- Wrong-scene ATE ratio: 0.761x
- Black-image Depth AbsRel ratio: 17.785x
- Visual evidence gate: **FAIL**
