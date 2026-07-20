# JWM Metrics v2

Training loss is an optimizer diagnostic, not evidence that the Eye understands
physical geometry. Checkpoint promotion uses held-out real data and mandatory
causal interventions.

## Logging levels

### 1. Optimization health (every 25 steps)

- Total and component losses.
- Gradient norm, learning rate and optimizer steps/second.
- Non-finite skip rate.
- MoE routing entropy/dead experts when the reasoner tower is active.

These metrics can stop an unstable run but cannot promote a checkpoint.

### 2. Physical accuracy (fixed validation windows)

- Depth: AbsRel, RMSE, log-RMSE, SILog and delta1/2/3.
- Pose: metric ATE, median ATE, absolute rotation, translational RPE and
  rotational RPE.
- Tracks: mean/median/p90 EPE, PCK@1, PCK@3 and outlier rate. Mean EPE is
  diagnostic only; p90 and outlier rate control promotion so one numerical
  explosion cannot erase otherwise interpretable tracking evidence.
- Confidence: track ECE and Brier score.
- Dynamics: precision, recall, F1 and IoU.
- Optimization geometry: bundle-adjustment residual reduction.

### 3. Causal dependence

The same held-out target is evaluated with normal, black, frozen, reversed,
wrong-window and wrong-intrinsics inputs. All mandatory gates must pass. This
prevents a constant-depth or identity-pose shortcut from receiving a good
checkpoint score.

### 4. OOD robustness

Metrics are reported per source and by worst source. The mean is never allowed
to hide collapse on dynamic real scenes. Synthetic-to-real gap is measured by
the same normalized capability score on synthetic and real-only validation.

### 5. Reader

- Free-running CER, ANLS and exact match.
- CTC CER and text-region IoU.
- Vision gain against shuffled/blind images.
- Worst-kind CER to expose failures on long paragraphs, documents or OOD text.

## Promotion rule

A checkpoint is deployable only when all required causal gates pass, real OOD
metrics do not regress, confidence calibration is acceptable, and held-out
progress has converged under the adaptive budget. A lower train loss alone is
never sufficient.
