# JWM — DAY 6 · Eye v3.2.1 Robust Causal Geometry

## Starting point

- The latest evaluation passed only **3/7 causal/OOD gates**.
- Remaining failure points were pose, temporal causality, tracking robustness, and dynamic-scene generalization.
- Root-cause analysis uncovered a critical target-construction bug: invalid rigid-flow values were interpolated before masking, allowing sentinel values to contaminate valid supervision and produce extreme EPE and gradient spikes.

## Research and design decision

- Reviewed current work on metric geometry, point tracking, optical flow, differentiable bundle adjustment, and dynamic-scene reconstruction.
- Rejected adding more steps to the old architecture because the primary failures were in target construction, uncertainty modeling, temporal supervision, and evaluator contracts.
- Moved to **JWM-Eye v3.2.1**, emphasizing verifiable causal geometry instead of training-loss reduction alone.

## Completed changes

- Added validity-normalized flow resizing to prevent invalid sentinel contamination.
- Added robust heteroscedastic tracking with separate confidence, visibility, and uncertainty predictions.
- Aligned confidence supervision with the evaluator contract `P(EPE ≤ 3 px)` and separated 1 px and 3 px calibration metrics.
- Added temporal compatibility learning with genuinely reordered negative windows.
- Added balanced dynamic focal loss and mandatory positive dynamic supervision in dataset admission.
- BA now weights correspondences by confidence, visibility, and static probability; its gate requires improvements in both residual and ATE.
- Pose evaluation is motion-stratified so near-static windows cannot make the identity prior appear successful.
- Redefined all seven promotion gates for depth, pose, BA, temporal pairing, tracking quality, reverse time, and incorrect intrinsics.

## Software verification

- Full workspace: **142 tests passed**.
- Public export repository: **121 tests passed**, with 4 non-blocking warnings.
- The exact-graph CPU smoke test remained finite with no NaN/Inf.
- Source was pushed by `anhsown` in commit `6c6ccfe`.

## v3.2.1 training

- Notebook: `jwm/kaggle/jwm_eye_v321_robust_causal_t4x2.ipynb`.
- Platform: Kaggle T4×2.
- The adaptive controller stopped training in `g0_calibrated_tracks` at step **1,000** with `stop_overfit`: training loss improved while held-out OOD performance regressed.
- The best controller score occurred at step **400** (`0.20374`), versus `-0.03962` at step 1,000.
- The run did not enter later temporal/dynamic stages because the promotion contract blocked it correctly.

## Training and benchmark result

| Item | Result |
|---|---|
| Final checkpoint | `jwm_eye_v321_blocked.pt` |
| Model | 381.927M parameters, 639 tensors, 1.527 GB |
| Executed steps | 1,000; stopped in stage g0 |
| Checkpoint integrity | No NaN/Inf tensors |
| Final real Depth AbsRel ↓ | 0.48775 |
| Final real Depth δ1 ↑ | 0.26889 |
| Final real ATE ↓ | 0.03282 m |
| Track EPE / P90 ↓ | 0.57980 / 1.06852 px |
| Track PCK@3 ↑ | 0.99469 |
| Track ECE@3 ↓ | 0.05842 |
| Dynamic F1 ↑ | 0.02993 |
| Causal/OOD gates | **2/7** |
| Decision | **BLOCKED — do not deploy to JARVIS** |

### Seven-gate breakdown

| Gate | Measured | Requirement | Result |
|---|---:|---:|---|
| Depth beats fixed prior | 0.838× | ≥1.20× | FAIL |
| Pose beats moving identity | 1.015× | ≥1.20× | FAIL |
| BA improves residual and pose | residual 0.99992; pose gain 22.63× | both conditions | **PASS** |
| Detects wrong temporal window | gap ≈ 0.000 | ≥0.15 | FAIL |
| Tracking usable and calibrated | quality 0.978 | ≥0.80 | **PASS** |
| Detects reverse time | 1.004× | ≥1.10× | FAIL |
| Detects incorrect intrinsics | 1.015× | ≥1.15× | FAIL |

### Technical conclusion

- The flow-mask and uncertainty correction worked: tracking EPE no longer explodes to hundreds of thousands of pixels, and tracking/calibration are now the checkpoint's strongest capabilities.
- The model still loses to the fixed-depth prior, barely matches the identity-pose prior, and does not distinguish correct from incorrect frame ordering or intrinsics.
- `temporal_compatibility≈0.49939` is effectively random guessing.
- `dynamic F1≈0.03` confirms that dynamic-scene understanding still collapses.
- The new 2/7 result is not perfectly comparable to the earlier 3/7 result because v3.2.1 changed the evaluator contract; under the new gates, the model is clearly not promotable.

Checkpoint SHA-256: `6967192C246BD52D22B530DB7F4DF350C67AE2B11429901EF7D841165966D4B7`.

The checkpoint is retained for failure analysis and selective warm-starting only; it must not be attached directly to JARVIS.
