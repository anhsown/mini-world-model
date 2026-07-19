# Eye Physical data curriculum and admission gates

Every source enters training to test a named hypothesis. Sample count alone is
not an admission criterion.

## Stages

| Stage | Mixture | Capability | Required gates |
|---|---|---|---|
| G0 geometry bootstrap | 100% JWM analytic scenes | camera convention, depth, pose, occlusion, dynamic masks | exact labels, SO(3), split isolation, reprojection >75% |
| G1 synthetic diversity | 50% TartanAir V2, 30% Hypersim, 20% G0 | photorealism, large pose/depth diversity, hard motion | valid intrinsics/units, no scene leak, depth/pose ranges, temporal overlap |
| G2 sim-to-real | 50% G1, 35% TUM RGB-D, 15% G0 | Kinect noise, real indoor texture, real 30-Hz camera motion | real-vs-synthetic domain report, ATE/RPE holdout, real depth validity |
| G3 streaming | long TartanAir/TUM sequences | anchor stability, local tracking, trajectory drift | causal tests, memory budget, ATE/RPE vs sequence length |
| G4 world/action | DROID/robot sequences after Eye passes | action-conditioned dynamics | action/video temporal alignment and held-out embodiment split |

## Official sources selected

- **TartanAir V2** — synchronized RGB, depth, pose, segmentation, flow and
  multiple camera models; CC BY 4.0. It supplies scalable motion and hard
  synthetic environments.
- **Hypersim** — 77,400 photorealistic frames from 461 indoor scenes with dense
  geometry and camera information; CC BY-SA 3.0. Distances must be converted to
  planar depth and asset coordinates must be converted with each scene scale.
- **TUM RGB-D** — real 640×480 RGB-D at 30 Hz with 100-Hz mocap trajectory;
  CC BY 4.0. It is the real-domain gate and trajectory benchmark, not merely
  more training data.
- **VinText** — Vietnamese scene text for the Eye Reader OOD branch. It closes
  the current document-only bias; source licensing must be checked and recorded
  before redistribution.
- **TranNhiem Vietnamese DocumentImage Reasoning** — retained for real
  document QA only. QA answers are never mislabeled as full-page transcripts.

## OCR v3.1 correction

The v3 checkpoint proved that more full-page samples do not solve a decoder
whose CTC time axis is a flattened 2D page. The next data contract therefore
adds exact **per-line boxes and transcripts**, random labels, font/size/layout
strata, real scene text, and hard negatives. Training moves to line ROI → 1D
sequence decoding. Full-page QA remains a later reasoner stage.

OCR admission gates:

1. all transcript characters survive tokenizer round-trip;
2. each line box contains visible ink and lies inside the image;
3. transcript fits the ROI decoder time axis after CTC collapse constraints;
4. document/page groups never cross train/val/test;
5. synthetic and real strata are reported separately;
6. blind/crop controls show positive image dependence before stage promotion.

The local analytic geometry report is written to
`data/geometry_validation_day05.json`. External sources are not considered
valid merely because they downloaded successfully; each adapter must emit the
same contract before training.

