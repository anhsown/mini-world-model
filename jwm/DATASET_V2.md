# JWM Real-Anchored Dataset v2

## Principle

Real samples define the deployment distribution. Synthetic samples provide
exact geometry, counterfactual interventions and long-tail coverage. A source
is not training data merely because it was downloaded: it must pass modality
admission and a small real-held-out ablation.

## Dataset branches

### Eye

- Real anchors: TUM RGB-D and Bonn Dynamic RGB-D.
- Simulation anchors: selected TartanAir trajectories; Dynamic Replica and
  PointOdyssey remain manual-license sources.
- Analytic synthetic: deterministic RGB/depth/pose/dynamic-mask/rigid-flow
  generated on demand by `RealAnchoredSyntheticGeometry`.
- Default logical size: 250,000 train sequences, with seed-disjoint validation
  and test namespaces. No images are stored for this branch.

### Reader

- Real anchor: TranNhiem Vietnamese DocumentImage Reasoning already available
  locally.
- Future manual source: DocVQA (official portal login required).
- Synthetic: Vietnamese glyph/line/paragraph/layout rendering. Benchmark pages
  such as MTVQA-VI remain evaluation-only.

### Generator and action

These remain separate from the Eye-v3.2 admission pack. Real video and robot
trajectory sources must preserve their own licenses and embodiment-specific
action units before entering Cosmos-style generator mid/post-training.

## Admission gates

1. Source/license/provenance recorded in `manifest.jsonl`.
2. Archive checksum and safe extraction.
3. Scene-level split disjointness.
4. Finite calibrated intrinsics and strictly increasing timestamps.
5. Metric depth and SE(3) pose validity.
6. RGB-D-pose reprojection agreement and static rigid-flow coverage.
7. Synthetic appearance/depth statistics close to real anchors.
8. Real + synthetic probe improves real-only held-out metrics and does not
   regress causal/OOD gates.

## Commands

```powershell
python scripts/prepare_real_anchor_data.py --tier starter --branch eye
python scripts/build_real_anchored_synthetic.py --samples 250000
```

Both download and generation are shutdown-safe. Downloads use `.part` files;
synthetic samples are deterministic functions of their seed.

