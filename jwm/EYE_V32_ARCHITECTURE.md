# JWM-Eye v3.2 — depth-ray causal world model

## Capacity

- Total parameters: **381,480,998 (381.48M)**.
- Physical geometry parameters: **14,659,201 (14.66M)**.
- MoT block parameters: **341,379,072 (341.38M)**.
- Backbone: `d_model=576`, 16 dual-tower MoT layers, 18 heads × 32 dims.
- Reasoner: 32 fine-grained experts, sigmoid top-4 routing + one shared expert.
- Physical eye: width 160, 128 tracks, six recurrent refinements, three BA steps.
- Temporal state: 16 scene registers, six causal layers at width 384.

The expected portable FP32 checkpoint is approximately 1.42 GiB before ZIP
compression. During Eye-only training only the supervised geometry graph is
optimized; intermediate checkpoints therefore store a ~56 MiB geometry delta
plus optimizer state. The final checkpoint contains the complete model.

## Corrective changes over v3.1

1. **Safe identity initialization.** JWM's global initializer previously
   overwrote the zero-delta pose and track heads. v3.2 restores those priors
   after global initialization and tests the invariant.
2. **Depth-ray factorization.** Metric depth is paired with calibrated camera
   rays and a bounded learned ray residual. This separates scene distance from
   viewing geometry instead of forcing one depth head to absorb both.
3. **Causal scene registers.** A compact register sequence aggregates only the
   current and past frames. It conditions depth and pairwise pose without full
   space-time attention or unbounded memory.
4. **Bidirectional track cycle.** Forward tracks are warped back to their source
   and penalized for cycle error, reducing drift and false correspondences.
5. **Calibrated confidence.** Track confidence directly predicts the measurable
   event `EPE <= 1 feature pixel`, matching the ECE evaluation contract.
6. **Scale-preserving warm start.** The 384×8 JWM-v4 subspace is copied into the
   576×16 model; inserted layers are residual identities and the new channels
   retain their initializer. The new eye itself is deliberately reset.

## Causal and promotion contracts

- Scene token at frame `t` cannot attend to any frame `> t`.
- Intrinsics are mandatory and wrong-intrinsics controls remain part of the
  seven-gate evaluation.
- Training loss cannot promote a checkpoint. Real held-out depth, pose, BA,
  wrong-window, reverse-time and wrong-intrinsics gates control the result.
- Synthetic data remains quarantined unless its real-heldout A/B verdict is
  `valid=true` and `decision=admit`.

## T4×2 execution

Use `jwm/kaggle/jwm_eye_v32_depth_ray_t4x2.ipynb`. The notebook runs unit
contracts, data validation, mechanism probes, a 100-step exact-graph canary,
adaptive training, one final real test and artifact hashing. Persistence must
be **Files only** and both `jwm_v4.pt` and the admitted synthetic verdict must
be attached as private Kaggle inputs.
