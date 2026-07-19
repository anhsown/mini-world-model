# LingBot-Map → JWM Eye Physical

This is a clean-room, mini-scale adaptation of the **architectural principle**
in LingBot-Map. JWM does not copy its checkpoint, ViT-L backbone or source
implementation.

## Reverse-engineered structure

LingBot-Map keeps three complementary context classes while streaming:

1. **Anchor context** — full tokens from the first frames establish metric
   scale and a stable coordinate system.
2. **Pose-reference window** — full dense tokens from the most recent frames
   provide overlap for local depth and camera-motion estimation.
3. **Trajectory memory** — frames evicted from the dense window retain only
   compact camera/register tokens for long-range drift correction.

For `M` visual tokens, `A` anchors, local window `K`, `S` summary tokens and
sequence length `T`, dense causal memory costs `O(MT)`. The structured JWM cache
costs `O(M(A+K) + ST)`, with a bounded history summary after the trajectory
limit. At JWM's Eye configuration, `M=256`, `A=2`, `K=8`, and `S=6`.

## JWM implementation

| LingBot-Map concept | JWM mini implementation |
|---|---|
| DINO/VGGT visual tokens | Existing JWM document/local vision stem |
| Alternating frame/global attention | Per-frame self-attention + context cross-attention |
| Anchor context | Two full anchor frames |
| Pose-reference window | Eight full recent frames |
| Trajectory memory | Camera + marker + four register tokens per evicted frame |
| Unbounded compact history | 256 summaries, then EMA history token |
| Camera/depth heads | 6D rotation + translation; positive depth + log uncertainty |
| Separate mapping model | Shared `world_tokens` for AR reasoner and DM generator |

The output C2W rotation is produced with the continuous 6D representation and
Gram–Schmidt projection onto SO(3). The training objective is

`L = λd Ldepth + λa(Lrot_abs + Ltrans_abs) + λr Lpose_relative`,

where depth is anchor-scale-normalized and uncertainty weighted. Relative pose
is supervised over all frame pairs inside the local window, not only adjacent
frames.

## Intentional differences

- The first anchor group is bidirectional; every later frame is causal.
- Old summaries are bounded by an EMA history token for laptop deployment.
- No claim of 20-FPS geometry inference: JARVIS captures/displays at 30 FPS but
  schedules expensive geometry on adaptive keyframes.
- OCR remains a separate spatial decoding path. Geometric memory does not fix
  the v3 full-page CTC blank-collapse failure.
- Dynamic-object modeling and loop closure remain explicit future stages.

## Verification contract

- SO(3) orthogonality and determinant tests.
- Relative-pose invariance to a common world-frame transform.
- Future-frame mutation must not change earlier streaming outputs.
- Anchor/local/trajectory retention counts are checked exactly.
- Depth stays positive; geometry loss is finite and backpropagates.
- All legacy JWM tests must remain green when geometry is disabled.

Implementation: `jwm/geometric_memory.py`; tests:
`tests/test_geometric_memory.py`.

