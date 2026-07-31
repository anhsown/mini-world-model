# Cosmos 3 MMAD shared checkpoints

This directory stores immutable JSONL shards produced by the NVIDIA Build and
Kaggle T4x2 runners. At startup, either runner pulls the repository and unions
all rows with `status == "ok"` by canonical `sample_id`.

- Failed, partial, and no-output attempts remain retryable and are not shared.
- Every row must match the canonical MMAD manifest hash in `manifest.json`.
- Shards are immutable and backend-specific to avoid concurrent append conflicts.
- No images, browser profiles, API keys, GitHub tokens, or NVIDIA credentials are stored here.
- Full local output files should still be retained for serving-failure analysis.

Kaggle writes require a private Kaggle Secret named `GITHUB_TOKEN`. The token is
read at runtime and is never written to the notebook or Git configuration.
