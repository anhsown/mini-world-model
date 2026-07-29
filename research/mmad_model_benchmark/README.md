# MMAD model benchmark: Cosmos 3 vs Qwen2-VL-2B

This package compares two models on one frozen MMAD zero-shot protocol:

- 140 multiple-choice questions: 20 for each of MMAD's seven canonical tasks.
- 138 unique query images, fetched from the official Hugging Face ZIP files by HTTP range.
- Near-balanced coverage of four source families: MVTec-AD, MVTec-LOCO, VisA and GoodsAD.
- The same neutral image filenames, system prompt, user prompt, parser and evaluator for both models.
- Manifest SHA-256: `7f6dcad2dda8bdd0a2f876c4b7a740239cf9437b3eb1636da88746ebd0aba50f`.

The benchmark is zero-shot because NVIDIA Build currently accepts one image per request. It is not a reproduction of MMAD's 1-shot and domain-knowledge settings.

## 1. Prepare the shared subset

```powershell
python prepare_subset.py
```

This downloads only `mmad.json` plus the 138 selected images. It does not download the complete ~28 GB MMAD image corpus.

## 2. Cosmos 3 Nano on NVIDIA Build

Close other Chrome processes that use the same automation profile, then run:

```powershell
$env:NVIDIA_EMAIL="your@email.com"
python models/cosmos3/run_nvidia_build.py
```

Complete password/MFA in the opened Chrome window if requested. The runner then continues automatically, checkpoints each answer to JSONL and resumes completed sample IDs after interruption.

Smoke test:

```powershell
python models/cosmos3/run_nvidia_build.py --limit 3 --delay 1
```

## 3. Qwen2-VL-2B on a free T4

Upload and run `models/qwen2_vl/Qwen2VL_2B_MMAD_ZeroShot.ipynb` in Colab or Kaggle with a T4 GPU. The notebook clones the public repository, materializes the exact shared subset, runs FP16 deterministic inference, plots metrics and creates `qwen2_vl_mmad_results.zip`.

## Outputs

Each model directory contains:

- `predictions.jsonl`: append-only raw responses, status and latency;
- `metrics.json`: aggregate and per-task/per-source metrics;
- `predictions_scored.csv`: row-level truth, prediction and correctness;
- error screenshots for recoverable NVIDIA UI failures.

Primary metrics are micro accuracy, macro seven-task accuracy, per-task accuracy, per-source accuracy, parse/completion rates, anomaly precision/recall/F1, miss/overkill rates, and latency mean/median/p95.

## Fairness and leakage controls

- Both systems use the same manifest hash and prompts.
- Query files are renamed to `sample_XXXX`; source, class and defect labels are never shown to the model.
- Generation is deterministic and the parser only accepts an unambiguous A-D answer.
- MMAD's official annotations remain evaluation-only; no selected answer is inserted into the prompt.
- Results are not directly comparable to the official MMAD leaderboard because this protocol is a small frozen zero-shot diagnostic subset.
