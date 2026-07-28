# HATRec x Cosmos 3 Nano Reasoner

This folder contains the complete, evidence-grounded evaluation workflow for
running NVIDIA Cosmos 3 Nano Reasoner on the HATRec industrial assembly video
dataset.

## Contents

- `run_batch.py`: recursively sends HATRec MP4 files to the NVIDIA hosted API.
- `run_all.py`: the only entrypoint needed for the complete automatic pipeline.
- `evaluate_results.py`: extracts task predictions and computes closed-set metrics.
- `inspect_dataset.py`: audits video counts, task balance, duration and resolution.
- `prompts/`: versioned system and user prompts.
- `set_api_key.ps1`: safely stores the NVIDIA key in the Windows user environment.
- `hatrec_cosmos3_reasoning_colab.ipynb`: earlier local-model/Colab workflow.

## Security

Never place an API key in source code. Any key pasted into chat must be revoked.
Generate a fresh key, then run:

```powershell
.\set_api_key.ps1
```

Open a new PowerShell window before running Python.

## One-command automatic run

The dataset is expected at `VideoDataset/Cycles` inside this folder. Run:

```powershell
python C:\Users\ASUS\OneDrive\Desktop\Jarvis-Vision\research\hatrec_cosmos3\run_all.py
```

This command installs missing dependencies, audits the dataset, processes all
videos, resumes past completed reports, calculates metrics, saves every artifact
under `outputs/`, and opens the output folder when complete.

## Backend requirement

The NVIDIA Build playground can run Cosmos 3 interactively. Its public Deploy
instructions currently expose a self-hosted NIM endpoint rather than a hosted
Cosmos 3 batch endpoint. The runner performs `/v1/models` preflight validation
and refuses to substitute an older Cosmos model silently.

For a running Cosmos 3 NIM, configure its OpenAI-compatible endpoint:

```powershell
$env:COSMOS3_API_BASE_URL="http://127.0.0.1:8000/v1"
python C:\Users\ASUS\OneDrive\Desktop\Jarvis-Vision\research\hatrec_cosmos3\run_all.py
```

## Cosmos 3 playground automation

The immediate Cosmos 3 evaluation path uses a visible Chrome session on the
official Experience page. It does not switch to an older model and does not
bypass authentication, CAPTCHA, rate limits, or trial quotas.

Run a balanced 14-video test (two videos per HATRec task):

```powershell
python C:\Users\ASUS\OneDrive\Desktop\Jarvis-Vision\research\hatrec_cosmos3\run_ui.py
```

On the first run, sign in and accept the trial notice in the opened Chrome
window, return to the console, and press Enter. Successful outputs are resumable
and saved under `outputs/ui_reports/`.

## Evaluation

```powershell
python evaluate_results.py --reports outputs/reports
```

The evaluator derives ground truth from the local HATRec filename only after
inference. The filename and path are never included in the model request.
