# Google Cloud runbook — Cosmos 3 Nano × MMAD Representative-1400

## Recommended first configuration

- Compute Engine VM: `g2-standard-24`
- GPUs: 2 × NVIDIA L4 (48 GB aggregate VRAM)
- RAM: 96 GB
- Persistent disk: 150 GB `pd-balanced`
- Provisioning: standard for the first 2-hour pilot; Spot only after checkpoint resume is verified
- Region/zone: any available G2 zone with approved quota; prefer a nearby region unless a US zone is materially cheaper or easier to allocate
- Notebook: `Cosmos3_Nano_MMAD_Representative1400_Optimized_GCP.ipynb`

This configuration is intentionally comparable to Kaggle T4×2 while providing
more VRAM per GPU, persistent storage, and uninterrupted paid runtime.

## Cost guardrails

At the published Iowa list price, `g2-standard-24` is approximately $2.0008/hour
before disk, network, taxes, and any account-specific discounts. A 2-hour pilot is
therefore about $4 of compute; 24 hours is about $48 of compute. Always confirm the
selected region in the Google Cloud Pricing Calculator before creation.

Create a project-scoped billing budget before starting. Budget alerts do not
automatically stop Compute Engine VMs, so also configure an idle shutdown or a
maximum run duration and stop the VM manually after the notebook finishes.

Suggested experiment budget:

- warning: $25
- escalation: $50
- hard internal stop for the first experiment: $75
- keep the remaining credit untouched until the 2-hour pilot is reviewed

## Before creating the VM

1. Create a dedicated project such as `cosmos3-mmad-research` and attach the credit-enabled billing account.
2. Enable Compute Engine API.
3. Check both global GPU quota and regional L4/G2 quota.
4. If quota is zero, request 2 L4 GPUs in a G2-supported region plus the global GPU quota.
5. Create the billing budget and email alerts.

## VM creation choices

Use the Google Cloud console for the first run:

1. Compute Engine → VM instances → Create instance.
2. Select a zone that lists G2/L4 capacity.
3. Machine configuration → GPU → `g2-standard-24`.
4. Use an Ubuntu image supported by G2 and install NVIDIA drivers when prompted.
5. Set a 150 GB balanced persistent boot/data disk.
6. Do not enable a public inbound Jupyter port. Use SSH port forwarding or Google-provided secure access.
7. Add labels such as `experiment=cosmos3-mmad` and `owner=anhsown` for billing filters.

## Persistent working directory

The notebook defaults to:

```text
/home/jupyter/jwm-work
```

If the persistent disk is mounted elsewhere, set:

```bash
export JWM_WORK_ROOT=/mnt/disks/jwm-data
```

The Hugging Face cache, repository, MMAD metadata, outputs, and checkpoint all
live below this directory and remain after a VM stop.

## Secrets

Never paste tokens into the notebook. Export them only for the current shell or
inject them through Secret Manager:

```bash
export HF_TOKEN='...'
export GITHUB_TOKEN='...'
```

The notebook also supports hidden prompts. Rotate any token that has previously
been pasted into chat, logs, or source files.

## Execution sequence

1. Open `Cosmos3_Nano_MMAD_Representative1400_Optimized_GCP.ipynb`.
2. Run the dependency cell once.
3. Restart the kernel once.
4. Run the environment, hardware, manifest, auth, and model-load cells.
5. Run the five-sample smoke test and inspect image/question pairing, reasoning, answer parsing, latency, and GPU memory.
6. Run Phase 2 with `MAX_RUNTIME_HOURS = 2.0`.
7. Confirm checkpoints were pushed and the local JSONL/archive exists.
8. Stop the VM immediately after collecting the pilot artifacts.

Only increase the runtime after comparing throughput and quality against Kaggle.

## Promotion rule

Continue the representative-1400 run on G2 only if all are true:

- smoke output coverage ≥ 80%;
- non-empty reasoning rate ≥ 80%;
- no image/question mismatch;
- successful rows are visible in the shared GitHub checkpoint;
- median latency is materially lower than the Kaggle baseline (~46.5 seconds);
- projected full representative-run cost remains within the experiment budget.

If one L4 is enough, downgrade to `g2-standard-12`. If two L4s still rely heavily
on CPU offload or do not improve latency, test `a2-highgpu-1g` (A100 40 GB) for one
hour before committing to a longer run.

## Official references

- GPU machine types: https://cloud.google.com/compute/docs/gpus
- Accelerator-optimized pricing: https://cloud.google.com/products/compute/pricing/accelerator-optimized
- GPU zones: https://cloud.google.com/compute/docs/regions-zones/gpu-regions-zones
- GPU VM creation and quotas: https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus
- Billing budgets: https://cloud.google.com/billing/docs/how-to/budgets
- Pricing calculator: https://cloud.google.com/products/calculator
