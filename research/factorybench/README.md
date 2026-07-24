# FactoryBench Industrial JWM Track

Artifacts in this folder audit FactoryBench as a research benchmark for:

```text
sensor history + control signals + machine context → text response
```

## Generate the audit

```bash
python scripts/audit_factorybench.py
```

The source dataset is downloaded into the Hugging Face cache, not committed to
this repository. Generated representative samples retain source provenance.

## Files

- `FACTORYBENCH_AUDIT.md`: decision-focused audit.
- `factorybench_audit.json`: machine-readable counts, leakage checks and gates.
- `representative_samples_l1_l4.json`: compact samples across causal levels.
- `COSMOS_JWM_MAPPING.md`: architecture and curriculum mapping.
- `factorybench_baseline_colab.ipynb`: reproducible audit and baseline notebook.
- `FACTORYBENCH_BASELINE.md`: interpretation of the context-blind score floor.
- `FACTORYBENCH_METRICS.md`: task-aware metrics, causal controls and seven
  provisional admission gates.

## License

FactoryBench is published under **CC BY-NC 4.0**. It is suitable for this
research evaluation, but direct commercial training/release requires separate
legal clearance or a commercially compatible replacement.

Source: https://huggingface.co/datasets/FactoryBench/FactoryBench
