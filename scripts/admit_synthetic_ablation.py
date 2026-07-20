"""Admit synthetic data only when it improves a real held-out probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normal(report: dict) -> dict:
    if "report" in report:
        report = report["report"]
    return report["controls"]["normal"]


def admission_report(base: dict, mixed: dict,
                     max_regression: float = .03) -> dict:
    base_m, mixed_m = normal(base), normal(mixed)
    lower = ("depth_abs_rel", "ate_metric", "rpe_translation", "track_epe_p90",
             "track_ece")
    comparisons = {}
    for key in lower:
        ratio = mixed_m[key] / max(base_m[key], 1e-8)
        comparisons[key] = {"real_only": base_m[key], "mixed": mixed_m[key],
                            "ratio": ratio,
                            "pass": ratio <= 1 + max_regression}
    base_gates = base.get("summary", base.get("report", {}).get("summary", {}))
    mixed_gates = mixed.get("summary", mixed.get("report", {}).get("summary", {}))
    causal_ok = mixed_gates.get("causal_gate_pass_rate", 0) >= \
        base_gates.get("causal_gate_pass_rate", 0)
    capability_ok = mixed_gates.get("probe_capability_score", 0) >= \
        base_gates.get("probe_capability_score", 0)
    worst_source_ok = mixed_gates.get("worst_source_score", 0) >= \
        base_gates.get("worst_source_score", 0) * (1 - max_regression)
    improved = sum(mixed_m[key] < base_m[key] for key in lower) >= 3
    valid = (all(row["pass"] for row in comparisons.values()) and causal_ok and
             capability_ok and worst_source_ok and improved)
    return {"valid": valid, "hypotheses": {
        "H_no_real_metric_regression": all(row["pass"] for row in comparisons.values()),
        "H_causal_gate_non_regression": causal_ok,
        "H_capability_score_non_regression": capability_ok,
        "H_worst_source_within_tolerance": worst_source_ok,
        "H_majority_real_metrics_improve": improved,
    }, "comparisons": comparisons,
       "decision": "admit" if valid else "quarantine"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-only", type=Path, required=True)
    parser.add_argument("--real-plus-synthetic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-regression", type=float, default=.03)
    args = parser.parse_args()
    base = json.loads(args.real_only.read_text(encoding="utf-8"))
    mixed = json.loads(args.real_plus_synthetic.read_text(encoding="utf-8"))
    report = admission_report(base, mixed, args.max_regression)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
