"""Build ten deterministic FactoryTraj-Bench examples from audited test samples."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jwm.factorytraj import SCHEMA_VERSION, factorybench_to_factorytraj


SOURCE = ROOT / "research" / "factorybench" / "representative_samples_l1_l4.json"
OUTPUT = ROOT / "research" / "factorytraj_bench" / "factorybench_samples_v0.1.json"

SELECTION = [
    (1, "predictive", 0),
    (1, "comparative", 0),
    (2, "anomaly detection", 0),
    (2, "comparative", 0),
    (3, "predictive", 0),
    (3, "intervention_outcome", 0),
    (3, "trajectory_outcome_multiselect", 0),
    (4, "troubleshooting", 0),
    (4, "troubleshooting", 1),
    (4, "optimization", 0),
]


def select_samples(source: list[dict]) -> list[dict]:
    selected: list[dict] = []
    for level, template, occurrence in SELECTION:
        matches = [
            item
            for item in source
            if item["level"] == level and item["template_type"] == template
        ]
        if occurrence >= len(matches):
            raise RuntimeError(
                f"missing representative sample L{level}/{template}/{occurrence}"
            )
        selected.append(matches[occurrence])
    return selected


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = [
        factorybench_to_factorytraj(item) for item in select_samples(source)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Ten compact, provenance-preserving FactoryBench test examples "
            "mapped to FactoryTraj-Bench v0.1. These are research-only "
            "visualization/contract examples, not a training release."
        ),
        "source_revision": "e2ad55f2c4a66d3f3190170cfe06a50cc93a2e46",
        "items": items,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(items)} items -> {OUTPUT}")


if __name__ == "__main__":
    main()
