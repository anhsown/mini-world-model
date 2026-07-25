"""Validate an SME-reviewed OPC UA export before B0 admission."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = {
    "source_id",
    "machine_id",
    "node_id",
    "browse_name",
    "data_type",
    "role",
    "relationships_json",
    "review_status",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    with args.csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing_columns = sorted(REQUIRED - set(rows[0] if rows else []))
    errors = []
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    for index, row in enumerate(rows, start=2):
        for field in REQUIRED:
            if not row.get(field):
                errors.append(f"row {index}: missing {field}")
        try:
            relations = json.loads(row.get("relationships_json", ""))
            if not isinstance(relations, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            errors.append(f"row {index}: relationships_json must be a JSON list")
        for prefix in ("instrument", "eu"):
            low, high = row.get(f"{prefix}_low"), row.get(f"{prefix}_high")
            if bool(low) != bool(high):
                errors.append(f"row {index}: {prefix} range requires both bounds")
            if low and high:
                try:
                    if float(low) >= float(high):
                        errors.append(f"row {index}: {prefix}_low must be < {prefix}_high")
                except ValueError:
                    errors.append(f"row {index}: {prefix} range must be numeric")
        if row.get("review_status") not in {"reviewed", "unreviewed"}:
            errors.append(f"row {index}: invalid review_status")
    reviewed = [row for row in rows if row.get("review_status") == "reviewed"]
    reviewed_range_count = sum(
        bool(row.get("instrument_low") and row.get("instrument_high"))
        or bool(row.get("eu_low") and row.get("eu_high"))
        for row in reviewed
    )
    report = {
        "valid": not errors,
        "rows": len(rows),
        "sources": len({row.get("source_id") for row in rows}),
        "machines": len({row.get("machine_id") for row in rows}),
        "reviewed_ratio": (
            sum(row.get("review_status") == "reviewed" for row in rows)
            / max(len(rows), 1)
        ),
        "unit_coverage": (
            sum(bool(row.get("engineering_unit_display")) for row in rows)
            / max(len(rows), 1)
        ),
        "range_coverage": (
            sum(
                bool(row.get("instrument_low") and row.get("instrument_high"))
                or bool(row.get("eu_low") and row.get("eu_high"))
                for row in rows
            )
            / max(len(rows), 1)
        ),
        "reviewed_range_records": reviewed_range_count,
        "b0_range_admission_ready": (
            len(reviewed) > 0
            and reviewed_range_count / len(reviewed) >= 0.5
        ),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
