"""Append an SME-reviewed OPC UA export to the FactoryTraj-B0 seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def optional_range(row: dict[str, str], prefix: str):
    low, high = row.get(f"{prefix}_low"), row.get(f"{prefix}_high")
    return {"low": float(low), "high": float(high)} if low and high else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("research/factorytraj_bench/b0_seed_v0.1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/factorytraj_bench/b0_seed_with_opcua_v0.1.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.seed.read_text(encoding="utf-8"))
    with args.csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviewed = [row for row in rows if row["review_status"] == "reviewed"]
    if not reviewed:
        raise ValueError("no SME-reviewed rows")
    existing = {item["item_id"] for item in payload["items"]}
    for index, row in enumerate(reviewed):
        item_id = f"opcua:{row['source_id']}:{row['node_id']}"
        if item_id in existing:
            raise ValueError(f"duplicate item_id: {item_id}")
        payload["items"].append(
            {
                "item_id": item_id,
                "source_dataset": row["source_id"],
                "split": "test",
                "input": {
                    "tag_name": row["browse_name"],
                    "anonymized_tag_id": f"opcua_tag_{index:05d}",
                    "declared_data_type": row["data_type"],
                    "representative_samples": [],
                    "documentation": row.get("description") or None,
                },
                "ground_truth": {
                    "data_type": row["data_type"],
                    "engineering_unit": row.get("engineering_unit_display") or None,
                    "instrument_range": optional_range(row, "instrument"),
                    "eu_range": optional_range(row, "eu"),
                    "role": row["role"],
                    "relationships": json.loads(row["relationships_json"]),
                },
                "provenance": {
                    "schema_and_type": row["node_id"],
                    "role": "SME-reviewed OPC UA export",
                    "unit": "OPC UA EngineeringUnits",
                    "range": "OPC UA InstrumentRange/EURange",
                    "relationship": "SME-reviewed OPC UA export",
                },
            }
        )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"appended {len(reviewed)} reviewed records -> {args.output}")


if __name__ == "__main__":
    main()
