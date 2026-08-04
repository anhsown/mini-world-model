"""Validate JSON Schema plus cross-field IWM-Episodes v1 invariants."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parent
schema = json.loads((ROOT / "episode.schema.json").read_text(encoding="utf-8"))
taxonomy = json.loads((ROOT / "industrial_event_taxonomy_v1.json").read_text(encoding="utf-8"))
payload = json.loads((ROOT / "golden_samples_v1.json").read_text(encoding="utf-8"))

Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema, format_checker=FormatChecker())
event_ids = {x["id"] for x in taxonomy["events"]}
action_ids = {x["id"] for x in taxonomy["actions"]}
outcomes = set(taxonomy["outcome_status"])
errors = []
split_groups = {}
episode_ids = set()

for episode in payload["samples"]:
    eid = episode["episode_id"]
    if eid in episode_ids:
        errors.append(f"{eid}: duplicate episode_id")
    episode_ids.add(eid)
    for error in sorted(validator.iter_errors(episode), key=lambda e: list(e.path)):
        path = ".".join(map(str, error.path))
        errors.append(f"{eid}:{path}: {error.message}")

    start = datetime.fromisoformat(episode["time"]["start_time"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(episode["time"]["end_time"].replace("Z", "+00:00"))
    duration = episode["time"]["duration_s"]
    if abs((end - start).total_seconds() - duration) > 1e-6:
        errors.append(f"{eid}: duration does not match timestamps")

    stream_ids = []
    for group in ("video", "audio", "telemetry", "other"):
        for item in episode["observations"][group]:
            stream_ids.append(item["stream_id"])
            if item["clock_error_ms"] > episode["time"]["max_clock_error_ms"]:
                errors.append(f"{eid}: stream clock error exceeds episode maximum")
    if len(stream_ids) != len(set(stream_ids)):
        errors.append(f"{eid}: duplicate stream_id")

    for action in episode["actions"]:
        if action["type"] not in action_ids:
            errors.append(f"{eid}: unknown action {action['type']}")
        if not (0 <= action["start_offset_s"] <= action["end_offset_s"] <= duration):
            errors.append(f"{eid}: invalid action interval")
        if action["executed"] and not action["safety_authorized"]:
            errors.append(f"{eid}: executed action lacks safety authorization")

    for event in episode["events"]:
        if event["taxonomy_id"] not in event_ids:
            errors.append(f"{eid}: unknown event {event['taxonomy_id']}")
        if not (0 <= event["start_offset_s"] <= event["end_offset_s"] <= duration):
            errors.append(f"{eid}: invalid event interval")

    if episode["outcome"]["status"] not in outcomes:
        errors.append(f"{eid}: unknown outcome")
    group = episode["split"]["split_group"]
    split = episode["split"]["split"]
    if group in split_groups and split_groups[group] != split:
        errors.append(f"{eid}: split-group leakage")
    split_groups[group] = split
    if episode["governance"]["permitted_use"] == "public":
        if not episode["governance"]["redistribution_allowed"]:
            errors.append(f"{eid}: public record cannot be redistributed")
        if episode["governance"]["privacy_review"] != "pass":
            errors.append(f"{eid}: public record lacks privacy pass")

with (ROOT / "industrial_feature_dictionary.csv").open(encoding="utf-8", newline="") as handle:
    features = list(csv.DictReader(handle))
if len(features) < 40:
    errors.append("feature dictionary is unexpectedly small")
paths = [row["json_path"] for row in features]
if any(not path.startswith("$.") for path in paths):
    errors.append("invalid feature dictionary JSON path")

report = {
    "valid": not errors,
    "schema_version": payload["schema_version"],
    "episodes": len(payload["samples"]),
    "features": len(features),
    "event_classes": len(event_ids),
    "action_classes": len(action_ids),
    "splits": sorted({x["split"]["split"] for x in payload["samples"]}),
    "asset_types": sorted({x["asset_type"] for x in payload["samples"]}),
    "errors": errors,
}
(ROOT / "golden_samples_validation_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["valid"] else 1)

