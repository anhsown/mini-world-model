import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


GROUND_TRUTH_RE = re.compile(r"_task_(\d+)$", re.IGNORECASE)
PREDICTION_PATTERNS = [
    re.compile(r"MOST LIKELY HATREC TASK[^\n]*?\b([0-6])\b", re.IGNORECASE),
    re.compile(r"(?:task|class)\s*[:#-]?\s*([0-6])\b", re.IGNORECASE),
]
CONFIDENCE_RE = re.compile(r"confidence[^\n%]*?([0-9]{1,3}(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def extract_prediction(text: str):
    for pattern in PREDICTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def extract_confidence(text: str):
    match = CONFIDENCE_RE.search(text)
    if not match:
        return None
    return min(100.0, float(match.group(1))) / 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.reports.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") not in {"ok", "partial"}:
            continue
        sample_id = record.get("sample_id", "")
        truth_match = GROUND_TRUTH_RE.search(sample_id)
        truth = int(truth_match.group(1)) if truth_match else None
        prediction = extract_prediction(record.get("report", ""))
        confidence = extract_confidence(record.get("report", ""))
        rows.append({
            "sample_id": sample_id,
            "truth": truth,
            "prediction": prediction,
            "confidence": confidence,
            "correct": truth is not None and prediction == truth,
            "latency_seconds": record.get("latency_seconds"),
            "completion": record.get("completion", "complete"),
        })

    scored = [r for r in rows if r["truth"] is not None and r["prediction"] is not None]
    correct = sum(r["correct"] for r in scored)
    per_task = {}
    for task in range(7):
        task_rows = [r for r in scored if r["truth"] == task]
        per_task[str(task)] = {
            "n": len(task_rows),
            "accuracy": sum(r["correct"] for r in task_rows) / len(task_rows) if task_rows else None,
        }
    confusion = Counter((r["truth"], r["prediction"]) for r in scored)
    summary = {
        "reports_found": len(rows),
        "scored": len(scored),
        "parse_failures": len(rows) - len(scored),
        "partial_reports": sum(
            r["completion"] in {"partial_timeout", "timeout_no_output"} for r in rows
        ),
        "accuracy": correct / len(scored) if scored else None,
        "mean_latency_seconds": (
            sum(r["latency_seconds"] for r in scored if r["latency_seconds"] is not None)
            / max(1, sum(r["latency_seconds"] is not None for r in scored))
        ),
        "per_task": per_task,
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print("saved", args.output.resolve(), "and", csv_path.resolve())


if __name__ == "__main__":
    main()
