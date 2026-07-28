import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2


TASK_RE = re.compile(r"_task_(\d+)\.mp4$", re.IGNORECASE)


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"valid": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "valid": fps > 0 and frames > 0,
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps if fps else None,
        "width": width,
        "height": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit.json"))
    args = parser.parse_args()

    videos = sorted(args.dataset.rglob("*.mp4"))
    task_counts = Counter()
    rows = []
    for video in videos:
        match = TASK_RE.search(video.name)
        if match:
            task_counts[int(match.group(1))] += 1
        rows.append({"path": str(video), **probe(video)})

    valid_rows = [row for row in rows if row["valid"]]
    report = {
        "dataset_root": str(args.dataset.resolve()),
        "video_count": len(videos),
        "valid_video_count": len(valid_rows),
        "task_counts": dict(sorted(task_counts.items())),
        "duration_seconds": {
            "min": min((r["duration_seconds"] for r in valid_rows), default=None),
            "max": max((r["duration_seconds"] for r in valid_rows), default=None),
            "mean": (
                sum(r["duration_seconds"] for r in valid_rows) / len(valid_rows)
                if valid_rows else None
            ),
        },
        "samples": rows[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print("saved", args.output.resolve())


if __name__ == "__main__":
    main()

