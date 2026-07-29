from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.mmad import evaluate_records, load_jsonl, write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary, rows = evaluate_records(manifest, load_jsonl(args.predictions))
    write_evaluation(args.output, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

