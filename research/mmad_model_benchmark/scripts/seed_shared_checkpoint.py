from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.shared_checkpoint import SharedCheckpointStore  # noqa: E402


def load_latest_success(paths: list[Path], manifest_sha256: str) -> list[dict]:
    latest: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok" or not row.get("sample_id"):
                continue
            row_hash = row.get("manifest_sha256")
            if row_hash and row_hash != manifest_sha256:
                raise ValueError(f"Manifest mismatch in {path}: {row_hash}")
            latest[row["sample_id"]] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed immutable shared Cosmos MMAD shards")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--nvidia", type=Path, action="append", default=[])
    parser.add_argument("--kaggle", type=Path, action="append", default=[])
    parser.add_argument("--shard-size", type=int, default=50)
    args = parser.parse_args()

    total = 0
    for backend, paths in (("nvidia_build", args.nvidia), ("kaggle_t4x2", args.kaggle)):
        rows = load_latest_success(paths, args.manifest_sha256)
        store = SharedCheckpointStore(
            args.repo,
            args.manifest_sha256,
            backend,
            push_every=args.shard_size,
        )
        accepted = store.seed(rows, push=False)
        total += accepted
        print(
            json.dumps(
                {
                    "backend": backend,
                    "source_success_rows": len(rows),
                    "new_shared_rows": accepted,
                    "shared_completed_total": len(store.completed_ids),
                },
                indent=2,
            )
        )
    print(f"Seed complete: {total} new successful rows")


if __name__ == "__main__":
    main()
