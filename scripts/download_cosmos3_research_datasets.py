"""Download a safe, auditable Cosmos 3 dataset research bundle.

The default mode downloads cards, repository manifests, licenses and selected
small annotation/index files. It intentionally does not materialize multi-TB
video corpora on a workstation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "dataset_registry.json"
OUT = ROOT / "research" / "datasets"
HF_API = "https://huggingface.co/api/datasets/{repo}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true"

# Explicit allowlist: no accidental video shards or multi-GB training JSON.
ALLOW = {
    "nvidia/Cosmos-HumanEval-v1": [
        "README.md", "hue-v1p2-i2v-public.json", "hue-v1p2-t2v-public.json"
    ],
    "nvidia/PhysicalAI-Traffic-Anomaly-Reasoning": [
        "README.md", "DOWNLOADING.md", "download_videos.py", "stitch_tad_frames.py",
        "test/clip_manifest.csv", "test/download_test_videos.py", "test/evaluate.py",
        "test/requirements.txt", "test/submission.example.csv", "test/test.json",
        "train/bcq.json", "train/bcq_openended.json", "train/causal_linkage.json",
        "train/mcq.json", "train/mcq_openended.json", "train/open_qa.json",
        "train/scene_description.json", "train/temporal_description.json",
        "train/temporal_localization.json", "train/video_summarization.json"
    ],
    "nvidia/PhysicalAI-Spatial-Intelligence-Warehouse": ["README.md"],
    "nvidia/Cosmos3-DROID": ["README.md"],
    "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes": ["README.md"],
    "nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes": [
        "README.md", "manifest_public.jsonl",
        "PhysicalAI-Cosmos-SDG-RobotSim-preview-samples.json"
    ],
    "nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios": ["README.md"],
    "nvidia/PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes": ["README.md"],
    "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes": [
        "README.md", "metadata/runs.parquet", "metadata/clips.parquet"
    ],
    "IPEC-COMMUNITY/EO-Data1.5M": ["README.md"],
    "nexar-ai/nexar_collision_prediction": ["README.md", "LICENSE"]
}


def get_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jwm-dataset-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def safe_name(repo: str) -> str:
    return repo.replace("/", "__")


def available_files(api: dict) -> dict[str, int | None]:
    files = {}
    for sibling in api.get("siblings", []):
        name = sibling.get("rfilename")
        if not name:
            continue
        size = sibling.get("size")
        lfs = sibling.get("lfs") or {}
        files[name] = size if size is not None else lfs.get("size")
    return files


def download_repo(repo: str, max_file_mb: float) -> dict:
    target = OUT / safe_name(repo)
    target.mkdir(parents=True, exist_ok=True)
    record = {"repo": repo, "status": "ok", "downloaded": [], "skipped": [], "errors": []}
    try:
        api_bytes = get_bytes(HF_API.format(repo=repo))
        (target / "hf_api.json").write_bytes(api_bytes)
        api = json.loads(api_bytes)
        files = available_files(api)
    except Exception as exc:
        record["status"] = "api-error"
        record["errors"].append(str(exc))
        return record

    (target / "file_inventory.json").write_text(
        json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for remote in ALLOW.get(repo, ["README.md"]):
        size = files.get(remote)
        if size is None and remote not in files:
            record["skipped"].append({"file": remote, "reason": "not listed or gated"})
            continue
        if size is not None and size > max_file_mb * 1024 * 1024:
            record["skipped"].append({"file": remote, "reason": f"larger than {max_file_mb} MiB", "bytes": size})
            continue
        local = target / remote
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            local.write_bytes(get_bytes(HF_RESOLVE.format(repo=repo, path=remote), timeout=300))
            record["downloaded"].append({"file": remote, "bytes": local.stat().st_size})
        except urllib.error.HTTPError as exc:
            record["errors"].append({"file": remote, "http": exc.code})
        except Exception as exc:
            record["errors"].append({"file": remote, "error": str(exc)})
    if record["errors"]:
        record["status"] = "partial"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-mb", type=float, default=128.0)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    args = parser.parse_args()

    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gb < args.min_free_gb:
        raise OSError(f"Only {free_gb:.2f} GB free; minimum is {args.min_free_gb:.2f} GB")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []
    for item in registry["huggingface"]:
        print(f"metadata: {item['id']}", flush=True)
        records.append(download_repo(item["id"], args.max_file_mb))

    for item in registry["external_official"]:
        d = OUT / "external_official" / item["name"].replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    status = {
        "mode": "metadata-first",
        "started_free_gb": round(free_gb, 3),
        "finished_free_gb": round(shutil.disk_usage(ROOT).free / (1024 ** 3), 3),
        "elapsed_seconds": round(time.time() - started, 2),
        "records": records,
    }
    (OUT / "download_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
