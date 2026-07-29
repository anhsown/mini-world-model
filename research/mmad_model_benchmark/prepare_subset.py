from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import requests
from PIL import Image
from remotezip import RemoteZip

from common.mmad import ARCHIVE_URLS, MMAD_JSON_URL, build_subset


ROOT = Path(__file__).resolve().parent


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def materialize_images(manifest: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_archive: dict[str, dict[str, str]] = {}
    for row in manifest["records"]:
        by_archive.setdefault(row["source_archive"], {})[
            row["source_image_path"]
        ] = row["image_file"]
    for archive, members in sorted(by_archive.items()):
        print(f"[{archive}] fetching {len(members)} unique images by HTTP range")
        with RemoteZip(ARCHIVE_URLS[archive]) as remote:
            available = set(remote.namelist())
            for index, (member, relative) in enumerate(sorted(members.items()), start=1):
                destination = output_dir.parent / relative
                if destination.exists():
                    continue
                if member not in available:
                    raise FileNotFoundError(f"{member} not found in {archive}.zip")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with remote.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                with Image.open(destination) as image:
                    image.verify()
                if index % 10 == 0 or index == len(members):
                    print(f"  {index}/{len(members)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--questions-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "mmad.json"
    download(MMAD_JSON_URL, raw_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    manifest = build_subset(raw, args.questions_per_task, args.seed)
    manifest_path = args.output / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.metadata_only:
        materialize_images(manifest, args.output / "images")
    print("manifest:", manifest_path)
    print("sha256:", manifest["manifest_sha256"])
    print("questions:", len(manifest["records"]))


if __name__ == "__main__":
    main()

