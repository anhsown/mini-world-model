from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
from remotezip import RemoteZip

from common.mmad import ARCHIVE_URLS, MMAD_JSON_URL, build_full
from prepare_subset import download


ROOT = Path(__file__).resolve().parent


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=300) as response:
        if existing and response.status_code != 206:
            existing = 0
            mode = "wb"
        response.raise_for_status()
        with partial.open(mode) as handle:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def materialize_all_images(
    manifest: dict,
    output: Path,
    cache: Path,
    archives: set[str] | None = None,
    range_download: bool = False,
) -> None:
    by_archive: dict[str, dict[str, str]] = defaultdict(dict)
    for row in manifest["records"]:
        by_archive[row["source_archive"]][row["source_image_path"]] = row["image_file"]

    for archive, members in sorted(by_archive.items()):
        if archives is not None and archive not in archives:
            continue
        missing = {
            member: relative
            for member, relative in members.items()
            if not valid_image(output / relative)
        }
        if not missing:
            print(f"[{archive}] already complete: {len(members)} images")
            continue
        if range_download:
            print(f"[{archive}] HTTP-range fetching {len(missing)} images")
            bundle: RemoteZip | None = None
            try:
                for index, (member, relative) in enumerate(sorted(missing.items()), 1):
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_suffix(destination.suffix + ".partial")
                    for attempt in range(1, 9):
                        try:
                            if bundle is None:
                                # Reopening refreshes Hugging Face's expiring signed CDN URL.
                                bundle = RemoteZip(ARCHIVE_URLS[archive])
                            if member not in bundle.namelist():
                                raise FileNotFoundError(f"{member} not found in {archive}.zip")
                            partial.unlink(missing_ok=True)
                            with bundle.open(member) as source, partial.open("wb") as target:
                                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                            if not valid_image(partial):
                                raise OSError(f"downloaded image failed validation: {member}")
                            partial.replace(destination)
                            break
                        except Exception as error:
                            partial.unlink(missing_ok=True)
                            if bundle is not None:
                                bundle.close()
                                bundle = None
                            if attempt == 8:
                                raise
                            delay = min(2**attempt, 60)
                            print(
                                f"  retry {attempt}/8 for {member}: "
                                f"{type(error).__name__}; waiting {delay}s"
                            )
                            time.sleep(delay)
                    if index % 250 == 0 or index == len(missing):
                        print(f"  fetched {index}/{len(missing)}")
            finally:
                if bundle is not None:
                    bundle.close()
            continue
        archive_path = cache / f"{archive}.zip"
        print(f"[{archive}] downloading archive for {len(missing)} missing images")
        download_archive(ARCHIVE_URLS[archive], archive_path)
        with zipfile.ZipFile(archive_path) as bundle:
            available = set(bundle.namelist())
            absent = sorted(set(missing) - available)
            if absent:
                raise FileNotFoundError(f"{archive}: {len(absent)} members missing; first={absent[0]}")
            for index, (member, relative) in enumerate(sorted(missing.items()), 1):
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                if index % 250 == 0 or index == len(missing):
                    print(f"  extracted {index}/{len(missing)}")
        archive_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data_full")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--archives",
        nargs="+",
        choices=sorted(ARCHIVE_URLS),
        help="Only materialize these source archives (useful for low-disk runtimes).",
    )
    parser.add_argument(
        "--range-download",
        action="store_true",
        help="Fetch individual ZIP members over HTTP ranges instead of storing archives.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.cache or args.output.parent / ".mmad_archive_cache"
    cache.mkdir(parents=True, exist_ok=True)

    raw_path = args.output / "mmad.json"
    download(MMAD_JSON_URL, raw_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    manifest = build_full(raw)
    manifest_path = args.output / "full_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.metadata_only:
        materialize_all_images(
            manifest,
            args.output,
            cache,
            set(args.archives) if args.archives else None,
            args.range_download,
        )
    print("manifest:", manifest_path)
    print("sha256:", manifest["manifest_sha256"])
    print("questions:", len(manifest["records"]))
    print("unique images:", manifest["unique_images"])


if __name__ == "__main__":
    main()
