"""Dataset provenance, resumable downloads and immutable manifests for JWM.

The registry deliberately separates *source availability* from *training
admission*. Downloading an asset never implies that it is valid training data;
the modality-specific admission gates must still pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetAsset:
    id: str
    branch: str
    source: str
    kind: str
    tier: tuple[str, ...]
    split: str
    scene_group: str
    license: str
    url: str
    archive: str
    size_bytes: int = 0
    sha256: str | None = None

    @classmethod
    def from_dict(cls, row: dict) -> "DatasetAsset":
        row = dict(row)
        row["tier"] = tuple(row.get("tier", ()))
        return cls(**row)


def load_registry(path: str | Path) -> tuple[dict, list[DatasetAsset]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, [DatasetAsset.from_dict(row) for row in payload["assets"]]


def select_assets(assets: Iterable[DatasetAsset], tier: str,
                  include: set[str] | None = None,
                  branch: str | None = None) -> list[DatasetAsset]:
    selected = [asset for asset in assets
                if tier in asset.tier and (branch is None or asset.branch == branch)]
    if include:
        selected = [asset for asset in selected if asset.id in include]
    return selected


def validate_registry_split_groups(assets: Iterable[DatasetAsset]) -> dict:
    """Reject a physical scene group assigned to more than one data split."""
    groups: dict[str, set[str]] = {}
    for asset in assets:
        if asset.split in ("train", "validation", "test"):
            groups.setdefault(asset.scene_group, set()).add(asset.split)
    leaked = {group: sorted(splits) for group, splits in groups.items()
              if len(splits) > 1}
    return {"valid": not leaked,
            "hypotheses": {"H_registry_scene_group_no_leak": not leaked},
            "leaked_scene_groups": leaked}


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(base: Path, names: Iterable[str]) -> None:
    root = base.resolve()
    for name in names:
        target = (base / name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"archive path escapes extraction root: {name}")


def safe_extract(archive: str | Path, output: str | Path, kind: str) -> None:
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if kind == "tgz":
        with tarfile.open(archive, "r:gz") as package:
            _safe_members(output, (member.name for member in package.getmembers()))
            # Python 3.10 has no ``filter=`` argument. `_safe_members` above
            # performs the path-traversal check before this extraction.
            package.extractall(output)
    elif kind == "zip":
        with zipfile.ZipFile(archive) as package:
            _safe_members(output, package.namelist())
            package.extractall(output)
    elif kind != "file":
        raise ValueError(f"unsupported archive type: {kind}")


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def download_resumable(url: str, destination: str | Path,
                       expected_bytes: int = 0,
                       timeout: int = 60) -> Path:
    """HTTP Range download using a .part file; safe to restart after shutdown."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and (not expected_bytes or destination.stat().st_size > 0):
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "JWM-Dataset/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # A server may ignore Range. Restart instead of silently duplicating bytes.
        append = offset > 0 and getattr(response, "status", None) == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        response_length = int(response.headers.get("Content-Length", "0") or 0)
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=8 << 20)
    if response_length and partial.stat().st_size < offset + response_length:
        raise IOError("HTTP body ended before Content-Length")
    # Published sizes are usually rounded decimal GB/MB. Treat them only as
    # quota estimates; archive parsing and optional SHA256 establish integrity.
    if expected_bytes and partial.stat().st_size < expected_bytes * 0.50:
        raise IOError(f"download implausibly small {partial.stat().st_size} < {expected_bytes}")
    os.replace(partial, destination)
    return destination


def append_manifest(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def archive_is_valid(path: str | Path, kind: str) -> bool:
    path = Path(path)
    try:
        if kind == "tgz":
            with tarfile.open(path, "r:gz") as package:
                # Reading all members validates the gzip/tar footer without
                # loading payloads into memory.
                package.getmembers()
        elif kind == "zip":
            with zipfile.ZipFile(path) as package:
                return package.testzip() is None
        else:
            return path.is_file() and path.stat().st_size > 0
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False
    return True


def materialize_asset(asset: DatasetAsset, root: str | Path,
                      reserve_free_gb: float = 25.0,
                      extract: bool = True) -> dict:
    root = Path(root)
    downloads, extracted = root / "downloads", root / "raw" / asset.source
    downloads.mkdir(parents=True, exist_ok=True)
    suffix = {"tgz": ".tgz", "zip": ".zip", "file": Path(asset.url).suffix}.get(
        asset.archive, ".bin")
    archive_path = downloads / f"{asset.id}{suffix}"
    partial = archive_path.with_suffix(archive_path.suffix + ".part")
    output = extracted / asset.id
    marker = output / ".jwm_extracted.json"
    # A verified extraction is the durable completion record. This permits a
    # low-disk environment to delete the compressed archive and safely resume.
    if extract and marker.exists():
        completed = json.loads(marker.read_text(encoding="utf-8"))
        return {
            "asset_id": asset.id, "source": asset.source, "branch": asset.branch,
            "kind": asset.kind, "split": asset.split,
            "scene_group": asset.scene_group, "license": asset.license,
            "url": asset.url,
            "archive": str(archive_path) if archive_path.exists() else None,
            "extracted": str(output),
            "bytes": archive_path.stat().st_size if archive_path.exists() else 0,
            "sha256": completed.get("sha256"), "elapsed_s": 0.0,
            "status": "already_ready",
        }
    # A prior HTTP transfer may have completed before a rounded-size check
    # rejected it. Recover it without another network request.
    if not archive_path.exists() and partial.exists() and archive_is_valid(partial, asset.archive):
        os.replace(partial, archive_path)
    required = max(asset.size_bytes, 1) + int(reserve_free_gb * (1 << 30))
    if _free_bytes(root) < required:
        raise OSError(f"insufficient free disk for {asset.id}; reserve={reserve_free_gb}GB")
    started = time.time()
    download_resumable(asset.url, archive_path, asset.size_bytes)
    if not archive_is_valid(archive_path, asset.archive):
        raise IOError(f"archive integrity check failed for {asset.id}")
    digest = sha256_file(archive_path)
    if asset.sha256 and digest.lower() != asset.sha256.lower():
        raise IOError(f"SHA256 mismatch for {asset.id}")
    if extract and asset.archive != "file" and not marker.exists():
        safe_extract(archive_path, output, asset.archive)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"asset": asset.id, "sha256": digest}),
                          encoding="utf-8")
    return {
        "asset_id": asset.id, "source": asset.source, "branch": asset.branch,
        "kind": asset.kind, "split": asset.split, "scene_group": asset.scene_group,
        "license": asset.license, "url": asset.url, "archive": str(archive_path),
        "extracted": str(output) if extract and asset.archive != "file" else None,
        "bytes": archive_path.stat().st_size, "sha256": digest,
        "elapsed_s": round(time.time() - started, 3), "status": "ready",
    }
