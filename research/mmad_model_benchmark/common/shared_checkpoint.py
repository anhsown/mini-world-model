from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_URL = "https://github.com/anhsown/mini-world-model.git"
CHECKPOINT_RELATIVE_ROOT = Path(
    "research/mmad_model_benchmark/checkpoints/cosmos3_mmad"
)


def _run_git(
    repo: Path,
    args: list[str],
    *,
    token: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    command = ["git", "-C", str(repo)]
    if token:
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        command += [
            "-c",
            "http.https://github.com/.extraheader="
            f"AUTHORIZATION: basic {encoded}",
        ]
    command += args
    return subprocess.run(
        command,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ensure_checkpoint_repo(
    repo: Path,
    repo_url: str = DEFAULT_REPO_URL,
) -> Path:
    repo = repo.resolve()
    if (repo / ".git").exists():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo)],
        check=True,
    )
    return repo


def discover_checkpoint_repo(code_repo: Path, cache_root: Path | None = None) -> Path:
    configured = os.environ.get("MMAD_CHECKPOINT_REPO")
    if configured:
        return ensure_checkpoint_repo(Path(configured))
    code_repo = code_repo.resolve()
    if (code_repo / ".git").exists():
        return code_repo
    sibling = code_repo.parent / "mini-world-model"
    if (sibling / ".git").exists():
        return sibling
    cache_root = cache_root or code_repo / ".checkpoint_sync"
    return ensure_checkpoint_repo(cache_root / "mini-world-model")


class SharedCheckpointStore:
    """Immutable Git shards shared by NVIDIA Build and Kaggle runners.

    Only successful, parseable records are shared. Local output files continue
    to retain failures for reliability analysis, while failed records remain
    eligible for retry on either backend.
    """

    def __init__(
        self,
        repo: Path,
        manifest_sha256: str,
        backend: str,
        *,
        push_every: int = 50,
        token: str | None = None,
        branch: str = "main",
        relative_root: Path | str | None = None,
    ) -> None:
        self.repo = ensure_checkpoint_repo(repo)
        self.manifest_sha256 = manifest_sha256
        self.backend = backend.replace(" ", "_").replace("/", "_").lower()
        self.push_every = max(1, int(push_every))
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.branch = branch
        # Benchmarks other than MMAD pass their own root so their shards do not
        # land in the MMAD checkpoint tree.
        self.root = self.repo / (relative_root or CHECKPOINT_RELATIVE_ROOT)
        self.backend_root = self.root / self.backend
        self.backend_root.mkdir(parents=True, exist_ok=True)
        self._known: set[str] = set()
        self._pending: dict[str, dict] = {}
        self.refresh()

    def sync_from_remote(self) -> None:
        result = _run_git(
            self.repo,
            ["pull", "--ff-only", "origin", self.branch],
            check=False,
        )
        if result.returncode:
            print("Shared checkpoint pull warning:", result.stdout.strip())
        self.refresh()

    def _iter_rows(self) -> Iterable[dict]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_hash = row.get("manifest_sha256")
                if row_hash and row_hash != self.manifest_sha256:
                    continue
                yield row

    def refresh(self) -> None:
        self._known = {
            row["sample_id"]
            for row in self._iter_rows()
            if row.get("sample_id") and row.get("status") == "ok"
        }

    @property
    def completed_ids(self) -> set[str]:
        return set(self._known) | set(self._pending)

    def successful_rows(self) -> list[dict]:
        latest: dict[str, dict] = {}
        for row in self._iter_rows():
            if row.get("status") == "ok" and row.get("sample_id"):
                latest[row["sample_id"]] = row
        latest.update(self._pending)
        return [latest[key] for key in sorted(latest)]

    def record(self, row: dict, *, push: bool = True) -> bool:
        sample_id = row.get("sample_id")
        if not sample_id or row.get("status") != "ok" or sample_id in self.completed_ids:
            return False
        item = dict(row)
        item.setdefault("manifest_sha256", self.manifest_sha256)
        item["shared_checkpoint_backend"] = self.backend
        self._pending[sample_id] = item
        if len(self._pending) >= self.push_every:
            self.flush(push=push)
        return True

    def seed(self, rows: Iterable[dict], *, push: bool = False) -> int:
        accepted = 0
        for row in rows:
            accepted += int(self.record(row, push=False))
        if self._pending:
            self.flush(push=push)
        return accepted

    def flush(self, *, push: bool = True) -> Path | None:
        if not self._pending:
            return None
        now = datetime.now(timezone.utc)
        name = (
            f"{now:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}_"
            f"{len(self._pending):04d}.jsonl"
        )
        path = self.backend_root / name
        rows = [self._pending[key] for key in sorted(self._pending)]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        self._known.update(self._pending)
        self._pending.clear()
        print(f"Shared checkpoint shard saved: {path} ({len(rows)} records)")
        if push:
            self._commit_and_push(path)
        return path

    def _commit_and_push(self, path: Path) -> bool:
        relative = path.relative_to(self.repo)
        _run_git(self.repo, ["config", "user.name", "anhsown"])
        _run_git(
            self.repo,
            ["config", "user.email", "92385316+anhsown@users.noreply.github.com"],
        )
        _run_git(self.repo, ["add", "--", relative.as_posix()])
        commit = _run_git(
            self.repo,
            ["commit", "-m", f"Checkpoint {self.backend}: {path.stem}"],
            check=False,
        )
        if commit.returncode:
            print("Shared checkpoint commit warning:", commit.stdout.strip())
            return False
        for attempt in range(1, 4):
            pull = _run_git(
                self.repo,
                ["pull", "--rebase", "origin", self.branch],
                token=self.token,
                check=False,
            )
            if pull.returncode:
                print(f"Shared checkpoint rebase attempt {attempt} failed")
                time.sleep(attempt * 2)
                continue
            push = _run_git(
                self.repo,
                ["push", "origin", f"HEAD:{self.branch}"],
                token=self.token,
                check=False,
            )
            if push.returncode == 0:
                print(f"Shared checkpoint pushed: {relative.as_posix()}")
                return True
            print(f"Shared checkpoint push attempt {attempt} failed")
            time.sleep(attempt * 2)
        print(f"Shard remains local and can be pushed later: {path}")
        return False
