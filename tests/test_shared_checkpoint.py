import json
import subprocess

from research.mmad_model_benchmark.common.shared_checkpoint import SharedCheckpointStore


def test_shared_checkpoint_deduplicates_and_retries_failures(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    store = SharedCheckpointStore(
        tmp_path,
        "manifest-a",
        "kaggle_t4x2",
        push_every=2,
    )
    assert store.record(
        {"sample_id": "mmad_full_00001", "status": "ok", "manifest_sha256": "manifest-a"},
        push=False,
    )
    assert not store.record(
        {"sample_id": "mmad_full_00002", "status": "error", "manifest_sha256": "manifest-a"},
        push=False,
    )
    assert not store.record(
        {"sample_id": "mmad_full_00001", "status": "ok", "manifest_sha256": "manifest-a"},
        push=False,
    )
    shard = store.flush(push=False)
    assert shard is not None
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in rows] == ["mmad_full_00001"]

    resumed = SharedCheckpointStore(tmp_path, "manifest-a", "nvidia_build")
    assert resumed.completed_ids == {"mmad_full_00001"}
    assert "mmad_full_00002" not in resumed.completed_ids


def test_shared_checkpoint_rejects_other_manifest(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    foreign = tmp_path / "research/mmad_model_benchmark/checkpoints/cosmos3_mmad/foreign"
    foreign.mkdir(parents=True)
    (foreign / "part.jsonl").write_text(
        json.dumps(
            {"sample_id": "mmad_full_00003", "status": "ok", "manifest_sha256": "manifest-b"}
        )
        + "\n",
        encoding="utf-8",
    )
    store = SharedCheckpointStore(tmp_path, "manifest-a", "kaggle_t4x2")
    assert not store.completed_ids


def test_shared_checkpoint_honours_custom_relative_root(tmp_path):
    """PIADE and other benchmarks must not land in the MMAD checkpoint tree."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    store = SharedCheckpointStore(
        tmp_path,
        "episodes-a",
        "kaggle_t4_text_direct",
        push_every=50,  # keep flushing explicit; push_every=1 auto-flushes in record()
        relative_root="research/piade_b0_b5/checkpoints",
    )
    assert store.record(
        {"sample_id": "text|direct|piade_test_M1_00012", "status": "ok",
         "manifest_sha256": "episodes-a"},
        push=False,
    )
    shard = store.flush(push=False)
    assert shard is not None
    assert "piade_b0_b5" in shard.as_posix()
    assert "mmad_model_benchmark" not in shard.as_posix()

    resumed = SharedCheckpointStore(
        tmp_path, "episodes-a", "other_backend",
        relative_root="research/piade_b0_b5/checkpoints",
    )
    assert resumed.completed_ids == {"text|direct|piade_test_M1_00012"}

    # The MMAD default root must not see PIADE shards.
    mmad = SharedCheckpointStore(tmp_path, "episodes-a", "kaggle_t4x2")
    assert not mmad.completed_ids


def test_shared_checkpoint_pushes_immutable_shard(tmp_path):
    remote = tmp_path / "remote.git"
    writer = tmp_path / "writer"
    reader = tmp_path / "reader"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(remote), str(writer)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(writer), "switch", "-c", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(writer), "config", "user.name", "anhsown"], check=True)
    subprocess.run(
        ["git", "-C", str(writer), "config", "user.email", "92385316+anhsown@users.noreply.github.com"],
        check=True,
    )
    (writer / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(writer), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(writer), "commit", "-m", "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(writer), "push", "-u", "origin", "main"], check=True, capture_output=True)

    store = SharedCheckpointStore(writer, "manifest-a", "kaggle_t4x2", push_every=1)
    assert store.record(
        {"sample_id": "mmad_full_00004", "status": "ok", "manifest_sha256": "manifest-a"}
    )
    subprocess.run(["git", "clone", "--branch", "main", str(remote), str(reader)], check=True, capture_output=True)
    shards = list(
        (reader / "research/mmad_model_benchmark/checkpoints/cosmos3_mmad/kaggle_t4x2").glob("*.jsonl")
    )
    assert len(shards) == 1
