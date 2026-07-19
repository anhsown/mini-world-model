from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jwm.geometry_v2_data import (
    TartanAirWindowDataset,
    load_tartanair_depth,
    normalize_pose_origin,
    procedural_v2_row,
    validate_geometry_source,
    validate_split_disjoint,
)


def _fake_tartan(root: Path, frames: int = 10):
    image_dir = root / "image_lcam_front"
    depth_dir = root / "depth_lcam_front"
    mask_dir = root / "dynamic_mask_lcam_front"
    image_dir.mkdir(parents=True); depth_dir.mkdir(); mask_dir.mkdir()
    poses = []
    for i in range(frames):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[..., 0] = 30 + i * 8
        image[5:15, 4 + i:10 + i, 1] = 220
        Image.fromarray(image).save(image_dir / f"{i:06d}.png")
        np.save(depth_dir / f"{i:06d}.npy",
                np.full((24, 32), 2.0 + i * 0.02, dtype=np.float32))
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[5:15, 4 + i:10 + i] = 255
        Image.fromarray(mask).save(mask_dir / f"{i:06d}.png")
        poses.append(f"{i * 0.01} 0 0 0 0 0 1")
    (root / "pose_lcam_front.txt").write_text("\n".join(poses))


def test_pose_normalization_sets_first_frame_to_identity():
    pose = torch.eye(4).view(1, 4, 4).repeat(3, 1, 1)
    pose[:, 0, 3] = torch.tensor([2.0, 2.1, 2.2])
    normalized = normalize_pose_origin(pose)
    assert torch.allclose(normalized[0], torch.eye(4), atol=1e-6)
    assert torch.allclose(normalized[:, 0, 3], torch.tensor([0.0, 0.1, 0.2]),
                          atol=1e-6)


def test_procedural_v2_row_has_metric_contract():
    row = procedural_v2_row(4, frames=4, size=32)
    assert torch.allclose(row["pose_c2w"][0], torch.eye(4), atol=1e-6)
    assert row["depth_valid"].all()
    assert row["source"] == "procedural_exact"


def test_tartanair_v2_rgba_depth_round_trip(tmp_path):
    expected = np.array([[0.5, 1.25], [3.75, 12.0]], dtype="<f4")
    rgba = expected.reshape(-1).view(np.uint8).reshape(2, 2, 4)
    path = tmp_path / "depth.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    decoded = load_tartanair_depth(path)
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, expected)


def test_tartanair_adapter_and_validator(tmp_path):
    trajectory = tmp_path / "env" / "easy" / "P000"
    _fake_tartan(trajectory)
    dataset = TartanAirWindowDataset([tmp_path], frames=4,
                                     frame_stride=1, window_stride=2,
                                     height=32, width=32)
    assert len(dataset) == 4
    row = dataset[0]
    assert row["image"].shape == (4, 3, 32, 32)
    assert row["depth"].shape == (4, 32, 32)
    assert row["dynamic_mask"].shape == (4, 32, 32)
    assert torch.allclose(row["pose_c2w"][0], torch.eye(4), atol=1e-6)
    report = validate_geometry_source(dataset, "tartanair-test", max_windows=4)
    assert report["valid"]
    assert report["metrics"]["dynamic_label_available"]


def test_split_validator_detects_same_trajectory_leak(tmp_path):
    trajectory = tmp_path / "trajectory"
    _fake_tartan(trajectory)
    train = TartanAirWindowDataset([trajectory], frames=4, height=32, width=32)
    validation = TartanAirWindowDataset([trajectory], frames=4, height=32, width=32)
    report = validate_split_disjoint(train, validation)
    assert not report["valid"]
    assert report["leaked_scene_ids"]


def test_static_tartanair_gets_zero_dynamic_labels(tmp_path):
    trajectory = tmp_path / "static" / "P000"
    _fake_tartan(trajectory)
    for path in (trajectory / "dynamic_mask_lcam_front").glob("*"):
        path.unlink()
    (trajectory / "dynamic_mask_lcam_front").rmdir()
    dataset = TartanAirWindowDataset([trajectory], frames=4,
                                     height=32, width=32,
                                     assume_static=True)
    row = dataset[0]
    assert "dynamic_mask" in row
    assert not row["dynamic_mask"].any()


def test_validator_blocks_implausible_metric_depth_scale(tmp_path):
    trajectory = tmp_path / "bad-depth" / "P000"
    _fake_tartan(trajectory)
    for path in (trajectory / "depth_lcam_front").glob("*.npy"):
        np.save(path, np.full((24, 32), 50_000.0, dtype=np.float32))
    dataset = TartanAirWindowDataset([trajectory], frames=4,
                                     height=32, width=32)
    report = validate_geometry_source(dataset, "bad-scale", max_windows=3)
    assert not report["valid"]
    assert not report["hypotheses"]["H_metric_depth_scale_plausible"]
