from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jwm.tum_rgbd import (TUMRGBDWindowDataset, quaternion_xyzw_to_matrix,
                          validate_tum_dataset)


def _fake_tum(root: Path, frames=8):
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb_lines, depth_lines, pose_lines = [], [], []
    for i in range(frames):
        t = i * 0.033
        Image.fromarray(np.full((24, 32, 3), 80 + i, np.uint8)).save(root / f"rgb/{i}.png")
        Image.fromarray(np.full((24, 32), 5000 + i * 10, np.uint16)).save(root / f"depth/{i}.png")
        rgb_lines.append(f"{t:.3f} rgb/{i}.png")
        depth_lines.append(f"{t + 0.002:.3f} depth/{i}.png")
        pose_lines.append(f"{t + 0.001:.3f} {i * 0.01} 0 0 0 0 0 1")
    (root / "rgb.txt").write_text("\n".join(rgb_lines))
    (root / "depth.txt").write_text("\n".join(depth_lines))
    (root / "groundtruth.txt").write_text("\n".join(pose_lines))


def test_quaternion_identity():
    assert torch.allclose(quaternion_xyzw_to_matrix(torch.tensor([0., 0., 0., 1.])),
                          torch.eye(3))


def test_tum_window_and_validation(tmp_path):
    root = tmp_path / "fr_test"
    _fake_tum(root)
    ds = TUMRGBDWindowDataset([root], frames=4, frame_stride=1,
                              window_stride=2, height=32, width=32)
    row = ds[0]
    assert row["image"].shape == (4, 3, 32, 32)
    assert row["depth"].shape == (4, 32, 32)
    assert torch.allclose(row["pose_c2w"][0], torch.eye(4), atol=1e-6)
    report = validate_tum_dataset(ds)
    assert report["valid"]
