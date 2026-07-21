import torch

from jwm.geometry_v3_data import (
    make_counterfactuals, procedural_v3_row, stack_geometry_v3_rows,
    validate_geometry_v3_source,
)


class TinyProcedural:
    def __len__(self): return 3
    def __getitem__(self, index): return procedural_v3_row(index, 4, 32)
    def scene_ids(self): return {f"s{i}" for i in range(3)}


def test_v3_row_and_stack_preserve_calibration_and_flow():
    row = procedural_v3_row(7, 4, 32)
    assert row["intrinsics"].shape == (4, 3, 3)
    assert row["rigid_flow"].shape == (3, 32, 32, 2)
    assert row["projection_y_sign"].item() == -1
    batch = stack_geometry_v3_rows([row, procedural_v3_row(8, 4, 32)])
    assert batch["intrinsics"].shape == (2, 4, 3, 3)
    assert batch["timestamp"].shape == (2, 4)


def test_counterfactual_intrinsics_are_actually_different():
    batch = stack_geometry_v3_rows([procedural_v3_row(9, 3, 32)])
    controls = make_counterfactuals(batch)
    assert not torch.allclose(controls["wrong_intrinsics"], batch["intrinsics"])
    assert torch.equal(controls["reverse_image"], batch["image"].flip(1))


def test_single_sample_wrong_window_breaks_local_chronology_not_circular_shift():
    batch = stack_geometry_v3_rows([procedural_v3_row(19, 6, 32)])
    images = batch["image"]
    wrong = make_counterfactuals(batch)["wrong_window_image"]
    expected = images[:, [0, 2, 4, 1, 3, 5]]
    assert torch.equal(wrong, expected)
    assert not torch.equal(wrong, images.roll(1, 1))


def test_procedural_v3_admission_passes_all_hypotheses():
    report = validate_geometry_v3_source(TinyProcedural(), "procedural")
    assert report["valid"], report
