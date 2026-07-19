"""Admission tests for the procedural Eye Physical geometry distribution."""

import torch

from jwm.geometry_data import (
    ProceduralGeometryDataset,
    render_geometry_sequence,
    validate_geometry_dataset,
)


def test_renderer_is_deterministic_and_seed_sensitive():
    a = render_geometry_sequence(42, frames=3, height=32, width=40)
    b = render_geometry_sequence(42, frames=3, height=32, width=40)
    c = render_geometry_sequence(43, frames=3, height=32, width=40)
    assert torch.equal(a.image, b.image)
    assert torch.equal(a.depth, b.depth)
    assert a.scene_id == b.scene_id and a.scene_id != c.scene_id
    assert not torch.equal(a.image, c.image)


def test_dataset_split_ids_are_disjoint():
    train = ProceduralGeometryDataset("train", 2, 3, 32, 32)
    val = ProceduralGeometryDataset("val", 2, 3, 32, 32)
    assert {train[i]["scene_id"] for i in range(2)}.isdisjoint(
        {val[i]["scene_id"] for i in range(2)})


def test_validator_accepts_exact_geometry_distribution():
    report = validate_geometry_dataset(samples_per_split=1, frames=3,
                                       height=32, width=32)
    assert report["valid"]
    assert all(report["hypotheses"].values())
