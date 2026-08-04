import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBSET = (
    ROOT / "research" / "mmad_model_benchmark" / "data_subsets"
    / "mmad_representative_1400_v1.json"
)


def test_representative_subset_contract():
    data = json.loads(SUBSET.read_text(encoding="utf-8"))
    rows = data["records"]
    strata = Counter((row["source_dataset"], row["question_type"]) for row in rows)

    assert data["validation"]["valid"] is True
    assert len(rows) == 1400
    assert len({row["sample_id"] for row in rows}) == 1400
    assert len({row["image_file"] for row in rows}) == 1400
    assert len(strata) == 28
    assert set(strata.values()) == {50}
    assert len({row["category"] for row in rows}) == 38
    assert all(row["sample_weight"] > 0 for row in rows)
    assert max(data["validation"]["weighted_js_divergence"].values()) < 0.05
