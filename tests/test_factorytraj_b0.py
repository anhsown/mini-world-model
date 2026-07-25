import json
from pathlib import Path

from jwm.factorytraj_b0 import rule_baseline, score_b0


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "factorytraj_bench" / "b0_seed_v0.1.json"


def _items():
    return json.loads(DATA.read_text(encoding="utf-8"))["items"]


def test_b0_seed_has_cross_source_holdout_and_no_fake_factorynet_ranges():
    items = _items()
    assert len(items) >= 100
    assert {"train", "validation", "test"} == {item["split"] for item in items}
    assert any("cartesian" in item["input"]["tag_name"] for item in items if item["split"] == "test")
    sources = {item["source_dataset"] for item in items}
    assert {"FactoryNet", "TennesseeEastman"} <= sources
    assert any(source.startswith("OPCFoundation-") for source in sources)
    factorynet = [item for item in items if item["source_dataset"] == "FactoryNet"]
    assert all(item["ground_truth"]["instrument_range"] is None for item in factorynet)
    assert all(item["ground_truth"]["eu_range"] is None for item in factorynet)
    assert any(
        item["ground_truth"]["eu_range"] == {"low": 0.0, "high": 100.0}
        for item in items
        if item["source_dataset"] == "TennesseeEastman"
    )


def test_b0_metrics_report_coverage_and_name_shortcut():
    items = [item for item in _items() if item["split"] != "train"]
    full = score_b0(items, rule_baseline(items))
    anonymous = score_b0(items, rule_baseline(items, anonymized=True))
    assert full["label_coverage"]["authoritative_range"] > 0.0
    assert full["components"]["range_score"] == 0.0
    assert full["components"]["role_macro_f1"] > anonymous["components"]["role_macro_f1"]


def test_b0_missing_predictions_are_rejected():
    items = _items()[:2]
    predictions = rule_baseline(items[:1])
    try:
        score_b0(items, predictions)
    except ValueError as error:
        assert "missing predictions" in str(error)
    else:
        raise AssertionError("missing predictions must fail")
