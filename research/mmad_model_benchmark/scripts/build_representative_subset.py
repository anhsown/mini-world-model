from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


SOURCES = ("GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA")
TASKS = (
    "Anomaly Detection",
    "Defect Analysis",
    "Defect Classification",
    "Defect Description",
    "Defect Localization",
    "Object Analysis",
    "Object Classification",
)


def _distribution(rows: list[dict], key: str, weight_key: str | None = None) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    for row in rows:
        value = str(row[key])
        counts[value] += float(row.get(weight_key, 1.0)) if weight_key else 1.0
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    score = 0.0
    for key in keys:
        p, q = left.get(key, 0.0), right.get(key, 0.0)
        m = (p + q) / 2
        if p:
            score += 0.5 * p * math.log2(p / m)
        if q:
            score += 0.5 * q * math.log2(q / m)
    return score


def _draw_candidate(records: list[dict], per_stratum: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["source_dataset"], row["question_type"])].append(row)

    selected: list[dict] = []
    used_images: set[str] = set()
    for source in SOURCES:
        for task in TASKS:
            group = list(groups[(source, task)])
            rng.shuffle(group)
            unique = []
            seen_here: set[str] = set()
            for row in group:
                image = row["image_file"]
                if image in used_images or image in seen_here:
                    continue
                unique.append(row)
                seen_here.add(image)
            assert len(unique) >= per_stratum, (source, task, len(unique))
            chosen = unique[:per_stratum]
            selected.extend(chosen)
            used_images.update(row["image_file"] for row in chosen)
    return selected


def _score_candidate(full: list[dict], sample: list[dict]) -> float:
    score = 0.0
    for key in ("category", "is_normal", "answer"):
        score += _js_divergence(_distribution(full, key), _distribution(sample, key))
    missing_categories = len(set(row["category"] for row in full) - set(row["category"] for row in sample))
    return score + missing_categories * 10


def build_subset(manifest: dict, per_stratum: int = 50, seed: int = 20260803,
                 candidate_seeds: int = 128) -> dict:
    records = manifest["records"]
    expected = len(SOURCES) * len(TASKS) * per_stratum
    best: tuple[float, int, list[dict]] | None = None
    for offset in range(candidate_seeds):
        candidate_seed = seed + offset
        sample = _draw_candidate(records, per_stratum, candidate_seed)
        score = _score_candidate(records, sample)
        if best is None or score < best[0]:
            best = score, candidate_seed, sample
    assert best is not None
    _, selected_seed, selected = best

    population = Counter((row["source_dataset"], row["question_type"]) for row in records)
    for row in selected:
        stratum = (row["source_dataset"], row["question_type"])
        row["sample_weight"] = population[stratum] / per_stratum

    selected.sort(key=lambda row: row["sample_id"])
    selected_ids = [row["sample_id"] for row in selected]
    subset_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    strata = Counter((row["source_dataset"], row["question_type"]) for row in selected)
    categories = {row["category"] for row in selected}
    images = {row["image_file"] for row in selected}

    weighted_js = {
        key: _js_divergence(
            _distribution(records, key), _distribution(selected, key, "sample_weight")
        )
        for key in ("source_dataset", "question_type", "category", "is_normal", "answer")
    }
    validation = {
        "valid": (
            len(selected) == expected
            and len(set(selected_ids)) == expected
            and len(images) == expected
            and len(strata) == len(SOURCES) * len(TASKS)
            and all(count == per_stratum for count in strata.values())
            and len(categories) == len({row["category"] for row in records})
            and max(weighted_js.values()) < 0.05
        ),
        "records": len(selected),
        "unique_images": len(images),
        "source_task_strata": len(strata),
        "categories_covered": len(categories),
        "categories_total": len({row["category"] for row in records}),
        "per_stratum": per_stratum,
        "weighted_js_divergence": weighted_js,
    }
    return {
        "benchmark": "MMAD-Representative",
        "version": "v1",
        "selection": "balanced source x task, globally unique images, post-stratification weights",
        "seed": selected_seed,
        "candidate_seeds": candidate_seeds,
        "parent_manifest_sha256": manifest["manifest_sha256"],
        "subset_sha256": subset_hash,
        "validation": validation,
        "records": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--candidate-seeds", type=int, default=128)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    subset = build_subset(manifest, args.per_stratum, args.seed, args.candidate_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in subset.items() if key != "records"}, indent=2))
    if not subset["validation"]["valid"]:
        raise SystemExit("Representative subset failed validation")


if __name__ == "__main__":
    main()
