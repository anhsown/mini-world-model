from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
SOURCE = HERE / "Cosmos3_Nano_MMAD_T4x2_UI_Parity.ipynb"
SUBSET = BENCH / "data_subsets" / "mmad_representative_1400_v1.json"
OUTPUT = HERE / "Cosmos3_Nano_MMAD_Representative1400_Optimized_T4x2.ipynb"


def main() -> None:
    nb = nbf.read(SOURCE, as_version=4)
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    ids = [row["sample_id"] for row in subset["records"]]
    payload = base64.b64encode(zlib.compress("\n".join(ids).encode(), 9)).decode()

    nb.cells[0].source = """# Cosmos 3 Nano — MMAD Representative-1400 Optimized (Kaggle T4×2)

This notebook runs a statistically locked **1,400-question MMAD subset**:

- 50 records for every `4 source datasets × 7 question types` stratum;
- 1,400 unique images and all 38 categories;
- post-stratification weights for population-level metrics;
- deterministic BNB8 decoding with a 256-token reasoning budget;
- early stop immediately after `</think>` plus an A/B/C/D answer;
- one concise 128-token retry for non-parseable output;
- isolated GitHub checkpoint namespace, so full-run checkpoints cannot contaminate this experiment.

Run the install cell once, restart the Kaggle session once, then run from the environment cell onward.
"""
    nb.cells[2].source = nb.cells[2].source.replace(
        "OUT = WORK / 'cosmos3_mmad_t4x2'\nSMOKE_OUT = OUT / 'smoke_1_5.jsonl'\nFULL_OUT = OUT / 'predictions_from_230.jsonl'",
        "OUT = WORK / 'cosmos3_mmad_rep1400_opt256'\n"
        "SMOKE_OUT = OUT / 'smoke_5.jsonl'\n"
        "FULL_OUT = OUT / 'predictions_rep1400.jsonl'",
    )

    nb.cells[4].source = f"""# Build canonical metadata, then select the immutable representative subset.
import base64, zlib
from collections import Counter

subprocess.run([
    sys.executable, str(BASE / 'prepare_full.py'),
    '--output', str(DATA), '--metadata-only'
], cwd=BASE, check=True)

from prepare_full import materialize_all_images
full_manifest = json.loads((DATA / 'full_manifest.json').read_text(encoding='utf-8'))
assert len(full_manifest['records']) == 39670

LOCKED_IDS_ZLIB_B64 = '{payload}'
locked_ids = zlib.decompress(base64.b64decode(LOCKED_IDS_ZLIB_B64)).decode().splitlines()
lookup = {{row['sample_id']: row for row in full_manifest['records']}}
records = [dict(lookup[sid]) for sid in locked_ids]
population = Counter((r['source_dataset'], r['question_type']) for r in full_manifest['records'])
sample_counts = Counter((r['source_dataset'], r['question_type']) for r in records)
for row in records:
    key = (row['source_dataset'], row['question_type'])
    row['sample_weight'] = population[key] / sample_counts[key]

manifest = {{
    'benchmark': 'MMAD-Representative',
    'setting': 'locked1400_opt256_reasoning',
    'manifest_sha256': '{subset['subset_sha256']}',
    'parent_manifest_sha256': full_manifest['manifest_sha256'],
    'records': records,
}}
record_number = {{row['sample_id']: i for i, row in enumerate(records, 1)}}
assert len(records) == 1400 and len({{r['image_file'] for r in records}}) == 1400
assert len(sample_counts) == 28 and set(sample_counts.values()) == {{50}}
assert len({{r['category'] for r in records}}) == 38
print('parent manifest:', full_manifest['manifest_sha256'])
print('locked subset:', manifest['manifest_sha256'])
print('questions:', len(records), 'unique images:', len({{r['image_file'] for r in records}}))
print('Validation: 28/28 strata, 38/38 categories, 1,400/1,400 unique images')
"""

    nb.cells[6].source = nb.cells[6].source.replace(
        "MAX_NEW_TOKENS = 512  # enough room for UI-style reasoning; old notebook used only 32",
        "MAX_NEW_TOKENS = 256\nMAX_RETRY_TOKENS = 128",
    ).replace(
        "assert free_gib >= 11, 'Need at least 11 GiB free for the Reasoner-only checkpoint.'",
        "assert free_gib >= 8, 'Need at least 8 GiB free for the Reasoner-only checkpoint.'",
    )

    source = nb.cells[7].source
    source = source.replace(
        "from common.shared_checkpoint import SharedCheckpointStore\nimport re",
        "from common.shared_checkpoint import SharedCheckpointStore\n"
        "from transformers import StoppingCriteria, StoppingCriteriaList\nimport re",
    )
    source = source.replace(
        "input_device = model.device",
        """class StopAfterThinkAnswer(StoppingCriteria):
    def __init__(self, tokenizer, prompt_tokens):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return bool(re.search(r'</think>\\s*[A-D](?:\\s|$)', text, re.I | re.S))

input_device = model.device""",
    )
    source = source.replace(
        "def infer_one(sample):",
        "def infer_one(sample, max_new_tokens=MAX_NEW_TOKENS, concise=False):",
    )
    source = source.replace(
        "{'type': 'text', 'text': sample['prompt']},",
        "{'type': 'text', 'text': sample['prompt'] + (\n"
        "    '\\nKeep the reasoning under 100 words and answer immediately.' if concise else '')},",
    )
    old_generate = """generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )"""
    new_generate = """generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([
                StopAfterThinkAnswer(processor.tokenizer, inputs.input_ids.shape[1])
            ]),
        )"""
    assert old_generate in source
    source = source.replace(old_generate, new_generate)
    old_call = """raw, normalized, parts, pred, latency = infer_one(sample)
            status, error = ('ok' if pred else 'parse_failure'), None"""
    new_call = """raw, normalized, parts, pred, latency = infer_one(sample)
            attempts = 1
            if not pred:
                raw2, normalized2, parts2, pred2, latency2 = infer_one(
                    sample, max_new_tokens=MAX_RETRY_TOKENS, concise=True
                )
                attempts = 2
                latency += latency2
                if pred2:
                    raw, normalized, parts, pred = raw2, normalized2, parts2, pred2
            status, error = ('ok' if pred else 'parse_failure'), None"""
    assert old_call in source
    source = source.replace(old_call, new_call)
    source = source.replace(
        "pred, latency, status, error = None, 0.0, 'error', f'{type(exc).__name__}: {exc}'",
        "pred, latency, attempts, status, error = None, 0.0, 1, 'error', f'{type(exc).__name__}: {exc}'",
    )
    source = source.replace(
        "'parse_format': parts['parse_format'],\n            'latency_seconds'",
        "'parse_format': parts['parse_format'],\n            'attempts': attempts,\n"
        "            'generation_budget': MAX_NEW_TOKENS,\n            'latency_seconds'",
    )
    source = source.replace(
        "'backend': 'Transformers/device_map=auto/T4x2',",
        "'backend': 'Transformers/device_map=auto/T4x2/opt256/deterministic',",
    )
    source = source.replace(
        "'precision': 'community BNB8 reasoner-only',",
        "'precision': 'community BNB8 reasoner-only; optimized deterministic decoding',",
    )
    nb.cells[7].source = source

    nb.cells[8].source = nb.cells[8].source.replace(
        "# PHASE 1 � run only MMAD questions 1 through 5.",
        "# PHASE 1 — run five locked representative smoke samples.",
    ).replace("smoke_records = records[:5]", "smoke_records = records[:5]")
    nb.cells[9].source = """## Manual smoke gate

Inspect the five outputs. Continue only when image/question pairing is correct, reasoning uses visible evidence, the final response is parseable, and smoke coverage is at least 80%.
"""

    nb.cells[10].source = """# PHASE 2 — run/resume the locked representative subset.
assert SMOKE_GATE['valid'], f'Smoke gate failed: {SMOKE_GATE}'
MAX_RUNTIME_HOURS = 2.0  # pilot run; change to 24.0 after reviewing speed and quality
started = time.perf_counter()
deadline = started + MAX_RUNTIME_HOURS * 3600

shared_store = SharedCheckpointStore(
    REPO, manifest['manifest_sha256'], 'cosmos3_bnb8_opt256_rep1400_shared',
    push_every=50, token=github_token,
)
print('1. Syncing isolated representative checkpoint...')
shared_store.sync_from_remote()
github_rows = shared_store.successful_rows()
seeded = {row['sample_id']: row for row in github_rows if row.get('status') == 'ok'}

for checkpoint in [FULL_OUT, *Path('/kaggle/input').rglob('predictions_rep1400.jsonl')]:
    if not checkpoint.exists():
        continue
    for row in load_jsonl(checkpoint):
        if (row.get('manifest_sha256') == manifest['manifest_sha256']
                and row.get('status') == 'ok'):
            seeded[row['sample_id']] = row

FULL_OUT.parent.mkdir(parents=True, exist_ok=True)
FULL_OUT.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\\n'
                            for _, row in sorted(seeded.items())), encoding='utf-8')
remaining = [row for row in records if row['sample_id'] not in seeded]
batches = list(group_by_image_batches(remaining, images_per_batch=16))
print(f'resumed={len(seeded)} remaining={len(remaining)} batches={len(batches)}')

for batch_index, batch in enumerate(batches, 1):
    if time.perf_counter() >= deadline:
        print('TIME BUDGET REACHED', flush=True)
        break
    print(f'\\n=== BATCH {batch_index}/{len(batches)}: {len(batch)} unique images ===', flush=True)
    materialize_batch(batch)
    try:
        run_range(batch, FULL_OUT, f'Representative batch {batch_index}',
                  deadline=deadline, shared_store=shared_store)
    finally:
        cleanup_batch(batch)

shared_store.flush(push=True)
predictions = load_jsonl(FULL_OUT)
summary, scored = evaluate_records(manifest, predictions)

# Population-weighted accuracy compensates for balanced source×task sampling.
truth = {row['sample_id']: row for row in records}
parsed = [row for row in predictions if row.get('prediction') in {'A','B','C','D'}]
weight_total = sum(truth[row['sample_id']]['sample_weight'] for row in parsed)
summary['weighted_accuracy'] = (
    sum(truth[row['sample_id']]['sample_weight'] *
        (row['prediction'] == truth[row['sample_id']]['answer']) for row in parsed) / weight_total
    if weight_total else None
)
summary['subset_sha256'] = manifest['manifest_sha256']
summary['runtime_budget_hours'] = MAX_RUNTIME_HOURS
summary['actual_runtime_hours'] = round((time.perf_counter() - started) / 3600, 4)
write_evaluation(OUT, summary, scored)
print(json.dumps(summary, ensure_ascii=False, indent=2))
archive = shutil.make_archive('/kaggle/working/cosmos3_mmad_rep1400_opt256_artifacts', 'zip', OUT)
print('DOWNLOAD:', archive)
"""
    nb.cells[11].source = """# Rebuild the portable archive at any time.
archive = shutil.make_archive('/kaggle/working/cosmos3_mmad_rep1400_opt256_artifacts', 'zip', OUT)
print('DOWNLOAD:', archive)
print('size MiB:', round(Path(archive).stat().st_size / 2**20, 2))
print('checkpoint:', FULL_OUT)
"""
    nb.metadata["cosmos_optimization"] = {
        "subset_sha256": subset["subset_sha256"],
        "records": 1400,
        "max_new_tokens": 256,
        "retry_tokens": 128,
        "early_stop": "</think> + A/B/C/D",
    }
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
