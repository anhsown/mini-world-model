from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("Cosmos3_Nano_MMAD_T4x2_UI_Parity.ipynb")


RESUME_BLOCK = r'''# Import successful Cosmos checkpoints attached as Kaggle Datasets.
# Accepted files include predictions.jsonl, predictions_from_230.jsonl, and
# predictions_combined.jsonl. Records are deduplicated by canonical sample_id.
resume_candidates = [FULL_OUT]
for pattern in ('predictions.jsonl', 'predictions_from_230.jsonl', 'predictions_combined.jsonl'):
    resume_candidates.extend(Path('/kaggle/input').rglob(pattern))

seeded = {}
resume_sources = []
for checkpoint in dict.fromkeys(resume_candidates):
    if not checkpoint.exists():
        continue
    accepted = 0
    for row in load_jsonl(checkpoint):
        sid = row.get('sample_id', '')
        if not sid.startswith('mmad_full_'):
            continue
        qnum = int(row.get('question_number') or sid.rsplit('_', 1)[1])
        if qnum < START_QUESTION or row.get('status') != 'ok':
            continue
        row_hash = row.get('manifest_sha256')
        if row_hash and row_hash != manifest['manifest_sha256']:
            continue
        model_name = f"{row.get('model', '')} {row.get('base_model', '')}".lower()
        if 'cosmos3' not in model_name and 'cosmos-3' not in model_name:
            continue
        seeded[sid] = row
        accepted += 1
    if accepted:
        resume_sources.append({'path': str(checkpoint), 'accepted_ok': accepted})

# Seed one portable local checkpoint. Future rows append to this same file.
FULL_OUT.parent.mkdir(parents=True, exist_ok=True)
FULL_OUT.write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n'
            for _, row in sorted(seeded.items())),
    encoding='utf-8',
)
completed_ids = set(seeded)
print('Cross-backend resume sources:', json.dumps(resume_sources, indent=2))
print('Successful Cosmos sample_ids seeded:', len(completed_ids))'''


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    auth_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "Optional Hugging Face authentication" in "".join(cell.get("source", []))
    )
    auth_source = "".join(auth_cell["source"])
    if "GITHUB_TOKEN" not in auth_source:
        auth_source += '''

# Optional GitHub write authentication for shared checkpoint shards.
# Create a Kaggle Secret named GITHUB_TOKEN; never paste the PAT into this notebook.
try:
    github_token = UserSecretsClient().get_secret('GITHUB_TOKEN')
except Exception:
    github_token = os.environ.get('GITHUB_TOKEN')
if github_token:
    os.environ['GITHUB_TOKEN'] = github_token
print('GitHub checkpoint push:', 'enabled' if github_token else 'read-only')
'''
        auth_cell["source"] = auth_source.splitlines(keepends=True)

    inference_cell = next(
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "def run_range" in "".join(cell.get("source", []))
    )
    inference_source = "".join(inference_cell["source"])
    if "from common.shared_checkpoint import SharedCheckpointStore" not in inference_source:
        inference_source = inference_source.replace(
            "from PIL import Image\n",
            "from PIL import Image\nfrom common.shared_checkpoint import SharedCheckpointStore\n",
        )
    inference_source = inference_source.replace(
        "def run_range(selected, output_path, label, deadline=None):",
        "def run_range(selected, output_path, label, deadline=None, shared_store=None):",
    )
    hook = "        append_jsonl(output_path, row)\n"
    if "shared_store.record(row)" not in inference_source:
        inference_source = inference_source.replace(
            hook,
            hook + "        if shared_store is not None:\n            shared_store.record(row)\n",
        )
    inference_cell["source"] = inference_source.splitlines(keepends=True)

    target = next(
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "PHASE 2" in "".join(cell.get("source", []))
    )
    source = "".join(target["source"])
    old = "completed_ids = {r['sample_id'] for r in load_jsonl(FULL_OUT) if r.get('status') == 'ok'}"
    if old in source:
        source = source.replace(old, RESUME_BLOCK)
    elif "Cross-backend resume sources:" not in source:
        raise RuntimeError("Phase 2 resume anchor was not found")
    store_setup = '''shared_store = SharedCheckpointStore(
    REPO, manifest['manifest_sha256'], 'kaggle_t4x2',
    push_every=50, token=github_token,
)
shared_store.sync_from_remote()
print('GitHub shared checkpoint completed:', len(shared_store.completed_ids))
'''
    anchor = "continuation_records = records[START_QUESTION - 1:]\n"
    if "GitHub shared checkpoint completed:" not in source:
        source = source.replace(anchor, anchor + store_setup)

    github_seed = '''
# Add immutable GitHub shards to the same deduplicated seed.
github_rows = shared_store.successful_rows()
for row in github_rows:
    sid = row['sample_id']
    qnum = int(row.get('question_number') or sid.rsplit('_', 1)[1])
    if qnum >= START_QUESTION:
        seeded[sid] = row
if github_rows:
    resume_sources.append({'path': 'GitHub shared checkpoint shards',
                           'accepted_ok': len(github_rows)})
'''
    seed_anchor = "# Seed one portable local checkpoint. Future rows append to this same file.\n"
    if "github_rows = shared_store.successful_rows()" not in source:
        source = source.replace(seed_anchor, github_seed + "\n" + seed_anchor)

    source = source.replace(
        "run_range(batch, FULL_OUT, f'MMAD batch {batch_index}', deadline=deadline)",
        "run_range(batch, FULL_OUT, f'MMAD batch {batch_index}', "
        "deadline=deadline, shared_store=shared_store)",
    )
    flush_anchor = "full_predictions = load_jsonl(FULL_OUT)\n"
    if "shared_store.flush(push=True)" not in source:
        source = source.replace(
            flush_anchor,
            "shared_store.flush(push=True)\n" + flush_anchor,
        )
    target["source"] = source.splitlines(keepends=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated: {NOTEBOOK}")


if __name__ == "__main__":
    main()
