from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
from common.mmad import (  # noqa: E402
    SYSTEM_PROMPT,
    append_jsonl,
    evaluate_records,
    load_jsonl,
    parse_prediction,
    write_evaluation,
)


def sync_checkpoint(local_file: Path, mirror_file: Path | None) -> None:
    if mirror_file is None or not local_file.exists():
        return
    mirror_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = mirror_file.with_suffix(mirror_file.suffix + ".tmp")
    shutil.copy2(local_file, temporary)
    temporary.replace(mirror_file)


def restore_checkpoint(local_file: Path, mirror_file: Path | None) -> None:
    if mirror_file and mirror_file.exists():
        if not local_file.exists() or mirror_file.stat().st_size > local_file.stat().st_size:
            local_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mirror_file, local_file)


def make_messages(sample: dict, image: Path) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data_full")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qwen2_vl_full")
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--archives",
        nargs="+",
        help="Run only records from these source archives.",
    )
    args = parser.parse_args()

    manifest_path = args.data / "full_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_predictions = args.output / "predictions.jsonl"
    mirror_predictions = args.mirror / "predictions.jsonl" if args.mirror else None
    restore_checkpoint(local_predictions, mirror_predictions)

    previous = load_jsonl(local_predictions)
    done = {
        row["sample_id"]
        for row in previous
        if row.get("status") in {"ok", "parse_failure"}
    }
    records = manifest["records"]
    if args.archives:
        selected_archives = set(args.archives)
        records = [row for row in records if row["source_archive"] in selected_archives]
    records = records[: args.limit or None]
    pending = [row for row in records if row["sample_id"] not in done]
    print(
        f"manifest={manifest['manifest_sha256']} total={len(records)} "
        f"done={len(records)-len(pending)} pending={len(pending)}"
    )

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    # Qwen's checkpoint contains sampling-only defaults. Null them for deterministic
    # greedy generation so Transformers does not emit ignored-flag warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    print(
        "device=", next(model.parameters()).device,
        "VRAM_GiB=", round(torch.cuda.memory_allocated() / 2**30, 2),
    )

    def emergency_sync(*_: object) -> None:
        sync_checkpoint(local_predictions, mirror_predictions)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, emergency_sync)
    signal.signal(signal.SIGINT, emergency_sync)

    batch_size = max(1, args.batch_size)
    cursor = 0
    completed_this_run = 0
    started_all = time.perf_counter()
    try:
        while cursor < len(pending):
            batch = pending[cursor : cursor + batch_size]
            conversations = [
                make_messages(sample, (args.data / sample["image_file"]).resolve())
                for sample in batch
            ]
            texts = [
                processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                for conversation in conversations
            ]
            try:
                image_inputs, video_inputs = process_vision_info(conversations)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                started_batch = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                elapsed = time.perf_counter() - started_batch
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                print(f"CUDA OOM: retrying with batch_size={batch_size}")
                continue

            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
            raw_answers = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            per_sample_latency = elapsed / len(batch)
            for sample, raw in zip(batch, raw_answers):
                raw = raw.strip()
                prediction = parse_prediction(raw)
                append_jsonl(
                    local_predictions,
                    {
                        "sample_id": sample["sample_id"],
                        "model": args.model,
                        "backend": f"Transformers FP16 SDPA batch={len(batch)}",
                        "manifest_sha256": manifest["manifest_sha256"],
                        "status": "ok" if prediction else "parse_failure",
                        "prediction": prediction,
                        "raw_response": raw,
                        "latency_seconds": round(per_sample_latency, 4),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                completed_this_run += 1
            cursor += len(batch)

            total_done = len(records) - len(pending) + cursor
            if completed_this_run % args.checkpoint_every < len(batch) or cursor == len(pending):
                sync_checkpoint(local_predictions, mirror_predictions)
                rate = completed_this_run / max(time.perf_counter() - started_all, 1e-6)
                eta_hours = (len(pending) - cursor) / max(rate, 1e-6) / 3600
                print(
                    f"[{total_done}/{len(records)}] batch={len(batch)} "
                    f"last={batch[-1]['sample_id']} rate={rate:.2f} q/s "
                    f"ETA={eta_hours:.2f}h"
                )
    finally:
        sync_checkpoint(local_predictions, mirror_predictions)

    predictions = load_jsonl(local_predictions)
    summary, scored = evaluate_records(manifest, predictions)
    write_evaluation(args.output, summary, scored)
    if args.mirror:
        args.mirror.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output / "metrics.json", args.mirror / "metrics.json")
        shutil.copy2(
            args.output / "predictions_scored.csv",
            args.mirror / "predictions_scored.csv",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
