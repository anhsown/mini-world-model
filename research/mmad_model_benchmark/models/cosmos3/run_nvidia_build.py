from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.mmad import (  # noqa: E402
    SYSTEM_PROMPT,
    append_jsonl,
    evaluate_records,
    image_input_profile,
    load_jsonl,
    parse_prediction,
    split_reasoning_response,
    write_evaluation,
)
from common.shared_checkpoint import (  # noqa: E402
    SharedCheckpointStore,
    discover_checkpoint_repo,
)


MODEL_PAGE = "https://build.nvidia.com/nvidia/cosmos3-nano-reasoner"
SIGNIN_PAGE = "https://build.nvidia.com/models?modal=signin"
DEFAULT_PROFILE = ROOT.parent / "hatrec_cosmos3" / ".nvidia_browser_sessions" / "cosmos3_authenticated"
SUBSET_DATA = ROOT / "data"
FULL_DATA = ROOT / "data_full"


def ensure_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright>=1.52,<2"], check=True)
        from playwright.sync_api import sync_playwright
    return sync_playwright


def safe_goto(page, url: str, attempts: int = 5) -> None:
    last_error = None
    for attempt in range(attempts):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            return
        except Exception as exc:  # NVIDIA OAuth can redirect while goto is pending.
            last_error = exc
            if "interrupted by another navigation" not in str(exc).lower():
                raise
            page.wait_for_timeout(2_000 + attempt * 1_000)
    raise last_error


def fill_email(page) -> None:
    email = os.environ.get("NVIDIA_EMAIL", "").strip()
    if not email:
        email_file = ROOT.parent / "hatrec_cosmos3" / ".nvidia_email"
        if email_file.exists():
            email = email_file.read_text(encoding="utf-8").strip()
    if not email:
        return
    for selector in ('input[type="email"]', 'input[name="email"]', 'input[autocomplete="email"]'):
        field = page.locator(selector)
        if field.count() and field.first.is_visible():
            field.first.fill(email)
            next_button = page.get_by_role("button", name=re.compile(r"^(next|continue)$", re.I))
            if next_button.count() and next_button.first.is_enabled():
                next_button.first.click()
            print("NVIDIA email filled; complete password/MFA if requested.")
            return


def image_input(page):
    return page.locator('input[type="file"][accept*="image"]')


def wait_for_auth(page) -> None:
    deadline = time.time() + 900
    while time.time() < deadline:
        safe_goto(page, MODEL_PAGE)
        page.wait_for_timeout(5_000)
        if image_input(page).count():
            print("NVIDIA authenticated session verified.")
            return
        safe_goto(page, SIGNIN_PAGE)
        page.wait_for_timeout(1_500)
        fill_email(page)
        print("Complete password/MFA in the NVIDIA Chrome window; no Enter is required.")
        while time.time() < deadline:
            fill_email(page)
            if "build.nvidia.com" in page.url.lower() and "login.nvgs" not in page.url.lower():
                break
            page.wait_for_timeout(2_000)
    raise TimeoutError("NVIDIA login was not completed within 15 minutes")


def dismiss_notices(page) -> None:
    for label in ("Acknowledge & Continue", "Accept All", "Accept"):
        button = page.get_by_role("button", name=label, exact=True)
        if button.count() and button.first.is_visible():
            button.first.click()
            page.wait_for_timeout(750)


def prompt_fields(page):
    user = page.get_by_placeholder("Prompt", exact=True)
    system = page.get_by_placeholder("System Prompt", exact=True)
    if not user.count():
        user = page.locator("textarea").nth(0)
    if not system.count():
        system = page.locator("textarea").nth(1)
    return user, system


def read_output(page) -> str:
    body = page.locator("body").inner_text(timeout=10_000).replace("\r\n", "\n")
    match = re.search(r"\bOutput\s*\n(?:Preview\s*\nJSON|JSON\s*\nPreview)\s*\n?", body)
    if not match:
        return ""
    output = body[match.end() :].strip()
    # Exclude static footer/navigation content when present.
    for marker in ("View Parameters", "Model Card", "Deploy\nSystem Card"):
        if marker in output:
            output = output.split(marker, 1)[0].strip()
    return output


def detect_blocker(page) -> str | None:
    text = page.locator("body").inner_text(timeout=10_000).lower()
    if "blocked_by_safety" in text or "input rejected by content safety" in text:
        return "safety_block"
    if "too many requests" in text or "rate limit" in text:
        return "rate_limit"
    if page.locator('iframe[src*="captcha"]:visible').count():
        return "captcha"
    if page.get_by_text("Login", exact=True).count() and not image_input(page).count():
        return "login_required"
    return None


def failure_category(status: str, completion: str, error: str = "") -> str | None:
    value = error.lower()
    for category in ("safety_block", "rate_limit", "captcha", "login_required"):
        if category in value:
            return category
    if status == "no_output" or completion == "timeout_no_output":
        return "no_output"
    if status == "partial":
        return "output_parse_failure"
    if status == "error":
        return "ui_or_upload_error"
    return None


def write_sample_artifacts(records_dir: Path, row: dict) -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    stem = records_dir / row["sample_id"]
    stem.with_suffix(".json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = (
        f"# {row['sample_id']}\n\n"
        f"- Status: `{row['status']}`\n"
        f"- Completion: `{row['completion']}`\n"
        f"- Prediction: `{row.get('prediction')}`\n"
        f"- Failure category: `{row.get('failure_category')}`\n\n"
        f"## Reasoning\n\n{row.get('reasoning') or '*No reasoning captured.*'}\n\n"
        f"## Response\n\n{row.get('response') or '*No response captured.*'}\n"
    )
    stem.with_suffix(".md").write_text(markdown, encoding="utf-8")


def full_data_ready(root: Path) -> bool:
    manifest_path = root / "full_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if len(manifest.get("records", [])) != 39_670:
        return False
    # Several questions can share one image. Checking the distinct paths keeps
    # startup inexpensive while still preventing an incomplete full run.
    images = {row["image_file"] for row in manifest["records"]}
    return len(images) == 8_366 and all((root / relative).exists() for relative in images)


def prepare_full_data(root: Path) -> None:
    if full_data_ready(root):
        print(f"Full MMAD data already complete: {root}")
        return
    print("Preparing/resuming all 8,366 MMAD images for the 39,670-question run...")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "prepare_full.py"),
            "--output", str(root),
            "--range-download",
        ],
        check=True,
    )
    if not full_data_ready(root):
        raise RuntimeError("Full MMAD preparation ended without all 8,366 valid images")


def wait_for_answer(
    page, previous: str, timeout_seconds: int, poll_seconds: float = 0.2,
) -> tuple[str, str]:
    deadline = time.time() + timeout_seconds
    last = ""
    stable = 0
    last_prediction = None
    prediction_stable = 0
    while time.time() < deadline:
        blocker = detect_blocker(page)
        if blocker:
            raise RuntimeError(f"NVIDIA UI blocked the run: {blocker}")
        current = read_output(page)
        stable = stable + 1 if current and current == last and current != previous else 0
        last = current
        separated = split_reasoning_response(current)
        parsed = parse_prediction(separated["response"] or current)
        if parsed and parsed == last_prediction:
            prediction_stable += 1
        elif parsed:
            last_prediction = parsed
            prediction_stable = 0
        else:
            last_prediction = None
            prediction_stable = 0
        changed = bool(current and current != previous)
        structured_final = separated["parse_format"] in {"think_tags", "nvidia_ui"}
        run = page.get_by_role("button", name="Run", exact=True)
        run_ready = run.count() and run.is_enabled()
        # A structured final answer after </think> is already an unambiguous
        # completion signal. Capture it immediately; the NVIDIA spinner may
        # remain active long after the answer is visible.
        if changed and structured_final and parsed:
            return current, "complete_output_captured"
        if parsed and stable >= 2 and run_ready:
            return current, "complete"
        if parsed and stable >= 4:
            return current, "complete_spinner_stuck"
        time.sleep(poll_seconds)
    current = read_output(page)
    if current and current != previous:
        return current, "partial_timeout"
    return "", "timeout_no_output"


def reset_page(page) -> None:
    reset = page.get_by_role("button", name="Reset", exact=True)
    if reset.count() and reset.is_enabled():
        reset.click()
        page.wait_for_timeout(700)
    else:
        safe_goto(page, MODEL_PAGE)
        page.wait_for_timeout(2_000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MMAD zero-shot on Cosmos 3 NVIDIA Build UI")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--full", action="store_true",
        help="Prepare/resume and run all 39,670 MMAD questions over 8,366 images",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument(
        "--poll-ms", type=int, default=200,
        help="Poll the NVIDIA output this often and leave immediately on a final answer",
    )
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--skip-attempted", action="store_true",
        help="Skip previous failures too; by default only successful records are skipped",
    )
    parser.add_argument(
        "--resume-from", type=Path, action="append", default=[],
        help=(
            "Additional JSONL checkpoint to use for sample_id deduplication. "
            "May be repeated. Successful rows are skipped; failed rows are retried "
            "unless --skip-attempted is also set."
        ),
    )
    parser.add_argument(
        "--no-shared-checkpoint", action="store_true",
        help="Disable automatic GitHub checkpoint pull/push.",
    )
    parser.add_argument(
        "--checkpoint-repo", type=Path,
        help="Git clone used for shared checkpoint shards (auto-discovered by default).",
    )
    parser.add_argument(
        "--checkpoint-push-every", type=int, default=50,
        help="Create and push an immutable shared shard after this many successful answers.",
    )
    parser.add_argument("--records-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.full:
        data_root = args.data_root or FULL_DATA
        prepare_full_data(data_root)
        args.data_root = data_root
        args.manifest = args.manifest or data_root / "full_manifest.json"
        args.output = args.output or ROOT / "outputs" / "cosmos3_full" / "predictions.jsonl"
    else:
        args.data_root = args.data_root or SUBSET_DATA
        args.manifest = args.manifest or args.data_root / "subset_manifest.json"
        args.output = args.output or ROOT / "outputs" / "cosmos3" / "predictions.jsonl"

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"][: args.limit or None]
    prior_rows = load_jsonl(args.output)
    resume_rows = list(prior_rows)
    for checkpoint in args.resume_from:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        extra_rows = load_jsonl(checkpoint)
        incompatible = {
            row.get("manifest_sha256")
            for row in extra_rows
            if row.get("manifest_sha256")
            and row.get("manifest_sha256") != manifest["manifest_sha256"]
        }
        if incompatible:
            raise ValueError(
                f"Resume checkpoint uses a different MMAD manifest: {checkpoint}"
            )
        resume_rows.extend(extra_rows)
        print(f"Loaded resume checkpoint: {checkpoint} ({len(extra_rows)} rows)")
    completed = {
        row["sample_id"] for row in resume_rows
        if row.get("status") == "ok" or args.skip_attempted
    }
    shared_store = None
    if not args.no_shared_checkpoint:
        try:
            code_repo = ROOT.parents[1]
            checkpoint_repo = args.checkpoint_repo or discover_checkpoint_repo(code_repo)
            shared_store = SharedCheckpointStore(
                checkpoint_repo,
                manifest["manifest_sha256"],
                "nvidia_build",
                push_every=args.checkpoint_push_every,
            )
            shared_store.sync_from_remote()
            completed.update(shared_store.completed_ids)
            print(
                f"Shared checkpoint repo={checkpoint_repo} "
                f"completed={len(shared_store.completed_ids)}"
            )
        except Exception as exc:
            print(f"Shared checkpoint disabled after initialization warning: {exc}")
            shared_store = None
    records_dir = args.records_dir or args.output.parent / "records"
    print(f"MMAD manifest={manifest['manifest_sha256']} records={len(records)} resumed={len(completed)}")
    args.profile.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sync_playwright = ensure_playwright()
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(args.profile), channel="chrome", headless=args.headless,
            viewport={"width": 1440, "height": 1000},
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            wait_for_auth(page)
            dismiss_notices(page)
            image_input(page).wait_for(state="attached", timeout=120_000)
            with tempfile.TemporaryDirectory(prefix="mmad_blind_") as temp_dir:
                for index, sample in enumerate(records, start=1):
                    if sample["sample_id"] in completed:
                        print(f"[{index}/{len(records)}] SKIP {sample['sample_id']}")
                        continue
                    source = args.data_root / sample["image_file"]
                    if not source.exists():
                        raise FileNotFoundError(f"Run prepare_subset.py first: {source}")
                    neutral = Path(temp_dir) / f"sample_{index:04d}{source.suffix.lower()}"
                    shutil.copy2(source, neutral)
                    profile = image_input_profile(source)
                    started = time.perf_counter()
                    try:
                        reset_page(page)
                        uploader = image_input(page)
                        uploader.set_input_files(str(neutral))
                        page.wait_for_function(
                            """name => { const i=document.querySelector('input[type=file][accept*=image]');
                            return !!(i && i.files && i.files[0] && i.files[0].name===name); }""",
                            arg=neutral.name, timeout=20_000,
                        )
                        user, system = prompt_fields(page)
                        user.fill(sample["prompt"])
                        system.fill(SYSTEM_PROMPT)
                        run = page.get_by_role("button", name="Run", exact=True)
                        run.wait_for(state="visible", timeout=60_000)
                        enable_deadline = time.time() + 60
                        while not run.is_enabled() and time.time() < enable_deadline:
                            time.sleep(1)
                        if not run.is_enabled():
                            raise RuntimeError("Run button did not enable")
                        previous = read_output(page)
                        run.click()
                        raw, completion = wait_for_answer(
                            page, previous, args.timeout, max(args.poll_ms, 50) / 1000.0,
                        )
                        separated = split_reasoning_response(raw)
                        prediction = parse_prediction(separated["response"] or raw)
                        status = "ok" if prediction else ("partial" if raw else "no_output")
                        row = {
                            "sample_id": sample["sample_id"],
                            "model": "nvidia/cosmos3-nano-reasoner",
                            "backend": "NVIDIA Build Experience UI",
                            "manifest_sha256": manifest["manifest_sha256"],
                            "status": status,
                            "completion": completion,
                            "failure_category": failure_category(status, completion),
                            "prediction": prediction,
                            "reasoning": separated["reasoning"],
                            "response": separated["response"],
                            "output_parse_format": separated["parse_format"],
                            "reasoning_chars": len(separated["reasoning"]),
                            "response_chars": len(separated["response"]),
                            "raw_response": raw,
                            "question_type": sample["question_type"],
                            "source_dataset": sample["source_dataset"],
                            "category": sample["category"],
                            "is_normal": sample["is_normal"],
                            "ground_truth": sample["answer"],
                            "image_file": sample["image_file"],
                            "image_profile": profile,
                            "latency_seconds": round(time.perf_counter() - started, 3),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        append_jsonl(args.output, row)
                        write_sample_artifacts(records_dir, row)
                        if shared_store is not None:
                            shared_store.record(row)
                        print(f"[{index}/{len(records)}] {status.upper()} pred={prediction} "
                              f"truth={sample['answer']} {row['latency_seconds']:.1f}s")
                    except Exception as exc:
                        screenshot = args.output.parent / "errors" / f"{sample['sample_id']}.png"
                        screenshot.parent.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(screenshot), full_page=True)
                        error_text = str(exc)
                        error_row = {
                            "sample_id": sample["sample_id"], "model": "nvidia/cosmos3-nano-reasoner",
                            "status": "error", "completion": "exception",
                            "failure_category": failure_category("error", "exception", error_text),
                            "error": error_text, "screenshot": str(screenshot),
                            "reasoning": "", "response": "", "prediction": None,
                            "question_type": sample["question_type"],
                            "source_dataset": sample["source_dataset"],
                            "category": sample["category"],
                            "is_normal": sample["is_normal"],
                            "ground_truth": sample["answer"],
                            "image_file": sample["image_file"],
                            "image_profile": profile,
                            "latency_seconds": round(time.perf_counter() - started, 3),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        append_jsonl(args.output, error_row)
                        write_sample_artifacts(records_dir, error_row)
                        print(f"[{index}/{len(records)}] ERROR {exc}")
                        if any(key in str(exc).lower() for key in ("rate_limit", "captcha", "login_required")):
                            break
                        safe_goto(page, MODEL_PAGE)
                        page.wait_for_timeout(2_000)
                    neutral.unlink(missing_ok=True)
                    time.sleep(args.delay)
        finally:
            if shared_store is not None:
                try:
                    shared_store.flush(push=True)
                except Exception as exc:
                    print(f"Shared checkpoint final flush warning: {exc}")
            context.close()

    summary, scored = evaluate_records(manifest, load_jsonl(args.output))
    write_evaluation(args.output.parent, summary, scored)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
