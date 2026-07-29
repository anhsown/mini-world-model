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
    load_jsonl,
    parse_prediction,
    write_evaluation,
)


MODEL_PAGE = "https://build.nvidia.com/nvidia/cosmos3-nano-reasoner"
SIGNIN_PAGE = "https://build.nvidia.com/models?modal=signin"
DEFAULT_PROFILE = ROOT.parent / "hatrec_cosmos3" / ".nvidia_browser_sessions" / "cosmos3_authenticated"


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
    if "too many requests" in text or "rate limit" in text:
        return "rate_limit"
    if page.locator('iframe[src*="captcha"]:visible').count():
        return "captcha"
    if page.get_by_text("Login", exact=True).count() and not image_input(page).count():
        return "login_required"
    return None


def wait_for_answer(page, previous: str, timeout_seconds: int) -> tuple[str, str]:
    deadline = time.time() + timeout_seconds
    last = ""
    stable = 0
    while time.time() < deadline:
        blocker = detect_blocker(page)
        if blocker:
            raise RuntimeError(f"NVIDIA UI blocked the run: {blocker}")
        current = read_output(page)
        stable = stable + 1 if current and current == last and current != previous else 0
        last = current
        parsed = parse_prediction(current)
        run = page.get_by_role("button", name="Run", exact=True)
        run_ready = run.count() and run.is_enabled()
        if parsed and stable >= 2 and run_ready:
            return current, "complete"
        if parsed and stable >= 4:
            return current, "complete_spinner_stuck"
        time.sleep(1)
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
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "subset_manifest.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cosmos3" / "predictions.jsonl")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"][: args.limit or None]
    completed = {
        row["sample_id"] for row in load_jsonl(args.output)
        if row.get("status") in {"ok", "parse_failure", "partial"}
    }
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
                        raw, completion = wait_for_answer(page, previous, args.timeout)
                        prediction = parse_prediction(raw)
                        status = "ok" if prediction else ("partial" if raw else "parse_failure")
                        row = {
                            "sample_id": sample["sample_id"],
                            "model": "nvidia/cosmos3-nano-reasoner",
                            "backend": "NVIDIA Build Experience UI",
                            "manifest_sha256": manifest["manifest_sha256"],
                            "status": status,
                            "completion": completion,
                            "prediction": prediction,
                            "raw_response": raw,
                            "latency_seconds": round(time.perf_counter() - started, 3),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        append_jsonl(args.output, row)
                        print(f"[{index}/{len(records)}] {status.upper()} pred={prediction} "
                              f"truth={sample['answer']} {row['latency_seconds']:.1f}s")
                    except Exception as exc:
                        screenshot = args.output.parent / "errors" / f"{sample['sample_id']}.png"
                        screenshot.parent.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(screenshot), full_page=True)
                        append_jsonl(args.output, {
                            "sample_id": sample["sample_id"], "model": "nvidia/cosmos3-nano-reasoner",
                            "status": "error", "error": str(exc), "screenshot": str(screenshot),
                            "latency_seconds": round(time.perf_counter() - started, 3),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        })
                        print(f"[{index}/{len(records)}] ERROR {exc}")
                        if any(key in str(exc).lower() for key in ("rate_limit", "captcha", "login_required")):
                            break
                        safe_goto(page, MODEL_PAGE)
                        page.wait_for_timeout(2_000)
                    neutral.unlink(missing_ok=True)
                    time.sleep(args.delay)
        finally:
            context.close()

    summary, scored = evaluate_records(manifest, load_jsonl(args.output))
    write_evaluation(args.output.parent, summary, scored)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
