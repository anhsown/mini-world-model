import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "VideoDataset" / "Cycles"
OUTPUT_ROOT = ROOT / "outputs" / "ui_reports"
PROFILE_ROOT = ROOT / ".nvidia_browser_sessions"
AUTH_PROFILE = PROFILE_ROOT / "cosmos3_authenticated"
EMAIL_FILE = ROOT / ".nvidia_email"
MODEL_PAGE = "https://build.nvidia.com/nvidia/cosmos3-nano-reasoner"
SIGNIN_PAGE = "https://build.nvidia.com/models?modal=signin"
TASK_RE = re.compile(r"_task_([0-6])\.mp4$", re.IGNORECASE)


def ensure_playwright():
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError:
        print("Installing Playwright Python package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "playwright>=1.52,<2"],
            check=True,
        )
        from playwright.sync_api import sync_playwright
        return sync_playwright


def select_videos(per_task: int, run_all: bool) -> list[Path]:
    groups = {task: [] for task in range(7)}
    for video in sorted(DATASET.rglob("*.mp4")):
        match = TASK_RE.search(video.name)
        if match:
            groups[int(match.group(1))].append(video)
    missing = [task for task, videos in groups.items() if not videos]
    if missing:
        raise SystemExit(f"Missing HATRec task groups: {missing}")
    if run_all:
        return sorted(video for videos in groups.values() for video in videos)
    return sorted(video for task in range(7) for video in groups[task][:per_task])


def prompts() -> tuple[str, str]:
    system = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8").strip()
    user = (ROOT / "prompts" / "user_prompt.txt").read_text(encoding="utf-8").strip()
    return system, user


def result_path(video: Path) -> Path:
    return OUTPUT_ROOT / video.relative_to(DATASET).with_suffix(".json")


def safe_goto(page, url: str, attempts: int = 5) -> None:
    last_error = None
    for attempt in range(attempts):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            return
        except Exception as exc:
            last_error = exc
            if "interrupted by another navigation" not in str(exc).lower():
                raise
            print(f"NVIDIA is still redirecting after login; waiting ({attempt + 1}/{attempts})...")
            page.wait_for_timeout(3_000)
    raise last_error


def autofill_nvidia_email(page) -> None:
    if not EMAIL_FILE.exists():
        return
    email = EMAIL_FILE.read_text(encoding="utf-8").strip()
    if not email:
        return
    selectors = [
        'input[type="email"]',
        'input[name="email"]',
        'input[autocomplete="email"]',
    ]
    for selector in selectors:
        field = page.locator(selector)
        if field.count() > 0 and field.first.is_visible():
            field.first.fill(email)
            next_button = page.get_by_role("button", name=re.compile(r"^(next|continue)$", re.I))
            if next_button.count() > 0 and next_button.first.is_enabled():
                next_button.first.click()
            print("NVIDIA email filled automatically; complete password/MFA if requested.")
            return


def wait_for_user_setup(page) -> None:
    auth_deadline = time.time() + 900
    while True:
        safe_goto(page, MODEL_PAGE)
        # The NVIDIA header hydrates after DOMContentLoaded. Checking immediately
        # can mistake a not-yet-rendered Login control for an authenticated state.
        page.wait_for_timeout(5_000)
        login = page.get_by_text("Login", exact=True)
        if login.count() == 0 or not login.first.is_visible():
            break

        if time.time() >= auth_deadline:
            raise TimeoutError("NVIDIA login was not completed within 15 minutes")
        safe_goto(page, SIGNIN_PAGE)
        page.wait_for_timeout(1_500)
        autofill_nvidia_email(page)
        print("\nComplete password/MFA in the NVIDIA Chrome window.")
        print("The runner detects successful login automatically; do not press Enter.")
        while time.time() < auth_deadline:
            if "login.nvgs.nvidia.com" in page.url.lower():
                autofill_nvidia_email(page)
                page.wait_for_timeout(2_000)
                continue
            # OAuth has returned to build.nvidia.com. Re-open the model page and
            # positively verify the fully rendered header in the outer loop.
            page.wait_for_timeout(3_000)
            break

    print("NVIDIA authenticated session verified; starting Cosmos 3 automatically.")
    for label in ("Acknowledge & Continue", "Accept All", "Accept"):
        button = page.get_by_role("button", name=label, exact=True)
        if button.count() > 0 and button.first.is_visible():
            button.first.click()
            page.wait_for_timeout(1_000)
    page.locator('input[type="file"][accept*="video/mp4"]').wait_for(
        state="attached", timeout=120_000
    )


def read_output_text(page) -> str:
    """Read the rendered output column, not only its Output/Preview/JSON header."""
    body = page.locator("body").inner_text(timeout=10_000)
    normalized = body.replace("\r\n", "\n")
    markers = ("Output\nPreview\nJSON", "Output\nJSON\nPreview")
    for marker in markers:
        if marker in normalized:
            return normalized.split(marker, 1)[1].strip()
    # Fallback for minor NVIDIA label/layout changes.
    match = re.search(r"\bOutput\s*\n(?:Preview\s*\nJSON|JSON\s*\nPreview)\s*\n?", normalized)
    return normalized[match.end():].strip() if match else ""


def detect_blocker(page) -> str | None:
    visible_text = page.locator("body").inner_text(timeout=10_000).lower()
    if "too many requests" in visible_text or "rate limit" in visible_text:
        return "rate_limit"
    if page.locator('iframe[src*="captcha"]:visible').count() > 0:
        return "captcha"
    return None


def wait_for_new_output(page, previous: str, timeout_seconds: int) -> tuple[str, str]:
    deadline = time.time() + timeout_seconds
    last_text = ""
    stable_polls = 0
    while time.time() < deadline:
        blocker = detect_blocker(page)
        if blocker:
            raise RuntimeError(f"NVIDIA UI blocked the run: {blocker}")
        current = read_output_text(page)
        if current == last_text and current != previous:
            stable_polls += 1
        else:
            stable_polls = 0
            last_text = current

        upper = current.upper()
        report_complete = (
            "CONFIDENCE AND LIMITATIONS" in upper
            and "MOST LIKELY HATREC TASK" in upper
        )
        final_answer_visible = bool(
            re.search(r"(?:ANSWER\s+IS\s*:|HATREC\s+TASK)[^0-6]{0,40}[0-6]", upper)
        )
        legacy_complete = "REASONING COMPLETE" in upper and "RESPONSE" in upper
        run_button = page.get_by_role("button", name="Run", exact=True)
        run_ready = run_button.count() > 0 and run_button.is_enabled()
        if current != previous and stable_polls >= 2 and run_ready and (
            report_complete or legacy_complete
        ):
            return current, "complete"
        # NVIDIA's UI can leave the Thinking spinner active after the final answer
        # is already rendered. A stable evidence report is safe to collect anyway;
        # the next Reset click cancels any stale UI state.
        if current != previous and stable_polls >= 5 and (
            report_complete or final_answer_visible
        ):
            return current, "complete_spinner_stuck"
        time.sleep(2)
    current = read_output_text(page)
    if current != previous and len(current) >= 100:
        return current, "partial_timeout"
    return current if current != previous else "", "timeout_no_output"


def final_response(raw_output: str) -> str:
    marker = "\nResponse\n"
    return raw_output.rsplit(marker, 1)[1].strip() if marker in raw_output else raw_output.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--all", action="store_true", help="Deprecated: all videos are now the default")
    parser.add_argument("--smoke", action="store_true", help="Only process --per-task videos per class")
    parser.add_argument("--delay", type=float, default=15.0)
    parser.add_argument("--timeout", type=int, default=15,
                        help="Save partial output and advance after this many seconds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not DATASET.exists():
        raise SystemExit(f"Dataset not found: {DATASET}")
    videos = select_videos(args.per_task, run_all=not args.smoke)
    print(f"Selected {len(videos)} Cosmos 3 UI evaluations")
    for video in videos:
        print(video.relative_to(DATASET))
    if args.dry_run:
        return

    sync_playwright = ensure_playwright()
    system_prompt, user_prompt = prompts()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

    AUTH_PROFILE.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = None
        try:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(AUTH_PROFILE),
                channel="chrome",
                headless=False,
                viewport={"width": 1440, "height": 1000},
            )
            page = context.pages[0] if context.pages else context.new_page()
            try:
                wait_for_user_setup(page)
            except Exception:
                setup_shot = ROOT / "outputs" / "ui_setup_error.png"
                setup_shot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(setup_shot), full_page=True)
                print("Setup screenshot:", setup_shot)
                raise

            with tempfile.TemporaryDirectory(prefix="hatrec_blind_") as temp_dir:
                for index, video in enumerate(videos, start=1):
                    destination = result_path(video)
                    if destination.exists():
                        try:
                            saved = json.loads(destination.read_text(encoding="utf-8"))
                            saved_report = saved.get("report", "").strip()
                            if (
                                saved.get("status") in {"ok", "partial"}
                                and saved.get("upload_verified") is True
                                and len(saved_report) >= 100
                                and saved_report not in {"Output\nPreview\nJSON", "Output\nJSON\nPreview"}
                            ):
                                print(f"[{index}/{len(videos)}] SKIP {video.name}")
                                continue
                        except (OSError, json.JSONDecodeError):
                            pass

                    print(f"[{index}/{len(videos)}] RUN {video.name}")
                    neutral_name = f"sample_{index:06d}.mp4"
                    neutral_video = Path(temp_dir) / neutral_name
                    shutil.copy2(video, neutral_video)
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        reset = page.get_by_role("button", name="Reset", exact=True)
                        if reset.is_enabled():
                            reset.click()
                            page.wait_for_timeout(500)
                        file_input = page.locator('input[type="file"][accept*="video/mp4"]')
                        file_input.set_input_files(str(neutral_video))
                        page.wait_for_function(
                            """(name) => {
                                const input = document.querySelector('input[type="file"][accept*="video/mp4"]');
                                return Boolean(input && input.files && input.files[0] && input.files[0].name === name);
                            }""",
                            arg=neutral_name,
                            timeout=15_000,
                        )
                        page.wait_for_timeout(1_000)
                        print(f"[{index}/{len(videos)}] UPLOAD_OK {neutral_name}")
                        page.get_by_placeholder("Prompt", exact=True).fill(user_prompt)
                        page.get_by_placeholder("System Prompt", exact=True).fill(system_prompt)

                        run_button = page.get_by_role("button", name="Run", exact=True)
                        run_button.wait_for(state="visible", timeout=60_000)
                        deadline = time.time() + 60
                        while not run_button.is_enabled() and time.time() < deadline:
                            time.sleep(1)
                        if not run_button.is_enabled():
                            raise RuntimeError("Run button did not become enabled after video upload")

                        previous = read_output_text(page)
                        started = time.perf_counter()
                        run_button.click()
                        raw_output, completion = wait_for_new_output(page, previous, args.timeout)
                        latency = time.perf_counter() - started
                        report = final_response(raw_output)

                        record = {
                            "status": "partial" if completion in {
                                "partial_timeout", "timeout_no_output"
                            } else "ok",
                            "completion": completion,
                            "sample_id": str(video.relative_to(DATASET).with_suffix("")),
                            "source_file": str(video.resolve()),
                            "uploaded_filename": neutral_name,
                            "upload_verified": True,
                            "model": "nvidia/cosmos3-nano-reasoner",
                            "backend": "NVIDIA Build Experience UI",
                            "latency_seconds": round(latency, 3),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "report": report,
                            "raw_output": raw_output,
                        }
                        destination.write_text(
                            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        destination.with_suffix(".md").write_text(report, encoding="utf-8")
                        print(f"[{index}/{len(videos)}] {completion.upper()} {latency:.1f}s")
                        if completion in {"partial_timeout", "timeout_no_output"}:
                            print(f"[{index}/{len(videos)}] RESET_STUCK_RUN")
                            safe_goto(page, MODEL_PAGE)
                            page.locator('input[type="file"][accept*="video/mp4"]').wait_for(
                                state="attached", timeout=120_000
                            )
                    except Exception as exc:
                        screenshot = destination.with_suffix(".error.png")
                        page.screenshot(path=str(screenshot), full_page=True)
                        destination.write_text(
                            json.dumps(
                                {
                                    "status": "blocked",
                                    "sample_id": str(video.relative_to(DATASET).with_suffix("")),
                                    "error": str(exc),
                                    "screenshot": str(screenshot),
                                    "created_at": datetime.now(timezone.utc).isoformat(),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print("STOPPED:", exc)
                        print("Screenshot:", screenshot)
                        blocker_error = any(
                            marker in str(exc).lower()
                            for marker in ("rate_limit", "captcha", "login_required")
                        )
                        if blocker_error:
                            break
                        print("RECOVERING: reloading Cosmos 3 and continuing with next video")
                        safe_goto(page, MODEL_PAGE)
                        page.locator('input[type="file"][accept*="video/mp4"]').wait_for(
                            state="attached", timeout=120_000
                        )
                        continue
                    neutral_video.unlink(missing_ok=True)
                    time.sleep(args.delay)
        finally:
            if context is not None:
                context.close()

    subprocess.run(
        [sys.executable, str(ROOT / "evaluate_results.py"), "--reports", str(OUTPUT_ROOT),
         "--output", str(ROOT / "outputs" / "ui_evaluation.json")],
        check=True,
    )
    print("Results:", OUTPUT_ROOT)
    print("Metrics:", ROOT / "outputs" / "ui_evaluation.json")


if __name__ == "__main__":
    main()
