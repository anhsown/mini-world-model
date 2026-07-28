import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "VideoDataset" / "Cycles"
OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"


def ensure_dependencies() -> None:
    required = ("requests", "cv2")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return
    print("Installing missing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )


def load_windows_user_key_if_needed() -> None:
    if os.getenv("NVIDIA_API_KEY"):
        return
    if sys.platform != "win32":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "NVIDIA_API_KEY")
        if value:
            os.environ["NVIDIA_API_KEY"] = value
    except OSError:
        pass


def run(command: list[str]) -> None:
    print("\n>", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


def main() -> None:
    print("HATRec x Cosmos 3 automatic pipeline")
    print("root:", ROOT)
    print("dataset:", DATASET)

    if not DATASET.exists():
        raise SystemExit(f"Dataset not found: {DATASET}")
    video_count = sum(1 for _ in DATASET.rglob("*.mp4"))
    if video_count == 0:
        raise SystemExit(f"No MP4 files found under: {DATASET}")
    print("videos:", video_count)

    ensure_dependencies()
    load_windows_user_key_if_needed()
    api_key = os.getenv("NVIDIA_API_KEY", "")
    api_base = os.getenv("COSMOS3_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    is_nvidia_hosted = "integrate.api.nvidia.com" in api_base
    if is_nvidia_hosted and not api_key.startswith("nvapi-"):
        raise SystemExit(
            "NVIDIA_API_KEY is missing. Run set_api_key.ps1 once, then rerun this command."
        )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        "inspect_dataset.py",
        "--dataset",
        str(DATASET),
        "--output",
        str(OUTPUTS / "dataset_audit.json"),
    ])
    run([
        sys.executable,
        "run_batch.py",
        "--dataset",
        str(DATASET),
        "--output",
        str(REPORTS),
        "--max-videos",
        "0",
    ])
    run([
        sys.executable,
        "evaluate_results.py",
        "--reports",
        str(REPORTS),
        "--output",
        str(OUTPUTS / "evaluation.json"),
    ])

    manifest = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET),
        "videos_discovered": video_count,
        "reports_folder": str(REPORTS),
        "evaluation": str(OUTPUTS / "evaluation.json"),
    }
    (OUTPUTS / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nPIPELINE COMPLETE")
    print("Reports:", REPORTS)
    print("Metrics:", OUTPUTS / "evaluation.json")
    if sys.platform == "win32":
        os.startfile(OUTPUTS)


if __name__ == "__main__":
    main()
