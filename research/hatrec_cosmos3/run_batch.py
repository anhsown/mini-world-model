import argparse
import base64
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


MODEL = "nvidia/cosmos3-nano-reasoner"
ROOT = Path(__file__).resolve().parent


def api_base_url() -> str:
    return os.getenv("COSMOS3_API_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")


def api_headers(api_key: str) -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def preflight(api_key: str, timeout: int) -> None:
    base = api_base_url()
    response = requests.get(
        f"{base}/models", headers=api_headers(api_key), timeout=min(timeout, 60)
    )
    if not response.ok:
        raise SystemExit(
            f"Cosmos 3 API preflight failed: HTTP {response.status_code}: {response.text[:500]}"
        )
    model_ids = {item.get("id") for item in response.json().get("data", [])}
    if MODEL not in model_ids:
        cosmos_ids = sorted(item for item in model_ids if item and "cosmos" in item.lower())
        raise SystemExit(
            "The selected API backend does not expose nvidia/cosmos3-nano-reasoner. "
            f"Available Cosmos models: {cosmos_ids or 'none'}. "
            "The NVIDIA Build playground is interactive, while the published Cosmos 3 "
            "API currently targets a self-hosted NIM at http://127.0.0.1:8000/v1. "
            "Do not silently substitute Cosmos Reason2 for this Cosmos 3 benchmark."
        )


def read_prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8").strip()


def encode_video(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def request_video(api_key: str, video: Path, fps: float, timeout: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": read_prompt("system_prompt.txt")},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": encode_video(video)}},
                    {"type": "text", "text": read_prompt("user_prompt.txt")},
                ],
            },
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 3000,
        "stream": False,
        "media_io_kwargs": {"video": {"fps": fps}},
    }
    response = requests.post(
        f"{api_base_url()}/chat/completions",
        headers=api_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def output_path(output_root: Path, dataset_root: Path, video: Path) -> Path:
    relative = video.relative_to(dataset_root).with_suffix(".json")
    return output_root / relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reports")
    parser.add_argument("--max-videos", type=int, default=7, help="0 means all videos")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    api_key = os.getenv("NVIDIA_API_KEY", "")
    is_nvidia_hosted = "integrate.api.nvidia.com" in api_base_url()
    if is_nvidia_hosted and not api_key:
        raise SystemExit("NVIDIA_API_KEY is missing. Run set_api_key.ps1, then open a new terminal.")
    if is_nvidia_hosted and not api_key.startswith("nvapi-"):
        raise SystemExit("NVIDIA_API_KEY does not have the expected nvapi- prefix.")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset folder does not exist: {args.dataset}")

    videos = sorted(args.dataset.rglob("*.mp4"))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit(f"No MP4 videos found below {args.dataset}")

    preflight(api_key, args.timeout)
    print(f"model={MODEL} videos={len(videos)} fps={args.fps}")
    for index, video in enumerate(videos, start=1):
        destination = output_path(args.output, args.dataset, video)
        if destination.exists():
            try:
                existing = json.loads(destination.read_text(encoding="utf-8"))
                if existing.get("status") == "ok":
                    print(f"[{index}/{len(videos)}] SKIP {video.name}")
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 1):
            started = time.perf_counter()
            try:
                raw = request_video(api_key, video, args.fps, args.timeout)
                latency = time.perf_counter() - started
                message = raw["choices"][0]["message"]
                record = {
                    "status": "ok",
                    "sample_id": str(video.relative_to(args.dataset).with_suffix("")),
                    "source_file": str(video.resolve()),
                    "model": MODEL,
                    "requested_fps": args.fps,
                    "latency_seconds": round(latency, 3),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "report": message.get("content", ""),
                    "reasoning": message.get("reasoning_content"),
                    "usage": raw.get("usage"),
                }
                destination.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                destination.with_suffix(".md").write_text(record["report"], encoding="utf-8")
                print(f"[{index}/{len(videos)}] OK {video.name} {latency:.1f}s")
                break
            except Exception as exc:
                wait = min(120.0, 10.0 * (2 ** (attempt - 1))) + random.random() * 2
                print(f"[{index}/{len(videos)}] attempt={attempt} error={exc}")
                if attempt == args.retries:
                    failure = {
                        "status": "failed",
                        "source_file": str(video.resolve()),
                        "error": str(exc),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    destination.write_text(
                        json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                else:
                    time.sleep(wait)
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
