from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ReasonerConfig:
    model_id: str = "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit"
    base_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    load_in_4bit: bool = True
    prequantized: bool = True
    max_new_tokens: int = 160
    max_image_edge: int = 448
    max_gpu_memory: str = "2300MiB"
    max_cpu_memory: str = "8GiB"


@dataclass
class ASRConfig:
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "vi"


@dataclass
class DemoConfig:
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    memory_frames: int = 8
    reasoning_frames: int = 3
    local_camera_index: int = 0
    auto_capture_local: bool = True
    session_root: str = "data/real_world_sessions"
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    mock_reasoner: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DemoConfig":
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        return cls(
            reasoner=ReasonerConfig(**payload.get("reasoner", {})),
            asr=ASRConfig(**payload.get("asr", {})),
            **{key: value for key, value in payload.items() if key not in {"reasoner", "asr"}},
        )
