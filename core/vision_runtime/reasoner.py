from __future__ import annotations

from abc import ABC, abstractmethod
import os
from time import perf_counter
from typing import Sequence

import numpy as np
from PIL import Image

from .config import ReasonerConfig
from .memory import FrameObservation
from .schemas import ReasonerResult, VisualEvidence, parse_reasoner_result


SYSTEM_PROMPT = """You are the visual Reasoner of a robot observing the real world.
Answer in the same language as the user's question. Use only visible evidence. If the
target is unclear, occluded, or too blurry, abstain instead of guessing.
Return EXACTLY one compact JSON line. No markdown, no code fence, no chain-of-thought.
Use at most 3 UNIQUE short attributes and then stop. bbox is [x1,y1,x2,y2] in the LAST
frame using coordinates 0..1000, or null. confidence is 0..1.
Exact schema: {"answer":"short answer","bbox":[x1,y1,x2,y2],"attributes":["visible fact"],"confidence":0.0,"abstain":false}"""


class VisionLanguageReasoner(ABC):
    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def analyze(self, frames: Sequence[FrameObservation], query: str) -> ReasonerResult: ...


class Qwen3VLReasoner(VisionLanguageReasoner):
    def __init__(self, config: ReasonerConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.loaded:
            return
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Qwen3-VL requires transformers>=4.57, accelerate and bitsandbytes for 4-bit loading."
            ) from exc

        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        # The installed torch/cuDNN pair crashes in Qwen's visual encoder while
        # resolving cudnnGetLibConfig. CUDA remains enabled; this selects the
        # stable non-cuDNN kernels for the visual convolution path.
        torch.backends.cudnn.enabled = False
        model_config = AutoConfig.from_pretrained(self.config.model_id)
        kwargs = {
            "config": model_config,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "dtype": torch.float16,
            "attn_implementation": "sdpa",
            "max_memory": {0: self.config.max_gpu_memory, "cpu": self.config.max_cpu_memory},
        }
        if self.config.load_in_4bit and self.config.prequantized:
            quantization_config = dict(getattr(model_config, "quantization_config", {}) or {})
            quantization_config["llm_int8_enable_fp32_cpu_offload"] = False
            quantization_config["bnb_4bit_compute_dtype"] = "float16"
            model_config.quantization_config = quantization_config
            kwargs["device_map"] = {"": 0}
            kwargs.pop("max_memory", None)
        elif self.config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.config.model_id, **kwargs)
        self._model.eval()

    def analyze(self, frames: Sequence[FrameObservation], query: str) -> ReasonerResult:
        if not frames:
            raise ValueError("at least one frame is required")
        if not query.strip():
            raise ValueError("query is required")
        self.load()
        import torch

        content: list[dict] = []
        for offset, frame in enumerate(frames):
            content.append({"type": "text", "text": f"Khung {offset + 1}/{len(frames)}, timestamp={frame.timestamp}"})
            content.append({"type": "image", "image": _resize_image(frame.image, self.config.max_image_edge)})
        content.append({"type": "text", "text": f"Câu hỏi của người dùng: {query.strip()}"})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

        started = perf_counter()
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        try:
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=4,
                )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "GPU hết bộ nhớ. Hãy giảm max_image_edge/reasoning_frames hoặc bật 4-bit."
            ) from exc
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        latency_ms = (perf_counter() - started) * 1000.0
        return parse_reasoner_result(
            text,
            model_id=self.config.model_id,
            latency_ms=latency_ms,
            frame_count=len(frames),
        )


class MockReasoner(VisionLanguageReasoner):
    """Deterministic reasoner used to verify the complete demo without model downloads."""

    loaded = True

    def load(self) -> None:
        return None

    def analyze(self, frames: Sequence[FrameObservation], query: str) -> ReasonerResult:
        if not frames:
            raise ValueError("at least one frame is required")
        image = frames[-1].image[:, :, :3].astype(np.float32)
        channel = int(np.argmax(image.mean(axis=(0, 1))))
        color = ["đỏ", "xanh lá", "xanh dương"][channel]
        return ReasonerResult(
            answer=f"Tôi quan sát thấy một vùng màu {color} nổi bật.",
            object_name=f"vật màu {color}",
            evidence=VisualEvidence(
                bbox_norm=(0.15, 0.15, 0.85, 0.85),
                visual_attributes=[f"màu chủ đạo: {color}"],
                visible=True,
            ),
            confidence=0.75,
            abstain=False,
            rationale_summary="Màu chủ đạo trong khung cuối được dùng làm bằng chứng kiểm thử.",
            raw_text="mock",
            model_id="mock-color-reasoner",
            latency_ms=0.0,
            frame_count=len(frames),
        )


def _resize_image(image: np.ndarray, max_edge: int) -> Image.Image:
    pil = Image.fromarray(image[:, :, :3].astype(np.uint8), mode="RGB")
    if max(pil.size) <= max_edge:
        return pil
    scale = max_edge / max(pil.size)
    size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
    return pil.resize(size, Image.Resampling.LANCZOS)
