"""Gộp adapter LoRA vào model gốc -> model độc lập chạy được.

Tách riêng khỏi train_local.py để: (1) chạy lại nhanh nếu bước gộp lỗi mà không
phải train lại, (2) gộp trên GPU (fp16 chạy mượt; fp16 trên CPU hay segfault).

Chạy:  python training/merge_model.py
"""

import os
import sys

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(HERE, "jarvis-lora")
OUT_DIR = os.path.join(HERE, "jarvis-model")


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(ADAPTER_DIR):
        print(f"❌ Chưa có adapter tại {ADAPTER_DIR}. Train trước: python training/train_local.py")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"💾 Gộp adapter vào model gốc trên {device.upper()}...")

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR)
    merged = merged.merge_and_unload()
    merged = merged.half()  # lưu fp16 cho gọn (~1 GB)

    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(OUT_DIR)

    print(f"\n✅ XONG! Bộ não JARVIS của bạn: {OUT_DIR}")
    print("   Chạy nó bằng:  python training/serve_brain.py")


if __name__ == "__main__":
    main()
