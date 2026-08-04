"""Train bộ não JARVIS NGAY TRÊN MÁY BẠN (GPU GTX 1650 4GB).

Kỹ thuật: LoRA fine-tune (fp16, không cần bitsandbytes/Ollama) — distill phong
cách JARVIS từ dataset.jsonl (do Claude viết) vào model nhỏ Qwen2.5.

Chạy:
    python training/train_local.py

Xong sẽ có model gộp tại training/jarvis-model/ — chạy bằng training/serve_brain.py.
"""

import json
import os
import sys

os.environ["USE_TF"] = "0"          # không nạp TensorFlow (không cần, tránh xung đột)
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---- Cấu hình (đổi ở đây nếu muốn) ----
# 0.5B: nhẹ, nhanh, chắc chắn vừa 4GB. Muốn thông minh hơn (và có 6GB+ VRAM):
#       đổi sang "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ = 1024
EPOCHS = 3
LR = 2e-4

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset.jsonl")
OUT_DIR = os.path.join(HERE, "jarvis-model")
ADAPTER_DIR = os.path.join(HERE, "jarvis-lora")


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not os.path.exists(DATASET):
        print("❌ Chưa có dataset.jsonl. Chạy 'python training/build_dataset.py' trước.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("⚠️  Không thấy GPU — sẽ train bằng CPU (rất chậm).")
    else:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {name} ({vram:.1f} GB)")

    print(f"📚 Model gốc (thầy đã dạy qua dataset): {BASE_MODEL}")
    print("⏳ Lần đầu sẽ tải model gốc (~1 GB cho 0.5B)...\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA: chỉ train ~1% tham số -> nhẹ VRAM, không phá kiến thức gốc
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- Nạp & mã hóa dữ liệu theo chat template của Qwen ----
    rows = [json.loads(l) for l in open(DATASET, encoding="utf-8") if l.strip()]
    print(f"\n📊 {len(rows)} mẫu hội thoại")

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        out = tokenizer(text, truncation=True, max_length=MAX_SEQ)
        return out

    dataset = Dataset.from_list(rows).map(
        tokenize, remove_columns=["messages"], desc="Mã hóa"
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=os.path.join(HERE, "_checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
        weight_decay=0.01,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("\n🚀 Bắt đầu huấn luyện...\n")
    trainer.train()

    # ---- Lưu LoRA (adapter) ----
    print("\n💾 Đang lưu adapter LoRA...")
    model.save_pretrained(ADAPTER_DIR)

    del model, trainer
    torch.cuda.empty_cache()

    # ---- Gộp vào model gốc (trên GPU — fp16 trên CPU hay segfault) ----
    print("💾 Đang gộp thành model độc lập...")
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload().half()
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"\n✅ XONG! Bộ não JARVIS của bạn nằm ở: {OUT_DIR}")
    print("   Chạy nó bằng:  python training/serve_brain.py")


if __name__ == "__main__":
    main()
