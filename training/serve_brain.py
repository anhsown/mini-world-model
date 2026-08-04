"""Chạy bộ não JARVIS (model bạn tự train) như một server nội bộ.

Mở endpoint chuẩn OpenAI (/v1/chat/completions) tại localhost -> Jarvis cắm vào
không cần sửa code. KHÔNG cần Ollama, không cần internet, không key.

Chạy:
    python training/serve_brain.py

Rồi trong config.py đặt:
    AI_BASE_URL = "http://localhost:8080/v1"
    AI_MODEL = "jarvis"
(và AI_API_KEY=local trong .env — hoặc bỏ qua, server không kiểm tra key)
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "jarvis-model")
PORT = 8080

_model = None
_tokenizer = None
_gen_lock = threading.Lock()


def load_model():
    global _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(MODEL_DIR):
        print(f"❌ Chưa có model tại {MODEL_DIR}")
        print("   Hãy train trước:  python training/train_local.py")
        sys.exit(1)

    print("⏳ Đang nạp bộ não JARVIS...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    _model.eval()
    print(f"✅ Sẵn sàng trên {device.upper()} — server chạy tại http://localhost:{PORT}/v1")
    print("   Để nguyên cửa sổ này. Mở Jarvis ở cửa sổ khác.\n")


def generate(messages, max_tokens=500, temperature=0.7):
    device = next(_model.parameters()).device
    inputs = _tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with _gen_lock, torch.no_grad():
        out = _model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.eos_token_id,
        )
    return _tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # tắt log rườm rà

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            messages = payload["messages"]
            max_tokens = int(payload.get("max_tokens", 500))
            temperature = float(payload.get("temperature", 0.7))
        except (ValueError, KeyError):
            self._send(400, {"error": "bad request"})
            return

        try:
            reply = generate(messages, max_tokens, temperature)
        except Exception as e:
            self._send(500, {"error": str(e)})
            return

        self._send(200, {
            "id": "chatcmpl-jarvis",
            "object": "chat.completion",
            "model": "jarvis",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
        })


def main():
    load_model()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")


if __name__ == "__main__":
    main()
