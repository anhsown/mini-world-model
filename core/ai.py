"""Bộ não AI của Jarvis — chuẩn OpenAI-compatible, KHÔNG khóa vào provider nào.

Bất kỳ dịch vụ nào nói chuẩn này đều cắm vào được (chỉ cần sửa config.py + .env):

  Groq (miễn phí, siêu nhanh, model mở):
      AI_BASE_URL = "https://api.groq.com/openai/v1"
      AI_MODEL    = "llama-3.3-70b-versatile"
  OpenRouter (một key dùng trăm model):
      AI_BASE_URL = "https://openrouter.ai/api/v1"
  Google Gemini:
      AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
      AI_MODEL    = "gemini-2.5-flash"
  Anthropic Claude:
      AI_BASE_URL = "https://api.anthropic.com/v1"
      AI_MODEL    = "claude-opus-4-8"
  Ollama chạy local (nếu sau này nâng cấp máy):
      AI_BASE_URL = "http://localhost:11434/v1"    (AI_API_KEY=ollama)

Kích hoạt: tạo file .env trong thư mục Jarvis với nội dung:
      AI_API_KEY=<key của dịch vụ bạn chọn>
"""

import os
import sys

import requests

import config

SYSTEM_PROMPT = (
    "You are JARVIS, the AI assistant from Iron Man — refined, calm, subtly witty, "
    "with impeccable British manners. Address the user as 'sir'. "
    "The user may speak or type in Vietnamese; understand it, but ALWAYS reply in English. "
    "Your replies will be READ ALOUD by a text-to-speech voice, so: keep them short "
    "(1-3 sentences), natural spoken English, no markdown, no special symbols, "
    "no bullet lists, no code unless explicitly requested."
)

_history: list[dict] = []


def _find_api_key() -> str | None:
    key = os.environ.get("AI_API_KEY")
    if key:
        return key.strip()
    # Đọc file .env trong thư mục gốc của project nếu có
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("AI_API_KEY"):
                        _, _, value = line.partition("=")
                        value = value.strip().strip('"').strip("'")
                        if value:
                            return value
        except OSError:
            pass
    return None


def _is_local() -> bool:
    url = getattr(config, "AI_BASE_URL", "")
    return "localhost" in url or "127.0.0.1" in url


def _local_port() -> int:
    url = getattr(config, "AI_BASE_URL", "")
    try:
        return int(url.split(":")[2].split("/")[0])
    except (IndexError, ValueError):
        return 8080


def _server_up() -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", _local_port())) == 0
    finally:
        s.close()


def ensure_local_server() -> None:
    """Nếu dùng bộ não local mà server chưa chạy -> tự bật ngầm."""
    if not _is_local() or _server_up():
        return
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "training", "serve_brain.py")
    if not os.path.exists(script):
        return
    env = dict(os.environ, USE_TF="0", USE_FLAX="0",
               TRANSFORMERS_NO_ADVISORY_WARNINGS="1")
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    print("🧠 Đang bật bộ não JARVIS (lần đầu mất ~10 giây nạp model)...")
    subprocess.Popen([sys.executable, script], cwd=root, env=env,
                    creationflags=flags,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def available() -> bool:
    # Server local (bộ não bạn tự train) không cần API key
    return _is_local() or _find_api_key() is not None


def _post(payload):
    return requests.post(
        config.AI_BASE_URL.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {_find_api_key() or 'local'}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )


def ask(text: str) -> str:
    """Gửi câu hỏi tới AI (chuẩn OpenAI chat completions), giữ ngữ cảnh hội thoại."""
    _history.append({"role": "user", "content": text})
    payload = {
        "model": config.AI_MODEL,
        "max_tokens": config.AI_MAX_TOKENS,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + _history,
    }

    try:
        response = _post(payload)
    except requests.RequestException:
        # Bộ não local có thể đã tắt -> tự bật lại rồi thử lại một lần
        if _is_local():
            import time
            ensure_local_server()
            for _ in range(12):
                time.sleep(1)
                if _server_up():
                    break
            try:
                response = _post(payload)
            except requests.RequestException:
                _history.pop()
                return "I'm just waking up, sir. Give me a moment and ask again."
        else:
            _history.pop()
            return "I cannot reach the AI server, sir. Please check the connection."

    if response.status_code == 401:
        _history.pop()
        return "The API key appears to be invalid, sir. Do check the .env file."
    if response.status_code == 429:
        _history.pop()
        return "I'm being rate limited, sir. Please try again in a moment."
    if response.status_code >= 400:
        _history.pop()
        return f"The AI server is having difficulties, sir. Error code {response.status_code}."

    try:
        reply = response.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError):
        _history.pop()
        return "The AI server returned unexpected data, sir."

    if not reply:
        _history.pop()
        return "I'm afraid I have no answer to that, sir. Perhaps rephrase the question."

    _history.append({"role": "assistant", "content": reply})

    # Cắt bớt lịch sử cũ (bỏ theo cặp hỏi-đáp để luôn bắt đầu bằng user)
    while len(_history) > config.AI_HISTORY_LIMIT:
        del _history[:2]

    return reply
