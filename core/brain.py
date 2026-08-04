"""Bộ điều phối của Jarvis.

Luồng xử lý một câu nói:
  1. Chuẩn hóa (bỏ dấu, viết thường) để so khớp lệnh dễ dàng
  2. Cho từng kỹ năng thử xử lý (nhắc nhở -> hệ thống -> ứng dụng -> web)
  3. Không kỹ năng nào nhận -> hỏi bộ não AI (nếu có API key)
  4. Chưa có AI -> trả lời dự phòng đơn giản
"""

import unicodedata

from core import ai
from core.skills import apps, reminders, system, web

SKILLS = [reminders, system, apps, web]


def normalize(text: str) -> str:
    """Chuẩn hóa để so khớp lệnh: bỏ dấu tiếng Việt, viết thường, bỏ dấu câu.

    'Mở Chrome giúp tôi!' -> 'mo chrome giup toi'
    "What's the time?"     -> 'whats the time'
    Whisper hay thêm dấu chấm/hỏi và viết hoa -> bước này gỡ hết cho khớp regex.
    """
    s = text.lower().strip().replace("đ", "d")
    s = s.replace("'", "").replace("’", "").replace("`", "")  # bỏ dấu nháy: what's -> whats
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")   # bỏ dấu tiếng Việt
    # đổi dấu câu thành khoảng trắng rồi gộp lại
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.split())


def handle(text: str) -> str:
    norm = normalize(text)

    for skill in SKILLS:
        try:
            reply = skill.handle(text, norm)
        except Exception as e:
            reply = f"Có lỗi khi thực hiện lệnh: {e}"
        if reply is not None:
            return reply

    if ai.available():
        return ai.ask(text)

    return _fallback(norm)


def _fallback(norm: str) -> str:
    """Trả lời đơn giản khi chưa gắn bộ não AI."""
    if any(k in norm for k in ("xin chao", "chao ban", "hello", "hi jarvis", "chao jarvis")):
        return "Good day, sir. How may I be of service?"
    if "ban la ai" in norm or "may la ai" in norm or "gioi thieu" in norm:
        return (
            "I am JARVIS, your personal assistant, sir. I can open applications, "
            "tell the time, check the weather and the news, set reminders and take notes."
        )
    if "cam on" in norm or "thank" in norm:
        return "You're most welcome, sir."
    if "lam duoc gi" in norm or "giup gi" in norm or "tro giup" in norm:
        return (
            "At your service, sir. I can open applications and websites, tell the time "
            "and date, adjust the volume, search Google, play music on YouTube, "
            "report the weather, read the news, check exchange rates, "
            "set reminders and take notes."
        )

    return (
        "I'm afraid I don't understand that yet, sir. My AI brain has not been "
        "installed — say 'tìm kiếm' followed by your query to search Google, "
        "or add a free AI key to the .env file to upgrade me. See the README, sir."
    )
