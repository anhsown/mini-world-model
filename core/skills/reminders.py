"""Kỹ năng: nhắc nhở, hẹn giờ và ghi chú.

Bạn ra lệnh tiếng Việt — JARVIS trả lời tiếng Anh (giọng phim).
Ví dụ: "nhắc tôi uống nước sau 30 phút"
       "nhắc tôi họp lúc 3 giờ chiều"
       "hẹn giờ 5 phút"
       "ghi chú mua sữa cho mẹ" / "đọc ghi chú" / "xóa ghi chú"
       "danh sách nhắc nhở"
"""

import json
import os
import re
import threading
from datetime import datetime, timedelta

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)
NOTES_FILE = os.path.join(DATA_DIR, "notes.txt")
REMINDERS_FILE = os.path.join(DATA_DIR, "reminders.json")

_speak = print          # jarvis.py sẽ thay bằng voice.speak khi khởi động
_timers: list[threading.Timer] = []


def set_speaker(fn) -> None:
    global _speak
    _speak = fn


def _spoken_time(when: datetime) -> str:
    period = "AM" if when.hour < 12 else "PM"
    h12 = when.hour % 12 or 12
    if when.minute == 0:
        return f"{h12} o'clock {period}"
    return f"{h12}:{when.minute:02d} {period}"


# ----------------------------------------------------------------
#  Xử lý lệnh
# ----------------------------------------------------------------

def handle(text: str, norm: str) -> str | None:
    # --- Ghi chú: đọc / xóa trước, rồi mới đến thêm ---
    if any(k in norm for k in ("doc ghi chu", "xem ghi chu", "co ghi chu gi",
                              "read my note", "read note", "my notes", "show my note",
                              "show note")):
        return _read_notes()
    if any(k in norm for k in ("xoa ghi chu", "xoa het ghi chu",
                              "clear note", "clear my note", "delete note",
                              "delete all note", "delete my note")):
        return _clear_notes()

    note = _capture(norm, r"^(?:ghi chu|ghi lai|luu y|note|take a note|write down) (.+)$")
    if note:
        return f"Note saved, sir: {note}." if _add_note(note) else None

    # --- Danh sách nhắc nhở ---
    if any(k in norm for k in ("danh sach nhac", "co nhac nho gi", "xem nhac nho",
                              "my reminders", "list reminder", "show reminder",
                              "what reminder")):
        return _list_reminders()

    # --- Hẹn giờ: "hẹn giờ 5 phút" / "set a timer for 5 minutes" ---
    match = re.match(
        r"^(?:hen gio|dat gio|dem nguoc|set a timer for|set timer for|timer for|set a timer of) "
        r"(\d+) (giay|phut|gio|tieng|seconds?|minutes?|hours?)$", norm)
    if match:
        seconds = _to_seconds(int(match.group(1)), match.group(2))
        return _schedule("Time is up!", datetime.now() + timedelta(seconds=seconds))

    # --- Nhắc sau X phút: "nhắc tôi uống nước sau 30 phút" / "remind me to drink in 30 minutes" ---
    match = re.match(
        r"^(?:nhac toi|nhac tui|nhac minh|nhac nho|nhac)\s?(.*?)\s?sau (\d+) "
        r"(giay|phut|gio|tieng)$", norm)
    if match and match.group(2):
        content = match.group(1).strip() or "your task"
        seconds = _to_seconds(int(match.group(2)), match.group(3))
        return _schedule(content, datetime.now() + timedelta(seconds=seconds))
    match = re.match(
        r"^remind me (?:to )?(.*?) in (\d+) (seconds?|minutes?|hours?)$", norm)
    if match:
        content = match.group(1).strip() or "your task"
        seconds = _to_seconds(int(match.group(2)), match.group(3))
        return _schedule(content, datetime.now() + timedelta(seconds=seconds))

    # --- Nhắc lúc H giờ M: "nhắc tôi họp lúc 3 giờ 30 chiều" ---
    match = re.match(
        r"^(?:nhac toi|nhac tui|nhac minh|nhac nho|dat bao thuc|nhac)\s?(.*?)\s?"
        r"(?:luc|vao luc|vao) (\d{1,2}) gio(?: (\d{1,2}))?(?: phut)?"
        r"( sang| trua| chieu| toi| dem)?$", norm)
    if match:
        content = match.group(1).strip() or "your task"
        when = _resolve_time(int(match.group(2)), int(match.group(3) or 0), match.group(4))
        return _schedule(content, when)
    # "remind me to X at 3 30 pm" / "remind me to X at 8 pm"
    match = re.match(
        r"^remind me (?:to )?(.*?) at (\d{1,2})(?:[: ](\d{2}))? ?(am|pm)?$", norm)
    if match:
        content = match.group(1).strip() or "your task"
        period = {"am": "sang", "pm": "chieu"}.get(match.group(4) or "", "")
        when = _resolve_time(int(match.group(2)), int(match.group(3) or 0), period)
        return _schedule(content, when)

    return None


# ----------------------------------------------------------------
#  Nhắc nhở
# ----------------------------------------------------------------

def _schedule(content: str, when: datetime) -> str:
    delay = (when - datetime.now()).total_seconds()
    if delay <= 0:
        return "I'm afraid that moment has already passed, sir. Please pick another time."

    timer = threading.Timer(delay, _fire, args=(content, when))
    timer.daemon = True
    timer.start()
    _timers.append(timer)
    _save_reminder(content, when)

    return f"Reminder set, sir: {content}, at {_spoken_time(when)}."


def _fire(content: str, when: datetime) -> None:
    _mark_done(content, when)
    _speak(f"Sir, this is your reminder: {content}!")


def _resolve_time(hour: int, minute: int, period: str | None) -> datetime:
    period = (period or "").strip()
    if period in ("chieu", "toi", "dem") and hour < 12:
        hour += 12
    now = datetime.now()
    when = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
    # Không nói rõ sáng/chiều và giờ đó đã qua -> hiểu là buổi chiều/tối hoặc ngày mai
    if when <= now and not period and hour < 12:
        when += timedelta(hours=12)
    if when <= now:
        when += timedelta(days=1)
    return when


def _to_seconds(amount: int, unit: str) -> int:
    unit = unit.rstrip("s")  # seconds -> second
    factor = {"giay": 1, "phut": 60, "gio": 3600, "tieng": 3600,
              "second": 1, "minute": 60, "hour": 3600}
    return amount * factor[unit]


def _load_reminders() -> list[dict]:
    try:
        with open(REMINDERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _write_reminders(items: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _save_reminder(content: str, when: datetime) -> None:
    items = _load_reminders()
    items.append({"content": content, "when": when.isoformat(), "done": False})
    _write_reminders(items)


def _mark_done(content: str, when: datetime) -> None:
    items = _load_reminders()
    for item in items:
        if item["content"] == content and item["when"] == when.isoformat():
            item["done"] = True
    _write_reminders(items)


def _list_reminders() -> str:
    pending = [i for i in _load_reminders() if not i["done"]]
    if not pending:
        return "You have no pending reminders, sir."
    parts = []
    for item in pending[:5]:
        when = datetime.fromisoformat(item["when"])
        parts.append(f"{item['content']} at {_spoken_time(when)}")
    label = "reminder" if len(pending) == 1 else "reminders"
    return f"You have {len(pending)} {label}, sir: " + ". ".join(parts) + "."


def restore_pending() -> None:
    """Khi khởi động lại Jarvis, đặt lại các nhắc nhở chưa tới giờ."""
    items = _load_reminders()
    changed = False
    for item in items:
        if item["done"]:
            continue
        when = datetime.fromisoformat(item["when"])
        if when <= datetime.now():
            item["done"] = True  # đã lỡ trong lúc Jarvis tắt
            changed = True
            continue
        timer = threading.Timer(
            (when - datetime.now()).total_seconds(), _fire, args=(item["content"], when)
        )
        timer.daemon = True
        timer.start()
        _timers.append(timer)
    if changed:
        _write_reminders(items)


# ----------------------------------------------------------------
#  Ghi chú
# ----------------------------------------------------------------

def _add_note(note: str) -> bool:
    os.makedirs(DATA_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {note}\n")
    return True


def _read_notes(limit: int = 5) -> str:
    try:
        with open(NOTES_FILE, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        lines = []
    if not lines:
        return "You have no notes yet, sir."
    recent = lines[-limit:]
    spoken = ". ".join(re.sub(r"^\[.*?\]\s*", "", line) for line in recent)
    label = "note" if len(lines) == 1 else "notes"
    return f"You have {len(lines)} {label}, sir. The most recent: {spoken}."


def _clear_notes() -> str:
    try:
        os.remove(NOTES_FILE)
    except OSError:
        pass
    return "All notes have been deleted, sir."


# ----------------------------------------------------------------
#  Tiện ích
# ----------------------------------------------------------------

def _capture(norm: str, pattern: str) -> str | None:
    """Khớp pattern trên chuỗi đã chuẩn hóa, trả về nhóm bắt được."""
    match = re.match(pattern, norm)
    if not match:
        return None
    return match.group(1).strip() or None
