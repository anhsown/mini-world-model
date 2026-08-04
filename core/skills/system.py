"""Kỹ năng: giờ, ngày tháng, âm lượng, khóa màn hình.

Bạn ra lệnh tiếng Việt — JARVIS trả lời tiếng Anh (giọng phim).
Ví dụ: "mấy giờ rồi", "hôm nay thứ mấy", "tăng âm lượng",
       "âm lượng 50", "tắt tiếng", "khóa màn hình"
"""

import ctypes
import re
from datetime import datetime

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def spoken_time(hour: int, minute: int) -> str:
    """15:05 -> '3:05 PM' (kiểu nói giờ của người Anh)."""
    period = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    if minute == 0:
        return f"{h12} o'clock {period}"
    return f"{h12}:{minute:02d} {period}"


def handle(text: str, norm: str) -> str | None:
    # --- Giờ ---
    if any(k in norm for k in ("may gio", "gio hien tai", "bay gio la may",
                              "what time", "the time", "time is it", "current time")):
        now = datetime.now()
        return f"It is currently {spoken_time(now.hour, now.minute)}, sir."

    # --- Ngày / thứ ---
    if any(k in norm for k in ("ngay may", "ngay bao nhieu", "thu may", "hom nay la",
                              "what day", "what date", "what is the date", "whats the date",
                              "today date", "day is it", "date today")):
        now = datetime.now()
        return (
            f"Today is {WEEKDAYS[now.weekday()]}, "
            f"{MONTHS[now.month - 1]} {now.day}, {now.year}."
        )

    # --- Âm lượng (kiểm tra unmute TRƯỚC vì "unmute" chứa "mute") ---
    if any(k in norm for k in ("mo tieng", "bat tieng", "bo tat tieng", "unmute")):
        return _set_mute(False)
    if any(k in norm for k in ("tat tieng", "cam tieng", "mute")):
        return _set_mute(True)

    match = re.search(r"(?:am luong|volume|set volume to) (\d{1,3})", norm)
    if match:
        level = min(100, int(match.group(1)))
        return _set_volume(level / 100)
    if any(k in norm for k in ("tang am luong", "to len", "volume up", "louder", "turn it up")):
        return _change_volume(+0.1)
    if any(k in norm for k in ("giam am luong", "nho lai", "nho xuong",
                              "volume down", "quieter", "turn it down")):
        return _change_volume(-0.1)
    if any(k in norm for k in ("am luong toi da", "am luong lon nhat",
                              "max volume", "maximum volume", "full volume")):
        return _set_volume(1.0)

    # --- Khóa màn hình ---
    if any(k in norm for k in ("khoa man hinh", "khoa may", "lock screen",
                              "lock the screen", "lock the computer")):
        ctypes.windll.user32.LockWorkStation()
        return "Locking the screen, sir."

    return None


def _volume_interface():
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    speakers = AudioUtilities.GetSpeakers()
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _set_volume(level: float) -> str:
    try:
        volume = _volume_interface()
        volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level)), None)
        return f"Volume set to {round(level * 100)} percent."
    except Exception:
        return "I'm afraid I cannot adjust the volume on this machine, sir."


def _change_volume(delta: float) -> str:
    try:
        volume = _volume_interface()
        current = volume.GetMasterVolumeLevelScalar()
        new = max(0.0, min(1.0, current + delta))
        volume.SetMasterVolumeLevelScalar(new, None)
        if delta > 0:
            return f"Volume increased to {round(new * 100)} percent."
        return f"Volume decreased to {round(new * 100)} percent."
    except Exception:
        return "I'm afraid I cannot adjust the volume on this machine, sir."


def _set_mute(mute: bool) -> str:
    try:
        volume = _volume_interface()
        volume.SetMute(1 if mute else 0, None)
        return "Audio muted, sir." if mute else "Audio restored, sir."
    except Exception:
        return "I'm afraid I cannot control the audio on this machine, sir."
