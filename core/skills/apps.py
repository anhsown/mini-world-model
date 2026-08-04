"""Kỹ năng: mở ứng dụng và website.

Ví dụ: "mở chrome", "mở máy tính", "mở youtube", "vào facebook"
"""

import subprocess
import webbrowser

# tên (không dấu) -> (lệnh Windows, tên hiển thị)
APPS = {
    "notepad": ("notepad", "Notepad"),
    "may tinh": ("calc", "Máy tính"),
    "calculator": ("calc", "Máy tính"),
    "chrome": ("start chrome", "Chrome"),
    "edge": ("start msedge", "Edge"),
    "trinh duyet": ("start chrome", "trình duyệt"),
    "paint": ("mspaint", "Paint"),
    "word": ("start winword", "Word"),
    "excel": ("start excel", "Excel"),
    "powerpoint": ("start powerpnt", "PowerPoint"),
    "cmd": ("start cmd", "Command Prompt"),
    "terminal": ("start cmd", "Terminal"),
    "explorer": ("explorer", "File Explorer"),
    "thu muc": ("explorer", "File Explorer"),
    "cai dat": ("start ms-settings:", "Cài đặt"),
    "settings": ("start ms-settings:", "Cài đặt"),
    "spotify": ("start spotify:", "Spotify"),
    "zalo": ("start zalo:", "Zalo"),
}

# tên (không dấu) -> URL
WEBSITES = {
    "youtube": "https://www.youtube.com",
    "facebook": "https://www.facebook.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "tiktok": "https://www.tiktok.com",
    "github": "https://github.com",
    "shopee": "https://shopee.vn",
    "bao moi": "https://baomoi.com",
    "vnexpress": "https://vnexpress.net",
}

TRIGGERS = ("mo ", "bat ", "khoi dong ", "open ", "vao ")

# những mục tiêu thuộc kỹ năng khác (nhạc, web...) — nhường lại cho skill web
SKIP_TARGETS = ("nhac", "bai hat", "video")


def handle(text: str, norm: str) -> str | None:
    target = None
    for trigger in TRIGGERS:
        if norm.startswith(trigger):
            target = norm[len(trigger):].strip()
            break
    if not target:
        return None
    if any(target.startswith(s) for s in SKIP_TARGETS):
        return None

    for key, (command, display) in APPS.items():
        if key in target:
            subprocess.Popen(command, shell=True)
            return f"Opening {display} for you, sir."

    for key, url in WEBSITES.items():
        if key in target:
            webbrowser.open(url)
            return f"Opening {key} for you, sir."

    # người dùng đọc thẳng tên miền, ví dụ "mở baomoi.com"
    if "." in target and " " not in target:
        webbrowser.open(f"https://{target}")
        return f"Opening {target}, sir."

    return None  # không biết mở gì -> nhường cho AI / fallback
