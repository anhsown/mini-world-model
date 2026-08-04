"""Miệng của Jarvis — chuyển văn bản thành giọng nói tiếng Việt.

Dùng edge-tts (miễn phí, giọng Microsoft) để tạo file mp3,
rồi phát bằng pygame. Cần kết nối internet để tạo giọng nói.
"""

import asyncio
import os
import tempfile
import threading
import time
import uuid

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import config

_lock = threading.Lock()      # tránh 2 luồng nói chồng lên nhau
_mixer_ready = None


def speak(text: str) -> None:
    """In ra màn hình, bung HUD nếu đang ẩn, và đọc thành tiếng."""
    from core import hud, telemetry

    print(f"\n🤖 Jarvis: {text}\n")
    with _lock:
        synthesis_started = time.perf_counter()
        try:
            path = _synthesize(text)
        except Exception as e:
            telemetry.event(
                "tts_failed",
                text=text,
                error=str(e),
                synthesis_ms=round((time.perf_counter() - synthesis_started) * 1000.0, 2),
            )
            print(f"   (Không tạo được giọng nói — kiểm tra mạng: {e})")
            return
        synthesis_ms = round((time.perf_counter() - synthesis_started) * 1000.0, 2)

        popped_up = False
        if not hud.is_visible():
            hud.show()
            popped_up = True
        hud.set_state("speaking")
        playback_started = time.perf_counter()
        try:
            _play(path)
        finally:
            telemetry.event(
                "tts_completed",
                text=text,
                synthesis_ms=synthesis_ms,
                playback_ms=round((time.perf_counter() - playback_started) * 1000.0, 2),
            )
            hud.set_state("listening")
            if popped_up:
                hud.hide()
            try:
                os.remove(path)
            except OSError:
                pass


def _synthesize(text: str) -> str:
    import edge_tts

    path = os.path.join(tempfile.gettempdir(), f"jarvis_{uuid.uuid4().hex}.mp3")

    async def run():
        await asyncio.wait_for(
            edge_tts.Communicate(
                text, config.VOICE, rate=config.SPEECH_RATE, pitch=config.PITCH
            ).save(path),
            timeout=float(getattr(config, "TTS_SYNTHESIS_TIMEOUT", 10)),
        )

    asyncio.run(run())
    return path


def _init_mixer() -> bool:
    global _mixer_ready
    if _mixer_ready is None:
        try:
            import pygame
            pygame.mixer.init()
            _mixer_ready = True
        except Exception:
            _mixer_ready = False
    return _mixer_ready


def _play(path: str) -> None:
    if not _init_mixer():
        print("   (Không tìm thấy thiết bị âm thanh — chỉ hiển thị chữ)")
        return
    import pygame

    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
