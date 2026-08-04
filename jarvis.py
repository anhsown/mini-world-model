"""J.A.R.V.I.S. — Trợ lý ảo giọng nói (ra lệnh tiếng Việt, trả lời giọng Anh như phim).

Chạy:  python jarvis.py

Wake word:
  - Jarvis chạy nền im lặng (chế độ CHỜ 💤)
  - Nói "Hey Jarvis"            -> tỉnh dậy, trả lời "Yes, sir?"
  - Nói "Hey Jarvis, mấy giờ rồi" -> thực hiện luôn, không cần chờ
  - Im lặng một lúc / "ngủ đi"  -> quay về chế độ chờ
  - "goodbye" / "goodnight"     -> thoát hẳn chương trình

Chế độ khác:
  - Không có micro -> tự chuyển sang gõ phím
  - "chế độ gõ" / "chế độ nói" để chuyển đổi thủ công
"""

import os
import socket
import sys
from datetime import datetime
from uuid import uuid4

# Chạy ẩn khi khởi động cùng Windows -> không có terminal, chuyển print vào devnull
if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stdout or _devnull
    sys.stderr = sys.stderr or _devnull
# Đảm bảo tiếng Việt hiển thị đúng trên terminal Windows
elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from core import ai, brain, hud, speech, telemetry, vision, voice, wake
from core.runtime import AssistantRuntime, AssistantState
from core.skills import reminders

# Thoát hẳn chương trình: "goodbye" / "goodnight" (kèm các cách Google nghe nhầm)
QUIT_WORDS = ("goodbye", "good bye", "gut bai", "tat jarvis", "thoat", "exit", "quit", "shutdown")
NIGHT_WORDS = ("goodnight", "good night", "gut nai", "chuc ngu ngon")
# Quay về chế độ chờ (chỉ có tác dụng ở chế độ giọng nói + wake word)
SLEEP_WORDS = ("ngu di", "tam biet", "nghi di", "standby", "bye")

BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
      "Hey Jarvis" — At your service, sir.
"""


def greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        part = "Good morning"
    elif 12 <= hour < 18:
        part = "Good afternoon"
    else:
        part = "Good evening"
    return f"{part}, sir. JARVIS at your service."


def strip_wake_word(text: str, norm: str) -> tuple[bool, str]:
    """Kiểm tra câu có chứa wake word không.

    Trả về (có_wake_word, phần_lệnh_còn_lại_từ_câu_gốc).
    Ví dụ: "Hey Jarvis mấy giờ rồi" -> (True, "mấy giờ rồi")
    """
    for wake in config.WAKE_WORDS:
        pos = norm.find(wake)
        if pos != -1:
            remainder = text[pos + len(wake):].strip(" ,.!?")
            return True, remainder
    return False, text


def print_standby_hint() -> None:
    # Standby is a wake/listening state, not a resource destructor. Camera
    # lifetime is controlled only by explicit camera/sleep/shutdown commands.
    hud.hide()
    engine = "engine chuyên dụng offline" if wake.available() else "Google STT dự phòng"
    print(f"💤 Chế độ chờ — nói 'Hey Jarvis' để gọi tôi dậy ({engine}, Ctrl+C để thoát)")


def handle_command(text: str, norm: str, mode: str) -> str:
    """Xử lý một câu lệnh. Trả về trạng thái tiếp theo: 'active' | 'standby' | 'quit' | mode mới."""
    if any(word in norm for word in NIGHT_WORDS):
        vision.shutdown()
        voice.speak("Goodnight, sir.")
        return "quit"

    if any(word in norm for word in QUIT_WORDS):
        vision.shutdown()
        voice.speak("Goodbye, sir.")
        return "quit"

    if any(word in norm for word in SLEEP_WORDS):
        # Standby closes only the sensor session; the isolated reasoner stays
        # warm so the next camera request is fast. Full shutdown unloads it.
        vision.stop()
        if mode == "voice" and config.WAKE_WORD_ENABLED:
            voice.speak("Entering standby, sir. Call me when you need me.")
            return "standby"
        voice.speak("Goodbye, sir.")
        return "quit"

    if "che do go" in norm:
        voice.speak("Switching to keyboard mode, sir.")
        return "text"
    if "che do noi" in norm:
        if speech.mic_available():
            voice.speak("Voice mode activated, sir.")
            return "voice"
        voice.speak("I'm afraid I cannot find a microphone, sir.")
        return "active"

    if vision.is_close_camera_command(norm):
        reply = vision.stop()
        hud.set_transcript(text, reply)
        voice.speak(reply)
        return "active"

    if vision.is_open_camera_command(norm):
        try:
            reply = vision.start()
        except Exception as exc:
            reply = f"I cannot access the camera, sir. {exc}"
        hud.set_transcript(text, reply)
        voice.speak(reply)
        return "active"

    if vision.active():
        reply = vision.ask(text)
        voice.speak(reply)
        return "active"

    reply = brain.handle(text)
    if reply:
        hud.set_transcript(text, reply)
        voice.speak(reply)
    return "active"


def main() -> None:
    print(BANNER)

    reminders.set_speaker(voice.speak)
    reminders.restore_pending()
    ai.ensure_local_server()  # tự bật bộ não local nếu config trỏ localhost
    wake.preload()            # nạp sẵn engine đánh thức "Hey Jarvis"
    vision.preload_async()    # warm model thị giác trước khi người dùng mở camera
    speech.preload()          # nạp sẵn Whisper (nhận diện lệnh offline)

    if speech.mic_available():
        mode = "voice"
    else:
        mode = "text"
        print("⌨️  Không tìm thấy micro — chạy chế độ GÕ PHÍM\n")

    voice.speak(greeting())

    runtime = AssistantRuntime(
        voice_mode=mode == "voice",
        wake_enabled=bool(config.WAKE_WORD_ENABLED),
        silence_limit=int(config.SILENCE_TO_STANDBY),
        recognition_failure_limit=int(getattr(config, "ASR_FAILURES_TO_STANDBY", 5)),
    )
    if runtime.state == AssistantState.STANDBY:
        print_standby_hint()

    while runtime.state != AssistantState.SHUTDOWN:
        # ---------- Lấy đầu vào ----------
        try:
            if mode == "text":
                text = input("⌨️  Bạn: ").strip()
            elif runtime.state == AssistantState.STANDBY and wake.available():
                # Engine chuyên dụng: chặn tới khi nghe đúng "Hey Jarvis"
                text = "__WAKE__" if wake.wait_for_wake() else None
            elif runtime.state == AssistantState.STANDBY:
                # Dự phòng: Google STT + đoán chữ (tiếng Anh cho từ khóa)
                text = speech.listen(quiet=True, language=config.WAKE_LANGUAGE, context="command")
            else:
                text = speech.listen(context=runtime.speech_context)
        except (EOFError, KeyboardInterrupt):
            voice.speak("Goodbye, sir.")
            break
        except speech.MicrophoneUnavailable as exc:
            telemetry.event("microphone_unavailable", error=str(exc))
            print("⚠️  Mất kết nối micro — chuyển sang chế độ gõ phím.")
            mode = "text"
            runtime.voice_mode = False
            runtime.transition(AssistantState.ACTIVE, reason="microphone_unavailable")
            continue
        except Exception as exc:
            telemetry.event("input_pipeline_error", error=repr(exc), state=runtime.state.value)
            runtime.recover(vision_active=vision.active(), error=repr(exc))
            try:
                voice.speak("I recovered from an input error, sir. You may try again.")
            except Exception:
                pass
            continue

        # ---------- Chế độ chờ: chỉ phản ứng với wake word ----------
        if mode == "voice" and runtime.state == AssistantState.STANDBY:
            if not text:
                continue
            if text != "__WAKE__":
                # nhánh dự phòng (Google STT): kiểm tra từ khóa trong câu nghe được
                print(f"👂 (nghe thấy: \"{text}\")")
                woke, _ = strip_wake_word(text, brain.normalize(text))
                if not woke:
                    continue
            # Đánh thức: bung HUD, đáp "Yes, sir?", chuyển sang nghe lệnh (tiếng Việt)
            telemetry.event("wake_detected")
            hud.enter_idle_mode()
            hud.show()
            voice.speak("Yes, sir?")
            runtime.wake()
            continue

        # ---------- Chế độ hoạt động ----------
        if not text:
            previous = runtime.state
            diagnostics = speech.last_diagnostics() if mode == "voice" else {}
            rejection_reason = str(diagnostics.get("reason") or "")
            if rejection_reason and rejection_reason != "listen_timeout":
                runtime.on_recognition_failure(reason=rejection_reason)
                if runtime.state != AssistantState.STANDBY:
                    retry_message = "I did not catch that clearly, sir. Please repeat the command."
                    hud.set_transcript("", retry_message)
                    if runtime.recognition_failures == 1:
                        try:
                            voice.speak(retry_message)
                        except Exception:
                            pass
            else:
                runtime.on_empty_input()
            if previous != AssistantState.STANDBY and runtime.state == AssistantState.STANDBY:
                print_standby_hint()
            continue

        runtime.command_received()
        if mode == "voice":
            print(f"🗣️  Bạn: {text}")

        # Đang hoạt động: xử lý lệnh THẲNG (wake đã do engine riêng lo ở chế độ chờ,
        # không kiểm tra từ khóa ở đây nữa -> "goodbye" thoát đúng, không nhầm "travis")
        turn_id = f"turn-{uuid4().hex[:12]}"
        diagnostics = speech.last_diagnostics() if mode == "voice" else {}
        telemetry.event(
            "turn_received",
            turn_id=turn_id,
            text=text,
            state=runtime.state.value,
            speech_context=runtime.speech_context,
            asr_audio_id=diagnostics.get("audio_id"),
            asr_decode_latency_ms=diagnostics.get("decode_latency_ms"),
        )
        norm = brain.normalize(text)
        try:
            result = handle_command(text, norm, mode)
        except Exception as exc:
            telemetry.event("turn_exception", turn_id=turn_id, error=repr(exc), text=text)
            runtime.recover(vision_active=vision.active(), error=repr(exc))
            recovery_reply = "I encountered an internal error but recovered, sir. The session is still available."
            hud.set_transcript(text, recovery_reply)
            try:
                voice.speak(recovery_reply)
            except Exception:
                pass
            continue

        telemetry.event(
            "turn_completed",
            turn_id=turn_id,
            result=result,
            vision_active=vision.active(),
            vision_status=vision.status(),
        )
        if result == "quit":
            runtime.transition(AssistantState.SHUTDOWN, reason="explicit_quit")
            break
        if result == "standby":
            runtime.transition(AssistantState.STANDBY, reason="explicit_sleep")
            print_standby_hint()
        elif result in ("text", "voice"):
            mode = result
            runtime.voice_mode = mode == "voice"
            runtime.transition(AssistantState.ACTIVE, reason="input_mode_changed")
        else:
            previous = runtime.state
            runtime.command_complete(
                vision_active=vision.active(),
                one_shot=bool(getattr(config, "ONE_SHOT_AFTER_WAKE", True)),
            )
            if previous != AssistantState.STANDBY and runtime.state == AssistantState.STANDBY:
                print_standby_hint()

    vision.shutdown()


def _acquire_single_instance():
    """Giữ một cổng cục bộ làm 'khóa' — bản Jarvis thứ hai sẽ không mở được."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", int(getattr(config, "INSTANCE_PORT", 47822))))
        return sock
    except OSError:
        return None


if __name__ == "__main__":
    _instance_lock = _acquire_single_instance()
    if _instance_lock is None:
        print("⚠️  Jarvis đang chạy rồi (có thể ở chế độ nền) — không mở thêm bản thứ hai.")
        sys.exit(0)

    if hud.available():
        # HUD chạy ở luồng chính (yêu cầu của giao diện), Jarvis chạy nền
        hud.start(main)
    else:
        main()
