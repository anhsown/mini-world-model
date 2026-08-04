"""Nhận diện từ khóa đánh thức "Hey Jarvis" — kiến trúc ĐÚNG (offline).

Khác với cách cũ (phiên âm mọi thứ bằng Google rồi đoán chữ — mong manh, hay
nhầm "Jarvis" thành "Travis"), module này dùng openWakeWord: một model nhỏ được
HUẤN LUYỆN RIÊNG để nhận diện chính xác âm thanh "Hey Jarvis".

  - Chạy 100% trên máy (không internet, không phiên âm, không đoán chữ)
  - Chỉ báo khi nghe đúng "Hey Jarvis" -> gần như không bao giờ nhầm
  - Nhẹ, chạy real-time trên CPU

Nếu máy không cài được openWakeWord, jarvis.py tự quay về cách cũ (Google STT).
"""

import config

RATE = 16000        # openWakeWord yêu cầu 16kHz mono
FRAME = 1280        # 80ms mỗi khung

_model = None
_available = None


def available() -> bool:
    global _available
    if _available is None:
        try:
            import openwakeword  # noqa: F401
            import pyaudio  # noqa: F401
            _available = True
        except Exception:
            _available = False
    return _available


def _get_model():
    global _model
    if _model is None:
        from openwakeword.model import Model
        _model = Model(
            wakeword_models=[config.WAKE_MODEL],
            inference_framework="onnx",
        )
    return _model


def preload() -> None:
    """Nạp sẵn model khi khởi động để lần đánh thức đầu không bị trễ."""
    if available():
        try:
            _get_model()
        except Exception:
            pass


def wait_for_wake() -> bool:
    """Chặn tới khi nghe đúng 'Hey Jarvis'. Trả về True khi phát hiện.

    Mở micro riêng, dò liên tục, rồi ĐÓNG micro trước khi trả về để nhường cho
    phần nghe lệnh (Google STT) dùng micro sau đó — tránh tranh chấp thiết bị.
    """
    import numpy as np
    import pyaudio
    from core import speech

    model = _get_model()
    model.reset()  # xóa bộ đệm dự đoán cũ

    pa = pyaudio.PyAudio()
    input_device_index = speech.pyaudio_input_device_index(pa)
    stream = pa.open(rate=RATE, channels=1, format=pyaudio.paInt16,
                    input=True, frames_per_buffer=FRAME,
                    input_device_index=input_device_index)
    try:
        while True:
            data = stream.read(FRAME, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            scores = model.predict(audio)
            if scores.get(config.WAKE_MODEL, 0.0) >= config.WAKE_THRESHOLD:
                return True
    except (OSError, KeyboardInterrupt):
        return False
    finally:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception:
            pass
