from __future__ import annotations

import numpy as np
import speech_recognition as sr

from core import speech


GOOD_DECODER_METRICS = {
    "avg_logprob": -0.25,
    "mean_word_probability": 0.82,
    "max_no_speech_probability": 0.08,
}


def test_microphone_hint_prefers_realtek_array() -> None:
    names = ["V380 FHD Camera", "Microphone Array (Realtek(R) Audio)", "Stereo Mix"]
    assert speech._resolve_name_index(names, "Microphone Array (Realtek") == 1


def test_signal_gate_rejects_silence() -> None:
    metrics = speech._signal_metrics(np.zeros(16000, dtype=np.float32))
    assert speech._signal_rejection(metrics) == "signal_too_quiet"


def test_language_gate_rejects_foreign_or_non_latin_transcript() -> None:
    assert speech._candidate_rejection("最重要的是鏡頭", "zh", 0.99, GOOD_DECODER_METRICS) == "disallowed_script"
    assert speech._candidate_rejection("Kvinnkapp och ämnegräkare", "sv", 0.91, GOOD_DECODER_METRICS).startswith(
        "unsupported_language"
    )


def test_background_english_is_not_treated_as_command() -> None:
    assert (
        speech._candidate_rejection(
            "Thank you guys for coming out.",
            "en",
            0.94,
            GOOD_DECODER_METRICS,
        )
        == "english_not_command"
    )


def test_vietnamese_and_explicit_english_commands_are_accepted() -> None:
    assert speech._candidate_rejection("Hãy truy cập camera", "vi", 0.88, GOOD_DECODER_METRICS) is None
    assert speech._candidate_rejection("Open the camera, Jarvis", "en", 0.91, GOOD_DECODER_METRICS) is None


def test_open_ended_english_question_is_allowed_only_in_vision_context() -> None:
    question = "What am I holding?"
    assert speech._candidate_rejection(question, "en", 0.83, GOOD_DECODER_METRICS) == "english_not_command"
    assert (
        speech._candidate_rejection(
            question,
            "en",
            0.83,
            GOOD_DECODER_METRICS,
            context="vision",
        )
        is None
    )


def test_cloud_command_resolver_repairs_short_camera_command() -> None:
    assert speech._resolve_supported_command("Truy cập Camera") == "Truy cập Camera"
    assert speech._resolve_supported_command("access to camera") == "access camera"
    assert speech._resolve_supported_command("camera") is None


def test_cloud_fallback_recovers_rejected_local_command(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("config.ASR_LOG_ROOT", str(tmp_path / "asr"))
    monkeypatch.setattr("config.RUNTIME_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(
        speech._recognizer,
        "recognize_google",
        lambda _audio, language: "Truy cập Camera" if language == "vi-VN" else "teacup camera",
    )
    audio = sr.AudioData(b"\x00\x00" * 16000, 16000, 2)
    text = speech._transcribe_cloud_fallback(
        audio,
        context="command",
        primary={"reason": "unsupported_language:id", "signal": {"duration_s": 1.0}},
    )
    assert text == "Truy cập Camera"
    assert speech.last_diagnostics()["model"] == "google-speech-fallback"
