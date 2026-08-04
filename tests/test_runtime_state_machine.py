from __future__ import annotations

from core.runtime import AssistantRuntime, AssistantState


def _isolate_telemetry(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("config.RUNTIME_EVENT_LOG", str(tmp_path / "events.jsonl"))


def test_silence_never_tears_down_a_vision_session(monkeypatch, tmp_path) -> None:
    _isolate_telemetry(monkeypatch, tmp_path)
    runtime = AssistantRuntime(voice_mode=True, wake_enabled=True, silence_limit=2)
    runtime.wake()
    runtime.sync_vision(True)

    for _ in range(10):
        assert runtime.on_empty_input() == AssistantState.VISION
    assert runtime.silent_rounds == 0
    assert runtime.speech_context == "vision"


def test_non_vision_session_still_returns_to_standby_after_silence(monkeypatch, tmp_path) -> None:
    _isolate_telemetry(monkeypatch, tmp_path)
    runtime = AssistantRuntime(voice_mode=True, wake_enabled=True, silence_limit=2)
    runtime.wake()
    assert runtime.on_empty_input() == AssistantState.ACTIVE
    assert runtime.on_empty_input() == AssistantState.STANDBY


def test_runtime_recovers_to_live_vision_after_turn_exception(monkeypatch, tmp_path) -> None:
    _isolate_telemetry(monkeypatch, tmp_path)
    runtime = AssistantRuntime(voice_mode=True, wake_enabled=True)
    runtime.wake()
    assert runtime.recover(vision_active=True, error="fixture") == AssistantState.VISION


def test_recognition_rejections_do_not_immediately_hide_hud(monkeypatch, tmp_path) -> None:
    _isolate_telemetry(monkeypatch, tmp_path)
    runtime = AssistantRuntime(
        voice_mode=True,
        wake_enabled=True,
        recognition_failure_limit=5,
    )
    runtime.wake()
    for _ in range(4):
        assert runtime.on_recognition_failure(reason="fixture") == AssistantState.ACTIVE
    assert runtime.on_recognition_failure(reason="fixture") == AssistantState.STANDBY


def test_recognition_rejections_never_close_live_vision(monkeypatch, tmp_path) -> None:
    _isolate_telemetry(monkeypatch, tmp_path)
    runtime = AssistantRuntime(voice_mode=True, wake_enabled=True, recognition_failure_limit=2)
    runtime.wake()
    runtime.sync_vision(True)
    for _ in range(10):
        assert runtime.on_recognition_failure(reason="fixture") == AssistantState.VISION
