from __future__ import annotations

import json

from core import hud, vision
from core.vision_runtime.schemas import parse_reasoner_result


def test_camera_voice_commands_are_normalized() -> None:
    assert vision.is_open_camera_command("jarvis hay truy cap camera")
    assert vision.is_open_camera_command("truy cap vao camera")
    assert vision.is_open_camera_command("jarvis hay mo giup toi cai camera")
    assert vision.is_open_camera_command("nhin xung quanh")
    assert vision.is_close_camera_command("tat camera")
    assert not vision.is_open_camera_command("may gio roi")


def test_user_camera_phrase_routes_to_vision_not_chat_brain(monkeypatch) -> None:
    import jarvis

    calls: list[str] = []
    monkeypatch.setattr(jarvis.vision, "start", lambda: calls.append("vision") or "Vision online")
    monkeypatch.setattr(jarvis.hud, "set_transcript", lambda *_: None)
    monkeypatch.setattr(jarvis.voice, "speak", lambda text: calls.append(text))
    monkeypatch.setattr(jarvis.brain, "handle", lambda _text: calls.append("brain") or "wrong route")

    result = jarvis.handle_command("truy cập vào camera", "truy cap vao camera", "voice")
    assert result == "active"
    assert calls == ["vision", "Vision online"]


def test_standby_hint_does_not_destroy_camera_session(monkeypatch) -> None:
    import jarvis

    calls: list[str] = []
    monkeypatch.setattr(jarvis.vision, "active", lambda: True)
    monkeypatch.setattr(jarvis.vision, "stop", lambda: calls.append("stop") or "stopped")
    monkeypatch.setattr(jarvis.hud, "hide", lambda: calls.append("hide"))
    monkeypatch.setattr(jarvis.wake, "available", lambda: True)

    jarvis.print_standby_hint()
    assert calls == ["hide"]


def test_fullscreen_hud_has_idle_and_camera_layouts() -> None:
    assert 'id="idleStage"' in hud.HTML
    assert 'id="cameraStage"' in hud.HTML
    assert 'id="cameraViewport"' in hud.HTML
    assert "setVisionResult" in hud.HTML


def test_vision_log_is_append_only_jsonl(tmp_path) -> None:
    controller = vision.VisionController()
    controller._session_id = "unit-session"
    controller._session_dir = tmp_path / "unit-session"
    controller._session_dir.mkdir(parents=True)
    controller._write_event("camera_started", {"camera_index": 0})
    controller._write_event("vision_error", {"error": "fixture"})
    rows = [json.loads(line) for line in (controller._session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == ["camera_started", "vision_error"]
    assert all(row["schema_version"] == "jarvis-vision-v1" for row in rows)


def test_runtime_recovers_small_vlm_json_drift() -> None:
    raw = '{"answer":"bag","bbox":[100,200,500,800],attributes:{"visible fact":"blue bag"}}'
    result = parse_reasoner_result(raw, model_id="unit", latency_ms=1, frame_count=1)
    assert result.answer == "bag"
    assert result.evidence.bbox_norm == (0.1, 0.2, 0.5, 0.8)
    assert not result.abstain


def test_runtime_honours_abstained_alias_and_discards_false_bbox() -> None:
    raw = '{"answer":"nothing","bbox":[412,551,650,750],"confidence":0,"abstained":true}'
    result = parse_reasoner_result(raw, model_id="unit", latency_ms=1, frame_count=1)
    assert result.abstain
    assert result.evidence.bbox_norm is None
    assert not result.evidence.visible
