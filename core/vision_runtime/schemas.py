from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any


@dataclass
class VisualEvidence:
    bbox_norm: tuple[float, float, float, float] | None = None
    visual_attributes: list[str] = field(default_factory=list)
    visible: bool = True


@dataclass
class ReasonerResult:
    answer: str
    object_name: str | None = None
    evidence: VisualEvidence = field(default_factory=VisualEvidence)
    confidence: float = 0.0
    abstain: bool = False
    rationale_summary: str = ""
    raw_text: str = ""
    model_id: str = ""
    latency_ms: float = 0.0
    frame_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    verdict: str
    failure_type: str | None
    answer_match: bool | None
    grounded: bool | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loads_relaxed_json(candidate: str) -> dict[str, Any]:
    variants = [candidate]
    repaired = re.sub(
        r'([,{]\s*)([^\W\d][\w-]*)(\s*:)',
        r'\1"\2"\3',
        candidate,
        flags=re.UNICODE,
    )
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if repaired != candidate:
        variants.append(repaired)
    last_error: json.JSONDecodeError | None = None
    for variant in variants:
        try:
            value = json.loads(variant)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(value, dict):
            raise ValueError("parsed JSON is not an object")
        return value
    if last_error is not None:
        raise last_error
    raise ValueError("parsed JSON is not an object")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return _loads_relaxed_json(stripped)
    except (ValueError, json.JSONDecodeError):
        pass
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string and char == "{":
            depth += 1
        elif not in_string and char == "}":
            depth -= 1
            if depth == 0:
                return _loads_relaxed_json(stripped[start : index + 1])
    raise ValueError("model response contains incomplete JSON")


def _extract_partial_payload(text: str) -> dict[str, Any] | None:
    """Recover answer/grounding when a small VLM truncates or slightly corrupts JSON."""
    answer_match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']+)', text, flags=re.IGNORECASE)
    if not answer_match:
        return None
    payload: dict[str, Any] = {"answer": answer_match.group(1).strip(), "abstain": False}
    bbox_match = re.search(
        r'["\'](?:bbox|bbox_norm)["\']\s*:\s*\[\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)',
        text,
        flags=re.IGNORECASE,
    )
    if bbox_match:
        payload["bbox"] = [float(value) for value in bbox_match.groups()]
    confidence_match = re.search(r'["\']confidence["\']\s*:\s*([-+\d.eE]+)', text, flags=re.IGNORECASE)
    if confidence_match:
        payload["confidence"] = float(confidence_match.group(1))
    return payload


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "y", "1", "co", "có"}:
            return True
        if normalized in {"false", "no", "n", "0", "khong", "không", ""}:
            return False
    return default


def _normalize_attributes(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value[:8]]
    if isinstance(value, dict):
        attributes: list[str] = []
        for key, item in value.items():
            if str(key).casefold() in {"confidence", "score", "visible"}:
                continue
            if isinstance(item, (str, int, float, bool)):
                attributes.append(str(item))
            else:
                attributes.append(f"{key}: {item}")
        return attributes[:8]
    if value in (None, ""):
        return []
    return [str(value)]


def parse_reasoner_result(text: str, *, model_id: str, latency_ms: float, frame_count: int) -> ReasonerResult:
    try:
        payload = _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        payload = _extract_partial_payload(text)
        if payload is None:
            return ReasonerResult(
                answer=text.strip() or "Không thể phân tích câu trả lời.",
                abstain=True,
                rationale_summary="Structured-output parsing failed.",
                raw_text=text,
                model_id=model_id,
                latency_ms=latency_ms,
                frame_count=frame_count,
            )
    evidence_payload = payload.get("evidence") or {}
    bbox = evidence_payload.get("bbox_norm") or payload.get("bbox_norm") or payload.get("bbox")
    normalized_bbox = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = [float(value) for value in bbox]
        if max(values) > 1.0:
            values = [value / 1000.0 for value in values]
        x1, y1, x2, y2 = [min(1.0, max(0.0, value)) for value in values]
        if x2 > x1 and y2 > y1:
            normalized_bbox = (x1, y1, x2, y2)
    raw_attributes = evidence_payload.get("visual_attributes") or payload.get("attributes") or []
    attributes = _normalize_attributes(raw_attributes)
    abstain_value = payload.get("abstain", False)
    for key, value in payload.items():
        normalized_key = str(key).casefold().translate(str.maketrans({"с": "s", "т": "t"}))
        if normalized_key in {"abstain", "abstained"}:
            abstain_value = value
            break
    abstain = _coerce_bool(abstain_value)
    if abstain:
        normalized_bbox = None
    answer = str(payload.get("answer") or "Không đủ thông tin.")
    object_value = payload.get("object_name")
    if object_value in (None, "", "null") and not abstain:
        object_value = answer
    raw_confidence = payload.get("confidence")
    if raw_confidence is None and isinstance(raw_attributes, dict):
        raw_confidence = raw_attributes.get("confidence")
    try:
        confidence = float(raw_confidence if raw_confidence is not None else 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return ReasonerResult(
        answer=answer,
        object_name=None if object_value in (None, "", "null") else str(object_value),
        evidence=VisualEvidence(
            bbox_norm=normalized_bbox,
            visual_attributes=attributes,
            visible=False if abstain else _coerce_bool(evidence_payload.get("visible"), default=normalized_bbox is not None),
        ),
        confidence=min(1.0, max(0.0, confidence)),
        abstain=abstain,
        rationale_summary=str(payload.get("rationale_summary") or payload.get("evidence_summary") or ""),
        raw_text=text,
        model_id=model_id,
        latency_ms=latency_ms,
        frame_count=frame_count,
    )
