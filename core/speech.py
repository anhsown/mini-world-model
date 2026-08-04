"""Robust speech-to-text pipeline for JARVIS.

Architecture:
  selected near-field microphone -> acoustic signal gate -> Silero VAD ->
  faster-whisper -> language/confidence/script policy -> accepted command.

Rejected attempts are logged locally so background-speech and hallucination
failures can be audited instead of silently reaching the conversational brain.
"""

from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import threading
from time import perf_counter
import unicodedata
from uuid import uuid4

import numpy as np
import speech_recognition as sr

import config


_recognizer = sr.Recognizer()
_recognizer.dynamic_energy_threshold = True
_recognizer.pause_threshold = float(getattr(config, "ASR_PAUSE_THRESHOLD", 0.65))
_recognizer.phrase_threshold = float(getattr(config, "ASR_PHRASE_THRESHOLD", 0.25))
_recognizer.non_speaking_duration = float(getattr(config, "ASR_NON_SPEAKING_DURATION", 0.35))
_recognizer.operation_timeout = float(getattr(config, "ASR_CLOUD_TIMEOUT", 5))

_calibrated = False
_whisper = None
_whisper_ready = None
_microphone_index = None
_microphone_resolved = False
_log_lock = threading.Lock()
_last_diagnostics: dict = {}


_COMMAND_PHRASES = (
    "truy cap camera",
    "mo camera",
    "tat camera",
    "nhin xung quanh",
    "open camera",
    "access camera",
    "close camera",
    "enable vision",
    "disable vision",
    "may gio roi",
    "what time is it",
    "thoi tiet",
    "weather",
    "mo chrome",
    "open chrome",
    "mo youtube",
    "search google",
    "tim kiem",
    "ghi chu",
    "note this",
    "nhac nho",
    "set reminder",
    "tang am luong",
    "giam am luong",
    "goodbye",
    "good night",
    "ngu di",
    "standby",
)


class MicrophoneUnavailable(Exception):
    """No configured microphone can be opened."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log_root() -> Path:
    configured = Path(getattr(config, "ASR_LOG_ROOT", "data/asr_logs")).expanduser()
    return configured.resolve() if configured.is_absolute() else (_project_root() / configured).resolve()


def _resolve_name_index(names: list[str], hint: str) -> int | None:
    if not hint.strip():
        return None
    needle = hint.casefold()
    for index, name in enumerate(names):
        if needle in str(name).casefold():
            return index
    return None


def microphone_device_index() -> int | None:
    """Return the SpeechRecognition/PyAudio input chosen for commands."""
    global _microphone_index, _microphone_resolved
    if _microphone_resolved:
        return _microphone_index
    explicit = getattr(config, "MICROPHONE_DEVICE_INDEX", None)
    if explicit is not None:
        _microphone_index = int(explicit)
    else:
        names = list(sr.Microphone.list_microphone_names())
        _microphone_index = _resolve_name_index(names, getattr(config, "MICROPHONE_NAME_HINT", ""))
    _microphone_resolved = True
    return _microphone_index


def pyaudio_input_device_index(audio_interface) -> int | None:
    """Resolve the same physical microphone in a raw PyAudio device list."""
    explicit = getattr(config, "MICROPHONE_DEVICE_INDEX", None)
    if explicit is not None:
        return int(explicit)
    hint = getattr(config, "MICROPHONE_NAME_HINT", "").casefold()
    if not hint:
        return None
    for index in range(audio_interface.get_device_count()):
        try:
            info = audio_interface.get_device_info_by_index(index)
        except Exception:
            continue
        if int(info.get("maxInputChannels", 0)) > 0 and hint in str(info.get("name", "")).casefold():
            return index
    return None


def selected_microphone_name() -> str:
    names = list(sr.Microphone.list_microphone_names())
    index = microphone_device_index()
    if index is None:
        return "system default input"
    if 0 <= index < len(names):
        return names[index]
    return f"input index {index}"


def mic_available() -> bool:
    try:
        with sr.Microphone(device_index=microphone_device_index()):
            pass
        return True
    except Exception:
        return False


def _whisper_available() -> bool:
    global _whisper_ready
    if _whisper_ready is None:
        if not getattr(config, "WHISPER_ENABLED", True):
            _whisper_ready = False
        else:
            try:
                import faster_whisper  # noqa: F401
                _whisper_ready = True
            except Exception:
                _whisper_ready = False
    return bool(_whisper_ready)


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )
    return _whisper


def preload() -> None:
    """Preload ASR and report the exact microphone/model selected."""
    if not _whisper_available():
        return
    try:
        _get_whisper()
        print(f"🎧 ASR ready — {config.WHISPER_MODEL} on {config.WHISPER_DEVICE}; mic: {selected_microphone_name()}")
    except Exception as exc:
        global _whisper_ready
        _whisper_ready = False
        _append_event("asr_load_failed", {"error": str(exc), "model": config.WHISPER_MODEL})


def _signal_metrics(samples: np.ndarray, sample_rate: int = 16000) -> dict:
    if samples.size == 0:
        return {"duration_s": 0.0, "rms_dbfs": -120.0, "peak": 0.0, "active_ratio": 0.0}
    values = samples.astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(values))) + 1e-12)
    peak = float(np.max(np.abs(values)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
    absolute = np.abs(values)
    noise_floor = float(np.percentile(absolute, 20))
    active_threshold = max(0.012, noise_floor * 3.5)
    active_ratio = float(np.mean(absolute >= active_threshold))
    return {
        "duration_s": round(float(values.size / sample_rate), 4),
        "rms_dbfs": round(rms_dbfs, 3),
        "peak": round(peak, 5),
        "active_ratio": round(active_ratio, 5),
    }


def _signal_rejection(metrics: dict) -> str | None:
    if metrics["duration_s"] < 0.25:
        return "audio_too_short"
    if metrics["rms_dbfs"] < float(getattr(config, "ASR_MIN_RMS_DBFS", -48.0)):
        return "signal_too_quiet"
    if metrics["active_ratio"] < float(getattr(config, "ASR_MIN_ACTIVE_RATIO", 0.015)):
        return "insufficient_speech_activity"
    return None


def _contains_disallowed_script(text: str) -> bool:
    for char in text:
        code = ord(char)
        if (
            0x0400 <= code <= 0x052F  # Cyrillic
            or 0x0600 <= code <= 0x06FF  # Arabic
            or 0x0E00 <= code <= 0x0E7F  # Thai
            or 0x3040 <= code <= 0x30FF  # Japanese kana
            or 0x3400 <= code <= 0x9FFF  # CJK
            or 0xAC00 <= code <= 0xD7AF  # Hangul
        ):
            return True
    return False


def _normalize_for_policy(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _english_is_command(text: str) -> bool:
    normalized = _normalize_for_policy(text)
    terms = (
        "jarvis",
        "camera",
        "open",
        "close",
        "enable vision",
        "disable vision",
        "time",
        "date",
        "weather",
        "volume",
        "search",
        "reminder",
        "note",
        "goodbye",
        "good night",
        "standby",
        "sleep",
        "hello",
    )
    return any(term in normalized for term in terms)


def _resolve_supported_command(text: str) -> str | None:
    """Resolve noisy cloud text against the complete command catalogue."""
    normalized = _normalize_for_policy(text)
    if not normalized:
        return None
    # Exact containment preserves arguments such as a search query or note.
    if any(phrase in normalized for phrase in _COMMAND_PHRASES):
        return text.strip()
    if len(normalized.split()) < 2:
        return None
    best_score, best_phrase = max(
        (SequenceMatcher(None, normalized, phrase).ratio(), phrase)
        for phrase in _COMMAND_PHRASES
    )
    # High-threshold fuzzy repair is used only after an explicit wake and only
    # after local ASR has already rejected a real speech segment.
    if best_score >= 0.78:
        return best_phrase
    return None


def _segment_metrics(segments: list) -> dict:
    durations = [max(0.01, float(segment.end - segment.start)) for segment in segments]
    total_duration = sum(durations) or 1.0
    avg_logprob = sum(float(segment.avg_logprob) * duration for segment, duration in zip(segments, durations)) / total_duration
    no_speech = max((float(segment.no_speech_prob) for segment in segments), default=1.0)
    compression_ratio = max((float(segment.compression_ratio) for segment in segments), default=0.0)
    word_probabilities = [
        float(word.probability)
        for segment in segments
        for word in (segment.words or [])
        if getattr(word, "probability", None) is not None
    ]
    return {
        "speech_duration_s": round(total_duration, 4),
        "avg_logprob": round(avg_logprob, 4),
        "max_no_speech_probability": round(no_speech, 4),
        "max_compression_ratio": round(compression_ratio, 4),
        "mean_word_probability": round(float(np.mean(word_probabilities)), 4) if word_probabilities else None,
        "word_count": len(word_probabilities),
    }


def _candidate_rejection(
    text: str,
    language: str,
    language_probability: float,
    metrics: dict,
    *,
    context: str = "command",
) -> str | None:
    if not text.strip() or len(text.strip()) < 2:
        return "empty_transcript"
    if _contains_disallowed_script(text):
        return "disallowed_script"
    allowed = tuple(getattr(config, "ASR_ALLOWED_LANGUAGES", ("vi", "en")))
    if language not in allowed:
        return f"unsupported_language:{language or 'unknown'}"
    is_vision = context == "vision"
    if is_vision:
        probability_key = (
            "ASR_VISION_MIN_LANGUAGE_PROBABILITY_VI"
            if language == "vi"
            else "ASR_VISION_MIN_LANGUAGE_PROBABILITY_EN"
        )
        default_probability = 0.32 if language == "vi" else 0.55
    else:
        probability_key = "ASR_MIN_LANGUAGE_PROBABILITY_VI" if language == "vi" else "ASR_MIN_LANGUAGE_PROBABILITY_EN"
        default_probability = 0.4 if language == "vi" else 0.72
    minimum_language_probability = float(getattr(config, probability_key, default_probability))
    if language_probability < minimum_language_probability:
        return "low_language_probability"
    if metrics["avg_logprob"] < float(getattr(config, "ASR_MIN_AVG_LOGPROB", -0.85)):
        return "low_decoder_logprob"
    word_probability = metrics.get("mean_word_probability")
    if word_probability is not None and word_probability < float(getattr(config, "ASR_MIN_WORD_PROBABILITY", 0.42)):
        return "low_word_probability"
    if metrics["max_no_speech_probability"] > float(getattr(config, "ASR_MAX_NO_SPEECH_PROBABILITY", 0.68)):
        return "high_no_speech_probability"
    words = _normalize_for_policy(text).split()
    if len(words) >= 4 and max(words.count(word) for word in set(words)) / len(words) > 0.6:
        return "repetitive_transcript"
    if (
        language == "en"
        and not is_vision
        and not getattr(config, "ASR_ALLOW_ENGLISH_FREEFORM", False)
        and not _english_is_command(text)
    ):
        return "english_not_command"
    return None


def _transcribe_whisper(audio: sr.AudioData, *, context: str = "command") -> str | None:
    global _last_diagnostics
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    signal = _signal_metrics(samples)
    rejection = _signal_rejection(signal)
    audio_id = f"asr-{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid4().hex[:8]}"
    if rejection:
        diagnostics = {
            "audio_id": audio_id,
            "accepted": False,
            "reason": rejection,
            "context": context,
            "signal": signal,
        }
        _last_diagnostics = diagnostics
        _record_attempt(audio, diagnostics)
        print(f"🛡️  ASR rejected audio: {rejection}")
        return None

    decode_started = perf_counter()
    try:
        initial_prompt = (
            getattr(config, "ASR_VISION_INITIAL_PROMPT", None)
            if context == "vision"
            else getattr(config, "ASR_INITIAL_PROMPT", None)
        )
        hotwords = (
            getattr(config, "ASR_VISION_HOTWORDS", None)
            if context == "vision"
            else getattr(config, "ASR_HOTWORDS", None)
        )
        segment_iter, info = _get_whisper().transcribe(
            samples,
            language=None,
            beam_size=int(getattr(config, "ASR_BEAM_SIZE", 2)),
            best_of=1,
            patience=1.0,
            temperature=0.0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-0.85,
            no_speech_threshold=0.62,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            word_timestamps=True,
            multilingual=False,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.55,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 420,
                "speech_pad_ms": 120,
                "max_speech_duration_s": float(getattr(config, "PHRASE_LIMIT", 12)),
            },
            max_new_tokens=96,
            hallucination_silence_threshold=1.0,
            hotwords=hotwords,
            language_detection_threshold=0.55,
            language_detection_segments=int(getattr(config, "ASR_LANGUAGE_DETECTION_SEGMENTS", 1)),
        )
        segments = list(segment_iter)
    except Exception as exc:
        diagnostics = {
            "audio_id": audio_id,
            "accepted": False,
            "reason": "decoder_error",
            "context": context,
            "error": str(exc),
            "signal": signal,
            "decode_latency_ms": round((perf_counter() - decode_started) * 1000.0, 2),
        }
        _last_diagnostics = diagnostics
        _record_attempt(audio, diagnostics)
        print(f"   (ASR decoder error: {exc})")
        return None

    text = "".join(segment.text for segment in segments).strip()
    segment_stats = _segment_metrics(segments)
    language = str(getattr(info, "language", "") or "")
    language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    rejection = _candidate_rejection(
        text,
        language,
        language_probability,
        segment_stats,
        context=context,
    )
    diagnostics = {
        "audio_id": audio_id,
        "accepted": rejection is None,
        "reason": rejection,
        "context": context,
        "text": text,
        "language": language,
        "language_probability": round(language_probability, 4),
        "signal": signal,
        "decoder": segment_stats,
        "model": config.WHISPER_MODEL,
        "microphone": selected_microphone_name(),
        "decode_latency_ms": round((perf_counter() - decode_started) * 1000.0, 2),
    }
    _last_diagnostics = diagnostics
    _record_attempt(audio, diagnostics)
    if rejection:
        print(f"🛡️  ASR rejected transcript ({rejection}): {text!r}")
        return None
    print(
        f"✅ ASR accepted [{language} {language_probability:.2f}; "
        f"logp {segment_stats['avg_logprob']:.2f}; {diagnostics['decode_latency_ms']:.0f} ms]: {text}"
    )
    return text


def _transcribe_cloud_fallback(audio: sr.AudioData, *, context: str, primary: dict) -> str | None:
    """Recover short/noisy bilingual commands after local Whisper rejects them."""
    global _last_diagnostics
    from core import telemetry

    started = perf_counter()
    candidates: list[dict] = []
    for language in tuple(getattr(config, "ASR_CLOUD_FALLBACK_LANGUAGES", ("vi-VN", "en-US"))):
        try:
            raw_text = _recognizer.recognize_google(audio, language=language).strip()
        except sr.UnknownValueError:
            candidates.append({"language": language, "reason": "unknown_value"})
            continue
        except sr.RequestError as exc:
            candidates.append({"language": language, "reason": "request_error", "error": str(exc)})
            continue
        if not raw_text or _contains_disallowed_script(raw_text):
            candidates.append({"language": language, "text": raw_text, "reason": "invalid_text"})
            continue
        resolved = raw_text if context == "vision" else _resolve_supported_command(raw_text)
        candidates.append({"language": language, "text": raw_text, "resolved": resolved})
        if not resolved:
            continue
        audio_id = f"asr-{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid4().hex[:8]}"
        diagnostics = {
            "audio_id": audio_id,
            "accepted": True,
            "reason": None,
            "context": context,
            "text": resolved,
            "raw_cloud_text": raw_text,
            "language": language.split("-", 1)[0],
            "language_probability": None,
            "signal": primary.get("signal", {}),
            "decoder": {},
            "model": "google-speech-fallback",
            "microphone": selected_microphone_name(),
            "decode_latency_ms": round((perf_counter() - started) * 1000.0, 2),
            "primary_rejection": primary.get("reason"),
        }
        _last_diagnostics = diagnostics
        _record_attempt(audio, diagnostics)
        telemetry.event(
            "asr_cloud_fallback_accepted",
            context=context,
            raw_text=raw_text,
            resolved_text=resolved,
            language=language,
            latency_ms=diagnostics["decode_latency_ms"],
            primary_rejection=primary.get("reason"),
        )
        print(f"☁️  ASR fallback accepted [{language}]: {resolved}")
        return resolved

    diagnostics = {
        **primary,
        "accepted": False,
        "reason": "cloud_fallback_failed",
        "context": context,
        "cloud_candidates": candidates,
        "cloud_latency_ms": round((perf_counter() - started) * 1000.0, 2),
        "primary_rejection": primary.get("reason"),
    }
    _last_diagnostics = diagnostics
    telemetry.event(
        "asr_cloud_fallback_failed",
        context=context,
        candidates=candidates,
        primary_rejection=primary.get("reason"),
    )
    return None


def _record_attempt(audio: sr.AudioData, diagnostics: dict) -> None:
    accepted = bool(diagnostics.get("accepted"))
    save_audio = bool(
        getattr(config, "ASR_SAVE_ACCEPTED_AUDIO", False)
        if accepted
        else getattr(config, "ASR_SAVE_REJECTED_AUDIO", True)
    )
    day_dir = _log_root() / datetime.now(timezone.utc).strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    audio_file = None
    if save_audio:
        audio_file = f"{diagnostics['audio_id']}.wav"
        (day_dir / audio_file).write_bytes(audio.get_wav_data(convert_rate=16000, convert_width=2))
    payload = {
        "schema_version": "jarvis-asr-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **diagnostics,
        "audio_file": audio_file,
    }
    _append_jsonl(day_dir / "asr_events.jsonl", payload)


def _append_event(event_type: str, payload: dict) -> None:
    day_dir = _log_root() / datetime.now(timezone.utc).strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": "jarvis-asr-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    _append_jsonl(day_dir / "asr_events.jsonl", event)


def _append_jsonl(path: Path, payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with _log_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def last_diagnostics() -> dict:
    return dict(_last_diagnostics)


def listen(quiet: bool = False, language: str | None = None, *, context: str = "command") -> str | None:
    global _calibrated, _last_diagnostics
    try:
        with sr.Microphone(device_index=microphone_device_index()) as source:
            if not _calibrated:
                print(f"🎚️  Calibrating near-field microphone: {selected_microphone_name()}")
                _recognizer.adjust_for_ambient_noise(
                    source,
                    duration=float(getattr(config, "ASR_AMBIENT_CALIBRATION_SECONDS", 1.0)),
                )
                _calibrated = True
            if not quiet:
                print("🎙️  Đang nghe... (nói gần microphone)")
            audio = _recognizer.listen(
                source,
                timeout=config.LISTEN_TIMEOUT,
                phrase_time_limit=config.PHRASE_LIMIT,
            )
    except sr.WaitTimeoutError:
        _last_diagnostics = {
            "accepted": False,
            "reason": "listen_timeout",
            "context": context,
        }
        return None
    except OSError as exc:
        raise MicrophoneUnavailable(str(exc)) from exc

    if _whisper_available():
        text = _transcribe_whisper(audio, context=context)
        if text:
            return text
        primary = last_diagnostics()
        acoustic_rejections = {"audio_too_short", "signal_too_quiet", "insufficient_speech_activity"}
        if (
            getattr(config, "ASR_CLOUD_FALLBACK_ENABLED", True)
            and primary.get("reason") not in acoustic_rejections
        ):
            return _transcribe_cloud_fallback(audio, context=context, primary=primary)
        return None

    # Google is used only when Whisper cannot be loaded, never as a fallback for
    # a low-confidence Whisper rejection.
    try:
        return _recognizer.recognize_google(audio, language=language or config.LANGUAGE)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as exc:
        print(f"   (Google STT unavailable: {exc})")
        return None
