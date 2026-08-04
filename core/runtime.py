"""Explicit session state machine for wake, conversation and persistent vision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core import telemetry


class AssistantState(str, Enum):
    STANDBY = "standby"
    ACTIVE = "active"
    VISION = "vision"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"


@dataclass
class AssistantRuntime:
    voice_mode: bool
    wake_enabled: bool
    silence_limit: int = 2
    recognition_failure_limit: int = 5

    def __post_init__(self) -> None:
        self.state = AssistantState.STANDBY if self.voice_mode and self.wake_enabled else AssistantState.ACTIVE
        self.silent_rounds = 0
        self.recognition_failures = 0
        telemetry.event("runtime_started", state=self.state.value)

    @property
    def speech_context(self) -> str:
        return "vision" if self.state == AssistantState.VISION else "command"

    def transition(self, state: AssistantState, *, reason: str) -> AssistantState:
        previous = self.state
        self.state = state
        if state != AssistantState.ACTIVE:
            self.silent_rounds = 0
        if state in {AssistantState.STANDBY, AssistantState.SHUTDOWN}:
            self.recognition_failures = 0
        if previous != state:
            telemetry.event("state_transition", previous=previous.value, state=state.value, reason=reason)
        return self.state

    def wake(self) -> AssistantState:
        self.silent_rounds = 0
        return self.transition(AssistantState.ACTIVE, reason="wake_detected")

    def sync_vision(self, vision_active: bool) -> AssistantState:
        if vision_active:
            return self.transition(AssistantState.VISION, reason="vision_active")
        if self.state == AssistantState.VISION:
            return self.transition(AssistantState.ACTIVE, reason="vision_closed")
        return self.state

    def on_empty_input(self) -> AssistantState:
        # A vision session owns its own lifetime. ASR silence/rejection is not a
        # camera-close command and must never hide or tear down the session.
        if self.state == AssistantState.VISION:
            telemetry.event("empty_input", state=self.state.value, action="keep_vision_alive")
            return self.state
        if self.state == AssistantState.ACTIVE and self.voice_mode and self.wake_enabled:
            self.silent_rounds += 1
            telemetry.event("empty_input", state=self.state.value, silent_rounds=self.silent_rounds)
            if self.silent_rounds >= max(1, self.silence_limit):
                return self.transition(AssistantState.STANDBY, reason="silence_limit")
        return self.state

    def command_received(self) -> None:
        self.silent_rounds = 0
        self.recognition_failures = 0

    def on_recognition_failure(self, *, reason: str) -> AssistantState:
        self.silent_rounds = 0
        self.recognition_failures += 1
        telemetry.event(
            "recognition_failure",
            state=self.state.value,
            reason=reason,
            consecutive_failures=self.recognition_failures,
        )
        if self.state == AssistantState.VISION:
            return self.state
        if self.recognition_failures >= max(1, self.recognition_failure_limit):
            return self.transition(AssistantState.STANDBY, reason="recognition_failure_limit")
        return self.state

    def command_complete(self, *, vision_active: bool, one_shot: bool) -> AssistantState:
        self.sync_vision(vision_active)
        if self.state == AssistantState.VISION:
            return self.state
        if self.voice_mode and self.wake_enabled and one_shot:
            return self.transition(AssistantState.STANDBY, reason="one_shot_complete")
        return self.transition(AssistantState.ACTIVE, reason="command_complete")

    def recover(self, *, vision_active: bool, error: str) -> AssistantState:
        self.transition(AssistantState.RECOVERING, reason="turn_exception")
        telemetry.event("turn_recovered", error=error, vision_active=vision_active)
        return self.transition(
            AssistantState.VISION if vision_active else AssistantState.ACTIVE,
            reason="recovery_complete",
        )
