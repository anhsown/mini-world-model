"""Metric-gated adaptive training budget for JWM stages.

The controller deliberately does not use training loss as its stopping
criterion.  Decisions are made from fixed, held-out/OOD metrics and mandatory
causal gates.  Training loss is used only to detect the characteristic
overfitting pattern where optimisation continues while OOD quality regresses.

The class is dependency-free so the exact same state can be used in local,
Kaggle and DDP training scripts.  Only rank zero should call ``observe`` and
``decide``; the resulting action can then be broadcast to the other ranks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping


class BudgetAction(str, Enum):
    CONTINUE = "continue"
    REDUCE_LR = "reduce_lr"
    ADVANCE_STAGE = "advance_stage"
    STOP_CONVERGED = "stop_converged"
    STOP_BLOCKED = "stop_blocked"
    STOP_OVERFIT = "stop_overfit"
    STOP_UNSTABLE = "stop_unstable"


@dataclass(frozen=True)
class MetricSpec:
    """One held-out metric normalized from a baseline to a target.

    ``direction`` is ``"min"`` when lower is better and ``"max"`` when
    higher is better.  A normalized progress of zero equals ``baseline`` and
    one equals ``target``.  Required metrics are hard gates; non-required
    metrics still contribute to the progress score and can expose regressions.
    """

    name: str
    direction: str
    baseline: float
    target: float
    weight: float = 1.0
    required: bool = True

    def __post_init__(self) -> None:
        if self.direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'")
        if not all(math.isfinite(v) for v in (self.baseline, self.target,
                                               self.weight)):
            raise ValueError("metric specification values must be finite")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if self.direction == "min" and not self.target < self.baseline:
            raise ValueError("a min target must be lower than its baseline")
        if self.direction == "max" and not self.target > self.baseline:
            raise ValueError("a max target must be higher than its baseline")

    def progress(self, value: float) -> float:
        if self.direction == "min":
            return (self.baseline - value) / (self.baseline - self.target)
        return (value - self.baseline) / (self.target - self.baseline)

    def passes(self, value: float) -> bool:
        return value <= self.target if self.direction == "min" else value >= self.target


@dataclass(frozen=True)
class BudgetConfig:
    min_steps: int
    max_steps: int
    eval_every: int
    slope_window: int = 5
    plateau_patience: int = 4
    overfit_patience: int = 3
    max_lr_decays: int = 2
    # Minimum normalized score improvement per evaluation, not per raw step.
    min_progress_per_eval: float = 0.01
    # A fall of this size below the best OOD score is treated as material.
    overfit_tolerance: float = 0.04
    # Gradient health is a veto, never a quality signal.
    min_grad_norm: float = 1e-8
    max_grad_norm: float = 1e4
    final_stage: bool = False

    def __post_init__(self) -> None:
        if self.min_steps < 0 or self.max_steps <= self.min_steps:
            raise ValueError("require 0 <= min_steps < max_steps")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.slope_window < 3:
            raise ValueError("slope_window must be at least three")
        if self.plateau_patience < 1 or self.overfit_patience < 1:
            raise ValueError("patience values must be positive")


@dataclass(frozen=True)
class BudgetObservation:
    step: int
    metrics: dict[str, float]
    train_loss: float
    grad_norm: float
    learning_rate: float
    score: float
    gates_passed: bool


@dataclass(frozen=True)
class BudgetDecision:
    action: BudgetAction
    reason: str
    step: int
    score: float
    best_step: int
    best_score: float
    slope_per_eval: float | None
    projected_total_steps: int | None


def _linear_slope(values: Iterable[float]) -> float | None:
    """OLS slope over equally spaced evaluations, used only after smoothing."""
    ys = list(values)
    n = len(ys)
    if n < 3:
        return None
    mean_x = (n - 1) / 2.0
    mean_y = sum(ys) / n
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return None
    return sum((i - mean_x) * (y - mean_y) for i, y in enumerate(ys)) / denominator


class AdaptiveTrainingBudget:
    """Sequential stage controller with hard caps and causal metric gates."""

    def __init__(self, specs: Iterable[MetricSpec], config: BudgetConfig):
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("at least one held-out metric is required")
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique")
        self.config = config
        self.history: list[BudgetObservation] = []
        self.lr_decays = 0
        self.last_lr_decay_step = -1
        self.best_step = 0
        self.best_score = -math.inf

    def _validate_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        missing = [spec.name for spec in self.specs if spec.name not in metrics]
        if missing:
            raise KeyError(f"missing held-out metrics: {missing}")
        selected = {spec.name: float(metrics[spec.name]) for spec in self.specs}
        return selected

    def _score(self, metrics: Mapping[str, float]) -> float:
        weighted = sum(spec.weight * spec.progress(metrics[spec.name])
                       for spec in self.specs)
        return weighted / sum(spec.weight for spec in self.specs)

    def _gates_pass(self, metrics: Mapping[str, float]) -> bool:
        return all((not spec.required) or spec.passes(metrics[spec.name])
                   for spec in self.specs)

    def observe(self, step: int, metrics: Mapping[str, float], train_loss: float,
                grad_norm: float, learning_rate: float) -> BudgetObservation:
        if self.history and step <= self.history[-1].step:
            raise ValueError("observation steps must increase strictly")
        selected = self._validate_metrics(metrics)
        score = self._score(selected) if all(map(math.isfinite, selected.values())) else -math.inf
        obs = BudgetObservation(
            step=int(step), metrics=selected, train_loss=float(train_loss),
            grad_norm=float(grad_norm), learning_rate=float(learning_rate),
            score=score, gates_passed=self._gates_pass(selected)
            if all(map(math.isfinite, selected.values())) else False,
        )
        self.history.append(obs)
        if score > self.best_score:
            self.best_score, self.best_step = score, int(step)
        return obs

    def acknowledge_lr_decay(self) -> None:
        if not self.history:
            raise RuntimeError("cannot acknowledge an LR decay before observation")
        current_step = self.history[-1].step
        if current_step == self.last_lr_decay_step:
            return
        self.lr_decays += 1
        self.last_lr_decay_step = current_step

    def _smoothed_scores(self) -> list[float]:
        # Three-point trailing median suppresses one noisy validation run.
        scores = [obs.score for obs in self.history]
        smooth: list[float] = []
        for index in range(len(scores)):
            chunk = sorted(scores[max(0, index - 2):index + 1])
            smooth.append(chunk[len(chunk) // 2])
        return smooth

    def slope_per_eval(self) -> float | None:
        scores = self._smoothed_scores()
        return _linear_slope(scores[-self.config.slope_window:])

    def projected_total_steps(self) -> int | None:
        """Advisory projection to normalized score 1; never overrides gates."""
        if not self.history:
            return None
        slope = self.slope_per_eval()
        current = self._smoothed_scores()[-1]
        if slope is None or slope <= 0 or current >= 1.0:
            return self.history[-1].step if current >= 1.0 else None
        remaining_evals = math.ceil((1.0 - current) / slope)
        estimate = self.history[-1].step + remaining_evals * self.config.eval_every
        return min(estimate, self.config.max_steps)

    def _unstable(self, obs: BudgetObservation) -> bool:
        finite = (math.isfinite(obs.score) and math.isfinite(obs.train_loss) and
                  math.isfinite(obs.grad_norm) and math.isfinite(obs.learning_rate))
        return (not finite or obs.grad_norm < self.config.min_grad_norm or
                obs.grad_norm > self.config.max_grad_norm)

    def _overfitting(self) -> bool:
        p = self.config.overfit_patience
        if len(self.history) < p + 1:
            return False
        recent = self.history[-p:]
        ood_worse = all(obs.score < self.best_score - self.config.overfit_tolerance
                        for obs in recent)
        train_still_improving = recent[-1].train_loss < self.history[-p - 1].train_loss
        return ood_worse and train_still_improving

    def _plateaued(self) -> bool:
        p = self.config.plateau_patience
        if len(self.history) < max(p, self.config.slope_window):
            return False
        slope = self.slope_per_eval()
        no_new_best = self.best_step < self.history[-p].step
        return (slope is not None and
                slope < self.config.min_progress_per_eval and no_new_best)

    def decide(self) -> BudgetDecision:
        if not self.history:
            raise RuntimeError("observe held-out metrics before deciding")
        obs = self.history[-1]
        slope = self.slope_per_eval()
        projected = self.projected_total_steps()

        def result(action: BudgetAction, reason: str) -> BudgetDecision:
            return BudgetDecision(action, reason, obs.step, obs.score,
                                  self.best_step, self.best_score, slope, projected)

        if self._unstable(obs):
            return result(BudgetAction.STOP_UNSTABLE,
                          "non-finite or unhealthy gradient/metric state")

        if obs.step >= self.config.max_steps:
            if obs.gates_passed:
                action = (BudgetAction.STOP_CONVERGED if self.config.final_stage
                          else BudgetAction.ADVANCE_STAGE)
                return result(action, "hard cap reached with all gates passed")
            return result(BudgetAction.STOP_BLOCKED,
                          "hard cap reached while mandatory OOD/causal gates failed")

        if obs.step < self.config.min_steps:
            return result(BudgetAction.CONTINUE,
                          "minimum evidence budget has not been reached")

        if self._overfitting():
            return result(BudgetAction.STOP_OVERFIT,
                          "training loss improved while held-out OOD score regressed")

        if self._plateaued():
            if obs.gates_passed:
                action = (BudgetAction.STOP_CONVERGED if self.config.final_stage
                          else BudgetAction.ADVANCE_STAGE)
                return result(action,
                              "mandatory gates passed and marginal OOD gain plateaued")
            if self.lr_decays < self.config.max_lr_decays:
                return result(BudgetAction.REDUCE_LR,
                              "OOD progress plateaued before gates; run one controlled LR decay")
            return result(BudgetAction.STOP_BLOCKED,
                          "OOD progress stayed flat after the allowed LR decays")

        return result(BudgetAction.CONTINUE,
                      "held-out progress remains material or evidence is insufficient")

    def state_dict(self) -> dict:
        return {
            "specs": [asdict(spec) for spec in self.specs],
            "config": asdict(self.config),
            "history": [asdict(obs) for obs in self.history],
            "lr_decays": self.lr_decays,
            "last_lr_decay_step": self.last_lr_decay_step,
            "best_step": self.best_step,
            "best_score": self.best_score,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "AdaptiveTrainingBudget":
        controller = cls(
            [MetricSpec(**item) for item in state["specs"]],
            BudgetConfig(**state["config"]),
        )
        controller.history = [BudgetObservation(**item) for item in state["history"]]
        controller.lr_decays = int(state["lr_decays"])
        controller.last_lr_decay_step = int(state["last_lr_decay_step"])
        controller.best_step = int(state["best_step"])
        controller.best_score = float(state["best_score"])
        return controller


def eye_v3_budget_specs() -> tuple[MetricSpec, ...]:
    """Default normalized gates for the first Eye-v3 odometry pilot.

    Baselines are deliberately expressed as ratios so this controller remains
    valid when the raw dataset values change.  A value above one means the
    baseline error divided by model error for ``*_gain`` metrics.
    """
    return (
        MetricSpec("depth_prior_gain", "max", 1.0, 1.20, weight=1.0),
        MetricSpec("pose_identity_gain", "max", 1.0, 1.20, weight=1.5),
        MetricSpec("ba_residual_reduction", "max", 0.0, 0.15, weight=1.0),
        MetricSpec("ba_pose_gain", "max", 1.0, 1.02, weight=1.5),
        MetricSpec("track_quality_score", "max", 0.0, 0.80, weight=1.5),
        MetricSpec("wrong_window_compatibility_gap", "max", 0.0, 0.15, weight=1.5),
        MetricSpec("reverse_time_rpe_ratio", "max", 1.0, 1.10, weight=1.0),
        MetricSpec("wrong_intrinsics_pose_ratio", "max", 1.0, 1.15, weight=1.0),
    )
