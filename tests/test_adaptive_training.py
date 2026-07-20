import math

from jwm.adaptive_training import (
    AdaptiveTrainingBudget,
    BudgetAction,
    BudgetConfig,
    MetricSpec,
)
from scripts.train_eye_v3_ddp import apply_distributed_lr_decision


def _controller(*, final=False, max_lr_decays=1):
    specs = (
        MetricSpec("ood_gain", "max", 1.0, 1.2),
        MetricSpec("wrong_ratio", "max", 1.0, 1.25),
    )
    cfg = BudgetConfig(
        min_steps=200, max_steps=2000, eval_every=100,
        slope_window=3, plateau_patience=3, overfit_patience=2,
        max_lr_decays=max_lr_decays, min_progress_per_eval=0.02,
        final_stage=final,
    )
    return AdaptiveTrainingBudget(specs, cfg)


def _observe(controller, step, gain, wrong, loss=2.0, grad=1.0):
    controller.observe(step, {"ood_gain": gain, "wrong_ratio": wrong},
                       train_loss=loss, grad_norm=grad, learning_rate=3e-4)


def test_minimum_budget_prevents_premature_stop():
    controller = _controller()
    _observe(controller, 100, 1.0, 1.0)
    assert controller.decide().action == BudgetAction.CONTINUE


def test_passed_plateau_advances_stage_and_reports_best():
    controller = _controller()
    for step, gain, wrong in (
        (200, 1.19, 1.24), (300, 1.21, 1.26), (400, 1.21, 1.26),
        (500, 1.21, 1.26), (600, 1.21, 1.26),
    ):
        _observe(controller, step, gain, wrong)
    decision = controller.decide()
    assert decision.action == BudgetAction.ADVANCE_STAGE
    assert decision.best_step in {300, 400, 500, 600}
    assert decision.projected_total_steps == 600


def test_failed_plateau_decays_lr_then_blocks():
    controller = _controller(max_lr_decays=1)
    for step in (200, 300, 400, 500, 600):
        _observe(controller, step, 1.05, 1.05)
    assert controller.decide().action == BudgetAction.REDUCE_LR
    controller.acknowledge_lr_decay()
    _observe(controller, 700, 1.05, 1.05)
    assert controller.decide().action == BudgetAction.STOP_BLOCKED


def test_ood_regression_while_train_loss_improves_stops_overfit():
    controller = _controller()
    _observe(controller, 200, 1.15, 1.15, loss=2.0)
    _observe(controller, 300, 1.05, 1.05, loss=1.8)
    _observe(controller, 400, 1.04, 1.04, loss=1.5)
    assert controller.decide().action == BudgetAction.STOP_OVERFIT


def test_unhealthy_gradient_is_a_veto():
    controller = _controller()
    _observe(controller, 200, 1.2, 1.25, grad=math.inf)
    assert controller.decide().action == BudgetAction.STOP_UNSTABLE


def test_state_round_trip_preserves_patience_and_best_checkpoint():
    controller = _controller()
    _observe(controller, 200, 1.1, 1.1)
    _observe(controller, 300, 1.12, 1.12)
    restored = AdaptiveTrainingBudget.from_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
    assert restored.decide() == controller.decide()


def test_ddp_lr_decay_mutates_only_owner_but_scales_every_rank():
    owner = _controller()
    _observe(owner, 200, 1.05, 1.05)
    worker = _controller()  # Rank 1 intentionally has no observations.

    owner_factor = apply_distributed_lr_decision(
        BudgetAction.REDUCE_LR, owner, 1.0, controller_owner=True)
    worker_factor = apply_distributed_lr_decision(
        BudgetAction.REDUCE_LR, worker, 1.0, controller_owner=False)

    assert owner_factor == worker_factor == 0.5
    assert owner.lr_decays == 1
    assert worker.lr_decays == 0
