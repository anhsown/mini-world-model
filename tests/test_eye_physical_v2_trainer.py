import math

from scripts.train_eye_physical_v2_ddp import cosine_lr


def test_cosine_lr_warms_then_decays_to_floor():
    base = 3e-4
    values = [cosine_lr(base, step, 1000) for step in range(1000)]
    assert math.isclose(values[0], base / 100)
    assert math.isclose(values[99], base)
    assert math.isclose(values[100], base)
    assert math.isclose(values[-1], base * 0.10)
    assert all(left >= right for left, right in zip(values[99:], values[100:]))
