"""T103 疊棧/碼垛測試：樣式產生器、靜力穩定判據、疊棧/金字塔執行、錯位塔倒塌層數與解析預測一致。"""

import numpy as np
import pytest

from sixaxis.commands import Executor, parse
from sixaxis.graspsim import GraspScene
from sixaxis.stacking import (
    FEED_XY,
    PITCH,
    A,
    H,
    Slot,
    column_slots,
    pyramid_slots,
    run_stack,
    stable_height,
    stack_margins,
    stack_program,
    tower_slots,
)
from sixaxis.tasks import Z_CLEAR, Z_OBJ


def _run(slots, **kw):
    s = GraspScene(workcell=True)
    try:
        return run_stack(s, Executor(s), slots, **kw)
    finally:
        s.close()


def _xy(slots):
    return np.array([[s.x, s.y] for s in slots])


def test_patterns():
    t = tower_slots(4, stagger=0.01)
    assert [s.layer for s in t] == [0, 1, 2, 3] and t[3].x - t[0].x == pytest.approx(0.03)
    assert t[2].z == pytest.approx(Z_OBJ + 2 * H)
    p = pyramid_slots(3)
    assert [s.layer for s in p] == [0, 0, 0, 1, 1, 2]
    assert p[3].x == pytest.approx((p[0].x + p[1].x) / 2) and p[5].x == pytest.approx(p[1].x)  # 壓縫
    c = column_slots(2, 3)
    assert len(c) == 6 and c[1].x - c[0].x == pytest.approx(PITCH) and c[4].x == pytest.approx(c[0].x)


def test_program_clearance_rises_with_layer():
    lo = parse(stack_program(FEED_XY, Slot(0.44, 0.0, 0)))
    hi = parse(stack_program(FEED_XY, Slot(0.44, 0.0, 6), dz_release=0.002))
    assert lo[4]["tcp"][2] == pytest.approx(Z_CLEAR)
    assert hi[4]["tcp"][2] > Z_OBJ + 6 * H + 0.05 and hi[5]["tcp"][2] == pytest.approx(Z_OBJ + 6 * H + 0.002)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_stagger_criterion_matches_closed_form(n):
    """等距錯位 d 的 n 層塔：穩定 ⇔ d < a/n。"""
    for d, stable in ((A / n * 0.98, True), (A / n * 1.02, False)):
        m = stack_margins(_xy(tower_slots(n, stagger=d)))
        assert bool(np.all(m > 0)) is stable
        assert stable_height(_xy(tower_slots(n + 2, stagger=d))) == (n if stable else n - 1)


def test_margin_of_aligned_tower_is_half_width():
    assert np.allclose(stack_margins(_xy(tower_slots(5))), A / 2)


def test_tower_of_four():
    m = _run(tower_slots(4))
    assert m["success"] and m["built"] == 4, m["error"]
    assert m["xy_max_mm"] < 1.5 and m["z_abs_max_mm"] < 0.5 and m["tilt_max_deg"] < 0.5
    assert m["torque_util"] < 1.0 and m["disturb_max_mm"] < 1.0
    z = [f["z"] for f in m["final"]]
    assert np.allclose(np.diff(z), H, atol=5e-4)


def test_pyramid_3_2_1():
    m = _run(pyramid_slots(3))
    assert m["success"] and m["built"] == 6, m["error"]
    assert m["xy_max_mm"] < 1.5 and m["tilt_max_deg"] < 0.5


@pytest.mark.parametrize("d", [0.012, 0.018])
def test_staggered_tower_collapses_at_predicted_height(d):
    slots = tower_slots(6, stagger=d)
    pred = stable_height(_xy(slots))
    m = _run(slots)
    assert pred < 6 and m["built"] == pred and m["error"] == f"COLLAPSE@{pred}"
