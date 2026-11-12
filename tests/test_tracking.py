"""T104 輸送帶追蹤抓取測試：α-β 估測、追蹤窗判據、同步追蹤夾取、開/閉迴路、後備靜態抓取、連續來料。"""

import numpy as np
import pytest

from sixaxis import scene as S
from sixaxis.commands import Executor
from sixaxis.graspsim import GraspScene
from sixaxis.tracking import (
    ALPHA,
    BELT_Y,
    BETA,
    CAM_DT,
    T_DESC,
    T_GRIP_NOM,
    T_LIFT,
    T_SYNC,
    X_DET,
    X_END_MAX,
    X_WAIT,
    BeltEstimator,
    alpha_beta_noise_ratio,
    run_flow,
    run_track_cycle,
    v_max_window,
    window_length,
    x_end_pred,
)

PC = S.pocket_centers()


def _cycle(**kw):
    s = GraspScene(workcell=True)
    try:
        return run_track_cycle(s, Executor(s), PC[0], **kw)
    finally:
        s.close()


def test_estimator_open_loop_and_alpha_beta_convergence():
    v = 0.12
    op = BeltEstimator("open", 0.10)
    ab = BeltEstimator("track", 0.10)
    for e in (op, ab):
        e.init(0.0, 0.2, BELT_Y)
    for k in range(1, 151):
        t = k * CAM_DT
        for e in (op, ab):
            e.update(t, 0.2 + v * t, BELT_Y + 0.004)
    t = 150 * CAM_DT
    assert op.predict(t)[0] == pytest.approx(0.2 + 0.10 * t)  # 開迴路誤差 = ε·v_nom·t
    x, vh, y = ab.predict(t + 0.01)
    assert vh == pytest.approx(v, abs=1e-5) and x == pytest.approx(0.2 + v * (t + 0.01), abs=1e-6)
    assert y == pytest.approx(BELT_Y + 0.004, abs=1e-6)
    with pytest.raises(ValueError):
        BeltEstimator("kalman", 0.1)


def test_alpha_beta_noise_reduction_matches_theory():
    rng = np.random.default_rng(5)
    sig, v = 1e-3, 0.1
    e = BeltEstimator("track", v)
    e.init(0.0, 0.0, BELT_Y)
    res = []
    for k in range(1, 40001):
        t = k * CAM_DT
        e.update(t, v * t + rng.normal(0, sig), BELT_Y)
        if k > 500:
            res.append(e.x - v * t)
    ratio = np.var(res) / sig**2
    assert ratio == pytest.approx(alpha_beta_noise_ratio(ALPHA, BETA), rel=0.08)
    assert ratio < 0.3


def test_window_criterion():
    assert window_length(0.2) == pytest.approx(0.2 * (T_SYNC / 2 + T_DESC + T_GRIP_NOM + T_LIFT / 2))
    assert x_end_pred(0.1) == pytest.approx(X_WAIT + window_length(0.1))  # 低速：觸發點在光電下游
    v_hi = 0.3
    assert x_end_pred(v_hi) == pytest.approx(X_DET + v_hi * (T_SYNC + T_DESC + T_GRIP_NOM + T_LIFT / 2))
    vm = v_max_window()
    assert x_end_pred(vm) == pytest.approx(X_END_MAX, abs=1e-9)
    assert 0.25 < vm < 0.28


@pytest.mark.parametrize("v", [0.05, 0.25])
def test_track_pick_and_place(v):
    m = _cycle(belt_speed=v)
    assert m["success"], m["error"]
    assert m["method"] == "track" and m["track_rms_mm"] < 1.0 and abs(m["dx_mm"]) < 1.5
    assert m["place"]["xy_err_mm"] < 1.5 and abs(m["place"]["z_err_mm"]) < 0.5
    assert m["vel_util"] < 1.0 and m["torque_util"] < 1.0
    assert m["x_end"] <= X_END_MAX + 1e-6


def test_too_fast_is_rejected_then_static_fallback():
    m = _cycle(belt_speed=0.30)
    assert not m["success"] and m["error"] == "WINDOW"
    m = _cycle(belt_speed=0.30, fallback=True)
    assert m["success"] and m["method"] == "static" and m["track_error"] == "WINDOW"


def test_speed_mismatch_open_vs_closed_loop():
    eps, v_nom = 0.2, 0.10
    op = _cycle(belt_speed=v_nom * (1 + eps), v_nom=v_nom, mode="open")
    pred = -eps * v_nom * (op["t_grip"] - op["t_det"]) * 1e3  # TCP − 工件 [mm]
    assert op["track_end_mm"] == pytest.approx(pred, abs=1.5)
    assert not op["success"] and op["error"] == "INHAND_OFFSET"
    cl = _cycle(belt_speed=v_nom * (1 + eps), v_nom=v_nom, mode="track")
    assert cl["success"] and abs(cl["track_end_mm"]) < 1.5
    assert cl["v_est_grip"] == pytest.approx(v_nom * (1 + eps), rel=0.05)


def test_lateral_offset():
    m = _cycle(belt_speed=0.10, y=BELT_Y + 0.010)
    assert m["success"] and abs(m["dy_mm"]) < 0.5
    m = _cycle(belt_speed=0.10, y=BELT_Y + 0.020)
    assert not m["success"] and m["error"] == "LATERAL"


@pytest.mark.parametrize("period, all_tracked", [(8.0, True), (5.0, False)])
def test_flow(period, all_tracked):
    s = GraspScene(workcell=True)
    try:
        f = run_flow(s, Executor(s), 4, period, 0.10, PC)
    finally:
        s.close()
    assert f["success"] and f["placed"] == 4
    assert (f["tracked"] == 4) == all_tracked
    assert f["tracked"] + f["static"] == 4
    assert max(i["xy_err_mm"] for i in f["items"]) < 1.5
