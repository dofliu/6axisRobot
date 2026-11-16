"""T108 去毛邊/拋光：墊片接觸模型、Preston 移除、導納力控、毛邊移除。"""

import numpy as np
import pytest

from sixaxis.finishing import (
    POL_L,
    T_POL_INV,
    TCP_LINK,
    FinishCell,
    FinishParams,
    ForceParams,
    PadModel,
    Surface,
    line_on,
    loop_model,
    normal_mass,
    pass_depth,
    predict_touchdown,
    removal_stats,
    routh_kcrit,
    tool_rot,
)
from sixaxis.welding import Q_SEED

C = (0.45, 0.0, 0.15)


def _flat(res=5e-4, size=(0.16, 0.10)):
    return Surface(C, size=size, res=res)


def test_tool_tcp_matches_fk():
    c = FinishCell()
    try:
        s = _flat()
        T = np.eye(4)
        T[:3, :3] = tool_rot(s)
        T[:3, 3] = s.c + np.array([0.0, 0.0, 0.02])
        q = c.r.ik(T @ T_POL_INV, Q_SEED)[0]
        c.sim.set_state(q)
        Tp = c.sim.link_pose(TCP_LINK)
        assert np.allclose(Tp[:3, 3], T[:3, 3], atol=1e-9)
        assert np.allclose(Tp[:3, 2], [0, 0, -1], atol=1e-9)
        assert np.isclose(np.linalg.norm(c.r.fk(q)[:3, 3] - Tp[:3, 3]), POL_L)
    finally:
        c.close()


def test_flat_contact_force_friction_torque():
    pad = PadModel(k_e=5e4)
    s = _flat(res=2.5e-4)
    d = 4e-4
    p = s.world(0.0, 0.0, -d)
    Fn, F, M, nc, dmax = pad.contact(s, p, -s.n, np.zeros(3), 0.0)
    assert Fn == pytest.approx(pad.k_e * d, rel=0.01)
    assert dmax == pytest.approx(d)
    assert np.linalg.norm(F[:2]) < 0.01 * Fn  # 對稱足印 → 摩擦合力 ≈ 0
    assert abs(M @ s.n) == pytest.approx(2 / 3 * pad.mu * Fn * pad.radius, rel=0.02)
    # 離開表面 → 無力
    assert pad.contact(s, s.world(0, 0, 1e-3), -s.n, np.zeros(3), 0.0)[0] == 0.0


def test_single_pass_depth_matches_preston():
    """墊以固定壓縮沿 u 等速掃過（運動學驅動），移除剖面 vs 解析 h(w)。"""
    pad = PadModel(k_e=5e4)
    s = _flat(res=2.5e-4, size=(0.10, 0.04))
    d, v, dt = 4e-4, 0.025, 1e-3
    for u in np.arange(-0.04, 0.04, v * dt):
        pad.contact(s, s.world(u, 0.0, -d), -s.n, s.u * v, dt)
    R = s.removal()[np.abs(s.gu) < 0.005].mean(0)
    w = s.gw
    h = pass_depth(w, pad.k_e * d, v, pad)
    core = np.abs(w) < 0.9 * pad.radius
    assert np.max(np.abs(R[core] - h[core])) < 0.03 * h.max()
    assert h[np.argmin(np.abs(w))] == pytest.approx(pad.k_p * pad.k_e * d * pad.omega / (np.pi * v), rel=0.01)


def test_ridge_and_burr_height():
    s = Surface(C, size=(0.06, 0.02), res=2.5e-4).set_shape(offset=1e-4)
    s.add_ridge(0.002, 1e-3, 3e-4, u_range=(-0.02, 0.02))
    b = s.burr()
    assert b.max() == pytest.approx(3e-4)
    assert b[np.abs(s.gu) > 0.025].max() == 0.0
    assert s.H0[np.argmin(np.abs(s.gu)), 0] == pytest.approx(1e-4)


def test_loop_model_and_routh():
    lm = loop_model(5e4, 2.0, 2 * np.sqrt(5e4 * 2.0))
    assert lm["zeta"] == pytest.approx(1.0)
    kc = routh_kcrit(2.0, 600.0, 50.0)
    # 剛性內環預測器：k_e 低於/高於臨界時 → 收斂/持續振盪（無 v 飽和影響：v_app 取大）
    lo = predict_touchdown(0.7 * kc, 2.0, 600.0, 20.0, 1.0, f_c=50.0, T=2.0)
    hi = predict_touchdown(1.3 * kc, 2.0, 600.0, 20.0, 1.0, f_c=50.0, T=2.0)
    assert lo["osc_std"] < 0.05 and lo["F_end"] == pytest.approx(20.0, rel=0.01)
    assert hi["osc_std"] > 1.0


@pytest.fixture(scope="module")
def flat_run():
    c = FinishCell()
    s = _flat()
    c.add_surface(s)
    m = c.finish(line_on(s, -0.03, 0.03, 0.0), FinishParams())
    lam = normal_mass(c, c.rec["q"][-1], s.n)
    c.close()
    return m, s, lam


def test_admittance_holds_force_on_flat(flat_run):
    m, s, _ = flat_run
    assert m["ok"] and m["error"] is None
    assert m["F_rms_err"] < 0.3 and m["loss_frac"] == 0.0
    assert m["lat_max_mm"] < 0.05 and m["torque_util"] < 0.5
    st = removal_stats(s, 0.01, 0.0015)
    h0 = pass_depth(0.0, 20.0, 0.025, PadModel())
    assert st["mean_um"] == pytest.approx(h0 * 1e6, rel=0.08)


def test_touchdown_matches_servo_model(flat_run):
    m, _, lam = flat_run
    assert 3.0 < lam < 8.0
    fp = ForceParams()
    pr = predict_touchdown(5e4, fp.M, fp.D, fp.F_d, fp.v_app, fp.f_c, lam=lam)
    assert m["land_overshoot"] == pytest.approx(pr["overshoot"], abs=0.03)


def test_offset_surface_position_vs_admittance():
    """實際表面比名目高 0.4 mm：位置控制力 ≈ F_d + k_e·Δh，導納維持 F_d。"""
    out = {}
    for mode in ("position", "admittance"):
        c = FinishCell()
        s = _flat().set_shape(offset=4e-4)
        c.add_surface(s)
        out[mode] = c.finish(line_on(s, -0.02, 0.02, 0.0), FinishParams(force=ForceParams(mode=mode)))
        c.close()
    assert out["position"]["F_mean"] == pytest.approx(20.0 + 5e4 * 4e-4, rel=0.05)
    assert out["admittance"]["F_rms_err"] < 0.3


def test_slope_ramp_error_and_integral():
    """沿行進方向 1.5° 上坡（表面以 ż_s = v·tanα 抬升）：穩態力超出 ≈ D·v·tanα；K_i > 0 時消除。"""
    out = {}
    for ki in (0.0, 20.0):
        c = FinishCell()
        s = _flat().set_shape(slope_u_deg=1.5)
        c.add_surface(s)
        fp = ForceParams(K_i=ki)
        m = c.finish(line_on(s, -0.03, 0.03, 0.0), FinishParams(force=fp))
        F = np.array(c.frec["Fn"][m["fi0"]:m["fi1"]])
        t = np.arange(len(F)) * c.dt
        mid = (t > 0.8) & (t < t[-1] - 0.8)
        out[ki] = F[mid].mean()
        c.close()
    e_pred = 600.0 * 0.025 * np.tan(np.radians(1.5))
    assert out[0.0] - 20.0 == pytest.approx(e_pred, rel=0.15)
    assert abs(out[20.0] - 20.0) < 0.15 * e_pred


def test_deburr_hard_pad_removes_burr():
    pad = PadModel(k_e=5e5)
    c = FinishCell(pad=pad)
    s = Surface(C, size=(0.08, 0.04), res=2.5e-4)
    s.add_ridge(0.0, 1e-3, 2e-4, u_range=(-0.02, 0.02))
    c.add_surface(s)
    seam = line_on(s, -0.03, 0.03, -0.0075)
    prm = FinishParams(speed=0.01, force=ForceParams(F_d=40.0, D=2000.0))
    b0 = s.burr().max()
    m = c.finish(seam, prm)
    b1 = s.burr().max()
    c.close()
    assert m["ok"] and m["F_rms_err"] < 1.0
    assert b1 < 0.8 * b0
    base = ~s.ridge & (np.abs(s.gu)[:, None] < 0.015)
    assert s.removal()[base].max() < 0.5 * (b0 - b1)  # 選擇性：毛邊降幅 > 2× 母材過切
