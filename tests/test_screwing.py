"""T107 鎖螺絲：螺紋力學、主軸策略與品質監控、手臂定位/搜尋/斜牙/進給/反作用。"""

import numpy as np
import pytest

from sixaxis.screwing import (
    RPM,
    DriverParams,
    FastenParams,
    JointSpec,
    ScrewCell,
    ScrewSpec,
    ThreadModel,
    bench_test,
    joint_stiffness,
    parse_job,
    run_job,
    tool_rotation,
)


# ------------------------------------------------------------------ 力學模型
def test_thread_geometry_and_vdi_formula():
    s = ScrewSpec()
    assert s.d2 == pytest.approx(3.5454e-3, rel=1e-4)  # ISO 724：M4×0.7
    assert s.d3 == pytest.approx(3.1412e-3, rel=1e-4)
    assert s.As == pytest.approx(8.78e-6, rel=5e-3)  # ISO 898-1 表列 8.78 mm²
    m = ThreadModel(s, JointSpec(mu_g=0.12, mu_k=0.12))
    K = 0.16 * s.P + 0.58 * 0.12 * s.d2 + 0.12 * (s.dk + s.dh) / 4
    assert m.K == pytest.approx(K)
    assert m.torque_to_force(m.torque(m.theta_s + 0.1)) == pytest.approx(m.clamp(m.theta_s + 0.1))
    # ν = 0.9 的鎖緊扭力約 2.8–2.9 N·m（與 VDI 2230 表列 M4 8.8、μ = 0.12 的量級一致）
    F90 = 0.9 / m.utilization(1.0)
    assert 2.7 < F90 * m.K < 3.0
    assert m.self_locking()
    assert ThreadModel(joint=JointSpec(mu_g=0.08, mu_k=0.08)).self_locking()
    assert s.phi_ct_deg == pytest.approx(5.0, abs=0.01)


def test_joint_stiffness_and_rundown():
    s, j = ScrewSpec(), JointSpec()
    c = joint_stiffness(s, j)
    assert 1e8 < c < 4e8
    m = ThreadModel(s, j)
    assert m.theta_s / (2 * np.pi) == pytest.approx((s.L - j.t_b) / s.P)
    assert m.depth(m.theta_s * 2) == pytest.approx(s.L - j.t_b)
    assert m.clamp(m.theta_s - 1.0) == 0.0
    # 座合後扭力–轉角斜率
    slope = (m.torque(m.theta_s + 0.02) - m.torque(m.theta_s + 0.01)) / 0.01
    assert slope == pytest.approx(m.k_theta, rel=1e-6)
    # 硬接合：snug 後到目標的轉角 < 30°（ISO 5393 硬接合）
    _, nr = bench_test(m, DriverParams())
    assert np.degrees(nr.angle_nom) < 30


def test_two_stage_vs_single_stage_overshoot():
    m = ThreadModel()
    r2, nr2 = bench_test(m, DriverParams())
    assert r2["ok"] and abs(r2["T_err_pct"]) < 1.0
    r1, nr1 = bench_test(m, DriverParams(two_stage=False))
    assert r1["error"] == "TORQUE_NG" and r1["T_err_pct"] > 20
    assert r1["t_spindle"] < r2["t_spindle"]  # 單段快但過衝
    # 能量法估計（以實際切斷轉速）與模擬同量級
    for r in (r1, r2):
        est = m.overshoot_estimate(r["w_cut"], DriverParams().J, r["T_cut"], DriverParams().T_max)
        assert r["T_peak"] - r["T_cut"] == pytest.approx(est, rel=0.35, abs=0.01)
    # 軟接合（剛性 1/30）單段式誤差小
    soft = ThreadModel(joint=JointSpec(c=m.c / 30))
    rs, _ = bench_test(soft, DriverParams(two_stage=False))
    assert abs(rs["T_err_pct"]) < 5


def test_friction_scatter_torque_vs_angle_control():
    nom = ThreadModel()
    Ft, Fa = [], []
    for mu in (0.08, 0.12, 0.16):
        m = ThreadModel(joint=JointSpec(mu_g=mu, mu_k=mu))
        rt, _ = bench_test(m, DriverParams(), nominal=nom)
        ra, _ = bench_test(m, DriverParams(strategy="angle"), nominal=nom)
        assert rt["F_clamp"] == pytest.approx(m.torque_to_force(2.5), rel=0.01)
        Ft.append(rt["F_clamp"])
        Fa.append(ra["F_clamp"])
    spread = lambda F: (max(F) - min(F)) / np.mean(F)  # noqa: E731
    assert spread(Fa) < spread(Ft) / 3


def test_cross_thread_and_no_snug_detection():
    m = ThreadModel()
    m.cross = True
    r, _ = bench_test(m, DriverParams())
    assert r["error"] == "CROSS_THREAD"
    stripped = ThreadModel(ScrewSpec(L=40e-3))  # 實際螺絲比標稱長很多 → 超過預期圈數仍未 snug
    r, _ = bench_test(stripped, DriverParams(), nominal=ThreadModel(), t_max=20)
    assert r["error"] == "NO_SNUG"


def test_tool_rotation():
    for a in ([0, 0, -1], [-np.sin(0.4), 0, -np.cos(0.4)], [0.3, -0.2, -0.9]):
        R = tool_rotation(a)
        assert np.allclose(R.T @ R, np.eye(3))
        assert np.allclose(R[:, 2], np.asarray(a) / np.linalg.norm(a))
    assert np.allclose(tool_rotation([0, 0, -1]), np.diag([1, -1, -1]))


def test_parse_job():
    items = parse_job({"holes": [{"name": "A", "p": [0.4, 0, 0.12], "torque": 2.0, "strategy": "angle"}]})
    h, p = items[0]
    assert h.name == "A" and p.driver.T_target == 2.0 and p.driver.strategy == "angle"
    for bad in ({}, {"holes": []}, {"holes": [{"p": [0, 0]}]}, {"holes": [{"p": [0, 0, 0], "torque": 9}]},
                {"holes": [{"p": [0, 0, 0], "strategy": "x"}]}, {"holes": [{"p": [0, 0, 0], "foo": 1}]}):
        with pytest.raises(ValueError):
            parse_job(bad)


# ------------------------------------------------------------------ 手臂
@pytest.fixture(scope="module")
def cell():
    c = ScrewCell()
    top, _, _ = c.add_plate("A", (0.45, 0.0, 0.10), (0.20, 0.14))
    c.top = top
    yield c
    c.close()


def test_fasten_nominal(cell):
    h = cell.add_hole("N1", cell.top + np.array([0.03, 0.02, 0]))
    r = cell.fasten(h, FastenParams(pick=True))
    assert r["ok"], r
    assert r["pos_err_mm"] < 0.05 and r["tilt_deg"] < 0.05
    assert abs(r["T_err_pct"]) < 1.0
    assert r["F_clamp"] == pytest.approx(ThreadModel().torque_to_force(2.5), rel=0.01)
    assert 20 < r["F_ax_min"] and r["F_ax_max"] < 30  # 跟隨進給：彈簧力 ≈ F0 + k·c_nom = 25 N
    assert r["roll_dev_deg"] < 0.05 and r["torque_util"] < 1.0
    assert r["turns_rundown"] == pytest.approx(r["turns_nom"], abs=0.05)


def test_fasten_offset_search(cell):
    h = cell.add_hole("S1", cell.top + np.array([-0.03, -0.02, 0]))
    r = cell.fasten(h, FastenParams(pick=False, hole_err=(1.2e-3, 0.5e-3, 0.0), search=False))
    assert r["error"] == "NO_ENGAGE"
    r = cell.fasten(h, FastenParams(pick=False, hole_err=(1.2e-3, 0.5e-3, 0.0)))
    assert r["ok"] and r["blocked"] and r["search_s"] < 2.0
    assert r["lat_cap_mm"] <= ((4.5 - 4.0) / 2 + 0.5) + 1e-6


def test_fasten_tilt_cross_thread(cell):
    h = cell.add_hole("X1", cell.top + np.array([0.0, 0.03, 0]))
    r = cell.fasten(h, FastenParams(pick=False, tilt_err_deg=3.0))
    assert r["ok"] and not r["cross"]
    r = cell.fasten(h, FastenParams(pick=False, tilt_err_deg=7.0))
    assert r["cross"] and r["error"] == "CROSS_THREAD"


def test_fasten_feed_and_reaction(cell):
    h = cell.add_hole("F1", cell.top + np.array([0.05, -0.03, 0]))
    r = cell.fasten(h, FastenParams(pick=False, feed="fixed"))
    assert r["error"] == "BIT_LIFT"  # 手臂不跟隨：9 mm 旋入 > 5 mm 預壓縮
    r = cell.fasten(h, FastenParams(pick=False, feed="fixed", c_start=9.5e-3))
    assert r["ok"] and r["F_ax_max"] - r["F_ax_min"] > 20
    r = cell.fasten(h, FastenParams(pick=False, react_ff=False))
    assert r["roll_dev_deg"] > 10  # 無前饋：計算力矩控制的 J6 等效剛性 M66·ωn² 太低


def test_tilted_plate_and_job():
    c = ScrewCell()
    try:
        a = np.radians(20)
        n = np.array([-np.sin(a), 0, np.cos(a)])
        top, _, R = c.add_plate("B", (0.40, -0.20, 0.13), (0.12, 0.10), n=n)
        r = c.fasten(c.add_hole("T1", top + R[:, 1] * 0.02, -n), FastenParams(pick=False))
        assert r["ok"] and r["tilt_deg"] < 0.05
        top2, _, _ = c.add_plate("A", (0.45, 0.10, 0.10), (0.12, 0.10))
        job = {"holes": [{"name": "J1", "p": list(top2), "torque": 2.0},
                         {"name": "J2", "p": list(top2 + [0.03, 0, 0]), "strategy": "angle", "torque": 2.0}]}
        out = run_job(c, job)
        assert out["ok"], out
        assert out["holes"][0]["T_peak"] == pytest.approx(2.0, rel=0.01)
    finally:
        c.close()


def test_rpm_constant():
    assert 60 * RPM == pytest.approx(2 * np.pi)
