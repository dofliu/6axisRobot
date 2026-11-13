"""T105 銲接路徑測試：銲槍 TCP、焊道幾何、等速時間律、銲槍角度、規劃與 PyBullet 模擬量測。"""

import json

import numpy as np
import pytest

from sixaxis import welding as W
from sixaxis.robot import SixAxisRobot
from sixaxis.welding import (
    Seam,
    SpeedProfile,
    WeldCell,
    WeldParams,
    bead_area,
    best_roll,
    heat_input,
    parse_job,
    plan_seam,
    run_job,
    tool_rotation,
    torch_angles,
    turtle_seam,
)

Z = 0.108  # 鋼板頂面
PLATE = ("plate", (0.50, 0.0, 0.104), (0.30, 0.30, 0.008), (0.6, 0.6, 0.65, 1))


def short_seam(r=0.02, **kw):
    return turtle_seam((0.42, -0.04, Z), (0, 1, 0), [("line", 0.04), ("arc", r, -90), ("line", 0.03)], **kw)


@pytest.fixture(scope="module")
def r():
    return SixAxisRobot()


@pytest.fixture
def cell():
    c = WeldCell()
    c.add_box(*PLATE)
    yield c
    c.close()


@pytest.fixture(scope="module")
def nominal():
    """短焊道（直線 40 + R20 圓弧 90° + 直線 30 mm）以 20 mm/s 銲一次，多個測試共用。"""
    c = WeldCell()
    c.add_box(*PLATE)
    s = short_seam()
    m = c.weld(s, WeldParams(speed=0.02))
    d = c.series(s, WeldParams(speed=0.02), m["i0"], m["i1"])
    c.close()
    return m, d


# ------------------------------------------------------------------ 銲槍與幾何
def test_tcp_matches_pybullet(cell, rng):
    for _ in range(20):
        q = rng.uniform(-2.5, 2.5, 6)
        cell.sim.set_state(q)
        assert np.abs(cell.tcp_pose() - cell.r.fk(q) @ W.T_TCP).max() < 1e-6
    # 銲絲方向與法蘭 z 夾角 = 彎頸角；TCP 距法蘭 = 兩段向量和
    assert np.degrees(np.arccos(W.T_TCP[2, 2])) == pytest.approx(W.NECK_DEG)
    ln = W.NECK_L + W.STICKOUT
    assert np.linalg.norm(W.T_TCP[:3, 3]) == pytest.approx(
        np.sqrt(W.BODY_L**2 + ln**2 + 2 * W.BODY_L * ln * np.cos(np.radians(W.NECK_DEG))))


def test_seam_geometry():
    s = short_seam()
    assert s.length == pytest.approx(0.04 + 0.02 * np.pi / 2 + 0.03)
    assert s.kink_deg < 1e-6
    assert np.allclose(s.p[0], [0.42, -0.04, Z]) and np.allclose(s.p[-1], [0.47, 0.02, Z], atol=1e-12)
    assert np.allclose(s.tangent(0.0), [0, 1, 0]) and np.allclose(s.tangent(s.length), [1, 0, 0], atol=1e-12)
    assert s.psi[-1] == pytest.approx(-np.pi / 2)  # 右轉 90°
    # 表上切線 = 位置差分；圓弧段各點到圓心距離 = R
    d = np.diff(s.p, axis=0) / np.diff(s.s)[:, None]
    assert np.abs(d - (s.t[:-1] + s.t[1:]) / 2).max() < 2e-3  # 跨接點的格子誤差 ~κ·ds/4
    arc = (s.s > 0.04) & (s.s < 0.04 + 0.02 * np.pi / 2)
    assert np.abs(np.linalg.norm(s.p[arc] - [0.44, 0.0, Z], axis=1) - 0.02).max() < 1e-12
    sn, pn, _ = s.nearest([[0.43, -0.02, Z + 0.01]])
    assert sn[0] == pytest.approx(0.02, abs=1e-4) and np.allclose(pn[0], [0.42, -0.02, Z], atol=1e-4)


def test_seam_validation():
    with pytest.raises(ValueError, match="不連續"):
        Seam((0, 0, 0), [("line", (0.1, 0, 0)), ("line", (0.1, 0.1, 0))])
    with pytest.raises(ValueError, match="平面"):
        Seam((0, 0, 0), [("line", (0.1, 0, 0.01))])
    with pytest.raises(ValueError, match="共線"):
        Seam((0, 0, 0), [("arc", (0.1, 0, 0), (0.2, 0, 0))])
    with pytest.raises(ValueError, match="平面"):
        Seam((0, 0, 0), [("arc", (0.1, 0, 0.1), (0.2, 0, 0))])
    c = turtle_seam((0.5, 0, 0), (1, 0, 0), [("arc", 0.03, 180), ("arc", 0.03, 180)])  # 整圓
    assert c.length == pytest.approx(2 * np.pi * 0.03) and np.allclose(c.p[0], c.p[-1], atol=1e-12)
    # 封閉焊道：s_hint 區分起點與終點
    assert c.nearest([c.p[0]], [c.length])[0][0] == pytest.approx(c.length)
    assert c.nearest([c.p[0]], [0.0])[0][0] == pytest.approx(0.0)


def test_speed_profile():
    p = SpeedProfile(0.1, 0.01, 0.2)
    assert p.T == pytest.approx(10.2)
    t = np.linspace(0, p.T, 20401)
    s, sd = p(t)
    assert s[0] == 0 and s[-1] == pytest.approx(0.1) and sd[0] == 0 and sd[-1] == pytest.approx(0, abs=1e-12)
    mid = (t >= 0.2) & (t <= p.T - 0.2)
    assert np.allclose(sd[mid], 0.01)
    assert np.abs(np.gradient(s, t) - sd).max() < 1e-6  # ṡ 與 s 一致
    acc = np.gradient(sd, t)
    assert np.abs(acc).max() == pytest.approx(1.5 * 0.01 / 0.2, rel=1e-3)  # smoothstep 峰值加速度 1.5 v/t_r
    with pytest.raises(ValueError, match="太短"):
        SpeedProfile(0.001, 0.01, 0.2)


@pytest.mark.parametrize("work,travel,roll",
                         [(90, 0, 0), (90, 10, 0), (45, 15, 90), (135, -15, -90), (60, 5, 180)])
def test_torch_angles_roundtrip(work, travel, roll):
    """tool_rotation 給定的 work/travel 角，沿直線與圓弧都能由 torch_angles 量回。"""
    s = short_seam()
    for x in np.linspace(0, s.length, 9):
        R, _ = tool_rotation(s, float(x), work, travel, roll)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1)
        w, t = torch_angles(s, [s.pos(float(x))], [R[:, 2]])
        assert w[0] == pytest.approx(work, abs=1e-6) and t[0] == pytest.approx(travel, abs=1e-6)
    R, _ = tool_rotation(s, 0.0, 90, 0)
    assert np.allclose(R[:, 2], [0, 0, -1]) and np.allclose(R[:, 0], [0, 1, 0])  # 垂直朝下、x = 行進方向
    R, _ = tool_rotation(s, 0.0, 90, 10)
    assert R[:, 2] @ s.tangent(0.0) == pytest.approx(np.sin(np.radians(10)))  # push：銲絲指向行進方向


def test_heading_blend():
    """航向平滑：接點處偏離切線 |κ|·b/4，dψ/ds 連續；無平滑時 dψ/ds 在接點階躍 κ。"""
    s, b = short_seam(), 0.004
    kap = -1 / 0.02
    assert s.heading(0.04 - 1e-5)[1] == pytest.approx(0.0, abs=1e-9)
    assert s.heading(0.04 + 2e-4)[1] == pytest.approx(kap)
    psi, dpsi = s.heading(0.04, b)
    assert psi - s.heading(0.04)[0] == pytest.approx(kap * b / 4, rel=1e-3)
    assert dpsi == pytest.approx(kap / 2, rel=1e-3)
    d = [s.heading(x, b)[1] for x in np.linspace(0.03, 0.05, 201)]
    assert np.abs(np.diff(d)).max() < abs(kap) * 1e-4 / (2 * b) * 1.5  # 斜率 ≤ κ/(2b)
    assert s.heading(0.02, b)[0] == pytest.approx(0.0, abs=1e-12)  # 遠離接點不受影響


def test_process_formulas():
    assert heat_input(24, 180, 0.010) == pytest.approx(345.6)  # 0.8·24·180/10 J/mm
    assert bead_area(1.2e-3, 0.1, 0.010) == pytest.approx(np.pi * 1.2**2 / 4 * 10)
    assert heat_input(24, 180, 0.005) == pytest.approx(2 * heat_input(24, 180, 0.010))


# ------------------------------------------------------------------ 規劃
def test_plan_follows_seam(r):
    s, prm = short_seam(), WeldParams(speed=0.02, work_deg=60, travel_deg=10, roll_deg=90)
    pl = plan_seam(r, s, prm, W.Q_SEED, dt=0.02)
    assert pl.ok and pl.ik_fail == 0 and pl.T == pytest.approx(s.length / 0.02 + 0.2)
    prof = SpeedProfile(s.length, 0.02, 0.2)
    for k in range(0, len(pl.t), 7):
        T = r.fk(pl.q[k]) @ W.T_TCP
        sk, sdk = prof(pl.t[k])
        assert np.linalg.norm(T[:3, 3] - s.pos(float(sk))) < 1e-8
        w, t = torch_angles(s, [T[:3, 3]], [T[:3, 2]])
        assert w[0] == pytest.approx(60, abs=0.05) and t[0] == pytest.approx(10, abs=0.05)
        tw = r.jacobian(pl.q[k]) @ pl.qd[k]
        v = tw[:3] + np.cross(tw[3:], T[:3, 3] - r.fk(pl.q[k])[:3, 3])
        assert np.linalg.norm(v) == pytest.approx(float(sdk), abs=1e-7)
    # 圓弧段銲槍角速度 = v/R（繞 −n，右轉）
    k = np.argmin(np.abs(prof(pl.t)[0] - 0.055))
    assert (r.jacobian(pl.q[k]) @ pl.qd[k])[3:] == pytest.approx([0, 0, -0.02 / 0.02], abs=1e-6)


def test_plan_limit_and_unreachable(r):
    pl = plan_seam(r, short_seam(), WeldParams(speed=0.15), W.Q_SEED, dt=0.02)  # 圓弧 ω = 7.5 rad/s > q̇max
    assert not pl.ok and pl.vel_ratio > 1.0
    far = turtle_seam((0.95, 0.0, Z), (0, 1, 0), [("line", 0.05)])
    assert plan_seam(r, far, WeldParams(), W.Q_SEED, dt=0.02).ik_fail > 0


def test_best_roll_circle(r):
    """整圓焊道：roll 0° 會撞 q6 限位（±2π），自動選到可行且 σ_min 最大者。"""
    c = turtle_seam((0.48, -0.03, Z), (1, 0, 0), [("arc", 0.03, 180), ("arc", 0.03, 180)])
    roll, rows = best_roll(r, c, WeldParams(work_deg=135), W.Q_SEED)
    by = {x["roll_deg"]: x for x in rows}
    assert not by[0.0]["limit_ok"] and by[0.0]["q6_min"] < -2 * np.pi
    assert by[roll]["ok"] and by[roll]["sigma_min"] == max(x["sigma_min"] for x in rows if x["ok"])
    assert by[roll]["q_range_deg"][5] == pytest.approx(360, abs=1)  # 腕 3 轉一整圈


# ------------------------------------------------------------------ 模擬
def test_weld_nominal(nominal):
    m, d = nominal
    assert m["ok"] and not m["collision"]
    assert m["path_max_mm"] < 0.01 and m["track_max_mm"] < 0.05
    assert m["speed_mean_mm_s"] == pytest.approx(20.0, abs=0.01) and m["speed_dev_max_pct"] < 1.0
    assert m["work_err_max_deg"] < 0.1 and m["travel_err_max_deg"] < 0.1 and m["heading_err_max_deg"] < 0.5
    assert m["vel_util"] < 1.0 and m["torque_util"] < 1.0 and m["end_err_mm"] < 0.01
    assert 5.0 < m["clear_mm"] < W.STICKOUT * 1e3  # 噴嘴離板面：小於伸出長度、未碰板
    assert m["weld_s"] == pytest.approx(m["length_mm"] / 20.0 + 0.2)
    # 平均延遲約 1 個控制週期（1 ms × 20 mm/s = 0.02 mm）
    lag = np.linalg.norm(d["p"] - d["p_ref"], axis=1)
    assert np.median(lag) == pytest.approx(0.02e-3, abs=0.01e-3)


def test_weld_records_arc_and_series(nominal):
    m, d = nominal
    assert len(d["t"]) == m["i1"] - m["i0"] and d["t"][-1] == pytest.approx(m["weld_s"], abs=2e-3)
    assert d["speed"].max() < 0.0205 and d["speed"][0] < 0.001  # 由靜止起弧、不超速


def test_blend_reduces_corner_transient(cell):
    """R5 小圓角：航向平滑降低關節加速度需求與速度波動，代價是接點航向偏差 ≈ κb/4。"""
    res = {}
    for b in (0.0, 0.002):
        c = WeldCell()
        c.add_box(*PLATE)
        res[b] = c.weld(short_seam(0.005), WeldParams(speed=0.01, blend=b))
        c.close()
    assert res[0.0]["ok"] and res[0.002]["ok"]
    assert res[0.002]["plan_qdd_max"] < 0.1 * res[0.0]["plan_qdd_max"]
    assert res[0.002]["speed_dev_max_pct"] < res[0.0]["speed_dev_max_pct"]
    assert res[0.002]["heading_err_max_deg"] == pytest.approx(np.degrees(0.002 / 0.005 / 4), rel=0.1)
    assert res[0.0]["heading_err_max_deg"] < 1.5


def test_errors(cell):
    m = cell.weld(short_seam(), WeldParams(speed=0.15))
    assert not m["ok"] and m["error"] == "LIMIT"
    m = cell.weld(turtle_seam((0.95, 0.0, Z), (0, 1, 0), [("line", 0.05)]))
    assert not m["ok"] and m["error"] == "UNREACHABLE"
    m = cell.weld(turtle_seam((0.45, 0.0, Z), (0, 1, 0), [("line", 0.001)]))
    assert not m["ok"] and m["error"] == "PLAN"
    assert cell.t == 0.0  # 失敗的焊道不動作


def test_precheck_collision(cell):
    """焊道埋在板面下 20 mm：預檢發現噴嘴穿入鋼板 → COLLISION，不執行。"""
    m = cell.weld(turtle_seam((0.45, 0.0, Z - 0.02), (0, 1, 0), [("line", 0.05)]), WeldParams(speed=0.02))
    assert m["error"] == "COLLISION" and m["precheck_clear_mm"] < 0 and cell.t == 0.0


def test_pd_worse_than_ctc(nominal):
    c = WeldCell(controller="pd")
    c.add_box(*PLATE)
    m = c.weld(short_seam(), WeldParams(speed=0.02))
    c.close()
    assert m["ok"] and m["path_max_mm"] < 1.0
    assert m["path_max_mm"] > 10 * nominal[0]["path_max_mm"]
    assert m["speed_dev_max_pct"] > nominal[0]["speed_dev_max_pct"]


# ------------------------------------------------------------------ JSON 工作
JOB = {"seams": [
    {"name": "A", "start": [0.42, -0.04, Z], "segments": [{"type": "line", "to": [0.42, 0.0, Z]}],
     "speed": 0.02, "travel_deg": 10},
    {"name": "B", "start": [0.50, 0.0, Z],
     "segments": [{"type": "arc", "via": [0.52, 0.02, Z], "to": [0.50, 0.04, Z]}],
     "speed": 0.02, "work_deg": 60, "travel_deg": 0},
]}


def test_parse_job_validation():
    seams = parse_job(json.dumps(JOB))
    assert [s.name for s, _ in seams] == ["A", "B"] and seams[1][1].work_deg == 60
    assert seams[1][0].length == pytest.approx(np.pi * 0.02)
    bad = [{}, {"seams": []}, {"seams": [{"start": [0, 0, 0]}]},
           {"seams": [{**JOB["seams"][0], "speed": 0.0}]}, {"seams": [{**JOB["seams"][0], "work_deg": 200}]},
           {"seams": [{**JOB["seams"][0], "foo": 1}]},
           {"seams": [{**JOB["seams"][0], "segments": [{"type": "spline"}]}]}]
    for b in bad:
        with pytest.raises(ValueError):
            parse_job(b)


def test_run_job(cell):
    out = run_job(cell, JOB)
    assert out["ok"] and len(out["seams"]) == 2
    for m in out["seams"]:
        assert m["path_max_mm"] < 0.01 and m["speed_dev_max_pct"] < 1.0 and not m["collision"]
        assert m["roll_tried"][-1]["error"] is None
    q, _ = cell.sim.get_state()
    assert np.abs(q - cell.q_home).max() < 1e-3  # 回到原點
    arc = np.array(cell.rec["arc"])
    assert np.count_nonzero(np.diff(arc.astype(int)) == 1) == 2  # 起弧兩次
