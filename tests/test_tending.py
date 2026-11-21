"""T112 機台上下料：門運動、成品質量性質、虎鉗定位/夾持力、週期模型、PLC 互鎖、上下料流程與故障恢復。"""

import numpy as np
import pytest

from sixaxis import tending as T
from sixaxis.graspsim import Q_REF
from sixaxis.gripper import CUBE
from sixaxis.planning import CollisionChecker
from sixaxis.robot import G
from sixaxis.scenarios import workcell_boxes
from sixaxis.sorting import ik_near


def test_door_trapezoid():
    s, v, a = T.DOOR_STROKE, T.DOOR_V, T.DOOR_A
    Td = T.door_time()
    assert Td == pytest.approx(s / v + v / a)
    ts = np.linspace(0, Td, 2001)
    x = np.array([T.door_pos(t) for t in ts])
    assert x[0] == 0 and x[-1] == pytest.approx(s)
    dx = np.diff(x) / np.diff(ts)
    assert np.all(dx >= -1e-12) and dx.max() <= v + 1e-6  # 單調、速度不超過 v_max
    assert np.max(np.abs(np.diff(dx) / np.diff(ts)[1:])) <= a * 1.01  # 加速度不超過 a_max
    # 短行程：三角形曲線
    assert T.door_time(0.01) == pytest.approx(2 * np.sqrt(0.01 / a))
    assert T.door_pos(T.door_time(0.01), 0.01) == pytest.approx(0.01)


def test_finished_mass_properties():
    fp = T.finished_props()
    a, w, d = T.PART_A, T.POCKET_W, T.POCKET_D
    rho = CUBE.mass / a**3
    assert fp["mass"] == pytest.approx(rho * (a**3 - w * w * d))
    # 質心：整塊（0）= 成品質心·m + 移除塊（a/2 − d/2）·m_cut
    m_cut = rho * w * w * d
    assert fp["com_z"] * fp["mass"] + m_cut * (a / 2 - d / 2) == pytest.approx(0.0, abs=1e-12)
    assert fp["com_z"] < 0


def test_slip_threshold_formula_and_sim():
    F = 100.0
    pred = T.slip_threshold(F)
    assert pred == pytest.approx(
        2 * T.MU_JAW * CUBE.friction * F + T.MU_FLOOR * CUBE.friction * CUBE.mass * G
    )
    r = T.clamp_hold_test(F)
    assert r["check"]["ok"]
    assert abs(r["jaw_force"]) == pytest.approx(F, rel=0.02)  # 馬達飽和 → 法向力 = F_c
    assert r["F_slip"] == pytest.approx(pred, rel=0.12)


def test_vise_datum_capture_and_monitoring():
    ok = T.vise_capture_test(0.004)  # 放置偏 +4 mm（開口間隙 6 mm 內）→ 被推回固定鉗口基準
    assert ok["ok"] and abs(ok["x_err_mm"]) < 0.1 and abs(ok["z_err_mm"]) < 0.1
    off = T.vise_capture_test(0.0, 0.003)  # y 方向不定位：誤差保留
    assert off["ok"] and off["y_err_mm"] == pytest.approx(3.0, abs=0.3)
    bad = T.vise_capture_test(0.007)  # 超出間隙 → 落在活動鉗口頂上
    assert not bad["ok"] and not bad["seat_ok"]
    chip = T.vise_capture_test(0.0, chip=True)  # 切屑墊在零件下 → 座面檢知失敗、鉗口位置窗仍 OK
    assert chip["jaw_ok"] and not chip["seat_ok"] and chip["seat_gap_mm"] > T.SEAT_TOL * 1e3


def test_cycle_model():
    m = T.cycle_model(np.array([5.0, 30.0]), 1.6, 12.0, 1.6, 15.0)
    assert m["T_c"][0] == pytest.approx(1.6 + 12.0 + 15.0)  # 手臂瓶頸
    assert m["T_c"][1] == pytest.approx(1.6 + 12.0 + 1.6 + 30.0)  # 機台瓶頸
    assert list(m["robot_bound"]) == [True, False]
    s1 = T.cycle_model(10.0, 1.6, 16.0, 1.6)
    assert float(s1["T_c"]) == pytest.approx(1.6 + 16.0 + 1.6 + 10.0)
    assert float(s1["U"]) == pytest.approx(10.0 / float(s1["T_c"]))


def test_door_entry_path_clear():
    """門外等待點 → 虎鉗上方的直線教導路徑：手臂＋夾持胚料對機台（門開）無碰撞。"""
    cc = CollisionChecker([b for b in workcell_boxes() if b.name != "cube"] + T.machine_boxes("open"))
    try:
        T_to = np.eye(4)
        T_to[2, 3] = T.PART_A / 2 - 0.02 + 0.003
        cc.attach([T.PART_A / 2] * 3, T_to)
        q = ik_near(T.P_OUT, 0.0, Q_REF, sym=False)
        for u in np.linspace(0, 1, 11):
            p = np.array(T.P_OUT) + u * (np.array(T.P_IN) - np.array(T.P_OUT))
            q = ik_near(tuple(p), 0.0, q, sym=False)
            assert q is not None and cc.distances(q)[0] > 0.02
    finally:
        cc.close()


def test_interlocks_and_curtain():
    c = T.TendCell(T.TendConfig(n_parts=1))
    try:
        m = c.m
        assert not m.cycle_start()  # 無零件、未夾緊
        m.open_door()
        assert c.wait_until(lambda: m.sig("door_open"), 5.0)
        part = c.pick_tray()
        assert part is not None
        c.to_door()
        assert not m.arm_in_zone()
        c.enter()
        assert m.arm_in_zone("inside")
        assert not m.close_door()  # 手臂在機台內 → 拒絕關門
        c.lin((T.X_PLACE, T.VY, T.Z_PART + 0.0095), "down")
        c.release()
        m.clamp(part.body)
        c.vise_part = part
        c.lin((T.X_PLACE, T.VY, T.Z_ENTRY), "up")
        assert c.wait_until(lambda: m.sig("clamped"), 3.0)
        assert m.check()["ok"]
        assert not m.cycle_start()  # 門開著 → 拒絕啟動
        c.lin((T.VX, T.FY + 0.02, T.Z_ENTRY), "curtain")
        m._door_move(0.0)  # 繞過關門互鎖：光幕須停止門
        c.s.wait(0.5)
        assert m.trip and m.door_s == pytest.approx(T.DOOR_STROKE)
        c.exit()
        assert m.close_door()
        m.request_start()
        assert c.wait_until(lambda: m.sig("cycle_busy"), 5.0)  # 門全關 → PLC 自動啟動
        assert not m.open_door()  # 循環中 → 拒絕開門
        kinds = [e["event"] for e in m.events]
        assert kinds.count("INTERLOCK_START") == 2 and "INTERLOCK_DOOR" in kinds and "CURTAIN_TRIP" in kinds
    finally:
        c.close()


def test_serial_run_with_empty_pocket():
    c = T.TendCell(T.TendConfig(strategy="serial", T_m=3.0, n_parts=3, empty_pockets=(1,)))
    try:
        r = c.run()
        assert r["all_ok"] and r["n_cycles"] == 2 and r["n_out"] == 2
        assert [f["fault"] for f in r["faults"]] == ["GRIP_EMPTY"]
        assert r["contacts_arm_machine"] == 0 and not r["errors"]
        assert r["torque_util"] < 0.5
        for ld in r["loads"]:  # 固定鉗口定位 x；y 為手臂放置精度
            assert abs(ld["err_x_mm"]) < 0.1 and abs(ld["err_y_mm"]) < 1.5 and ld["ok"]
        (pt,) = T.period_terms(r["events"], r["segments"])
        model = T.cycle_model(3.0, pt["T_open"], pt["T_ex"], pt["T_close"], pt["T_side"])
        assert pt["period"] == pytest.approx(float(model["T_c"]), abs=0.05)
        assert r["cut_slip_mm"] < 0.05  # 切削力 35 N 下零件不滑移
    finally:
        c.close()


def test_buffer_run_chip_recovery():
    c = T.TendCell(T.TendConfig(strategy="buffer", T_m=4.0, n_parts=2, chip_at=(1,)))
    try:
        r = c.run()
        assert r["all_ok"] and r["n_cycles"] == 2
        kinds = [f["fault"] for f in r["faults"]]
        assert kinds == ["CHIP_INJECTED", "NOT_SEATED"]
        assert r["faults"][1]["chips_removed"] == 1
        assert r["loads"][1]["retries"] == 1 and r["loads"][1]["ok"]
        assert r["loads"][1]["checks"][0]["seat_gap_mm"] > 1.0
    finally:
        c.close()
