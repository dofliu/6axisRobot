"""T007 指令介面測試：解析/驗證、錯誤回報、運動學 dry-run、物理執行（取放、直線度、落空夾持、碰撞）。"""

import json

import numpy as np
import pybullet as pb
import pytest

from sixaxis import scene as S
from sixaxis.commands import CommandError, Executor, parse
from sixaxis.graspsim import GraspScene
from sixaxis.gripper import CUBE

PCK = S.pocket_centers()[3]
Z_PLACE = S.BENCH_TOP + S.TRAY_PLATE_T + CUBE.size[2] / 2
PROGRAM_DSL = f"""
# 取放：桌面 (0.45,0) → 料盤格 3
move_j tcp=0.45,0,0.30
move_j tcp=0.45,0,0.22
move_l tcp=0.45,0,0.125
grip force=40
move_l tcp=0.45,0,0.25
move_c via=0.52,-0.0625,0.25 tcp={PCK[0]},{PCK[1]},0.25
move_l tcp={PCK[0]},{PCK[1]},{Z_PLACE}
release
move_l tcp={PCK[0]},{PCK[1]},0.25
wait T=0.2
"""


@pytest.fixture
def sc():
    s = GraspScene(CUBE, workcell=True)
    yield s
    s.close()


def code(program, scene=None):
    rep = Executor(scene).run(program)
    assert not rep.ok
    return rep.error.code, rep


# ------------------------------------------------------------------ 解析
def test_dsl_and_json_equivalent():
    dsl = "move_j tcp=0.4,0,0.3\ngrip force=30\nwait T=0.5  # c\n"
    js = json.dumps([{"cmd": "move_j", "tcp": [0.4, 0, 0.3]}, {"cmd": "grip", "force": 30},
                     {"cmd": "wait", "T": 0.5}])
    a, b = parse(dsl), parse(js)
    for x in (*a, *b):
        x.pop("id")
    assert a == b
    assert [c["cmd"] for c in parse({"commands": json.loads(js)})] == ["move_j", "grip", "wait"]
    assert [c["cmd"] for c in a] == ["move_j", "grip", "wait"] and a[1]["force"] == 30.0


def test_defaults_and_deg():
    c = parse([{"cmd": "grip"}, {"cmd": "move_j", "q_deg": [0, -90, 0, 0, 0, 0]}])
    assert c[0]["force"] == 40.0 and abs(c[1]["q"][1] + np.pi / 2) < 1e-12


@pytest.mark.parametrize("prog,expect", [
    ([{"cmd": "fly"}], "SYNTAX"),
    ([{"cmd": "move_l"}], "SYNTAX"),  # 缺 tcp
    ([{"cmd": "move_l", "tcp": [0, 0]}], "SYNTAX"),
    ([{"cmd": "move_l", "tcp": [0.4, 0, 0.3], "foo": 1}], "SYNTAX"),
    ([{"cmd": "move_j"}], "SYNTAX"),
    ([{"cmd": "move_j", "q": [0] * 6, "tcp": [0.4, 0, 0.3]}], "SYNTAX"),
    ([{"cmd": "wait", "T": "x"}], "SYNTAX"),
    ([{"cmd": "wait", "T": -1}], "RANGE"),
    ([{"cmd": "grip", "force": 1e4}], "RANGE"),
    ([{"cmd": "set_speed", "speed": 0}], "RANGE"),
    ([{"cmd": "move_l", "tcp": [0.4, 0, 0.3], "profile": "sine"}], "RANGE"),
    ("[{bad json", "SYNTAX"),
    ("move_l tcp", "SYNTAX"),
    (42, "SYNTAX"),
])
def test_validation_errors(prog, expect):
    with pytest.raises(CommandError) as e:
        parse(prog)
    assert e.value.code == expect


def test_error_reports_index():
    rep = Executor().run([{"cmd": "wait", "T": 0.1}, {"cmd": "wait", "T": -2}])
    assert rep.error.index == 1 and not rep.results  # 先整體驗證，不執行任何指令


# ------------------------------------------------------------------ dry-run（運動學）
def test_dry_run_program_ok_and_reaches_targets():
    rep = Executor().run(PROGRAM_DSL)
    assert rep.ok and len(rep.results) == 10
    for r in rep.results:
        if r["cmd"].startswith("move"):
            assert r["target_err_mm"] < 1e-6
            assert r["vel_ratio"] <= 0.5 + 1e-9 and r["torque_ratio"] <= 0.5 + 1e-9
    assert rep.t_total == pytest.approx(sum(r["t_end"] - r["t_start"] for r in rep.results), rel=1e-9)


def test_unreachable_and_collinear_and_limit():
    c, rep = code("move_j tcp=2,0,0.3")
    assert c == "UNREACHABLE" and rep.error.index == 0
    c, _ = code("move_j tcp=0.45,0,0.3\nmove_c via=0.45,0,0.35 tcp=0.45,0,0.4")
    assert c == "RANGE"
    c, _ = code("move_j tcp=0.45,0,0.3\nmove_l tcp=0.45,0.3,0.3 T=0.1")  # 0.3 m 在 0.1 s → 超速
    assert c == "LIMIT"
    c, _ = code([{"cmd": "move_j", "q": [10, 0, 0, 0, 0, 0]}])  # 超出 ±2π
    assert c == "LIMIT"


def test_speed_scales_duration():
    base = "move_j tcp=0.45,0,0.3\nmove_l tcp=0.45,0.2,0.3"
    d1 = Executor().run(base).results[1]["duration"]
    d2 = Executor().run("set_speed speed=0.5\n" + base).results[-1]["duration"]
    assert d2 > 1.5 * d1


def test_explicit_T_respected():
    rep = Executor().run("move_j tcp=0.45,0,0.3\nmove_l tcp=0.45,0.05,0.3 T=2.0")
    assert rep.results[1]["duration"] == 2.0


# ------------------------------------------------------------------ 物理執行
def test_pick_and_place_in_sim(sc):
    ex = Executor(sc)
    rep = ex.run(PROGRAM_DSL)
    assert rep.ok, rep.error
    p = np.array(pb.getBasePositionAndOrientation(sc.obj, physicsClientId=sc.cid)[0])
    assert np.linalg.norm(p[:2] - np.array(PCK)) < 2e-3 and abs(p[2] - Z_PLACE) < 1.5e-3
    for r in rep.results:
        if r["cmd"].startswith("move"):
            assert r["tcp_err_mm"] < 2.0, r
    grip = next(r for r in rep.results if r["cmd"] == "grip")
    assert grip["holding"] and 30 < grip["fL_N"] < 50
    assert not ex.holding and rep.t_total == pytest.approx(sc.t, rel=1e-6)


def test_move_l_straightness_in_sim(sc):
    ex = Executor(sc)
    assert ex.run("move_j tcp=0.45,0,0.30\nmove_l tcp=0.45,0.0,0.30").ok
    n0 = len(sc.rec["tcp"])
    a, b = np.array([0.45, 0.0, 0.30]), np.array([0.45, 0.20, 0.30])
    assert ex.run("move_l tcp=0.45,0.2,0.3").ok
    P = np.array(sc.rec["tcp"][n0:])
    d = b - a
    dev = np.linalg.norm(np.cross(P - a, d / np.linalg.norm(d)), axis=1)
    assert dev.max() < 1.5e-3


def test_grip_empty_detected(sc):
    rep = Executor(sc).run("move_j tcp=0.45,0.1,0.30\ngrip force=30")
    assert rep.error and rep.error.code == "GRIP_EMPTY" and rep.error.index == 1
    assert rep.results[-1]["status"] == "error"


def test_collision_detected(sc):
    rep = Executor(sc).run("move_j tcp=0.45,0,0.3\nmove_l tcp=0.45,0,0.07")  # 指墊壓進工作台
    assert rep.error and rep.error.code == "COLLISION"


def test_continue_on_error():
    rep = Executor().run("move_j tcp=2,0,0.3\nwait T=0.1", stop_on_error=False)
    assert [r["status"] for r in rep.results] == ["error", "ok"] and rep.error.code == "UNREACHABLE"
