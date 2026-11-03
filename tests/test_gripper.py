"""T005 夾爪與物件測試：抓取力學、抬升不滑、釋放、摩擦下限與解析式比對。"""

import pybullet as pb
import pytest

from sixaxis.graspsim import GraspScene, run_pick_lift, static_slip
from sixaxis.gripper import BLOCK, CUBE, CYLINDER, F_MAX, S_MAX, TCP_Z, f_min
from sixaxis.robot import G


def test_f_min_formula():
    assert f_min(0.2, 0.8) == pytest.approx(0.2 * G / 1.6, rel=1e-6)
    assert f_min(0.2, 0.8, accel=3.0) > f_min(0.2, 0.8)


def test_gripper_model_geometry():
    s = GraspScene(CUBE)
    try:
        g = s.g
        info = [pb.getJointInfo(s.sim.id, j, physicsClientId=s.cid) for j in g.fingers]
        assert all(i[2] == pb.JOINT_PRISMATIC for i in info)
        assert all(i[8] == 0.0 and i[9] == pytest.approx(S_MAX) for i in info)
        assert all(i[10] == pytest.approx(F_MAX) for i in info)
        # TCP 位於指墊中心：左右指墊 AABB 中心 z 與 TCP z 相差 < 1 mm
        for j in g.fingers:
            lo, hi = pb.getAABB(s.sim.id, j, physicsClientId=s.cid)
            assert abs((lo[2] + hi[2]) / 2 - g.tcp_pos()[2]) < 1e-3
        assert TCP_Z == pytest.approx(0.095)
    finally:
        s.close()


@pytest.fixture(scope="module")
def cube_run():
    s, m = run_pick_lift(CUBE, force=40.0, lift=0.15, T_lift=1.0)
    s.close()
    return m


def test_cube_grasp_force_and_symmetry(cube_run):
    m = cube_run
    assert m["fL_N"] == pytest.approx(40.0, abs=1.0) and m["fR_N"] == pytest.approx(40.0, abs=1.0)
    assert abs(m["finger_stroke_mm"][0] - m["finger_stroke_mm"][1]) < 0.1  # 兩指連動
    assert m["finger_stroke_mm"][0] == pytest.approx(15.0, abs=0.2)  # (80-50)/2 mm


def test_cube_lift_no_slip(cube_run):
    m = cube_run
    assert m["held"] and m["slip_lift_mm"] < 0.1
    assert m["z_gain_mm"] == pytest.approx(150.0, abs=2.0)


def test_cube_release_stable(cube_run):
    assert cube_run["release_drift_mm"] < 1.0 and cube_run["tilt_deg"] < 2.0


@pytest.mark.parametrize("spec", [BLOCK, CYLINDER])
def test_other_objects_lift(spec):
    s, m = run_pick_lift(spec, force=40.0, lift=0.10, T_lift=1.0)
    s.close()
    assert m["held"] and m["slip_lift_mm"] < 0.5
    assert m["z_gain_mm"] == pytest.approx(100.0, abs=2.0)


def test_high_acceleration_lift_held():
    s, m = run_pick_lift(CUBE, force=40.0, lift=0.15, T_lift=0.5)
    s.close()
    assert m["a_peak_ms2"] > 3.0 and m["held"] and m["slip_lift_mm"] < 0.5


def test_friction_threshold_matches_analytic():
    fmin = f_min(CUBE.mass, CUBE.friction)  # 1.226 N
    assert static_slip(CUBE, 0.8 * fmin) > 5e-3  # 低於下限 → 明顯下滑
    assert static_slip(CUBE, 1.5 * fmin) < 1e-4  # 高於下限 → 不滑
