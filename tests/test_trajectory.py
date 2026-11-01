"""T003 軌跡規劃庫測試。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sixaxis import SixAxisRobot
from sixaxis.robot import rot_log
from sixaxis.trajectory import (
    ArcPath,
    LinePath,
    Quintic,
    Trapezoid,
    check_samples,
    concat,
    fit_duration,
    joint_ptp,
    min_duration,
    sample_cartesian,
    sample_joint,
)

Q0 = np.array([0.3, -1.1, 1.3, -0.7, 0.9, 0.2])


@pytest.fixture(scope="module")
def robot():
    return SixAxisRobot()


def _pose_target(robot, dp, rotvec=(0, 0, 0)):
    T0 = robot.fk(Q0)
    T1 = T0.copy()
    T1[:3, 3] += dp
    T1[:3, :3] = T0[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
    return T0, T1


@pytest.mark.parametrize("prof", [Quintic(2.0), Trapezoid(2.0, 0.25), Trapezoid(1.0, 0.1)])
def test_profile_boundaries_monotone_and_derivatives(prof):
    s0, v0, _ = prof(0.0)
    s1, v1, _ = prof(prof.T)
    assert (s0, v0) == (0.0, 0.0) and abs(s1 - 1) < 1e-12 and abs(v1) < 1e-12
    t = np.linspace(0, prof.T, 4001)
    s, sd, sdd = prof(t)
    assert np.all(np.diff(s) >= -1e-12)
    h = t[1] - t[0]
    assert np.max(np.abs(np.gradient(s, h)[2:-2] - sd[2:-2])) < 2e-3 * sd.max()
    assert np.isclose(sd.max(), prof.peak_vel_factor / prof.T, rtol=1e-3)
    assert np.isclose(np.abs(sdd).max(), prof.peak_acc_factor / prof.T**2, rtol=1e-3)


def test_quintic_zero_boundary_acceleration():
    _, _, a = Quintic(1.0)(np.array([0.0, 1.0]))
    assert np.allclose(a, 0, atol=1e-12)


def test_joint_ptp_endpoints_and_sync(robot):
    q1 = Q0 + np.array([0.5, -0.3, 0.2, 0.8, -0.4, 0.1])
    tr = joint_ptp(Q0, q1, 1.5)
    q_a, qd_a, _ = tr.state(0.0)
    q_b, qd_b, _ = tr.state(1.5)
    assert np.allclose(q_a, Q0, atol=1e-14) and np.allclose(q_b, q1, atol=1e-12)
    assert np.allclose(qd_a, 0) and np.allclose(qd_b, 0, atol=1e-12)
    smp = sample_joint(robot, tr, 0.01)
    frac = (smp.q - Q0) / (q1 - Q0)  # 各軸歸一化進度相同 = 同步
    assert np.ptp(frac, axis=1).max() < 1e-12


def test_min_duration_makes_velocity_exactly_limit(robot):
    q1 = Q0 + np.array([1.0, -0.5, 0.2, 0.8, -0.4, 0.1])
    for kind in ("quintic", "trapezoid"):
        tr = joint_ptp(Q0, q1, None, kind, robot=robot)
        smp = sample_joint(robot, tr, 0.002)
        assert np.max(np.abs(smp.qd) / robot.qd_max) == pytest.approx(1.0, rel=2e-3)
    assert min_duration(q1 - Q0, robot.qd_max, qdd_max=np.full(6, 0.5)) > min_duration(
        q1 - Q0, robot.qd_max
    )


def test_line_is_straight_and_slerp_constant_rate(robot):
    T0, T1 = _pose_target(robot, [0.1, 0.2, -0.05], [0.0, 0.4, 0.6])
    path = LinePath(T0, T1)
    chord = T1[:3, 3] - T0[:3, 3]
    ss = np.linspace(0, 1, 101)
    P = np.array([path.pose(s)[:3, 3] for s in ss])
    dev = np.linalg.norm(np.cross(P - T0[:3, 3], chord / np.linalg.norm(chord)), axis=1)
    assert dev.max() < 1e-12
    angs = [np.linalg.norm(rot_log(T0[:3, :3].T @ path.pose(s)[:3, :3])) for s in ss]
    assert np.allclose(angs, ss * path.angle, atol=1e-10)  # 角度對 s 線性
    assert np.allclose(path.pose(1.0), T1, atol=1e-12) and np.allclose(path.pose(0.0), T0)


def test_arc_geometry(robot):
    T0 = robot.fk(Q0)
    A = T0[:3, 3]
    B, C = A + [0.0, 0.1, 0.1], A + [0.0, 0.2, 0.0]
    TC = T0.copy()
    TC[:3, 3] = C
    arc = ArcPath(T0, B, TC)
    ss = np.linspace(0, 1, 201)
    P = np.array([arc.pose(s)[:3, 3] for s in ss])
    assert np.abs(np.linalg.norm(P - arc.center, axis=1) - arc.radius).max() < 1e-12
    assert np.allclose(P[0], A) and np.allclose(P[-1], C, atol=1e-12)
    assert np.allclose(arc.pose(arc.theta_B / arc.theta)[:3, 3], B, atol=1e-12)  # 通過中間點
    # 切線連續、速率恆定（弧長參數化）
    d = np.array([arc.derivs(s)[0] for s in ss])
    assert np.allclose(np.linalg.norm(d, axis=1), arc.length)
    with pytest.raises(ValueError):
        ArcPath(T0, A + [0, 0.1, 0], _line_T(A, [0, 0.2, 0]))


def _line_T(A, d):
    T = np.eye(4)
    T[:3, 3] = A + np.array(d)
    return T


def test_cartesian_tracking_and_joint_consistency(robot):
    T0, T1 = _pose_target(robot, [0.05, 0.2, 0.03], [0.0, 0.0, 0.7])
    path = LinePath(T0, T1)
    smp = sample_cartesian(robot, path, Quintic(2.0), Q0, dt=0.01)
    assert smp.ik_fail == 0
    assert np.abs(smp.pos - smp.target_pos).max() < 1e-8
    final = robot.fk(smp.q[-1])
    assert np.linalg.norm(final[:3, 3] - T1[:3, 3]) < 1e-8
    assert np.linalg.norm(rot_log(T1[:3, :3] @ final[:3, :3].T)) < 1e-8
    assert np.abs(np.diff(smp.q, axis=0)).max() < 0.05  # 關節連續，無分支跳躍
    # 解析 J⁺·twist 速度與位置差分一致
    fd = np.gradient(smp.q, smp.t, axis=0)
    assert np.abs(fd - smp.qd)[3:-3].max() < 5e-3
    # 末端線速度峰值 = 1.875 L / T
    v = np.linalg.norm(np.gradient(smp.pos, smp.t, axis=0), axis=1)
    assert v.max() == pytest.approx(Quintic.peak_vel_factor * path.length / 2.0, rel=5e-3)


def test_arc_cartesian_samples_on_circle(robot):
    T0 = robot.fk(Q0)
    A = T0[:3, 3]
    TC = T0.copy()
    TC[:3, 3] = A + [0.0, 0.2, 0.0]
    arc = ArcPath(T0, A + [0.0, 0.1, 0.1], TC)
    smp = sample_cartesian(robot, arc, Trapezoid(2.0), Q0, dt=0.02)
    assert smp.ik_fail == 0
    r = np.linalg.norm(smp.pos - arc.center, axis=1)
    assert np.abs(r - arc.radius).max() < 1e-8


def test_check_and_fit_duration_time_scaling(robot):
    q1 = Q0 + np.array([2.0, -1.5, 1.0, 1.5, -1.0, 1.0])
    build = lambda T: sample_joint(robot, joint_ptp(Q0, q1, T), 0.01)  # noqa: E731
    fast = build(0.4)
    c = check_samples(robot, fast)
    assert not c["ok"] and c["vel_ratio"] > 1  # 太快：偵測到超限
    T, smp, hist = fit_duration(robot, build, 0.4)
    c2 = check_samples(robot, smp)
    assert T > 0.4 and c2["ok"] and smp.meta["feasible"]
    assert c2["vel_ratio"] <= 0.95 + 1e-9 and c2["torque_ratio"] <= 0.95 + 1e-9
    assert len(hist) >= 2


def test_concat_continuity(robot):
    q1, q2 = Q0 + 0.3, Q0 - 0.2
    a = sample_joint(robot, joint_ptp(Q0, q1, 1.0), 0.01)
    b = sample_joint(robot, joint_ptp(q1, q2, 1.0), 0.01)
    c = concat([a, b])
    assert len(c) == len(a) + len(b) - 1
    assert np.abs(np.diff(c.q, axis=0)).max() < 0.05
    assert np.all(np.diff(c.t) > 0)
