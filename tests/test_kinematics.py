import time

import numpy as np

from sixaxis.robot import rot_log


def rand_q(rng, n=None):
    return rng.uniform(-np.pi, np.pi, 6 if n is None else (n, 6))


def test_dh_fk_equals_urdf_chain(robot, rng):
    for q in rand_q(rng, 200):
        assert np.abs(robot.fk(q) - robot.link_frames(q)[6]).max() < 1e-12


def test_fk_vs_pybullet_1000(robot, pbr, rng):
    pos_err, rot_err = [], []
    for q in rand_q(rng, 1000):
        pbr.set_state(q)
        Tp, Tm = pbr.link_pose(6), robot.fk(q)
        pos_err.append(np.linalg.norm(Tp[:3, 3] - Tm[:3, 3]))
        rot_err.append(np.linalg.norm(rot_log(Tp[:3, :3] @ Tm[:3, :3].T)))
    assert max(pos_err) < 1e-6, max(pos_err)
    assert max(rot_err) < 1e-5, max(rot_err)


def test_link_frames_vs_pybullet(robot, pbr, rng):
    for q in rand_q(rng, 100):
        pbr.set_state(q)
        frames = robot.link_frames(q)
        for i in range(7):
            assert np.linalg.norm(pbr.link_pose(i)[:3, 3] - frames[i][:3, 3]) < 1e-6


def test_ik_roundtrip_1000(robot, rng):
    n_sol = []
    for q in rand_q(rng, 1000):
        T = robot.fk(q)
        sols = robot.ik_all(T, q)
        n_sol.append(len(sols))
        assert sols, "至少應有解"
        for s in sols:
            Te = robot.fk(s)
            assert np.linalg.norm(Te[:3, 3] - T[:3, 3]) < 1e-9
            assert np.linalg.norm(Te[:3, :3] - T[:3, :3]) < 1e-9
        assert any(
            np.allclose(np.cos(s), np.cos(q), atol=1e-6) and np.allclose(np.sin(s), np.sin(q), atol=1e-6)
            for s in sols
        ), "原始關節解必須在解集合中"
        best = robot.ik(T, q)[0]
        assert np.abs(best - q).max() < 1e-6 or np.abs(np.sin(best - q)).max() < 1e-6
    assert max(n_sol) == 8


def test_ik_selects_nearest(robot):
    q = np.array([0.4, -1.2, 1.5, -0.7, 1.1, 0.3])
    sols = robot.ik(robot.fk(q), q + 0.05)
    d = [np.linalg.norm(s - (q + 0.05)) for s in sols]
    assert d == sorted(d)
    assert np.allclose(sols[0], q, atol=1e-8)


def test_ik_unreachable(robot):
    T = np.eye(4)
    T[:3, 3] = [3.0, 0.0, 0.5]
    assert robot.ik(T) == []


def test_ik_numeric_converges(robot, rng):
    """非奇異目標（σ_min>0.02）、初值偏離 0.2 rad 內，LM 數值 IK 應全數收斂。"""
    tried = 0
    for q in rand_q(rng, 80):
        if robot.sigma_min(q) < 0.02:
            continue
        tried += 1
        q0 = q + rng.normal(0, 0.2, 6)
        T = robot.fk(q)
        qs, conv, _ = robot.ik_numeric(T, q0)
        Te = robot.fk(qs)
        assert conv and np.linalg.norm(Te[:3, 3] - T[:3, 3]) < 1e-8
    assert tried >= 50


def test_jacobian_vs_numeric(robot, rng):
    h = 1e-6
    worst = 0.0
    for q in rand_q(rng, 200):
        J = robot.jacobian(q)
        Jn = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = h
            Tp, Tm = robot.fk(q + dq), robot.fk(q - dq)
            Jn[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
            Jn[3:, i] = rot_log(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
        worst = max(worst, np.abs(J - Jn).max())
    assert worst < 1e-6, worst


def test_singularity_detection(robot):
    generic = np.array([0.3, -1.1, 1.4, -0.6, 1.0, 0.2])
    assert robot.singularities(generic) == []
    assert robot.sigma_min(generic) > 0.05
    wrist = generic.copy()
    wrist[4] = 0.0
    assert "wrist" in robot.singularities(wrist)
    assert robot.sigma_min(wrist) < 1e-6
    elbow = generic.copy()
    elbow[2] = 0.0
    assert "elbow" in robot.singularities(elbow)
    assert robot.sigma_min(elbow) < 1e-6
    # 肩奇異：腕心位於軸 1 上方（在臂平面內離軸距離 0）
    sh = np.array([0.0, -np.pi / 2, 0.0, 0.0, 1.0, 0.0])
    q = np.array([0.0, -1.0, 2.2, -1.2, 1.0, 0.0])
    found = None
    for th2 in np.linspace(-3, 3, 6001):
        q[1] = th2
        if "shoulder" in robot.singularities(q, tol=1e-2):
            found = q.copy()
            break
    assert found is not None
    assert robot.sigma_min(found) < 5e-3
    del sh


def test_ik_at_wrist_singularity(robot):
    q = np.array([0.3, -1.1, 1.4, -0.6, 0.0, 0.2])
    T = robot.fk(q)
    sols = robot.ik(T, q)
    assert sols
    for s in sols:
        assert np.linalg.norm(robot.fk(s)[:3, 3] - T[:3, 3]) < 1e-8
        assert np.linalg.norm(robot.fk(s)[:3, :3] - T[:3, :3]) < 1e-8


def test_limits(robot):
    q = np.zeros(6)
    assert robot.check_limits(q, np.zeros(6), np.zeros(6)) == {}
    bad = robot.check_limits(
        np.array([7, 0, 0, 0, 0, 0.0]), np.array([0, 4, 0, 0, 0, 0.0]), np.array([0, 0, 0, 0, 0, 30.0])
    )
    assert bad == {"joint_pos": [0], "joint_vel": [1], "torque": [5]}
    # IK 套用限位：縮小 j1 範圍後，解需都在範圍內
    robot2 = type(robot)(type(robot.p)(q_min=(-1.0,) + robot.p.q_min[1:], q_max=(1.0,) + robot.p.q_max[1:]))
    qq = np.array([0.5, -1.0, 1.0, -0.5, 1.0, 0.0])
    for s in robot2.ik(robot2.fk(qq), qq):
        assert -1.0 <= s[0] <= 1.0


def test_performance(robot, rng):
    qs = rand_q(rng, 200)
    t = time.perf_counter()
    for q in qs:
        robot.fk(q)
    fk_us = (time.perf_counter() - t) / len(qs) * 1e6
    Ts = [robot.fk(q) for q in qs]
    t = time.perf_counter()
    for T in Ts:
        robot.ik(T)
    ik_ms = (time.perf_counter() - t) / len(qs) * 1e3
    assert fk_us < 500 and ik_ms < 20, (fk_us, ik_ms)
