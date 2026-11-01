import time

import numpy as np

from sixaxis.robot import G


def test_inverse_dynamics_vs_pybullet(robot, pbr, rng):
    worst = 0.0
    for _ in range(500):
        q = rng.uniform(-np.pi, np.pi, 6)
        qd = rng.uniform(-2, 2, 6)
        qdd = rng.uniform(-5, 5, 6)
        worst = max(worst, np.abs(pbr.inverse_dynamics(q, qd, qdd) - robot.rne(q, qd, qdd)).max())
    assert worst < 1e-3, worst


def test_gravity_term_vs_pybullet(robot, pbr, rng):
    z = np.zeros(6)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, 6)
        assert np.abs(pbr.inverse_dynamics(q, z, z) - robot.gravity_torque(q)).max() < 1e-4


def test_mass_matrix_properties(robot, pbr, rng):
    z = np.zeros(6)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 6)
        M = robot.mass_matrix(q)
        assert np.allclose(M, M.T, atol=1e-12)
        assert np.linalg.eigvalsh(M).min() > 1e-6
        g = pbr.inverse_dynamics(q, z, z)
        for i in range(6):
            e = np.zeros(6)
            e[i] = 1.0
            assert np.abs((pbr.inverse_dynamics(q, z, e) - g) - M[:, i]).max() < 1e-4


def test_forward_inverse_roundtrip(robot, rng):
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, 6)
        qd = rng.uniform(-2, 2, 6)
        qdd = rng.uniform(-5, 5, 6)
        tau = robot.rne(q, qd, qdd)
        assert np.abs(robot.forward_dynamics(q, qd, tau) - qdd).max() < 1e-8


def test_external_wrench_static(robot, rng):
    q = rng.uniform(-1, 1, 6)
    w = np.array([5.0, -3.0, 10.0, 0.5, 0.2, -0.1])
    tau = robot.gravity_torque(q) - robot.jacobian(q).T @ w
    qdd = robot.forward_dynamics(q, np.zeros(6), tau, wrench_ext=w)
    assert np.abs(qdd).max() < 1e-9


def test_energy_conservation_free_swing(robot):
    q0 = np.array([0.3, -0.4, 0.8, -0.5, 0.6, 0.2])
    qd0 = np.array([0.5, -0.3, 0.4, 1.0, -0.8, 0.6])
    t, q, qd = robot.simulate(q0, qd0, 2e-3, 500)  # 1 s
    E = np.array([sum(robot.energy(a, b)) for a, b in zip(q[::10], qd[::10])])
    ke0, _ = robot.energy(q0, qd0)
    assert ke0 > 0.1
    rel = (E.max() - E.min()) / abs(E[0])
    assert rel < 1e-5, rel
    assert np.ptp(q, axis=0).max() > 0.3  # 確實有擺動


def test_free_swing_vs_pybullet_short(robot, pbr):
    q0 = np.array([0.3, -0.4, 0.8, -0.5, 0.6, 0.2])
    qd0 = np.array([0.5, -0.3, 0.4, 1.0, -0.8, 0.6])
    t, q, _ = robot.simulate(q0, qd0, 1e-3, 500)  # 0.5 s
    _, qp, _ = pbr.free_swing(q0, qd0, 1e-4, 5000, sample_every=10)
    assert np.abs(q - qp).max() < 2e-3


def test_gravity_compensation_holds_in_pybullet(robot, pbr):
    q0 = np.array([0.2, -1.0, 1.2, -0.8, 0.9, 0.1])
    pbr.make_frictionless()
    import pybullet as pb

    pb.setTimeStep(1e-3, physicsClientId=pbr.cid)
    pbr.set_state(q0)
    for _ in range(500):
        q, _ = pbr.get_state()
        pbr.apply_torques(robot.gravity_torque(q))
        pbr.step()
    q, qd = pbr.get_state()
    assert np.abs(q - q0).max() < 2e-3, np.abs(q - q0).max()


def test_dynamics_performance(robot):
    q = np.full(6, 0.3)
    t = time.perf_counter()
    for _ in range(100):
        robot.rne(q, q, q)
    rne_us = (time.perf_counter() - t) / 100 * 1e6
    assert rne_us < 3000
    assert G == 9.81
