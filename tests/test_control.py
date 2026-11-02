"""T004 控制器測試：PyBullet TORQUE_CONTROL 閉迴路（dt = 1 ms）。"""

import numpy as np
import pytest

from sixaxis import SixAxisRobot
from sixaxis.control import (
    Admittance,
    CartesianImpedance,
    CartesianPositionTracker,
    ComputedTorque,
    PDGravity,
    critical_gains,
)
from sixaxis.ctrlsim import prepare, run
from sixaxis.pbsim import PBRobot
from sixaxis.trajectory import joint_ptp

Q0 = np.array([0.3, -1.1, 1.3, -0.7, 0.9, 0.2])
Q1 = Q0 + np.array([0.3, 0.2, -0.2, 0.3, -0.3, 0.4])
F_EXT = np.array([0.0, 0.0, -5.0])


@pytest.fixture(scope="module")
def r():
    return SixAxisRobot()


@pytest.fixture(scope="module")
def sim(r):
    p = PBRobot(r)
    yield p
    p.close()


def pd_gains(r, q, wn=30.0):
    """Kp = wn²·M_ii, Kd = 2·wn·M_ii（依各軸等效慣量縮放，避免小慣量腕軸離散不穩定）。"""
    m = np.diag(r.mass_matrix(q))
    return m * wn**2, 2 * wn * m


def test_critical_gains():
    kp, kd = critical_gains(10.0)
    assert np.allclose(kp, 100) and np.allclose(kd, 20)


def test_pd_gravity_setpoint(r, sim):
    prepare(sim, Q0)
    kp, kd = pd_gains(r, Q0)
    pd = PDGravity(r, kp, kd)
    rec = run(sim, lambda t, q, qd: pd(q, qd, Q1), lambda t: (Q1, np.zeros(6)), 3.0)
    assert np.abs(rec["q"][-1] - Q1).max() < 1e-4
    assert np.abs(rec["qd"][-1]).max() < 1e-4


def _track(r, sim, ctrl, T=1.5, model_ctrl=None):
    traj = joint_ptp(Q0, Q1, T, "quintic")
    prepare(sim, Q0)

    def ref(t):
        q, qd, _ = traj.state(min(t, T))
        return q, qd

    def c(t, q, qd):
        qr, qdr = ref(t)
        _, _, qddr = traj.state(min(t, T))
        return ctrl(q, qd, qr, qdr, qddr)

    return run(sim, c, ref, T, tau_limit=None)


def _rms(rec):
    return float(np.sqrt(np.mean((rec["q"] - rec["q_ref"]) ** 2)))


def test_computed_torque_beats_pd_tracking(r, sim):
    ct = ComputedTorque(r, *critical_gains(40.0))
    kp, kd = pd_gains(r, Q0)
    pd = PDGravity(r, kp, kd)
    rec_ct = _track(r, sim, ct)
    e_ct = np.abs(rec_ct["q"] - rec_ct["q_ref"]).max()
    rec_pd = _track(r, sim, pd)
    e_pd = np.abs(rec_pd["q"] - rec_pd["q_ref"]).max()
    assert e_ct < 1e-4
    assert e_pd > 10 * e_ct


def test_computed_torque_model_mismatch_degrades_but_stable(r, sim):
    from sixaxis.robot import RobotParams

    p = RobotParams(mass=tuple(1.2 * m for m in RobotParams().mass))
    wrong = SixAxisRobot(p)
    exact = _rms(_track(r, sim, ComputedTorque(r, *critical_gains(40.0))))
    bad = _rms(_track(r, sim, ComputedTorque(r, *critical_gains(40.0), model=wrong)))
    assert exact < bad < 0.05  # 質量誤差 20%：誤差變大但有界


def test_cartesian_impedance_stiffness(r, sim):
    prepare(sim, Q0)
    T0 = r.fk(Q0)
    k = 2000.0
    imp = CartesianImpedance(r, [k] * 3 + [30.0] * 3, [80.0] * 3 + [0.3] * 3)
    f = np.array([0.0, 0.0, -20.0])
    rec = run(sim, lambda t, q, qd: imp(q, qd, T0), lambda t: (Q0, np.zeros(6)), 2.0,
              ext_force=lambda t: f if t > 0.2 else np.zeros(3))
    dz = r.fk(rec["q"][-1])[:3, 3] - T0[:3, 3]
    assert dz[2] == pytest.approx(f[2] / k, rel=0.05)
    assert np.abs(dz[:2]).max() < 1e-4


def test_admittance_steady_state(r, sim):
    prepare(sim, Q0)
    T0 = r.fk(Q0)
    ka = 200.0
    adm = Admittance(2.0, 30.0, ka)
    tr = CartesianPositionTracker(r, ComputedTorque(r, *critical_gains(60.0)), Q0)

    def ext(t):
        return F_EXT if t > 0.2 else np.zeros(3)

    def c(t, q, qd):
        x, v = adm.step(ext(t), 1e-3)
        T = T0.copy()
        T[:3, 3] += x
        return tr.step(q, qd, T, v, 1e-3)

    rec = run(sim, c, lambda t: (Q0, np.zeros(6)), 3.0, ext_force=ext)
    dz = r.fk(rec["q"][-1])[:3, 3] - T0[:3, 3]
    assert dz[2] == pytest.approx(F_EXT[2] / ka, rel=0.15)
