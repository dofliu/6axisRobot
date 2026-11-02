"""PyBullet TORQUE_CONTROL 閉迴路模擬輔助（T004）。控制週期 = 物理步長 = dt（預設 1 ms）。"""

from __future__ import annotations

import numpy as np
import pybullet as pb

from .pbsim import PBRobot


def prepare(pbr: PBRobot, q0, dt: float = 1e-3):
    """關閉預設馬達/阻尼/限位影響，設初始狀態與步長。"""
    pbr.make_frictionless()
    pb.setTimeStep(dt, physicsClientId=pbr.cid)
    pbr.set_state(q0, np.zeros(6))


def run(pbr: PBRobot, controller, ref, T: float, dt: float = 1e-3, tau_limit=None,
        ext_force=None, ext_link: int = 6):
    """執行 T 秒。
    controller(t, q, qd) -> tau；ref(t) -> (q_ref, qd_ref) 僅供記錄誤差。
    ext_force(t) -> 世界座標外力 (3,)（作用於 ext_link 原點），None 則無。
    回傳 dict：t, q, qd, tau, q_ref, qd_ref。"""
    n = int(round(T / dt))
    rec = {k: [] for k in ("t", "q", "qd", "tau", "q_ref", "qd_ref")}
    for k in range(n):
        t = k * dt
        q, qd = pbr.get_state()
        tau = np.asarray(controller(t, q, qd), float)
        if tau_limit is not None:
            tau = np.clip(tau, -tau_limit, tau_limit)
        pbr.apply_torques(tau)
        if ext_force is not None:
            f = ext_force(t)
            if np.any(f):
                s = pb.getLinkState(pbr.id, ext_link, computeForwardKinematics=1, physicsClientId=pbr.cid)
                pb.applyExternalForce(pbr.id, ext_link, list(map(float, f)), list(s[4]),
                                      pb.WORLD_FRAME, physicsClientId=pbr.cid)
        pbr.step()
        qr, qdr = ref(t)
        for key, v in zip(rec, (t, q, qd, tau, qr, qdr), strict=True):
            rec[key].append(v)
    return {k: np.array(v) for k, v in rec.items()}
