"""關節/笛卡兒控制器（T004）。皆輸出關節力矩 τ，供 PyBullet TORQUE_CONTROL 使用。

公式依據（Spong, Hutchinson, Vidyasagar《Robot Modeling and Control》；Siciliano《Robotics》第 8 章）：
* PD + 重力補償：τ = g(q) + Kp(q_d - q) + Kd(qd_d - q̇)          （全域漸近穩定，Takegaki & Arimoto 1981）
* 計算力矩（逆動力學）：τ = M(q)(q̈_d + Kd ė + Kp e) + C(q,q̇)q̇ + g(q)   → 誤差動態 ë + Kd ė + Kp e = 0
* 笛卡兒阻抗：F = K(x_d - x) + D(ẋ_d - ẋ)，τ = Jᵀ F + g(q)  （Hogan 1985；平移 + 姿態誤差用旋轉向量）
* 導納：M_a ë_x + D_a ė_x + K_a e_x = F_ext，把外力轉成參考位置偏移，再交給內環位置控制（計算力矩）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot import SixAxisRobot, rot_log


def critical_gains(wn: float | np.ndarray, zeta: float = 1.0):
    """由自然頻率與阻尼比得 Kp = wn², Kd = 2 ζ wn（逐軸）。"""
    wn = np.broadcast_to(np.asarray(wn, float), (6,)).copy()
    return wn**2, 2.0 * zeta * wn


class PDGravity:
    """PD + 重力補償（只需 g(q)，不需完整慣量）。Kp、Kd 為 N·m/rad、N·m·s/rad。"""

    def __init__(self, robot: SixAxisRobot, kp, kd):
        self.r = robot
        self.kp = np.broadcast_to(np.asarray(kp, float), (6,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, float), (6,)).copy()

    def __call__(self, q, qd, q_d, qd_d=None, qdd_d=None):
        qd_d = np.zeros(6) if qd_d is None else qd_d
        return self.r.gravity_torque(q) + self.kp * (q_d - q) + self.kd * (qd_d - qd)


class ComputedTorque:
    """計算力矩控制。kp、kd 為 1/s² 與 1/s 單位的誤差增益（見 critical_gains）。
    `model` 可傳入參數有偏差的模型來測試對模型不確定性的魯棒性。"""

    def __init__(self, robot: SixAxisRobot, kp, kd, model: SixAxisRobot | None = None):
        self.r = model or robot
        self.kp = np.broadcast_to(np.asarray(kp, float), (6,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, float), (6,)).copy()

    def __call__(self, q, qd, q_d, qd_d=None, qdd_d=None):
        qd_d = np.zeros(6) if qd_d is None else qd_d
        qdd_d = np.zeros(6) if qdd_d is None else qdd_d
        a = qdd_d + self.kd * (qd_d - qd) + self.kp * (q_d - q)
        return self.r.mass_matrix(q) @ a + self.r.bias_torque(q, qd)


def pose_error(T_d: np.ndarray, T: np.ndarray) -> np.ndarray:
    """6 維誤差 [Δp; Δφ]（世界座標；Δφ = log(R_d Rᵀ)）。"""
    return np.concatenate([T_d[:3, 3] - T[:3, 3], rot_log(T_d[:3, :3] @ T[:3, :3].T)])


class CartesianImpedance:
    """笛卡兒阻抗控制 τ = Jᵀ(K e + D(ẋ_d − J q̇)) + g(q)。K、D 為 6 維（N/m、N·m/rad；N·s/m、N·m·s/rad）。
    comp_coriolis=True 時再加上 C(q,q̇)q̇ 補償。"""

    def __init__(self, robot: SixAxisRobot, k, d, comp_coriolis: bool = True):
        self.r = robot
        self.k = np.broadcast_to(np.asarray(k, float), (6,)).copy()
        self.d = np.broadcast_to(np.asarray(d, float), (6,)).copy()
        self.cc = comp_coriolis

    def __call__(self, q, qd, T_d, v_d=None):
        v_d = np.zeros(6) if v_d is None else v_d
        J = self.r.jacobian(q)
        e = pose_error(T_d, self.r.fk(q))
        F = self.k * e + self.d * (v_d - J @ qd)
        tau = J.T @ F
        tau += self.r.bias_torque(q, qd) if self.cc else self.r.gravity_torque(q)
        return tau


@dataclass
class AdmittanceState:
    x: np.ndarray  # 參考位置偏移（3）
    v: np.ndarray


class Admittance:
    """平移導納：M_a ẍ + D_a ẋ + K_a x = F_ext（x 為相對期望位置的偏移）。
    `step(F_ext, dt)` 以半隱式 Euler 積分，回傳偏移 x（加到期望位置上形成內環參考）。"""

    def __init__(self, m: float, d: float, k: float):
        self.m, self.d, self.k = m, d, k
        self.s = AdmittanceState(np.zeros(3), np.zeros(3))

    def reset(self):
        self.s = AdmittanceState(np.zeros(3), np.zeros(3))

    def step(self, f_ext, dt: float):
        s = self.s
        a = (np.asarray(f_ext, float) - self.d * s.v - self.k * s.x) / self.m
        s.v = s.v + a * dt
        s.x = s.x + s.v * dt
        return s.x.copy(), s.v.copy()


class CartesianPositionTracker:
    """導納外環的內環：把笛卡兒參考 (p_ref, v_ref) 以 J⁺ 轉成關節參考並用計算力矩追蹤。
    q_ref 以速度積分（q_ref += J⁺ (v_ref + Kc·e_p) dt）維持，避免逐步 IK。"""

    def __init__(self, robot: SixAxisRobot, inner: ComputedTorque, q0, kc: float = 20.0, lam: float = 1e-6):
        self.r, self.inner, self.kc, self.lam = robot, inner, kc, lam
        self.q_ref = np.array(q0, float)
        self.qd_ref = np.zeros(6)

    def step(self, q, qd, T_ref, v_ref, dt: float):
        J = self.r.jacobian(self.q_ref)
        e = pose_error(T_ref, self.r.fk(self.q_ref))
        twist = np.concatenate([v_ref, np.zeros(3)]) + self.kc * e
        self.qd_ref = J.T @ np.linalg.solve(J @ J.T + self.lam * np.eye(6), twist)
        self.q_ref = self.q_ref + self.qd_ref * dt
        return self.inner(q, qd, self.q_ref, self.qd_ref, np.zeros(6))
