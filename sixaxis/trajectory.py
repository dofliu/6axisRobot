"""軌跡規劃函式庫（T003）。

構成
----
* 時間律 `Quintic` / `Trapezoid`：把路徑參數 s∈[0,1] 隨時間展開，回傳 (s, ṡ, s̈)。
  - 五次多項式 s = 10τ³ − 15τ⁴ + 6τ⁵（τ=t/T），邊界 s,ṡ,s̈ 皆為 rest-to-rest；
    峰值 ṡ = 15/(8T)=1.875/T，峰值 |s̈| = 10/(√3 T²)=5.7735/T²（Craig, Ch.7）。
  - 梯形速度：加/減速段各佔 f·T，巡航速度 1/((1−f)T)，加速度 1/(f(1−f)T²)。
* `JointPTP`：關節空間點到點，各軸共用同一個 s(t)（同步到達）。
* `LinePath` / `ArcPath`：笛卡兒直線與圓弧（三點定圓），姿態以 SLERP 由起點到終點等角速度插值。
  路徑以弧長比例 s 參數化，`derivs(s)` 給出對 s 的線速度/角速度係數；實際
  twist = ṡ · derivs(s)。
* `sample_cartesian`：沿路徑取樣 → 逐點 IK（以前一點為初值，保持連續）→ 關節速度（阻尼偽逆 J⁺ twist）
  → 關節加速度（數值微分）→ 逆動力學力矩。
* `check_samples` / `fit_duration`：檢查位置/速度/力矩限制，並自動拉長時間直到合格（時間縮放）。

姿態 SLERP：R(s)=R0·Exp(s·φ)，φ=Log(R0ᵀR1)；世界座標角速度 ω = R0·φ·ṡ（Shoemake 1985）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import SixAxisRobot, rot_log


# ------------------------------------------------------------------ 時間律
class Quintic:
    """五次多項式 rest-to-rest 時間律（含速度、加速度邊界為 0）。"""

    name = "quintic"
    peak_vel_factor = 15.0 / 8.0  # max ṡ · T
    peak_acc_factor = 10.0 / np.sqrt(3.0)  # max |s̈| · T²

    def __init__(self, T: float):
        if T <= 0:
            raise ValueError("duration must be > 0")
        self.T = float(T)

    def __call__(self, t):
        T = self.T
        tau = np.clip(np.asarray(t, float) / T, 0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
        return s, sd, sdd


class Trapezoid:
    """梯形速度時間律；accel_frac=加速段（與減速段）占總時間比例，需 <0.5。"""

    name = "trapezoid"

    def __init__(self, T: float, accel_frac: float = 0.25):
        if T <= 0 or not 0 < accel_frac < 0.5:
            raise ValueError("need T>0 and 0<accel_frac<0.5")
        self.T = float(T)
        self.f = float(accel_frac)
        self.ta = self.f * self.T
        self.v = 1.0 / (self.T - self.ta)
        self.a = self.v / self.ta
        self.peak_vel_factor = 1.0 / (1.0 - self.f)
        self.peak_acc_factor = 1.0 / (self.f * (1.0 - self.f))

    def __call__(self, t):
        t = np.clip(np.asarray(t, float), 0.0, self.T)
        T, ta, v, a = self.T, self.ta, self.v, self.a
        acc = t < ta
        dec = t > T - ta
        s = np.where(acc, 0.5 * a * t**2,
                     np.where(dec, 1 - 0.5 * a * (T - t) ** 2, 0.5 * a * ta**2 + v * (t - ta)))
        sd = np.where(acc, a * t, np.where(dec, a * (T - t), v))
        sdd = np.where(acc, a, np.where(dec, -a, 0.0))
        return s, sd, sdd


def make_profile(kind: str, T: float, **kw):
    if kind == "quintic":
        return Quintic(T)
    if kind == "trapezoid":
        return Trapezoid(T, **kw)
    raise ValueError(f"unknown profile {kind!r}")


def min_duration(delta, qd_max, qdd_max=None, kind: str = "quintic", accel_frac: float = 0.25):
    """滿足各軸速度（與可選加速度）上限的最短時間（同步運動，|Δ| 最大者決定）。"""
    d = np.abs(np.asarray(delta, float))
    if kind == "quintic":
        kv, ka = Quintic.peak_vel_factor, Quintic.peak_acc_factor
    else:
        kv, ka = 1.0 / (1.0 - accel_frac), 1.0 / (accel_frac * (1.0 - accel_frac))
    T = float(np.max(kv * d / np.asarray(qd_max)))
    if qdd_max is not None:
        T = max(T, float(np.sqrt(np.max(ka * d / np.asarray(qdd_max)))))
    return max(T, 1e-3)


# ------------------------------------------------------------------ 關節空間
@dataclass
class JointPTP:
    q0: np.ndarray
    q1: np.ndarray
    profile: object

    @property
    def duration(self) -> float:
        return self.profile.T

    def state(self, t):
        s, sd, sdd = self.profile(t)
        dq = np.asarray(self.q1, float) - np.asarray(self.q0, float)
        return self.q0 + s * dq, sd * dq, sdd * dq


def joint_ptp(q0, q1, T=None, kind="quintic", robot: SixAxisRobot | None = None, qdd_max=None, **kw):
    """T=None 時由機器人速度上限自動求最短時間。"""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if T is None:
        T = min_duration(q1 - q0, robot.qd_max, qdd_max, kind, kw.get("accel_frac", 0.25))
    return JointPTP(q0, q1, make_profile(kind, T, **kw))


# ------------------------------------------------------------------ 笛卡兒路徑
def _slerp_parts(R0, R1):
    phi = rot_log(R0.T @ R1)
    return phi, R0 @ phi  # 局部旋轉向量、世界角速度係數（對 s）


class LinePath:
    """直線（位置）+ SLERP（姿態）。"""

    def __init__(self, T0: np.ndarray, T1: np.ndarray):
        self.T0, self.T1 = T0, T1
        self.length = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        self.phi, self.w_coef = _slerp_parts(T0[:3, :3], T1[:3, :3])
        self.angle = float(np.linalg.norm(self.phi))

    def pose(self, s: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.T0[:3, :3] @ Rotation.from_rotvec(s * self.phi).as_matrix()
        T[:3, 3] = self.T0[:3, 3] + s * (self.T1[:3, 3] - self.T0[:3, 3])
        return T

    def derivs(self, s: float):
        return self.T1[:3, 3] - self.T0[:3, 3], self.w_coef


class ArcPath:
    """過 A→B→C 三點的圓弧（位置），姿態由 A 的姿態 SLERP 到 C 的姿態。"""

    def __init__(self, TA: np.ndarray, B: np.ndarray, TC: np.ndarray):
        A, C = TA[:3, 3], TC[:3, 3]
        u, w = B - A, C - A
        n = np.cross(u, w)
        if np.linalg.norm(n) < 1e-9:
            raise ValueError("three points are collinear")
        n = n / np.linalg.norm(n)
        M = np.array([u, w, n])
        self.center = A + np.linalg.solve(M, np.array([u @ u / 2, w @ w / 2, 0.0]))
        self.radius = float(np.linalg.norm(A - self.center))
        self.e1 = (A - self.center) / self.radius
        self.e2 = np.cross(n, self.e1)
        self.theta = self._angle(C) or 2 * np.pi  # A→B→C 同向，掃過角 (0,2π)
        self.theta_B = self._angle(B)
        self.length = self.radius * self.theta
        self.TA, self.TC = TA, TC
        self.phi, self.w_coef = _slerp_parts(TA[:3, :3], TC[:3, :3])
        self.angle = float(np.linalg.norm(self.phi))

    def _angle(self, P: np.ndarray) -> float:
        d = P - self.center
        return float(np.arctan2(d @ self.e2, d @ self.e1) % (2 * np.pi))

    def pose(self, s: float) -> np.ndarray:
        th = s * self.theta
        T = np.eye(4)
        T[:3, :3] = self.TA[:3, :3] @ Rotation.from_rotvec(s * self.phi).as_matrix()
        T[:3, 3] = self.center + self.radius * (np.cos(th) * self.e1 + np.sin(th) * self.e2)
        return T

    def derivs(self, s: float):
        th = s * self.theta
        dp = self.radius * self.theta * (-np.sin(th) * self.e1 + np.cos(th) * self.e2)
        return dp, self.w_coef


@dataclass
class Samples:
    """取樣後的關節軌跡（等間隔 dt）。"""

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray | None = None
    pos: np.ndarray | None = None  # 末端位置（FK）
    target_pos: np.ndarray | None = None
    ik_fail: int = 0
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)


def sample_joint(robot: SixAxisRobot, traj: JointPTP, dt: float = 0.01) -> Samples:
    n = int(round(traj.duration / dt))
    t = np.linspace(0.0, traj.duration, n + 1)
    q, qd, qdd = (np.array(x) for x in zip(*[traj.state(x) for x in t], strict=True))
    tau = np.array([robot.rne(a, b, c) for a, b, c in zip(q, qd, qdd, strict=True)])
    pos = np.array([robot.fk(a)[:3, 3] for a in q])
    return Samples(t, q, qd, qdd, tau, pos)


def sample_cartesian(robot: SixAxisRobot, path, profile, q0, dt: float = 0.01,
                     lam: float = 1e-4) -> Samples:
    """沿笛卡兒路徑取樣並轉成關節軌跡。q0 為起點附近的關節姿態（作為 IK 初值）。"""
    n = int(round(profile.T / dt))
    t = np.linspace(0.0, profile.T, n + 1)
    s, sd, _ = profile(t)
    q = np.zeros((n + 1, 6))
    qd = np.zeros((n + 1, 6))
    tgt = np.zeros((n + 1, 3))
    fail = 0
    qk = np.asarray(q0, float)
    for k in range(n + 1):
        T = path.pose(float(s[k]))
        qk, ok, _ = robot.ik_numeric(T, qk, tol=1e-10, damping=1e-8)
        fail += int(not ok)
        dp, dw = path.derivs(float(s[k]))
        J = robot.jacobian(qk)
        twist = np.concatenate([dp, dw]) * sd[k]
        qd[k] = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), twist)
        q[k], tgt[k] = qk, T[:3, 3]
    qdd = np.gradient(qd, t, axis=0, edge_order=2)
    tau = np.array([robot.rne(a, b, c) for a, b, c in zip(q, qd, qdd, strict=True)])
    pos = np.array([robot.fk(a)[:3, 3] for a in q])
    return Samples(t, q, qd, qdd, tau, pos, tgt, fail)


def concat(parts: list[Samples]) -> Samples:
    """串接多段取樣（各段終點=下一段起點時略去重複點）。"""
    t0, ts, qs, qds, qdds, taus, poss = 0.0, [], [], [], [], [], []
    for i, p in enumerate(parts):
        sl = slice(1 if i else 0, None)
        ts.append(p.t[sl] + t0)
        qs.append(p.q[sl])
        qds.append(p.qd[sl])
        qdds.append(p.qdd[sl])
        taus.append(p.tau[sl])
        poss.append(p.pos[sl])
        t0 += p.t[-1]
    return Samples(np.concatenate(ts), np.vstack(qs), np.vstack(qds), np.vstack(qdds),
                   np.vstack(taus), np.vstack(poss), None, sum(p.ik_fail for p in parts))


# ------------------------------------------------------------------ 限制檢查與時間縮放
def check_samples(robot: SixAxisRobot, s: Samples) -> dict:
    """回傳各項限制的使用率（max|x|/limit）與是否超限。"""
    pos_bad = bool(np.any(s.q < robot.q_min) or np.any(s.q > robot.q_max))
    r_v = float(np.max(np.abs(s.qd) / robot.qd_max))
    r_t = float(np.max(np.abs(s.tau) / robot.tau_max))
    return {"pos_violation": pos_bad, "vel_ratio": r_v, "torque_ratio": r_t,
            "vel_ok": r_v <= 1.0, "torque_ok": r_t <= 1.0, "ok": (not pos_bad) and r_v <= 1 and r_t <= 1}


def fit_duration(robot: SixAxisRobot, build, T0: float, margin: float = 0.95, max_iter: int = 40):
    """時間縮放：從 T0 起逐步拉長，直到速度/力矩使用率 ≤ margin。

    build(T) -> Samples。回傳 (T, samples, history[(T, vel_ratio, torque_ratio)])；
    若 max_iter 內仍超限（例如靜態重力力矩已超限），samples 的 meta['feasible']=False。
    """
    T, hist = float(T0), []
    for _ in range(max_iter):
        smp = build(T)
        c = check_samples(robot, smp)
        hist.append((T, c["vel_ratio"], c["torque_ratio"]))
        if c["vel_ratio"] <= margin and c["torque_ratio"] <= margin:
            smp.meta["feasible"] = True
            return T, smp, hist
        T *= max(c["vel_ratio"] / margin, 1.05, np.sqrt(c["torque_ratio"] / margin) * 0.9 + 0.1)
    smp.meta["feasible"] = False
    return T, smp, hist
