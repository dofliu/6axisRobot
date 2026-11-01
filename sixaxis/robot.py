"""六軸手臂數學模型：DH 參數、正/逆運動學、Jacobian、牛頓-尤拉動力學。

模型依據
--------
* 幾何：UR5 類工業手臂標準 DH 參數（Universal Robots 公開資料；Hawkins 2013,
  "Analytic Inverse Kinematics for the Universal Robots UR-5/UR-10 arms"）。
  標準 DH：  ``T_{i-1,i} = Rz(θ_i) Tz(d_i) Tx(a_i) Rx(α_i)``。
* 逆運動學：Hawkins 的 UR 型封閉解（最多 8 組），每組皆以 FK 驗證。
* Jacobian：幾何法 ``J_v = z_{i-1} × (p_e - p_{i-1})``, ``J_w = z_{i-1}``。
* 動力學：遞迴牛頓-尤拉（Craig, *Introduction to Robotics*, 第 6 章；Luh-Walker-Paul
  1980），質量矩陣以 RNE 逐欄求得，正動力學 ``q̈ = M⁻¹(τ - h)``。

URDF 對應（供 PyBullet 與動力學共用的連桿座標系）
--------------------------------------------------
把 DH 連乘重新分組::

    T = [Rz(θ1)] [Tz(d1)Tx(a1)Rx(α1)] [Rz(θ2)] [Tz(d2)Tx(a2)Rx(α2)] ... [Rz(θ6)] [Tz(d6)Tx(a6)Rx(α6)]

故第 i 關節 = 固定轉換 ``F_i = Tz(d_{i-1})Tx(a_{i-1})Rx(α_{i-1})``（F_1 = I）後繞 z 轉 θ_i；
最後 ``F_7`` 為法蘭（flange）。連桿座標系 L_i 的 z 軸即關節軸，慣性參數定義於 L_i。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81


def rotz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """標準 DH 齊次轉換 Rz(θ)Tz(d)Tx(a)Rx(α)。"""
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def cross(a, b) -> np.ndarray:
    """3 維外積（比 np.cross 快約 10 倍，動力學內迴圈用）。"""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def wrap_pi(x):
    """角度包到 (-π, π]。"""
    return np.pi - np.mod(np.pi - np.asarray(x), 2.0 * np.pi)


def rot_z_to(u: np.ndarray) -> np.ndarray:
    """回傳把 +z 轉到單位向量 u 的旋轉矩陣（Rodrigues）。"""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    c = float(z @ u)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else rotx(np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def rot_log(R: np.ndarray) -> np.ndarray:
    """SO(3) 對數映射（旋轉向量）。"""
    c = (np.trace(R) - 1.0) / 2.0
    th = np.arccos(np.clip(c, -1.0, 1.0))
    if th < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:
        w = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(w))
        s = np.sign(R[k, :] + R[:, k])
        s[k] = 1.0
        return th * w * s
    return th / (2.0 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


@dataclass
class RobotParams:
    """UR5 類六軸手臂參數。長度 m、質量 kg、角度 rad。"""

    a: tuple = (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0)
    d: tuple = (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823)
    alpha: tuple = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
    # 連桿質量（UR5 公開值）
    mass: tuple = (3.7, 8.393, 2.275, 1.219, 1.219, 0.1879)
    # 連桿等效圓柱半徑（決定慣性張量與視覺外觀）
    radius: tuple = (0.060, 0.055, 0.045, 0.040, 0.040, 0.035)
    q_min: tuple = tuple([-2 * np.pi] * 6)
    q_max: tuple = tuple([2 * np.pi] * 6)
    qd_max: tuple = tuple([np.pi] * 6)  # 180 deg/s
    tau_max: tuple = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    flange_mass: float = 1e-6  # 僅為讓 PyBullet 固定連桿有慣性資料
    extra: dict = field(default_factory=dict)


def default_params() -> RobotParams:
    return RobotParams()


class SixAxisRobot:
    """六軸手臂數學模型（運動學 + 動力學）。"""

    def __init__(self, params: RobotParams | None = None):
        self.p = params or default_params()
        p = self.p
        self.a = np.array(p.a)
        self.d = np.array(p.d)
        self.alpha = np.array(p.alpha)
        self.q_min = np.array(p.q_min)
        self.q_max = np.array(p.q_max)
        self.qd_max = np.array(p.qd_max)
        self.tau_max = np.array(p.tau_max)
        self.m = np.array(p.mass)

        # 固定轉換 F_i (i=1..7)：F_1=I, F_i=Tz(d_{i-1})Tx(a_{i-1})Rx(α_{i-1})
        self.RF: list[np.ndarray] = []
        self.pF: list[np.ndarray] = []
        for i in range(7):
            if i == 0:
                self.RF.append(np.eye(3))
                self.pF.append(np.zeros(3))
            else:
                self.RF.append(rotx(self.alpha[i - 1]))
                self.pF.append(np.array([self.a[i - 1], 0.0, self.d[i - 1]]))
        # 連桿等效圓柱：沿 L_i 原點→L_{i+1} 原點，質心在中點
        self.seg: list[np.ndarray] = [self.pF[i + 1].copy() for i in range(6)]
        self.com: list[np.ndarray] = [s / 2.0 for s in self.seg]
        self.R_in: list[np.ndarray] = []  # 主慣性軸 → L_i
        self.I_diag: list[np.ndarray] = []
        self.I_link: list[np.ndarray] = []  # 對質心、於 L_i 座標的慣性張量
        for i in range(6):
            L = float(np.linalg.norm(self.seg[i]))
            r = p.radius[i]
            m = self.m[i]
            i_ax = 0.5 * m * r * r
            i_tr = m * (3 * r * r + L * L) / 12.0
            Rin = rot_z_to(self.seg[i])
            diag = np.array([i_tr, i_tr, i_ax])
            self.R_in.append(Rin)
            self.I_diag.append(diag)
            self.I_link.append(Rin @ np.diag(diag) @ Rin.T)

    # ------------------------------------------------------------ 運動學
    def dh_frames(self, q) -> list[np.ndarray]:
        """回傳 [T_0 ... T_6]（DH 座標系，相對基座）。"""
        T = np.eye(4)
        out = [T]
        for i in range(6):
            T = T @ dh_matrix(self.a[i], self.alpha[i], self.d[i], q[i])
            out.append(T)
        return out

    def fk(self, q) -> np.ndarray:
        """法蘭 4x4 位姿（標準 DH 連乘）。"""
        T = np.eye(4)
        for i in range(6):
            T = T @ dh_matrix(self.a[i], self.alpha[i], self.d[i], q[i])
        return T

    def link_frames(self, q) -> list[np.ndarray]:
        """URDF 風格連桿座標系 L_1..L_6 與法蘭 (共 7 個 4x4)。"""
        T = np.eye(4)
        out = []
        for i in range(6):
            F = np.eye(4)
            F[:3, :3], F[:3, 3] = self.RF[i], self.pF[i]
            Rz = np.eye(4)
            Rz[:3, :3] = rotz(q[i])
            T = T @ F @ Rz
            out.append(T)
        F = np.eye(4)
        F[:3, :3], F[:3, 3] = self.RF[6], self.pF[6]
        out.append(T @ F)
        return out

    def jacobian(self, q) -> np.ndarray:
        """幾何 Jacobian (6x6)，基座座標、參考點為法蘭原點；上 3 列線速度、下 3 列角速度。"""
        Ts = self.dh_frames(q)
        pe = Ts[6][:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z = Ts[i][:3, 2]
            J[:3, i] = np.cross(z, pe - Ts[i][:3, 3])
            J[3:, i] = z
        return J

    # --- 逆運動學 (Hawkins 封閉解) ---
    def ik_all(self, T, q_ref=None, eps: float = 1e-9) -> list[np.ndarray]:
        """回傳目標位姿 T 的全部封閉解（各軸 wrap 到 (-π, π]，尚未套用限位）。

        奇異點：腕奇異 (sinθ5≈0) 時 θ6 取 q_ref[5]（無則 0）；肩/肘奇異時兩解合併。
        """
        T = np.asarray(T, float)
        a2, a3 = self.a[1], self.a[2]
        d1, d4, d5, d6 = self.d[0], self.d[3], self.d[4], self.d[5]
        th6_ref = 0.0 if q_ref is None else float(q_ref[5])
        p06 = T[:3, 3]
        p05 = p06 - d6 * T[:3, 2]
        r = float(np.hypot(p05[0], p05[1]))
        if r < d4 - 1e-9:
            return []  # 腕心離軸 1 太近，無解
        ratio = min(d4 / r, 1.0) if r > 1e-12 else 1.0
        phi = np.arctan2(p05[1], p05[0])
        off = np.arccos(ratio)
        T60 = np.linalg.inv(T)
        sols: list[np.ndarray] = []
        for s1 in (1.0, -1.0):
            th1 = phi + s1 * off + np.pi / 2
            c1, s1_ = np.cos(th1), np.sin(th1)
            arg = (p06[0] * s1_ - p06[1] * c1 - d4) / d6
            if abs(arg) > 1.0 + 1e-9:
                continue
            arg = float(np.clip(arg, -1.0, 1.0))
            for s5 in (1.0, -1.0):
                th5 = s5 * np.arccos(arg)
                sin5 = np.sin(th5)
                if abs(sin5) < 1e-7:
                    th6 = th6_ref
                else:
                    th6 = np.arctan2(
                        (-T60[1, 0] * s1_ + T60[1, 1] * c1) / sin5,
                        (T60[0, 0] * s1_ - T60[0, 1] * c1) / sin5,
                    )
                T01 = dh_matrix(0.0, self.alpha[0], d1, th1)
                T45 = dh_matrix(0.0, self.alpha[4], d5, th5)
                T56 = dh_matrix(0.0, self.alpha[5], d6, th6)
                T14 = np.linalg.inv(T01) @ T @ np.linalg.inv(T56) @ np.linalg.inv(T45)
                p13 = (T14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
                n13 = float(np.linalg.norm(p13))
                c3 = (n13**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
                if abs(c3) > 1.0 + 1e-9:
                    continue
                c3 = float(np.clip(c3, -1.0, 1.0))
                for s3 in (1.0, -1.0):
                    th3 = s3 * np.arccos(c3)
                    th2 = -np.arctan2(p13[1], -p13[0]) + np.arcsin(np.clip(a3 * np.sin(th3) / n13, -1.0, 1.0))
                    T12 = dh_matrix(a2, 0.0, 0.0, th2)
                    T23 = dh_matrix(a3, 0.0, 0.0, th3)
                    T34 = np.linalg.inv(T12 @ T23) @ T14
                    th4 = np.arctan2(T34[1, 0], T34[0, 0])
                    q = wrap_pi(np.array([th1, th2, th3, th4, th5, th6]))
                    Te = self.fk(q)
                    err = np.linalg.norm(Te[:3, 3] - T[:3, 3]) + np.linalg.norm(Te[:3, :3] - T[:3, :3])
                    if err < 1e-6 and not any(np.allclose(q, s, atol=1e-7) for s in sols):
                        sols.append(q)
        return sols

    def ik(self, T, q_ref=None, respect_limits: bool = True) -> list[np.ndarray]:
        """封閉解 → 對每軸挑 ±2πk 中離 q_ref 最近且在限位內者；依與 q_ref 距離由近到遠排序。"""
        q_ref = np.zeros(6) if q_ref is None else np.asarray(q_ref, float)
        out = []
        for s in self.ik_all(T, q_ref):
            q = s.copy()
            ok = True
            for j in range(6):
                cands = [s[j] + 2 * np.pi * k for k in (-2, -1, 0, 1, 2)]
                if respect_limits:
                    cands = [c for c in cands if self.q_min[j] - 1e-12 <= c <= self.q_max[j] + 1e-12]
                if not cands:
                    ok = False
                    break
                q[j] = min(cands, key=lambda c: abs(c - q_ref[j]))
            if ok:
                out.append(q)
        out.sort(key=lambda q: float(np.linalg.norm(q - q_ref)))
        return out

    def ik_numeric(self, T, q0, tol=1e-10, max_iter=500, damping=1e-3, max_step=0.3):
        """阻尼最小平方 (Levenberg-Marquardt) 數值 IK，回傳 (q, 是否收斂, 迭代數)。"""
        q = np.array(q0, float)
        for it in range(max_iter):
            Tc = self.fk(q)
            e = np.concatenate([T[:3, 3] - Tc[:3, 3], rot_log(T[:3, :3] @ Tc[:3, :3].T)])
            if np.linalg.norm(e) < tol:
                return q, True, it
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            nrm = np.linalg.norm(dq)
            if nrm > max_step:
                dq *= max_step / nrm  # 限制單步，避免遠離奇異點附近跳躍
            q = q + dq
        return q, False, max_iter

    # --- 奇異點與限位 ---
    def manipulability(self, q) -> float:
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def sigma_min(self, q) -> float:
        return float(np.linalg.svd(self.jacobian(q), compute_uv=False)[-1])

    def singularities(self, q, tol: float = 1e-2) -> list[str]:
        """偵測 UR 型三種奇異：shoulder / elbow / wrist（tol 為無因次門檻）。"""
        Ts = self.dh_frames(q)
        T6 = Ts[6]
        p05 = T6[:3, 3] - self.d[5] * T6[:3, 2]
        r = float(np.hypot(p05[0], p05[1]))
        s_in = np.sqrt(max(r * r - self.d[3] ** 2, 0.0))  # 腕心在臂平面內離軸1的距離
        out = []
        if s_in < tol * 0.1:  # 腕心過軸 1 (m)
            out.append("shoulder")
        if abs(np.sin(q[2])) < tol:
            out.append("elbow")
        if abs(np.sin(q[4])) < tol:
            out.append("wrist")
        return out

    def check_limits(self, q, qd=None, tau=None) -> dict:
        """回傳超限資訊：dict(joint_pos=[...], joint_vel=[...], torque=[...])，空表示都合格。"""
        res = {}
        q = np.asarray(q, float)
        bad = np.where((q < self.q_min) | (q > self.q_max))[0]
        if bad.size:
            res["joint_pos"] = bad.tolist()
        if qd is not None:
            bad = np.where(np.abs(qd) > self.qd_max)[0]
            if bad.size:
                res["joint_vel"] = bad.tolist()
        if tau is not None:
            bad = np.where(np.abs(tau) > self.tau_max)[0]
            if bad.size:
                res["torque"] = bad.tolist()
        return res

    # ------------------------------------------------------------ 動力學
    def rne(self, q, qd, qdd, gravity: bool = True) -> np.ndarray:
        """遞迴牛頓-尤拉逆動力學：回傳關節力矩 τ (6,)。無摩擦。"""
        z = np.array([0.0, 0.0, 1.0])
        w = np.zeros(3)
        wd = np.zeros(3)
        vd = np.array([0.0, 0.0, G]) if gravity else np.zeros(3)
        Fs, Ns, Rs = [], [], []
        for i in range(6):
            R = self.RF[i] @ rotz(q[i])
            Rt = R.T
            p = self.pF[i]
            vd = Rt @ (cross(wd, p) + cross(w, cross(w, p)) + vd)
            wd_new = Rt @ wd + cross(Rt @ w, z) * qd[i] + z * qdd[i]
            w = Rt @ w + z * qd[i]
            wd = wd_new
            c = self.com[i]
            vc = cross(wd, c) + cross(w, cross(w, c)) + vd
            I = self.I_link[i]
            Fs.append(self.m[i] * vc)
            Ns.append(I @ wd + cross(w, I @ w))
            Rs.append(R)
        f = np.zeros(3)
        n = np.zeros(3)
        tau = np.zeros(6)
        for i in range(5, -1, -1):
            if i == 5:
                Rf = np.zeros((3, 3))
                p_next = self.pF[6]
            else:
                Rf = Rs[i + 1]
                p_next = self.pF[i + 1]
            Rf_f = Rf @ f
            n = Ns[i] + Rf @ n + cross(self.com[i], Fs[i]) + cross(p_next, Rf_f)
            f = Rf_f + Fs[i]
            tau[i] = n[2]
        return tau

    def gravity_torque(self, q) -> np.ndarray:
        """重力補償力矩 g(q)。"""
        return self.rne(q, np.zeros(6), np.zeros(6), True)

    def bias_torque(self, q, qd) -> np.ndarray:
        """h(q,q̇) = C(q,q̇)q̇ + g(q)。"""
        return self.rne(q, qd, np.zeros(6), True)

    def mass_matrix(self, q) -> np.ndarray:
        M = np.zeros((6, 6))
        z = np.zeros(6)
        for i in range(6):
            e = np.zeros(6)
            e[i] = 1.0
            M[:, i] = self.rne(q, z, e, gravity=False)
        return 0.5 * (M + M.T)

    def forward_dynamics(self, q, qd, tau, wrench_ext=None) -> np.ndarray:
        """q̈ = M⁻¹(τ - h + Jᵀ w_ext)；w_ext 為環境作用於法蘭的 6 維力/力矩（基座座標）。"""
        rhs = np.asarray(tau, float) - self.bias_torque(q, qd)
        if wrench_ext is not None:
            rhs = rhs + self.jacobian(q).T @ np.asarray(wrench_ext, float)
        return np.linalg.solve(self.mass_matrix(q), rhs)

    def com_positions(self, q) -> list[np.ndarray]:
        out = []
        for i, T in enumerate(self.link_frames(q)[:6]):
            out.append(T[:3, :3] @ self.com[i] + T[:3, 3])
        return out

    def energy(self, q, qd) -> tuple[float, float]:
        """(動能, 位能)；位能零點為基座平面。"""
        M = self.mass_matrix(q)
        ke = 0.5 * float(np.asarray(qd) @ M @ np.asarray(qd))
        pe = sum(self.m[i] * G * c[2] for i, c in enumerate(self.com_positions(q)))
        return ke, pe

    def simulate(self, q0, qd0, dt: float, steps: int, tau_fn=None):
        """RK4 積分正動力學。tau_fn(t,q,qd)->τ；None 為零力矩（自由擺動）。
        回傳 (t, q[N+1,6], qd[N+1,6])。"""
        q = np.array(q0, float)
        qd = np.array(qd0, float)
        ts = [0.0]
        qs = [q.copy()]
        qds = [qd.copy()]

        def f(t, x):
            qq, vv = x[:6], x[6:]
            tau = np.zeros(6) if tau_fn is None else tau_fn(t, qq, vv)
            return np.concatenate([vv, self.forward_dynamics(qq, vv, tau)])

        x = np.concatenate([q, qd])
        t = 0.0
        for _ in range(steps):
            k1 = f(t, x)
            k2 = f(t + dt / 2, x + dt / 2 * k1)
            k3 = f(t + dt / 2, x + dt / 2 * k2)
            k4 = f(t + dt, x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            ts.append(t)
            qs.append(x[:6].copy())
            qds.append(x[6:].copy())
        return np.array(ts), np.array(qs), np.array(qds)
