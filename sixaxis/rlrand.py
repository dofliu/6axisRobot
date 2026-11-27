"""T205 Gymnasium 環境：隨機化物件位置與姿態＋domain randomization 的 pick-and-place。

在 T204 `PlaceEnv`（目標條件化取放）上加入：

1. 物件姿態隨機：長方塊（尺寸 L × w × h，物件 x 軸為長邊 L ≥ w）初始 yaw θ_o ~ U[−π/2, π/2)
   （長方塊對 z 軸 180° 對稱，故此範圍涵蓋所有相異方位）。平行夾爪最大開口 2(W0 − PAD_T/2) = 80 mm，
   當長邊 L 接近或超過開口時，只能「橫跨短邊 w」抓取，策略必須先把夾爪轉到對的方位。
2. 夾爪方位動作：動作改為 5 維 a = [v_x, v_y, v_z, ψ̇, grip]（[−1, 1]）。工具保持朝下，但可繞世界 z 軸轉 ψ：
   R_d(ψ) = R_z(ψ)·R_DOWN，ψ_ref ← clip(ψ_ref + a₃ ω_ψ Δt, ±ψ_lim)；姿態控制
   ω = k_R log(R_d Rᵀ) + ψ̇ ẑ（比例回授＋前饋），再以 DLS 解關節速度（同 T203/T204）。
3. Domain randomization（Tobin et al. 2017 "Domain Randomization for Transferring Deep Neural Networks from
   Simulation to the Real World"；Peng et al. 2018 "Sim-to-Real Transfer of Robotic Control with Dynamics
   Randomization"）：每回合均勻抽樣
     物件尺寸 w ∈ [35, 55] mm、L ∈ [max(w, 50), 100] mm、h ∈ [45, 60] mm；質量 m ∈ [0.1, 0.4] kg；
     物件摩擦 μ_obj ∈ [0.5, 1.0]（指墊 μ = 1 → PyBullet 有效 μ = μ_obj）；夾持力 F_grip ∈ [30, 50] N；
     關節伺服頻寬 ω_n ∈ [20, 30] rad/s（K_p = ω_n²、K_d = 2ω_n）；
   觀測雜訊（每步獨立高斯）：物件位置 σ_p = 2 mm、物件 yaw σ_ψ = 2°（模擬視覺位姿估測誤差）。
   獎勵與成功判定一律用真值（雜訊只進觀測）。dr=False 時只隨機化位置與 yaw，
   物件固定為標稱 70×45×50 mm 長方塊。

觀測（42 維）：T204 36 維（物件位置/四元數為含雜訊估測）＋ [cos 2Δ, sin 2Δ]（Δ = 物件短邊軸與夾爪指軸的
相對 yaw，對 π 取模；2Δ 使 180° 對稱的兩個解等價）＋ 物件尺寸 (L, w, h)/0.1 ＋ ψ_ref/π。

接近階段獎勵乘上方位對齊倍率 0.6 + 0.4 cos²Δ（`PlaceEnv._reach_scale`），其餘分階段獎勵同 T204。

夾持幾何（報告中以模擬驗證）：方位誤差 Δ 時，若為「全寬」夾爪，兩指間需跨越長方形在指軸方向的投影寬度
  w_eff(Δ) = w |cos Δ| + L |sin Δ|（Δ_max：w cos Δ + L sin Δ = 80 mm 的根）；
本夾爪指墊寬僅 b = 20 mm，指墊只掃過寬 b 的帶狀區，需跨越的是帶內物件長度 `strip_extent`（≤ w_eff），
「指墊不碰物件頂面即可下降」的條件為 strip_extent ≤ 80 mm（`clear_misalignment`）。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pybullet as pb
from gymnasium import spaces

from .gripper import PAD_L, PAD_T, PAD_W, T_TCP, W0, ObjectSpec
from .rlenv import R_DOWN
from .rlpick import TABLE_TOP
from .rlplace import PlaceEnv
from .robot import rot_log

OPENING = 2 * (W0 - PAD_T / 2)  # 全開時兩指墊內表面間距 = 0.080 m
NOMINAL = ObjectSpec("bar", "box", (0.070, 0.045, 0.050), 0.20, friction=0.8, rgba=(0.85, 0.45, 0.1, 1.0))
DR_RANGES = {
    "w": (0.035, 0.055),
    "L": (0.050, 0.100),
    "h": (0.045, 0.060),
    "mass": (0.10, 0.40),
    "mu": (0.5, 1.0),
    "f_grip": (30.0, 50.0),
    "wn": (20.0, 30.0),
}
SIGMA_P = 0.002
SIGMA_YAW = np.deg2rad(2.0)


def wrap_half_pi(a: float) -> float:
    """角度對 π 取模到 [−π/2, π/2)。"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def effective_width(w: float, L: float, delta: float) -> float:
    """方位誤差 Δ 時長方形 (L × w) 在夾爪指軸方向的投影寬度 w|cosΔ| + L|sinΔ|。"""
    return float(w * abs(np.cos(delta)) + L * abs(np.sin(delta)))


def max_misalignment(w: float, L: float, opening: float = OPENING) -> float:
    """w cosΔ + L sinΔ = opening 在 [0, π/2] 的最小根（無解且 w ≤ opening → π/2；w > opening → 0）。
    令 w cosΔ + L sinΔ = R cos(Δ − φ)，R = √(w² + L²)、φ = atan2(L, w) → Δ = φ − acos(opening/R)。"""
    if w > opening:
        return 0.0
    R = np.hypot(w, L)
    if R <= opening:
        return float(np.pi / 2)
    return float(np.arctan2(L, w) - np.arccos(opening / R))


def strip_extent(w: float, L: float, delta: float, b: float = PAD_W) -> float:
    """寬 b 的指墊沿指軸下降時，所掃過帶狀區（|垂直距離| ≤ b/2）內物件的全長 2·t_max。

    物件座標 |x| ≤ L/2、|y| ≤ w/2；指軸 u = (sinΔ, cosΔ)、垂直 n = (cosΔ, −sinΔ)，帶內點 t·u + s·n。
    正 t 側的有效約束為 t sinΔ + s cosΔ ≤ L/2（f₁(s) 對 s 遞減）與 t cosΔ − s sinΔ ≤ w/2（f₂(s) 遞增），
    此線性規劃的最大 t 在兩約束交點（物件角點）s_c = (L/2)cosΔ − (w/2)sinΔ，夾到 [−b/2, b/2]：
      t_max = min(f₁(s_c), f₂(s_c))。
    b → ∞ 時退化為投影寬度 w|cosΔ| + L|sinΔ|（全寬夾爪）；b = 0 為中心弦長 min(w/|cosΔ|, L/|sinΔ|)。
    指墊能「不碰物件頂面」地下降到兩側的條件：strip_extent ≤ 開口 80 mm。"""
    d = abs(wrap_half_pi(delta))
    sn, cs = np.sin(d), np.cos(d)
    sc = float(np.clip(L / 2 * cs - w / 2 * sn, -b / 2, b / 2))
    f1 = (L / 2 - sc * cs) / sn if sn > 1e-12 else np.inf
    f2 = (w / 2 + sc * sn) / cs if cs > 1e-12 else np.inf
    return float(2 * min(f1, f2))


def clear_misalignment(w: float, L: float, b: float = PAD_W, opening: float = OPENING) -> float:
    """strip_extent(Δ) ≤ opening 的最大 |Δ|（0.05° 格點＋二分；全區可行回傳 π/2）。"""
    grid = np.radians(np.arange(0.0, 90.0001, 0.05))
    ok = [strip_extent(w, L, x, b) <= opening for x in grid]
    if all(ok):
        return float(np.pi / 2)
    i = int(np.argmin(ok))
    if i == 0:
        return 0.0
    lo, hi = grid[i - 1], grid[i]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if strip_extent(w, L, mid, b) <= opening else (lo, mid)
    return float(lo)


def _obb_overlap(c1, ax1, h1, c2, ax2, h2) -> bool:
    """二維有向矩形重疊（分離軸定理：4 個邊法向量上投影皆重疊才相交）。ax：2×2，列為單位軸；h：半長。"""
    d = np.asarray(c2, float) - np.asarray(c1, float)
    for n in (*ax1, *ax2):
        r1 = sum(h * abs(float(n @ a)) for a, h in zip(ax1, h1, strict=True))
        r2 = sum(h * abs(float(n @ a)) for a, h in zip(ax2, h2, strict=True))
        if abs(float(n @ d)) > r1 + r2:
            return False
    return True


def pads_hit_box(p_tcp_xy, finger_yaw: float, obj_xy, short_yaw: float, L: float, w: float,
                 stroke: float = 0.0, b: float = PAD_W) -> bool:
    """以實際位姿判斷指墊（俯視為 PAD_T × b 矩形，中心在 TCP ± (W0 − stroke) 沿指軸）是否與物件
    （L × w 矩形，短邊軸 yaw = short_yaw）重疊：True 表示指墊下降時會碰到物件頂面/稜邊。
    `strip_extent` 為其理想情形（TCP 對準物件中心、無側向偏移）的解析解。"""
    u = np.array([np.cos(finger_yaw), np.sin(finger_yaw)])
    x = np.array([-u[1], u[0]])
    v = np.array([np.cos(short_yaw), np.sin(short_yaw)])
    ob = (np.array([-v[1], v[0]]), v)  # 物件長邊軸、短邊軸
    for sg in (1.0, -1.0):
        c = np.asarray(p_tcp_xy, float) + sg * (W0 - stroke) * u
        if _obb_overlap(c, (u, x), (PAD_T / 2, b / 2), obj_xy, ob, (L / 2, w / 2)):
            return True
    return False


def short_axis_yaw(quat) -> float:
    """物件短邊軸（物件 y 軸）的世界 yaw。"""
    R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return float(np.arctan2(R[1, 1], R[0, 1]))


def finger_axis_yaw(R: np.ndarray) -> float:
    """夾爪指軸（工具 y 軸）的世界 yaw。"""
    return float(np.arctan2(R[1, 1], R[0, 1]))


class RandPlaceEnv(PlaceEnv):
    def __init__(self, dr: bool = True, obs_noise: bool | None = None, omega_psi: float = 1.5,
                 psi_lim: float = np.pi / 2 + 0.35, yaw_low: float = -np.pi / 2, yaw_high: float = np.pi / 2,
                 dr_ranges: dict | None = None, **kw):
        super().__init__(obj=NOMINAL, **kw)
        self.dr = dr
        self.obs_noise = dr if obs_noise is None else obs_noise
        self.omega_psi, self.psi_lim = omega_psi, psi_lim
        self.yaw_low, self.yaw_high = yaw_low, yaw_high
        self.ranges = dict(DR_RANGES, **(dr_ranges or {}))
        self.nominal = {"f_grip": self.f_grip, "wn": float(np.sqrt(self.kp))}
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (42,), np.float32)
        self.psi_ref, self.psi_rate = 0.0, 0.0
        self.params = {}
        self._psi_init = True

    @property
    def demo_policy(self):
        return ScriptedRandPlacePolicy

    # ------------------------------------------------------------ 隨機化
    def sample_params(self) -> dict:
        rng, R = self.np_random, self.ranges
        if not self.dr:
            L, w, h = NOMINAL.size
            return {"L": L, "w": w, "h": h, "mass": NOMINAL.mass, "mu": NOMINAL.friction, **self.nominal}
        w = rng.uniform(*R["w"])
        L = rng.uniform(max(w, R["L"][0]), R["L"][1])
        return {"L": float(L), "w": float(w), "h": float(rng.uniform(*R["h"])),
                "mass": float(rng.uniform(*R["mass"])), "mu": float(rng.uniform(*R["mu"])),
                "f_grip": float(rng.uniform(*R["f_grip"])), "wn": float(rng.uniform(*R["wn"]))}

    def apply_params(self, p: dict):
        self.params = dict(p)
        self.part = ObjectSpec("rbox", "box", (p["L"], p["w"], p["h"]), p["mass"], friction=p["mu"],
                               rgba=NOMINAL.rgba)
        self.z_obj0 = TABLE_TOP + p["h"] / 2
        self.f_grip = p["f_grip"]
        self.kp, self.kd = p["wn"] ** 2, 2.0 * p["wn"]

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        gym.Env.reset(self, seed=seed)
        p = self.sample_params()
        p.update({k: v for k, v in options.get("params", {}).items()})
        self.apply_params(p)
        if "obj_yaw" not in options:
            options["obj_yaw"] = float(self.np_random.uniform(self.yaw_low, self.yaw_high))
        self.obj_yaw0 = options["obj_yaw"]
        self.psi_rate = 0.0
        self._psi_init = True
        obs, info = super().reset(seed=None, options=options)
        self.yaw_rel0 = self.yaw_rel(self.last)  # 初始（真值）相對 yaw：策略需轉夾爪的角度
        info.update({"params": dict(self.params), "obj_yaw": self.obj_yaw0, "yaw_rel": self.yaw_rel0})
        return obs, info

    # ------------------------------------------------------------ 觀測
    def yaw_rel(self, m, noise: float = 0.0) -> float:
        """Δ = 物件短邊軸 yaw − 夾爪指軸 yaw（對 π 取模）；把夾爪轉 +Δ 即對齊。"""
        return wrap_half_pi(short_axis_yaw(m["qo"]) + noise - finger_axis_yaw(m["R"]))

    def _obs(self, m):
        if self._psi_init:  # reset 時以實際工具方位初始化 ψ_ref（q0 含隨機擾動）
            self.psi_ref = float(np.arctan2(m["R"][1, 0], m["R"][0, 0]))
            self._psi_init = False
        m2 = dict(m)
        n_yaw = 0.0
        if self.obs_noise:
            m2["po"] = m["po"] + self.np_random.normal(0.0, SIGMA_P, 3)
            n_yaw = float(self.np_random.normal(0.0, SIGMA_YAW))
            q_yaw = pb.getQuaternionFromEuler([0, 0, n_yaw])
            q_n = pb.multiplyTransforms([0, 0, 0], q_yaw, [0, 0, 0], m["qo"])[1]
            m2["qo"] = np.array(q_n)
            m2["h"] = float(m2["po"][2] - self.z_obj0)
        d = self.yaw_rel(m, n_yaw)
        m2["yaw_rel"] = d
        self.obs_m = m2
        o = super()._obs(m2)
        L, w, h = self.part.size
        extra = [np.cos(2 * d), np.sin(2 * d), L / 0.1, w / 0.1, h / 0.1, self.psi_ref / np.pi]
        return np.concatenate([o, np.asarray(extra, np.float32)])

    def _reach_scale(self, m) -> float:
        return 0.6 + 0.4 * np.cos(self.yaw_rel(m)) ** 2

    # ------------------------------------------------------------ 控制
    def joint_cmd(self, v) -> np.ndarray:
        """TCP 線速度 v ＋ 繞世界 z 的工具轉速 ψ̇ → 關節速度
        （DLS；同 PickEnv.joint_cmd，但姿態目標隨 ψ_ref 轉）。"""
        psi_new = float(np.clip(self.psi_ref + self.psi_rate * self.dt_c, -self.psi_lim, self.psi_lim))
        rate = (psi_new - self.psi_ref) / self.dt_c
        self.psi_ref = psi_new
        q, _ = self.pbr.get_state()
        T = self.robot.fk(q)
        R = T[:3, :3]
        R_d = rz(self.psi_ref) @ R_DOWN
        w = self.k_rot * rot_log(R_d @ R.T) + np.array([0.0, 0.0, rate])
        r = R @ T_TCP[:3, 3]
        v_fl = v - np.cross(w, r)
        J = self.robot.jacobian(q)
        u = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), np.concatenate([v_fl, w]))
        s = np.max(np.abs(u)) / self.omega_max
        return u / s if s > 1.0 else u

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.psi_rate = float(a[3]) * self.omega_psi
        obs, r, terminated, truncated, info = super().step(np.array([a[0], a[1], a[2], a[4]]))
        info["yaw_rel"] = self.yaw_rel(self.last)
        info["psi_ref"] = self.psi_ref
        return obs, r, terminated, truncated, info


# ---------------------------------------------------------------- 傳統基準：含方位對齊的腳本取放
class ScriptedRandPlacePolicy:
    """T204 腳本 FSM 加上方位對齊：APPROACH 期間同時以 ψ̇ = k_ψ Δ 把指軸轉到物件短邊軸，
    |Δ| < tol_yaw 且到達上方才 DESCEND；其餘 CLOSE → LIFT → CARRY → LOWER → OPEN → RETREAT 同 T204。
    物件位姿讀「含雜訊觀測」（env.obs_m，與 RL 相同資訊），TCP 位置讀本體感測（無雜訊）。
    noise > 0 時對速度指令加高斯雜訊（示範多樣化）。"""

    def __init__(self, env: RandPlaceEnv, kp: float = 4.0, k_psi: float = 4.0, z_app: float = 0.08,
                 tol: float = 0.006, tol_yaw: float = np.deg2rad(4.0), n_close: int = 6, n_open: int = 5,
                 h_carry: float = 0.08, h_rel: float = 0.002, noise: float = 0.0, seed: int = 0):
        self.env, self.kp, self.k_psi, self.z_app = env, kp, k_psi, z_app
        self.tol, self.tol_yaw = tol, tol_yaw
        self.n_close, self.n_open, self.h_carry, self.h_rel = n_close, n_open, h_carry, h_rel
        self.noise, self.rng = noise, np.random.default_rng(seed)

    def reset(self):
        self.phase, self.n = "approach", 0
        self.po_f = None

    def __call__(self, obs):
        e = self.env
        p = e.last["p"]
        mo = e.obs_m
        # 估測位置一階低通（α = 0.5）降低 2 mm 量測雜訊
        self.po_f = mo["po"].copy() if self.po_f is None else 0.5 * self.po_f + 0.5 * mo["po"]
        po, g = self.po_f, e.goal
        z_min = TABLE_TOP + PAD_L / 2 + 0.004  # 指墊下緣不碰台面
        grip, yaw_cmd = -1.0, 0.0
        if self.phase == "approach":
            tgt = po + [0, 0, self.z_app]
            yaw_cmd = self.k_psi * mo["yaw_rel"] / e.omega_psi
            if np.linalg.norm(tgt - p) < 2 * self.tol and abs(mo["yaw_rel"]) < self.tol_yaw:
                self.phase = "descend"
        if self.phase == "descend":
            tgt = np.array([po[0], po[1], max(po[2], z_min)])
            yaw_cmd = self.k_psi * mo["yaw_rel"] / e.omega_psi
            if np.linalg.norm(tgt - p) < self.tol:
                self.phase, self.n = "close", 0
        if self.phase == "close":
            tgt, grip, yaw_cmd = p.copy(), 1.0, 0.0
            self.n += 1
            if self.n >= self.n_close:
                self.phase, self.off = "lift", p - po
        if self.phase == "lift":
            tgt, grip = np.array([*p[:2], e.z_obj0 + self.h_carry]), 1.0
            if abs(tgt[2] - p[2]) < 2 * self.tol:
                self.phase = "carry"
        if self.phase == "carry":
            grip = 1.0
            tgt = np.array([*(g[:2] + self.off[:2]), e.z_obj0 + self.h_carry])
            if np.linalg.norm(tgt - p) < self.tol:
                self.phase = "lower"
        if self.phase == "lower":
            grip = 1.0
            tgt = np.array([*(g[:2] + self.off[:2]), e.z_obj0 + self.h_rel + self.off[2]])
            if np.linalg.norm(tgt - p) < 0.5 * self.tol:
                self.phase, self.n = "open", 0
        if self.phase == "open":
            tgt = p.copy()
            self.n += 1
            if self.n >= self.n_open:
                self.phase, self.ret = "retreat", p.copy()
        if self.phase == "retreat":
            tgt = self.ret + [0, 0, self.z_app]
        v = self.kp * (tgt - p)
        n = np.linalg.norm(v)
        if n > e.v_max:
            v *= e.v_max / n
        a = np.clip(v / e.v_max, -1, 1)
        if self.noise > 0:
            a = np.clip(a + self.rng.normal(0.0, self.noise, 3), -1, 1)
        return np.array([*a, float(np.clip(yaw_cmd, -1, 1)), grip])


class RandomRandPlacePolicy:
    def __init__(self, env: RandPlaceEnv, seed: int = 0):
        self.env, self.rng = env, np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, 5)


gym.register(id="SixAxisRandPlace-v0", entry_point="sixaxis.rlrand:RandPlaceEnv", max_episode_steps=None)
