"""T201 Gymnasium 環境：reach（法蘭到達隨機目標點並停住）。

物理：PyBullet（DIRECT）UR5 類手臂 + 地面，物理步長 2 ms；策略頻率 20 Hz（每個動作 25 個子步）。

動作 → 力矩（低階控制器，與 T004 計算力矩相同的定律）：
* action_mode="joint"：a ∈ [−1,1]⁶ 為關節速度指令 u = a·ω_max（rad/s）。
* action_mode="ee"   ：a ∈ [−1,1]³ 為法蘭線速度指令 v = a·v_max（m/s）；姿態保持 ω = k_R·log(R₀Rᵀ)；
  以阻尼最小平方 q̇ = Jᵀ(JJᵀ + λ²I)⁻¹[v; ω]（Wampler 1986；Nakamura & Hanafusa 1986）換成關節速度，
  超過 ω_max 時等比縮放（保持方向）。
兩者皆積分成參考 q_ref ← q_ref + u·Δt（限制 |q_ref − q| ≤ e_max 防止積分飽和、並夾在關節限位內），
子步間線性內插，內環計算力矩 τ = M(q)(K_p e + K_d ė) + C(q,q̇)q̇ + g(q)
= ID(q, q̇, K_p e + K_d ė)（PyBullet calculateInverseDynamics；T002 已驗證與自寫 RNE 一致），
K_p = ω_n²、K_d = 2ζω_n（臨界阻尼，Spong《Robot Modeling and Control》式 8.4），並依 τ_max 飽和。

觀測（27 維，float32，夾在 ±10）：q/π、q̇/q̇_max、p_ee、p_goal、p_goal − p_ee、
上一步動作（ee 模式補零到 6 維）。

獎勵（dense，預設）：r = −d − c_a‖a‖² + R_succ·[成功] − R_coll·[碰撞]，d = ‖p_goal − p_ee‖（m）。
「sparse」：r = −1（未成功）/ 0（成功）− R_coll·[碰撞]。
成功：d < tol 且 ‖ṗ_ee‖ < v_tol（到達並停住）；碰撞（手臂–地面、非相鄰連桿自碰）→ 終止。
截斷：max_steps（預設 100 步 = 5 s）。

目標取樣：在前方工作空間方塊內均勻取樣，並要求工具朝下的封閉解 IK 存在且腕心離軸 1 ≥ r_min，
確保目標在姿態保持下可達（規劃基準也以此姿態求解）。
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import pybullet as pb
from gymnasium import spaces

from .pbsim import FLANGE, PBRobot
from .robot import SixAxisRobot, rot_log
from .trajectory import joint_ptp, min_duration

R_DOWN = np.diag([1.0, -1.0, -1.0])  # 工具（法蘭 z）朝下
Q_SEED = np.array([0.0, -1.6, -2.0, -1.1, 1.57, 0.0])  # 肘上、工具朝下解的選解參考（同 T105）
HOME_P = np.array([0.40, 0.0, 0.45])
GOAL_LOW = np.array([0.25, -0.40, 0.10])
GOAL_HIGH = np.array([0.65, 0.40, 0.55])


def pose(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        action_mode: str = "joint",
        reward_type: str = "dense",
        control_hz: float = 20.0,
        sim_hz: float = 500.0,
        max_steps: int = 100,
        tol: float = 0.02,
        v_tol: float = 0.10,
        omega_max: float = 1.0,
        v_max: float = 0.30,
        wn: float = 25.0,
        e_max: float = 0.30,
        q0_noise: float = 0.10,
        c_action: float = 0.01,
        r_success: float = 5.0,
        r_collision: float = 10.0,
        terminate_on_success: bool = True,
        goal_low=GOAL_LOW,
        goal_high=GOAL_HIGH,
        render_mode: str | None = None,
        render_size=(400, 300),
    ):
        assert action_mode in ("joint", "ee") and reward_type in ("dense", "sparse")
        self.action_mode, self.reward_type = action_mode, reward_type
        self.dt_c = 1.0 / control_hz
        self.dt = 1.0 / sim_hz
        self.n_sub = int(round(sim_hz / control_hz))
        self.max_steps, self.tol, self.v_tol = max_steps, tol, v_tol
        self.omega_max, self.v_max, self.e_max = omega_max, v_max, e_max
        self.kp, self.kd = wn**2, 2.0 * wn
        self.q0_noise = q0_noise
        self.c_action, self.r_success, self.r_collision = c_action, r_success, r_collision
        self.terminate_on_success = terminate_on_success
        self.goal_low, self.goal_high = np.asarray(goal_low, float), np.asarray(goal_high, float)
        self.render_mode, self.render_size = render_mode, render_size

        self.robot = SixAxisRobot()
        self.pbr = PBRobot(self.robot, plane=True)
        self.cid, self.body = self.pbr.cid, self.pbr.id
        self.pbr.make_frictionless()
        pb.setTimeStep(self.dt, physicsClientId=self.cid)
        self.goal_marker = None
        self.q_home = self.robot.ik(pose(R_DOWN, HOME_P), Q_SEED)[0]
        # 非相鄰連桿對（自碰檢查）；link 0..5 = link1..6
        self.self_pairs = [(i, j) for i in range(6) for j in range(i + 2, 6)]

        nact = 6 if action_mode == "joint" else 3
        self.action_space = spaces.Box(-1.0, 1.0, (nact,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (27,), np.float32)
        self.q_ref = self.q_home.copy()
        self.goal = HOME_P.copy()
        self.prev_a = np.zeros(6)
        self.k = 0
        self.substep_cb = None  # 量測用：每個物理子步呼叫 cb(q_d, qd_d, tau)

    # ------------------------------------------------------------ 目標
    def goal_reachable(self, p) -> bool:
        sols = self.robot.ik(pose(R_DOWN, p), Q_SEED)
        if not sols:
            return False
        p05 = np.asarray(p) + self.robot.d[5] * np.array([0, 0, 1.0])  # 腕心（工具朝下）
        return bool(np.hypot(p05[0], p05[1]) >= 0.22)

    def sample_goal(self) -> np.ndarray:
        for _ in range(1000):
            p = self.np_random.uniform(self.goal_low, self.goal_high)
            if self.goal_reachable(p):
                return p
        raise RuntimeError("找不到可達目標")

    # ------------------------------------------------------------ 狀態
    def ee(self):
        s = pb.getLinkState(self.body, FLANGE, computeLinkVelocity=1, computeForwardKinematics=1,
                            physicsClientId=self.cid)
        return np.array(s[4]), np.array(s[6])

    def _obs(self):
        q, qd = self.pbr.get_state()
        p, _ = self.ee()
        o = np.concatenate([q / np.pi, qd / self.robot.qd_max, p, self.goal, self.goal - p, self.prev_a])
        return np.clip(o, -10.0, 10.0).astype(np.float32)

    def _collision(self) -> bool:
        # 基座與 link1（只繞 z 旋轉、外殼底面貼地）不列入地面碰撞
        if any(c[3] >= 1 for c in pb.getContactPoints(self.body, self.pbr.plane, physicsClientId=self.cid)):
            return True
        for i, j in self.self_pairs:
            if pb.getClosestPoints(self.body, self.body, 0.0, i, j, physicsClientId=self.cid):
                return True
        return False

    # ------------------------------------------------------------ Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        q0 = options.get("q0")
        if q0 is None:
            q0 = self.q_home + self.np_random.uniform(-self.q0_noise, self.q0_noise, 6)
        self.goal = np.asarray(options["goal"], float) if "goal" in options else self.sample_goal()
        q0 = np.asarray(q0, float)
        self.pbr.set_state(q0, np.zeros(6))
        self.q_ref = q0.copy()
        self.R0 = self.robot.fk(q0)[:3, :3]
        self.prev_a = np.zeros(6)
        self.k = 0
        self.ep_tau_ratio = 0.0
        if self.goal_marker is not None:
            pb.resetBasePositionAndOrientation(self.goal_marker, self.goal.tolist(), [0, 0, 0, 1],
                                               physicsClientId=self.cid)
        p, _ = self.ee()
        d = float(np.linalg.norm(self.goal - p))
        return self._obs(), {"distance": d, "goal": self.goal.copy(), "success": False}

    def joint_cmd(self, a) -> np.ndarray:
        """動作 → 關節速度指令 u（rad/s）。"""
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        if self.action_mode == "joint":
            return a * self.omega_max
        q, _ = self.pbr.get_state()
        T = self.robot.fk(q)
        tw = np.concatenate([a * self.v_max, 2.0 * rot_log(self.R0 @ T[:3, :3].T)])
        J = self.robot.jacobian(q)
        u = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), tw)
        s = np.max(np.abs(u)) / self.omega_max
        return u / s if s > 1.0 else u

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        u = self.joint_cmd(a)
        q, _ = self.pbr.get_state()
        q_prev = self.q_ref.copy()
        q_new = np.clip(self.q_ref + u * self.dt_c, q - self.e_max, q + self.e_max)
        q_new = np.clip(q_new, self.robot.q_min, self.robot.q_max)
        u_eff = (q_new - q_prev) / self.dt_c
        tau_ratio = 0.0
        tmax = self.robot.tau_max
        for s in range(self.n_sub):
            qd_ref = u_eff
            q_d = q_prev + u_eff * (s + 1) * self.dt
            q, qd = self.pbr.get_state()
            acc = self.kp * (q_d - q) + self.kd * (qd_ref - qd)
            tau = np.array(pb.calculateInverseDynamics(self.body, q.tolist(), qd.tolist(), acc.tolist(),
                                                       physicsClientId=self.cid))
            tau = np.clip(tau, -tmax, tmax)
            tau_ratio = max(tau_ratio, float(np.max(np.abs(tau) / tmax)))
            self.pbr.apply_torques(tau)
            pb.stepSimulation(physicsClientId=self.cid)
            if self.substep_cb is not None:
                self.substep_cb(q_d, qd_ref, tau)
        self.q_ref = q_new
        self.k += 1
        self.ep_tau_ratio = max(self.ep_tau_ratio, tau_ratio)
        self.prev_a = np.zeros(6)
        self.prev_a[: a.size] = a

        p, v = self.ee()
        d = float(np.linalg.norm(self.goal - p))
        speed = float(np.linalg.norm(v))
        success = d < self.tol and speed < self.v_tol
        collision = self._collision()
        if self.reward_type == "dense":
            r = -d - self.c_action * float(a @ a) + self.r_success * success
        else:
            r = 0.0 if success else -1.0
        r -= self.r_collision * collision
        terminated = bool(collision or (success and self.terminate_on_success))
        truncated = bool(self.k >= self.max_steps and not terminated)
        info = {"distance": d, "ee_speed": speed, "success": bool(success), "collision": bool(collision),
                "tau_ratio": tau_ratio, "tau": tau, "ee_pos": p}
        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.goal_marker is None:
            self.goal_marker = self.pbr.add_marker(self.goal, 0.02, (0.1, 0.85, 0.2, 0.9))
        return self.pbr.render(target=(0.25, 0.0, 0.3), distance=1.5, yaw=50, pitch=-22,
                               size=self.render_size, supersample=1)

    def close(self):
        self.pbr.close()


# ---------------------------------------------------------------- 基準策略（規劃/控制）
class PlannerPolicy:
    """傳統規劃基準：reset 後以封閉解 IK（工具朝下、離 q_ref 最近解）求目標關節角，
    建立關節五次多項式 PTP（T003；最短時間使峰值速度 = ω_max·speed），
    每步動作 a = (q_traj(t_{k+1}) − q_ref)/(Δt·ω_max) 讓參考精確沿軌跡（joint 模式）。"""

    def __init__(self, env: ReachEnv, speed: float = 0.9):
        self.env, self.speed = env, speed

    def reset(self):
        e = self.env
        sols = e.robot.ik(pose(R_DOWN, e.goal), e.q_ref)
        q1 = sols[0]
        T = min_duration(q1 - e.q_ref, np.full(6, e.omega_max * self.speed), kind="quintic")
        T = max(T, 2 * e.dt_c)
        self.traj = joint_ptp(e.q_ref.copy(), q1, T=T)
        self.k = 0

    def __call__(self, obs):
        e = self.env
        self.k += 1
        t = min(self.k * e.dt_c, self.traj.duration)
        q_t, _, _ = self.traj.state(t)
        return np.clip((q_t - e.q_ref) / (e.dt_c * e.omega_max), -1, 1)


class ResolvedRatePolicy:
    """笛卡兒比例控制基準（resolved-rate，Whitney 1969）：v = K_p(p_goal − p)（|v| ≤ v_max），
    姿態保持，DLS 換成關節速度。joint 模式輸出 u/ω_max；ee 模式直接輸出 v/v_max。"""

    def __init__(self, env: ReachEnv, kp: float = 2.5):
        self.env, self.kp = env, kp

    def reset(self):
        pass

    def __call__(self, obs):
        e = self.env
        p, _ = e.ee()
        v = self.kp * (e.goal - p)
        n = np.linalg.norm(v)
        if n > e.v_max:
            v *= e.v_max / n
        if e.action_mode == "ee":
            return np.clip(v / e.v_max, -1, 1)
        q, _ = e.pbr.get_state()
        T = e.robot.fk(q)
        tw = np.concatenate([v, 2.0 * rot_log(e.R0 @ T[:3, :3].T)])
        J = e.robot.jacobian(q)
        u = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), tw)
        return np.clip(u / e.omega_max, -1, 1)


class RandomPolicy:
    def __init__(self, env: ReachEnv, seed: int = 0):
        self.env, self.rng = env, np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, self.env.action_space.shape)


def run_episode(env: ReachEnv, policy, seed: int, record: bool = False, frames: bool = False):
    """執行一回合，回傳量測 dict（與 T202 學習策略比較的共同指標）。"""
    obs, info = env.reset(seed=seed)
    policy.reset()
    d0 = info["distance"]
    ret, path, t_succ, tau_ratio = 0.0, 0.0, None, 0.0
    p_prev = env.ee()[0]
    log = {k: [] for k in ("t", "q", "qd", "d", "a", "tau", "q_ref")} if record else None
    imgs = [env.render()] if frames else None
    terminated = truncated = False
    info = {"success": False, "collision": False, "distance": d0}
    while not (terminated or truncated):
        a = policy(obs)
        obs, r, terminated, truncated, info = env.step(a)
        ret += r
        p = info["ee_pos"]
        path += float(np.linalg.norm(p - p_prev))
        p_prev = p
        tau_ratio = max(tau_ratio, info["tau_ratio"])
        if info["success"] and t_succ is None:
            t_succ = env.k * env.dt_c
        if record:
            q, qd = env.pbr.get_state()
            for key, val in zip(log, (env.k * env.dt_c, q, qd, info["distance"], np.asarray(a, float),
                                      info["tau"], env.q_ref.copy()), strict=True):
                log[key].append(val)
        if frames:
            imgs.append(env.render())
    out = {
        "seed": seed, "d0": d0, "success": bool(info["success"]), "collision": bool(info["collision"]),
        "d_final": float(info["distance"]), "t_success": t_succ, "steps": env.k, "return": ret,
        "path": path, "path_ratio": path / max(d0, 1e-9), "tau_ratio": tau_ratio,
    }
    if record:
        out["log"] = {k: np.array(v) for k, v in log.items()}
    if frames:
        out["frames"] = imgs
    return out


def throughput(env: ReachEnv, n: int = 500, seed: int = 0) -> float:
    """隨機動作下每秒環境步數（含物理子步）。"""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
        if term or trunc:
            env.reset()
    return n / (time.perf_counter() - t0)


gym.register(id="SixAxisReach-v0", entry_point="sixaxis.rlenv:ReachEnv", max_episode_steps=None)
