"""T203 Gymnasium 環境：pick（夾爪抓起桌上方塊並抬高、保持）。

場景：PyBullet（DIRECT）UR5 類手臂＋T005 平行夾爪（同一 multibody，私有暫存 URDF，可多行程平行）＋地面＋工作台
（頂面 z = 0.10 m）＋50 mm 方塊（0.2 kg，μ_obj = 0.8，指墊 μ = 1.0 → PyBullet 有效 μ = 0.8）。
物件位置於台面方塊區均勻隨機（姿態固定 yaw = 0；隨機姿態留待 T205）。

動作（4 維，[−1,1]）：a[:3] 為夾持中心（TCP）線速度指令 v = a·v_max；工具姿態保持朝下（R_DOWN，
ω = k_R·log(R_d Rᵀ)）；以阻尼最小平方換成關節速度（同 T201 ee 模式）→ q_ref 積分 → 計算力矩內環
τ = ID(q, q̇, K_p e + K_d ė)（含夾爪與手指的 8 自由度 PyBullet 逆動力學）。
a[3] > 0 → 夾爪以力上限 F_grip 閉合（T005 位置馬達＋力上限，接觸後法向力 ≈ F_grip），否則張開。

觀測（30 維）：p_tcp、ṗ_tcp、p_obj、p_obj − p_tcp、物件四元數、ṗ_obj、指開度 s/S_MAX、左右指接觸旗標、
夾爪指令狀態、q/π、物件抬升高度 h（m）。

獎勵（分階段 dense，依 robosuite Lift 任務的分段獎勵：Zhu et al. 2020, "robosuite: A Modular Simulation
Framework and Benchmark for Robot Learning"）：
  r = (1 − tanh(10·d)) + w_g·[grasped] + w_l·[grasped]·min(h/h_lift, 1) + w_s·[success]
      − c_a‖a_xyz‖² − R_coll·[碰撞] − R_drop·[掉落]
d = ‖p_obj − p_tcp‖；grasped = 兩指墊皆與物件接觸（法向力 > 0.5 N）且夾爪指令為閉合；
success = grasped 且 h ≥ h_lift（預設 0.10 m）。依 T202 發現（成功即終止使「到達後」行為未受訓練），
本環境成功「不終止」，跑滿 max_steps；回合成功 = 最後一步仍 success（抓住並保持）。
終止：手臂連桿（link1–6）碰地或碰台、物件掉出台面（z < 台面 − 3 cm）。

抓持力學：保持需要 2μF_n ≥ m(g + a)（T005 `f_min`）；F_grip = 40 N 遠大於 0.2 kg 方塊的 1.2 N，
抓取失敗主要來自幾何（未對中、指墊壓到物件頂面、推倒物件）而非摩擦不足。
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as pb
from gymnasium import spaces

from .gripper import CUBE, FINGER_LINKS, S_MAX, T_TCP, Gripper, ObjectSpec, make_arm_with_gripper
from .pbsim import FLANGE
from .rlenv import Q_SEED, R_DOWN, pose
from .robot import SixAxisRobot, rot_log

TABLE_C = np.array([0.475, 0.0])  # 台面中心 (x, y)
TABLE_HALF = np.array([0.225, 0.32])
TABLE_TOP = 0.10
OBJ_LOW = np.array([0.35, -0.20])  # 物件中心取樣範圍 (x, y)
OBJ_HIGH = np.array([0.60, 0.20])
HOME_TCP = np.array([0.45, 0.0, 0.32])
TCP_OFF = T_TCP[2, 3]  # 夾持中心離法蘭（沿工具 z）


def tcp_pose(p_tcp) -> np.ndarray:
    """工具朝下、夾持中心在 p_tcp 的法蘭位姿。"""
    return pose(R_DOWN, np.asarray(p_tcp, float) - R_DOWN @ T_TCP[:3, 3])


class PickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        control_hz: float = 20.0,
        sim_hz: float = 500.0,
        solver_iters: int = 50,
        max_steps: int = 100,
        v_max: float = 0.30,
        omega_max: float = 1.5,
        wn: float = 25.0,
        e_max: float = 0.30,
        k_rot: float = 2.0,
        f_grip: float = 40.0,
        h_lift: float = 0.10,
        q0_noise: float = 0.10,
        w_grasp: float = 0.5,
        w_lift: float = 2.0,
        w_success: float = 2.0,
        c_action: float = 0.01,
        r_collision: float = 10.0,
        r_drop: float = 10.0,
        obj: ObjectSpec = CUBE,
        obj_low=OBJ_LOW,
        obj_high=OBJ_HIGH,
        render_mode: str | None = None,
        render_size=(400, 300),
    ):
        self.dt_c, self.dt = 1.0 / control_hz, 1.0 / sim_hz
        self.n_sub = int(round(sim_hz / control_hz))
        self.max_steps, self.v_max, self.omega_max, self.e_max = max_steps, v_max, omega_max, e_max
        self.kp, self.kd, self.k_rot = wn**2, 2.0 * wn, k_rot
        self.f_grip, self.h_lift, self.q0_noise = f_grip, h_lift, q0_noise
        self.w_grasp, self.w_lift, self.w_success = w_grasp, w_lift, w_success
        self.c_action, self.r_collision, self.r_drop = c_action, r_collision, r_drop
        self.part = obj
        self.obj_low, self.obj_high = np.asarray(obj_low, float), np.asarray(obj_high, float)
        self.render_mode, self.render_size = render_mode, render_size

        self.robot = SixAxisRobot()
        self._td = tempfile.TemporaryDirectory()
        self.pbr = make_arm_with_gripper(self.robot, plane=True, urdf_path=Path(self._td.name) / "arm.urdf")
        self.cid, self.body = self.pbr.cid, self.pbr.id
        self.pbr.make_frictionless()
        pb.setTimeStep(self.dt, physicsClientId=self.cid)
        pb.setPhysicsEngineParameter(numSolverIterations=solver_iters, physicsClientId=self.cid)
        self.g = Gripper(self.pbr)
        he = [TABLE_HALF[0], TABLE_HALF[1], TABLE_TOP / 2]
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.cid)
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[0.55, 0.57, 0.62, 1],
                                   physicsClientId=self.cid)
        self.table = pb.createMultiBody(0, col, vis, [TABLE_C[0], TABLE_C[1], TABLE_TOP / 2],
                                        physicsClientId=self.cid)
        pb.changeDynamics(self.table, -1, lateralFriction=1.0, physicsClientId=self.cid)
        self.z_obj0 = TABLE_TOP + obj.height / 2
        self.obj = self.g.spawn(obj, [TABLE_C[0], 0.0, self.z_obj0])
        self.q_home = self.robot.ik(tcp_pose(HOME_TCP), Q_SEED)[0]

        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (30,), np.float32)
        self.q_ref = self.q_home.copy()
        self.grip_cmd = 0.0
        self.k = 0
        self.substep_cb = None

    # ------------------------------------------------------------ 狀態
    def tcp(self):
        s = pb.getLinkState(self.body, FLANGE, computeLinkVelocity=1, computeForwardKinematics=1,
                            physicsClientId=self.cid)
        R = np.array(pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)
        r = R @ T_TCP[:3, 3]
        w = np.array(s[7])
        return np.array(s[4]) + r, np.array(s[6]) + np.cross(w, r), R

    def obj_state(self):
        p, qt = pb.getBasePositionAndOrientation(self.obj, physicsClientId=self.cid)
        v, _ = pb.getBaseVelocity(self.obj, physicsClientId=self.cid)
        return np.array(p), np.array(qt), np.array(v)

    def finger_contacts(self):
        out = []
        for lk in FINGER_LINKS:
            cs = pb.getContactPoints(self.body, self.obj, lk, -1, physicsClientId=self.cid)
            out.append(float(sum(c[9] for c in cs)))
        return out

    def _measure(self):
        p, v, R = self.tcp()
        po, qo, vo = self.obj_state()
        fc = self.finger_contacts()
        contact = [f > 0.5 for f in fc]
        grasped = bool(all(contact) and self.grip_cmd > 0)
        h = float(po[2] - self.z_obj0)
        return {"p": p, "v": v, "R": R, "po": po, "qo": qo, "vo": vo, "fc": fc, "contact": contact,
                "grasped": grasped, "h": h, "d": float(np.linalg.norm(po - p))}

    def _obs(self, m):
        q, _ = self.pbr.get_state()
        s = float(np.mean(self.g.finger_pos())) / S_MAX
        o = np.concatenate([m["p"], m["v"], m["po"], m["po"] - m["p"], m["qo"], m["vo"], [s],
                            np.array(m["contact"], float), [self.grip_cmd], q / np.pi, [m["h"]]])
        return np.clip(o, -10.0, 10.0).astype(np.float32)

    def _collision(self) -> bool:
        for other in (self.pbr.plane, self.table):
            cs = pb.getContactPoints(self.body, other, physicsClientId=self.cid)
            if any(1 <= c[3] <= FLANGE for c in cs):
                return True
        return False

    # ------------------------------------------------------------ Gym API
    def sample_obj(self) -> np.ndarray:
        return self.np_random.uniform(self.obj_low, self.obj_high)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        xy = np.asarray(options["obj_xy"], float) if "obj_xy" in options else self.sample_obj()
        q0 = options.get("q0")
        if q0 is None:
            q0 = self.q_home + self.np_random.uniform(-self.q0_noise, self.q0_noise, 6)
        q0 = np.asarray(q0, float)
        self.pbr.set_state(q0, np.zeros(6))
        for lk in FINGER_LINKS:
            pb.resetJointState(self.body, lk, 0.0, 0.0, physicsClientId=self.cid)
        self.g.open()
        self.obj_xy0 = xy.copy()
        # 每回合重建物件：清除上一回合留下的接觸流形與暖啟動衝量，使相同種子＋動作序列逐位元重現
        pb.removeBody(self.obj, physicsClientId=self.cid)
        self.obj = self.g.spawn(self.part, [xy[0], xy[1], self.z_obj0])
        self.q_ref = q0.copy()
        self.grip_cmd = 0.0
        self.k = 0
        self.ep_tau_ratio = 0.0
        self.prev_a = np.zeros(4)
        m = self._measure()
        self.last = m
        return self._obs(m), {"distance": m["d"], "obj_xy": xy.copy(), "success": False, "grasped": False,
                              "h": m["h"]}

    def joint_cmd(self, v) -> np.ndarray:
        """TCP 線速度 v → 關節速度（姿態保持朝下；DLS；超過 ω_max 等比縮放）。"""
        q, _ = self.pbr.get_state()
        T = self.robot.fk(q)
        R = T[:3, :3]
        w = self.k_rot * rot_log(R_DOWN @ R.T)
        r = R @ T_TCP[:3, 3]
        v_fl = v - np.cross(w, r)  # 法蘭速度 = TCP 速度 − ω × r
        J = self.robot.jacobian(q)
        u = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), np.concatenate([v_fl, w]))
        s = np.max(np.abs(u)) / self.omega_max
        return u / s if s > 1.0 else u

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        u = self.joint_cmd(a[:3] * self.v_max)
        new_grip = 1.0 if a[3] > 0 else 0.0
        if new_grip != self.grip_cmd:
            if new_grip > 0:
                self.g.close(self.f_grip)
            else:
                self.g.open()
            self.grip_cmd = new_grip
        q, _ = self.pbr.get_state()
        q_prev = self.q_ref.copy()
        q_new = np.clip(self.q_ref + u * self.dt_c, q - self.e_max, q + self.e_max)
        q_new = np.clip(q_new, self.robot.q_min, self.robot.q_max)
        u_eff = (q_new - q_prev) / self.dt_c
        tmax = self.robot.tau_max
        tau_ratio = 0.0
        for s in range(self.n_sub):
            q_d = q_prev + u_eff * (s + 1) * self.dt
            q, qd = self.pbr.get_state()
            js = pb.getJointStates(self.body, FINGER_LINKS, physicsClientId=self.cid)
            acc = self.kp * (q_d - q) + self.kd * (u_eff - qd)
            tau = pb.calculateInverseDynamics(
                self.body, [*q.tolist(), js[0][0], js[1][0]], [*qd.tolist(), js[0][1], js[1][1]],
                [*acc.tolist(), 0.0, 0.0], physicsClientId=self.cid)
            tau = np.clip(np.asarray(tau[:6]), -tmax, tmax)
            tau_ratio = max(tau_ratio, float(np.max(np.abs(tau) / tmax)))
            self.pbr.apply_torques(tau)
            pb.stepSimulation(physicsClientId=self.cid)
            if self.substep_cb is not None:
                self.substep_cb(self)
        self.q_ref = q_new
        self.k += 1
        self.ep_tau_ratio = max(self.ep_tau_ratio, tau_ratio)
        self.prev_a = a

        m = self._measure()
        self.last = m
        success = bool(m["grasped"] and m["h"] >= self.h_lift)
        collision = self._collision()
        drop = bool(m["po"][2] < TABLE_TOP - 0.03)
        lift = min(max(m["h"], 0.0) / self.h_lift, 1.0)
        r = (1.0 - np.tanh(10.0 * m["d"]) + self.w_grasp * m["grasped"]
             + self.w_lift * m["grasped"] * lift + self.w_success * success
             - self.c_action * float(a[:3] @ a[:3]) - self.r_collision * collision - self.r_drop * drop)
        terminated = bool(collision or drop)
        truncated = bool(self.k >= self.max_steps and not terminated)
        info = {"distance": m["d"], "h": m["h"], "grasped": m["grasped"], "success": success,
                "collision": collision, "drop": drop, "tau_ratio": tau_ratio, "tau": tau, "tcp": m["p"],
                "obj": m["po"], "fc": m["fc"]}
        return self._obs(m), float(r), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self.pbr.render(target=(0.42, 0.0, 0.28), distance=1.45, yaw=55, pitch=-24,
                               size=self.render_size, supersample=1)

    def close(self):
        self.pbr.close()
        self._td.cleanup()


# ---------------------------------------------------------------- 傳統基準：腳本式取物
class ScriptedPickPolicy:
    """傳統有限狀態機基準（教導式取物）：
    APPROACH（TCP 到物件上方 z_app，張開）→ DESCEND（下到物件中心）→ CLOSE（停住閉合 n_close 步）
    → LIFT（抬到 h_lift + 2 cm）。各段為笛卡兒比例控制 v = K_p(p_t − p)（|v| ≤ v_max，
    resolved-rate，Whitney 1969），讀取真實物件位置（完美感知）。"""

    def __init__(self, env: PickEnv, kp: float = 4.0, z_app: float = 0.08, tol: float = 0.006,
                 n_close: int = 6):
        self.env, self.kp, self.z_app, self.tol, self.n_close = env, kp, z_app, tol, n_close

    def reset(self):
        self.phase, self.n = "approach", 0

    def __call__(self, obs):
        e = self.env
        m = e.last
        p, po = m["p"], m["po"]
        grip = -1.0
        if self.phase == "approach":
            tgt = po + [0, 0, self.z_app]
            if np.linalg.norm(tgt - p) < 2 * self.tol:
                self.phase = "descend"
        if self.phase == "descend":
            tgt = po.copy()
            if np.linalg.norm(tgt - p) < self.tol:
                self.phase = "close"
        if self.phase == "close":
            tgt, grip = p.copy(), 1.0
            self.n += 1
            if self.n >= self.n_close:
                self.phase = "lift"
                self.lift_xy = p[:2].copy()
        if self.phase == "lift":
            tgt, grip = np.array([*self.lift_xy, e.z_obj0 + e.h_lift + 0.02]), 1.0
        v = self.kp * (tgt - p)
        n = np.linalg.norm(v)
        if n > e.v_max:
            v *= e.v_max / n
        return np.array([*np.clip(v / e.v_max, -1, 1), grip])


class RandomPickPolicy:
    def __init__(self, env: PickEnv, seed: int = 0):
        self.env, self.rng = env, np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, 4)


def run_pick_episode(env: PickEnv, policy, seed: int, record: bool = False, frames: bool = False,
                     options: dict | None = None):
    """執行一回合（跑滿 max_steps 或終止），回傳共同指標。
    success = 最後一步仍抓住且抬升 ≥ h_lift；t_success = 首次 success 時間；t_grasp = 首次 grasped 時間。"""
    obs, info = env.reset(seed=seed, options=options)
    policy.reset()
    ret, t_succ, t_grasp, n_succ = 0.0, None, None, 0
    tau_ratio, h_max, regrasp, prev_g = 0.0, 0.0, 0, False
    keys = ("t", "q", "qd", "tau", "tcp", "obj", "h", "d", "a", "fc", "grasped", "s")
    log = {k: [] for k in keys} if record else None
    imgs = [env.render()] if frames else None
    terminated = truncated = False
    info = {"success": False, "collision": False, "drop": False, "grasped": False, "h": 0.0}
    while not (terminated or truncated):
        a = np.asarray(policy(obs), float)
        obs, r, terminated, truncated, info = env.step(a)
        ret += r
        tau_ratio = max(tau_ratio, info["tau_ratio"])
        h_max = max(h_max, info["h"])
        t = env.k * env.dt_c
        if info["grasped"] and t_grasp is None:
            t_grasp = t
        if info["grasped"] and not prev_g and t_grasp is not None and t_grasp < t:
            regrasp += 1
        prev_g = info["grasped"]
        if info["success"]:
            n_succ += 1
            if t_succ is None:
                t_succ = t
        if record:
            q, qd = env.pbr.get_state()
            vals = (t, q, qd, info["tau"], info["tcp"], info["obj"], info["h"], info["distance"], a,
                    info["fc"], info["grasped"], float(np.mean(env.g.finger_pos())))
            for key, val in zip(log, vals, strict=True):
                log[key].append(val)
        if frames:
            imgs.append(env.render())
    out = {
        "seed": seed, "success": bool(info["success"]), "collision": bool(info["collision"]),
        "drop": bool(info["drop"]), "t_success": t_succ, "t_grasp": t_grasp, "steps": env.k, "return": ret,
        "h_final": float(info["h"]), "h_max": h_max, "hold_frac": n_succ / max(env.k, 1), "regrasp": regrasp,
        "tau_ratio": tau_ratio, "obj_xy": env.obj_xy0.tolist(),
    }
    if record:
        out["log"] = {k: np.array(v) for k, v in log.items()}
    if frames:
        out["frames"] = imgs
    return out


def throughput(env: PickEnv, n: int = 300, seed: int = 0) -> float:
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 4))
        if term or trunc:
            env.reset()
    return n / (time.perf_counter() - t0)


gym.register(id="SixAxisPick-v0", entry_point="sixaxis.rlpick:PickEnv", max_episode_steps=None)
