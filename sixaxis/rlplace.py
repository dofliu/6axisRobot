"""T204 Gymnasium 環境：place／完整 pick-and-place（抓起桌上方塊、搬到目標位置放下、鬆爪退離）。

場景與動作沿用 T203 `PickEnv`（手臂＋T005 平行夾爪＋工作台＋50 mm 方塊；4 維動作 = TCP 線速度＋夾爪開閉；
工具朝下姿態保持＋DLS → q_ref → 8 自由度計算力矩內環）。另加放置目標 g（台面上，xy 均勻隨機且與物件起點
相距 ≥ d_min；只有視覺標記，無碰撞）。目標條件化（goal-conditioned）觀測：
T203 的 30 維＋g＋(p_obj − g) = 36 維。

放置判據（全部成立才算「在目標」at_goal）：
  ‖p_obj,xy − g_xy‖ < tol_xy（預設 2 cm）、|z_obj − z_rest| < tol_z（1 cm，即靜置在台面上）、
  物件傾角 θ = acos(ẑ_obj·ẑ) < tilt_max（10°）。
success = at_goal ∧ 夾爪指令為張開 ∧ 兩指墊皆無接觸（已鬆開退離，物件靠自身靜置）。
成功「不終止」（同 T203），回合成功 = 最後一步仍 success（放好且未被再次碰動）。

獎勵（分階段取最大值，依 robosuite PickPlace 的 staged reward：r = max(r_reach, r_grasp, r_lift, r_hover)，
Zhu et al. 2020；取 max 而非加總可避免各階段獎勵互相抵消、並保證「越後面的階段獎勵越高」）：
  r_reach = 1 − tanh(10 d)                                d = ‖p_obj − p_tcp‖               ∈ [0, 1]
  r_carry = [grasped]·(1.5 + 2.5(1 − tanh(5 e)))          e = ‖p_obj − (g_xy, z_rest)‖      ∈ [1.5, 4]
  r_rel   = 4.5·[at_goal ∧ ¬grasped]                      已在目標但尚未完全退離（鬆爪過渡）
  r       = R_s（= 6）若 success，否則 max(r_reach, r_carry, r_rel)
            − c_a‖a_xyz‖² − R_coll·[碰撞] − R_drop·[掉落]
各階段上界單調遞增（1 < 1.5…4 < 4.5 < 6），因此最佳策略為 抓 → 搬 → 放 → 鬆 → 保持。

示範（demonstration）：`ScriptedPlacePolicy` 為 T203 腳本 FSM 的延伸（APPROACH → DESCEND → CLOSE → LIFT →
CARRY → LOWER → OPEN → RETREAT），在 SAC 訓練前以加噪示範預填回放緩衝（DDPGfD：Vecerik et al. 2017；
Nair et al. 2018 "Overcoming Exploration in RL with Demonstrations"），見 `rltrain.train(demo_episodes=…)`。

放置精度的物理模型（報告中驗證）：放下時若物件底面離台面 h_rel，鬆爪後自由落體 t_f = √(2h_rel/g)、
著地速度 v = √(2 g h_rel)；指墊張開時兩指接觸力不同時消失，切向摩擦力的衝量使物件產生側向位移，
故放置誤差 = TCP 對位誤差 + 夾持偏移 + 鬆爪/落下擾動。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pybullet as pb
from gymnasium import spaces

from .rlpick import TABLE_TOP, PickEnv

GOAL_LOW = np.array([0.35, -0.20])
GOAL_HIGH = np.array([0.60, 0.20])


def tilt_angle(quat) -> float:
    """物件 z 軸與世界 z 軸夾角 [rad]（方塊任一面朝上皆視為正立：取 |cos| 的三軸最大值）。"""
    R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    c = float(np.max(np.abs(R[2, :])))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


class PlaceEnv(PickEnv):
    def __init__(
        self,
        max_steps: int = 160,
        goal_low=GOAL_LOW,
        goal_high=GOAL_HIGH,
        d_min: float = 0.12,
        tol_xy: float = 0.02,
        tol_z: float = 0.01,
        tilt_max: float = np.deg2rad(10.0),
        r_success: float = 6.0,
        h_lift: float = 0.05,
        **kw,
    ):
        # h_lift 只用於 info["lifted"]（抓住且抬升 ≥ h_lift）與 t_lift 統計，不進入獎勵
        super().__init__(max_steps=max_steps, h_lift=h_lift, **kw)
        self.goal_low, self.goal_high = np.asarray(goal_low, float), np.asarray(goal_high, float)
        self.d_min, self.tol_xy, self.tol_z, self.tilt_max = d_min, tol_xy, tol_z, tilt_max
        self.r_success = r_success
        self.observation_space = spaces.Box(-10.0, 10.0, (36,), np.float32)
        he = [0.03, 0.03, 0.0005]
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[0.15, 0.75, 0.3, 0.55],
                                   physicsClientId=self.cid)
        self.marker = pb.createMultiBody(0, -1, vis, [0.5, 0.0, TABLE_TOP + 0.0006], physicsClientId=self.cid)
        self.goal = np.array([0.5, 0.0, self.z_obj0])

    # ------------------------------------------------------------ 目標
    def sample_goal(self, obj_xy) -> np.ndarray:
        for _ in range(1000):
            g = self.np_random.uniform(self.goal_low, self.goal_high)
            if np.linalg.norm(g - obj_xy) >= self.d_min:
                return g
        raise RuntimeError("無法取樣目標")

    def _place_state(self, m):
        e_xy = float(np.linalg.norm(m["po"][:2] - self.goal[:2]))
        e_z = float(abs(m["po"][2] - self.z_obj0))
        tilt = tilt_angle(m["qo"])
        at_goal = bool(e_xy < self.tol_xy and e_z < self.tol_z and tilt < self.tilt_max)
        released = bool(self.grip_cmd == 0 and not any(f > 0.0 for f in m["fc"]))
        return {"e_xy": e_xy, "e_z": e_z, "tilt": tilt, "at_goal": at_goal, "released": released,
                "success": bool(at_goal and released),
                "e_goal": float(np.linalg.norm(m["po"] - self.goal))}

    def _obs(self, m):
        o = super()._obs(m)
        extra = np.concatenate([self.goal, m["po"] - self.goal])
        return np.concatenate([o, np.clip(extra, -10, 10).astype(np.float32)])

    # ------------------------------------------------------------ Gym API
    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        gym.Env.reset(self, seed=seed)
        if "obj_xy" not in options:
            options["obj_xy"] = self.sample_obj()
        if "goal_xy" in options:
            g = np.asarray(options["goal_xy"], float)
        else:
            g = self.sample_goal(options["obj_xy"])
        self.goal = np.array([g[0], g[1], self.z_obj0])
        pb.resetBasePositionAndOrientation(self.marker, [g[0], g[1], TABLE_TOP + 0.0006], [0, 0, 0, 1],
                                           physicsClientId=self.cid)
        # PickEnv.reset 內的 super().reset(seed=None) 不會重設亂數，q0 由同一亂數流取樣 → 決定性
        obs, info = super().reset(seed=None, options=options)
        ps = self._place_state(self.last)
        info.update({"goal": self.goal.copy(), **ps, "success": False})
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        m = self.last
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        ps = self._place_state(m)
        if ps["success"]:
            r = self.r_success
        else:
            r_reach = 1.0 - np.tanh(10.0 * m["d"])
            r_carry = m["grasped"] * (1.5 + 2.5 * (1.0 - np.tanh(5.0 * ps["e_goal"])))
            r_rel = 4.5 * (ps["at_goal"] and not m["grasped"])
            r = max(r_reach, r_carry, r_rel)
        r = float(r - self.c_action * float(a[:3] @ a[:3]) - self.r_collision * info["collision"]
                  - self.r_drop * info["drop"])
        info.update({"lifted": info["success"], **ps})  # PickEnv 的 success（抓住且抬升）改名 lifted
        info["success"] = ps["success"]
        return obs, r, terminated, truncated, info


# ---------------------------------------------------------------- 傳統基準：腳本式取放
class ScriptedPlacePolicy:
    """教導式取放有限狀態機（T203 ScriptedPickPolicy 延伸）：
    APPROACH（物件上方 z_app）→ DESCEND → CLOSE（停 n_close 步）→ LIFT（抬 h_carry）→ CARRY（到目標上方）
    → LOWER（物件底面離台面 h_rel）→ OPEN（停 n_open 步）→ RETREAT（上升 z_app）。
    各段為笛卡兒比例控制 v = K_p(p_t − p)，|v| ≤ v_max（resolved-rate）；完美感知（讀真實物件位置）。
    noise > 0 時對速度指令加高斯雜訊（產生多樣化示範供回放緩衝）。"""

    def __init__(self, env: PlaceEnv, kp: float = 4.0, z_app: float = 0.08, tol: float = 0.006,
                 n_close: int = 6, n_open: int = 5, h_carry: float = 0.08, h_rel: float = 0.002,
                 noise: float = 0.0, seed: int = 0):
        self.env, self.kp, self.z_app, self.tol = env, kp, z_app, tol
        self.n_close, self.n_open, self.h_carry, self.h_rel = n_close, n_open, h_carry, h_rel
        self.noise, self.rng = noise, np.random.default_rng(seed)

    def reset(self):
        self.phase, self.n = "approach", 0

    def __call__(self, obs):
        e = self.env
        m = e.last
        p, po, g = m["p"], m["po"], e.goal
        grip = -1.0
        if self.phase == "approach":
            tgt = po + [0, 0, self.z_app]
            if np.linalg.norm(tgt - p) < 2 * self.tol:
                self.phase = "descend"
        if self.phase == "descend":
            tgt = po.copy()
            if np.linalg.norm(tgt - p) < self.tol:
                self.phase, self.n = "close", 0
        if self.phase == "close":
            tgt, grip = p.copy(), 1.0
            self.n += 1
            if self.n >= self.n_close:
                self.phase, self.off = "lift", p - po  # 夾持偏移（TCP 相對物件）
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
        return np.array([*a, grip])


class RandomPlacePolicy:
    def __init__(self, env: PlaceEnv, seed: int = 0):
        self.env, self.rng = env, np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, 4)


def run_place_episode(env: PlaceEnv, policy, seed: int, record: bool = False, frames: bool = False,
                      options: dict | None = None, frame_every: int = 1):
    """執行一回合（跑滿 max_steps 或終止），回傳取放指標。
    success = 最後一步仍 success；t_grasp/t_lift/t_place/t_success =
    首次抓住／抬升 ≥ h_lift／at_goal／success 的時間；
    e_xy_final = 最後物件中心與目標水平距離。"""
    obs, info = env.reset(seed=seed, options=options)
    policy.reset()
    goal = env.goal.copy()
    obj0 = env.obj_xy0.copy()
    ret, tau_ratio = 0.0, 0.0
    t_grasp = t_lift = t_place = t_succ = None
    n_succ = 0
    keys = ("t", "q", "qd", "tau", "tcp", "obj", "a", "fc", "grasped", "s", "e_goal", "r")
    log = {k: [] for k in keys} if record else None
    imgs = [env.render()] if frames else None
    terminated = truncated = False
    while not (terminated or truncated):
        a = np.asarray(policy(obs), float)
        obs, r, terminated, truncated, info = env.step(a)
        ret += r
        tau_ratio = max(tau_ratio, info["tau_ratio"])
        t = env.k * env.dt_c
        if info["grasped"] and t_grasp is None:
            t_grasp = t
        if info["lifted"] and t_lift is None:
            t_lift = t
        if info["at_goal"] and t_place is None:
            t_place = t
        if info["success"]:
            n_succ += 1
            if t_succ is None:
                t_succ = t
        if record:
            q, qd = env.pbr.get_state()
            vals = (t, q, qd, info["tau"], info["tcp"], info["obj"], a, info["fc"], info["grasped"],
                    float(np.mean(env.g.finger_pos())), info["e_goal"], r)
            for key, val in zip(log, vals, strict=True):
                log[key].append(val)
        if frames and env.k % frame_every == 0:
            imgs.append(env.render())
    out = {
        "seed": seed, "success": bool(info["success"]), "collision": bool(info["collision"]),
        "drop": bool(info["drop"]), "tipped": bool(info["tilt"] >= env.tilt_max and not info["drop"]),
        "t_grasp": t_grasp, "t_lift": t_lift, "t_place": t_place, "t_success": t_succ, "steps": env.k,
        "return": ret, "e_xy_final": info["e_xy"], "e_z_final": info["e_z"], "tilt_final": info["tilt"],
        "hold_frac": n_succ / max(env.k, 1), "tau_ratio": tau_ratio, "obj_xy": obj0.tolist(),
        "goal_xy": goal[:2].tolist(), "dist": float(np.linalg.norm(goal[:2] - obj0)),
    }
    if record:
        out["log"] = {k: np.array(v) for k, v in log.items()}
    if frames:
        out["frames"] = imgs
    return out


def collect_demos(env: PlaceEnv, n_episodes: int, noise: float = 0.15, seed0: int = 20_000,
                  success_only: bool = True):
    """以加噪腳本示範收集轉移 (o, a, r, o', terminated, truncated)；預設只保留成功回合。"""
    out, n_ok = [], 0
    k = 0
    while n_ok < n_episodes and k < 4 * n_episodes:
        pol = ScriptedPlacePolicy(env, noise=noise, seed=seed0 + k)
        obs, _ = env.reset(seed=seed0 + k)
        pol.reset()
        ep, term, trunc, info = [], False, False, {}
        while not (term or trunc):
            a = pol(obs)
            nobs, r, term, trunc, info = env.step(a)
            ep.append((obs, a.astype(np.float32), float(r), nobs, term, trunc))
            obs = nobs
        k += 1
        if info.get("success") or not success_only:
            out.extend(ep)
            n_ok += 1
    return out, n_ok, k


gym.register(id="SixAxisPlace-v0", entry_point="sixaxis.rlplace:PlaceEnv", max_episode_steps=None)
