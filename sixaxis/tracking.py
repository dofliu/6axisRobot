"""T104 輸送帶追蹤抓取：移動工件的位置/速度估測、TCP 同步追蹤（CLIK）、動態夾取與放入料盤。

流程（單一工件）
1. 手臂在等待點（輸送帶上游、懸停高度）待命；工件通過光電開關（x = X_DET）時取得第一筆位置。
2. 估測器預測工件位置 x̂(t)、速度 v̂：
   * ``open``  開迴路：只用光電觸發的一筆位置 + 標稱帶速 v_nom → x̂ = x_det + v_nom·(t − t_det)；
   * ``track`` 閉迴路：相機以 CAM_DT 週期量測（含高斯雜訊），以 α-β 濾波器更新位置與速度。
3. 同步（T_SYNC）：TCP 由靜止以五次時間律 s(τ) 混入工件運動  p = p_w + s·(p̂ − p_w)，
   結束時位置/速度皆與工件一致（ṗ = ṡ·(p̂ − p_w) + s·v̂ → v̂）。
4. 追蹤下降（T_DESC）→ 追蹤中閉合夾爪（兩指法向力 ≥ 0.5·F 持續 0.1 s 即完成，最長 T_GRIP_MAX）→
   抬升（T_LIFT）同時 x 向速度以 (1 − s) 漸減到 0（行程 v̂·T_LIFT/2）。
5. 抬升後量測手內偏移 d（工件中心 − TCP），放置時以 TCP = 料盤格 − d 補償，再回等待點。

追蹤窗（可行性判據）：自觸發同步到抬升結束，TCP 沿帶向行進
    L(v) = v·(T_SYNC/2 + T_DESC + t_grip + T_LIFT/2)，
須滿足 x_w + L(v) ≤ X_END_MAX（擋料板前的安全位置）。同步在 x̂ = x_w − v·T_SYNC/2 觸發；帶速高到觸發點落在
光電開關上游時只能在光電處觸發，x_end = X_DET + v·(T_SYNC + T_DESC + t_grip + T_LIFT/2)（見 `x_end_pred`）。
估測超出追蹤窗時不追（回報 WINDOW），可改在擋料板前靜態抓取（`static_pick`）。

依據
* α-β 濾波器（等速目標的穩態卡爾曼濾波）：預測 x⁻ = x + v·T；殘差 r = z − x⁻；x = x⁻ + α·r；v = v + (β/T)·r。
  對等速目標穩態誤差為 0（Kalata, "The tracking index", IEEE Trans. AES 20(2), 1984；
  Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation*, 2001, §6.5）。
* 開迴路預測誤差：帶速相對誤差 ε 時，e(t) = ε·v_nom·(t − t_det)（等速運動直接積分）。
* 閉迴路逆運動學 CLIK：q̇ = J⁻¹(q)·[ṗ_d + K·(p_d − p(q)); K·e_o]，e_o = log(R_d·Rᵀ)；誤差動態 ė + K·e = 0
  （Siciliano, Sciavicco, Villani & Oriolo, *Robotics: Modelling, Planning and Control*, 2009, §3.7）。
  q̇ 積分得 q_cmd，再交給 T005 的計算力矩控制器；q̈ 前饋以 q̇ 一階差分取得。
* 輸送帶為庫侖限幅的速度追隨模型（`scene.Workcell.pre_step`）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as pb

from . import scene as S
from .commands import Executor
from .graspsim import R_DOWN
from .gripper import CUBE, T_TCP
from .robot import rot_log
from .tasks import Z_CLEAR, Z_OBJ, Z_POCKET, _tilt_yaw, pick_place_program, transfer_program

BELT_Y = (S.BELT_Y0 + S.BELT_Y1) / 2  # 帶中心線 y
X_FEED = 0.18  # 上料位置（工件中心）
X_DET = 0.20  # 光電開關
X_WAIT = 0.23  # 等待點（TCP）
Z_HOVER = Z_OBJ + 0.06  # 懸停高度（指尖離工件頂面 15 mm）
X_END_MAX = 0.46  # 抬升結束時 TCP 的最大 x（工件前緣 0.485 < 擋料板 0.52）
T_SYNC, T_DESC, T_LIFT = 0.30, 0.30, 0.50
T_GRIP_MAX, T_GRIP_NOM = 0.5, 0.15  # 夾爪閉合最長時間；可行性判斷用的標稱閉合時間
CAM_DT = 0.02  # 相機量測週期 [s]
ALPHA, BETA = 0.3, 0.03  # α-β 濾波器增益
GRIP_DWELL = 100  # 兩指力達標後持續的步數（0.1 s）才視為夾穩
K_CLIK = 100.0  # CLIK 位置/姿態誤差增益 [1/s]
MAX_DX = 0.006  # 可補償的手內 x 偏移上限（指墊 ±10 mm、料盤側銷內緣 17 mm → 餘裕 7 mm）
Y_TOL = 0.012  # 允許的工件橫向偏移（手指開口 80 mm、方塊 50 mm → 單邊 15 mm）


# ------------------------------------------------------------------ 估測
class BeltEstimator:
    """工件沿帶向位置/速度估測（open：光電 + 標稱帶速；track：α-β 濾波）。"""

    def __init__(self, mode: str, v_nom: float, alpha: float = ALPHA, beta: float = BETA, dt: float = CAM_DT):
        if mode not in ("open", "track"):
            raise ValueError("mode 需為 open 或 track")
        self.mode, self.alpha, self.beta, self.dt = mode, alpha, beta, dt
        self.x, self.v, self.y, self.t = 0.0, float(v_nom), BELT_Y, 0.0
        self.n = 0

    def init(self, t: float, x: float, y: float):
        self.t, self.x, self.y, self.n = float(t), float(x), float(y), 1

    def update(self, t: float, zx: float, zy: float):
        """相機量測更新（open 模式忽略）。"""
        if self.mode == "open":
            return
        T = t - self.t
        xp = self.x + self.v * T
        r = zx - xp
        self.x = xp + self.alpha * r
        self.v += self.beta / T * r
        self.y += self.alpha * (zy - self.y)
        self.t = float(t)
        self.n += 1

    def predict(self, t: float) -> tuple[float, float, float]:
        return self.x + self.v * (t - self.t), self.v, self.y


def window_length(v: float, t_grip: float = T_GRIP_NOM) -> float:
    """自等待點起算，同步到抬升結束 TCP 沿帶向的行程 L(v)（同步在 x̂ = x_w − v·T_SYNC/2 觸發時）。"""
    return v * (T_SYNC / 2 + T_DESC + t_grip + T_LIFT / 2)


def x_end_pred(v: float, t_grip: float = T_GRIP_NOM, x_wait: float = X_WAIT, x_first: float = X_DET) -> float:
    """抬升結束時 TCP 的 x。x_first：最早能開始同步的工件位置
    （光電開關；帶速高時觸發點在光電上游，只能晚觸發）。"""
    x_trig = max(x_wait - v * T_SYNC / 2, x_first)
    return x_trig + v * (T_SYNC + T_DESC + t_grip + T_LIFT / 2)


def v_max_window(t_grip: float = T_GRIP_NOM, x_wait: float = X_WAIT, x_first: float = X_DET) -> float:
    """追蹤窗允許的最大帶速：x_end_pred(v) ≤ X_END_MAX（x_end_pred 對 v 單調遞增 → 二分法）。"""
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if x_end_pred(mid, t_grip, x_wait, x_first) <= X_END_MAX else (lo, mid)
    return lo


def alpha_beta_noise_ratio(alpha: float = ALPHA, beta: float = BETA) -> float:
    """α-β 濾波器穩態位置估測變異數 / 量測變異數（Bar-Shalom et al. 2001, §6.5）：
    (2α² + 2β − 3αβ) / (α·(4 − 2α − β))。"""
    return (2 * alpha**2 + 2 * beta - 3 * alpha * beta) / (alpha * (4 - 2 * alpha - beta))


def _quintic(tau: float) -> tuple[float, float, float]:
    """s, ds/dτ, ∫s dτ（τ∈[0,1]）。"""
    u = min(max(tau, 0.0), 1.0)
    return (10 * u**3 - 15 * u**4 + 6 * u**5, 30 * u**2 - 60 * u**3 + 30 * u**4,
            2.5 * u**4 - 3 * u**5 + u**6)


# ------------------------------------------------------------------ 追蹤抓取
@dataclass
class Sensor:
    """理想相機 + 高斯雜訊：讀取模擬器中的工件位置。"""

    sigma: float = 0.0005
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def read(self, scene, body) -> np.ndarray:
        p = np.array(pb.getBasePositionAndOrientation(body, physicsClientId=scene.cid)[0])
        return p[:2] + self.rng.normal(0.0, self.sigma, 2) if self.sigma > 0 else p[:2]


def feed_part(scene, body: int | None = None, y: float = BELT_Y, x: float = X_FEED, rgba=None) -> int:
    """在輸送帶上游放一個方塊（body=None 則新生成），初速 = 帶速。"""
    cid = scene.cid
    pos = [x, y, Z_OBJ + 1e-3]
    if body is None:
        keep, spec = scene.g.obj, scene.g.obj_spec
        body = scene.g.spawn(CUBE, pos)
        scene.g.obj, scene.g.obj_spec = keep, spec
        scene.wc.add_part(body)
    else:
        pb.resetBasePositionAndOrientation(body, pos, [0, 0, 0, 1], physicsClientId=cid)
    pb.resetBaseVelocity(body, [scene.wc.belt_speed, 0, 0], [0, 0, 0], physicsClientId=cid)
    if rgba is not None:
        pb.changeVisualShape(body, -1, rgbaColor=rgba, physicsClientId=cid)
    return body


def _part(scene, body):
    pos, quat = pb.getBasePositionAndOrientation(body, physicsClientId=scene.cid)
    return np.array(pos), quat


class Tracker:
    """CLIK 線上追蹤：每個物理步給 TCP 目標位置/速度（工具朝下），積分出關節指令交給計算力矩控制。"""

    def __init__(self, scene, ex: Executor, hook=None):
        self.sc, self.ex, self.r, self.hook = scene, ex, scene.r, hook
        self.off = R_DOWN @ T_TCP[:3, 3]
        self.q = scene.sim.get_state()[0].copy()
        self.qd = np.zeros(6)
        self.vel_util = self.torque_util = 0.0
        self.k = 0

    def step(self, p_tcp, v_tcp):
        sc, dt = self.sc, self.sc.dt
        T = self.r.fk(self.q)
        e = np.concatenate([np.asarray(p_tcp) - self.off - T[:3, 3], rot_log(R_DOWN @ T[:3, :3].T)])
        xd = np.concatenate([np.asarray(v_tcp), np.zeros(3)]) + K_CLIK * e
        qd = np.linalg.solve(self.r.jacobian(self.q), xd)
        qdd = (qd - self.qd) / dt
        self.q = self.q + qd * dt
        self.qd = qd
        sc._hold(self.q, qd, qdd)
        sc.step()
        self.k += 1
        self.vel_util = max(self.vel_util, float(np.max(np.abs(qd) / self.r.qd_max)))
        if self.k % 10 == 0:
            tau = self.r.rne(self.q, qd, qdd)
            self.torque_util = max(self.torque_util, float(np.max(np.abs(tau) / self.r.tau_max)))
        if self.hook and self.k % 25 == 0:
            self.hook(sc)

    def finish(self):
        self.sc._hold(self.q)
        self.ex.t, self.ex.q = self.sc.t, self.q


def goto_wait(ex: Executor, x_wait: float = X_WAIT, y: float = BELT_Y, **kw) -> bool:
    """到等待點（懸停高度）。TCP 低於搬運高度時先垂直抬起，避免 PTP 橫掃到治具/料盤。"""
    prog = []
    tcp = ex.tcp_of(ex._cur_q())
    if tcp[2] < Z_CLEAR - 1e-3:
        up = [float(tcp[0]), float(tcp[1]), Z_CLEAR]
        prog.append({"cmd": "move_l", "tcp": up, "id": "lift_clear", **kw})
    prog.append({"cmd": "move_j", "tcp": [x_wait, y, Z_HOVER], "id": "to_wait", **kw})
    return ex.run(prog).ok


def track_pick(scene, ex: Executor, body: int, v_nom: float, mode: str = "track",
               sensor: Sensor | None = None, force: float = 40.0, hook=None, timeout: float = 15.0) -> dict:
    """追蹤並夾起輸送帶上移動中的 body。手臂須已在等待點附近（懸停高度）。回傳量測 dict（含 ok/error）。"""
    sensor = sensor or Sensor()
    dt = scene.dt
    cam_n = int(round(CAM_DT / dt))
    scene.obj = scene.g.obj = body
    est = BeltEstimator(mode, v_nom)
    tr = Tracker(scene, ex, hook)
    p_w = ex.tcp_of(tr.q)
    m = {"ok": False, "error": None, "mode": mode, "v_nom": v_nom, "t_start": scene.t}
    k = 0
    est_log: list = []
    m["est_log"] = est_log

    def sense():
        nonlocal k
        k += 1
        if k % cam_n == 0 and est.n:
            est.update(scene.t, *sensor.read(scene, body))
            est_log.append((scene.t, est.x, est.v))

    # --- 等待：光電觸發 → 等工件到同步觸發位置
    t_lim = scene.t + timeout
    while True:
        if not est.n:
            p = _part(scene, body)[0]
            if p[0] >= X_DET:
                est.init(scene.t, *sensor.read(scene, body))
                m["t_det"], m["x_det"] = scene.t, float(est.x)
        else:
            xh, vh, _ = est.predict(scene.t)
            if xh >= p_w[0] - vh * T_SYNC / 2:
                break
        if scene.t > t_lim:
            tr.finish()
            m["error"] = "TIMEOUT"
            return m
        tr.step(p_w, np.zeros(3))
        sense()
    xh, vh, yh = est.predict(scene.t)
    m.update(t_sync=scene.t, v_est_sync=vh, y_est=yh)
    # 可行性：追蹤窗、橫向偏移
    x_end = x_end_pred(vh, x_first=xh)
    m["x_end_pred"] = float(x_end)
    if x_end > X_END_MAX or abs(yh - BELT_Y) > Y_TOL:
        tr.finish()
        m["error"] = "WINDOW" if x_end > X_END_MAX else "LATERAL"
        return m

    err: list = []  # 追蹤誤差（TCP − 工件實際位置，xy），夾爪閉合前

    def target(z, zd):
        xh, vh, yh = est.predict(scene.t)
        return np.array([xh, yh, z]), np.array([vh, 0.0, zd])

    def log_err():
        d = scene.g.tcp_pos()[:2] - _part(scene, body)[0][:2]
        err.append(d)

    # --- 同步
    n = int(round(T_SYNC / dt))
    for i in range(1, n + 1):
        s, sd, _ = _quintic(i / n)
        pt, vt = target(Z_HOVER, 0.0)
        tr.step(p_w + s * (pt - p_w), sd / T_SYNC * (pt - p_w) + s * vt)
        sense()
    # --- 追蹤下降
    n = int(round(T_DESC / dt))
    for i in range(1, n + 1):
        s, sd, _ = _quintic(i / n)
        tr.step(*target(Z_HOVER + s * (Z_OBJ - Z_HOVER), sd / T_DESC * (Z_OBJ - Z_HOVER)))
        sense()
        log_err()
    m["t_grip"] = scene.t
    e = np.array(err)
    m.update(track_rms_mm=float(np.sqrt(np.mean(e[:, 0] ** 2)) * 1e3),
             track_max_mm=float(np.max(np.abs(e[:, 0])) * 1e3),
             track_end_mm=float(e[-1, 0] * 1e3), track_y_end_mm=float(e[-1, 1] * 1e3),
             v_est_grip=float(est.predict(scene.t)[1]))
    # --- 追蹤中夾持：兩指法向力 ≥ 0.5·F 持續 GRIP_DWELL 步即完成
    scene.g.close(force)
    n = int(round(T_GRIP_MAX / dt))
    held = 0
    for i in range(1, n + 1):
        tr.step(*target(Z_OBJ, 0.0))
        sense()
        fl, fr = scene.g.contact_forces()
        held = held + 1 if min(fl, fr) >= 0.5 * force else 0
        if held >= GRIP_DWELL:
            break
    m.update(grip_s=i * dt, fL_N=fl, fR_N=fr)
    if held < GRIP_DWELL:
        scene.g.open()
        p0 = ex.tcp_of(tr.q)
        for i in range(1, 301):  # 鬆開並原地上升，避免撞擋料板
            s = _quintic(i / 300)[0]
            tr.step(np.array([p0[0], p0[1], Z_OBJ + s * (Z_CLEAR - Z_OBJ)]), np.zeros(3))
        tr.finish()
        ex.holding = False
        m["error"] = "GRIP_EMPTY"
        return m
    # --- 抬升 + x 向減速到 0（估測凍結）
    x0, v0, y0 = est.predict(scene.t)
    n = int(round(T_LIFT / dt))
    for i in range(1, n + 1):
        s, sd, si = _quintic(i / n)
        tau = i / n
        p = np.array([x0 + v0 * T_LIFT * (tau - si), y0, Z_OBJ + s * (Z_CLEAR - Z_OBJ)])
        tr.step(p, np.array([v0 * (1 - s), 0.0, sd / T_LIFT * (Z_CLEAR - Z_OBJ)]))
    for _ in range(100):  # 靜止 0.1 s
        tr.step(p, np.zeros(3))
    tr.finish()
    ex.holding = True
    pos, quat = _part(scene, body)
    d = pos - scene.g.tcp_pos()
    m.update(t_end=scene.t, pick_s=scene.t - m["t_sync"], x_end=float(p[0]), dx_mm=float(d[0] * 1e3),
             dy_mm=float(d[1] * 1e3), dz_mm=float(d[2] * 1e3), inhand_tilt_deg=_tilt_yaw(quat)[0],
             vel_util=tr.vel_util, torque_util=tr.torque_util)
    if abs(d[2]) > 0.004 or m["inhand_tilt_deg"] > 3.0:
        m["error"] = "DROPPED"
    elif abs(d[0]) > MAX_DX:
        m["error"] = "INHAND_OFFSET"
    else:
        m["ok"] = True
    return m


def place_held(scene, ex: Executor, place_xy, d_xy=(0.0, 0.0), z_place: float = Z_POCKET, **kw) -> dict:
    """把手上的工件放到 place_xy（TCP 目標 = place − 手內偏移 d），並量測放置誤差。"""
    body = scene.obj
    tgt = (place_xy[0] - d_xy[0], place_xy[1] - d_xy[1])
    rep = ex.run(transfer_program(tgt, tgt, z_place=z_place, **kw)[4:])
    pos, quat = _part(scene, body)
    tilt, yaw = _tilt_yaw(quat)
    mv = [r for r in rep.results if r["cmd"].startswith("move") and r["status"] == "ok"]
    return {"ok": rep.ok, "error": None if rep.ok else f"{rep.error.code}@{rep.error.index}",
            "xy_err_mm": float(np.hypot(pos[0] - place_xy[0], pos[1] - place_xy[1]) * 1e3),
            "z_err_mm": float((pos[2] - z_place) * 1e3), "tilt_deg": tilt, "yaw_deg": yaw,
            "torque_util": max((r["torque_ratio"] for r in mv), default=0.0),
            "vel_util": max((r["vel_ratio"] for r in mv), default=0.0)}


def placed_ok(p: dict) -> bool:
    return bool(p["ok"] and p["xy_err_mm"] <= 3.0 and abs(p["z_err_mm"]) <= 1.5 and p["tilt_deg"] <= 2.0
                and p["yaw_deg"] <= 5.0)


def static_pick(scene, ex: Executor, body: int, place_xy, sensor: Sensor | None = None, timeout: float = 10.0,
                hook=None, **kw) -> dict:
    """後備：等工件被擋料板擋停後，以一般取放程式抓取並放到 place_xy。"""
    sensor = sensor or Sensor(0.0)
    scene.obj = scene.g.obj = body
    t_lim = scene.t + timeout
    while np.linalg.norm(pb.getBaseVelocity(body, physicsClientId=scene.cid)[0]) > 1e-3 or \
            _part(scene, body)[0][0] < 0.40:
        if scene.t > t_lim:
            return {"ok": False, "error": "TIMEOUT"}
        scene.wait(0.05)
        if hook:
            hook(scene)
    ex.t = scene.t
    xy = sensor.read(scene, body)
    rep = ex.run(pick_place_program((float(xy[0]), float(xy[1])), place_xy, **kw))
    pos, quat = _part(scene, body)
    tilt, yaw = _tilt_yaw(quat)
    return {"ok": rep.ok, "error": None if rep.ok else f"{rep.error.code}@{rep.error.index}",
            "xy_err_mm": float(np.hypot(pos[0] - place_xy[0], pos[1] - place_xy[1]) * 1e3),
            "z_err_mm": float((pos[2] - Z_POCKET) * 1e3), "tilt_deg": tilt, "yaw_deg": yaw}


def run_track_cycle(scene, ex: Executor, place_xy, belt_speed: float, v_nom: float | None = None,
                    mode: str = "track", y: float = BELT_Y, sigma: float = 0.0005, seed: int = 0,
                    compensate: bool = True, fallback: bool = False, hook=None) -> dict:
    """單一週期：到等待點 → 上料 → 追蹤夾取 → 放入料盤格 place_xy。
    belt_speed 為實際帶速，v_nom 為控制器假設的標稱帶速（預設相同）。"""
    scene.wc.belt_speed = belt_speed
    v_nom = belt_speed if v_nom is None else v_nom
    if not goto_wait(ex):
        return {"success": False, "error": "WAIT_POSE"}
    body = feed_part(scene, scene.obj, y=y)
    t0 = scene.t
    sensor = Sensor(sigma, seed)
    m = track_pick(scene, ex, body, v_nom, mode, sensor, hook=hook)
    m.update(belt_speed=belt_speed, y_feed=y, sigma_mm=sigma * 1e3, method="track")
    if m["ok"]:
        d = (m["dx_mm"] * 1e-3, m["dy_mm"] * 1e-3) if compensate else (0.0, 0.0)
        pl = place_held(scene, ex, place_xy, d)
        m["place"] = pl
        m["error"] = pl["error"]
        m["success"] = placed_ok(pl)
        if m["success"]:
            m["torque_util"] = max(m["torque_util"], pl["torque_util"])
    else:
        m["success"] = False
        if fallback:
            if ex.tcp_of(ex._cur_q())[2] < Z_CLEAR - 1e-3:
                xy = ex.tcp_of(ex._cur_q())[:2].tolist()
                ex.run([{"cmd": "move_l", "tcp": [*xy, Z_CLEAR], "id": "abort_up"}])
            pl = static_pick(scene, ex, body, place_xy, hook=hook)
            m.update(place=pl, method="static", track_error=m["error"], error=pl["error"],
                     success=pl["ok"] and placed_ok(pl))
    m["cycle_s"] = scene.t - t0
    return m


def run_flow(scene, ex: Executor, n: int, period: float, belt_speed: float, places, ys=None,
             sigma: float = 0.0005, seed: int = 0, hook=None, rgbas=None) -> dict:
    """連續來料：每 period 秒上料一個（共 n 個），依序追蹤夾取放入 places。

    策略：目標 = 帶上最下游的未取工件；仍在追蹤窗內 → 追蹤夾取；否則等它被擋料板擋停後靜態抓取。
    上料由每 25 步呼叫的 hook 依時間觸發（手臂執行 T007 指令期間也持續上料）。
    """
    scene.wc.belt_speed = belt_speed
    cid = scene.cid
    ys = [BELT_Y] * n if ys is None else list(ys)
    fed: list = []
    # 藏起場景預設方塊（改為靜態、移到地面下）
    pb.resetBasePositionAndOrientation(scene.obj, [0.45, 0.0, -1.0], [0, 0, 0, 1], physicsClientId=cid)
    pb.changeDynamics(scene.obj, -1, mass=0.0, physicsClientId=cid)
    hidden = scene.obj
    scene.wc.parts.remove(hidden)

    def tick(sc):
        if len(fed) < n and sc.t - t0 >= len(fed) * period:
            b = feed_part(sc, None, y=ys[len(fed)], rgba=rgbas[len(fed) % len(rgbas)] if rgbas else None)
            fed.append({"body": b, "t_feed": sc.t})
        if hook:
            hook(sc)

    items, done = [], 0
    goto_wait(ex)
    t0 = scene.t
    old_cb, old_every = ex.frame_cb, ex.cb_every
    ex.frame_cb, ex.cb_every = tick, 25
    try:
        while done < n:
            if done >= len(fed):  # 尚未上料：原地等待
                scene.wait(0.025)
                tick(scene)
                ex.t = scene.t
                continue
            f = fed[done]
            body = f["body"]
            px = _part(scene, body)[0][0]
            it = {"index": done, "t_feed": f["t_feed"] - t0, "x_at_start": float(px)}
            m = {"ok": False, "error": "WINDOW"}
            if px < X_END_MAX - window_length(belt_speed):
                m = track_pick(scene, ex, body, belt_speed, "track", Sensor(sigma, seed + done), hook=tick)
                it.update({k: m.get(k) for k in ("track_rms_mm", "track_max_mm", "dx_mm", "grip_s")})
            if m["ok"]:
                pl = place_held(scene, ex, places[done], (m["dx_mm"] * 1e-3, m["dy_mm"] * 1e-3))
                it.update(method="track", **pl)
            elif m["error"] not in ("WINDOW", "LATERAL"):
                it.update(method="track", ok=False, error=m["error"])
            else:  # 追不上（或橫向偏移過大）→ 擋料板前靜態抓取
                it["track_error"] = m["error"]
                pl = static_pick(scene, ex, body, places[done], hook=tick)
                it.update(method="static", **pl)
            it["t_done"] = scene.t - t0
            it["success"] = bool(it["ok"] and placed_ok(it))
            items.append(it)
            done += 1
            if not it["ok"]:
                break
            if done < n:
                goto_wait(ex)
    finally:
        ex.frame_cb, ex.cb_every = old_cb, old_every
    total = scene.t - t0
    ok = sum(i["success"] for i in items)
    return {"n": n, "period_s": period, "belt_speed": belt_speed, "items": items, "placed": ok,
            "tracked": sum(i["method"] == "track" and i["success"] for i in items),
            "static": sum(i["method"] == "static" and i["success"] for i in items),
            "total_s": total, "throughput_ppm": 60.0 * ok / total if total > 0 else 0.0,
            "success": ok == n, "bodies": [f["body"] for f in fed]}


__all__ = ["BeltEstimator", "Sensor", "Tracker", "feed_part", "goto_wait", "track_pick", "place_held",
           "static_pick", "run_track_cycle", "run_flow", "window_length", "x_end_pred", "v_max_window",
           "alpha_beta_noise_ratio", "placed_ok", "BELT_Y", "X_FEED", "X_DET", "X_WAIT", "Z_HOVER",
           "X_END_MAX", "T_SYNC", "T_DESC", "T_LIFT", "T_GRIP_NOM", "CAM_DT", "ALPHA", "BETA", "MAX_DX",
           "Y_TOL"]
