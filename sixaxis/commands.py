"""指令介面（T007）：JSON / DSL 指令的解析、驗證與執行器。

指令（皆可帶 ``id`` 標籤）
* ``move_j``  關節空間 PTP：``q``（rad，6 個）或 ``q_deg`` 或 ``tcp``（[x,y,z]，工具朝下）；可選 ``T``、``speed``、``profile``
* ``move_l``  TCP 直線（姿態保持）：``tcp``；可選 ``T``、``speed``、``profile``
* ``move_c``  TCP 圓弧（過 ``via`` 到 ``tcp``，姿態保持）：``via``、``tcp``；可選 ``T``、``speed``、``profile``
* ``grip``    夾持：可選 ``force``（N，預設 40）；夾到底（空夾）回報 GRIP_EMPTY
* ``release`` 釋放
* ``wait``    等待 ``T`` 秒
* ``set_speed`` 設定後續運動的速度比例 ``speed`` ∈ (0,1]

時間：未給 ``T`` 時以 :func:`trajectory.fit_duration` 求「速度/力矩使用率 ≤ 0.5·speed」的最短時間
（見 T003 時間縮放）；給了 ``T`` 但超限 → LIMIT 錯誤（不默默改時間）。

錯誤碼：SYNTAX（格式/未知指令/缺參數/型別）、RANGE（參數範圍）、UNREACHABLE（IK 無解/路徑 IK 不收斂）、
LIMIT（關節限位/速度/力矩超限）、COLLISION（手臂與場景靜態體接觸）、GRIP_EMPTY（夾持落空）。

DSL 範例（每行一個指令，``#`` 為註解，值可為 JSON 或以逗號分隔的數字）::

    move_j tcp=0.45,0,0.25
    move_l tcp=0.45,0,0.125 speed=0.5
    grip force=40
    wait T=0.2

執行器可在無物理（``scene=None``，僅運動學的 dry-run）或 :class:`graspsim.GraspScene` 上執行；
遇到第一個錯誤即停止並回報（含指令序號、錯誤碼）。
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from .graspsim import PED_X, Q_REF, tool_pose
from .gripper import F_MAX, S_MAX, T_TCP
from .robot import SixAxisRobot
from .trajectory import (
    ArcPath,
    LinePath,
    check_samples,
    fit_duration,
    joint_ptp,
    make_profile,
    min_duration,
    sample_cartesian,
    sample_joint,
)

DEFAULT_FORCE = 40.0
MIN_T = 0.1
NOMINAL = 0.5  # speed=1 對應速度/力矩使用率 0.5（保留餘裕給負載與抓持穩定性）
PROFILES = ("quintic", "trapezoid")


class CommandError(Exception):
    def __init__(self, index: int, cmd: str, code: str, message: str):
        super().__init__(f"[#{index} {cmd}] {code}: {message}")
        self.index, self.cmd, self.code, self.message = index, cmd, code, message

    def as_dict(self) -> dict:
        return {"index": self.index, "cmd": self.cmd, "code": self.code, "message": self.message}


# ------------------------------------------------------------------ 解析
def _parse_value(s: str):
    try:
        return json.loads(s)
    except ValueError:
        pass
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        return s


def parse_dsl(text: str) -> list[dict]:
    cmds = []
    for n, raw in enumerate(text.splitlines()):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = shlex.split(line)
        c = {"cmd": toks[0]}
        for t in toks[1:]:
            if "=" not in t:
                raise CommandError(len(cmds), toks[0], "SYNTAX", f"第 {n + 1} 行：參數需為 key=value（{t!r}）")
            k, v = t.split("=", 1)
            c[k] = _parse_value(v)
        cmds.append(c)
    return cmds


def _num(c, key, i, lo=None, hi=None, lo_open=False):
    v = c[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise CommandError(i, c["cmd"], "SYNTAX", f"{key} 需為數值")
    if lo is not None and (v <= lo if lo_open else v < lo) or hi is not None and v > hi:
        raise CommandError(i, c["cmd"], "RANGE", f"{key}={v} 超出範圍")
    return float(v)


def _vec(c, key, i, n):
    v = c[key]
    ok = isinstance(v, (list, tuple)) and len(v) == n and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) for x in v)
    if not ok:
        raise CommandError(i, c["cmd"], "SYNTAX", f"{key} 需為 {n} 個數值")
    return [float(x) for x in v]


# 指令 → (必要鍵, 可選鍵)
SPEC = {
    "move_j": ((), ("q", "q_deg", "tcp", "T", "speed", "profile")),
    "move_l": (("tcp",), ("T", "speed", "profile")),
    "move_c": (("via", "tcp"), ("T", "speed", "profile")),
    "grip": ((), ("force",)),
    "release": ((), ()),
    "wait": (("T",), ()),
    "set_speed": (("speed",), ()),
}


def validate_one(c: dict, i: int) -> dict:
    """回傳正規化後的指令 dict，否則拋 CommandError。"""
    if not isinstance(c, dict) or "cmd" not in c:
        raise CommandError(i, "?", "SYNTAX", "每個指令需為含 'cmd' 的物件")
    name = str(c["cmd"]).lower()
    if name not in SPEC:
        raise CommandError(i, name, "SYNTAX", f"未知指令；可用：{', '.join(SPEC)}")
    c = {**c, "cmd": name}
    req, opt = SPEC[name]
    for k in req:
        if k not in c:
            raise CommandError(i, name, "SYNTAX", f"缺少參數 {k}")
    for k in c:
        if k not in ("cmd", "id", *req, *opt):
            raise CommandError(i, name, "SYNTAX", f"未知參數 {k}")
    out = {"cmd": name, "id": c.get("id", f"{name}#{i}")}
    for k in ("tcp", "via"):
        if k in c:
            out[k] = _vec(c, k, i, 3)
    if "q" in c:
        out["q"] = _vec(c, "q", i, 6)
    if "q_deg" in c:
        out["q"] = list(np.radians(_vec(c, "q_deg", i, 6)))
    if name == "move_j" and sum(k in c for k in ("q", "q_deg", "tcp")) != 1:
        raise CommandError(i, name, "SYNTAX", "move_j 需恰好給 q / q_deg / tcp 其中之一")
    if "T" in c:
        out["T"] = _num(c, "T", i, 0.0 if name == "wait" else MIN_T)
    if "speed" in c:
        out["speed"] = _num(c, "speed", i, 0.0, 1.0, lo_open=True)
    if "profile" in c:
        if c["profile"] not in PROFILES:
            raise CommandError(i, name, "RANGE", f"profile 需為 {PROFILES}")
        out["profile"] = c["profile"]
    if name == "grip":
        out["force"] = _num(c, "force", i, 1.0, F_MAX) if "force" in c else DEFAULT_FORCE
    return out


def parse(program) -> list[dict]:
    """program：list[dict]、{"commands": [...]}、JSON 字串或 DSL 文字。驗證失敗拋 CommandError。"""
    if isinstance(program, str):
        s = program.strip()
        if s[:1] in "[{":
            try:
                program = json.loads(s)
            except ValueError as e:
                raise CommandError(0, "?", "SYNTAX", f"JSON 解析失敗：{e}") from e
        else:
            program = parse_dsl(program)
    if isinstance(program, dict):
        program = program.get("commands", None)
    if not isinstance(program, list):
        raise CommandError(0, "?", "SYNTAX", "程式需為指令陣列")
    return [validate_one(c, i) for i, c in enumerate(program)]


# ------------------------------------------------------------------ 執行
@dataclass
class Report:
    results: list = field(default_factory=list)
    error: CommandError | None = None
    t_total: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class Executor:
    """scene=None → 僅運動學 dry-run（時間照算、不跑物理）；否則在 GraspScene 上以計算力矩控制執行。"""

    def __init__(self, scene=None, robot: SixAxisRobot | None = None, speed: float = 1.0, frame_cb=None,
                 cb_every: int = 25, collision_check: bool = True):
        self.scene = scene
        self.r = scene.r if scene is not None else (robot or SixAxisRobot())
        self.speed = speed
        self.frame_cb, self.cb_every = frame_cb, cb_every
        self.collision_check = collision_check and scene is not None and scene.wc is not None
        self.q = scene.sim.get_state()[0] if scene is not None else self.r.ik(
            tool_pose([PED_X, 0, 0.25]), Q_REF)[0]
        self.t = 0.0
        self.holding = False

    # --- 運動學輔助
    def tcp_of(self, q) -> np.ndarray:
        T = self.r.fk(q)
        return T[:3, 3] + T[:3, :3] @ T_TCP[:3, 3]

    def _cur_q(self):
        return self.scene.sim.get_state()[0] if self.scene is not None else self.q

    # --- 執行單一運動（共用）
    def _play(self, i, c, traj, T):
        """traj(t_array)->(q,qd,qdd)。dry-run 只更新時間與狀態。"""
        sc = self.scene
        q_end = traj(np.array([T]))[0][0]
        if sc is None:
            self.t += T
            self.q = q_end
            return {"tcp_err_mm": 0.0}
        n = int(round(T / sc.dt))
        t = np.arange(1, n + 1) * sc.dt
        q, qd, qdd = traj(t)
        for k in range(n):
            sc._hold(q[k], qd[k], qdd[k])
            sc.step()
            if k % 50 == 49 and self.collision_check:
                hit = sc.wc.arm_collisions()
                if hit:
                    raise CommandError(i, c["cmd"], "COLLISION", f"手臂與 {hit} 接觸（t={sc.t:.3f}s）")
            if self.frame_cb and k % self.cb_every == 0:
                self.frame_cb(sc)
        sc._hold(q_end)
        self.t = sc.t
        self.q = q_end
        act = sc.sim.get_state()[0]
        return {"tcp_err_mm": float(np.linalg.norm(self.tcp_of(act) - self.tcp_of(q_end)) * 1e3)}

    def _joint_traj(self, i, c, q1, speed):
        q0 = self._cur_q()
        kind = c.get("profile", "quintic")
        if np.any(q1 < self.r.q_min) or np.any(q1 > self.r.q_max):
            raise CommandError(i, c["cmd"], "LIMIT", "目標關節角超出限位")
        if np.max(np.abs(q1 - q0)) < 1e-9:
            return None, 0.0, {}

        def build(T):
            return sample_joint(self.r, joint_ptp(q0, q1, T, kind), 0.01)

        T, smp = self._time(i, c, build, min_duration(q1 - q0, self.r.qd_max * speed * NOMINAL, None, kind), speed)
        return (lambda t, tr=joint_ptp(q0, q1, T, kind): tuple(np.array(x) for x in zip(
            *[tr.state(x) for x in t], strict=True))), T, self._ratios(smp)

    def _time(self, i, c, build, T_min, speed):
        if "T" in c:
            smp = build(c["T"])
            chk = check_samples(self.r, smp)
            if smp.ik_fail:
                raise CommandError(i, c["cmd"], "UNREACHABLE", f"路徑上 {smp.ik_fail} 點 IK 不收斂")
            if not chk["ok"]:
                raise CommandError(i, c["cmd"], "LIMIT", f"T={c['T']}s 超限：速度 {chk['vel_ratio']:.2f}、"
                                   f"力矩 {chk['torque_ratio']:.2f}、限位違反={chk['pos_violation']}")
            return c["T"], smp
        T, smp, _ = fit_duration(self.r, build, max(T_min, MIN_T), margin=NOMINAL * speed)
        if smp.ik_fail:
            raise CommandError(i, c["cmd"], "UNREACHABLE", f"路徑上 {smp.ik_fail} 點 IK 不收斂")
        chk = check_samples(self.r, smp)
        if chk["pos_violation"] or not smp.meta.get("feasible", False):
            raise CommandError(i, c["cmd"], "LIMIT", "路徑超出關節限位或無可行時間（力矩/速度）")
        return T, smp

    def _ratios(self, smp):
        c = check_samples(self.r, smp)
        return {"vel_ratio": c["vel_ratio"], "torque_ratio": c["torque_ratio"]}

    def _cart_traj(self, i, c, path, speed):
        q0 = self._cur_q()
        kind = c.get("profile", "quintic")
        if path.length < 1e-6 and path.angle < 1e-6:
            return None, 0.0, {}
        T_min = max(path.length / (0.15 * speed), MIN_T)  # 起始猜測：TCP 約 0.25 m/s

        def build(T):
            return sample_cartesian(self.r, path, make_profile(kind, T), q0, 0.01)

        T, smp = self._time(i, c, build, T_min, speed)
        spl = CubicHermiteSpline(smp.t, smp.q, smp.qd)
        d1, d2 = spl.derivative(1), spl.derivative(2)
        return (lambda t: (spl(t), d1(t), d2(t))), T, self._ratios(smp)

    # --- 指令
    def run_one(self, i: int, c: dict) -> dict:
        name = c["cmd"]
        sc = self.scene
        res = {"index": i, "id": c["id"], "cmd": name, "t_start": self.t}
        speed = c.get("speed", self.speed)
        if name == "set_speed":
            self.speed = c["speed"]
        elif name == "wait":
            if sc:
                sc.wait(c["T"])
                self.t = sc.t
            else:
                self.t += c["T"]
        elif name == "grip":
            res.update(self._grip(i, c))
        elif name == "release":
            if sc:
                sc.g.open()
                sc.wait(0.4)
                self.t = sc.t
            else:
                self.t += 0.4
            self.holding = False
        else:
            q0 = self._cur_q()
            T0 = self.r.fk(q0)
            R0 = T0[:3, :3]
            tcp_target = None
            if name == "move_j":
                if "q" in c:
                    q1 = np.array(c["q"])
                    tcp_target = self.tcp_of(q1)
                else:
                    tcp_target = np.array(c["tcp"])
                    sols = self.r.ik(tool_pose(tcp_target), q0)
                    if not sols:
                        raise CommandError(i, name, "UNREACHABLE", f"tcp={c['tcp']} 無可達 IK 解（限位內）")
                    q1 = sols[0]
                traj, T, rat = self._joint_traj(i, c, q1, speed)
            else:
                tcp_target = np.array(c["tcp"])
                flange = lambda p: np.block([[R0, (p - R0 @ T_TCP[:3, 3])[:, None]], [np.zeros((1, 3)), 1.0]])  # noqa: E731
                TA, TC = T0, flange(tcp_target)
                if name == "move_l":
                    path = LinePath(TA, TC)
                else:
                    try:
                        path = ArcPath(TA, np.array(c["via"]) - R0 @ T_TCP[:3, 3], TC)
                    except ValueError as e:
                        raise CommandError(i, name, "RANGE", f"圓弧三點共線：{e}") from e
                traj, T, rat = self._cart_traj(i, c, path, speed)
            if traj is not None:
                res.update(self._play(i, c, traj, T))
            res.update({"duration": T, **rat})
            res["tcp_target"] = tcp_target.tolist()
            res["tcp_end"] = self.tcp_of(self._cur_q()).tolist()
            res["target_err_mm"] = float(np.linalg.norm(np.array(res["tcp_end"]) - tcp_target) * 1e3)
        res["t_end"] = self.t
        return res

    def _grip(self, i, c):
        sc = self.scene
        if sc is None:
            self.t += 0.5
            self.holding = True
            return {"holding": True, "dry_run": True}
        sc.g.close(c["force"])
        sc.wait(0.5)
        self.t = sc.t
        stroke = float(np.mean(sc.g.finger_pos()))
        fl, fr = sc.g.contact_forces()
        if stroke > S_MAX - 1e-3 or min(fl, fr) < 0.2 * c["force"]:
            self.holding = False
            raise CommandError(i, "grip", "GRIP_EMPTY",
                               f"夾持落空：指行程 {stroke * 1e3:.1f} mm、接觸力 L/R = {fl:.1f}/{fr:.1f} N")
        self.holding = True
        return {"holding": True, "stroke_mm": stroke * 1e3, "fL_N": fl, "fR_N": fr}

    def run(self, program, stop_on_error: bool = True) -> Report:
        rep = Report()
        try:
            cmds = parse(program)
        except CommandError as e:
            rep.error = e
            return rep
        for i, c in enumerate(cmds):
            try:
                rep.results.append({**self.run_one(i, c), "status": "ok"})
            except CommandError as e:
                rep.results.append({"index": i, "id": c["id"], "cmd": c["cmd"], "status": "error",
                                    "code": e.code, "message": e.message, "t_start": self.t})
                rep.error = rep.error or e
                if stop_on_error:
                    break
        rep.t_total = self.t
        return rep


__all__ = ["CommandError", "Executor", "Report", "parse", "parse_dsl", "validate_one"]
