"""銲接路徑（T105）：銲槍工具、焊道幾何（直線＋圓弧）、固定銲槍姿態與等速時間律、模擬與量測。

銲槍與 TCP
----------
彎頸 MIG 銲槍以固定關節接在法蘭：槍身沿法蘭 z 軸 ``BODY_L``，之後繞法蘭 y 軸彎 ``NECK_DEG``，
槍頸＋噴嘴長 ``NECK_L``，銲絲伸出 ``STICKOUT``。TCP = 銲絲尖端，TCP 座標 z 軸 = 銲絲方向::

    T_TCP = Tz(BODY_L) · Ry(β) · Tz(NECK_L + STICKOUT)

焊道座標與銲槍角度（AWS A3.0 的 travel angle / work angle 定義）
----------------------------------------------------------------
焊道位於法向量 n 的平面上，以弧長 s 參數化；t(s) 為切線（行進方向）、b = n × t（行進方向左側）。

* 工作角 work（由工件表面量起，90° = 垂直表面；<90° 槍身倒向 +b，>90° 倒向 −b）：
  槍身方向（由尖端指向槍身）u = sin(work)·n + cos(work)·b
* 行進角 travel（+ = 前進法 push，銲絲指向行進方向；− = 後退法 drag）：
  銲絲方向 a = −cos(travel)·u + sin(travel)·t
* 工具 x 軸 = t 在 ⟂a 平面的投影（再繞 a 轉 roll），決定彎頸所在平面。

「固定姿態」＝ 相對焊道座標 (t, n, b) 的 work / travel 角固定：直線段為世界座標固定姿態，
圓弧段銲槍繞 n 以角速度 κ·v 旋轉（κ = ±1/R）。平面焊道的切線航向 ψ(s) = ∫κ ds，故
R_tcp(s) = Rot(n, ψ_o(s)) · R_tcp(0)，ω = n · ψ_o'(s) · ṡ。

直線–圓弧接點曲率不連續 → 角速度階躍 κ·v（角加速度無限大）。可選「航向平滑」：
ψ_o(s) = (1/2b)∫_{s−b}^{s+b} ψ(σ)dσ（半寬 b 的移動平均），使角速度連續，代價是接點附近
銲槍方位偏離切線最多 |κ|·b/4（由積分式可得；b ≤ 弧長時）。

等速時間律
----------
弧長速度 ṡ：起/收弧各以 smoothstep 3τ²−2τ³ 在 ``t_ramp`` 內加/減速（加速度連續），中間等速 v。
加速段行程 v·t_ramp/2，總時間 T = L/v + t_ramp。

製程量（僅作指標換算，不模擬電弧物理）
--------------------------------------
* 線入熱量 Q = η·U·I / v  [J/mm]（EN 1011-1：Q = k·U·I/v；GMAW 的熱效率 k = 0.8）
* 焊道截面積 A = (π d²/4)·WFS / v  [mm²]（銲絲體積守恆；d 線徑、WFS 送絲速度）
兩者皆 ∝ 1/v，故 TCP 速度均勻度直接決定焊道均勻度。

控制
----
計算力矩控制（ωn = 60 rad/s 臨界阻尼），逆動力學用 PyBullet 的 calculateInverseDynamics（含銲槍質量）；
另提供 PD + 重力補償（`controller="pd"`）作對照。力矩以 τ_max 限幅。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as pb
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .pbsim import PBRobot
from .robot import SixAxisRobot
from .trajectory import LinePath, Quintic, joint_ptp, min_duration, sample_cartesian
from .urdf import write_urdf

TORCH_URDF = Path(__file__).resolve().parents[1] / "robots" / "ur5_torch.urdf"

# ------------------------------------------------------------------ 銲槍
NECK_DEG = 22.0  # 彎頸角
BODY_L = 0.130  # 安裝座 + 槍身（沿法蘭 z）
NECK_L = 0.110  # 槍頸 + 噴嘴（到導電嘴端）
STICKOUT = 0.015  # 銲絲伸出長度（CTWD）
TORCH_MASS = 1.2
TORCH_COM = (0.008, 0.0, 0.095)
NOZZLE_R = 0.011
TORCH_LINK, TCP_LINK = 7, 8


def _ry(b: float) -> np.ndarray:
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


_B = np.radians(NECK_DEG)
T_TCP = np.eye(4)
T_TCP[:3, :3] = _ry(_B)
T_TCP[:3, 3] = np.array([0.0, 0.0, BODY_L]) + _ry(_B) @ np.array([0.0, 0.0, NECK_L + STICKOUT])
T_TCP_INV = np.linalg.inv(T_TCP)


def torch_xml(parent: str = "flange") -> str:
    """銲槍連桿 XML：torch（質量、外觀、碰撞）＋ tcp（銲絲尖端座標系）。"""

    def cyl(z0, z1, r, mat, bent, col=True):
        zc, ln = (z0 + z1) / 2, z1 - z0
        if bent:
            c = np.array([0.0, 0.0, BODY_L]) + _ry(_B) @ np.array([0.0, 0.0, zc])
            o = f'<origin xyz="{c[0]:.6f} 0 {c[2]:.6f}" rpy="0 {_B:.6f} 0"/>'
        else:
            o = f'<origin xyz="0 0 {zc:.6f}"/>'
        g = f'{o}<geometry><cylinder length="{ln:.6f}" radius="{r}"/></geometry>'
        s = f'<visual>{g}<material name="{mat}"/></visual>'
        return s + (f"<collision>{g}</collision>" if col else "")

    L = TORCH_MASS * 0.22**2 / 12  # 橫向慣量：長 0.22 m 細桿對質心 m·l²/12
    U = '  <material name="tbody"><color rgba="0.13 0.13 0.15 1"/></material>\n'
    U += '  <material name="tneck"><color rgba="0.70 0.72 0.75 1"/></material>\n'
    U += '  <material name="tnozzle"><color rgba="0.80 0.50 0.25 1"/></material>\n'
    U += '  <material name="twire"><color rgba="0.95 0.90 0.80 1"/></material>\n'
    U += f'  <link name="torch"><inertial><mass value="{TORCH_MASS}"/>'
    U += f'<origin xyz="{TORCH_COM[0]} {TORCH_COM[1]} {TORCH_COM[2]}"/>'
    U += f'<inertia ixx="{L:.6f}" iyy="{L:.6f}" izz="3e-4" ixy="0" ixz="0" iyz="0"/></inertial>'
    U += cyl(0.0, 0.030, 0.034, "tbody", False)  # 安裝座（含防撞感測器）
    U += cyl(0.030, BODY_L, 0.019, "tbody", False)
    U += cyl(-0.010, NECK_L - 0.045, 0.012, "tneck", True)
    U += cyl(NECK_L - 0.045, NECK_L, NOZZLE_R, "tnozzle", True)
    U += cyl(NECK_L, NECK_L + STICKOUT, 0.0012, "twire", True, col=False)
    U += "</link>\n"
    U += ('  <link name="tcp"><inertial><mass value="1e-6"/>'
          '<inertia ixx="1e-12" iyy="1e-12" izz="1e-12" ixy="0" ixz="0" iyz="0"/></inertial></link>\n')
    U += f'  <joint name="torch_mount" type="fixed"><parent link="{parent}"/><child link="torch"/></joint>\n'
    p = T_TCP[:3, 3]
    U += ('  <joint name="tcp_joint" type="fixed"><parent link="torch"/><child link="tcp"/>'
          f'<origin xyz="{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}" rpy="0 {_B:.9f} 0"/></joint>\n')
    return U


def write_arm_with_torch(robot: SixAxisRobot | None = None, path: str | Path = TORCH_URDF) -> Path:
    """手臂 URDF + 銲槍。連桿 index：7 = torch，8 = tcp。"""
    robot = robot or SixAxisRobot()
    write_urdf(robot, path)
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    path.write_text(txt.replace("</robot>", torch_xml() + "</robot>"), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 製程量
def heat_input(volt: float, amp: float, v: float, eta: float = 0.8) -> float:
    """線入熱量 [J/mm]；v 為 m/s。"""
    return eta * volt * amp / (v * 1e3)


def bead_area(d_wire: float, wfs: float, v: float) -> float:
    """焊道截面積 [mm²]；d_wire [m]、送絲速度 wfs [m/s]、行進速度 v [m/s]。"""
    return np.pi * (d_wire * 1e3) ** 2 / 4 * wfs / v


# ------------------------------------------------------------------ 焊道幾何
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot(axis, ang) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(axis, float) * ang).as_matrix()


@dataclass
class _Piece:
    kind: str
    length: float
    kappa: float  # 繞 n 的有號曲率
    p0: np.ndarray
    t0: np.ndarray
    center: np.ndarray | None = None


class Seam:
    """平面焊道：起點 + 直線/圓弧段。segments：("line", to) 或 ("arc", via, to)。

    以 ds 建表：位置 p(s)、切線 t(s)、航向 ψ(s)（相對起點切線，繞 n）。`kink_deg` 為段間切線最大折角。
    """

    def __init__(self, start, segments, normal=(0.0, 0.0, 1.0), ds: float = 1e-4, max_kink_deg: float = 1.0,
                 name: str = "seam"):
        self.name = name
        self.n = _unit(normal)
        self.start = np.asarray(start, float)
        self.pieces: list[_Piece] = []
        p = self.start
        for seg in segments:
            kind = seg[0]
            if kind == "line":
                to = np.asarray(seg[1], float)
                d = to - p
                ln = float(np.linalg.norm(d))
                if ln < 1e-9:
                    raise ValueError("直線段長度為 0")
                if abs(d @ self.n) > 1e-6 * max(ln, 1.0):
                    raise ValueError("直線段不在焊道平面內")
                self.pieces.append(_Piece("line", ln, 0.0, p, d / ln))
            elif kind == "arc":
                via, to = np.asarray(seg[1], float), np.asarray(seg[2], float)
                u, w = via - p, to - p
                nn = np.cross(u, w)
                if np.linalg.norm(nn) < 1e-12:
                    raise ValueError("圓弧三點共線")
                nn = nn / np.linalg.norm(nn)
                if abs(abs(nn @ self.n) - 1.0) > 1e-6:
                    raise ValueError("圓弧不在焊道平面內")
                c = p + np.linalg.solve(np.array([u, w, nn]), np.array([u @ u / 2, w @ w / 2, 0.0]))
                r = float(np.linalg.norm(p - c))
                e1 = (p - c) / r
                e2 = np.cross(nn, e1)
                th = float(np.arctan2((to - c) @ e2, (to - c) @ e1) % (2 * np.pi))
                sgn = float(np.sign(nn @ self.n))
                self.pieces.append(_Piece("arc", r * th, sgn / r, p, e2, c))
            else:
                raise ValueError(f"未知焊道段 {kind!r}")
            p = np.asarray(seg[-1], float)
        if not self.pieces:
            raise ValueError("焊道至少需一段")
        self.length = float(sum(pc.length for pc in self.pieces))
        self.s_break = np.concatenate([[0.0], np.cumsum([pc.length for pc in self.pieces])])
        # 段間折角
        self.kink_deg = 0.0
        for a, b in zip(self.pieces[:-1], self.pieces[1:], strict=True):
            ta = _rot(self.n, a.kappa * a.length) @ a.t0
            self.kink_deg = max(self.kink_deg, float(np.degrees(np.arccos(np.clip(ta @ b.t0, -1, 1)))))
        if self.kink_deg > max_kink_deg:
            raise ValueError(f"焊道切線不連續（折角 {self.kink_deg:.2f}°）")
        self.psi_break = np.concatenate([[0.0], np.cumsum([pc.kappa * pc.length for pc in self.pieces])])
        self.t0 = self.pieces[0].t0
        # 建表（供最近點搜尋）
        m = max(int(np.ceil(self.length / ds)), 2)
        self.s = np.linspace(0.0, self.length, m + 1)
        self.p, self.t, self.psi = self.eval(self.s)
        self._tree = cKDTree(self.p)

    # --- 解析求值
    def _index(self, s):
        return np.clip(np.searchsorted(self.s_break, s, side="right") - 1, 0, len(self.pieces) - 1)

    def eval(self, s):
        """弧長 s（陣列）→ (p, t, ψ)，逐段解析計算。"""
        s = np.atleast_1d(np.asarray(s, float))
        idx = self._index(s)
        p, t, psi = np.zeros((len(s), 3)), np.zeros((len(s), 3)), np.zeros(len(s))
        for k, pc in enumerate(self.pieces):
            sel = idx == k
            if not sel.any():
                continue
            u = s[sel] - self.s_break[k]
            if pc.kind == "line":
                p[sel] = pc.p0 + u[:, None] * pc.t0
                t[sel] = pc.t0
            else:
                ang = pc.kappa * u
                r0 = pc.p0 - pc.center
                c_, s_ = np.cos(ang)[:, None], np.sin(ang)[:, None]
                # Rodrigues：繞 n 轉 ang（r0、t0 皆 ⟂ n）
                p[sel] = pc.center + c_ * r0 + s_ * np.cross(self.n, r0)
                t[sel] = c_ * pc.t0 + s_ * np.cross(self.n, pc.t0)
            psi[sel] = self.psi_break[k] + pc.kappa * u
        return p, t, psi

    def pos(self, s):
        p = self.eval(s)[0]
        return p[0] if np.ndim(s) == 0 else p

    def tangent(self, s):
        t = self.eval(s)[1]
        return t[0] if np.ndim(s) == 0 else t

    def heading(self, s, blend: float = 0.0):
        """(ψ_o, dψ_o/ds)：blend > 0 時為半寬 blend 的移動平均（端點外以端點曲率線性外插）。"""
        if blend <= 0.0:
            return float(self.eval(s)[2][0]), self.pieces[int(self._index(s))].kappa
        g = np.linspace(s - blend, s + blend, 401)
        v = self._psi_ext(g)
        mean = float((v[:-1] + v[1:]).sum() / 2 / (len(g) - 1))
        return mean, float((v[-1] - v[0]) / (2 * blend))

    def _psi_ext(self, s):
        s = np.asarray(s, float)
        v = self.eval(np.clip(s, 0.0, self.length))[2]
        lo, hi = s < 0, s > self.length
        v[lo] += self.pieces[0].kappa * s[lo]
        v[hi] += self.pieces[-1].kappa * (s[hi] - self.length)
        return v

    def nearest(self, pts, s_hint=None, window: float = 0.005):
        """最近焊道點：回傳 (s, p, t)。先取最近表格點，再沿切線一階投影修正 s（p、t 以修正後的 s 重算）。
        s_hint（各點的預期弧長）給定時只在 s_hint ± window 內搜尋，避免封閉焊道（起點＝終點）取錯圈。"""
        pts = np.atleast_2d(pts)
        if s_hint is None:
            _, i = self._tree.query(pts)
        else:
            ds = self.s[1] - self.s[0]
            k = int(np.ceil(window / ds))
            off = np.arange(-k, k + 1)
            c = np.rint(np.atleast_1d(s_hint) / ds).astype(int)
            i = np.zeros(len(pts), int)
            for a in range(0, len(pts), 4000):
                idx = np.clip(c[a:a + 4000, None] + off[None, :], 0, len(self.s) - 1)
                d = np.linalg.norm(self.p[idx] - pts[a:a + 4000, None, :], axis=2)
                i[a:a + 4000] = idx[np.arange(len(idx)), np.argmin(d, axis=1)]
        s = np.clip(self.s[i] + np.einsum("ij,ij->i", pts - self.p[i], self.t[i]), 0.0, self.length)
        p, t, _ = self.eval(s)
        return s, p, t


def turtle_seam(start, heading, moves, normal=(0.0, 0.0, 1.0), name: str = "seam", **kw) -> Seam:
    """以「前進／轉彎」描述平面焊道：moves 為 ("line", 長度) 或 ("arc", 半徑, 轉角°)；
    轉角 > 0 為繞 n 左轉（逆時針），< 0 右轉；|轉角| ≤ 180（整圓用兩段 180°）。heading 為起點切線方向。"""
    n = _unit(normal)
    t = np.asarray(heading, float)
    t = _unit(t - (t @ n) * n)
    p = np.asarray(start, float)
    segs = []
    for mv in moves:
        if mv[0] == "line":
            p = p + t * float(mv[1])
            segs.append(("line", p))
        elif mv[0] == "arc":
            r, ang = float(mv[1]), np.radians(float(mv[2]))
            if r <= 0 or not 0 < abs(ang) <= np.pi + 1e-9:
                raise ValueError("圓弧需 R > 0、0 < |轉角| ≤ 180°")
            c = p + np.sign(ang) * r * np.cross(n, t)
            via = c + _rot(n, ang / 2) @ (p - c)
            p_new = c + _rot(n, ang) @ (p - c)
            segs.append(("arc", via, p_new))
            p, t = p_new, _rot(n, ang) @ t
        else:
            raise ValueError(f"未知動作 {mv[0]!r}")
    return Seam(start, segs, normal, name=name, **kw)


def tool_rotation(seam: Seam, s: float, work_deg: float = 90.0, travel_deg: float = 0.0, roll_deg: float = 0.0,
                  blend: float = 0.0):
    """TCP 姿態 R(s) 與 dψ_o/ds。見模組說明。"""
    n, t0 = seam.n, seam.t0
    b0 = np.cross(n, t0)
    w, a_ = np.radians(work_deg), np.radians(travel_deg)
    u = np.sin(w) * n + np.cos(w) * b0
    z = -np.cos(a_) * u + np.sin(a_) * t0
    x = _unit(t0 - (t0 @ z) * z)
    R0 = _rot(z, np.radians(roll_deg)) @ np.column_stack([x, np.cross(z, x), z])
    psi, dpsi = seam.heading(s, blend)
    return _rot(n, psi) @ R0, dpsi


def torch_angles(seam: Seam, tcp_pos, tool_z, s_hint=None):
    """由實際 TCP 位置與銲絲方向，對最近焊道點的 (t, n, b) 量 work / travel 角 [deg]。"""
    _, _, t = seam.nearest(tcp_pos, s_hint)
    a = np.atleast_2d(tool_z)
    b = np.cross(seam.n, t)
    travel = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", a, t), -1, 1)))
    work = np.degrees(np.arctan2(-a @ seam.n, -np.einsum("ij,ij->i", a, b)))
    return work, travel


def heading_error(seam: Seam, tcp_pos, tool_x, prm, s_hint=None) -> np.ndarray:
    """實際工具 x 軸（投影到焊道平面）相對「切線航向下應有方向」的繞 n 偏角 [deg]。"""
    s, _, _ = seam.nearest(tcp_pos, s_hint)
    R0, _ = tool_rotation(seam, 0.0, prm.work_deg, prm.travel_deg, prm.roll_deg)
    R0 = _rot(seam.n, -seam.heading(0.0)[0]) @ R0
    x0 = R0[:, 0] - (R0[:, 0] @ seam.n) * seam.n
    e1 = _unit(x0)
    e2 = np.cross(seam.n, e1)
    x = np.atleast_2d(tool_x)
    ang = np.arctan2(x @ e2, x @ e1) - seam.eval(s)[2]
    return np.degrees((ang + np.pi) % (2 * np.pi) - np.pi)


class SpeedProfile:
    """弧長時間律：smoothstep 加速 → 等速 v → smoothstep 減速。"""

    def __init__(self, length: float, v: float, t_ramp: float = 0.2):
        if v <= 0 or t_ramp <= 0:
            raise ValueError("v、t_ramp 需 > 0")
        if v * t_ramp > length:
            raise ValueError(f"焊道太短（{length * 1e3:.1f} mm）無法在 t_ramp 內達到 {v * 1e3:.1f} mm/s")
        self.L, self.v, self.tr = float(length), float(v), float(t_ramp)
        self.T = length / v + t_ramp

    def __call__(self, t):
        t = np.clip(np.asarray(t, float), 0.0, self.T)
        v, tr, T = self.v, self.tr, self.T

        def ramp(x):  # x∈[0,tr] → (距離, 速度)
            tau = x / tr
            return v * tr * (tau**3 - tau**4 / 2), v * (3 * tau**2 - 2 * tau**3)

        d_up, v_up = ramp(np.minimum(t, tr))
        d_dn, v_dn = ramp(np.minimum(T - t, tr))
        up, dn = t < tr, t > T - tr
        s = np.where(up, d_up, np.where(dn, self.L - d_dn, v * tr / 2 + v * (t - tr)))
        sd = np.where(up, v_up, np.where(dn, v_dn, v))
        return s, sd


# ------------------------------------------------------------------ 規劃
@dataclass
class WeldParams:
    speed: float = 0.010  # 行進速度 [m/s]
    work_deg: float = 90.0
    travel_deg: float = 10.0
    roll_deg: float = 0.0
    blend: float = 0.0  # 航向平滑半寬 [m]
    t_ramp: float = 0.2
    t_pre: float = 0.3  # 起弧停留
    t_crater: float = 0.3  # 收弧填弧坑停留
    approach: float = 0.05  # 沿銲絲方向的進/退刀距離


@dataclass
class WeldPlan:
    seam: Seam
    prm: WeldParams
    T: float
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    ik_fail: int
    vel_ratio: float
    torque_ratio: float
    sigma_min: float
    limit_ok: bool
    spline: object = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.ik_fail == 0 and self.limit_ok and self.vel_ratio <= 1.0 and self.torque_ratio <= 1.0

    def state(self, t):
        return self.spline(t), self._d1(t), self._d2(t)


def tcp_pose(seam: Seam, s: float, prm: WeldParams):
    T = np.eye(4)
    T[:3, :3], dpsi = tool_rotation(seam, s, prm.work_deg, prm.travel_deg, prm.roll_deg, prm.blend)
    T[:3, 3] = seam.pos(s)
    return T, dpsi


def plan_seam(robot: SixAxisRobot, seam: Seam, prm: WeldParams, q_seed, dt: float = 0.01) -> WeldPlan:
    """焊道 → 關節軌跡：沿等速時間律取樣，逐點數值 IK（前一點為初值），q̇ = J⁻¹·twist（法蘭）。"""
    prof = SpeedProfile(seam.length, prm.speed, prm.t_ramp)
    n = max(int(round(prof.T / dt)), 2)
    t = np.linspace(0.0, prof.T, n + 1)
    s, sd = prof(t)
    T0, _ = tcp_pose(seam, 0.0, prm)
    sols = robot.ik(T0 @ T_TCP_INV, q_seed)
    if not sols:
        return WeldPlan(seam, prm, prof.T, t, np.zeros((n + 1, 6)), np.zeros((n + 1, 6)), n + 1, 0, 0, 0, False)
    qk = sols[0]
    q, qd = np.zeros((n + 1, 6)), np.zeros((n + 1, 6))
    fail, sig = 0, np.inf
    for k in range(n + 1):
        Tt, dpsi = tcp_pose(seam, float(s[k]), prm)
        Tf = Tt @ T_TCP_INV
        qk, ok, _ = robot.ik_numeric(Tf, qk, tol=1e-10, damping=1e-8)
        fail += int(not ok)
        w = seam.n * dpsi * sd[k]
        v_tcp = seam.tangent(float(s[k])) * sd[k]
        v_f = v_tcp - np.cross(w, Tt[:3, 3] - Tf[:3, 3])
        J = robot.jacobian(qk)
        sig = min(sig, float(np.linalg.svd(J, compute_uv=False)[-1]))
        qd[k] = J.T @ np.linalg.solve(J @ J.T + 1e-10 * np.eye(6), np.concatenate([v_f, w]))
        q[k] = qk
    spl = CubicHermiteSpline(t, q, qd)
    qdd = spl.derivative(2)(t)
    tau = np.array([robot.rne(a, b, c) for a, b, c in zip(q, qd, qdd, strict=True)])
    plan = WeldPlan(seam, prm, prof.T, t, q, qd, fail,
                    float(np.max(np.abs(qd) / robot.qd_max)), float(np.max(np.abs(tau) / robot.tau_max)), sig,
                    bool(np.all(q >= robot.q_min) and np.all(q <= robot.q_max)), spl)
    plan._d1, plan._d2 = spl.derivative(1), spl.derivative(2)
    plan.meta = {"q_range_deg": np.degrees(q.max(0) - q.min(0)).tolist(),
                 "qdd_max": float(np.max(np.abs(qdd)))}
    return plan


def best_roll(robot: SixAxisRobot, seam: Seam, prm: WeldParams, q_seed, rolls=(0.0, 90.0, 180.0, -90.0),
              dt: float = 0.05):
    """銲絲為軸對稱 → 繞銲絲的 roll 是製程上的自由度。對候選 roll 各規劃一次（粗取樣），
    選「可行且 σ_min 最大」者。回傳 (roll 或 None, 各候選摘要 list)。"""
    rows, best = [], None
    for roll in rolls:
        p = WeldParams(**{**prm.__dict__, "roll_deg": float(roll)})
        try:
            pl = plan_seam(robot, seam, p, q_seed, dt)
        except ValueError:
            continue
        row = {"roll_deg": float(roll), "ok": pl.ok, "ik_fail": pl.ik_fail, "limit_ok": pl.limit_ok,
               "vel_ratio": pl.vel_ratio, "sigma_min": pl.sigma_min,
               "q6_min": float(pl.q[:, 5].min()), "q6_max": float(pl.q[:, 5].max()),
               "q_range_deg": pl.meta.get("q_range_deg")}
        rows.append(row)
        if pl.ok and (best is None or pl.sigma_min > best[1]):
            best = (float(roll), pl.sigma_min)
    return (best[0] if best else None), rows


# ------------------------------------------------------------------ 工作站與模擬
BENCH_TOP = 0.10
Q_SEED = np.array([0.0, -1.6, -2.0, -1.1, 1.57, 0.0])  # 肘向外、工具朝下的 IK 選解參考
HOME_TCP = (0.35, 0.0, 0.35)  # 原點：TCP 位置（銲絲垂直朝下）
WN = 60.0


class WeldCell:
    """手臂 + 銲槍 + 工作台 + 工件（靜態）。step() 以力矩控制追蹤關節參考並記錄。"""

    def __init__(self, render: bool = False, dt: float = 1e-3, controller: str = "ctc", wn: float = WN,
                 wn_pd: float = 30.0):
        if controller not in ("ctc", "pd"):
            raise ValueError("controller 需為 'ctc' 或 'pd'")
        self.r = SixAxisRobot()
        self.sim = PBRobot(self.r, urdf=write_arm_with_torch(self.r), plane=render)
        self.cid, self.dt, self.controller, self.wn, self.wn_pd = self.sim.cid, dt, controller, wn, wn_pd
        self.sim.make_frictionless()
        pb.setTimeStep(dt, physicsClientId=self.cid)
        self.work: dict[str, int] = {}
        self.add_box("bench", (0.50, 0.0, BENCH_TOP / 2), (0.60, 0.70, BENCH_TOP), (0.45, 0.47, 0.52, 1))
        Th = np.eye(4)
        Th[:3, :3] = np.diag([1.0, -1.0, -1.0])
        Th[:3, 3] = HOME_TCP
        self.q_home = self.r.ik(Th @ T_TCP_INV, Q_SEED)[0]
        self.sim.set_state(self.q_home)
        self._ref = (self.q_home.copy(), np.zeros(6), np.zeros(6))
        self.t, self.arc = 0.0, False
        self.rec = {k: [] for k in ("t", "q", "qd", "tau", "q_ref", "tcp", "tool_z", "tool_x", "arc",
                                    "clear")}
        self.min_clear, self._clear, self._k = np.inf, np.inf, 0
        self.beads: list[int] = []
        self.hook = None

    def close(self):
        self.sim.close()

    # --- 工件
    def add_box(self, name, center, size, rgba, quat=(0, 0, 0, 1)) -> int:
        he = [x / 2 for x in size]
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.cid)
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=rgba, physicsClientId=self.cid)
        b = pb.createMultiBody(0, col, vis, [float(c) for c in center], list(quat), physicsClientId=self.cid)
        self.work[name] = b
        return b

    def add_cylinder(self, name, center, radius, height, rgba) -> int:
        col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.cid)
        vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba,
                                   physicsClientId=self.cid)
        b = pb.createMultiBody(0, col, vis, [float(c) for c in center], physicsClientId=self.cid)
        self.work[name] = b
        return b

    def add_wall(self, seam: Seam, side: float = -1.0, height: float = 0.04, thick: float = 0.006,
                 step: float = 0.004, rgba=(0.62, 0.64, 0.68, 1)):
        """沿焊道在 side·b 側立一道牆（T 形接頭立板），牆面貼齊焊道線。以小方塊近似圓弧段。"""
        for k, s in enumerate(np.arange(step / 2, seam.length, step)):
            t = seam.tangent(float(s))
            b = np.cross(seam.n, t)
            c = seam.pos(float(s)) + side * b * thick / 2 + seam.n * height / 2
            R = np.column_stack([t, b, seam.n])
            self.add_box(f"wall_{seam.name}_{k}", c, (step * 1.3, thick, height), rgba,
                         Rotation.from_matrix(R).as_quat())

    # --- 量測
    def tcp_pose(self) -> np.ndarray:
        return self.sim.link_pose(TCP_LINK)

    def clearance(self, search: float = 0.05) -> float:
        """手臂（含銲槍，不含銲絲）與工作台/工件的最小距離 [m]。"""
        d = [search]
        for b in self.work.values():
            d += [c[8] for c in pb.getClosestPoints(self.sim.id, b, search, physicsClientId=self.cid)]
        return float(min(d))

    def precheck(self, plan: WeldPlan, every: int = 5) -> float:
        """運動學預檢：把手臂依序擺到規劃姿態（不跑物理）量最小間隙 [m]，結束後還原狀態。"""
        q0, qd0 = self.sim.get_state()
        d = np.inf
        for q in plan.q[::every]:
            self.sim.set_state(q)
            d = min(d, self.clearance())
        self.sim.set_state(q0, qd0)
        return float(d)

    # --- 控制
    def _torque(self, q, qd):
        qr, qdr, qddr = self._ref
        if self.controller == "ctc":
            a = qddr + 2 * self.wn * (qdr - qd) + self.wn**2 * (qr - q)
            tau = pb.calculateInverseDynamics(self.sim.id, list(map(float, q)), list(map(float, qd)),
                                              list(map(float, a)), physicsClientId=self.cid)
        else:  # PD + 重力補償；增益依各軸等效慣量縮放（T004）
            g = pb.calculateInverseDynamics(self.sim.id, list(map(float, q)), [0.0] * 6, [0.0] * 6,
                                            physicsClientId=self.cid)
            m = np.diag(np.array(pb.calculateMassMatrix(self.sim.id, list(map(float, q)),
                                                        physicsClientId=self.cid)))
            tau = np.asarray(g) + m * self.wn_pd**2 * (qr - q) + 2 * self.wn_pd * m * (qdr - qd)
        return np.clip(np.asarray(tau), -self.r.tau_max, self.r.tau_max)

    def step(self, q_ref=None, qd_ref=None, qdd_ref=None):
        if q_ref is not None:
            z = np.zeros(6)
            self._ref = (np.asarray(q_ref, float), z if qd_ref is None else np.asarray(qd_ref, float),
                         z if qdd_ref is None else np.asarray(qdd_ref, float))
        q, qd = self.sim.get_state()
        tau = self._torque(q, qd)
        self.sim.apply_torques(tau)
        pb.stepSimulation(physicsClientId=self.cid)
        self.t += self.dt
        self._k += 1
        if self._k % 20 == 0:
            self._clear = self.clearance()
            self.min_clear = min(self.min_clear, self._clear)
        T = self.tcp_pose()
        q1, qd1 = self.sim.get_state()
        for key, v in zip(self.rec, (self.t, q1, qd1, tau, self._ref[0], T[:3, 3].copy(), T[:3, 2].copy(),
                                     T[:3, 0].copy(), self.arc, self._clear), strict=True):
            self.rec[key].append(v)
        if self.hook:
            self.hook(self)

    def wait(self, T: float):
        for _ in range(int(round(T / self.dt))):
            self.step()

    def move_j(self, q1, T: float | None = None, util: float = 0.3):
        """關節 PTP（五次多項式）；T=None 時以速度使用率 util 求時間（≥ 0.5 s）。"""
        q0 = self._ref[0]
        q1 = np.asarray(q1, float)
        if T is None:
            T = max(min_duration(q1 - q0, self.r.qd_max * util), 0.5)
        tr = joint_ptp(q0, q1, T, "quintic")
        for k in range(1, int(round(T / self.dt)) + 1):
            self.step(*tr.state(k * self.dt))
        self.step(q1)
        return T

    def move_l(self, T_flange, T: float = 1.0) -> bool:
        """法蘭直線移動（姿態 SLERP），用於進/退刀。"""
        q0 = self._ref[0]
        smp = sample_cartesian(self.r, LinePath(self.r.fk(q0), T_flange), Quintic(T), q0, 0.01, lam=1e-10)
        spl = CubicHermiteSpline(smp.t, smp.q, smp.qd)
        d1, d2 = spl.derivative(1), spl.derivative(2)
        for k in range(1, int(round(T / self.dt)) + 1):
            tk = k * self.dt
            self.step(spl(tk), d1(tk), d2(tk))
        return smp.ik_fail == 0

    def add_bead(self, pos, radius: float = 0.0035, rgba=(1.0, 0.55, 0.1, 1.0)):
        vs = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.cid)
        self.beads.append(pb.createMultiBody(0, -1, vs, [float(x) for x in pos], physicsClientId=self.cid))

    # --- 銲一條焊道
    def weld_auto(self, seam: Seam, prm: WeldParams | None = None, rolls=(0.0, 90.0, 180.0, -90.0), **kw) -> dict:
        """自動選 roll：依 σ_min 由大到小嘗試各可行候選，第一個通過碰撞預檢者執行。"""
        prm = prm or WeldParams()
        _, rows = best_roll(self.r, seam, prm, Q_SEED, rolls)
        tried, m = [], {"seam": seam.name, "ok": False, "error": "UNREACHABLE"}
        for row in sorted((r for r in rows if r["ok"]), key=lambda r: -r["sigma_min"]):
            m = self.weld(seam, WeldParams(**{**prm.__dict__, "roll_deg": row["roll_deg"]}), **kw)
            tried.append({"roll_deg": row["roll_deg"], "sigma_min": row["sigma_min"], "error": m["error"]})
            if m["error"] != "COLLISION":
                break
        return {**m, "roll_candidates": rows, "roll_tried": tried}

    def weld(self, seam: Seam, prm: WeldParams | None = None, q_seed=None, bead_pitch: float | None = None,
             volt: float = 24.0, amp: float = 180.0, margin: float = 5e-4) -> dict:
        """進刀 → 起弧 → 等速銲接 → 收弧 → 退刀。回傳量測 dict（"ok"=False 時含 "error"）。
        錯誤碼：PLAN（焊道/時間律不成立）、UNREACHABLE（IK）、LIMIT（限位/速度/力矩）、
        COLLISION（預檢間隙 < margin，不執行；或執行中接觸）。"""
        prm = prm or WeldParams()
        try:
            plan = plan_seam(self.r, seam, prm, Q_SEED if q_seed is None else q_seed)
        except ValueError as e:
            return {"ok": False, "error": "PLAN", "message": str(e)}
        base = {"seam": seam.name, "length_mm": seam.length * 1e3, "speed_cmd_mm_s": prm.speed * 1e3,
                "work_deg": prm.work_deg, "travel_deg": prm.travel_deg, "roll_deg": prm.roll_deg,
                "blend_mm": prm.blend * 1e3,
                "plan_vel_ratio": plan.vel_ratio, "plan_torque_ratio": plan.torque_ratio,
                "sigma_min": plan.sigma_min, "ik_fail": plan.ik_fail, "plan_qdd_max": plan.meta.get("qdd_max"),
                "q_range_deg": plan.meta.get("q_range_deg")}
        if plan.ik_fail:
            return {**base, "ok": False, "error": "UNREACHABLE"}
        if not plan.ok:
            return {**base, "ok": False, "error": "LIMIT"}
        pre = self.precheck(plan)
        base["precheck_clear_mm"] = pre * 1e3
        if pre < margin:
            return {**base, "ok": False, "error": "COLLISION"}
        T0, _ = tcp_pose(seam, 0.0, prm)
        T1, _ = tcp_pose(seam, seam.length, prm)

        def back(Tt):  # 沿銲絲反方向退 approach 的法蘭位姿
            Tb = Tt.copy()
            Tb[:3, 3] -= Tt[:3, 2] * prm.approach
            return Tb @ T_TCP_INV

        q_app, ok, _ = self.r.ik_numeric(back(T0), plan.q[0], damping=1e-8)
        if not ok:
            return {**base, "ok": False, "error": "UNREACHABLE"}
        t_begin = self.t
        self.move_j(q_app)
        self.move_l(T0 @ T_TCP_INV, 0.8)
        self.step(plan.q[0])
        clear0 = self.min_clear
        self.min_clear = np.inf
        self.arc = True
        self.wait(prm.t_pre)
        i0 = len(self.rec["t"])
        prof, s_bead = SpeedProfile(seam.length, prm.speed, prm.t_ramp), 0.0
        for k in range(1, int(round(plan.T / self.dt)) + 1):
            self.step(*plan.state(k * self.dt))
            if bead_pitch and float(prof(k * self.dt)[0]) >= s_bead:
                self.add_bead(self.rec["tcp"][-1])
                s_bead += bead_pitch
        self.step(plan.q[-1])
        i1 = len(self.rec["t"])
        self.wait(prm.t_crater)
        self.arc = False
        weld_clear = self.min_clear
        self.min_clear = min(clear0, weld_clear)
        self.move_l(back(T1), 0.8)
        m = self.analyze(seam, plan, i0, i1, volt, amp)
        m.update(base)
        hit = bool(weld_clear <= 0.0)
        m.update({"ok": not hit, "error": "COLLISION" if hit else None, "weld_s": plan.T,
                  "cycle_s": self.t - t_begin, "clear_mm": weld_clear * 1e3, "collision": hit, "i0": i0, "i1": i1})
        return m

    def series(self, seam: Seam, prm: WeldParams, i0: int, i1: int) -> dict:
        """銲接段 [i0, i1) 的時間序列（供量測與繪圖）。speed 每 5 步取樣（ts）。"""
        sl = slice(i0, i1)
        t = np.array(self.rec["t"][sl])
        t = t - t[0] + self.dt
        p = np.array(self.rec["tcp"][sl])
        q, qd = np.array(self.rec["q"][sl]), np.array(self.rec["qd"][sl])
        prof = SpeedProfile(seam.length, prm.speed, prm.t_ramp)
        s_ref, sd_ref = prof(t)
        s_n, p_n, t_n = seam.nearest(p, s_ref)
        e = p - p_n
        # 實際 TCP 速度：v = Jv q̇ + ω × (p_tcp − p_flange)
        dec = slice(0, None, 5)
        spd = []
        for qk, qdk, pk in zip(q[dec], qd[dec], p[dec], strict=True):
            tw = self.r.jacobian(qk) @ qdk
            spd.append(np.linalg.norm(tw[:3] + np.cross(tw[3:], pk - self.r.fk(qk)[:3, 3])))
        work, travel = torch_angles(seam, p, np.array(self.rec["tool_z"][sl]), s_ref)
        return {"t": t, "T": prof.T, "p": p, "q": q, "qd": qd, "tau": np.array(self.rec["tau"][sl]),
                "q_ref": np.array(self.rec["q_ref"][sl]), "s": s_n, "s_ref": s_ref, "sd_ref": sd_ref,
                "lat": np.einsum("ij,ij->i", e, np.cross(seam.n, t_n)), "hgt": e @ seam.n,
                "p_ref": seam.pos(s_ref), "ts": t[dec], "speed": np.array(spd), "work": work, "travel": travel,
                "heading": heading_error(seam, p, np.array(self.rec["tool_x"][sl]), prm, s_ref),
                "clear": np.array(self.rec["clear"][sl])}

    def analyze(self, seam: Seam, plan: WeldPlan, i0: int, i1: int, volt: float = 24.0, amp: float = 180.0):
        """銲接段的量測：路徑誤差（對焊道幾何）、速度、銲槍角度、力矩/速度使用率。"""
        prm = plan.prm
        d = self.series(seam, prm, i0, i1)
        lat, hgt, p = d["lat"], d["hgt"], d["p"]
        steady = (d["ts"] > prm.t_ramp + 0.05) & (d["ts"] < plan.T - prm.t_ramp - 0.05)
        vs = d["speed"][steady]
        dwork, dtravel = d["work"] - prm.work_deg, d["travel"] - prm.travel_deg
        return {
            "path_rms_mm": float(np.sqrt(np.mean(lat**2 + hgt**2)) * 1e3),
            "path_max_mm": float(np.max(np.hypot(lat, hgt)) * 1e3),
            "lat_max_mm": float(np.max(np.abs(lat)) * 1e3), "hgt_max_mm": float(np.max(np.abs(hgt)) * 1e3),
            "lag_max_mm": float(np.max(np.abs(d["s_ref"] - d["s"])) * 1e3),
            "track_rms_mm": float(np.sqrt(np.mean(np.sum((p - d["p_ref"]) ** 2, axis=1))) * 1e3),
            "track_max_mm": float(np.max(np.linalg.norm(p - d["p_ref"], axis=1)) * 1e3),
            "speed_mean_mm_s": float(vs.mean() * 1e3), "speed_std_mm_s": float(vs.std() * 1e3),
            "speed_dev_max_pct": float(np.max(np.abs(vs / prm.speed - 1)) * 100),
            "heat_J_mm": float(heat_input(volt, amp, prm.speed)),
            "heat_dev_max_pct": float(np.max(np.abs(prm.speed / vs - 1)) * 100),
            "work_err_max_deg": float(np.max(np.abs(dwork))),
            "work_err_rms_deg": float(np.sqrt(np.mean(dwork**2))),
            "travel_err_max_deg": float(np.max(np.abs(dtravel))),
            "travel_err_rms_deg": float(np.sqrt(np.mean(dtravel**2))),
            "heading_err_max_deg": float(np.max(np.abs(d["heading"]))),
            "joint_err_max_mrad": float(np.max(np.abs(d["q"] - d["q_ref"])) * 1e3),
            "vel_util": float(np.max(np.abs(d["qd"]) / self.r.qd_max)),
            "torque_util": float(np.max(np.abs(d["tau"]) / self.r.tau_max)),
            "end_err_mm": float(np.linalg.norm(p[-1] - seam.p[-1]) * 1e3),
        }


# ------------------------------------------------------------------ JSON 工作描述
def parse_job(job) -> list[tuple[Seam, WeldParams]]:
    """JSON 銲接工作 → [(Seam, WeldParams)]。格式::

        {"seams": [{"name": "J", "start": [x,y,z], "normal": [0,0,1],
                    "segments": [{"type": "line", "to": [..]}, {"type": "arc", "via": [..], "to": [..]}],
                    "speed": 0.010, "work_deg": 90, "travel_deg": 10, "blend": 0.005}]}

    格式錯誤拋 ValueError。
    """
    if isinstance(job, str):
        job = json.loads(job)
    if not isinstance(job, dict) or not isinstance(job.get("seams"), list) or not job["seams"]:
        raise ValueError("工作需為含 'seams' 陣列的物件")
    keys = {"speed", "work_deg", "travel_deg", "roll_deg", "blend", "t_ramp", "t_pre", "t_crater", "approach"}
    out = []
    for i, sd in enumerate(job["seams"]):
        if "start" not in sd or "segments" not in sd:
            raise ValueError(f"seams[{i}] 缺少 start / segments")
        segs = []
        for g in sd["segments"]:
            if g.get("type") == "line" and "to" in g:
                segs.append(("line", g["to"]))
            elif g.get("type") == "arc" and "via" in g and "to" in g:
                segs.append(("arc", g["via"], g["to"]))
            else:
                raise ValueError(f"seams[{i}] 段格式錯誤：{g}")
        unknown = set(sd) - keys - {"name", "start", "segments", "normal"}
        if unknown:
            raise ValueError(f"seams[{i}] 未知參數 {sorted(unknown)}")
        prm = WeldParams(**{k: float(sd[k]) for k in keys if k in sd})
        if not 0 < prm.speed <= 0.5:
            raise ValueError(f"seams[{i}] speed 超出範圍 (0, 0.5] m/s")
        if not 0 < prm.work_deg < 180 or abs(prm.travel_deg) > 45:
            raise ValueError(f"seams[{i}] 銲槍角度超出範圍（work 0–180°、|travel| ≤ 45°）")
        seam = Seam(sd["start"], segs, sd.get("normal", (0, 0, 1)), name=sd.get("name", f"seam{i}"))
        out.append((seam, prm))
    return out


def run_job(cell: WeldCell, job, bead_pitch: float | None = None) -> dict:
    """依序銲完所有焊道後回原點；任何一條失敗即停止。
    焊道未指定 roll_deg 時以 :meth:`WeldCell.weld_auto` 自動選擇繞銲絲的 roll。"""
    res = []
    raw = (json.loads(job) if isinstance(job, str) else job)["seams"]
    for i, (seam, prm) in enumerate(parse_job(job)):
        fn = cell.weld if "roll_deg" in raw[i] else cell.weld_auto
        m = fn(seam, prm, bead_pitch=bead_pitch)
        res.append(m)
        if not m["ok"]:
            break
    cell.move_j(cell.q_home)
    return {"ok": all(m["ok"] for m in res), "seams": res, "t_total": cell.t}


__all__ = ["Seam", "SpeedProfile", "WeldCell", "WeldParams", "bead_area", "best_roll", "heading_error",
           "heat_input", "parse_job", "plan_seam", "run_job", "tool_rotation", "torch_angles", "turtle_seam",
           "write_arm_with_torch", "T_TCP"]
