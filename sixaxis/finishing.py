"""去毛邊/拋光（T108）：力控研磨主軸工具、墊片接觸與材料移除模型、法向力導納控制、量測。

工具（``robots/ur5_polisher.urdf``）
-----------------------------------
法蘭 → 六軸力感測器 → 主軸馬達 → 背板 → 研磨墊（半徑 ``PAD_R``）。TCP = 未壓縮墊面中心，TCP z 軸 = 主軸軸線
（朝工件）。研磨墊不設碰撞幾何：墊與工件的接觸力由下述解析模型計算並以外力/外力矩施加到工具連桿
（同 T107 起子頭的作法），避免 PyBullet 剛體接觸與模型重複計力。

墊片接觸（Winkler 彈性基礎；Johnson,《Contact Mechanics》§5.9）
--------------------------------------------------------------
工件表面以高度場 H(u, w)（沿名目法向 n，相對名目平面）表示，格距 ``res``。墊面高度 h_p = (p_TCP − c)·n。
墊面圓形足印內每格的壓縮量 d = H − h_p，局部壓力 p = k_w·max(d, 0)，k_w = k_e / (π R²) 使平面接觸的
等效剛性為 k_e [N/m]。法向力 F_n = Σ p·ΔA。

摩擦（Coulomb）：墊上一點相對工件的速度 v_i = v_TCP + ω_s·a × r_i（a = 主軸軸向、r_i 自墊心起算），
摩擦力 f_i = −μ·p_i·ΔA·v̂_i。合力與對 TCP 的合力矩一併施加；均勻壓力且 ω_s·R ≫ v 時合力 ≈ 0、
合力矩 ≈ (2/3)·μ·F_n·R（圓盤摩擦扭矩）。

材料移除（Preston 方程，Preston 1927, J. Soc. Glass Technol. 11:214）
--------------------------------------------------------------------
∂H/∂t = −K_p·p·|v_i|，|v_i| ≈ ω_s·r_i（主軸線速度 ≫ 進給）。故墊心移除率為零、外緣最大（圓盤拋光的
W 形足印）。單道沿進給方向 v_f 掃過，平面上的移除深度 ∝ K_p·F_n·ω_s / v_f。
K_p 為製程係數（研磨材、工件材料、冷卻）：本專案取代表值 6e-11 m²/N（使 20 N、25 mm/s 單道約數 µm），
非量測值。

毛邊：工件頂面沿 u 的窄稜（寬 b、高 h_b(u)），以高度場表示。硬墊（k_e 大）時毛邊承受集中壓力，
移除率遠大於母材（壓力比 = (d_b + h_b)/d_b）；軟墊則壓力分散、母材過切（gouge）增加。

法向力導納控制（Hogan 1985；Siciliano《Robotics》§9.4 間接力控）
----------------------------------------------------------------
z 為 TCP 沿 −n（壓入工件）相對名目路徑的位移，外環::

    M·z̈ + D·ż = (F_d − F_m) + K_i∫(F_d − F_m)dt,   |ż| ≤ v_app

F_m 為力感測器讀值（高斯雜訊 σ_F、一階低通 f_c）。內環：名目路徑的關節軌跡 q_n(t) 加上
Δq = J(q_n)⁻¹·[−n·z; 0]（姿態固定 → 法蘭平移 = TCP 平移），Δq̇、Δq̈ 同理，交給計算力矩控制。
可選力前饋 τ_ff = −J_TCPᵀ·w_m（以量得的接觸力螺旋抵消其對伺服的擾動）。

理想內環時接觸力動態為二階系統（平面、剛性 k_e、K_i = 0）::

    M·z̈ + D·ż + k_e·z = F_d + k_e·z_s   →   ω_a = √(k_e/M)、ζ = D / (2√(k_e·M))

步階無穩態誤差（D·ż 項無彈簧 → 對力為積分作用）；表面以速率 ż_s 變化（斜面 ż_s = v_f·tanα）時
穩態力誤差 e_ss = D·ż_s（K_i > 0 時消除）。

純位置控制（基準）：沿名目表面規劃壓入量 δ_0 = F_d / k_e，實際表面偏差 Δh 造成力誤差 k_e·Δh。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

from .robot import SixAxisRobot
from .spraying import Panel, plan_spray
from .urdf import write_urdf
from .welding import BENCH_TOP, Q_SEED, Seam, SpeedProfile, WeldCell, _unit, turtle_seam

POLISHER_URDF = Path(__file__).resolve().parents[1] / "robots" / "ur5_polisher.urdf"

# ------------------------------------------------------------------ 工具
PAD_R = 0.0125  # 研磨墊半徑
POL_L = 0.195  # 法蘭 → 墊面（未壓縮）
POL_MASS = 2.2
POL_COM = (0.0, 0.0, 0.080)
T_POL = np.eye(4)
T_POL[2, 3] = POL_L
T_POL_INV = np.linalg.inv(T_POL)
TOOL_LINK, TCP_LINK = 7, 8
RPM = 2 * np.pi / 60


def polisher_xml(parent: str = "flange") -> str:
    """研磨主軸連桿 XML：polisher（力感測器/馬達/主軸/背板，含碰撞）＋ 研磨墊（僅外觀）＋ tcp。"""

    def cyl(z0, z1, r, mat, col=True, x=0.0):
        g = (f'<origin xyz="{x} 0 {(z0 + z1) / 2:.6f}"/>'
             f'<geometry><cylinder length="{z1 - z0:.6f}" radius="{r}"/></geometry>')
        c = f"<collision>{g}</collision>" if col else ""
        return f'<visual>{g}<material name="{mat}"/></visual>' + c

    Lt = POL_MASS * 0.19**2 / 12
    U = '  <material name="pft"><color rgba="0.72 0.74 0.78 1"/></material>\n'
    U += '  <material name="pbody"><color rgba="0.85 0.45 0.10 1"/></material>\n'
    U += '  <material name="pdark"><color rgba="0.18 0.18 0.20 1"/></material>\n'
    U += '  <material name="ppad"><color rgba="0.55 0.20 0.15 1"/></material>\n'
    U += f'  <link name="polisher"><inertial><mass value="{POL_MASS}"/>'
    U += f'<origin xyz="{POL_COM[0]} {POL_COM[1]} {POL_COM[2]}"/>'
    U += f'<inertia ixx="{Lt:.6f}" iyy="{Lt:.6f}" izz="8e-4" ixy="0" ixz="0" iyz="0"/></inertial>'
    U += cyl(0.0, 0.022, 0.036, "pft")  # 六軸力感測器
    U += cyl(0.022, 0.140, 0.032, "pbody")  # 主軸馬達
    U += cyl(0.060, 0.110, 0.008, "pdark", x=0.036)  # 電纜/氣管接頭
    U += cyl(0.140, 0.172, 0.018, "pdark")  # 主軸鼻
    U += cyl(0.172, 0.184, 0.0135, "pft")  # 背板
    U += cyl(0.184, POL_L, PAD_R, "ppad", col=False)  # 研磨墊（接觸由模型處理）
    U += "</link>\n"
    U += ('  <link name="tcp"><inertial><mass value="1e-6"/>'
          '<inertia ixx="1e-12" iyy="1e-12" izz="1e-12" ixy="0" ixz="0" iyz="0"/></inertial></link>\n')
    U += f'  <joint name="pol_mount" type="fixed"><parent link="{parent}"/><child link="polisher"/></joint>\n'
    U += ('  <joint name="tcp_joint" type="fixed"><parent link="polisher"/><child link="tcp"/>'
          f'<origin xyz="0 0 {POL_L:.9f}"/></joint>\n')
    return U


def write_arm_with_polisher(robot: SixAxisRobot | None = None, path: str | Path = POLISHER_URDF) -> Path:
    """手臂 URDF + 研磨主軸。連桿 index：7 = polisher，8 = tcp（墊面中心）。"""
    robot = robot or SixAxisRobot()
    write_urdf(robot, path)
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    path.write_text(txt.replace("</robot>", polisher_xml() + "</robot>"), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 工件表面
class Surface:
    """工件頂面高度場 H(u, w)（沿 n，相對名目平面；單位 m）。
    座標同 :class:`Panel`（中心 c、行進方向 u、w = n×u）。
    `H0` 為初始（含形狀誤差與毛邊）；`base` 為不含毛邊的母材初始面；`ridge` 為毛邊格遮罩。"""

    def __init__(self, center, normal=(0, 0, 1), u=(1, 0, 0), size=(0.16, 0.10), res: float = 5e-4,
                 name: str = "work"):
        self.panel = Panel(center, normal, u, size, res=min(size) / 2, name=name)  # 只取座標系
        self.c, self.n, self.u, self.w = self.panel.c, self.panel.n, self.panel.u, self.panel.w
        self.L, self.W, self.res, self.name = float(size[0]), float(size[1]), float(res), name
        self.nu, self.nw = int(round(self.L / res)), int(round(self.W / res))
        self.gu = (np.arange(self.nu) + 0.5) * res - self.L / 2
        self.gw = (np.arange(self.nw) + 0.5) * res - self.W / 2
        self.base = np.zeros((self.nu, self.nw))
        self.ridge = np.zeros((self.nu, self.nw), bool)
        self.H = self.base.copy()
        self.H0 = self.H.copy()
        self.dA = res * res

    def set_shape(self, offset: float = 0.0, slope_u_deg: float = 0.0, slope_w_deg: float = 0.0,
                  wave_amp: float = 0.0, wave_len: float = 0.05, wave_phase: float = 0.0):
        """形狀誤差（規劃端未知）：偏移 + 沿 u/w 的斜率 + 沿 u 的正弦起伏。"""
        U, Wg = np.meshgrid(self.gu, self.gw, indexing="ij")
        self.base = (offset + U * np.tan(np.radians(slope_u_deg)) + Wg * np.tan(np.radians(slope_w_deg))
                     + wave_amp * np.sin(2 * np.pi * U / wave_len + wave_phase))
        self.H = np.where(self.ridge, self.H - self.H0 + self.base, self.base)
        self.H0 = self.H.copy()
        return self

    def add_ridge(self, w0: float, width: float, heights, u_range=None):
        """沿 u 的毛邊窄稜：中心 w0、寬 width、高 heights(u) [m]（陣列或函式）。"""
        if callable(heights):
            h = heights(self.gu)
        else:
            h = np.broadcast_to(np.asarray(heights, float), self.gu.shape)
        mw = np.abs(self.gw - w0) <= width / 2 + 1e-12
        mu = np.ones(self.nu, bool) if u_range is None else (self.gu >= u_range[0]) & (self.gu <= u_range[1])
        m = mu[:, None] & mw[None, :]
        self.ridge |= m
        self.H = np.where(m, self.base + h[:, None], self.H)
        self.H0 = self.H.copy()
        return self

    def local(self, p):
        d = np.asarray(p, float) - self.c
        return float(d @ self.u), float(d @ self.w), float(d @ self.n)

    def world(self, u, w, h=0.0):
        return self.c + u * self.u + w * self.w + h * self.n

    def window(self, u, w, R):
        """以 (u, w) 為心、半徑 R 的足印：回傳 (切片 iu, iw, 格點相對位置 du, dw, 遮罩)。"""
        i0 = max(int(np.floor((u - R + self.L / 2) / self.res)), 0)
        i1 = min(int(np.ceil((u + R + self.L / 2) / self.res)) + 1, self.nu)
        j0 = max(int(np.floor((w - R + self.W / 2) / self.res)), 0)
        j1 = min(int(np.ceil((w + R + self.W / 2) / self.res)) + 1, self.nw)
        if i0 >= i1 or j0 >= j1:
            return None
        du = self.gu[i0:i1, None] - u
        dw = self.gw[None, j0:j1] - w
        return slice(i0, i1), slice(j0, j1), du, dw, du**2 + dw**2 <= R * R

    def removal(self) -> np.ndarray:
        return self.H0 - self.H

    def burr(self) -> np.ndarray:
        """各 u 截面的殘留毛邊高 [m]：毛邊格最高點 − 鄰近（毛邊兩側 1 mm 內）母材目前高度的最大值。"""
        if not self.ridge.any():
            return np.zeros(self.nu)
        jr = np.where(self.ridge.any(0))[0]
        k = max(int(round(1e-3 / self.res)), 1)
        jn = np.r_[max(jr[0] - k, 0):jr[0], jr[-1] + 1:min(jr[-1] + 1 + k, self.nw)]
        top = np.where(self.ridge, self.H, -np.inf).max(1)
        nb = self.H[:, jn].max(1)
        return np.where(self.ridge.any(1), np.maximum(top - nb, 0.0), 0.0)


@dataclass
class PadModel:
    """研磨墊接觸與移除參數。"""

    k_e: float = 5e4  # 平面接觸等效剛性 [N/m]
    radius: float = PAD_R
    mu: float = 0.30  # 墊–工件摩擦係數
    rpm: float = 3000.0
    k_p: float = 6e-11  # Preston 係數 [m²/N]

    @property
    def k_w(self) -> float:
        return self.k_e / (np.pi * self.radius**2)

    @property
    def omega(self) -> float:
        return self.rpm * RPM

    def contact(self, surf: Surface, p_tcp, axis, v_tcp, dt: float = 0.0, spin: bool = True):
        """墊與工件接觸：回傳 (法向力 F_n、作用於工具的力 F[3]、對 TCP 的力矩 M[3]、接觸格數、最大壓縮)；
        dt > 0 且 spin 時依 Preston 更新 surf.H。"""
        u, w, h = surf.local(p_tcp)
        win = surf.window(u, w, self.radius)
        z3 = np.zeros(3)
        if win is None:
            return 0.0, z3, z3, 0, 0.0
        iu, iw, du, dw, m = win
        d = surf.H[iu, iw] - h
        c = m & (d > 0)
        nc = int(c.sum())
        if nc == 0:
            return 0.0, z3, z3, 0, float(d[m].max()) if m.any() else 0.0
        p = self.k_w * d[c]
        dA = surf.dA
        Fn = float(p.sum() * dA)
        duc = np.broadcast_to(du, d.shape)[c]
        dwc = np.broadcast_to(dw, d.shape)[c]
        r = duc[:, None] * surf.u + dwc[:, None] * surf.w  # 墊心→格點（世界）
        a = _unit(axis)
        om = self.omega if spin else 0.0
        vrel = v_tcp[None, :] + om * np.cross(a, r)
        sp = np.linalg.norm(vrel, axis=1)
        dirs = vrel / np.maximum(sp, 1e-9)[:, None]
        ff = -self.mu * (p * dA)[:, None] * dirs
        F = surf.n * Fn + ff.sum(0)
        M = np.cross(r, ff).sum(0)
        if dt > 0 and spin:
            dH = np.zeros(d.shape)
            dH[c] = self.k_p * p * sp * dt
            surf.H[iu, iw] -= dH
        return Fn, F, M, nc, float(d[c].max())


# ------------------------------------------------------------------ 力控參數與理論
@dataclass
class ForceParams:
    F_d: float = 20.0  # 期望法向力 [N]
    M: float = 2.0  # 導納質量 [kg]
    D: float = 600.0  # 導納阻尼 [N·s/m]
    K_i: float = 0.0  # 力誤差積分增益 [1/s]
    v_app: float = 0.010  # |ż| 上限（接近/觸地速度）[m/s]
    z_app: float = 0.003  # 觸地前起始高度（名目面上方）[m]
    mode: str = "admittance"  # "admittance" | "position"
    force_ff: bool = True  # 力前饋
    sigma_F: float = 0.3  # 感測雜訊 σ [N]
    f_c: float = 50.0  # 感測低通截止 [Hz]
    settle_tol: float = 0.05  # 觸地穩定判定（相對 F_d）
    t_settle: float = 0.15  # 誤差持續在容許內的時間 [s]
    t_land_max: float = 2.0


def loop_model(k_e: float, M: float, D: float) -> dict:
    """理想內環下的接觸力二階模型：ω_a、ζ、步階超越量（ζ<1）、2% 安定時間。"""
    wa = float(np.sqrt(k_e / M))
    z = float(D / (2 * np.sqrt(k_e * M)))
    os_ = float(np.exp(-np.pi * z / np.sqrt(1 - z * z))) if z < 1 else 0.0
    ts = float(4 / (z * wa)) if z < 1 else float(4 / (wa * (z - np.sqrt(z * z - 1))))
    return {"omega_a": wa, "zeta": z, "overshoot": os_, "t_settle_2pct": ts}


def predict_touchdown(k_e: float, M: float, D: float, F_d: float, v_app: float, f_c: float = 50.0,
                      lam: float | None = None, wn: float = 60.0, force_ff: bool = True, dt: float = 2e-5,
                      T: float = 1.0, series: bool = False) -> dict:
    """觸地暫態的集中參數模型（沿 n 一維）：自 ż = v_app 接觸起積分。

    * 導納：M z̈ + D ż = F_d − F_m，|ż| ≤ v_app；感測 F_m = 一階低通(F)，截止 f_c。
    * 伺服柔度（lam 為 TCP 沿 n 的等效質量 Λ_n = 1/(nᵀ J_v M_q⁻¹ J_vᵀ n)；None = 剛性內環）：
      計算力矩控制對外力的誤差動態 Λ_n(ë + 2ω_n ė + ω_n² e) = −F + F_m（力前饋時），實際位置 x = z + e。
    * 接觸：F = k_e·max(x, 0)。
    dt 取 1e-3 可近似 1 kHz 控制/接觸更新的離散效應。
    回傳峰值力、超越量、最後 0.1 s 的力標準差（> 0 表持續振盪）。"""
    z, zd, e, ed, Fm, t = 0.0, v_app, 0.0, 0.0, 0.0, 0.0
    tau = 1 / (2 * np.pi * f_c)
    a = dt / (dt + tau)
    Fs = []
    while t < T:
        F = k_e * max(z + e, 0.0)
        if lam is not None:
            edd = -2 * wn * ed - wn * wn * e + (-F + (Fm if force_ff else 0.0)) / lam
            ed += edd * dt
            e += ed * dt
        Fm += a * (F - Fm)
        zdd = (F_d - Fm - D * zd) / M
        zd = float(np.clip(zd + zdd * dt, -v_app, v_app))
        z += zd * dt
        Fs.append(F)
        t += dt
    Fs = np.array(Fs)
    tail = Fs[-max(int(0.1 / dt), 1):]
    out = {"F_peak": float(Fs.max()), "overshoot": float(Fs.max() / F_d - 1), "osc_std": float(tail.std()),
           "F_end": float(tail.mean())}
    if series:
        out.update(t=np.arange(len(Fs)) * dt, F=Fs)
    return out


def routh_kcrit(M: float, D: float, f_c: float) -> float:
    """剛性內環 + 感測一階低通（τ = 1/2πf_c）時的臨界接觸剛性：特徵式 Mτ s³ + (M + Dτ) s² + D s + k_e，
    Routh–Hurwitz 穩定條件 (M + Dτ)·D > Mτ·k_e。"""
    tau = 1 / (2 * np.pi * f_c)
    return float(D * (M + D * tau) / (M * tau))


def pass_depth(w, F_n: float, v: float, pad: PadModel) -> np.ndarray:
    """平面單道（均勻壓力 p0 = F/πR²、ω_s r ≫ v）的移除深度剖面 [m]，w 為相對道中心的橫向距離：
    h(w) = K_p·p0·ω/v · ∫_{−c}^{c} √(x² + w²) dx = K_p·p0·ω/v · (c·R + w²·ln((R + c)/|w|))，c = √(R² − w²)。
    w = 0 時 h = K_p·F·ω/(π·v)。"""
    w = np.abs(np.asarray(w, float))
    R = pad.radius
    c = np.sqrt(np.clip(R * R - w * w, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(w > 0, w * w * np.log((R + c) / np.where(w > 0, w, 1.0)), 0.0)
    p0 = F_n / (np.pi * R * R)
    return np.where(w < R, pad.k_p * p0 * pad.omega / v * (c * R + lg), 0.0)


def normal_mass(cell, q, n) -> float:
    """TCP 沿 n 的等效（操作空間）質量 Λ_n = 1/(nᵀ J_v M_q⁻¹ J_vᵀ n)，M_q 取 PyBullet（含工具）。"""
    Mq = np.array(pb.calculateMassMatrix(cell.sim.id, list(map(float, q)), physicsClientId=cell.cid))
    q0, qd0 = cell.sim.get_state()
    cell.sim.set_state(q)
    Jv = cell._jac_tcp(q)[:3]
    cell.sim.set_state(q0, qd0)
    return float(1.0 / (n @ Jv @ np.linalg.solve(Mq, Jv.T) @ n))


# ------------------------------------------------------------------ 路徑
def raster_on(surf: Surface, spacing: float, n: int, length: float, w_c: float = 0.0, name: str = "raster"):
    """工件名目平面上的 zig-zag 路徑（TCP 在名目面上）：n 道、道長 length、180° 圓弧換道。"""
    ws = w_c + (np.arange(n) - (n - 1) / 2) * spacing
    start = surf.world(-length / 2, ws[0])
    moves = []
    for k in range(n):
        moves.append(("line", length))
        if k < n - 1:
            moves.append(("arc", spacing / 2, 180.0 if k % 2 == 0 else -180.0))
    return turtle_seam(start, surf.u, moves, normal=surf.n, name=name), ws


def line_on(surf: Surface, u0: float, u1: float, w: float, name: str = "line") -> Seam:
    return Seam(surf.world(u0, w), [("line", surf.world(u1, w))], surf.n, name=name)


def tool_rot(surf: Surface) -> np.ndarray:
    """主軸軸向 = −n、TCP x = u 的固定姿態。"""
    z = -surf.n
    x = surf.u
    return np.column_stack([x, np.cross(z, x), z])


# ------------------------------------------------------------------ 工作站
@dataclass
class FinishParams:
    speed: float = 0.025  # 進給速度 [m/s]
    t_ramp: float = 0.3
    approach: float = 0.03  # 進/退刀高度
    force: ForceParams = field(default_factory=ForceParams)


class FinishCell(WeldCell):
    """手臂 + 研磨主軸 + 工作台 + 工件。法向力導納外環 + 計算力矩內環。"""

    def __init__(self, render: bool = False, pad: PadModel | None = None, seed: int = 0, **kw):
        super().__init__(render=render, tool_writer=write_arm_with_polisher, t_tcp=T_POL, **kw)
        self.pad = pad or PadModel()
        self.surf: Surface | None = None
        self.ext = (np.zeros(3), np.zeros(3))
        self.w_meas = np.zeros(6)
        self.fp = ForceParams()
        self.force_on, self.spin = False, False
        self.rng = np.random.default_rng(seed)
        self.frec = {k: [] for k in ("t", "Fn", "Fm", "z", "zd", "h_true", "nc", "dmax", "phase", "s")}
        self.phase, self._s = "idle", 0.0
        self.burr_bodies: list[tuple[int, float, float, int]] = []

    # --- 場景
    def add_surface(self, surf: Surface, thick: float = 0.02, rgba=(0.62, 0.64, 0.68, 1),
                    burr_rgba=(0.95, 0.75, 0.20, 1), burr_seg: float = 0.004, burr_vis_scale: float = 1.0):
        """工件塊（頂面 = 名目平面）＋ 毛邊外觀（分段小方塊，隨移除下沉）。
        burr_vis_scale 只放大毛邊外觀的高度（寬度放大一半），不影響模型。"""
        self.surf = surf
        self.burr_scale = float(burr_vis_scale)
        q = Rotation.from_matrix(surf.panel.rotation()).as_quat()
        self.add_box(surf.name, surf.c - surf.n * thick / 2, (surf.L, surf.W, thick), rgba, q)
        lo = surf.c - surf.n * thick
        if lo[2] > BENCH_TOP + 0.01:
            h = lo[2] - BENCH_TOP
            self.add_box(f"{surf.name}_post", (lo[0], lo[1], BENCH_TOP + h / 2), (0.08, 0.06, h - 1e-3),
                         (0.35, 0.36, 0.40, 1))
        if surf.ridge.any():
            jr = np.where(surf.ridge.any(0))[0]
            wc = float(surf.gw[jr].mean())
            width = (jr[-1] - jr[0] + 1) * surf.res
            for u0 in np.arange(-surf.L / 2 + burr_seg / 2, surf.L / 2, burr_seg):
                iu = np.abs(surf.gu - u0) <= burr_seg / 2
                if not surf.ridge[iu].any():
                    continue
                hmax = float(surf.H0[iu][:, jr].max()) * self.burr_scale
                if hmax <= 0:
                    continue
                hw = width * (self.burr_scale + 1) / 4
                vs = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[burr_seg / 2 * 0.98, hw, hmax / 2],
                                          rgbaColor=burr_rgba, physicsClientId=self.cid)
                b = pb.createMultiBody(0, -1, vs, list(surf.world(u0, wc, hmax / 2)), q,
                                       physicsClientId=self.cid)
                self.burr_bodies.append((b, float(u0), wc, hmax))

    def update_burr_visual(self):
        s = self.surf
        if s is None or not self.burr_bodies:
            return
        b_h = s.burr()
        q = Rotation.from_matrix(s.panel.rotation()).as_quat()
        for b, u0, wc, hmax in self.burr_bodies:
            iu = np.abs(s.gu - u0) <= 2e-3
            h = float(b_h[iu].max()) * self.burr_scale
            pb.resetBasePositionAndOrientation(b, list(s.world(u0, wc, h - hmax / 2)), q,
                                               physicsClientId=self.cid)

    # --- 控制（外力 + 力前饋）
    def _jac_tcp(self, q):
        J = self.r.jacobian(q)
        r = self.tcp_pose()[:3, 3] - self.r.fk(q)[:3, 3]
        Jv = J[:3] - np.cross(r, J[3:].T).T
        return np.vstack([Jv, J[3:]])

    def _torque(self, q, qd):
        tau = super()._torque(q, qd)
        if self.fp.force_ff and np.any(self.w_meas):
            tau = np.clip(tau - self._jac_tcp(q).T @ self.w_meas, -self.r.tau_max, self.r.tau_max)
        return tau

    def step(self, q_ref=None, qd_ref=None, qdd_ref=None):
        F, M = self.ext
        if F.any() or M.any():
            p = self.tcp_pose()[:3, 3]
            pb.applyExternalForce(self.sim.id, TOOL_LINK, list(map(float, F)), list(map(float, p)),
                                  pb.WORLD_FRAME, physicsClientId=self.cid)
            pb.applyExternalTorque(self.sim.id, TOOL_LINK, list(map(float, M)), pb.WORLD_FRAME,
                                   physicsClientId=self.cid)
        super().step(q_ref, qd_ref, qdd_ref)
        self._contact_update()

    def _contact_update(self):
        s = self.surf
        if s is None:
            return
        T = self.tcp_pose()
        st = pb.getLinkState(self.sim.id, TCP_LINK, computeLinkVelocity=1, physicsClientId=self.cid)
        v = np.array(st[6])
        Fn, F, M, nc, dmax = self.pad.contact(s, T[:3, 3], T[:3, 2], v, self.dt, self.spin)
        self.ext = (F, M)
        # 力感測器（量測作用於工具的接觸力螺旋；法向分量加雜訊）＋ 一階低通
        w = np.concatenate([F, M])
        if self.fp.sigma_F > 0:
            w = w + np.concatenate([s.n * self.rng.normal(0, self.fp.sigma_F), np.zeros(3)])
        a = self.dt / (self.dt + 1 / (2 * np.pi * self.fp.f_c))
        self.w_meas = self.w_meas + a * (w - self.w_meas)
        u, ww, h = s.local(T[:3, 3])
        win = s.window(u, ww, 1e-9 + s.res)
        h_true = float(s.H[win[0], win[1]].max()) if win is not None else 0.0
        r = self.frec
        for k, val in zip(r, (self.t, Fn, float(self.w_meas[:3] @ s.n), getattr(self, "_z", 0.0),
                              getattr(self, "_zd", 0.0), h_true, nc, dmax, self.phase, self._s), strict=True):
            r[k].append(val)

    # --- 導納
    def _adm_reset(self, z0: float):
        self._z, self._zd, self._zdd, self._int = z0, 0.0, 0.0, 0.0

    def _adm_step(self):
        fp = self.fp
        if fp.mode == "position":
            return
        e = fp.F_d - float(self.w_meas[:3] @ self.surf.n)
        self._int += fp.K_i * e * self.dt
        zdd = (e + self._int - fp.D * self._zd) / fp.M
        zd = float(np.clip(self._zd + zdd * self.dt, -fp.v_app, fp.v_app))
        self._zdd = (zd - self._zd) / self.dt
        self._zd = zd
        self._z += zd * self.dt

    def _ref_from(self, qn, qdn, qddn):
        """名目關節狀態 + 沿 −n 位移 z 的修正。"""
        J = self.r.jacobian(qn)
        n = self.surf.n

        def dq(x):
            return np.linalg.solve(J, np.concatenate([-n * x, np.zeros(3)]))

        return qn + dq(self._z), qdn + dq(self._zd), qddn + dq(self._zdd)

    # --- 製程
    def finish(self, seam: Seam, prm: FinishParams | None = None, q_seed=None, retract: bool = True,
               spin: bool = True) -> dict:
        """接近 → 觸地（力控）→ 沿路徑力控研磨/拋光 → 抬起退刀。回傳量測 dict（含 i0/i1、fi0/fi1）。
        錯誤碼：UNREACHABLE、LIMIT、NO_CONTACT（觸地逾時）、OVERFORCE（> 3 F_d 或 > 150 N）。"""
        prm = prm or FinishParams()
        fp = self.fp = prm.force
        s = self.surf
        R = tool_rot(s)
        plan = plan_spray(self.r, seam, R, prm.speed, Q_SEED if q_seed is None else q_seed, prm.t_ramp,
                          prm=prm, t_tool_inv=T_POL_INV)
        base = {"path_len_m": seam.length, "speed": prm.speed, "F_d": fp.F_d, "mode": fp.mode,
                "k_e": self.pad.k_e,
                "M": fp.M, "D": fp.D, "K_i": fp.K_i, "plan_vel_ratio": plan.vel_ratio,
                "plan_torque_ratio": plan.torque_ratio, "ik_fail": plan.ik_fail}
        if plan.ik_fail:
            return {**base, "ok": False, "error": "UNREACHABLE"}
        if not plan.ok:
            return {**base, "ok": False, "error": "LIMIT"}
        t_begin = self.t
        fi_begin = len(self.frec["t"])
        # 接近：名目起點上方 approach → 直線降到 z_app
        self._adm_reset(-prm.approach)
        q0n = plan.q[0]
        self.phase, self._s = "approach", 0.0
        self.move_j(self._ref_from(q0n, np.zeros(6), np.zeros(6))[0])
        self.spin = spin
        z_start = -fp.z_app if fp.mode == "admittance" else fp.F_d / self.pad.k_e
        n_app = int(round(0.6 / self.dt))
        for k in range(1, n_app + 1):
            x = k / n_app
            sm = 10 * x**3 - 15 * x**4 + 6 * x**5
            self._z = -prm.approach + (z_start + prm.approach) * sm
            self.step(*self._ref_from(q0n, np.zeros(6), np.zeros(6)))
        self._zd = 0.0
        # 觸地
        self.phase = "land"
        fl0 = len(self.frec["t"])
        t_land, ok_cnt, landed = 0.0, 0, False
        n_need = int(round(fp.t_settle / self.dt))
        while t_land < (fp.t_land_max if fp.mode == "admittance" else 0.2):
            self._adm_step()
            self.step(*self._ref_from(q0n, np.zeros(6), np.zeros(6)))
            t_land += self.dt
            Fn = self.frec["Fn"][-1]
            ok_cnt = ok_cnt + 1 if abs(Fn - fp.F_d) <= fp.settle_tol * fp.F_d else 0
            if ok_cnt >= n_need:
                landed = True
                break
        fl1 = len(self.frec["t"])
        Fl = np.array(self.frec["Fn"][fl0:fl1])
        land = {"land_s": t_land, "land_peak_N": float(Fl.max()) if len(Fl) else 0.0,
                "land_overshoot": float(Fl.max() / fp.F_d - 1) if len(Fl) else -1.0}
        if not landed and fp.mode == "admittance":
            self._retract(q0n, prm)
            return {**base, **land, "ok": False, "error": "NO_CONTACT", "cycle_s": self.t - t_begin}
        # 加工
        self.phase = "run"
        i0, fi0 = len(self.rec["t"]), len(self.frec["t"])
        prof = SpeedProfile(seam.length, prm.speed, prm.t_ramp)
        over = False
        for k in range(1, int(round(plan.T / self.dt)) + 1):
            tk = k * self.dt
            self._s = float(prof(tk)[0])
            self._adm_step()
            qn, qdn, qddn = plan.state(tk)
            self.step(*self._ref_from(qn, qdn, qddn))
            if self.frec["Fn"][-1] > max(3 * fp.F_d, 150.0):
                over = True
                break
            if self.burr_bodies and k % 50 == 0:
                self.update_burr_visual()
        i1, fi1 = len(self.rec["t"]), len(self.frec["t"])
        qn_end = plan.state(min(k * self.dt, plan.T))[0]
        if retract:
            self._retract(qn_end, prm)
        self.update_burr_visual()
        m = {**base, **land, "ok": not over, "error": "OVERFORCE" if over else None, "run_s": plan.T,
             "cycle_s": self.t - t_begin, "i0": i0, "i1": i1, "fi0": fi0, "fi1": fi1, "fi_begin": fi_begin}
        m.update(self.force_metrics(fi0, fi1, prm))
        m.update(self.motion_metrics(seam, prm, i0, i1))
        return m

    def _retract(self, qn, prm: FinishParams):
        self.phase = "retract"
        z0 = self._z
        n = int(round(0.5 / self.dt))
        for k in range(1, n + 1):
            x = k / n
            self._z = z0 + (-prm.approach - z0) * (10 * x**3 - 15 * x**4 + 6 * x**5)
            self._zd = self._zdd = 0.0
            self.step(*self._ref_from(qn, np.zeros(6), np.zeros(6)))
        self.spin = False
        self.phase = "idle"

    # --- 量測
    def force_metrics(self, fi0: int, fi1: int, prm: FinishParams, skip: float | None = None) -> dict:
        """加工段力量測（排除起/停 t_ramp 段）：RMS/最大誤差、脫離比例、最大力。"""
        fp = prm.force
        t = np.array(self.frec["t"][fi0:fi1])
        if len(t) == 0:
            return {}
        t = t - t[0]
        F = np.array(self.frec["Fn"][fi0:fi1])
        skip = prm.t_ramp if skip is None else skip
        st = (t >= skip) & (t <= t[-1] - skip)
        e = F[st] - fp.F_d
        return {"F_mean": float(F[st].mean()), "F_rms_err": float(np.sqrt(np.mean(e**2))),
                "F_max_err": float(np.abs(e).max()), "F_max": float(F.max()), "F_min": float(F[st].min()),
                "loss_frac": float(np.mean(F[st] < 0.2 * fp.F_d)), "F_std": float(F[st].std())}

    def motion_metrics(self, seam: Seam, prm: FinishParams, i0: int, i1: int) -> dict:
        sl = slice(i0, i1)
        t = np.array(self.rec["t"][sl])
        t = t - t[0] + self.dt
        p = np.array(self.rec["tcp"][sl])
        s_ref, _ = SpeedProfile(seam.length, prm.speed, prm.t_ramp)(t)
        _, p_n, t_n = seam.nearest(p, s_ref, window=0.02)
        e = p - p_n
        lat = np.einsum("ij,ij->i", e, np.cross(seam.n, t_n))
        tz = np.array(self.rec["tool_z"][sl])
        ax = np.degrees(np.arccos(np.clip(tz @ -seam.n, -1, 1)))
        return {"lat_max_mm": float(np.abs(lat).max() * 1e3),
                "lat_rms_mm": float(np.sqrt(np.mean(lat**2)) * 1e3),
                "axis_err_max_deg": float(ax.max()),
                "torque_util": float(np.max(np.abs(np.array(self.rec["tau"][sl])) / self.r.tau_max)),
                "vel_util": float(np.max(np.abs(np.array(self.rec["qd"][sl])) / self.r.qd_max))}

    def fseries(self, fi0: int, fi1: int) -> dict:
        return {k: np.array(v[fi0:fi1]) for k, v in self.frec.items() if k != "phase"}


def removal_stats(surf: Surface, u_lim, w_lim) -> dict:
    """矩形區域 |u| ≤ u_lim、|w| ≤ w_lim 內的移除深度統計（µm）。"""
    R = surf.removal()
    m = (np.abs(surf.gu)[:, None] <= u_lim) & (np.abs(surf.gw)[None, :] <= w_lim) & ~surf.ridge
    r = R[m] * 1e6
    return {"mean_um": float(r.mean()), "std_um": float(r.std()),
            "cv": float(r.std() / r.mean()) if r.mean() else 0.0,
            "min_um": float(r.min()), "max_um": float(r.max())}
