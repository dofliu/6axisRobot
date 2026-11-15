"""鎖螺絲（T107）：電動起子（nutrunner）工具、螺紋鎖固力學、主軸動力學、定位/姿態/旋轉的工作流程與品質監控。

螺絲與接合（M4×12 內六角螺栓 ISO 4762、強度等級 8.8，鎖入鋼製螺紋孔，夾緊 3 mm 鋼板）
--------------------------------------------------------------------------------------
* 螺紋幾何（ISO 724）：d2 = d − 0.649519·P、d3 = d − 1.226869·P；應力截面 A_s = π/4·((d2+d3)/2)²。
* 扭力–夾緊力（VDI 2230 Part 1 (2015) 式 (R13)；ISO 16047 亦同）::

      T_A = F_M · (0.16·P + 0.58·μ_G·d2 + μ_K·D_Km/2) = F_M · K'

  0.16P 為導程（彈性、可逆）項，0.58μ_G·d2 為螺紋摩擦（60° 牙角），μ_K·D_Km/2 為頭下摩擦，
  D_Km = (d_w + d_h)/2（頭部承面外徑與通孔徑平均）。
* 自鎖條件：導程角 φ = atan(P/(π·d2)) < 螺紋摩擦角 ρ' = atan(μ_G/cos30°)。M4、μ = 0.08 時
  φ = 3.6° < ρ' = 5.3°，故停轉後夾緊力保持（模型在主軸停止後凍結轉角）。
* 接合剛性（VDI 2230 柔度法，簡化）：螺栓 δ_S = (0.5d/A_N + l_K/A_d3 + 0.5d/A_d3 + 0.33d/A_N)/E
  （頭部、夾持段、嚙合螺紋、螺紋孔各段；0.33d 為螺紋孔接合的替代長度），被夾件以頭部承面環
  A = π/4·(d_w² − d_h²) 近似 δ_P = l_K/(E·A)。c = 1/(δ_S + δ_P)。座合後夾緊力 F = c·P·Δθ/(2π)。
* 應力利用率（VDI 2230 式 (R14)、k_τ = 1）：σ = F/A_s、τ = F·(0.16P + 0.58μ_G·d2)/W_p，
  W_p = π·d0³/16、d0 = (d2+d3)/2；σ_red = √(σ² + 3τ²)，ν = σ_red / R_p0.2（ISO 898-1：8.8 → 640 MPa）。

鎖固階段（依螺絲轉角 θ，自螺紋開始嚙合起算）
--------------------------------------------
* 旋入（rundown）：深度 z = P·θ/(2π)，負載扭力 = 殘留摩擦扭力 T_pv + F_ax·(0.16P + 0.58μ_G·d2)（起子進給力）。
* 座合（θ ≥ θ_s = 2π·l_e/P，l_e = L − t_b 為嚙合長度）：F = c·P·(θ − θ_s)/(2π)，T = T_pv + F·K'。
* 斜牙（cross-thread）：起始嚙合時螺絲軸對孔軸傾角 > φ_ct 則咬錯牙，旋入扭力額外 k_ct·θ（牙面擠壓）。
  φ_ct 的簡化幾何判據：傾角使螺絲端面兩側沿軸向相差 d·tanφ，若 > P/2 則牙峰落入錯誤牙溝，
  φ_ct = atan(P/(2d))（M4：5.0°）。此為本專案的簡化判據，非標準值。

電動起子主軸（以 h = 0.1 ms 子步積分，控制器 10 kHz 取樣）
------------------------------------------------------------
J_s·ω̇ = T_m − T_load，速度 PI 控制 T_m = sat(K_p(ω_ref − ω) + K_i∫(ω_ref − ω))，|T_m| ≤ T_max。
二段式策略：快速旋入 n_run → 預期座合前 n_slow 圈降速 n_fin → 到目標扭力（或 snug 後轉角）即切斷並
以 −T_max 主動煞車至 ω ≤ 0。過衝的能量估計（座合段 k_θ = c·P·K'/(2π)）：
½·J·ω² ≈ ½·k_θ·Δθ² + (T_A + T_max)·Δθ，ΔT ≈ k_θ·Δθ（另加取樣延遲 k_θ·ω·h）。
反作用：馬達定子承受 −T_m（繞螺絲軸）→ 以外力矩施加到工具連桿；起子彈簧力 F_ax 反推工具。
機器人可用起子內建扭力/行程感測做反作用前饋 τ_ff = −J_tcpᵀ·w_ext。

起子頭浮動彈簧與定位
--------------------
TCP = 彈簧在標稱壓縮量 c_nom 時的起子頭尖端。壓縮 c = c_nom + (z_tcp − z_head)（沿螺絲軸、朝材料為正），
F_ax = F_0 + k_s·c（c > 0）；c ≤ 0 起子頭脫離（BIT_LIFT）、c ≥ stroke 觸底（OVERFORCE）。
螺絲端經過板面時若 TCP 對真孔軸的徑向誤差 ≤ e_cap（通孔間隙 (d_h − d)/2 + 倒角 0.5 mm）則入孔，
否則頂在板面（壓縮增加 → 偵測 NO_ENGAGE，可選阿基米德螺旋搜尋：間距 < 2·e_cap 保證覆蓋）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

from .robot import SixAxisRobot
from .trajectory import LinePath, Quintic, sample_cartesian
from .urdf import write_urdf
from .welding import BENCH_TOP, WeldCell, _unit

DRIVER_URDF = Path(__file__).resolve().parents[1] / "robots" / "ur5_driver.urdf"

# ------------------------------------------------------------------ 起子工具
DRIVER_L = 0.270  # 法蘭 → 起子頭尖端（彈簧標稱壓縮時）
DRIVER_MASS = 1.6
DRIVER_COM = (0.0, 0.0, 0.105)
T_DRV = np.eye(4)
T_DRV[2, 3] = DRIVER_L
T_DRV_INV = np.linalg.inv(T_DRV)
TOOL_LINK, TCP_LINK = 7, 8


def driver_xml(parent: str = "flange") -> str:
    """起子連桿 XML：driver（馬達/減速機/扭力感測器/浮動套筒，含碰撞）＋ tcp（起子頭尖端）。"""

    def cyl(z0, z1, r, mat, col=True, x=0.0):
        g = (f'<origin xyz="{x} 0 {(z0 + z1) / 2:.6f}"/>'
             f'<geometry><cylinder length="{z1 - z0:.6f}" radius="{r}"/></geometry>')
        c = f"<collision>{g}</collision>" if col else ""
        return f'<visual>{g}<material name="{mat}"/></visual>' + c

    Lt = DRIVER_MASS * 0.24**2 / 12
    U = '  <material name="dbody"><color rgba="0.16 0.30 0.55 1"/></material>\n'
    U += '  <material name="dgear"><color rgba="0.72 0.74 0.78 1"/></material>\n'
    U += '  <material name="dsleeve"><color rgba="0.25 0.25 0.28 1"/></material>\n'
    U += '  <material name="dbit"><color rgba="0.85 0.86 0.88 1"/></material>\n'
    U += f'  <link name="driver"><inertial><mass value="{DRIVER_MASS}"/>'
    U += f'<origin xyz="{DRIVER_COM[0]} {DRIVER_COM[1]} {DRIVER_COM[2]}"/>'
    U += f'<inertia ixx="{Lt:.6f}" iyy="{Lt:.6f}" izz="6e-4" ixy="0" ixz="0" iyz="0"/></inertial>'
    U += cyl(0.0, 0.025, 0.034, "dsleeve")  # 安裝座
    U += cyl(0.025, 0.150, 0.026, "dbody")  # 馬達
    U += cyl(0.060, 0.120, 0.008, "dgear", x=0.030)  # 電纜接頭
    U += cyl(0.150, 0.195, 0.022, "dgear")  # 減速機 + 扭力感測器
    U += cyl(0.195, 0.235, 0.011, "dsleeve")  # 浮動彈簧套筒
    U += cyl(0.235, DRIVER_L, 0.0022, "dbit", col=False)  # 起子頭（不參與碰撞：與螺絲接觸由模型處理）
    U += "</link>\n"
    U += ('  <link name="tcp"><inertial><mass value="1e-6"/>'
          '<inertia ixx="1e-12" iyy="1e-12" izz="1e-12" ixy="0" ixz="0" iyz="0"/></inertial></link>\n')
    U += (f'  <joint name="driver_mount" type="fixed"><parent link="{parent}"/>'
          '<child link="driver"/></joint>\n')
    U += ('  <joint name="tcp_joint" type="fixed"><parent link="driver"/><child link="tcp"/>'
          f'<origin xyz="0 0 {DRIVER_L:.9f}"/></joint>\n')
    return U


def write_arm_with_driver(robot: SixAxisRobot | None = None, path: str | Path = DRIVER_URDF) -> Path:
    """手臂 URDF + 電動起子。連桿 index：7 = driver，8 = tcp（起子頭尖端）。"""
    robot = robot or SixAxisRobot()
    write_urdf(robot, path)
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    path.write_text(txt.replace("</robot>", driver_xml() + "</robot>"), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 螺絲與接合力學
E_STEEL = 2.05e11


@dataclass
class ScrewSpec:
    """M4×12 內六角螺栓（ISO 4762）、8.8 級（ISO 898-1）。長度單位 m。"""

    d: float = 4.0e-3
    P: float = 0.7e-3
    L: float = 12.0e-3  # 螺桿長
    dk: float = 7.0e-3  # 頭部外徑（= 承面外徑 d_w 近似）
    k: float = 4.0e-3  # 頭高
    dh: float = 4.5e-3  # 通孔徑（ISO 273 medium）
    Rp: float = 640e6  # R_p0.2

    @property
    def d2(self) -> float:
        return self.d - 0.649519 * self.P

    @property
    def d3(self) -> float:
        return self.d - 1.226869 * self.P

    @property
    def As(self) -> float:
        return np.pi / 4 * ((self.d2 + self.d3) / 2) ** 2

    @property
    def Dkm(self) -> float:
        return (self.dk + self.dh) / 2

    @property
    def phi_ct_deg(self) -> float:
        """斜牙臨界傾角（簡化幾何判據）atan(P/(2d))。"""
        return float(np.degrees(np.arctan(self.P / (2 * self.d))))


@dataclass
class JointSpec:
    """被鎖固接合：夾緊板厚 t_b、摩擦係數、剛性（None → 由 VDI 2230 柔度法計算）。"""

    t_b: float = 3.0e-3
    mu_g: float = 0.12  # 螺紋摩擦
    mu_k: float = 0.12  # 頭下摩擦
    c: float | None = None  # 接合軸向剛性 [N/m]
    T_pv: float = 0.03  # 旋入殘留摩擦扭力 [N·m]
    k_ct: float = 0.10  # 斜牙時旋入扭力斜率 [N·m/rad]


def joint_stiffness(s: ScrewSpec, j: JointSpec, E: float = E_STEEL) -> float:
    """VDI 2230 柔度法（簡化）：δ_S（頭、夾持段、嚙合螺紋、螺紋孔）+ δ_P（承面環）。"""
    AN, Ad3 = np.pi / 4 * s.d**2, np.pi / 4 * s.d3**2
    dS = (0.5 * s.d / AN + j.t_b / Ad3 + 0.5 * s.d / Ad3 + 0.33 * s.d / AN) / E
    dP = j.t_b / (E * np.pi / 4 * (s.dk**2 - s.dh**2))
    return float(1.0 / (dS + dP))


class ThreadModel:
    """螺絲轉角 θ → 深度、夾緊力、負載扭力（見模組說明）。"""

    def __init__(self, screw: ScrewSpec | None = None, joint: JointSpec | None = None):
        self.s = screw or ScrewSpec()
        self.j = joint or JointSpec()
        self.c = self.j.c if self.j.c is not None else joint_stiffness(self.s, self.j)
        s, j = self.s, self.j
        self.k_th = 0.16 * s.P + 0.58 * j.mu_g * s.d2  # 螺紋項 [m]
        self.K = self.k_th + j.mu_k * s.Dkm / 2  # K' [m]
        self.l_e = s.L - j.t_b  # 嚙合長度
        self.theta_s = 2 * np.pi * self.l_e / s.P  # 座合轉角
        self.k_theta = self.c * s.P / (2 * np.pi) * self.K  # 座合後扭力–轉角斜率 [N·m/rad]
        self.cross = False

    def depth(self, th: float) -> float:
        return min(self.s.P * th / (2 * np.pi), self.l_e)

    def clamp(self, th: float) -> float:
        return max(0.0, self.c * self.s.P * (th - self.theta_s) / (2 * np.pi))

    def torque(self, th: float, f_ax: float = 0.0) -> float:
        if th < self.theta_s:
            t = self.j.T_pv + f_ax * self.k_th
            return t + (self.j.k_ct * th if self.cross else 0.0)
        return self.j.T_pv + self.clamp(th) * self.K

    def torque_to_force(self, T: float) -> float:
        return max(0.0, (T - self.j.T_pv) / self.K)

    def utilization(self, F: float) -> float:
        """VDI 2230 等效應力利用率 ν = σ_red / R_p0.2。"""
        s = self.s
        d0 = (s.d2 + s.d3) / 2
        sig = F / s.As
        tau = F * self.k_th / (np.pi * d0**3 / 16)
        return float(np.sqrt(sig**2 + 3 * tau**2) / s.Rp)

    def self_locking(self) -> bool:
        lead = np.arctan(self.s.P / (np.pi * self.s.d2))
        return bool(lead < np.arctan(self.j.mu_g / np.cos(np.radians(30))))

    def overshoot_estimate(self, omega: float, J: float, T_target: float, T_brake: float,
                           h: float = 1e-4) -> float:
        """能量法過衝估計 ΔT [N·m]（座合段、主動煞車、取樣延遲）。"""
        k = self.k_theta
        b = T_target + T_brake
        dth = (-b + np.sqrt(b * b + k * J * omega**2)) / k
        return float(k * (dth + omega * h))


# ------------------------------------------------------------------ 主軸（起子馬達）
RPM = 2 * np.pi / 60


@dataclass
class DriverParams:
    T_target: float = 2.5  # 目標扭力 [N·m]
    strategy: str = "torque"  # "torque"（扭力控制）／"angle"（扭力–轉角：snug 後轉 angle_deg）
    angle_deg: float | None = None  # None → 依標稱模型由 T_target 換算
    snug_frac: float = 0.2  # snug 扭力 = snug_frac·T_target
    n_run: float = 600.0  # 旋入轉速 [rpm]
    n_fin: float = 60.0  # 最終轉速 [rpm]
    n_slow: float = 0.5  # 二段式：預期座合前多少圈降速
    two_stage: bool = True
    J: float = 3.0e-4  # 主軸等效慣量（輸出軸）[kg·m²]
    T_max: float = 4.0  # 馬達最大扭力 [N·m]
    w_bw: float = 300.0  # 速度迴路頻寬 [rad/s]
    h: float = 1e-4  # 子步 / 控制週期 [s]
    # 監控窗（相對標稱值）
    turns_tol: float = 0.5  # 旋入圈數 ±
    angle_win: tuple = (0.5, 1.7)  # snug→停止 轉角 / 標稱
    torque_win: tuple = (0.95, 1.10)  # 峰值扭力 / 目標（扭力控制）
    torque_win_angle: tuple = (0.70, 1.30)  # 轉角控制時的扭力監控窗
    theta_over: float = 2.0  # 超過預期座合轉角幾圈仍未 snug → NO_SNUG


class Nutrunner:
    """主軸動力學 + 鎖固策略狀態機 + 品質監控。step(dt, f_ax, dphi_tool) 以 h 子步積分。"""

    def __init__(self, model: ThreadModel, prm: DriverParams | None = None,
                 nominal: ThreadModel | None = None):
        self.m, self.p = model, prm or DriverParams()
        nom = nominal or model  # 控制器只知道標稱接合（窗口與轉角換算）
        self.nom = nom
        p = self.p
        self.T_snug = p.snug_frac * p.T_target
        th_snug_nom = nom.theta_s + 2 * np.pi * nom.torque_to_force(self.T_snug) / (nom.c * nom.s.P)
        th_tgt_nom = nom.theta_s + 2 * np.pi * nom.torque_to_force(p.T_target) / (nom.c * nom.s.P)
        self.angle_nom = th_tgt_nom - th_snug_nom
        self.angle_cmd = np.radians(p.angle_deg) if p.angle_deg is not None else self.angle_nom
        self.turns_nom = th_snug_nom / (2 * np.pi)
        self.theta_slow = nom.theta_s - 2 * np.pi * p.n_slow if p.two_stage else np.inf
        self.Kp = p.J * p.w_bw
        self.Ki = self.Kp * p.w_bw / 4
        self.th = self.w = self.ie = self.Tm = self.Tl = 0.0
        self.phase = "run"
        self.th_snug = None
        self.T_peak = 0.0
        self.w_cut = self.T_cut = None
        self.t = 0.0
        self.t_done = None
        self.log = {k: [] for k in ("t", "th", "w", "Tm", "Tl", "F", "phase")}

    @property
    def done(self) -> bool:
        return self.phase in ("done", "abort")

    def w_ref(self) -> float:
        p = self.p
        if self.phase != "run":
            return 0.0
        n = p.n_fin if self.th >= self.theta_slow else p.n_run
        return n * RPM

    def step(self, dt: float, f_ax: float = 0.0, dphi_tool: float = 0.0):
        p = self.p
        nsub = max(int(round(dt / p.h)), 1)
        for _ in range(nsub):
            if self.done:  # 自鎖：螺絲轉角凍結
                self.Tm, self.w = 0.0, 0.0
                continue
            self.th += dphi_tool / nsub  # 工具繞軸轉動 → 螺絲絕對轉角（主軸編碼器量相對角）
            if self.phase == "run":
                e = self.w_ref() - self.w
                self.ie += e * p.h
                u = self.Kp * e + self.Ki * self.ie
                self.Tm = float(np.clip(u, -p.T_max, p.T_max))
                if self.Tm != u:  # 抗積分飽和
                    self.ie -= e * p.h
            else:  # brake
                self.Tm = -p.T_max
            Tl = self.m.torque(self.th, f_ax)
            self.w += (self.Tm - Tl) * p.h / p.J
            if self.phase == "brake" and self.w <= 0.0:
                self.w = 0.0  # 自鎖：停止後轉角凍結
                self.phase = "done"
                self.t_done = self.t
            self.th += max(self.w, 0.0) * p.h
            self.w = max(self.w, 0.0)
            self.Tl = self.m.torque(self.th, f_ax)  # 扭力感測器讀值（嚙合處負載）
            self.T_peak = max(self.T_peak, self.Tl)
            self.t += p.h
            if self.phase == "run":
                if self.th_snug is None and self.Tl >= self.T_snug:
                    self.th_snug = self.th
                stop = (self.Tl >= p.T_target) if p.strategy == "torque" else (
                    self.th_snug is not None and self.th - self.th_snug >= self.angle_cmd)
                if stop:
                    self.phase = "brake"
                    self.w_cut, self.T_cut = self.w, self.Tl
                elif self.th_snug is None and self.th > self.nom.theta_s + 2 * np.pi * p.theta_over:
                    self.phase = "abort"
                    self.t_done = self.t
        lg = self.log
        for k, v in zip(lg, (self.t, self.th, self.w, self.Tm, self.Tl, self.m.clamp(self.th), self.phase),
                        strict=True):
            lg[k].append(v)

    def result(self) -> dict:
        """品質判定（扭力窗、旋入圈數窗、snug 後轉角窗）。"""
        p, m = self.p, self.m
        F = m.clamp(self.th)
        r = {"T_peak": self.T_peak, "T_err_pct": (self.T_peak / p.T_target - 1) * 100, "F_clamp": F,
             "theta_final_deg": float(np.degrees(self.th)), "util": m.utilization(F),
             "turns_rundown": None if self.th_snug is None else self.th_snug / (2 * np.pi),
             "angle_snug_deg": None if self.th_snug is None else float(np.degrees(self.th - self.th_snug)),
             "t_spindle": self.t_done, "turns_nom": self.turns_nom,
             "angle_nom_deg": float(np.degrees(self.angle_nom)),
             "w_cut": self.w_cut, "T_cut": self.T_cut}
        err = None
        if self.phase == "abort":
            err = "NO_SNUG"
        elif self.phase != "done":
            err = "TIMEOUT"
        elif abs(r["turns_rundown"] - self.turns_nom) > p.turns_tol:
            err = "CROSS_THREAD" if r["turns_rundown"] < self.turns_nom else "LATE_SNUG"
        elif not (lo_hi := p.torque_win if p.strategy == "torque" else p.torque_win_angle)[0] \
                <= self.T_peak / p.T_target <= lo_hi[1]:
            err = "TORQUE_NG"
        elif p.strategy == "torque" and not (
                p.angle_win[0] <= (self.th - self.th_snug) / self.angle_nom <= p.angle_win[1]):
            err = "ANGLE_NG"
        r["error"] = err
        r["ok"] = err is None
        return r


def bench_test(model: ThreadModel, prm: DriverParams, f_ax: float = 25.0, t_max: float = 10.0,
               nominal: ThreadModel | None = None) -> tuple[dict, Nutrunner]:
    """接合模擬台（不含手臂；如 ISO 5393 的工具測試台）：固定進給力下鎖到停止。"""
    nr = Nutrunner(model, prm, nominal)
    while not nr.done and nr.t < t_max:
        nr.step(1e-3, f_ax)
    return nr.result(), nr


# ------------------------------------------------------------------ 工作站
@dataclass
class Hole:
    """螺紋孔：p = 板面（螺絲頭座合面）上的孔中心，a = 鎖入方向（朝材料）。"""

    name: str
    p: np.ndarray
    a: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))

    def __post_init__(self):
        self.p = np.asarray(self.p, float)
        self.a = _unit(self.a)


@dataclass
class FastenParams:
    driver: DriverParams = field(default_factory=DriverParams)
    c_nom: float = 5.0e-3  # 彈簧標稱壓縮
    stroke: float = 10.0e-3
    k_s: float = 3000.0  # 彈簧剛性 [N/m]
    F0: float = 10.0  # 彈簧預壓 [N]
    feed: str = "follow"  # "follow"：依主軸編碼器跟隨旋入深度；"fixed"：手臂不動（只靠彈簧）
    c_start: float | None = None  # fixed 模式起始壓縮（None → c_nom）
    approach: float = 0.03
    t_descent: float = 0.5
    e_cap_extra: float = 0.5e-3  # 通孔倒角
    search: bool = True
    search_pitch: float = 1.0e-3
    search_v: float = 0.010
    search_rmax: float = 3.0e-3
    react_ff: bool = True  # 反作用力前饋
    hole_err: tuple = (0.0, 0.0, 0.0)  # 孔位實際偏差（手臂不知道）
    tilt_err_deg: float = 0.0  # 注入的工具傾角誤差
    pick: bool = True
    t_spin_max: float = 30.0  # 主軸逾時 [s]


def tool_rotation(a) -> np.ndarray:
    """TCP z 軸 = a；以「工具朝下的原點姿態 diag(1,−1,−1)」經最小旋轉得到。"""
    a = _unit(a)
    R0 = np.diag([1.0, -1.0, -1.0])
    z0 = R0[:, 2]
    ax = np.cross(z0, a)
    s, c = np.linalg.norm(ax), float(z0 @ a)
    if s < 1e-12:
        return R0 if c > 0 else np.diag([1.0, 1.0, 1.0]) @ np.diag([1, -1, 1]) @ R0
    return Rotation.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix() @ R0


FEEDER = np.array([0.32, 0.28, BENCH_TOP + 0.040])  # 供料器：螺絲頭頂面位置（立在料槽中）


class ScrewCell(WeldCell):
    """手臂 + 電動起子 + 工作台 + 底座/夾緊板 + 螺絲供料器。"""

    def __init__(self, render: bool = False, screw: ScrewSpec | None = None, **kw):
        super().__init__(render=render, tool_writer=write_arm_with_driver, t_tcp=T_DRV, **kw)
        self.screw = screw or ScrewSpec()
        self.holes: dict[str, Hole] = {}
        self.screws: list[int] = []
        self.ext = (np.zeros(3), np.zeros(3))  # 施加於 TCP 的外力、外力矩（世界座標）
        self.react_ff = True
        self.nr: Nutrunner | None = None
        self.srec = {k: [] for k in ("t", "c", "F_ax", "lat", "Tm", "Tl", "th", "w", "F", "roll", "phase")}
        self._sstate = None
        self.add_box("feeder", (FEEDER[0], FEEDER[1], (BENCH_TOP + FEEDER[2] - 0.016) / 2),
                     (0.05, 0.05, FEEDER[2] - 0.016 - BENCH_TOP), (0.85, 0.55, 0.12, 1))

    # --- 場景
    def add_plate(self, name, center, size, n=(0, 0, 1), base_h: float = 0.02, t_b: float = 3e-3,
                  rgba_base=(0.55, 0.57, 0.60, 1), rgba_plate=(0.20, 0.45, 0.75, 1)):
        """底座（螺紋孔）+ 夾緊板；回傳板面（頂面）中心與法向量。center 為底座底面中心。"""
        n = _unit(n)
        R = tool_rotation(-n) @ np.diag([1.0, -1.0, -1.0])  # z = n
        q = Rotation.from_matrix(R).as_quat()
        c = np.asarray(center, float)
        self.add_box(f"{name}_base", c + n * base_h / 2, (size[0], size[1], base_h), rgba_base, q)
        self.add_box(f"{name}_plate", c + n * (base_h + t_b / 2), (size[0] * 0.8, size[1] * 0.8, t_b),
                     rgba_plate, q)
        return c + n * (base_h + t_b), n, R

    def add_hole(self, name, p, a=(0, 0, -1), visual: bool = True) -> Hole:
        h = Hole(name, p, a)
        self.holes[name] = h
        if visual:
            R = tool_rotation(h.a)
            vs = pb.createVisualShape(pb.GEOM_CYLINDER, radius=self.screw.dh / 2, length=4e-4,
                                      rgbaColor=(0.05, 0.05, 0.06, 1), physicsClientId=self.cid)
            pb.createMultiBody(0, -1, vs, list(h.p - h.a * 1e-4), Rotation.from_matrix(R).as_quat(),
                               physicsClientId=self.cid)
        return h

    def make_screw(self) -> int:
        s = self.screw
        shapes = [pb.GEOM_CYLINDER, pb.GEOM_CYLINDER, pb.GEOM_BOX]
        vs = pb.createVisualShapeArray(
            shapes, radii=[s.dk / 2, s.d / 2, 0], lengths=[s.k, s.L, 0],
            halfExtents=[[0, 0, 0], [0, 0, 0], [s.dk * 0.42, 6e-4, 3e-4]],
            rgbaColors=[[0.12, 0.12, 0.13, 1], [0.55, 0.56, 0.58, 1], [0.95, 0.15, 0.10, 1]],
            visualFramePositions=[[0, 0, s.k / 2], [0, 0, s.k + s.L / 2], [0, 0, -2e-4]],
            physicsClientId=self.cid)
        b = pb.createMultiBody(0, -1, vs, [0, 0, -1], physicsClientId=self.cid)
        self.screws.append(b)
        return b

    def place_screw(self, b, head_top, a, x_dir):
        z = _unit(a)
        x = _unit(x_dir - (x_dir @ z) * z)
        R = np.column_stack([x, np.cross(z, x), z])
        pb.resetBasePositionAndOrientation(b, [float(v) for v in head_top], Rotation.from_matrix(R).as_quat(),
                                           physicsClientId=self.cid)

    # --- 控制（加入外力與反作用前饋）
    def _jac_tcp(self, q):
        J = self.r.jacobian(q)
        r = self.tcp_pose()[:3, 3] - self.r.fk(q)[:3, 3]
        Jv = J[:3] - np.cross(r, J[3:].T).T
        return np.vstack([Jv, J[3:]])

    def _torque(self, q, qd):
        tau = super()._torque(q, qd)
        F, M = self.ext
        if self.react_ff and (F.any() or M.any()):
            tau = tau - self._jac_tcp(q).T @ np.concatenate([F, M])
            tau = np.clip(tau, -self.r.tau_max, self.r.tau_max)
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
        if self._sstate is not None:
            self._screw_update()

    # --- 螺絲/起子狀態（每個控制週期）
    def _axial(self, hole: Hole, T_tcp):
        """TCP 沿孔軸座標 z（板面 = 0、朝材料為正）與對真孔軸的徑向誤差。"""
        d = T_tcp[:3, 3] - hole.p
        z = float(d @ hole.a)
        return z, float(np.linalg.norm(d - z * hole.a))

    def _screw_update(self):
        st = self._sstate
        hole, prm, s = st["hole"], st["prm"], self.screw
        T = self.tcp_pose()
        z_tcp, lat = self._axial(hole, T)
        roll = float(np.arctan2(T[:3, 0] @ st["e2"], T[:3, 0] @ st["e1"]))
        droll = (roll - st["roll"] + np.pi) % (2 * np.pi) - np.pi
        st["roll"] = roll
        st["roll_acc"] += droll
        mode = st["mode"]
        if mode == "carry":  # 螺絲剛性跟著起子頭（未接觸時彈簧力為工具內力）
            z_ht = z_tcp
            tip = z_ht + s.k + s.L
            if not st["captured"] and tip > -1e-4 and st["below"] is None:
                st["captured"] = lat <= st["e_cap"]
                st["below"] = True
                st["lat_cap"] = lat
            if st["below"] and not st["captured"]:
                z_ht = min(z_ht, -s.k - s.L)  # 頂在板面
            elif st["captured"] and tip > self.nr_model.j.t_b:  # 到螺紋口即嚙合開始
                z_ht = min(z_ht, -s.k - s.L + self.nr_model.j.t_b)
        else:  # "thread"：由螺紋決定螺絲位置
            z_ht = -s.k - s.L + self.nr_model.j.t_b + self.nr_model.depth(self.nr.th if self.nr else 0.0)
        c = prm.c_nom + (z_tcp - z_ht)
        F_ax = prm.F0 + prm.k_s * c if c > 0 else 0.0
        contact = mode == "thread" or z_ht < z_tcp - 1e-9
        st.update(c=c, F_ax=F_ax, z_ht=z_ht, lat=lat)
        st["c_min"], st["c_max"] = min(st["c_min"], c), max(st["c_max"], c)
        if mode == "thread":
            st["F_min"], st["F_max"] = min(st["F_min"], F_ax), max(st["F_max"], F_ax)
        # 外力：彈簧反推工具（−a）；起子馬達反作用（−T_m·a）
        a = hole.a
        Tm = self.nr.Tm if (self.nr and mode == "thread") else 0.0
        self.ext = (-a * F_ax if contact else np.zeros(3), -a * Tm)
        # 螺絲外觀
        ang = st["roll_acc"] + (self.nr.th if (self.nr and mode == "thread") else 0.0)
        xd = np.cos(ang) * st["e1"] + np.sin(ang) * st["e2"]
        lat_vec = T[:3, 3] - hole.p - z_tcp * a
        if mode == "thread":
            lat_vec = np.zeros(3)
        self.place_screw(st["screw"], hole.p + lat_vec + a * z_ht, a, xd)
        r = self.srec
        nr = self.nr if mode == "thread" else None
        for k, v in zip(r, (self.t, c, F_ax, lat, 0.0 if nr is None else nr.Tm, 0.0 if nr is None else nr.Tl,
                            0.0 if nr is None else nr.th, 0.0 if nr is None else nr.w,
                            0.0 if nr is None else self.nr_model.clamp(nr.th), st["roll_acc"],
                            mode if nr is None else nr.phase), strict=True):
            r[k].append(v)

    # --- 運動
    def _goto_tcp(self, T_tcp, T: float, stop=None) -> bool:
        """TCP 直線移動（五次時間律、姿態 SLERP）；stop() 為真時立即停止並保持。回傳是否被中止。"""
        q0 = self._ref[0]
        smp = sample_cartesian(self.r, LinePath(self.r.fk(q0), T_tcp @ self.T_tcp_inv), Quintic(T), q0, 0.005,
                               lam=1e-10)
        from scipy.interpolate import CubicHermiteSpline

        spl = CubicHermiteSpline(smp.t, smp.q, smp.qd)
        d1, d2 = spl.derivative(1), spl.derivative(2)
        for k in range(1, int(round(T / self.dt)) + 1):
            tk = k * self.dt
            self.step(spl(tk), d1(tk), d2(tk))
            if stop is not None and stop():
                q, _ = self.sim.get_state()
                self._ref = (q.copy(), np.zeros(6), np.zeros(6))
                return True
        return False

    def _track(self, T_tcp, v_tcp, w_tcp=np.zeros(3)):
        """單步追蹤：TCP 位姿 → 數值 IK（前一參考為初值）、q̇ = J⁻¹·twist。"""
        Tf = T_tcp @ self.T_tcp_inv
        q, ok, _ = self.r.ik_numeric(Tf, self._ref[0], tol=1e-11, damping=1e-9)
        J = self.r.jacobian(q)
        v_f = v_tcp - np.cross(w_tcp, T_tcp[:3, 3] - Tf[:3, 3])
        qd = np.linalg.solve(J, np.concatenate([v_f, w_tcp]))
        self.step(q, qd)
        return ok

    def tcp_target(self, hole_p, a, z, tilt_deg: float = 0.0):
        T = np.eye(4)
        R = tool_rotation(a)
        if tilt_deg:
            R = Rotation.from_rotvec(R[:, 0] * np.radians(tilt_deg)).as_matrix() @ R
        T[:3, :3] = R
        T[:3, 3] = hole_p + a * z
        return T

    # --- 取螺絲
    def pick_screw(self, prm: FastenParams, screw_body: int):
        a = np.array([0.0, 0.0, -1.0])
        top = FEEDER.copy()
        self.place_screw(screw_body, top, a, np.array([1.0, 0, 0]))
        T_up = self.tcp_target(top, a, -0.04)
        q_up = self.r.ik(T_up @ self.T_tcp_inv, self._ref[0])[0]
        self.move_j(q_up)
        self._goto_tcp(self.tcp_target(top, a, 0.0), 0.4)
        self.wait(0.10)  # 真空吸附
        return T_up

    # --- 鎖一顆螺絲
    def fasten(self, hole: Hole, prm: FastenParams | None = None, model: ThreadModel | None = None) -> dict:
        """（取螺絲）→ 接近 → 下降入孔（偵測/螺旋搜尋）→ 旋入 → 鎖緊（停止、煞車）→ 退回。"""
        prm = prm or FastenParams()
        self.react_ff = prm.react_ff
        model = model or ThreadModel(self.screw)
        self.nr_model = model
        nominal = ThreadModel(self.screw, JointSpec(t_b=model.j.t_b))
        s = self.screw
        t0 = self.t
        b = self.make_screw()
        true = Hole(hole.name, hole.p + np.asarray(prm.hole_err, float), hole.a)
        e1 = tool_rotation(hole.a)[:, 0]
        st = {"hole": true, "prm": prm, "screw": b, "mode": "carry", "captured": False, "below": None,
              "e_cap": (s.dh - s.d) / 2 + prm.e_cap_extra, "lat_cap": None, "roll": 0.0, "roll_acc": 0.0,
              "e1": e1, "e2": np.cross(hole.a, e1), "c_min": np.inf, "c_max": -np.inf, "F_min": np.inf,
              "F_max": -np.inf}
        if prm.pick:
            self._sstate = None
            self.pick_screw(prm, b)
            self._goto_tcp(self.tcp_target(FEEDER, np.array([0, 0, -1.0]), -0.04), 0.4)
        self._sstate = st
        T_tcp = self.tcp_pose()
        st["roll"] = float(np.arctan2(T_tcp[:3, 0] @ st["e2"], T_tcp[:3, 0] @ st["e1"]))
        self._screw_update()
        z_eng = -s.k - s.L + model.j.t_b  # 螺絲端到螺紋口時的頭頂（= TCP）位置
        T_app = self.tcp_target(hole.p, hole.a, z_eng - prm.approach, prm.tilt_err_deg)
        sols = self.r.ik(T_app @ self.T_tcp_inv, self._ref[0])
        if not sols:
            return self._finish(st, t0, "UNREACHABLE")
        self.move_j(sols[0])
        t_pos = self.t
        T_eng = self.tcp_target(hole.p, hole.a, z_eng, prm.tilt_err_deg)
        blocked = self._goto_tcp(T_eng, prm.t_descent, stop=lambda: st["c"] > prm.c_nom + 1.5e-3)
        res: dict = {}
        if blocked or not st["captured"]:
            res["blocked"] = True
            if not prm.search:
                return self._finish(st, t0, "NO_ENGAGE", res)
            ok, info = self._spiral(st, hole, prm, z_eng)
            res.update(info)
            if not ok:
                return self._finish(st, t0, "NO_ENGAGE", res)
            st["below"] = True
            Tc = self.tcp_pose()
            p_plane = Tc[:3, 3] - hole.a * self._axial(hole, Tc)[0]  # 搜尋到的孔位（板面上）
            T_eng = self.tcp_target(p_plane, hole.a, z_eng, prm.tilt_err_deg)
            self._goto_tcp(T_eng, 0.25)
        self.wait(0.05)
        T_now = self.tcp_pose()
        tilt = float(np.degrees(np.arccos(np.clip(T_now[:3, 2] @ true.a, -1, 1))))
        res.update({"pos_err_mm": st["lat"] * 1e3, "tilt_deg": tilt, "lat_cap_mm": (st["lat_cap"] or 0) * 1e3,
                    "t_position": self.t - t_pos})
        # --- 旋入與鎖緊
        model.cross = tilt > s.phi_ct_deg
        self.nr = Nutrunner(model, prm.driver, nominal)
        st["mode"] = "thread"
        st["c_min"], st["c_max"] = np.inf, -np.inf
        i0 = len(self.srec["t"])
        T_base = T_eng.copy()
        T_base[:3, 3] = T_now[:3, 3] - hole.a * ((T_now[:3, 3] - T_eng[:3, 3]) @ hole.a)
        c_start = prm.c_start if prm.c_start is not None else prm.c_nom
        T_base[:3, 3] += hole.a * (c_start - prm.c_nom)
        if prm.feed == "fixed" and c_start != prm.c_nom:
            self._goto_tcp(T_base, 0.2)
        t_run, lifted, bottom = self.t, False, False
        while not self.nr.done and self.t - t_run < prm.t_spin_max:
            dphi = 0.0  # 由 _screw_update 的 roll 累計
            roll0 = st["roll_acc"]
            self.nr.step(self.dt, st["F_ax"], dphi)
            if prm.feed == "follow":
                dz = model.s.P * self.nr.th / (2 * np.pi) if self.nr.th < nominal.theta_s else nominal.l_e
                v = model.s.P * self.nr.w / (2 * np.pi) if self.nr.th < nominal.theta_s else 0.0
                Tt = T_base.copy()
                Tt[:3, 3] += hole.a * min(dz, nominal.l_e)
                self._track(Tt, hole.a * v)
            else:
                self.step(self._ref[0])
            if not self.nr.done:
                self.nr.th += st["roll_acc"] - roll0  # 工具被反作用轉動 → 螺絲絕對轉角修正
            if st["c"] <= 0.0:
                lifted = True
                self.nr.phase = "abort"
                break
            if st["c"] >= prm.stroke:
                bottom = True
                self.nr.phase = "abort"
                break
        i1 = len(self.srec["t"])
        t_spin = self.t - t_run
        self.ext = (np.zeros(3), np.zeros(3))
        rr = self.nr.result()
        err = "BIT_LIFT" if lifted else "OVERFORCE" if bottom else rr["error"]
        if model.cross and rr["error"] is None and not lifted:
            err = "CROSS_THREAD_UNDETECTED"
        res.update({k: v for k, v in rr.items() if k not in ("ok", "error")})
        res.update({"t_spin": t_spin, "F_ax_min": st["F_min"], "F_ax_max": st["F_max"],
                    "c_min_mm": st["c_min"] * 1e3, "c_max_mm": st["c_max"] * 1e3, "cross": model.cross,
                    "roll_dev_deg": float(np.degrees(np.max(np.abs(np.array(self.srec["roll"][i0:i1])
                                                                   - self.srec["roll"][i0])))),
                    "lat_dev_mm": float(np.max(np.abs(np.array(self.srec["lat"][i0:i1])
                                                      - self.srec["lat"][i0])) * 1e3),
                    "torque_util": float(np.max(np.abs(np.array(self.rec["tau"][-(i1 - i0):])
                                                       / self.r.tau_max))),
                    "i0": i0, "i1": i1, "mu_g": model.j.mu_g, "c_joint": model.c})
        # --- 退回
        self.wait(0.05)
        st["mode"] = "release"
        self._sstate = None
        self._goto_tcp(self._retract_pose(hole, prm), 0.4)
        return self._finish(st, t0, err, res)

    def _retract_pose(self, hole: Hole, prm: FastenParams):
        T = self.tcp_pose()
        T[:3, 3] = T[:3, 3] - hole.a * prm.approach
        return T

    def _finish(self, st, t0, err, res=None):
        self._sstate = None
        self.ext = (np.zeros(3), np.zeros(3))
        out = {"hole": st["hole"].name, "ok": err is None, "error": err, "cycle_s": self.t - t0,
               **(res or {})}
        if err is not None and st["mode"] != "release":  # 失敗：先退回（螺絲留在原處）
            T = self.tcp_pose()
            T[:3, 3] -= st["hole"].a * st["prm"].approach
            self._goto_tcp(T, 0.4)
            out["cycle_s"] = self.t - t0
        return out

    def _spiral(self, st, hole: Hole, prm: FastenParams, z_eng: float):
        """阿基米德螺旋搜尋：r = b·φ（b = pitch/2π），等線速度 v；螺絲端輕壓板面（c ≈ c_nom + 1 mm）。"""
        s = self.screw
        T0 = self.tcp_pose()
        z_tcp, _ = self._axial(hole, T0)
        z_press = -s.k - s.L + 1.0e-3  # 螺絲端在板面時壓縮 +1 mm
        base = T0.copy()
        base[:3, 3] = T0[:3, 3] + hole.a * (z_press - z_tcp)
        self._goto_tcp(base, 0.15)
        e1, e2 = st["e1"], st["e2"]
        b = prm.search_pitch / (2 * np.pi)
        phi, t_s = 0.0, self.t
        st["captured"], st["searching"] = False, True
        while b * phi < prm.search_rmax:
            r = max(b * phi, 1e-6)
            phi += prm.search_v * self.dt / np.hypot(r, b)  # ds = √(r² + b²)·dφ
            r = b * phi
            u = r * (np.cos(phi) * e1 + np.sin(phi) * e2)
            du = b * (np.cos(phi) * e1 + np.sin(phi) * e2) + r * (-np.sin(phi) * e1 + np.cos(phi) * e2)
            v = du / np.linalg.norm(du) * prm.search_v
            Tt = base.copy()
            Tt[:3, 3] = base[:3, 3] + u
            self._track(Tt, v)
            if st["lat"] <= st["e_cap"]:
                st["captured"], st["searching"] = True, False
                st["lat_cap"] = st["lat"]
                return True, {"search_s": self.t - t_s, "search_r_mm": float(r * 1e3)}
        st["searching"] = False
        return False, {"search_s": self.t - t_s, "search_r_mm": float(b * phi * 1e3)}

    def series(self, i0: int, i1: int) -> dict:
        r = {k: np.array(v[i0:i1]) if k != "phase" else v[i0:i1] for k, v in self.srec.items()}
        r["t"] = r["t"] - r["t"][0]
        return r


# ------------------------------------------------------------------ JSON 工作描述
def parse_job(job) -> list[tuple[Hole, FastenParams]]:
    """JSON 鎖附工作 → [(Hole, FastenParams)]。格式::

        {"holes": [{"name": "H1", "p": [x,y,z], "a": [0,0,-1], "torque": 2.5, "strategy": "torque"}]}

    格式錯誤拋 ValueError。
    """
    if isinstance(job, str):
        job = json.loads(job)
    if not isinstance(job, dict) or not isinstance(job.get("holes"), list) or not job["holes"]:
        raise ValueError("工作需為含 'holes' 陣列的物件")
    out = []
    for i, h in enumerate(job["holes"]):
        if "p" not in h or len(h["p"]) != 3:
            raise ValueError(f"holes[{i}] 缺少 p 或格式錯誤")
        unknown = set(h) - {"name", "p", "a", "torque", "strategy", "angle_deg", "search"}
        if unknown:
            raise ValueError(f"holes[{i}] 未知參數 {sorted(unknown)}")
        T = float(h.get("torque", 2.5))
        if not 0.2 <= T <= 3.5:
            raise ValueError(f"holes[{i}] torque 超出範圍 [0.2, 3.5] N·m")
        strat = h.get("strategy", "torque")
        if strat not in ("torque", "angle"):
            raise ValueError(f"holes[{i}] strategy 需為 torque / angle")
        drv = replace(DriverParams(), T_target=T, strategy=strat, angle_deg=h.get("angle_deg"))
        out.append((Hole(h.get("name", f"H{i}"), h["p"], h.get("a", (0, 0, -1))),
                    FastenParams(driver=drv, search=bool(h.get("search", True)))))
    return out


def run_job(cell: ScrewCell, job, model_fn=None) -> dict:
    """依序鎖完所有孔後回原點；model_fn(i) 可給各孔的實際接合模型（摩擦離散等）。"""
    res = []
    for i, (hole, prm) in enumerate(parse_job(job)):
        res.append(cell.fasten(hole, prm, model_fn(i) if model_fn else None))
    cell.move_j(cell.q_home)
    return {"ok": all(m["ok"] for m in res), "holes": res, "t_total": cell.t}


__all__ = ["DriverParams", "FastenParams", "Hole", "JointSpec", "Nutrunner", "ScrewCell", "ScrewSpec",
           "ThreadModel", "bench_test", "joint_stiffness", "parse_job", "run_job", "tool_rotation",
           "write_arm_with_driver"]
