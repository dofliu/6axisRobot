"""軸孔插入 peg-in-hole（T109）：RCC 柔順腕 + 銷體接觸動力學 + 軸向力導納/側向力歸零 + 螺旋搜孔。

工具（``robots/ur5_peg.urdf``）
------------------------------
法蘭 → 六軸力感測器 → RCC（remote center compliance）上板（以上屬工具連桿 7，隨法蘭剛性運動）。
RCC 下板 + 夾爪 + 銷（以下稱「銷體」）相對上板以 6 自由度彈簧–阻尼連接，另以子步積分（預設 20 kHz）
模擬其剛體動力學與接觸。TCP（連桿 8）= 銷體「名目」端點（RCC 未變形時銷端面中心）。

RCC 柔順中心（Whitney 1982, J. Dyn. Sys. Meas. Control 104:65–77；Whitney《Mechanical Assemblies》2004 ch.10）
--------------------------------------------------------------------------------------------------
柔順中心 C 位於銷端上方 L_c（沿銷軸）。以手部座標表示：C 點位移 δ、銷體相對轉角 θ（小角度向量），
彈簧力/力矩（作用於銷體，作用點 C）::

    F_s = −R_h (K_t δ_h + C_t δ̇_h)，  K_t = diag(K_lat, K_lat, K_ax)
    M_s = −R_h (K_r θ_h + C_r θ̇_h)，  K_r = diag(K_rot, K_rot, K_tor)

反作用 (−F_s, −M_s) 施加於工具連桿，也就是力感測器量到的負載。L_c = 0（柔順中心在銷端）時，作用在銷端的
側向力只造成平移、不造成轉動 → 側向誤差與角度誤差解耦，是 RCC 的設計目的；L_c 大（一般柔順腕）時側向力
會使銷反向傾斜而提高卡死（jamming）/楔住（wedging）風險。

銷體動力學：m v̇ = F_s + m g + ΣF_c；I_w ω̇ = M_C − ω × I_w ω（對質心），半隱式 Euler、Rodrigues 更新姿態。

接觸（罰函數；Cundall & Strack 1979, Géotechnique 29:47 的切向彈簧–滑移摩擦）
---------------------------------------------------------------------------
孔：軸 n（朝外）、入口頂面圓心 c_h、孔徑 R、45° 倒角寬 w_c、孔深 L_h。工件實體的有號距離：
頂面 h（h 為沿 −n 的深度）、倒角面 (ρ − R − w_c + h)/√2、孔壁 ρ − R、孔底 h − L_h。兩組取樣點：

* 銷端面外緣 N_r 點 vs 工件實體（端角觸頂面/倒角/孔壁/孔底）；
* 孔口兩條稜線（倒角下緣 ρ = R, h = w_c；上緣 ρ = R + w_c, h = 0）各 N_e 點 vs 銷圓柱（側面或端面）——
  傾斜銷的「兩點接觸」第二點即孔口稜線抵住銷側面。

每點法向力 f_n = max(k·pen − c·ṗen, 0)；切向以錨點彈簧 u（∫v_t dt）表示：f_t = −k_t u − c_t v_t，
|f_t| > μ f_n 時錨點滑移（u 縮放到摩擦錐上）→ 可表示靜摩擦黏著（卡死時銷真正停住，而非正則化摩擦的潛變）。

準靜態理論（本模組函式，報告與測試比對）
--------------------------------------
* 捕捉條件：側向誤差 e ≤ w_c + c（c = R − r 徑向間隙）時銷端遠側落在倒角上，可被導入；否則落在頂面（MISS）。
* 倒角滑入（45° 倒角、摩擦 μ）：倒角對銷的合力 F = N(−sinα + μ cosα, cosα + μ sinα)，
  軸向/側向比 F_z/F_x = (1 + μ tanα)/(tanα − μ)；
  RCC 側向剛性下側向力 F_x = K_lat·(e − c)（穿過倒角時的最大偏移）。
* 傾斜銷幾何極限：轉角固定為 θ 時能進入的最大深度 l ≈ (D_h − D cosθ)/sinθ（兩點接觸開始；
  精確式見 ``tilt_depth_limit``）。
* 楔住（Whitney）：兩點接觸深度 l_2 < μ·D 時兩接觸點摩擦錐重疊，可能楔住；即 θ ≳ 2c/(μD)。
* 不卡死傾角（本專案推導，見 ``jam_free_tilt``）：θ_max = 2√(F_z c/(μ K_rot))。
* 兩點接觸靜力平衡（推導見 ``two_point_statics``）：給定施加力 F_x、F_z 與對銷端力矩 M、深度 l，
  N1 = −(M + (μr + l)F_x)/l、N2 = N1 + F_x；F_z ≤ μ(N1 + N2) 則卡死。

控制
----
外環（手部 TCP 相對名目孔口的位移，孔座標 u, w, 深度 z）：

* position：沿孔軸以 v_ins 直線下壓至目標深度（無力回授，僅力保護）。
* admittance：軸向 M z̈ + D ż = F_d − F_z,m（|ż| ≤ v_ins）；
  側向 M_l ẍ + D_l ẋ = F_lat,m（順著工具受到的側向力移動 → 側向力歸零，主動柔順）。
* search：頂面著陸（軸向力達 F_d 但深度 < d_land）時沿阿基米德螺旋 r = p_s·φ/2π 搜孔，深度增加即停止。

內環：每步以封閉解 IK 求關節參考（固定姿態）、q̇ = J⁻¹[v; 0]，交給 :class:`WeldCell` 的計算力矩控制；
並以力感測器讀值做力前饋 τ_ff = −Jᵀw_m（同 T108）。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as pb

from .robot import SixAxisRobot
from .urdf import write_urdf
from .welding import BENCH_TOP, Q_SEED, WeldCell, _unit

PEG_URDF = Path(__file__).resolve().parents[1] / "robots" / "ur5_peg.urdf"
TOOL_LINK, TCP_LINK = 7, 8
G = np.array([0.0, 0.0, -9.81])

# 工具尺寸（法蘭起算，沿工具 z）
FT_L = 0.022  # 力感測器
RCC_TOP_L = 0.030  # RCC 上板 + 彈性體上半
RCC_BOT_L = 0.025  # RCC 下板（銷體側）
GRIP_L = 0.040  # 夾爪本體
PEG_OUT = 0.035  # 銷伸出夾爪長度
TOOL_L = FT_L + RCC_TOP_L + RCC_BOT_L + GRIP_L + PEG_OUT  # 法蘭 → 名目銷端
T_PEG = np.eye(4)
T_PEG[2, 3] = TOOL_L
T_PEG_INV = np.linalg.inv(T_PEG)


# ------------------------------------------------------------------ 規格
@dataclass
class PegSpec:
    r: float = 0.006  # 銷半徑（D = 12 mm）
    length: float = 0.060  # 銷長（接觸檢查範圍：端面起 0..length）
    mass: float = 0.25  # 銷體質量（銷 + 夾爪 + RCC 下板）
    h_com: float = 0.045  # 質心距銷端
    I_xx: float = 3.0e-4  # 對質心（銷 + 夾爪 + RCC 下板，長約 0.1 m）
    I_zz: float = 3.0e-5


@dataclass
class HoleSpec:
    R: float = 0.00608  # 孔半徑（徑向間隙 c = 80 µm，間隙比 c* = 2c/D ≈ 0.013）
    w_c: float = 0.0010  # 45° 倒角寬
    depth: float = 0.030
    block: tuple = (0.07, 0.07, 0.040)  # 工件塊尺寸（頂面 = 孔口）


@dataclass
class RCCSpec:
    K_lat: float = 4.0e3  # 側向 [N/m]
    K_ax: float = 1.0e5  # 軸向
    K_rot: float = 15.0  # 彎曲 [N·m/rad]
    K_tor: float = 30.0  # 扭轉
    L_c: float = 0.0  # 柔順中心距銷端（沿銷軸，> 0 往手部）
    zeta: float = 0.7


RIGID = RCCSpec(K_lat=2.0e5, K_ax=5.0e5, K_rot=400.0, K_tor=400.0, L_c=0.0, zeta=0.7)


@dataclass
class ContactSpec:
    mu: float = 0.30
    k_rim: float = 5.0e5  # 銷端外緣每點法向剛性 [N/m]
    c_rim: float = 100.0
    k_edge: float = 1.0e6  # 孔口稜線每點
    c_edge: float = 100.0
    kt_ratio: float = 0.5  # 切向/法向剛性比
    n_rim: int = 36
    n_edge: int = 72


# ------------------------------------------------------------------ 準靜態理論
def capture_radius(peg: PegSpec, hole: HoleSpec) -> float:
    """倒角可導入的最大側向誤差 e_cap = w_c + c。"""
    return hole.w_c + (hole.R - peg.r)


def chamfer_ratio(mu: float, alpha_deg: float = 45.0) -> float:
    """倒角滑入時軸向/側向力比 F_z/F_x = (1 + μ tanα)/(tanα − μ)。
    α：倒角面與水平夾角；tanα ≤ μ 時無法滑入 → inf。"""
    ta = np.tan(np.radians(alpha_deg))
    return float((1 + mu * ta) / (ta - mu)) if ta > mu else float("inf")


def chamfer_forces(e: float, peg: PegSpec, hole: HoleSpec, rcc: RCCSpec, mu: float) -> dict:
    """RCC（L_c = 0）準靜態穿越倒角：最大側向偏移 e − c → 側向力 K_lat(e − c)、軸向力 × chamfer_ratio。"""
    c = hole.R - peg.r
    fx = rcc.K_lat * max(e - c, 0.0)
    return {"F_lat": fx, "F_ax": fx * chamfer_ratio(mu)}


def tilt_depth_limit(theta: float, peg: PegSpec, hole: HoleSpec) -> float:
    """轉角固定為 θ [rad] 的銷可進入的最大深度（兩點接觸：最低端角觸孔壁、同側銷側面觸孔口稜線）。

    平面剖面幾何：最低端角在 x_c + r cosθ、深度 d + r sinθ；同側銷側面由端角沿軸 (sinθ, cosθ) 上升，
    在孔口稜線高度的 x = x_c + r cosθ + (d_b + r sinθ) tanθ；另一側端角 x_c − r cosθ ≥ −R。
    可行條件 (d_b + r sinθ) tanθ + D cosθ ≤ D_h →
    d_b = (D_h − D cosθ)/tanθ − r sinθ（銷端中心在孔口稜線（倒角下緣）下方的深度）。
    小角度時即經典式 l ≈ (D_h − D cosθ)/sinθ（Whitney）。θ → 0 時為 inf。"""
    th = abs(theta)
    if th < 1e-12:
        return float("inf")
    return float((2 * hole.R - 2 * peg.r * np.cos(th)) / np.tan(th) - peg.r * np.sin(th))


def wedge_angle(peg: PegSpec, hole: HoleSpec, mu: float) -> float:
    """Whitney 楔住條件 l_2 < μD 的臨界初始傾角 θ_w ≈ 2c/(μD) [rad]。"""
    return float(2 * (hole.R - peg.r) / (mu * 2 * peg.r))


def jam_free_tilt(F_z: float, peg: PegSpec, hole: HoleSpec, mu: float, K_rot: float) -> float:
    """柔順銷（L_c = 0、側向力已歸零 F_x ≈ 0）不卡死的最大初始傾角 θ_max [rad]（本專案推導）。

    兩點接觸後幾何約束銷的轉角 θ(l) ≈ 2c/l（l：孔口稜線到銷端的插入深度，小角度），RCC 回復力矩
    |M| = K_rot·(θ0 − 2c/l)。由 ``two_point_statics``（F_x = 0）：N1 = N2 = |M|/l，滑動所需推力 2μ|M|/l。
    不卡死 ⇔ 對所有 l：F_z·l/(2μ) > K_rot(θ0 − 2c/l)。左右差 f(l) = F_z l/(2μ) + 2cK_rot/l − K_rot θ0 在
    l* = √(4μcK_rot/F_z) 有最小值 2√(F_z c K_rot/μ) − K_rot θ0 ≥ 0 →

        θ_max = 2·√(F_z·c / (μ·K_rot))

    （Whitney 1982 的結論「柔順中心在銷端且角剛性小 → 不卡死」的定量形式；F_z ↑、μ ↓、K_rot ↓ 皆放寬。）"""
    c = hole.R - peg.r
    return float(2 * np.sqrt(F_z * c / (mu * K_rot)))


def jam_depth(theta0: float, F_z: float, peg: PegSpec, hole: HoleSpec, mu: float, K_rot: float) -> float:
    """θ0 > θ_max 時開始卡死的插入深度（自孔口稜線起算）：f(l) = 0 的較小根；θ0 ≤ θ_max 回傳 inf。
    F_z l²/(2μ) − K_rot θ0 l + 2cK_rot = 0。"""
    c = hole.R - peg.r
    a, b, cc = F_z / (2 * mu), -K_rot * theta0, 2 * c * K_rot
    disc = b * b - 4 * a * cc
    if disc < 0:
        return float("inf")
    return float((-b - np.sqrt(disc)) / (2 * a))


def two_point_statics(F_x: float, F_z: float, M: float, l: float, r: float, mu: float) -> dict:
    """平面兩點接觸準靜態（Whitney 1982 的推導，本專案重推）：

    銷端角 A = (−r, 0) 觸左壁（法向力 N1 → +x、摩擦 μN1 向上），
    孔口稜線 B = (r, l) 觸右側面（N2 → −x、μN2 向上）；
    施加於銷端中心 O 的力 (F_x, −F_z)（F_z 為向下推力）與力矩 M（逆時針為正）。
    x：F_x + N1 − N2 = 0；對 O 力矩：M − μrN1 + (μr + l)N2 = 0
    → N1 = −(M + (μr + l)F_x)/l、N2 = N1 + F_x；滑動所需推力 F_req = μ(N1 + N2)。
    兩點接觸成立需 N1, N2 ≥ 0；F_z ≤ F_req 時卡死（jam）。"""
    N1 = -(M + (mu * r + l) * F_x) / l
    N2 = N1 + F_x
    F_req = mu * (N1 + N2)
    valid = N1 >= 0 and N2 >= 0
    return {"N1": N1, "N2": N2, "F_req": F_req, "two_point": bool(valid), "jam": bool(valid and F_z <= F_req)}


# ------------------------------------------------------------------ 小工具
def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _expm(w):
    th = float(np.linalg.norm(w))
    if th < 1e-14:
        return np.eye(3) + _skew(w)
    K = _skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _rot(axis, ang):
    return _expm(_unit(np.asarray(axis, float)) * ang)


def _vee_small(A):
    """R 的小角度轉角向量（反對稱部分）。"""
    return 0.5 * np.array([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]])


# ------------------------------------------------------------------ 孔
class Hole:
    """工件上的孔：頂面圓心 c、軸 n（朝外）、側向基底 u、w。"""

    def __init__(self, center, spec: HoleSpec | None = None, normal=(0, 0, 1), u=(1, 0, 0), n_edge: int = 72):
        self.spec = spec or HoleSpec()
        self.c = np.asarray(center, float)
        self.n = _unit(normal)
        u = np.asarray(u, float)
        self.u = _unit(u - (u @ self.n) * self.n)
        self.w = np.cross(self.n, self.u)
        s = self.spec
        ph = np.linspace(0, 2 * np.pi, n_edge, endpoint=False)
        ring = np.cos(ph)[:, None] * self.u + np.sin(ph)[:, None] * self.w
        e1 = self.c + s.R * ring - s.w_c * self.n  # 倒角下緣（孔口）
        e2 = self.c + (s.R + s.w_c) * ring  # 倒角上緣
        self.edges = np.vstack([e1, e2])

    def frame(self) -> np.ndarray:
        return np.column_stack([self.u, self.w, self.n])

    def sdf_solid(self, P):
        """點 P（N×3）在工件實體內的穿透深度與作用於侵入物的法向（世界）。
        回傳 (pen ≥ 0, normal)，pen = 0 表無接觸。"""
        s = self.spec
        d = P - self.c
        dn = d @ self.n
        h = -dn  # 深度
        rad = d - dn[:, None] * self.n
        rho = np.linalg.norm(rad, axis=1)
        er = rad / np.maximum(rho, 1e-12)[:, None]
        rw = s.R + np.clip(s.w_c - h, 0.0, None)
        in_col = rho < rw
        pen = np.zeros(len(P))
        nrm = np.tile(self.n, (len(P), 1))
        # 孔底
        mb = in_col & (h > s.depth)
        pen[mb] = h[mb] - s.depth
        # 頂面 / 倒角 / 孔壁
        ms = (~in_col) & (h > 0)
        if ms.any():
            hs, rs, es = h[ms], rho[ms], er[ms]
            cham = hs < s.w_c
            pw = np.where(cham, (rs - s.R - s.w_c + hs) / np.sqrt(2), rs - s.R)
            nw = np.where(cham[:, None], (-es + self.n) / np.sqrt(2), -es)
            top = hs <= pw
            pen[ms] = np.where(top, hs, pw)
            nrm[ms] = np.where(top[:, None], self.n, nw)
        return pen, nrm

    def depth_of(self, p) -> float:
        return float(-(np.asarray(p) - self.c) @ self.n)

    def lateral_of(self, p) -> np.ndarray:
        d = np.asarray(p) - self.c
        return np.array([d @ self.u, d @ self.w])


# ------------------------------------------------------------------ 銷體（RCC 下側）動力學
class PegBody:
    """銷體 6 自由度剛體 + RCC 彈簧 + 與孔的接觸，子步積分。"""

    def __init__(self, peg: PegSpec, rcc: RCCSpec, contact: ContactSpec, hole: Hole, R_grip=None):
        self.peg, self.rcc, self.cs, self.hole = peg, rcc, contact, hole
        # 銷在手部座標的名目夾持姿態誤差
        self.R_g = np.eye(3) if R_grip is None else np.asarray(R_grip, float)
        ph = np.linspace(0, 2 * np.pi, contact.n_rim, endpoint=False)
        self.rim_b = np.column_stack([peg.r * np.cos(ph), peg.r * np.sin(ph), np.zeros_like(ph)])  # 銷座標
        self.Ib = np.diag([peg.I_xx, peg.I_xx, peg.I_zz])
        m = peg.mass
        self.Kt = np.array([rcc.K_lat, rcc.K_lat, rcc.K_ax])
        self.Ct = 2 * rcc.zeta * np.sqrt(self.Kt * m)
        self.Kr = np.array([rcc.K_rot, rcc.K_rot, rcc.K_tor])
        self.Cr = 2 * rcc.zeta * np.sqrt(self.Kr * np.array([peg.I_xx + m * peg.h_com**2] * 2 + [peg.I_zz]))
        self.u_rim = np.zeros((contact.n_rim, 3))
        self.u_edge = np.zeros((len(hole.edges), 3))
        self.p = np.zeros(3)  # 銷端中心（世界）
        self.R = np.eye(3)  # 銷座標（z = 銷軸，由銷端指向手部）
        self.v = np.zeros(3)  # 質心速度
        self.w = np.zeros(3)
        self.n_contact = 0
        self.contact_kind = (0, 0)  # (端緣點數, 稜線點數)

    # 名目（RCC 未變形）銷姿態：手部 TCP 座標 z 為工具軸（朝工件）→ 銷軸 = −z_tool
    def nominal(self, T_h):
        Rn = T_h[:3, :3] @ np.diag([1.0, -1.0, -1.0]) @ self.R_g
        return T_h[:3, 3].copy(), Rn

    def com(self):
        return self.p + self.R[:, 2] * self.peg.h_com

    def reset(self, T_h):
        """放在準靜態平衡（僅重力下垂）。"""
        p, Rn = self.nominal(T_h)
        Rh = T_h[:3, :3]
        sag = Rh @ ((Rh.T @ (self.peg.mass * G)) / self.Kt)
        self.p, self.R = p + sag, Rn
        self.v, self.w = np.zeros(3), np.zeros(3)
        self.u_rim[:] = 0
        self.u_edge[:] = 0

    def _rcc(self, ph, Rn, vh, wh, pc):
        """RCC 彈簧–阻尼對銷體的力（作用於 C_p）與力矩。"""
        Lc = self.rcc.L_c
        Cp = self.p + self.R[:, 2] * Lc
        Ch = ph + Rn[:, 2] * Lc
        Rh = Rn  # 以名目銷座標表示剛性矩陣（z = 銷軸）
        dl = Rh.T @ (Cp - Ch)
        vCp = self.v + np.cross(self.w, Cp - pc)
        vCh = vh + np.cross(wh, Ch - ph)
        dld = Rh.T @ (vCp - vCh)
        F = -Rh @ (self.Kt * dl + self.Ct * dld)
        th = _vee_small(Rh.T @ self.R)
        thd = Rh.T @ (self.w - wh)
        M = -Rh @ (self.Kr * th + self.Cr * thd)
        return F, M, Cp

    def _contacts(self, pc, dt):
        cs, hole = self.cs, self.hole
        F = np.zeros(3)
        Mc = np.zeros(3)
        n_r = n_e = 0
        # 1) 銷端外緣 vs 工件實體
        P = self.p + self.rim_b @ self.R.T
        pen, nrm = hole.sdf_solid(P)
        c = pen > 0
        self.u_rim[~c] = 0
        if c.any():
            n_r = int(c.sum())
            f, Mi = self._point_forces(P[c], pen[c], nrm[c], self.u_rim, c, pc, dt, cs.k_rim, cs.c_rim)
            F += f
            Mc += Mi
        # 2) 孔口稜線 vs 銷圓柱
        E = hole.edges
        a = self.R[:, 2]
        d = E - self.p
        s = d @ a
        rad = d - s[:, None] * a
        dist = np.linalg.norm(rad, axis=1)
        r = self.peg.r
        c = (s > 0) & (s < self.peg.length) & (dist < r)
        self.u_edge[~c] = 0
        if c.any():
            n_e = int(c.sum())
            side = (r - dist[c]) <= s[c]
            pen_e = np.where(side, r - dist[c], s[c])
            nrm_e = np.where(side[:, None], -rad[c] / np.maximum(dist[c], 1e-12)[:, None], a)
            f, Mi = self._point_forces(E[c], pen_e, nrm_e, self.u_edge, c, pc, dt, cs.k_edge, cs.c_edge)
            F += f
            Mc += Mi
        self.contact_kind = (n_r, n_e)
        return F, Mc

    def _point_forces(self, X, pen, nrm, U, mask, pc, dt, k, cdamp):
        """接觸點（世界 X）上的法向罰力 + 錨點切向摩擦；回傳作用於銷體的合力與對質心力矩。"""
        mu, kt = self.cs.mu, self.cs.kt_ratio * k
        r = X - pc
        vrel = self.v + np.cross(self.w, r)  # 銷上該點相對工件（工件靜止）
        vn = np.einsum("ij,ij->i", vrel, nrm)
        fn = np.maximum(k * pen - cdamp * vn, 0.0)
        vt = vrel - vn[:, None] * nrm
        u = U[mask] + vt * dt
        u -= np.einsum("ij,ij->i", u, nrm)[:, None] * nrm
        ft = -kt * u - cdamp * vt
        ftn = np.linalg.norm(ft, axis=1)
        lim = mu * fn
        slip = ftn > lim
        if slip.any():
            sc = lim[slip] / np.maximum(ftn[slip], 1e-12)
            ft[slip] *= sc[:, None]
            u[slip] = -ft[slip] / kt  # 錨點移到摩擦錐邊界（不含阻尼部分的近似）
        U[mask] = u
        f = fn[:, None] * nrm + ft
        return f.sum(0), np.cross(r, f).sum(0)

    def far_from_hole(self, margin: float = 0.002) -> bool:
        """銷端所有外緣點都在頂面上方 margin 以上（無接觸可能）。"""
        P = self.p + self.rim_b @ self.R.T
        return bool(np.all((P - self.hole.c) @ self.hole.n > margin))

    def advance(self, T_h, v_h, w_h, dt: float, n_sub: int):
        """以手部（TCP）位姿/速度驅動，積分 n_sub 個子步；
        回傳作用於工具（RCC 上板）的平均力、對 TCP 的力矩。"""
        ph0, Rn = self.nominal(T_h)
        h = dt / n_sub
        m = self.peg.mass
        Fh = np.zeros(3)
        Mh = np.zeros(3)
        nmax = 0
        for k in range(n_sub):
            ph = ph0 + v_h * (k * h)
            pc = self.com()
            Fs, Ms, Cp = self._rcc(ph, Rn, v_h, w_h, pc)
            Fc, Mc = self._contacts(pc, h)
            nmax = max(nmax, sum(self.contact_kind))
            Ft = Fs + m * G + Fc
            Mt = np.cross(Cp - pc, Fs) + Ms + Mc
            Iw = self.R @ self.Ib @ self.R.T
            self.v = self.v + Ft / m * h
            self.w = self.w + np.linalg.solve(Iw, Mt - np.cross(self.w, Iw @ self.w)) * h
            pc = pc + self.v * h
            self.R = _expm(self.w * h) @ self.R
            self.p = pc - self.R[:, 2] * self.peg.h_com
            Fh -= Fs
            Mh -= Ms + np.cross(Cp - (ph0 + v_h * (k * h)), Fs)
        self.n_contact = nmax
        return Fh / n_sub, Mh / n_sub

    def quasi_static(self, T_h, v_h, w_h):
        """無接觸時直接放在準靜態平衡；回傳工具上的力（= 銷體重量）與對 TCP 的力矩。"""
        self.reset(T_h)
        self.v, self.w = np.asarray(v_h, float).copy(), np.asarray(w_h, float).copy()
        self.n_contact = 0
        self.contact_kind = (0, 0)
        F = self.peg.mass * G
        M = np.cross(self.com() - T_h[:3, 3], F)
        return F, M

    def tilt_deg(self, T_h) -> float:
        """銷軸與名目（手部）銷軸夾角 [deg]。"""
        _, Rn = self.nominal(T_h)
        return float(np.degrees(np.arccos(np.clip(self.R[:, 2] @ Rn[:, 2], -1, 1))))

    def axis_hole_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.R[:, 2] @ self.hole.n, -1, 1))))


# ------------------------------------------------------------------ 工具 URDF 與外觀
def peg_tool_xml(parent: str = "flange") -> str:
    """工具連桿：力感測器 + RCC 上板（含碰撞）＋ tcp。銷體外觀另建（隨模擬狀態擺放）。"""

    def cyl(z0, z1, r, mat):
        g = (f'<origin xyz="0 0 {(z0 + z1) / 2:.6f}"/>'
             f'<geometry><cylinder length="{z1 - z0:.6f}" radius="{r}"/></geometry>')
        return f'<visual>{g}<material name="{mat}"/></visual><collision>{g}</collision>'

    m = 0.9
    L = FT_L + RCC_TOP_L
    It = m * L**2 / 12 + m * 0.035**2 / 4
    U = '  <material name="ptft"><color rgba="0.72 0.74 0.78 1"/></material>\n'
    U += '  <material name="ptrcc"><color rgba="0.20 0.45 0.80 1"/></material>\n'
    U += f'  <link name="peg_tool"><inertial><mass value="{m}"/><origin xyz="0 0 {L / 2:.4f}"/>'
    U += f'<inertia ixx="{It:.6f}" iyy="{It:.6f}" izz="6e-4" ixy="0" ixz="0" iyz="0"/></inertial>'
    U += cyl(0.0, FT_L, 0.036, "ptft")
    U += cyl(FT_L, FT_L + RCC_TOP_L, 0.040, "ptrcc")
    U += "</link>\n"
    U += ('  <link name="tcp"><inertial><mass value="1e-6"/>'
          '<inertia ixx="1e-12" iyy="1e-12" izz="1e-12" ixy="0" ixz="0" iyz="0"/></inertial></link>\n')
    U += f'  <joint name="peg_mount" type="fixed"><parent link="{parent}"/><child link="peg_tool"/></joint>\n'
    U += ('  <joint name="tcp_joint" type="fixed"><parent link="peg_tool"/><child link="tcp"/>'
          f'<origin xyz="0 0 {TOOL_L:.9f}"/></joint>\n')
    return U


def write_arm_with_peg_tool(robot: SixAxisRobot | None = None, path: str | Path = PEG_URDF) -> Path:
    """手臂 URDF + 力感測器/RCC。連桿 index：7 = peg_tool，8 = tcp（名目銷端）。"""
    robot = robot or SixAxisRobot()
    write_urdf(robot, path)
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    path.write_text(txt.replace("</robot>", peg_tool_xml() + "</robot>"), encoding="utf-8")
    return path


def hole_block_obj(hole: Hole, path: Path, n: int = 64, cut: bool = False) -> Path:
    """工件塊（頂面含 45° 倒角圓孔、孔底）OBJ 網格，局部座標（孔口中心為原點、z = n）。
    cut=True 時去掉 y > 0 半邊並補上 y = 0 剖面（剖視，便於觀察孔內的銷）。三角形雙面輸出。"""
    s = hole.spec
    Lx, Ly, H = s.block
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V: list = []
    Fc: list = []

    def ring(rad, z):
        i0 = len(V) + 1
        V.extend((rad * np.cos(a), rad * np.sin(a), z) for a in ph)
        return i0

    def square_ring(z):
        i0 = len(V) + 1
        for a in ph:
            c, s_ = np.cos(a), np.sin(a)
            t = min(Lx / 2 / max(abs(c), 1e-9), Ly / 2 / max(abs(s_), 1e-9))
            V.append((t * c, t * s_, z))
        return i0

    def band(a0, a1, flip=False):
        for k in range(n):
            k1 = (k + 1) % n
            q = (a0 + k, a0 + k1, a1 + k1, a1 + k)
            if flip:
                q = q[::-1]
            Fc.append((q[0], q[1], q[2]))
            Fc.append((q[0], q[2], q[3]))

    top_o = square_ring(0.0)
    top_i = ring(s.R + s.w_c, 0.0)
    ch_lo = ring(s.R, -s.w_c)
    bore_lo = ring(s.R, -s.depth)
    band(top_o, top_i, flip=True)
    band(top_i, ch_lo, flip=True)
    band(ch_lo, bore_lo, flip=True)
    cb = len(V) + 1
    V.append((0.0, 0.0, -s.depth))
    for k in range(n):
        Fc.append((cb, bore_lo + k, bore_lo + (k + 1) % n))
    # 外側面與底面
    bot = square_ring(-H)
    band(top_o, bot)
    cb2 = len(V) + 1
    V.append((0.0, 0.0, -H))
    for k in range(n):
        Fc.append((cb2, bot + (k + 1) % n, bot + k))
    if cut:
        Fc = [f for f in Fc if sum(V[i - 1][1] for i in f) / 3 <= 1e-9]
        rt = s.R + s.w_c
        for sg in (-1.0, 1.0):  # 剖面：孔兩側（凸多邊形，扇形三角化）
            poly = [(sg * Lx / 2, 0.0), (sg * rt, 0.0), (sg * s.R, -s.w_c), (sg * s.R, -s.depth),
                    (sg * Lx / 2, -s.depth)]
            i0 = len(V) + 1
            V.extend((x, 0.0, z) for x, z in poly)
            Fc.extend((i0, i0 + k, i0 + k + 1) for k in range(1, len(poly) - 1))
        i0 = len(V) + 1
        V.extend([(-Lx / 2, 0.0, -s.depth), (Lx / 2, 0.0, -s.depth), (Lx / 2, 0.0, -H), (-Lx / 2, 0.0, -H)])
        Fc.extend([(i0, i0 + 1, i0 + 2), (i0, i0 + 2, i0 + 3)])
    Fc = Fc + [(a, c, b) for a, b, c in Fc]  # 雙面
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V] + [f"f {a} {b} {c}" for a, b, c in Fc]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ 控制參數
@dataclass
class InsertParams:
    strategy: str = "admittance"  # "position" | "admittance"
    search: bool = False
    depth: float = 0.020  # 目標插入深度（銷端低於孔口）
    v_ins: float = 0.010  # 插入速度上限 [m/s]
    z_app: float = 0.003  # 觸碰前起始高度（孔口上方）
    approach: float = 0.040
    F_d: float = 15.0  # 軸向期望力
    M: float = 1.0
    D: float = 1500.0
    M_lat: float = 1.0
    D_lat: float = 1500.0
    lateral: bool = True  # 側向力歸零
    F_max: float = 80.0  # 力保護
    sigma_F: float = 0.2
    f_c: float = 60.0
    t_jam: float = 0.5  # 無進展時間
    v_jam: float = 5e-4  # 進展速度門檻
    t_max: float = 8.0
    # 搜孔
    d_land: float = 0.4e-3  # 著陸判定深度（< 此值且力 ≥ 0.8 F_d → 在頂面）
    F_search: float = 8.0
    pitch: float = 1.5e-3
    v_search: float = 0.006
    r_search: float = 4.0e-3
    d_found: float = 0.5e-3


# ------------------------------------------------------------------ 工作站
class InsertCell(WeldCell):
    """手臂 + 力感測器/RCC 工具 + 工作台 + 孔工件。"""

    def __init__(self, hole_c=(0.45, 0.0, 0.14), hole_spec: HoleSpec | None = None, hole_normal=(0, 0, 1),
                 hole_err=(0.0, 0.0), peg: PegSpec | None = None, rcc: RCCSpec | None = None,
                 contact: ContactSpec | None = None, grip_tilt_deg: float = 0.0, grip_tilt_axis=(0, 1, 0),
                 render: bool = False, n_sub: int = 20, seed: int = 0, urdf_path=None, **kw):
        """hole_c 為「規劃端」以為的孔口中心；實際孔口 = hole_c + (e_x, e_y, 0)（規劃端未知）。
        hole_normal 為實際孔軸（規劃端以 +z 為名目）。grip_tilt_deg：銷在夾爪中的角度誤差（繞手部座標軸）。"""
        path = PEG_URDF if urdf_path is None else Path(urdf_path)
        super().__init__(render=render, tool_writer=lambda r: write_arm_with_peg_tool(r, path), t_tcp=T_PEG,
                         **kw)
        self.peg, self.rcc = peg or PegSpec(), rcc or RCCSpec()
        self.cs = contact or ContactSpec()
        self.plan_c = np.asarray(hole_c, float)
        self.plan_n = np.array([0.0, 0.0, 1.0])
        nrm = _unit(hole_normal)
        u0 = np.array([1.0, 0, 0])
        u0 = _unit(u0 - (u0 @ nrm) * nrm)
        true_c = self.plan_c + hole_err[0] * np.array([1.0, 0, 0]) + hole_err[1] * np.array([0, 1.0, 0])
        self.hole = Hole(true_c, hole_spec, nrm, u0, n_edge=self.cs.n_edge)
        Rg = _rot(grip_tilt_axis, np.radians(grip_tilt_deg)) if grip_tilt_deg else np.eye(3)
        self.body = PegBody(self.peg, self.rcc, self.cs, self.hole, Rg)
        self.n_sub = n_sub
        self.rng = np.random.default_rng(seed)
        self.ip = InsertParams()
        self.w_meas = np.zeros(6)
        self.w_ff = np.zeros(6)
        self.wr_true = np.zeros(6)
        self.ext = (np.zeros(3), np.zeros(3))
        self.phase = "idle"
        self.render_on = render
        self.peg_vis = None
        self._add_workpiece(render)
        self.body.reset(self.tcp_pose())
        self.frec = {k: [] for k in ("t", "F", "M", "Fm", "depth", "hand_depth", "lat", "hand_lat", "tilt",
                                     "n_rim", "n_edge", "phase")}

    # --- 場景
    def _add_workpiece(self, render: bool):
        s, h = self.hole.spec, self.hole
        Lx, Ly, H = s.block
        Rw = h.frame()
        quat = _quat(Rw)
        # 孔工件只有外觀（網格）；銷–孔接觸由解析模型處理
        c = h.c - h.n * H / 2
        tmp = Path(tempfile.mkdtemp(prefix="t109_")) / "hole_block.obj"
        hole_block_obj(h, tmp, cut=render)
        vis = pb.createVisualShape(pb.GEOM_MESH, fileName=str(tmp), rgbaColor=(0.62, 0.64, 0.68, 1),
                                   visualFramePosition=[0, 0, H / 2], physicsClientId=self.cid)
        b = pb.createMultiBody(0, -1, vis, list(map(float, c)), quat, physicsClientId=self.cid)
        self.block_body = b
        lo = h.c - h.n * H
        if lo[2] > BENCH_TOP + 0.005:
            hh = lo[2] - BENCH_TOP
            self.add_box("fixture", (lo[0], lo[1], BENCH_TOP + hh / 2), (0.06, 0.06, hh - 1e-3),
                         (0.35, 0.36, 0.40, 1))
        if render:
            self._make_peg_visual()

    def _make_peg_visual(self):
        p = self.peg
        shapes = [  # (type, dims, rgba, z 中心（銷座標，自銷端向手部）)
            (pb.GEOM_CYLINDER, (p.r, PEG_OUT + 0.02), (0.80, 0.82, 0.86, 1), (PEG_OUT + 0.02) / 2),
            (pb.GEOM_BOX, (0.022, 0.030, GRIP_L / 2), (0.22, 0.23, 0.26, 1), PEG_OUT + GRIP_L / 2),
            (pb.GEOM_CYLINDER, (0.040, RCC_BOT_L), (0.14, 0.32, 0.62, 1), PEG_OUT + GRIP_L + RCC_BOT_L / 2),
        ]
        ids = []
        for typ, dims, rgba, z in shapes:
            if typ == pb.GEOM_CYLINDER:
                v = pb.createVisualShape(typ, radius=dims[0], length=dims[1], rgbaColor=rgba,
                                         visualFramePosition=[0, 0, z], physicsClientId=self.cid)
            else:
                v = pb.createVisualShape(typ, halfExtents=[dims[0], dims[1], dims[2]], rgbaColor=rgba,
                                         visualFramePosition=[0, 0, z], physicsClientId=self.cid)
            ids.append(pb.createMultiBody(0, -1, v, [0, 0, 0], physicsClientId=self.cid))
        self.peg_vis = ids

    def update_peg_visual(self):
        if not self.peg_vis:
            return
        q = _quat(self.body.R)
        for b in self.peg_vis:
            pb.resetBasePositionAndOrientation(b, list(map(float, self.body.p)), q, physicsClientId=self.cid)

    # --- 控制
    def _jac_tcp(self, q):
        J = self.r.jacobian(q)
        r = self.tcp_pose()[:3, 3] - self.r.fk(q)[:3, 3]
        Jv = J[:3] - np.cross(r, J[3:].T).T
        return np.vstack([Jv, J[3:]])

    def _torque(self, q, qd):
        tau = super()._torque(q, qd)
        if np.any(self.w_ff):
            tau = np.clip(tau - self._jac_tcp(q).T @ self.w_ff, -self.r.tau_max, self.r.tau_max)
        return tau

    def step(self, q_ref=None, qd_ref=None, qdd_ref=None):
        F, M = self.ext
        p = self.tcp_pose()[:3, 3]
        pb.applyExternalForce(self.sim.id, TOOL_LINK, list(map(float, F)), list(map(float, p)),
                              pb.WORLD_FRAME, physicsClientId=self.cid)
        pb.applyExternalTorque(self.sim.id, TOOL_LINK, list(map(float, M)), pb.WORLD_FRAME,
                               physicsClientId=self.cid)
        super().step(q_ref, qd_ref, qdd_ref)
        self._peg_update()

    def _peg_update(self):
        T = self.tcp_pose()
        st = pb.getLinkState(self.sim.id, TCP_LINK, computeLinkVelocity=1, physicsClientId=self.cid)
        v, w = np.array(st[6]), np.array(st[7])
        b = self.body
        if b.n_contact == 0 and b.far_from_hole():
            F, M = b.quasi_static(T, v, w)
        else:
            F, M = b.advance(T, v, w, self.dt, self.n_sub)
        self.ext = (F, M)
        # 力感測器：作用於工具的力螺旋（對 TCP），去除銷體靜重（感測器歸零），加雜訊與一階低通
        Fg = self.peg.mass * G
        Mg = np.cross(T[:3, :3] @ np.array([0, 0, -self.peg.h_com]), Fg)
        wr = np.concatenate([F - Fg, M - Mg])
        if self.ip.sigma_F > 0:
            wr = wr + np.concatenate([self.rng.normal(0, self.ip.sigma_F, 3), np.zeros(3)])
        a = self.dt / (self.dt + 1 / (2 * np.pi * self.ip.f_c))
        self.w_meas = self.w_meas + a * (wr - self.w_meas)
        self.wr_true = wr
        self.w_ff = self.w_meas + np.concatenate([Fg, Mg])  # 力前饋（含銷體靜重，感測器歸零前的值）
        if self.render_on:
            self.update_peg_visual()
        h = self.hole
        r = self.frec
        vals = (self.t, F - Fg, M - Mg, self.w_meas[:3].copy(), h.depth_of(b.p), self.hand_depth(T),
                h.lateral_of(b.p), self.hand_lat(T), b.axis_hole_deg(), b.contact_kind[0], b.contact_kind[1],
                self.phase)
        for k, val in zip(r, vals, strict=True):
            r[k].append(val)

    # 規劃端座標（名目孔：plan_c、+z）
    def hand_depth(self, T=None) -> float:
        T = self.tcp_pose() if T is None else T
        return float(-(T[:3, 3] - self.plan_c) @ self.plan_n)

    def hand_lat(self, T=None) -> np.ndarray:
        T = self.tcp_pose() if T is None else T
        return (T[:3, 3] - self.plan_c)[:2].copy()

    def _target(self, lat, z):
        """手部 TCP 目標（規劃端座標）：孔口中心 + 側向 lat(2) − n·z，工具軸朝 −z。"""
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = self.plan_c + np.array([lat[0], lat[1], 0.0]) - self.plan_n * z
        return T

    def _ref_at(self, lat, z, v3, a3, q_seed):
        T = self._target(lat, z)
        sols = self.r.ik(T @ self.T_tcp_inv, q_seed)
        if not sols:
            raise RuntimeError("UNREACHABLE")
        q = sols[0]
        J = self._jac_flange_tcp(q)
        qd = np.linalg.solve(J, np.concatenate([v3, np.zeros(3)]))
        qdd = np.linalg.solve(J, np.concatenate([a3, np.zeros(3)]))
        return q, qd, qdd

    def _jac_flange_tcp(self, q):
        J = self.r.jacobian(q)
        R = self.r.fk(q)[:3, :3]
        r = R @ self.T_tcp[:3, 3]
        Jv = J[:3] - np.cross(r, J[3:].T).T
        return np.vstack([Jv, J[3:]])

    # --- 插入製程
    def insert(self, ip: InsertParams | None = None) -> dict:
        """接近 → 觸碰（位置或導納）→（搜孔）→ 插入至目標深度 → 量測。銷保持在孔內（不退出）。
        錯誤碼：MISS（落在頂面且未搜孔/搜孔失敗）、JAM（孔內無進展）、OVERFORCE、TIMEOUT、UNREACHABLE。"""
        ip = self.ip = ip or InsertParams()
        t0 = self.t
        fi0 = len(self.frec["t"])
        out = {"strategy": ip.strategy, "search": ip.search, "F_d": ip.F_d, "v_ins": ip.v_ins,
               "K_lat": self.rcc.K_lat, "L_c": self.rcc.L_c, "mu": self.cs.mu}
        try:
            q_up = self._ref_at((0, 0), -ip.approach, np.zeros(3), np.zeros(3), Q_SEED)[0]
        except RuntimeError:
            return {**out, "ok": False, "error": "UNREACHABLE"}
        self.phase = "approach"
        self.move_j(q_up)
        # 直線降到 z_app 上方（五次）
        n_app = int(round(0.5 / self.dt))
        q = q_up
        for k in range(1, n_app + 1):
            x = k / n_app
            sm = 10 * x**3 - 15 * x**4 + 6 * x**5
            z = -ip.approach + (ip.approach - ip.z_app) * sm
            q, qd, qdd = self._ref_at((0, 0), z, np.zeros(3), np.zeros(3), q)
            self.step(q, qd, qdd)
        self.phase = "insert"
        lat = np.zeros(2)
        latd = np.zeros(2)
        z, zd = -ip.z_app, 0.0
        n = self.plan_n
        t_stall, err = 0.0, None
        searching, sphi, s_center = False, 0.0, np.zeros(2)
        land_cnt = 0
        t_contact = None
        t_ins0 = self.t
        fi_ins = len(self.frec["t"])
        while True:
            Fm = self.w_meas[:3]
            Fz = float(Fm @ n)
            Flat = Fm[:2]
            if t_contact is None and abs(Fz) > 2.0:
                t_contact = self.t
            if ip.strategy == "position":
                zd_new = ip.v_ins
                zdd = 0.0
            else:
                Fd = ip.F_search if searching else ip.F_d
                zdd = (Fd - Fz - ip.D * zd) / ip.M
                zd_new = float(np.clip(zd + zdd * self.dt, -ip.v_ins, ip.v_ins))
                zdd = (zd_new - zd) / self.dt
            zd = zd_new
            z = min(z + zd * self.dt, ip.depth + 0.002)
            a_lat = np.zeros(2)
            if searching:
                # 阿基米德螺旋：弧長速度 v_search → dφ = v·dt / sqrt(ρ² + (p/2π)²)
                b = ip.pitch / (2 * np.pi)
                rr = b * sphi
                dphi = ip.v_search * self.dt / np.hypot(rr, b)
                sphi += dphi
                rr = b * sphi
                new = s_center + rr * np.array([np.cos(sphi), np.sin(sphi)])
                latd = (new - lat) / self.dt
                lat = new
                if self.hand_depth() > ip.d_found:
                    searching = False
                    out["search_s"] = self.t - t0 - out["search_start_s"]
                    out["search_r_mm"] = rr * 1e3
                    latd = np.zeros(2)
                elif rr > ip.r_search:
                    err = "MISS"
                    break
            elif ip.strategy == "admittance" and ip.lateral:
                ldd = (Flat - ip.D_lat * latd) / ip.M_lat
                latd = latd + ldd * self.dt
                a_lat = ldd
                lat = lat + latd * self.dt
            v3 = np.concatenate([latd, [0.0]]) - n * zd
            a3 = np.concatenate([a_lat, [0.0]]) - n * zdd
            q, qd, qdd = self._ref_at(lat, z, v3, a3, q)
            self.step(q, qd, qdd)
            # 判定
            depth = self.hole.depth_of(self.body.p)
            Fz_true = float(self.wr_true[:3] @ n)
            if np.linalg.norm(self.wr_true[:3]) > ip.F_max:
                err = "OVERFORCE"
                break
            if depth >= ip.depth - 2e-4:
                break
            hd = self.hand_depth()
            if ip.strategy == "admittance" and not searching:
                if abs(zd) < ip.v_jam and Fz_true > 0.7 * ip.F_d:
                    t_stall += self.dt
                    if hd < ip.d_land:
                        land_cnt += 1
                else:
                    t_stall, land_cnt = 0.0, 0
                if ip.search and land_cnt * self.dt >= 0.1:
                    searching, sphi, s_center = True, 0.0, lat.copy()
                    t_stall, land_cnt = 0.0, 0
                    out["search_start_s"] = self.t - t0
                elif t_stall >= ip.t_jam:
                    err = "MISS" if hd < ip.d_land else "JAM"
                    break
            if self.t - t0 > ip.t_max:
                err = "TIMEOUT"
                break
        fi1 = len(self.frec["t"])
        out.update(self.metrics(fi_ins, fi1))
        depth = self.hole.depth_of(self.body.p)
        out.update({"ok": err is None, "error": err, "cycle_s": self.t - t0,
                    "insert_s": self.t - t_ins0,
                    "contact_s": self.t - t_contact if t_contact is not None else 0.0,
                    "depth_mm": depth * 1e3, "fi0": fi0, "fi_ins": fi_ins, "fi1": fi1,
                    "searched": "search_start_s" in out})
        self.phase = "hold"
        return out

    def hold(self, T: float = 0.3):
        """保持手部參考 T 秒（插入後鬆力觀察）。"""
        for _ in range(int(round(T / self.dt))):
            self.step()

    def metrics(self, fi0: int, fi1: int) -> dict:
        r = {k: np.array(v[fi0:fi1]) for k, v in self.frec.items() if k != "phase"}
        if len(r["t"]) == 0:
            return {}
        Fz = r["F"] @ self.hole.n  # 作用於工具的力沿 n（插入阻力為正）
        Flat = np.linalg.norm(r["F"] - np.outer(r["F"] @ self.hole.n, self.hole.n), axis=1)
        Mmag = np.linalg.norm(r["M"], axis=1)
        tp = (r["n_rim"] > 0) & (r["n_edge"] > 0) & (r["depth"] > self.hole.spec.w_c + 2e-4)
        tp_on = float(r["depth"][np.argmax(tp)] * 1e3) if tp.any() else float("nan")
        tau = np.array(self.rec["tau"][-len(r["t"]):])
        return {"F_ax_max": float(Fz.max()), "F_lat_max": float(Flat.max()), "M_max": float(Mmag.max()),
                "F_max": float(np.linalg.norm(r["F"], axis=1).max()),
                "tilt_max_deg": float(r["tilt"].max()),
                "lat_final_mm": float(np.linalg.norm(r["lat"][-1]) * 1e3),
                "two_point_frac": float(np.mean((r["n_rim"] > 0) & (r["n_edge"] > 0))), "tp_onset_mm": tp_on,
                "torque_util": float(np.max(np.abs(tau) / self.r.tau_max))}

    def series(self, fi0: int, fi1: int) -> dict:
        return {k: np.array(v[fi0:fi1]) for k, v in self.frec.items() if k != "phase"}


def _quat(R) -> list:
    from scipy.spatial.transform import Rotation

    return list(map(float, Rotation.from_matrix(np.asarray(R)).as_quat()))


__all__ = ["PegSpec", "HoleSpec", "RCCSpec", "RIGID", "ContactSpec", "Hole", "PegBody", "InsertParams",
           "InsertCell", "capture_radius", "chamfer_ratio", "chamfer_forces", "tilt_depth_limit",
           "wedge_angle", "two_point_statics", "jam_free_tilt", "jam_depth", "write_arm_with_peg_tool",
           "hole_block_obj"]
