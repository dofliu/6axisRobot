"""噴塗表面覆蓋（T106）：噴槍工具、塗料沉積模型、光柵（zig-zag）路徑、模擬與膜厚量測。

噴槍與 TCP
----------
直式自動噴槍以固定關節接在法蘭：安裝座＋槍身沿法蘭 z 軸，TCP = 噴嘴口（連桿 8），
TCP z 軸 = 噴霧中心軸（與法蘭 z 同向）::

    T_GUN = Tz(GUN_L)

沉積模型（點源、beta 分佈）
---------------------------
參考 Balkan & Arikan (1999)「beta 分佈噴霧通量」與 Conner et al. (2005, IEEE T-ASE 2(4))
「Paint deposition modeling for trajectory planning on automotive surfaces」的一般曲面推廣：

* 參考平面（垂直噴軸、距離 h_ref）上的膜厚速率為 beta 分佈::

      f(r) = Q·β/(π R²) · (1 − r²/R²)^(β−1),   r < R = h_ref·tan φ

  其中 Q 為沉積膜體積流量 [m³/s]（已含轉移效率），φ 為噴霧錐半角；∫ f dA = Q。
* 改寫為角通量（每立體角）：平面上 dA = h²/cos³θ dΩ，故
  F(θ) = f(h_ref·tan θ) · h_ref² / cos³θ。
* 任意曲面點 x（法向量 m）、噴嘴 p、噴軸 a：d = x − p、ρ = |d|、cos θ = d·a/ρ、
  入射角 cos γ = −d·m/ρ（背面 cos γ ≤ 0 不沉積；不考慮遮蔽）::

      ḣ(x) = F(θ) · cos γ / ρ²

  在垂直、距離 h 的平面上 ḣ(r) = f(r·h_ref/h)·(h_ref/h)²：圖樣隨 h 等比放大、峰值 ∝ 1/h²，總量仍為 Q。

單道與光柵的解析解
------------------
噴槍垂直、以速度 v 直線掃過（距離 h_ref），橫向 y 處的膜厚（沿行進方向積分）::

    g(y) = (Q·β / (π R² v)) · B(½, β) · R^(2−2β) · (R² − y²)^(β−½)

（∫ g dy = Q/v，因 B(½,β)·B(½,β+½) = π/β）。間距 Δ 的平行多道膜厚 H(y) = Σ_k g(y − kΔ)，
內部平均膜厚 = Q/(v·Δ)（質量守恆），均勻度（變異係數 CV）只與 Δ/R 與 β 有關。

光柵路徑
--------
噴嘴路徑位於與面板平行、距離 h 的平面：每道長 L + 2·overrun（兩端超出面板），道與道之間以
半徑 Δ/2 的 180° 圓弧轉向（切線連續，可沿用 T105 的 :class:`Seam` 與等速時間律）。
噴槍姿態全程固定（不隨航向旋轉，否則轉向時腕部角速度 = v/(Δ/2) 過大）。
扳機：噴嘴沿行進方向的座標 |u| ≤ L/2 + trigger 時開槍，轉向段關槍。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.special import beta as beta_fn

from .robot import SixAxisRobot
from .urdf import write_urdf
from .welding import Q_SEED, Seam, SpeedProfile, WeldCell, WeldPlan, _unit, turtle_seam

GUN_URDF = Path(__file__).resolve().parents[1] / "robots" / "ur5_spray.urdf"

# ------------------------------------------------------------------ 噴槍
GUN_L = 0.160  # 法蘭 → 噴嘴口
GUN_MASS = 0.9
GUN_COM = (0.0, 0.0, 0.070)
T_GUN = np.eye(4)
T_GUN[2, 3] = GUN_L
T_GUN_INV = np.linalg.inv(T_GUN)
GUN_LINK, TCP_LINK = 7, 8


def gun_xml(parent: str = "flange") -> str:
    """噴槍連桿 XML：gun（質量、外觀、碰撞）＋ tcp（噴嘴口座標系）。"""

    def cyl(z0, z1, r, mat, x=0.0):
        g = (f'<origin xyz="{x} 0 {(z0 + z1) / 2:.6f}"/>'
             f'<geometry><cylinder length="{z1 - z0:.6f}" radius="{r}"/></geometry>')
        return f'<visual>{g}<material name="{mat}"/></visual><collision>{g}</collision>'

    def box(z0, z1, sx, sy, mat, x=0.0):
        g = (f'<origin xyz="{x} 0 {(z0 + z1) / 2:.6f}"/>'
             f'<geometry><box size="{sx} {sy} {z1 - z0:.6f}"/></geometry>')
        return f'<visual>{g}<material name="{mat}"/></visual><collision>{g}</collision>'

    Lt = GUN_MASS * 0.16**2 / 12  # 橫向慣量：長 0.16 m 細桿 m·l²/12
    U = '  <material name="gbody"><color rgba="0.20 0.22 0.26 1"/></material>\n'
    U += '  <material name="gcap"><color rgba="0.78 0.80 0.83 1"/></material>\n'
    U += '  <material name="ghose"><color rgba="0.15 0.35 0.65 1"/></material>\n'
    U += f'  <link name="gun"><inertial><mass value="{GUN_MASS}"/>'
    U += f'<origin xyz="{GUN_COM[0]} {GUN_COM[1]} {GUN_COM[2]}"/>'
    U += f'<inertia ixx="{Lt:.6f}" iyy="{Lt:.6f}" izz="4e-4" ixy="0" ixz="0" iyz="0"/></inertial>'
    U += cyl(0.0, 0.025, 0.034, "gbody")  # 安裝座
    U += box(0.025, 0.115, 0.050, 0.040, "gbody")  # 槍身（含氣/料閥）
    U += cyl(0.040, 0.100, 0.007, "ghose", x=0.030)  # 料管接頭
    U += cyl(0.115, 0.145, 0.016, "gcap")  # 空氣帽
    U += cyl(0.145, GUN_L, 0.006, "gcap")  # 噴嘴
    U += "</link>\n"
    U += ('  <link name="tcp"><inertial><mass value="1e-6"/>'
          '<inertia ixx="1e-12" iyy="1e-12" izz="1e-12" ixy="0" ixz="0" iyz="0"/></inertial></link>\n')
    U += f'  <joint name="gun_mount" type="fixed"><parent link="{parent}"/><child link="gun"/></joint>\n'
    U += ('  <joint name="tcp_joint" type="fixed"><parent link="gun"/><child link="tcp"/>'
          f'<origin xyz="0 0 {GUN_L:.9f}"/></joint>\n')
    return U


def write_arm_with_gun(robot: SixAxisRobot | None = None, path: str | Path = GUN_URDF) -> Path:
    """手臂 URDF + 噴槍。連桿 index：7 = gun，8 = tcp（噴嘴口）。"""
    robot = robot or SixAxisRobot()
    write_urdf(robot, path)
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    path.write_text(txt.replace("</robot>", gun_xml() + "</robot>"), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 沉積模型
CC_MIN = 1e-6 / 60  # 1 cm³/min → m³/s


@dataclass
class SprayModel:
    """beta 分佈點源噴霧。flow：沉積膜體積流量 [m³/s]；h_ref：參考噴距 [m]；half_angle：錐半角 [deg]。"""

    flow: float = 20.0 * CC_MIN
    h_ref: float = 0.20
    half_angle: float = 20.0
    beta: float = 2.5

    @property
    def R(self) -> float:
        """參考噴距下的圖樣半徑 [m]。"""
        return self.h_ref * np.tan(np.radians(self.half_angle))

    def plane(self, r, h: float | None = None):
        """垂直噴軸、距離 h 的平面上，離中心 r 處的膜厚速率 [m/s]。"""
        h = self.h_ref if h is None else h
        rr = np.asarray(r, float) * self.h_ref / h
        x = np.clip(1.0 - rr**2 / self.R**2, 0.0, None)
        return self.flow * self.beta / (np.pi * self.R**2) * x ** (self.beta - 1) * (self.h_ref / h) ** 2

    def rate(self, pts, normals, nozzle, axis):
        """一般曲面點的膜厚速率 [m/s]：ḣ = F(θ)·cos γ / ρ²（見模組說明）。pts、normals 為 (N,3)。"""
        d = np.asarray(pts, float) - np.asarray(nozzle, float)
        rho = np.linalg.norm(d, axis=-1)
        ct = (d @ np.asarray(axis, float)) / rho
        cg = -np.einsum("...i,...i->...", d, np.asarray(normals, float)) / rho
        ok = (ct > 1e-9) & (cg > 0.0)
        ct_s = np.where(ok, ct, 1.0)
        tan2 = (1.0 - ct_s**2) / ct_s**2
        x = np.clip(1.0 - tan2 * self.h_ref**2 / self.R**2, 0.0, None)
        F = self.flow * self.beta / (np.pi * self.R**2) * x ** (self.beta - 1) * self.h_ref**2 / ct_s**3
        return np.where(ok, F * cg / rho**2, 0.0)

    def deposit(self, pts, normals, nozzles, axes, dt, chunk: int = 64):
        """多個噴槍位姿（各持續 dt [s]，dt 可為陣列）累積到各點的膜厚 [m]。"""
        pts, normals = np.asarray(pts, float), np.asarray(normals, float)
        nozzles, axes = np.atleast_2d(nozzles), np.atleast_2d(axes)
        dt = np.broadcast_to(np.asarray(dt, float), (len(nozzles),))
        H = np.zeros(len(pts))
        for a in range(0, len(nozzles), chunk):
            P, A, w = nozzles[a:a + chunk], axes[a:a + chunk], dt[a:a + chunk]
            d = pts[None, :, :] - P[:, None, :]  # (k, N, 3)
            rho = np.linalg.norm(d, axis=2)
            ct = np.einsum("knj,kj->kn", d, A) / rho
            cg = -np.einsum("knj,nj->kn", d, normals) / rho
            ok = (ct > 1e-9) & (cg > 0.0)
            ct_s = np.where(ok, ct, 1.0)
            tan2 = (1.0 - ct_s**2) / ct_s**2
            x = np.clip(1.0 - tan2 * self.h_ref**2 / self.R**2, 0.0, None)
            F = x ** (self.beta - 1) / ct_s**3
            H += (w[:, None] * np.where(ok, F * cg / rho**2, 0.0)).sum(0)
        return H * self.flow * self.beta / (np.pi * self.R**2) * self.h_ref**2

    # --- 解析
    def stroke(self, y, v: float):
        """單道（垂直、距離 h_ref、速度 v）的橫向膜厚 g(y) [m]。"""
        R, b = self.R, self.beta
        y2 = np.clip(R**2 - np.asarray(y, float) ** 2, 0.0, None)
        return self.flow * b / (np.pi * R**2 * v) * beta_fn(0.5, b) * R ** (2 - 2 * b) * y2 ** (b - 0.5)

    def raster(self, y, v: float, spacing: float, n: int):
        """n 道、間距 spacing、以 0 為中心的平行道膜厚 H(y)。"""
        y = np.asarray(y, float)
        ys = (np.arange(n) - (n - 1) / 2) * spacing
        return sum(self.stroke(y - c, v) for c in ys)

    def ripple(self, spacing: float, n_per: int = 400):
        """無限多道時一個週期內的 (CV, (max−min)/mean)；與 v、Q 無關。"""
        y = np.linspace(0.0, spacing, n_per, endpoint=False)
        k = int(np.ceil(self.R / spacing)) + 1
        H = sum(self.stroke(y - j * spacing, 1.0) for j in range(-k, k + 1))
        return float(H.std() / H.mean()), float((H.max() - H.min()) / H.mean())

    def max_spacing(self, cv_max: float = 0.02, lo: float = 0.2, hi: float = 1.6, step: float = 0.0025):
        """由 Δ/R = lo 往上掃，回傳「CV 首次超過 cv_max 之前」的最大間距 [m]（保守：CV(Δ) 非單調，
        較大 Δ 雖有孤立甜蜜點但對噴距/圖樣變化敏感，故不採用）。lo 即不合格時回傳 None。"""
        best = None
        for r in np.arange(lo, hi + 1e-12, step):
            if self.ripple(r * self.R)[0] > cv_max:
                break
            best = float(r * self.R)
        return best


# ------------------------------------------------------------------ 面板與光柵路徑
class Panel:
    """平面矩形面板（工件表面）：中心 c、法向 n、行進方向 u（道方向）、尺寸 (L 沿 u, W 沿 w = n×u)。"""

    def __init__(self, center, normal=(0, 0, 1), u=(0, 1, 0), size=(0.40, 0.30), res: float = 0.0025,
                 name: str = "panel"):
        self.c = np.asarray(center, float)
        self.n = _unit(normal)
        uu = np.asarray(u, float)
        self.u = _unit(uu - (uu @ self.n) * self.n)
        self.w = np.cross(self.n, self.u)
        self.L, self.W = float(size[0]), float(size[1])
        self.name, self.res = name, res
        nu, nw = int(round(self.L / res)), int(round(self.W / res))
        self.gu = (np.arange(nu) + 0.5) * self.L / nu - self.L / 2
        self.gw = (np.arange(nw) + 0.5) * self.W / nw - self.W / 2
        U, Wg = np.meshgrid(self.gu, self.gw, indexing="ij")
        self.pts = self.c + U.reshape(-1, 1) * self.u + Wg.reshape(-1, 1) * self.w
        self.normals = np.broadcast_to(self.n, self.pts.shape)
        self.shape = (nu, nw)
        self.cell_area = (self.L / nu) * (self.W / nw)

    def to_local(self, p):
        d = np.atleast_2d(p) - self.c
        return d @ self.u, d @ self.w, d @ self.n

    def rotation(self) -> np.ndarray:
        return np.column_stack([self.u, self.w, self.n])


def raster_path(panel: Panel, spacing: float, standoff: float = 0.20, overrun: float = 0.08,
                edge: float = 0.0, name: str = "raster") -> tuple[Seam, np.ndarray]:
    """面板上方（沿 n 距離 standoff）的 zig-zag 噴嘴路徑。道數 n = ceil((W + 2·edge)/Δ) + 1，
    以面板中心對稱排列（edge > 0 時兩側道超出面板邊）。回傳 (Seam, 各道中心 w 座標)。"""
    if spacing <= 0 or standoff <= 0:
        raise ValueError("spacing、standoff 需 > 0")
    n = int(np.ceil((panel.W + 2 * edge) / spacing - 1e-9)) + 1
    ws = (np.arange(n) - (n - 1) / 2) * spacing
    length = panel.L + 2 * overrun
    start = panel.c + panel.n * standoff - panel.u * length / 2 + panel.w * ws[0]
    moves = []
    for k in range(n):
        moves.append(("line", length))
        if k < n - 1:  # 往 +w 方向換道：沿 +u 時左轉（繞 n 逆時針），沿 −u 時右轉
            moves.append(("arc", spacing / 2, 180.0 if k % 2 == 0 else -180.0))
    return turtle_seam(start, panel.u, moves, normal=panel.n, name=name), ws


def gun_rotation(axis, u) -> np.ndarray:
    """噴軸 = axis（TCP z）、TCP x 盡量沿 u 的固定姿態。"""
    z = _unit(axis)
    x = _unit(np.asarray(u, float) - (np.asarray(u, float) @ z) * z)
    return np.column_stack([x, np.cross(z, x), z])


@dataclass
class SprayParams:
    speed: float = 0.30  # 噴嘴行進速度 [m/s]
    spacing: float = 0.050  # 道間距 [m]
    standoff: float = 0.20  # 噴距 [m]（沿面板法向）
    overrun: float = 0.08  # 每道兩端超出面板的長度 [m]
    trigger: float = 0.03  # 開槍區：|u| ≤ L/2 + trigger
    edge: float = 0.0  # 道排列超出面板側邊的量 [m]
    tilt_deg: float = 0.0  # 噴軸相對 −n 繞 u 的傾角（0 = 垂直面板；> 0 噴軸偏向 +w）
    t_ramp: float = 0.4
    approach: float = 0.06


def gun_axis(panel: Panel, prm: SprayParams) -> np.ndarray:
    t = np.radians(prm.tilt_deg)
    return _unit(-np.cos(t) * panel.n + np.sin(t) * panel.w)


def plan_spray(robot: SixAxisRobot, seam: Seam, R_tool: np.ndarray, speed: float, q_seed, t_ramp: float = 0.4,
               dt: float = 0.01, prm=None, t_tool_inv=None) -> WeldPlan:
    """噴嘴路徑（固定姿態）→ 關節軌跡：等速時間律取樣、逐點數值 IK（前一點為初值）、q̇ = J⁻¹·[v; 0]。
    t_tool_inv：TCP→法蘭的 4×4（預設噴槍；T108 研磨主軸沿用）。"""
    T_inv = T_GUN_INV if t_tool_inv is None else np.asarray(t_tool_inv, float)
    prof = SpeedProfile(seam.length, speed, t_ramp)
    n = max(int(round(prof.T / dt)), 2)
    t = np.linspace(0.0, prof.T, n + 1)
    s, sd = prof(t)
    P, Tg = seam.eval(s)[0], seam.eval(s)[1]
    Tt = np.eye(4)
    Tt[:3, :3] = R_tool
    Tt[:3, 3] = P[0]
    sols = robot.ik(Tt @ T_inv, q_seed)
    if not sols:
        z = np.zeros((n + 1, 6))
        return WeldPlan(seam, prm, prof.T, t, z, z.copy(), n + 1, 0, 0, 0, False)
    qk = sols[0]
    q, qd = np.zeros((n + 1, 6)), np.zeros((n + 1, 6))
    fail, sig = 0, np.inf
    for k in range(n + 1):
        Tt[:3, 3] = P[k]
        qk, ok, _ = robot.ik_numeric(Tt @ T_inv, qk, tol=1e-10, damping=1e-8)
        fail += int(not ok)
        J = robot.jacobian(qk)
        sig = min(sig, float(np.linalg.svd(J, compute_uv=False)[-1]))
        tw = np.concatenate([Tg[k] * sd[k], np.zeros(3)])  # 姿態固定 → 法蘭速度 = 噴嘴速度
        qd[k] = J.T @ np.linalg.solve(J @ J.T + 1e-10 * np.eye(6), tw)
        q[k] = qk
    spl = CubicHermiteSpline(t, q, qd)
    qdd = spl.derivative(2)(t)
    tau = np.array([robot.rne(a, b, c) for a, b, c in zip(q, qd, qdd, strict=True)])
    plan = WeldPlan(seam, prm, prof.T, t, q, qd, fail,
                    float(np.max(np.abs(qd) / robot.qd_max)), float(np.max(np.abs(tau) / robot.tau_max)), sig,
                    bool(np.all(q >= robot.q_min) and np.all(q <= robot.q_max)), spl)
    plan._d1, plan._d2 = spl.derivative(1), spl.derivative(2)
    plan.meta = {"qdd_max": float(np.max(np.abs(qdd))), "s": s}
    return plan


# ------------------------------------------------------------------ 工作站
BENCH_TOP = 0.10


class SprayCell(WeldCell):
    """手臂 + 噴槍 + 工作台 + 面板。沿用 :class:`WeldCell` 的力矩控制、記錄與碰撞量測；
    rec["arc"] 記錄扳機（開槍）狀態。"""

    def __init__(self, model: SprayModel | None = None, **kw):
        super().__init__(tool_writer=write_arm_with_gun, t_tcp=T_GUN, **kw)
        self.model = model or SprayModel()
        self.panels: dict[str, Panel] = {}

    def add_panel(self, panel: Panel, thick: float = 0.004, rgba=(0.80, 0.81, 0.83, 1), support: bool = True):
        """面板（薄板，頂面 = panel 平面）；support=True 時加一根撐柱到工作台。"""
        from scipy.spatial.transform import Rotation

        q = Rotation.from_matrix(panel.rotation()).as_quat()
        self.add_box(panel.name, panel.c - panel.n * thick / 2, (panel.L, panel.W, thick), rgba, q)
        lo = panel.c - panel.n * thick
        if support and lo[2] > BENCH_TOP + 0.01:
            h = lo[2] - BENCH_TOP
            self.add_box(f"{panel.name}_post", (lo[0], lo[1], BENCH_TOP + h / 2 - 0.005),
                         (0.04, 0.04, h - 0.01), (0.35, 0.36, 0.40, 1))
        self.panels[panel.name] = panel

    def spray(self, panel: Panel, prm: SprayParams | None = None, q_seed=None, margin: float = 5e-3) -> dict:
        """進槍 → 光柵噴塗（扳機依 |u| 判定）→ 退槍。回傳 dict（含 i0/i1 供 :meth:`thickness`）。
        錯誤碼：PLAN、UNREACHABLE、LIMIT（限位/速度/力矩）、COLLISION（預檢間隙 < margin 或執行中接觸）。"""
        prm = prm or SprayParams()
        try:
            R_tool = gun_rotation(gun_axis(panel, prm), panel.u)
            a = R_tool[:, 2]
            cos_n = float(-a @ panel.n)
            if cos_n < 0.2:
                raise ValueError("噴軸與面板法向夾角過大")
            # 噴嘴 = 瞄準點 − a·standoff/cos：噴軸傾斜時整條路徑平移，使噴霧中心仍落在道中心線上
            shift = -a * prm.standoff / cos_n - panel.n * prm.standoff
            seam, ws = raster_path(Panel(panel.c + shift, panel.n, panel.u, (panel.L, panel.W), 0.05),
                                   prm.spacing, prm.standoff, prm.overrun, prm.edge)
            plan = plan_spray(self.r, seam, R_tool, prm.speed, Q_SEED if q_seed is None else q_seed,
                              prm.t_ramp, prm=prm)
        except ValueError as e:
            return {"ok": False, "error": "PLAN", "message": str(e)}
        base = {"panel": panel.name, "strokes": len(ws), "path_len_m": seam.length, "speed_cmd": prm.speed,
                "spacing": prm.spacing, "standoff": prm.standoff, "plan_vel_ratio": plan.vel_ratio,
                "plan_torque_ratio": plan.torque_ratio, "sigma_min": plan.sigma_min, "ik_fail": plan.ik_fail,
                "plan_qdd_max": plan.meta.get("qdd_max")}
        if plan.ik_fail:
            return {**base, "ok": False, "error": "UNREACHABLE"}
        if not plan.ok:
            return {**base, "ok": False, "error": "LIMIT"}
        pre = self.precheck(plan)
        base["precheck_clear_mm"] = pre * 1e3
        if pre < margin:
            return {**base, "ok": False, "error": "COLLISION"}
        Tf0 = np.eye(4)
        Tf0[:3, :3] = R_tool
        Tf0[:3, 3] = seam.p[0]
        Tf1 = Tf0.copy()
        Tf1[:3, 3] = seam.p[-1]
        back = np.eye(4)
        back[2, 3] = -prm.approach  # 沿噴軸反向退
        q_app, ok, _ = self.r.ik_numeric(Tf0 @ back @ T_GUN_INV, plan.q[0], damping=1e-8)
        if not ok:
            return {**base, "ok": False, "error": "UNREACHABLE"}
        t_begin = self.t
        self.move_j(q_app)
        self.move_l(Tf0 @ T_GUN_INV, 0.6)
        self.step(plan.q[0])
        self.min_clear = np.inf
        i0 = len(self.rec["t"])
        lim = panel.L / 2 + prm.trigger
        for k in range(1, int(round(plan.T / self.dt)) + 1):
            aim = self.rec["tcp"][-1] + a * prm.standoff / cos_n  # 噴軸與面板平面交點（命令姿態）
            self.arc = bool(abs((aim - panel.c) @ panel.u) <= lim)
            self.step(*plan.state(k * self.dt))
        self.arc = False
        self.step(plan.q[-1])
        i1 = len(self.rec["t"])
        run_clear = self.min_clear
        self.move_l(Tf1 @ back @ T_GUN_INV, 0.6)
        hit = bool(run_clear <= 0.0)
        m = {**base, "ok": not hit, "error": "COLLISION" if hit else None, "spray_s": plan.T,
             "cycle_s": self.t - t_begin, "clear_mm": run_clear * 1e3, "i0": i0, "i1": i1}
        m.update(self.motion_metrics(seam, prm, a, i0, i1))
        return m

    # --- 量測
    def motion_metrics(self, seam: Seam, prm: SprayParams, axis, i0: int, i1: int) -> dict:
        sl = slice(i0, i1)
        t = np.array(self.rec["t"][sl])
        t = t - t[0] + self.dt
        p = np.array(self.rec["tcp"][sl])
        q, qd = np.array(self.rec["q"][sl]), np.array(self.rec["qd"][sl])
        on = np.array(self.rec["arc"][sl])
        prof = SpeedProfile(seam.length, prm.speed, prm.t_ramp)
        s_ref, _ = prof(t)
        _, p_n, t_n = seam.nearest(p, s_ref, window=0.02)
        e = p - p_n
        off = np.linalg.norm(e - np.einsum("ij,ij->i", e, t_n)[:, None] * t_n, axis=1)
        vel = np.gradient(p, t, axis=0)
        spd = np.linalg.norm(vel, axis=1)
        tz = np.array(self.rec["tool_z"][sl])
        axis_err = np.degrees(np.arctan2(np.linalg.norm(np.cross(tz, axis), axis=1), tz @ axis))
        trig = on & (t > prm.t_ramp) & (t < prof.T - prm.t_ramp)
        return {
            "path_dev_max_mm": float(off.max() * 1e3),
            "path_dev_rms_mm": float(np.sqrt(np.mean(off**2)) * 1e3),
            "path_dev_on_max_mm": float(off[on].max() * 1e3) if on.any() else None,
            "speed_dev_on_max_pct": (float(np.max(np.abs(spd[trig] / prm.speed - 1)) * 100)
                                     if trig.any() else None),
            "axis_err_max_deg": float(axis_err.max()),
            "vel_util": float(np.max(np.abs(qd) / self.r.qd_max)),
            "torque_util": float(np.max(np.abs(np.array(self.rec["tau"][sl])) / self.r.tau_max)),
            "on_s": float(on.sum() * self.dt), "q_range_deg": np.degrees(q.max(0) - q.min(0)).tolist(),
        }

    def thickness(self, panel: Panel, i0: int, i1: int, model: SprayModel | None = None,
                  every: int = 1) -> np.ndarray:
        """記錄段 [i0, i1) 中開槍時刻的實際噴嘴位姿 → 面板各格點膜厚 [m]（形狀 panel.shape）。
        every > 1 時每 every 步取一個位姿、權重 every·dt（矩形積分）。"""
        model = model or self.model
        on = np.array(self.rec["arc"][i0:i1], bool)
        idx = np.arange(i0, i1)[on][::every]
        cnt = np.add.reduceat(on[on].astype(float), np.arange(0, on.sum(), every)) if len(idx) else []
        P = np.array(self.rec["tcp"])[idx] if len(idx) else np.zeros((0, 3))
        A = np.array(self.rec["tool_z"])[idx] if len(idx) else np.zeros((0, 3))
        return model.deposit(panel.pts, panel.normals, P, A, np.asarray(cnt) * self.dt).reshape(panel.shape)


def film_metrics(H: np.ndarray, panel: Panel, target: float | None = None, border: float = 0.0,
                 tol: float = 0.10) -> dict:
    """膜厚統計：平均、CV、(max−min)/mean、目標 ±tol 內面積比、覆蓋率（≥ (1−tol)·target）。
    border > 0 時排除面板外緣該寬度。target 預設為平均值。單位：µm。"""
    U, Wg = np.meshgrid(panel.gu, panel.gw, indexing="ij")
    sel = (np.abs(U) <= panel.L / 2 - border) & (np.abs(Wg) <= panel.W / 2 - border)
    h = H[sel]
    mean = float(h.mean())
    tg = mean if target is None else target
    return {"mean_um": mean * 1e6, "min_um": float(h.min() * 1e6), "max_um": float(h.max() * 1e6),
            "cv_pct": float(h.std() / mean * 100), "range_pct": float((h.max() - h.min()) / mean * 100),
            "within_tol_pct": float(np.mean(np.abs(h / tg - 1) <= tol) * 100),
            "coverage_pct": float(np.mean(h >= (1 - tol) * tg) * 100), "target_um": tg * 1e6,
            "border_mm": border * 1e3}


def transfer_efficiency(H: np.ndarray, panel: Panel, model: SprayModel, on_s: float) -> float:
    """幾何轉移效率 = 面板上沉積體積 / 開槍期間噴出體積（模型的 Q 已含霧化損失外的轉移率）。"""
    return float(H.sum() * panel.cell_area / (model.flow * on_s))


# ------------------------------------------------------------------ JSON 工作描述
def parse_job(job):
    """JSON 噴塗工作 → (SprayModel, [(Panel, SprayParams)])。格式::

        {"gun": {"flow_cc_min": 20, "h_ref": 0.2, "half_angle": 20, "beta": 2.5},
         "panels": [{"name": "door", "center": [x,y,z], "normal": [0,0,1], "u": [0,1,0],
                     "size": [0.4, 0.3], "speed": 0.3, "spacing": 0.05, "standoff": 0.2}]}

    格式錯誤拋 ValueError。
    """
    if isinstance(job, str):
        job = json.loads(job)
    if not isinstance(job, dict) or not isinstance(job.get("panels"), list) or not job["panels"]:
        raise ValueError("工作需為含 'panels' 陣列的物件")
    g = job.get("gun", {})
    unknown = set(g) - {"flow_cc_min", "h_ref", "half_angle", "beta"}
    if unknown:
        raise ValueError(f"gun 未知參數 {sorted(unknown)}")
    model = SprayModel(flow=float(g.get("flow_cc_min", 20.0)) * CC_MIN, h_ref=float(g.get("h_ref", 0.2)),
                       half_angle=float(g.get("half_angle", 20.0)), beta=float(g.get("beta", 2.5)))
    if not (0 < model.half_angle < 60 and model.beta >= 1.0 and model.flow > 0):
        raise ValueError("噴槍參數超出範圍（0 < half_angle < 60、beta ≥ 1、flow > 0）")
    keys = set(SprayParams.__dataclass_fields__)
    out = []
    for i, pd in enumerate(job["panels"]):
        if "center" not in pd or "size" not in pd:
            raise ValueError(f"panels[{i}] 缺少 center / size")
        unknown = set(pd) - keys - {"name", "center", "normal", "u", "size"}
        if unknown:
            raise ValueError(f"panels[{i}] 未知參數 {sorted(unknown)}")
        prm = SprayParams(**{k: float(pd[k]) for k in keys if k in pd})
        if not 0 < prm.speed <= 1.0 or not 0 < prm.spacing <= 0.3 or not 0.05 <= prm.standoff <= 0.5:
            raise ValueError(f"panels[{i}] speed/spacing/standoff 超出範圍")
        panel = Panel(pd["center"], pd.get("normal", (0, 0, 1)), pd.get("u", (0, 1, 0)), pd["size"],
                      name=pd.get("name", f"panel{i}"))
        out.append((panel, prm))
    return model, out


__all__ = ["CC_MIN", "Panel", "SprayCell", "SprayModel", "SprayParams", "T_GUN", "film_metrics",
           "gun_rotation", "parse_job", "plan_spray", "raster_path", "transfer_efficiency",
           "write_arm_with_gun"]
