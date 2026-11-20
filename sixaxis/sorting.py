"""T111 多物件分類揀選：俯視相機感知 → 抓取規劃（夾爪掃掠體 vs 鄰物）→ 序列最佳化 → 避障搬運 → 分類放置。

1. 工作站（沿用 T006）：工作台取料區散置三類物件
   * 紅色方塊 50 mm（`CUBE`）→ 料盤 4 格（精密放置，格內單邊間隙 4 mm）
   * 藍色立方柱 40×60×70 mm → A 箱；綠色圓柱 Ø50×60 mm → B 箱
   料箱放在工作台 −y 側獨立台座上；兩箱之間後方的立柱＋橫臂為俯視相機支架（同時是障礙物）。

2. 感知（`OverheadCamera`）：PyBullet 針孔相機（TinyRenderer）RGB＋深度。
   * 反投影：像素 (u,v)、深度緩衝 d → NDC (x, y, 2d−1, 1)，世界座標 X = (P V)^{-1} x_ndc 再除以 w
     （OpenGL 投影管線的逆運算；P、V 由 computeProjectionMatrixFOV / computeViewMatrix 給出，column-major）。
   * 前景 = 取料區 ROI 內、高於桌面 6 mm 且低於 0.20 m 的點（以高度分割，不受陰影/光照影響）；
     8-連通標記；頂面點 = z ≥ z_98% − 4 mm。
   * 位置 = 頂面點 xy 質心；尺寸/方位 = 頂面凸包的最小面積外接矩形
     （旋轉卡尺：最小外接矩形必有一邊與凸包某邊共線，Freeman & Shapira 1975）；
     圓形度 = 凸包面積 / 外接矩形面積（圓 π/4 ≈ 0.785、矩形 ≈ 1）。
   * 類別 = 色度 (r,g,b)/Σ 與各類參考色度最近者（對亮度/陰影不敏感），並以形狀/高度做一致性檢查。

3. 抓取規劃（`grasp_candidates`）：候選 yaw ψ（方塊：兩組對邊；圓柱：每 22.5°），夾持寬度 ≤ 開口 − 2·餘裕。
   夾爪碰撞幾何（兩指墊、掌體）在 ψ 下的俯視矩形，與「頂面高於該部件下緣」的鄰物俯視多邊形距離 ≥ 餘裕才可行；
   垂直下降掃掠體 = 俯視矩形 × [部件下緣, ∞)，故此 2.5D 檢查即為下降全程的碰撞檢查。凸多邊形距離以 SAT 判交、
   否則取頂點–邊最小距離。被鄰物擋住的物件需等鄰物取走後才可抓（序列相依）。

4. 序列（`plan_sequence`）：每次揀選成本 = 前一放置點上方 → 物件上方 → 放置點上方 兩段關節 PTP 的最短時間估計
   （`min_duration`，與 T007 執行器同一速度上限）＋固定的升降/夾放時間。
   放置點依類別依序填格，故狀態 =（剩餘集合, 上一物件），
   以 Held–Karp 動態規劃（O(2^n n²)）求含可抓相依限制的最佳順序；比較：掃描順序、貪婪最近鄰。
   執行時為滾動時域：每次放置後重新拍照、重新規劃、只執行第一個。

5. 搬運：`CollisionChecker`（T110）加入其餘物件（可移動障礙物）與夾持工件（附在 TCP 的方塊）；
   關節直線邊通過保證式檢查則直接 PTP，否則 Informed RRT*（第一解）＋捷徑。

6. 失敗恢復：抬升後檢查物件是否在夾爪內且置中（T102 `inhand_ok` 同判據）；否則放回、鬆開、重新拍照重抓。
"""

from __future__ import annotations

import itertools
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as pb
from scipy import ndimage
from scipy.spatial import ConvexHull

from . import scene as S
from .commands import NOMINAL, Executor
from .graspsim import Q_REF, R_DOWN, GraspScene
from .gripper import CUBE, PAD_L, PAD_T, PAD_W, S_MAX, T_TCP, TCP_Z, W0, ObjectSpec
from .planning import Box, CollisionChecker, RRTStar, path_length, shortcut
from .robot import rotz, wrap_pi
from .scenarios import workcell_boxes
from .tasks import Z_POCKET
from .trajectory import min_duration

# ---------------------------------------------------------------------- 物件類別與目的地
BLOCK_S = ObjectSpec("block", "box", (0.04, 0.06, 0.07), 0.25, rgba=(0.2, 0.45, 0.85, 1.0))
CYL_S = ObjectSpec("cyl", "cylinder", (0.025, 0.06), 0.15, rgba=(0.2, 0.7, 0.35, 1.0))


@dataclass(frozen=True)
class PartClass:
    name: str
    spec: ObjectSpec
    dest: str  # "tray" / "binA" / "binB"

    @property
    def chroma(self) -> np.ndarray:
        c = np.array(self.spec.rgba[:3])
        return c / c.sum()


CLASSES = {
    "red": PartClass("red", CUBE, "tray"),
    "blue": PartClass("blue", BLOCK_S, "binA"),
    "green": PartClass("green", CYL_S, "binB"),
}

# ---------------------------------------------------------------------- 佈置（單位 m）
ZONE = (0.32, 0.60, -0.07, 0.115)  # 物件俯視輪廓須在此矩形內（避開料盤 y < −0.084）
GAP = 0.012  # 散置時物件輪廓最小間距
Z_CLEAR = 0.27  # 搬運 TCP 高度（最高物件頂 0.17 + 夾持工件下垂 0.035 + 餘裕）
BIN_C = {"binA": (0.30, -0.40), "binB": (0.56, -0.40)}
BIN_IN_X, BIN_IN_Y = 0.065, 0.09  # 箱內半寬（x 窄、y 長）
BIN_WALL_T, BIN_WALL_H, BIN_PLATE_T = 0.006, 0.05, 0.004  # 壁厚、壁高、底板厚
STAND_TOP = 0.10
BIN_FLOOR = STAND_TOP + BIN_PLATE_T
BIN_WALL_TOP = BIN_FLOOR + BIN_WALL_H
BIN_SLOT = 0.04  # 箱內投放點沿 y 偏移 ±40 mm（兩點）
BIN_PSI = np.pi / 2  # 箱內放置時兩指沿 x 閉合：指墊外緣 |x| ≤ 55 mm < 內半寬 65 mm，可下降到箱內
# 放入箱內時物件底面離箱底的高度：0 = 放到底再鬆開
# （曾用 5 mm 懸空鬆爪：PyBullet 圓柱–指墊接觸法向偏斜＋錨點摩擦，偶發圓柱黏在指墊上被帶走）
BIN_LIFT = 0.0
PAD_OVER = 0.003  # 夾取時指墊上緣高出物件頂面量
MAST_XY, MAST_HALF, MAST_TOP = (0.435, -0.50), 0.02, 1.0
CAM_EYE, CAM_TARGET = (0.46, 0.02, 0.95), (0.46, 0.02, 0.10)
Q_HOME = None  # 於 _home_q() 計算：TCP (0.22, −0.22, 0.42)，相機視野外

# 夾爪碰撞部件（夾爪座標系 x、y 範圍與相對 TCP 的下緣高度）
GRIP_MARGIN = 0.004
PAD_RECTS = [((-PAD_W / 2, PAD_W / 2), (s * (W0 - PAD_T / 2), s * (W0 + PAD_T / 2))) for s in (1, -1)]
PAD_BOTTOM = -PAD_L / 2  # 指墊下緣（相對 TCP）
PALM_RECT = ((-0.0375, 0.0375), (-0.055, 0.055))
PALM_BOTTOM = TCP_Z - 0.050  # 掌體下緣（法蘭下 50 mm）在 TCP 上方 45 mm
OPEN_W = 2 * (W0 - PAD_T / 2)  # 全開內距 80 mm


def tray_slots():
    return [(x, y) for x, y in S.pocket_centers()]


def bin_slots(name: str):
    cx, cy = BIN_C[name]
    return [(cx, cy - BIN_SLOT), (cx, cy + BIN_SLOT)]


def dest_slots(dest: str):
    return tray_slots() if dest == "tray" else bin_slots(dest)


def grasp_offset(height: float) -> float:
    """夾取 TCP 高於物件中心的量：指墊上緣高出物件頂面 PAD_OVER（TCP = 頂面 − PAD_L/2 + PAD_OVER）。
    比夾在中心高 h/2 − 17 mm，使夾 50 mm 方塊時掌體下緣（TCP 上方 45 mm）高出 70 mm 鄰物頂面 8 mm。
    不取「齊平」：指墊上緣與物件頂面共面時，方塊–方塊接觸流形出現共線稜邊，實測 PyBullet 接觸法向翻轉、
    鬆爪時物件被指墊黏住帶走。"""
    return max(height / 2 - PAD_L / 2 + PAD_OVER, 0.0)


def place_height(cls: PartClass, height: float | None = None) -> float:
    """放置時 TCP 高度。料盤：物件中心 = Z_POCKET；料箱：物件底面離箱底 BIN_LIFT。"""
    h = cls.spec.height if height is None else height
    if cls.dest == "tray":
        return Z_POCKET + grasp_offset(h)
    return BIN_FLOOR + BIN_LIFT + h / 2 + grasp_offset(h)


def place_yaw(cls: PartClass, psi_grasp: float = 0.0, T_tcp_obj=None) -> float:
    """放置時夾爪 yaw。料盤：ψ = 0（兩指須沿 y 伸入側銷之間；夾取時 ψ 已對齊方塊面 → 放入時方塊亦對齊格）。
    料箱：ψ = 90°（兩指沿 x 閉合，指墊不碰長邊箱壁，物件可放到箱底而非由壁頂上方掉落）。
    兩者都只取決於目的地；物件相對夾爪 yaw δ（T_tcp_obj）在料盤情形由夾取時對齊面保證為 90° 倍數。"""
    return 0.0 if cls.dest == "tray" else BIN_PSI


def sort_boxes() -> list[Box]:
    """分類工作站新增的靜態方塊：兩箱台座、底板、四壁，相機立柱與橫臂。"""
    out = []
    for nm, (cx, cy) in BIN_C.items():
        hx, hy = BIN_IN_X + BIN_WALL_T, BIN_IN_Y + BIN_WALL_T
        out.append(Box((cx, cy, STAND_TOP / 2), (hx + 0.004, hy + 0.004, STAND_TOP / 2), f"{nm}_stand"))
        out.append(Box((cx, cy, STAND_TOP + BIN_PLATE_T / 2), (hx, hy, BIN_PLATE_T / 2), f"{nm}_floor"))
        zc = BIN_FLOOR + BIN_WALL_H / 2
        ox, oy = BIN_IN_X + BIN_WALL_T / 2, BIN_IN_Y + BIN_WALL_T / 2
        walls = ((ox, 0, BIN_WALL_T / 2, hy), (-ox, 0, BIN_WALL_T / 2, hy),
                 (0, oy, BIN_IN_X, BIN_WALL_T / 2), (0, -oy, BIN_IN_X, BIN_WALL_T / 2))
        for k, (dx, dy, hx, hy) in enumerate(walls):
            out.append(Box((cx + dx, cy + dy, zc), (hx, hy, BIN_WALL_H / 2), f"{nm}_wall{k}"))
    mx, my = MAST_XY
    out.append(Box((mx, my, MAST_TOP / 2), (MAST_HALF, MAST_HALF, MAST_TOP / 2), "mast"))
    out.append(Box((mx, (my + CAM_EYE[1]) / 2, MAST_TOP + 0.015), (0.03, (CAM_EYE[1] - my) / 2 + 0.03, 0.015),
                   "boom"))
    return out


_STATIC_FP = None


def static_footprints() -> list:
    """取料區附近靜態體（料盤、銷、治具…，不含工作台本身）的俯視多邊形與頂面高度，供抓取檢查。"""
    global _STATIC_FP
    if _STATIC_FP is None:
        x0, x1, y0, y1 = ZONE
        out = []
        for b in workcell_boxes():
            if b.name in ("bench", "cube"):
                continue
            c, h = b.center, b.half
            if (c[0] + h[0] < x0 - 0.1 or c[0] - h[0] > x1 + 0.1
                    or c[1] + h[1] < y0 - 0.1 or c[1] - h[1] > y1 + 0.1):
                continue
            out.append((rect_poly(c[:2], b.yaw, h[0], h[1]), c[2] + h[2]))
        _STATIC_FP = out
    return _STATIC_FP


def fixture_keepout() -> list[np.ndarray]:
    """治具（工作台上靜態體）俯視多邊形，散置時物件須遠離。"""
    fx, fy = S.FIX_NOM
    return [rect_poly((fx + 0.025 + S.FIX_WALL_T / 2, fy), 0.0, S.FIX_WALL_T / 2, 0.05),
            rect_poly((fx + 0.019, fy + 0.025 + S.FIX_WALL_T / 2), 0.0, 0.006, S.FIX_WALL_T / 2)]


# ---------------------------------------------------------------------- 2D 幾何
def rect_poly(c, yaw, hx, hy) -> np.ndarray:
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return pts @ R.T + np.asarray(c, float)


def circle_poly(c, r, n: int = 24) -> np.ndarray:
    """外切正 n 邊形（包含圓 → 保守）。"""
    a = np.arange(n) * 2 * np.pi / n
    rr = r / np.cos(np.pi / n)
    return np.asarray(c, float) + rr * np.c_[np.cos(a), np.sin(a)]


def footprint(spec: ObjectSpec, xy, yaw) -> np.ndarray:
    if spec.shape == "box":
        return rect_poly(xy, yaw, spec.size[0] / 2, spec.size[1] / 2)
    return circle_poly(xy, spec.size[0])


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _overlap(A, B) -> bool:
    """分離軸定理（凸多邊形）：所有邊法向投影皆重疊才相交。"""
    for P in (A, B):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            n = np.array([-e[1], e[0]])
            pa, pb_ = A @ n, B @ n
            if pa.max() < pb_.min() or pb_.max() < pa.min():
                return False
    return True


def poly_dist(A, B) -> float:
    """兩凸多邊形距離（相交為 0）。"""
    if _overlap(A, B):
        return 0.0
    d = np.inf
    for P, Q in ((A, B), (B, A)):
        for p in P:
            for i in range(len(Q)):
                d = min(d, _seg_dist(p, Q[i], Q[(i + 1) % len(Q)]))
    return float(d)


# ---------------------------------------------------------------------- 散置
@dataclass
class Part:
    cls: PartClass
    xy: tuple
    yaw: float
    body: int = -1

    @property
    def poly(self) -> np.ndarray:
        return footprint(self.cls.spec, self.xy, self.yaw)

    @property
    def top(self) -> float:
        return S.BENCH_TOP + self.cls.spec.height


DEFAULT_MIX = (("red", 2), ("blue", 2), ("green", 2))


def scatter(seed: int, mix=DEFAULT_MIX, gap: float = GAP, zone=ZONE, max_tries: int = 20000) -> list[Part]:
    """在取放區隨機散置（拒絕取樣）：輪廓在 zone 內、彼此距離 ≥ gap、離治具 ≥ 10 mm；位置/yaw 均勻分佈。"""
    rng = np.random.default_rng(seed)
    keep = fixture_keepout()
    for _ in range(200):
        parts: list[Part] = []
        names = [n for n, k in mix for _ in range(k)]
        rng.shuffle(names)
        ok = True
        for nm in names:
            cls = CLASSES[nm]
            for _t in range(max_tries // 200):
                xy = (rng.uniform(zone[0], zone[1]), rng.uniform(zone[2], zone[3]))
                yaw = rng.uniform(-np.pi, np.pi)
                P = footprint(cls.spec, xy, yaw)
                if (P[:, 0].min() < zone[0] or P[:, 0].max() > zone[1] or P[:, 1].min() < zone[2]
                        or P[:, 1].max() > zone[3]):
                    continue
                if any(poly_dist(P, K) < 0.01 for K in keep):
                    continue
                if any(poly_dist(P, q.poly) < gap for q in parts):
                    continue
                parts.append(Part(cls, (float(xy[0]), float(xy[1])), float(yaw)))
                break
            else:
                ok = False
                break
        if ok:
            return parts
    raise RuntimeError("散置失敗：物件過多或間距過大")


# ---------------------------------------------------------------------- 感知
@dataclass
class Detection:
    cls: str
    xy: np.ndarray
    yaw: float  # 外接矩形一邊（寬 w 方向）的角度
    w: float
    l: float
    height: float
    roundness: float
    n_px: int
    chroma: np.ndarray
    consistent: bool = True

    @property
    def spec(self) -> ObjectSpec:
        return CLASSES[self.cls].spec

    @property
    def poly(self) -> np.ndarray:
        if self.roundness < ROUND_T:
            return circle_poly(self.xy, max(self.w, self.l) / 2)
        return rect_poly(self.xy, self.yaw, self.w / 2, self.l / 2)

    @property
    def top(self) -> float:
        return S.BENCH_TOP + self.height

    def box(self, name="det") -> Box:
        """規劃用障礙物方塊（圓柱以外接正方形近似）。"""
        h = self.height / 2
        return Box((float(self.xy[0]), float(self.xy[1]), S.BENCH_TOP + h), (self.w / 2, self.l / 2, h), name,
                   float(self.yaw))


OCCL_PX = 50  # ROI 內手臂像素超過此值即重新觀測
ROUND_T = 0.90  # 凸包/外接矩形面積比 < 0.90 判為圓形


def min_area_rect(pts) -> tuple[float, float, float, float]:
    """最小面積外接矩形（旋轉卡尺）：回傳 (角度 a, 沿 a 方向邊長 w, 垂直邊長 l, 凸包面積)。"""
    hull = ConvexHull(pts)
    H = pts[hull.vertices]
    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        a = np.arctan2(e[1], e[0])
        u = np.array([np.cos(a), np.sin(a)])
        v = np.array([-u[1], u[0]])
        pu, pv = H @ u, H @ v
        w, l = pu.max() - pu.min(), pv.max() - pv.min()
        if best is None or w * l < best[1] * best[2]:
            best = (float(a), float(w), float(l))
    return (*best, float(hull.volume))


class OverheadCamera:
    """俯視針孔相機；`detect` 只用 RGB/深度影像（不讀模擬真值）。"""

    def __init__(self, eye=CAM_EYE, target=CAM_TARGET, fov: float = 21.0, size: int = 480, near: float = 0.5,
                 far: float = 1.2, roi=ZONE, roi_margin: float = 0.02):
        self.eye, self.target, self.fov, self.size = eye, target, fov, size
        self.near, self.far = near, far
        self.roi = (roi[0] - roi_margin, roi[1] + roi_margin, roi[2] - roi_margin, roi[3] + roi_margin)
        self.bg_z = None  # 空場景高度圖（背景模型）
        self.view = pb.computeViewMatrix(list(eye), list(target), [1, 0, 0])
        self.proj = pb.computeProjectionMatrixFOV(fov, 1.0, near, far)
        PV = np.array(self.proj).reshape(4, 4).T @ np.array(self.view).reshape(4, 4).T
        self.inv = np.linalg.inv(PV)
        n = size
        u, v = np.meshgrid(np.arange(n), np.arange(n))
        self._xn = (2 * (u + 0.5) / n - 1).ravel()
        self._yn = (1 - 2 * (v + 0.5) / n).ravel()

    @property
    def px_size(self) -> float:
        """桌面處像素尺寸 [m]。"""
        return 2 * (self.eye[2] - S.BENCH_TOP) * np.tan(np.radians(self.fov / 2)) / self.size

    def capture(self, cid: int):
        img = pb.getCameraImage(self.size, self.size, self.view, self.proj, renderer=pb.ER_TINY_RENDERER,
                                lightDirection=[0.3, -0.4, 2.0], shadow=1, physicsClientId=cid)
        n = self.size
        rgb = np.reshape(img[2], (n, n, 4))[:, :, :3].astype(np.float32)
        depth = np.reshape(img[3], (n, n)).astype(np.float64)
        seg = np.reshape(img[4], (n, n))
        return rgb, depth, seg

    def deproject(self, depth) -> np.ndarray:
        X = np.c_[self._xn, self._yn, 2 * depth.ravel() - 1, np.ones(depth.size)] @ self.inv.T
        return (X[:, :3] / X[:, 3:4]).reshape(depth.shape + (3,))

    def calibrate(self, cid: int):
        """背景模型：空工作台（未放物件）的高度圖；之後前景 = 高出背景 6 mm（排除料盤銷、治具等靜態體）。"""
        self.bg_z = self.deproject(self.capture(cid)[1])[..., 2]

    def detect(self, cid: int, robot_id: int | None = None, min_px: int = 150, out: dict | None = None):
        rgb, depth, seg = self.capture(cid)
        P = self.deproject(depth)
        x0, x1, y0, y1 = self.roi
        base = S.BENCH_TOP if self.bg_z is None else self.bg_z
        fg = ((P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
              & (P[..., 2] > base + 0.006) & (P[..., 2] < 0.20))
        occluded = 0
        if robot_id is not None:  # 機器人自我遮罩（已知自身模型與姿態）：統計 ROI 內被手臂遮住的像素
            in_roi = (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
            occluded = int(np.sum(in_roi & ((seg & ((1 << 24) - 1)) == robot_id)))
        lab, n = ndimage.label(fg, structure=np.ones((3, 3)))
        dets = []
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < min_px:
                continue
            pts = P[m]
            zt = np.percentile(pts[:, 2], 98)
            top = pts[pts[:, 2] >= zt - 0.004]
            if len(top) < 10:
                continue
            a, w, l, area = min_area_rect(top[:, :2])
            ch = rgb[m].mean(axis=0)
            ch = ch / ch.sum()
            cls = min(CLASSES, key=lambda c: np.linalg.norm(CLASSES[c].chroma - ch))
            d = Detection(cls, top[:, :2].mean(axis=0), wrap_pi(a), w, l, float(zt - S.BENCH_TOP),
                          area / (w * l), int(m.sum()), ch)
            spec = CLASSES[cls].spec
            d.consistent = bool(abs(d.height - spec.height) < 0.01 and ((d.roundness < ROUND_T)
                                                                         == (spec.shape == "cylinder")))
            dets.append(d)
        if out is not None:
            out.update(rgb=rgb, fg=fg, labels=lab, occluded_px=occluded)
        return dets


def match(dets, parts) -> list[int]:
    """偵測 ↔ 真值物件（最近 xy；量測與簿記用，不參與控制）。"""
    out = []
    for d in dets:
        dist = [np.hypot(*(np.asarray(p.xy) - d.xy)) for p in parts]
        out.append(int(np.argmin(dist)) if dist and min(dist) < 0.03 else -1)
    return out


def yaw_error(det: Detection, part: Part) -> float:
    """方位誤差 [deg]，依物件對稱性取模（方塊 90°、長方柱 180°、圓柱不計 → nan）。"""
    sp = part.cls.spec
    if sp.shape == "cylinder":
        return float("nan")
    a = det.yaw
    if sp.size[0] != sp.size[1] and det.w > det.l:  # 偵測的 w 邊對應物件 y 邊 → 物件 x 軸 = a + 90°
        a = a + np.pi / 2
    sym = np.pi / 2 if abs(sp.size[0] - sp.size[1]) < 1e-9 else np.pi
    e = (a - part.yaw + sym / 2) % sym - sym / 2
    return float(np.degrees(abs(e)))


# ---------------------------------------------------------------------- 抓取規劃
@dataclass
class Grasp:
    psi: float  # 夾爪 yaw（tool_pose 再繞世界 z 轉 ψ）；兩指閉合方向 = (sin ψ, −cos ψ)
    width: float
    z: float  # TCP 高度
    clearance: float  # 夾爪部件到鄰物俯視距離（無相關鄰物 = 0.1）


def gripper_polys(xy, psi):
    """(兩指墊多邊形, 掌體多邊形)。夾爪 x 軸 = (cos ψ, sin ψ)、y 軸 = (sin ψ, −cos ψ)。"""
    gx = np.array([np.cos(psi), np.sin(psi)])
    gy = np.array([np.sin(psi), -np.cos(psi)])

    def poly(rx, ry):
        return np.array([np.asarray(xy) + a * gx + b * gy for a, b in
                         ((rx[0], ry[0]), (rx[1], ry[0]), (rx[1], ry[1]), (rx[0], ry[1]))])

    return [poly(*r) for r in PAD_RECTS], poly(*PALM_RECT)


def norm_psi(psi: float) -> float:
    """ψ 正規化到 [−π/2, π/2)（平行夾爪 180° 對稱）。"""
    return float((psi + np.pi / 2) % np.pi - np.pi / 2)


def grasp_candidates(det: Detection, others: list, check: bool = True, margin: float = GRIP_MARGIN,
                     psi_pref: float = 0.0) -> list[Grasp]:
    """others：[(俯視多邊形, 頂面高度)]（另自動加入靜態體 `static_footprints`）。
    回傳可行抓取（clearance 大者優先，其次 ψ 接近 psi_pref）。
    check=False：不檢查鄰物（消融比較用），只要求寬度可夾。"""
    z = S.BENCH_TOP + det.height / 2 + grasp_offset(det.height)
    if det.roundness < ROUND_T:
        cands = [(k * np.pi / 8, max(det.w, det.l)) for k in range(8)]
    else:  # 閉合方向沿 w 邊（角度 a）→ ψ = a + 90°，寬 = w；沿 l 邊 → ψ = a，寬 = l
        cands = [(det.yaw + np.pi / 2, det.w), (det.yaw, det.l)]
    out = []
    for psi, width in cands:
        if width > OPEN_W - 2 * margin:
            continue
        pads, palm = gripper_polys(det.xy, psi)
        clear = 0.1
        self_ok = all(poly_dist(p, det.poly) >= margin for p in pads)
        for poly, top in list(others) + static_footprints():
            if top > z + PAD_BOTTOM - margin:  # 垂直方向同樣保留餘裕（掌體下緣恰與鄰物頂面齊平也算干涉）
                clear = min(clear, *(poly_dist(p, poly) for p in pads))
            if top > z + PALM_BOTTOM - margin:
                clear = min(clear, poly_dist(palm, poly))
        if check and (clear < margin or not self_ok):
            continue
        out.append(Grasp(norm_psi(psi), float(width), float(z), float(clear)))
    out.sort(key=lambda g: (-min(g.clearance, 0.03), abs(wrap_pi(g.psi - psi_pref))))
    return out


def blockers(dets, i, check: bool = True) -> bool:
    """dets[i] 在其餘 dets 存在時是否可抓。"""
    others = [(d.poly, d.top) for j, d in enumerate(dets) if j != i]
    return bool(grasp_candidates(dets[i], others, check))


# ---------------------------------------------------------------------- 運動學輔助
_R = None


def _robot():
    global _R
    if _R is None:
        from .robot import SixAxisRobot

        _R = SixAxisRobot()
    return _R


def pose(xyz, psi: float = 0.0) -> np.ndarray:
    """工具朝下、繞世界 z 轉 ψ 的法蘭位姿，使 TCP 位於 xyz。"""
    T = np.eye(4)
    T[:3, :3] = rotz(psi) @ R_DOWN
    T[:3, 3] = np.asarray(xyz, float) - T[:3, :3] @ T_TCP[:3, 3]
    return T


def ik_near(xyz, psi, q_near, sym: bool = True, cc: CollisionChecker | None = None):
    """ψ（與 ψ+π，若夾持對稱）之所有 IK 解中，關節角包到 [−π, π]、（cc 給定時）無碰撞且最接近 q_near 者。"""
    r = _robot()
    best, bd = None, np.inf
    for p in ([psi, psi + np.pi] if sym else [psi]):
        for q in r.ik_all(pose(xyz, p), q_near):
            q = np.asarray(wrap_pi(np.asarray(q)), float)
            if np.linalg.norm(r.fk(q)[:3, 3] - pose(xyz, p)[:3, 3]) > 1e-6 or q[1] > 0:
                continue
            if cc is not None and not cc.free(q):
                continue
            d = float(np.sum(np.abs(q - q_near)))
            if d < bd:
                best, bd = q, d
    return best


def home_q():
    return ik_near((0.22, -0.22, 0.42), 0.0, Q_REF)


def move_time(qa, qb) -> float:
    """與 T007 執行器相同速度上限（qd_max·NOMINAL）的五次多項式最短時間。"""
    dq = np.asarray(qb) - np.asarray(qa)
    if np.max(np.abs(dq)) < 1e-9:
        return 0.0
    return float(min_duration(dq, _robot().qd_max * NOMINAL, None, "quintic"))


# ---------------------------------------------------------------------- 序列規劃
T_FIXED = 3.2  # 每件固定時間估計（下降/夾/抬升/下降/放/抬升）：以 T101 量測的直線段時間為量級，比較序列時相消


def plan_sequence(dets, q_start, method: str = "optimal", check: bool = True) -> tuple[list[int], float]:
    """回傳（順序, 估計搬運時間）。method：'scan'（掃描順序：x 再 y）、'greedy'、'optimal'（Held–Karp）。
    每類物件依序填該類放置格；被鄰物擋住（無可行抓取）的物件不可在其鄰物之前。
    若不存在取完全部的序列（死結），回傳可取最多件的部分順序，估計時間 = ∞。"""
    n = len(dets)
    q_obj, slot_q, gcache = {}, {}, {}
    polys = [(d.poly, d.top) for d in dets]

    def qo(i, rem):
        if (i, rem) not in gcache:
            g = grasp_candidates(dets[i], [polys[j] for j in rem if j != i], check)
            gcache[(i, rem)] = g[0].psi if g else None
        psi = gcache[(i, rem)]
        if psi is None:
            return None
        key = (i, round(psi, 6))
        if key not in q_obj:
            q_obj[key] = ik_near((*dets[i].xy, Z_CLEAR), psi, Q_REF)
        return q_obj[key]

    def qs(cls, k):
        if (cls, k) not in slot_q:
            sl = dest_slots(CLASSES[cls].dest)
            slot_q[(cls, k)] = ik_near((*sl[k % len(sl)], Z_CLEAR), place_yaw(CLASSES[cls]), Q_REF)
        return slot_q[(cls, k)]

    def placed_count(rem, cls):
        return sum(1 for j in range(n) if j not in rem and dets[j].cls == cls)

    def step_cost(q_prev, i, rem):
        q1 = qo(i, rem)
        if q1 is None:
            return None, None
        k = placed_count(rem, dets[i].cls)
        q2 = qs(dets[i].cls, k)
        return move_time(q_prev, q1) + move_time(q1, q2) + T_FIXED, q2

    full = frozenset(range(n))
    if method in ("scan", "greedy"):
        rem, order, q, tot = set(full), [], q_start, 0.0
        while rem:
            cand = []
            for i in sorted(rem, key=lambda j: (round(dets[j].xy[0], 3), dets[j].xy[1])):
                c, q2 = step_cost(q, i, frozenset(rem))
                if c is not None:
                    cand.append((c, i, q2))
                    if method == "scan":
                        break
            if not cand:
                return order, np.inf
            c, i, q2 = min(cand) if method == "greedy" else cand[0]
            order.append(i)
            rem.discard(i)
            tot += c
            q = q2
        return order, tot
    # Held–Karp：f[S][i] = 最小成本，S = 已取集合，i = 最後一件
    f: dict = {}
    for i in range(n):
        c, q2 = step_cost(q_start, i, full)
        if c is not None:
            f[(frozenset([i]), i)] = (c, None)
    for size in range(2, n + 1):
        for Sset in itertools.combinations(range(n), size):
            Sf = frozenset(Sset)
            for i in Sset:
                prev = Sf - {i}
                rem = full - prev
                best = None
                for j in prev:
                    if (prev, j) not in f:
                        continue
                    q_prev = qs(dets[j].cls, placed_count(full - prev, dets[j].cls) - 1)
                    c, _ = step_cost(q_prev, i, rem)
                    if c is None:
                        continue
                    v = f[(prev, j)][0] + c
                    if best is None or v < best[0]:
                        best = (v, j)
                if best:
                    f[(Sf, i)] = best
    if not f:
        return [], np.inf
    # 完整序列不存在（剩餘物件互相阻擋＝死結）時，退而求能取走最多件的部分序列，估計時間記為 ∞
    m = max(len(k[0]) for k in f)
    Sf = min((k for k in f if len(k[0]) == m), key=lambda k: f[k][0])[0]
    ends = [(f[(Sf, i)][0], i) for i in range(n) if (Sf, i) in f]
    tot, i = min(ends)
    if m < n:
        tot = np.inf
    order = [i]
    while f[(Sf, i)][1] is not None:
        j = f[(Sf, i)][1]
        Sf = Sf - {i}
        order.append(j)
        i = j
    return order[::-1], float(tot)


# ---------------------------------------------------------------------- 分類工作站（物理執行）
@dataclass
class SortConfig:
    vision: bool = True  # False：使用真值位姿（消融）
    grasp_check: bool = True  # False：不檢查夾爪 vs 鄰物（消融）
    fixed_yaw: bool = False  # True：ψ 固定 0（T102 式，不用視覺方位；消融）
    sequence: str = "optimal"
    plan: bool = True  # False：一律關節直線 PTP（不避障）
    max_attempts: int = 3
    rrt_time: float = 30.0
    force: float = 40.0


@dataclass
class SortCell:
    parts: list
    cfg: SortConfig = field(default_factory=SortConfig)
    render: bool = False
    extra: list = field(default_factory=list)  # 額外靜態障礙物（Box），例如光柵立柱
    frame_cb: object = None
    cb_every: int = 25

    def __post_init__(self):
        self._td = tempfile.TemporaryDirectory()
        self.s = GraspScene(CUBE, workcell=True, render=self.render,
                            urdf_path=Path(self._td.name) / "arm_gripper.urdf")
        s = self.s
        pb.removeBody(s.obj, physicsClientId=s.cid)
        s.wc.parts.clear()
        for b in sort_boxes():
            col = (0.35, 0.38, 0.42, 1) if ("stand" in b.name or "mast" in b.name or "boom" in b.name) else (
                (0.25, 0.45, 0.8, 1) if "binA" in b.name else (0.25, 0.65, 0.35, 1))
            s.wc._box(b.name, b.center, b.half, col)
        for b in self.extra:
            s.wc._box(f"obs_{b.name}", b.center, b.half, (0.85, 0.55, 0.15, 1))
        q0 = home_q()
        s.sim.set_state(q0)
        s._hold(q0)
        self.cam = OverheadCamera()
        self.cam.calibrate(s.cid)
        for p in self.parts:
            sp = p.cls.spec
            z = S.BENCH_TOP + sp.height / 2 + 1e-3
            p.body = s.g.spawn(sp, [p.xy[0], p.xy[1], z], [0, 0, np.sin(p.yaw / 2), np.cos(p.yaw / 2)])
            s.wc.add_part(p.body)
        s.wait(0.3)
        self.ex = Executor(s, frame_cb=self.frame_cb, cb_every=self.cb_every)
        static = [b for b in workcell_boxes() if b.name != "cube"] + sort_boxes() + list(self.extra)
        self.cc = CollisionChecker(static)
        self.used = {"tray": set(), "binA": set(), "binB": set()}
        self.log: list = []
        self.vision_log: list = []
        self.last_view: dict = {}
        self.motion_log: list = []

    def close(self):
        self.cc.close()
        self.s.close()
        self._td.cleanup()

    # --- 感知
    def perceive(self):
        s = self.s
        remaining = [p for p in self.parts if self._on_bench(p)]
        if not self.cfg.vision:
            dets = []
            for p in remaining:
                pos, quat = pb.getBasePositionAndOrientation(p.body, physicsClientId=s.cid)
                yaw = _yaw_of(quat)
                sp = p.cls.spec
                w, l = (sp.size[0], sp.size[1]) if sp.shape == "box" else (2 * sp.size[0],) * 2
                dets.append(Detection(p.cls.name, np.array(pos[:2]), yaw, w, l, sp.height,
                                      1.0 if sp.shape == "box" else 0.785, 0, p.cls.chroma))
            return dets, remaining
        info: dict = {}
        dets = self.cam.detect(s.cid, s.sim.id, out=info)
        reobs = False
        if info["occluded_px"] > OCCL_PX:  # 手臂遮住取料區 → 退到相機視野外的觀測姿態再拍
            reobs = True
            occl0 = info["occluded_px"]
            self.goto(home_q(), "observe")
            dets = self.cam.detect(s.cid, s.sim.id, out=info)
        self.last_view = {"rgb": info["rgb"], "fg": info["fg"], "dets": dets, "t": s.t}
        idx = match(dets, remaining)
        rec = {"t": s.t, "n_true": len(remaining), "n_det": len(dets), "occluded_px": info["occluded_px"],
               "reobserve": reobs, "occluded_px0": occl0 if reobs else info["occluded_px"], "items": []}
        for d, j in zip(dets, idx, strict=True):
            it = {"cls": d.cls, "consistent": d.consistent, "matched": j >= 0}
            if j >= 0:
                p = remaining[j]
                pos = pb.getBasePositionAndOrientation(p.body, physicsClientId=s.cid)[0]
                it.update(true_cls=p.cls.name, pos_err_mm=float(np.hypot(*(np.array(pos[:2]) - d.xy)) * 1e3),
                          yaw_err_deg=yaw_error(d, Part(p.cls, pos[:2], self._yaw(p))),
                          h_err_mm=float((d.height - p.cls.spec.height) * 1e3))
            rec["items"].append(it)
        self.vision_log.append(rec)
        return dets, remaining

    def _yaw(self, p) -> float:
        quat = pb.getBasePositionAndOrientation(p.body, physicsClientId=self.s.cid)[1]
        return _yaw_of(quat)

    def _on_bench(self, p) -> bool:
        pos = pb.getBasePositionAndOrientation(p.body, physicsClientId=self.s.cid)[0]
        return bool(ZONE[0] - 0.05 < pos[0] < ZONE[1] + 0.05 and ZONE[2] - 0.05 < pos[1] < ZONE[3] + 0.05
                    and pos[2] < S.BENCH_TOP + 0.1)

    # --- 運動
    def goto(self, q_goal, tag: str) -> bool:
        """目前姿態 → q_goal：保證式直線檢查，不通則 RRT*（第一解）＋捷徑；以 move_j(q) 逐段執行。"""
        q0 = self.s.sim.get_state()[0]
        q0w = np.asarray(wrap_pi(q0), float)
        if np.max(np.abs(q0w - q0)) > 1e-9:  # 起點角度包回 [−π, π]（同一姿態）
            q_goal = q_goal + (q0 - q0w)
        rec = {"tag": tag, "planned": False, "t_plan": 0.0, "ok": True}
        path = [q0, q_goal]
        if self.cfg.plan:
            cc = self.cc
            if not cc.edge_free(q0w, q_goal - (q0 - q0w)):
                t_plan = 0.0
                for k in range(2):  # 第一次未在時限內找到解 → 換亂數種子重試一次
                    rrt = RRTStar(cc, seed=len(self.motion_log) + 1000 * k)
                    res = rrt.plan(q0w, [q_goal - (q0 - q0w)], max_iter=5000, time_limit=self.cfg.rrt_time,
                                   stop_at_first=True)
                    t_plan += res.t_plan
                    if res.success:
                        break
                rec.update(planned=True, t_plan=t_plan, rrt_ok=res.success, rrt_tries=k + 1)
                if not res.success:
                    rec["ok"] = False
                    self.motion_log.append(rec)
                    return False
                sc = shortcut(cc, res.path, 100, seed=0)
                rec.update(raw_len=path_length(res.path), sc_len=path_length(sc), n_seg=len(sc) - 1)
                path = [q + (q0 - q0w) for q in sc]
        prog = [{"cmd": "move_j", "q": [float(x) for x in q], "id": f"{tag}{k}"}
                for k, q in enumerate(path[1:])]
        rep = self.ex.run(prog)
        rec["ok"] = rep.ok
        rec["error"] = None if rep.ok else rep.error.code
        rec["results"] = rep.results
        self.motion_log.append(rec)
        return rep.ok

    def stroke_clearance(self, xy, psi, z_lo, n: int = 6) -> float:
        """垂直升降行程（Z_CLEAR → z_lo）上 n 個等距 TCP 高度的 IK 姿態，
        手臂（不含夾持物）對障礙物的最小距離 [m]。move_l 不經規劃器，故以取樣檢查（非保證式）挑選放置格；
        夾持物本身在箱內/格內與壁的間隙由幾何設計保證。"""
        q = np.asarray(wrap_pi(self.s.sim.get_state()[0]), float)
        held, self.cc.held = self.cc.held, None
        dmin = np.inf
        try:
            for z in np.linspace(Z_CLEAR, z_lo, n):
                q = ik_near((*xy, z), psi, q)
                if q is None:
                    return -1.0
                dmin = min(dmin, self.cc.distances(q)[0])
        finally:
            self.cc.held = held
        return float(dmin)

    def choose_slot(self, dest, psi, z_lo, d_min: float = 0.003) -> tuple[int, float]:
        """依序找第一個未用、且垂直放置行程手臂間隙 > d_min 的放置格；都不行則取間隙最大者。"""
        slots = dest_slots(dest)
        order = [k for k in range(len(slots)) if k not in self.used[dest]] or list(range(len(slots)))
        best = (-np.inf, order[0])
        for k in order:
            c = self.stroke_clearance(slots[k], psi, z_lo)
            if c > d_min:
                return k, c
            best = max(best, (c, k))
        return best[1], best[0]

    def lin(self, z, tag) -> bool:
        xy = self.ex.tcp_of(self.s.sim.get_state()[0])[:2].tolist()
        rep = self.ex.run([{"cmd": "move_l", "tcp": [*xy, float(z)], "id": tag}])
        self.motion_log.append({"tag": tag, "planned": False, "ok": rep.ok, "results": rep.results,
                                "error": None if rep.ok else rep.error.code})
        return rep.ok

    # --- 主流程
    def run(self, max_cycles: int = 20) -> dict:
        s, cfg = self.s, self.cfg
        t0 = s.t
        attempts = {}
        failures = 0
        for _cyc in range(max_cycles):
            dets, remaining = self.perceive()
            if not dets:
                break
            q_now = np.asarray(wrap_pi(s.sim.get_state()[0]), float)
            order, est = plan_sequence(dets, q_now, cfg.sequence, cfg.grasp_check)
            if not order:
                self.log.append({"event": "NO_FEASIBLE_GRASP", "t": s.t})
                break
            i = order[0]
            d = dets[i]
            j = match([d], remaining)[0]
            part = remaining[j] if j >= 0 else None
            key = id(part)
            attempts[key] = attempts.get(key, 0) + 1
            others = [(o.poly, o.top) for k, o in enumerate(dets) if k != i]
            gr = grasp_candidates(d, others, cfg.grasp_check)
            g = gr[0]
            psi = 0.0 if cfg.fixed_yaw else g.psi
            entry = {"cycle": _cyc, "cls": d.cls, "true_cls": part.cls.name if part else None, "psi": psi,
                     "width_mm": g.width * 1e3, "clear_mm": g.clearance * 1e3, "t_start": s.t, "est_seq": est,
                     "order": order, "attempt": attempts[key]}
            # 接近
            self.cc.set_movable([o.box(f"d{k}") for k, o in enumerate(dets)])
            q_pre = ik_near((*d.xy, Z_CLEAR), psi, q_now, cc=self.cc)
            if q_pre is None:
                entry["fail_reason"] = "NO_FREE_IK_PICK"
            ok = q_pre is not None and self.goto(q_pre, "approach")
            if ok:
                ok = self.lin(g.z, "descend")
            grip_ok = False
            if ok and part is not None:
                s.g.obj = part.body
                rep = self.ex.run([{"cmd": "grip", "force": cfg.force, "id": "grip"}])
                grip_ok = rep.ok
                entry["grip"] = rep.results[-1]
            if ok:
                ok = self.lin(Z_CLEAR, "lift")
            inhand = {}
            if ok and grip_ok:
                pos, quat = pb.getBasePositionAndOrientation(part.body, physicsClientId=s.cid)
                tcp = self.ex.tcp_of(s.sim.get_state()[0])
                dd = np.array(pos) - tcp
                R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
                tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
                inhand = {"dx_mm": dd[0] * 1e3, "dy_mm": dd[1] * 1e3, "dz_mm": dd[2] * 1e3, "tilt_deg": tilt}
                dz0 = -grasp_offset(d.height)  # 物件中心在 TCP 下方 grasp_offset
                grip_ok = bool(np.hypot(dd[0], dd[1]) <= 0.003 and abs(dd[2] - dz0) <= 0.004 and tilt <= 5.0)
            entry["inhand"] = inhand
            # 近鄰物件被擾動量
            entry["disturb_mm"] = self._disturbance(remaining, part)
            if not (ok and grip_ok):
                entry["result"] = "PICK_FAIL"
                failures += 1
                s.g.open()
                s.wait(0.3)
                self.ex.holding = False
                self.ex.t = s.t
                self.lin(Z_CLEAR, "recover")
                entry["t_end"] = s.t
                self.log.append(entry)
                if attempts[key] >= cfg.max_attempts:
                    entry["result"] = "GIVE_UP"
                    self.log.append({"event": "GIVE_UP", "t": s.t})
                    break
                continue
            # 搬運：附加夾持工件
            dest = part.cls.dest
            psi_pl = place_yaw(part.cls)
            self.cc.set_movable([o.box(f"d{kk}") for kk, o in enumerate(dets) if kk != i])
            k, stroke = self.choose_slot(dest, psi_pl, place_height(part.cls, d.height))
            slot = dest_slots(dest)[k]
            entry["stroke_clear_mm"] = stroke * 1e3
            R_tcp = rotz(psi) @ R_DOWN
            pos, quat = pb.getBasePositionAndOrientation(part.body, physicsClientId=s.cid)
            R_obj = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
            T_to = np.eye(4)
            T_to[:3, :3] = R_tcp.T @ R_obj
            T_to[:3, 3] = R_tcp.T @ (np.array(pos) - self.ex.tcp_of(s.sim.get_state()[0]))
            sp = part.cls.spec
            half = [x / 2 for x in sp.size] if sp.shape == "box" else [sp.size[0], sp.size[0], sp.size[1] / 2]
            self.cc.attach(half, T_to)
            q_now = s.sim.get_state()[0]
            q_place = ik_near((*slot, Z_CLEAR), psi_pl, np.asarray(wrap_pi(q_now)), sym=True, cc=self.cc)
            if q_place is None:
                entry["fail_reason"] = "NO_FREE_IK_PLACE"
            ok = q_place is not None and self.goto(q_place, "transport")
            self.cc.detach()
            if ok:
                ok = self.lin(place_height(part.cls, d.height), "lower")
            if ok:
                self.ex.run([{"cmd": "release", "id": "release"}])
                ok = self.lin(Z_CLEAR, "retreat")
            self.used[dest].add(k)
            entry.update(result="PLACED" if ok else "MOTION_FAIL", slot=list(slot), dest=dest, t_end=s.t,
                         psi_place=psi_pl)
            self.log.append(entry)
            if not ok:
                break
        return self.metrics(s.t - t0, failures)

    def _disturbance(self, remaining, picked) -> float:
        """其餘物件相對本輪開始時的最大位移 [mm]（夾爪碰到鄰物時 > 0）。"""
        mx = 0.0
        for p in remaining:
            if p is picked:
                continue
            pos = pb.getBasePositionAndOrientation(p.body, physicsClientId=self.s.cid)[0]
            mx = max(mx, float(np.hypot(pos[0] - p.xy[0], pos[1] - p.xy[1]) * 1e3))
        return mx

    def final_state(self) -> list[dict]:
        out = []
        cid = self.s.cid
        for p in self.parts:
            pos, quat = pb.getBasePositionAndOrientation(p.body, physicsClientId=cid)
            R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
            tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            where = locate(pos)
            rec = {"cls": p.cls.name, "dest": p.cls.dest, "where": where, "pos": list(pos), "tilt_deg": tilt,
                   "correct": where == p.cls.dest}
            if where == "tray":
                c = min(tray_slots(), key=lambda q: np.hypot(q[0] - pos[0], q[1] - pos[1]))
                yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                rec.update(pocket_err_mm=float(np.hypot(c[0] - pos[0], c[1] - pos[1]) * 1e3),
                           z_err_mm=float((pos[2] - Z_POCKET) * 1e3), yaw_deg=abs((yaw + 45) % 90 - 45))
                rec["seated"] = bool(rec["pocket_err_mm"] <= 4.0 and abs(rec["z_err_mm"]) <= 1.5
                                     and tilt <= 2.0)
            out.append(rec)
        return out

    def metrics(self, T: float, failures: int) -> dict:
        fs = self.final_state()
        picks = [e for e in self.log if "cycle" in e]
        mv = [r for m in self.motion_log for r in m.get("results", []) if r.get("status") == "ok"
              and r["cmd"].startswith("move")]
        return {
            "n_parts": len(self.parts), "n_correct": sum(f["correct"] for f in fs),
            "n_seated": sum(f.get("seated", False) for f in fs if f["dest"] == "tray"),
            "n_tray": sum(1 for f in fs if f["dest"] == "tray"),
            "success": all(f["correct"] for f in fs), "cycle_s": T, "pick_fail": failures,
            "n_picks": len(picks), "n_planned": sum(1 for m in self.motion_log if m.get("planned")),
            "t_plan": sum(m.get("t_plan", 0.0) for m in self.motion_log),
            "torque_util": max((r.get("torque_ratio", 0.0) for r in mv), default=0.0),
            "vel_util": max((r.get("vel_ratio", 0.0) for r in mv), default=0.0),
            "errors": [m["error"] for m in self.motion_log if m.get("error")]
            + [e["fail_reason"] for e in picks if e.get("fail_reason")],
            "final": fs, "log": picks, "vision": self.vision_log,
            "max_disturb_mm": max((e.get("disturb_mm", 0.0) for e in picks), default=0.0),
            "n_reobserve": sum(v.get("reobserve", False) for v in self.vision_log),
        }


def _yaw_of(quat) -> float:
    """四元數 [x,y,z,w] 的 yaw（ZYX 尤拉角的 z 分量）。"""
    x, y, z, w = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))


def locate(pos) -> str:
    """物件最終所在：'tray'（料盤格內）、'binA'/'binB'（箱內）、'bench'、'floor'、'other'。"""
    x, y, z = pos
    for nm, (cx, cy) in BIN_C.items():
        if abs(x - cx) < BIN_IN_X and abs(y - cy) < BIN_IN_Y and BIN_FLOOR < z < BIN_WALL_TOP + 0.05:
            return nm
    h = S.TRAY_PITCH / 2 + S.POCKET / 2
    if abs(x - S.TRAY_C[0]) < h and abs(y - S.TRAY_C[1]) < h and z < S.BENCH_TOP + 0.06:
        return "tray"
    if z < 0.05:
        return "floor"
    if abs(x - S.BENCH_X) < S.BENCH_HX and abs(y) < S.BENCH_HY:
        return "bench"
    return "other"


__all__ = ["BLOCK_S", "CLASSES", "CYL_S", "Detection", "Grasp", "OverheadCamera", "Part", "PartClass",
           "SortCell", "SortConfig", "S_MAX", "ZONE", "bin_slots", "blockers", "dest_slots", "footprint",
           "grasp_candidates", "gripper_polys", "home_q", "ik_near", "locate", "match", "min_area_rect",
           "move_time", "place_yaw", "plan_sequence", "poly_dist", "pose", "scatter", "sort_boxes",
           "static_footprints",
           "tray_slots", "yaw_error"]
