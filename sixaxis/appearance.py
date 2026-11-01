"""手臂外觀（視覺/碰撞幾何），參考 UR 系列外型。

只影響 URDF 的 <visual>/<collision>，**不影響**質量、質心、慣性與運動學
（PyBullet 以 URDF_USE_INERTIA_FROM_FILE 載入，慣性取自 robot.py 的模型）。

幾何依據
--------
UR5 實機在平行的關節 2、3、4 軸方向上有側向偏置：肩偏置 0.13585 m、肘偏置 -0.1197 m、
腕 1 偏置 0.093 m，三者相加即標準 DH 的 d4 = 0.10915 m。沿「平行關節軸」平移連桿外型
不改變運動學，因此可把上臂畫在 z2 = 0.13585、前臂畫在 z3 = 0.01615 的平面上，
外觀與真實 UR 一致而 DH 不變。

所有座標皆在各連桿座標系 L_i（z 軸 = 關節 i 軸，見 robot.py）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot import SixAxisRobot

SHOULDER_OFFSET = 0.13585
ELBOW_OFFSET = -0.1197
UPPER_ARM_Z = SHOULDER_OFFSET  # 上臂所在平面（沿 z2）
FOREARM_Z = SHOULDER_OFFSET + ELBOW_OFFSET  # 前臂所在平面（沿 z3）

# 材質（RGBA）
MATERIALS = {
    "housing": (0.86, 0.87, 0.89, 1.0),  # 關節外殼：淺灰
    "tube": (0.72, 0.74, 0.77, 1.0),  # 臂管：鋁銀
    "cap": (0.16, 0.42, 0.78, 1.0),  # 端蓋：UR 藍
    "ring": (0.22, 0.23, 0.26, 1.0),  # 密封環：深灰
    "base": (0.28, 0.29, 0.32, 1.0),  # 底座
    "plate": (0.18, 0.18, 0.20, 1.0),  # 安裝板
    "flange": (0.62, 0.64, 0.67, 1.0),  # 工具法蘭
    "tcp": (0.95, 0.55, 0.10, 1.0),  # 工具中心標記
}

# 關節外殼 (中心沿關節軸的位置 c, 長度 h, 半徑 r)，單位 m，位於子連桿 L_j 座標
HOUSINGS = {
    1: (0.060, 0.120, 0.060),  # 底座旋轉柱（z 0~0.12）
    2: (0.068, 0.170, 0.060),  # 肩：涵蓋 z2=0（立柱）到上臂平面 0.136
    3: (0.076, 0.160, 0.050),  # 肘：涵蓋上臂平面與前臂平面 0.016
    4: (0.052, 0.112, 0.042),  # 腕 1：從前臂平面到腕 2 軸 (z4=d4)
    5: (0.0535, 0.177, 0.042),  # 腕 2：z5 -0.035~0.142，包住腕 3 根部（腕 3 軸在 z5=d5）
    6: (0.0275, 0.085, 0.038),  # 腕 3（法蘭端，z6 -0.015~0.07，法蘭面 z6=d6）
}


@dataclass
class Prim:
    kind: str  # "cylinder" | "box" | "sphere"
    center: np.ndarray
    axis: np.ndarray  # 圓柱軸方向（單位向量）
    size: tuple  # cylinder:(length, radius)  box:(x,y,z)  sphere:(radius,)
    material: str
    collision: bool = False
    mesh: str | None = None  # 視覺網格種類（meshgen.PROFILES），None 則用 URDF 基本形狀


def _cyl(center, axis, length, radius, material, collision=False, mesh=None):
    return Prim(
        "cylinder",
        np.asarray(center, float),
        np.asarray(axis, float),
        (length, radius),
        material,
        collision,
        mesh,
    )


def _inside_housing(p, j) -> bool:
    """點 p（L_j 座標）是否落在關節 j 外殼內（含 3 mm 餘裕）。"""
    c, h, r = HOUSINGS[j]
    return bool(np.hypot(p[0], p[1]) < r + 0.003 and abs(p[2] - c) < h / 2 + 0.003)


def exposed_ends(robot: SixAxisRobot, j: int) -> tuple[bool, bool]:
    """關節 j 外殼兩端（-z, +z）是否外露，用來決定是否加藍色端蓋。

    端面中心在關節 j 軸上，相鄰關節繞自己的軸旋轉時，此點相對於相鄰外殼的
    「離軸距離」與「軸向座標」不變，因此判斷與姿態無關。
    """
    c, h, _ = HOUSINGS[j]
    i = j - 1
    out = []
    for s in (-1.0, 1.0):
        p = np.array([0.0, 0.0, c + s * h / 2])
        hidden = False
        if j == 1:
            hidden = s < 0  # 底部埋在底座內
        else:  # 父連桿外殼（L_{j-1}）：p_parent = RF_i p + pF_i
            hidden |= _inside_housing(robot.RF[i] @ p + robot.pF[i], j - 1)
        if j < 6:  # 子連桿外殼（L_{j+1}）
            hidden |= _inside_housing(robot.RF[i + 1].T @ (p - robot.pF[i + 1]), j + 1)
        out.append(not hidden)
    return out[0], out[1]


def _housing(c, h, r, ends=(True, True)):
    """沿 z 的關節外殼：本體 + 外露端的藍色端蓋 + 深灰密封環。"""
    z = np.array([0.0, 0.0, 1.0])
    cap_t = 0.010
    out = [_cyl([0, 0, c], z, h, r, "housing", collision=True, mesh="housing")]
    for s, show_cap in zip((-1.0, 1.0), ends):
        if show_cap:
            out.append(
                _cyl([0, 0, c + s * (h / 2 + cap_t / 2 - 0.002)], z, cap_t, r * 0.86, "cap", mesh="cap")
            )
        out.append(_cyl([0, 0, c + s * (h / 2 - 0.026)], z, 0.010, r * 1.02, "ring", mesh="ring"))
    return out


def _tube(p0, p1, r, material="tube"):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    return _cyl(
        (p0 + p1) / 2,
        v / np.linalg.norm(v),
        float(np.linalg.norm(v)),
        r,
        material,
        collision=True,
        mesh="tube",
    )


def base_prims() -> list[Prim]:
    z = np.array([0.0, 0.0, 1.0])
    return [
        Prim("box", np.array([0, 0, 0.005]), z, (0.22, 0.22, 0.010), "plate", True),
        # 底座頂面須低於肩外殼底緣 (d1 - r2 = 0.0292 m)，避免與 link2 幾何重疊
        _cyl([0, 0, 0.0185], z, 0.017, 0.078, "base", collision=True, mesh="housing"),
        _cyl([0, 0, 0.0255], z, 0.003, 0.080, "ring", mesh="ring"),
    ]


def link_prims(robot: SixAxisRobot, i: int) -> list[Prim]:
    """第 i 個連桿（0-based，對應 link{i+1}）的外觀幾何。"""
    j = i + 1
    prims = _housing(*HOUSINGS[j], ends=exposed_ends(robot, j))
    a2, a3 = robot.a[1], robot.a[2]
    if j == 2:  # 上臂：在 z2 = 肩偏置平面，從肩軸到肘軸
        prims.append(_tube([0, 0, UPPER_ARM_Z], [a2, 0, UPPER_ARM_Z], 0.044))
        prims.append(_cyl([a2 * 0.5, 0, UPPER_ARM_Z], [1, 0, 0], 0.012, 0.047, "ring", mesh="ring"))
    elif j == 3:  # 前臂：在 z3 = 肩偏置 + 肘偏置平面
        prims.append(_tube([0, 0, FOREARM_Z], [a3, 0, FOREARM_Z], 0.036))
        prims.append(_cyl([a3 * 0.5, 0, FOREARM_Z], [1, 0, 0], 0.012, 0.039, "ring", mesh="ring"))
    elif j == 6:  # 工具法蘭（末端面 = 法蘭座標原點 z6 = d6）
        d6 = robot.d[5]
        z = np.array([0.0, 0.0, 1.0])
        prims.append(_cyl([0, 0, d6 - 0.006], z, 0.012, 0.032, "flange", collision=True, mesh="ring"))
        prims.append(_cyl([0, 0, d6 - 0.0005], z, 0.001, 0.010, "tcp"))
    return prims
