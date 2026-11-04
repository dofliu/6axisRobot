"""工作站場景（T006）：工作台、輸送帶（含擋料板/護欄）、料盤（2×2 角柱定位格）、固定治具、碰撞群組。

座標：世界系 z 向上，手臂基座在原點；工作台頂面 z = BENCH_TOP（= graspsim.PED_TOP，與 T005 抓取高度一致）。

配置（單位 m）
* 工作台：x∈[0.28,0.62]、y∈[-0.29,0.29]，頂面 0.10；取料點 (0.45, 0)。
* 料盤：中心 (0.45,-0.17)，2×2 格、間距 0.09；每格由 ±x 兩片端板 + ±y 側 4 根小銷定位（銷在 x=±22 mm，
  兩指墊從 y 邊中間伸入不干涉）。
* 治具：後擋板（+x）+ 側定位銷（+y）構成 2D 基準角，標稱零件中心 (0.45, 0.14)。
* 輸送帶：x∈[0.15,0.55]、y∈[0.35,0.55]，帶面 0.10，沿 +x 前進；末端擋料板，零件停在 (0.495, 0.45)。

輸送帶模型（PyBullet 無表面速度接觸）：帶面摩擦設為 0，改以「庫侖限幅的速度追隨」施力——
對每個與帶面接觸的零件施加 F = m·clip((v_belt − v_x)/dt, ±a_max)，a_max = μ_belt·g。
物理意義：理想摩擦驅動帶，零件加速度受摩擦極限 μg 限制（見 Siciliano 等；輸送帶為標準摩擦傳動模型）。
碰撞群組：以 setCollisionFilterGroupMask，規則 (gA & mB) && (gB & mA) 才碰撞，見 GROUPS / MASKS。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as pb

from .robot import G

BENCH_TOP = 0.10
BENCH_X, BENCH_HX, BENCH_HY = 0.45, 0.17, 0.29
PICK_XY = (0.45, 0.0)

TRAY_C = (0.45, -0.17)
TRAY_PITCH = 0.09
POCKET = 0.058  # 格內淨寬（方塊 50 mm → 單邊間隙 4 mm）
POST, POST_H = 0.012, 0.020  # 擋牆/側銷厚度與高度
PIN_X = 0.022  # y 向側銷 x 位置（距格中心；指墊 x 範圍 ±10 mm，銷內緣 ≥ 17 mm，餘裕 7 mm）
TRAY_PLATE_T = 0.004

FIX_NOM = (0.45, 0.14)  # 治具標稱零件中心
FIX_WALL_T, FIX_H = 0.010, 0.030

BELT_X0, BELT_X1, BELT_Y0, BELT_Y1 = 0.15, 0.55, 0.35, 0.55
BELT_T = 0.010
BELT_SPEED = 0.10  # [m/s]，+x
BELT_MU = 0.5
STOP_X = 0.52  # 擋料板內面 x
BELT_PICK_XY = (0.495, 0.45)  # 方塊（50 mm）抵擋料板時的中心

ARM, STATIC, BELT, PART = 1, 2, 4, 8
GROUPS = {"arm": ARM, "static": STATIC, "belt": BELT, "part": PART}
MASKS = {"arm": STATIC | BELT | PART, "static": ARM | PART, "belt": ARM | PART,
         "part": ARM | STATIC | BELT | PART}


def pocket_centers() -> list[tuple[float, float]]:
    """料盤 4 格中心（先 y 再 x 排序），z 取工作台頂面。"""
    h = TRAY_PITCH / 2
    return [(TRAY_C[0] + sx * h, TRAY_C[1] + sy * h) for sx in (-1, 1) for sy in (-1, 1)]


@dataclass
class Workcell:
    """在既有 PyBullet client 中建立工作站。arm_id 為已載入的手臂（+夾爪）multibody。"""

    cid: int
    arm_id: int
    belt_speed: float = BELT_SPEED
    bodies: dict = field(default_factory=dict)
    parts: list = field(default_factory=list)

    def __post_init__(self):
        self._build()
        self.set_arm_groups()

    # ------------------------------------------------------------ 建模
    def _box(self, name, center, half, rgba, group="static", friction=0.8):
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(half), physicsClientId=self.cid)
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=list(half), rgbaColor=rgba,
                                   physicsClientId=self.cid)
        b = pb.createMultiBody(0, col, vis, [float(c) for c in center], physicsClientId=self.cid)
        pb.changeDynamics(b, -1, lateralFriction=friction, physicsClientId=self.cid)
        pb.setCollisionFilterGroupMask(b, -1, GROUPS[group], MASKS[group], physicsClientId=self.cid)
        self.bodies[name] = b
        return b

    def _build(self):
        T = BENCH_TOP
        self._box("bench", (BENCH_X, 0, T / 2), (BENCH_HX, BENCH_HY, T / 2), (0.55, 0.57, 0.62, 1),
                  friction=1.0)
        # 料盤：底板 + 每格 2 端板 + 4 側銷
        tw = TRAY_PITCH + POCKET + 2 * POST
        self._box("tray", (TRAY_C[0], TRAY_C[1], T + TRAY_PLATE_T / 2), (tw / 2, tw / 2, TRAY_PLATE_T / 2),
                  (0.25, 0.3, 0.38, 1), friction=1.0)
        zt = T + TRAY_PLATE_T
        o = POCKET / 2 + POST / 2
        for i, (px, py) in enumerate(pocket_centers()):
            for j, sx in enumerate((-1, 1)):  # ±x 端板：沿 y 全長，限制 x
                self._box(f"wall{i}{j}", (px + sx * o, py, zt + POST_H / 2),
                          (POST / 2, POCKET / 2, POST_H / 2), (0.95, 0.75, 0.15, 1))
            for j, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):  # ±y 側銷，限制 y
                self._box(f"pin{i}{j}", (px + sx * PIN_X, py + sy * o, zt + POST_H / 2),
                          (POST / 4 + 0.001, POST / 2, POST_H / 2), (0.95, 0.75, 0.15, 1))
        # 治具：+x 後擋板、+y 側銷（銷靠 +x 端，避開兩指 x 範圍 ±10 mm）；導面低摩擦（μ_eff≈0.3）
        fx, fy = FIX_NOM
        self._box("fix_back", (fx + 0.025 + FIX_WALL_T / 2, fy, T + FIX_H / 2),
                  (FIX_WALL_T / 2, 0.05, FIX_H / 2), (0.9, 0.5, 0.1, 1), friction=0.4)
        self._box("fix_pin", (fx + 0.019, fy + 0.025 + FIX_WALL_T / 2, T + FIX_H / 2),
                  (0.006, FIX_WALL_T / 2, FIX_H / 2), (0.9, 0.5, 0.1, 1), friction=0.4)
        # 輸送帶：機架、帶面（群組 belt、摩擦 0）、護欄、擋料板
        cx, cy = (BELT_X0 + BELT_X1) / 2, (BELT_Y0 + BELT_Y1) / 2
        hx, hy = (BELT_X1 - BELT_X0) / 2, (BELT_Y1 - BELT_Y0) / 2
        self._box("belt_frame", (cx, cy, (T - BELT_T) / 2), (hx, hy, (T - BELT_T) / 2), (0.2, 0.22, 0.25, 1))
        self._box("belt", (cx, cy, T - BELT_T / 2), (hx, hy, BELT_T / 2), (0.12, 0.12, 0.13, 1),
                  group="belt", friction=0.0)
        for nm, y in (("rail0", BELT_Y0 - 0.005), ("rail1", BELT_Y1 + 0.005)):
            self._box(nm, (cx, y, T + 0.01), (hx, 0.005, 0.01), (0.7, 0.72, 0.75, 1))
        self._box("stopper", (STOP_X + 0.005, cy, T + 0.02), (0.005, hy, 0.02), (0.85, 0.2, 0.15, 1))

    def set_arm_groups(self):
        """手臂（含夾爪）所有連桿設為 arm 群組。"""
        for link in range(-1, pb.getNumJoints(self.arm_id, physicsClientId=self.cid)):
            pb.setCollisionFilterGroupMask(self.arm_id, link, GROUPS["arm"], MASKS["arm"],
                                           physicsClientId=self.cid)

    def add_part(self, body: int):
        """註冊零件：設 part 群組、加入輸送帶驅動清單。"""
        pb.setCollisionFilterGroupMask(body, -1, GROUPS["part"], MASKS["part"], physicsClientId=self.cid)
        self.parts.append(body)

    def add_ghost(self, pos, radius=0.02) -> int:
        """群組/遮罩皆為 0 的虛擬體（驗證群組過濾：不與任何物體碰撞）。"""
        col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius, physicsClientId=self.cid)
        b = pb.createMultiBody(0.1, col, -1, [float(c) for c in pos], physicsClientId=self.cid)
        pb.setCollisionFilterGroupMask(b, -1, 0, 0, physicsClientId=self.cid)
        return b

    # ------------------------------------------------------------ 輸送帶
    def on_belt(self, p) -> bool:
        return BELT_X0 <= p[0] <= BELT_X1 and BELT_Y0 <= p[1] <= BELT_Y1

    def pre_step(self):
        """每個物理步之前呼叫：對帶上零件施加庫侖限幅的速度追隨力。"""
        dt = pb.getPhysicsEngineParameters(physicsClientId=self.cid)["fixedTimeStep"]
        for b in self.parts:
            if not pb.getContactPoints(b, self.bodies["belt"], physicsClientId=self.cid):
                continue
            m = pb.getDynamicsInfo(b, -1, physicsClientId=self.cid)[0]
            v = np.array(pb.getBaseVelocity(b, physicsClientId=self.cid)[0])
            a = float(np.clip((self.belt_speed - v[0]) / dt, -BELT_MU * G, BELT_MU * G))
            pos = pb.getBasePositionAndOrientation(b, physicsClientId=self.cid)[0]
            pb.applyExternalForce(b, -1, [m * a, 0, 0], pos, pb.WORLD_FRAME, physicsClientId=self.cid)

    # ------------------------------------------------------------ 碰撞查詢（供路徑規劃 T110 使用）
    def arm_collisions(self, margin: float = 0.0) -> list[str]:
        """目前姿態下，手臂（含夾爪）與場景靜態體（工作台、治具、料盤、輸送帶…）距離 ≤ margin 者名稱。"""
        hit = []
        for nm, b in self.bodies.items():
            if pb.getClosestPoints(self.arm_id, b, margin, physicsClientId=self.cid):
                hit.append(nm)
        return hit

    def min_clearance(self, search: float = 0.3) -> float:
        """手臂與場景靜態體最小距離 [m]（負值＝穿透；search 內無點時回傳 search）。"""
        d = [search]
        for b in self.bodies.values():
            d += [c[8] for c in pb.getClosestPoints(self.arm_id, b, search, physicsClientId=self.cid)]
        return float(min(d))
