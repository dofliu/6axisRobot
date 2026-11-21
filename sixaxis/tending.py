"""T112 機台上下料（machine tending）：CNC 加工機（自動門、虎鉗、主軸）＋六軸手臂
＋料盤（胚料入料）/緩衝台/出料輸送帶。

1. 工作站（世界座標，m；手臂基座在原點，沿用 T006 工作台/料盤/輸送帶）
   * 加工機在 −y 側：機殼 x ∈ [X0, X1]、y ∈ [Y_BACK, FY]；前牆（面向手臂）y = FY，
     門洞 x ∈ [DX0, DX1]、z ∈ [SILL, DOOR_TOP]；機台台面 z = 0.10；虎鉗中心（夾緊後零件中心）(VX, VY)。
   * 料盤（T006，4 格）＝胚料入料；輸送帶（T006，+x 前進到擋料板）＝成品出料；
     緩衝台（門外台座，頂面 0.20 m，兩格 A = 成品、B = 胚料）供「緩衝交換」策略使用。

2. 加工機模型（`Machine`，每個物理步 `tick`）
   * 自動拉門：運動學驅動，梯形速度曲線 v_max、a_max：行程 s ≥ v²/a 時 T = s/v + v/a
     （否則三角形 T = 2√(s/a)）。關門中光幕區（門洞平面）偵測到手臂 → 立即停止並鎖存 CURTAIN_TRIP
     （IEC 61496 光幕的功能等效；只做幾何偵測，不模擬停止距離 ISO 13855）。
   * 虎鉗：固定鉗口（基準面 x = VX − a/2）＋活動鉗口（PyBullet 稜柱關節，POSITION_CONTROL、
     力上限 = 夾緊力 F_c）。活動鉗口頂住零件時馬達飽和，法向力 = F_c。
     x 方向由固定鉗口定位（推向基準，修正放置誤差），y 方向不定位（= 手臂放置精度）。
     監控：夾緊後活動鉗口位置窗（零件寬度一致）＋座面檢知（零件底面 4 角離座面 ≤ SEAT_TOL，
     背壓式氣密座面檢知的等效量）。
   * 夾持力（Coulomb）：側向力使零件滑移的門檻 F_slip = 2 μ_j F_c + μ_f m g
     （兩鉗口面各 μ_j F_c、底面 μ_f m g；PyBullet 的接觸摩擦係數 = 兩物體係數相乘）。
   * 加工循環：主軸下降 → 切削（側向切削力 F_cut = k_c a_p h，比切削力 k_c 取常數的
     Kienzle 型簡化，方向每道次交替 ±y）→ 主軸上升；上升開始時胚料換成成品
     （頂面口袋 30×30×15 mm：質量與質心依移除體積重算）。
   * PLC 互鎖（信號以 (t, 名稱, 值) 記錄成時序圖）：
     - 關門命令：手臂任一連桿在機台區（門洞平面或機殼內）→ 拒絕（INTERLOCK_DOOR）。
     - 循環啟動：須門全關、已夾緊、夾緊監控 OK → 否則拒絕（INTERLOCK_START）；
       `request_start` 預約後門全關時 PLC 自動啟動。循環中開門 → 拒絕。
     - 手臂側：門未全開不進入（`TendCell.enter` 等待 door_open 信號）。

3. 手臂側流程（`TendCell`）：大範圍移動用 T110 `CollisionChecker`（機殼/門/主軸/緩衝台為靜態障礙，
   夾持件 attach）保證式直線檢查，不通則 Informed RRT*＋捷徑；進出門洞與升降用 move_l（教導路徑）。
   * 上料：門外 P_out → move_l 進門到虎鉗上方 → 下降到開口中心 → 釋放 → 夾緊（同時抬升）
     → 監控 → 退出 → 關門＋預約啟動。
   * 下料：等循環結束 → 開門 → 進門 → 夾取 → 鬆鉗 → 抬升 → 退出。
   * 策略 S1「依序」：下料 → 成品送輸送帶 → 料盤取胚料 → 上料（交換期間機台閒置）。
   * 策略 S2「緩衝交換」：加工期間手臂把上一件成品由緩衝 A 送輸送帶、把下一件胚料由料盤放到
     緩衝 B，回門外等待；交換時只做 下料 → 放 A → 取 B → 上料。
   * 週期模型（`cycle_model`）：T_c = T_open + T_ex + max(T_close + T_m, T_side)，
     T_ex = 門全開 → 關門命令（機台必閒置的交換時間）、T_side = 加工期間手臂側工作（S1 = 0）；
     稼動率 U = T_m / T_c。即「工作單元週期 = 機台時間 + 不重疊的手臂時間」
     （Groover, Automation, Production Systems, and Computer-Integrated Manufacturing, 機台照顧/
     工作單元週期分析）。
   * 故障恢復：座面檢知失敗（切屑墊在零件下）→ 鬆鉗 → 夾回 → 抬起 → 吹氣（移除切屑）→ 重放 → 再檢；
     料盤空格（GRIP_EMPTY）→ 跳下一格。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as pb

from . import scene as S
from .commands import Executor
from .graspsim import Q_REF, GraspScene
from .gripper import CUBE, ObjectSpec
from .planning import Box, CollisionChecker, RRTStar, path_length, shortcut
from .robot import G, wrap_pi
from .scenarios import workcell_boxes
from .sorting import grasp_offset, ik_near
from .tasks import Z_POCKET

# ---------------------------------------------------------------------- 佈置
VX, VY = 0.22, -0.54  # 虎鉗：夾緊後零件中心
FY = -0.37  # 前牆面向手臂的面
WALL_T = 0.02
X0, X1, Y_BACK = -0.02, 0.84, -0.84  # 右側機殼較寬：開門時門板滑到右前牆外側（不突出機殼）
DX0, DX1 = VX - 0.18, VX + 0.18  # 門洞
SILL, DOOR_TOP, ROOF = 0.16, 0.70, 0.86
TABLE_TOP = 0.10
DOOR_T, DOOR_GAP = 0.012, 0.006  # 門板厚、與前牆間隙（門在牆外側，手臂側）
DOOR_Y = FY + DOOR_GAP + DOOR_T / 2
DOOR_STROKE = DX1 - DX0 + 0.04
DOOR_V, DOOR_A = 0.30, 1.0

PART_A = CUBE.size[0]  # 50 mm 立方胚料
VISE_PLATE = 0.02
VISE_FLOOR = TABLE_TOP + VISE_PLATE
JAW_T, JAW_H, JAW_L = 0.015, 0.025, 0.060
JAW_OPEN = 0.012  # 活動鉗口全開時與零件面的間隙（開口 62 mm）
X_FIX = VX - PART_A / 2  # 固定鉗口基準面
X_PLACE = VX + JAW_OPEN / 2  # 放置目標：開口中心
Z_PART = VISE_FLOOR + PART_A / 2
F_CLAMP = 200.0
JAW_VMAX = 0.03
MU_JAW, MU_FLOOR = 0.4, 0.3  # 鉗口/座面材料摩擦係數（與零件 0.8 相乘 → 有效 0.32 / 0.24）
SEAT_TOL = 3e-4  # 座面檢知門檻 0.3 mm
JAW_WIN = 5e-4  # 夾緊位置窗 ±0.5 mm

SPINDLE_R, SPINDLE_L = 0.045, 0.20
SPINDLE_HOME = 0.64  # 主軸頭下緣（待機）
SPINDLE_CUT = Z_PART + PART_A / 2 + 0.065  # 切削時主軸頭下緣（刀具 65 mm 伸入口袋位置）
T_SPINDLE = 1.0  # 主軸下降/上升時間

Z_ENTRY = 0.30  # 進出門洞時 TCP 高度
P_OUT = (VX, -0.24, Z_ENTRY)  # 門外等待點
P_IN = (VX, VY, Z_ENTRY)
BUF_TOP = 0.20
BUF_C = (0.10, -0.245)
BUF_SLOTS = {"A": (0.06, -0.245), "B": (0.14, -0.245)}
BELT_DROP = (0.25, 0.45)
Z_TRAVEL = 0.27
V_LIN = 0.5  # 直線段 TCP 峰值速度 [m/s]

# 成品：頂面口袋
POCKET_W, POCKET_D = 0.030, 0.015
RHO = CUBE.mass / PART_A**3
K_C, A_P, H_CHIP = 700.0, 1.0, 0.05  # 比切削力 [N/mm²]（鋁合金量級）、切深 [mm]、切屑厚 [mm]
F_CUT = K_C * A_P * H_CHIP  # = 35 N
PASS_T = 2.0  # 每道次時間（切削力方向交替）


def door_time(s: float = DOOR_STROKE, v: float = DOOR_V, a: float = DOOR_A) -> float:
    """梯形速度曲線行程時間。"""
    return s / v + v / a if s >= v * v / a else 2.0 * np.sqrt(s / a)


def door_pos(t: float, s: float = DOOR_STROKE, v: float = DOOR_V, a: float = DOOR_A) -> float:
    """梯形速度曲線下 t 秒時的位移 [m]（0 ≤ 位移 ≤ s）。"""
    T = door_time(s, v, a)
    if t <= 0:
        return 0.0
    if t >= T:
        return s
    if s < v * v / a:  # 三角形
        v = np.sqrt(s * a)
    ta = v / a
    if t < ta:
        return 0.5 * a * t * t
    if t < T - ta:
        return 0.5 * a * ta * ta + v * (t - ta)
    r = T - t
    return s - 0.5 * a * r * r


def finished_props() -> dict:
    """成品（立方體頂面挖口袋）的質量與質心（相對外形中心）：m = ρ(a³ − w²d)，
    z_c = −(w²d)(a/2 − d/2)/m·ρ（挖去體積的質心在 a/2 − d/2）。"""
    a, w, d = PART_A, POCKET_W, POCKET_D
    v_cut = w * w * d
    m = RHO * (a**3 - v_cut)
    zc = -(RHO * v_cut) * (a / 2 - d / 2) / m
    return {"mass": m, "com_z": zc, "removed_g": RHO * v_cut * 1e3}


def slip_threshold(F_c: float, mass: float = CUBE.mass, mu_part: float = CUBE.friction) -> float:
    """Coulomb 夾持門檻：F_slip = 2 μ_j F_c + μ_f m g（有效摩擦係數 = 材料係數乘積）。"""
    return 2 * MU_JAW * mu_part * F_c + MU_FLOOR * mu_part * mass * G


def machine_boxes(door: str = "both") -> list[Box]:
    """加工機靜態方塊（規劃用）：機台、前牆（左/右/楣/檻）、側/後牆、頂、虎鉗座、固定鉗口、主軸頭（待機位）、
    緩衝台；door = 'closed' / 'open' / 'both'（門板在兩個位置都視為障礙物，保守）。活動鉗口以全開位置計。"""
    yc = (FY + Y_BACK) / 2
    hy = (FY - Y_BACK) / 2
    zc, hz = ROOF / 2, ROOF / 2
    fwy = FY - WALL_T / 2
    out = [
        Box(((X0 + X1) / 2, yc, TABLE_TOP / 2), ((X1 - X0) / 2, hy, TABLE_TOP / 2), "m_table"),
        Box(((X0 + DX0) / 2, fwy, zc), ((DX0 - X0) / 2, WALL_T / 2, hz), "m_front_l"),
        Box(((DX1 + X1) / 2, fwy, zc), ((X1 - DX1) / 2, WALL_T / 2, hz), "m_front_r"),
        Box(
            (VX, fwy, (DOOR_TOP + ROOF) / 2), ((DX1 - DX0) / 2, WALL_T / 2, (ROOF - DOOR_TOP) / 2), "m_lintel"
        ),
        Box((VX, fwy, SILL / 2), ((DX1 - DX0) / 2, WALL_T / 2, SILL / 2), "m_sill"),
        Box((X0 - WALL_T / 2, yc, zc), (WALL_T / 2, hy, hz), "m_side_l"),
        Box((X1 + WALL_T / 2, yc, zc), (WALL_T / 2, hy, hz), "m_side_r"),
        Box(((X0 + X1) / 2, Y_BACK - WALL_T / 2, zc), ((X1 - X0) / 2 + WALL_T, WALL_T / 2, hz), "m_back"),
        Box(
            ((X0 + X1) / 2, yc, ROOF + WALL_T / 2),
            ((X1 - X0) / 2 + WALL_T, hy + WALL_T, WALL_T / 2),
            "m_roof",
        ),
        Box((VX, VY, TABLE_TOP + VISE_PLATE / 2), (0.07, 0.05, VISE_PLATE / 2), "vise_base"),
        Box((X_FIX - JAW_T / 2, VY, VISE_FLOOR + JAW_H / 2), (JAW_T / 2, JAW_L / 2, JAW_H / 2), "jaw_fixed"),
        Box((VX, VY, SPINDLE_HOME + SPINDLE_L / 2), (SPINDLE_R, SPINDLE_R, SPINDLE_L / 2), "spindle"),
        Box(
            (VX, VY, (SPINDLE_HOME + SPINDLE_L + ROOF) / 2),
            (0.06, 0.06, (ROOF - SPINDLE_HOME - SPINDLE_L) / 2),
            "spindle_col",
        ),
        Box((*BUF_C, BUF_TOP / 2), (0.085, 0.045, BUF_TOP / 2), "buffer"),
    ]
    mj = X_FIX + PART_A + JAW_OPEN + JAW_T / 2
    out.append(Box((mj, VY, VISE_FLOOR + JAW_H / 2), (JAW_T / 2, JAW_L / 2, JAW_H / 2), "jaw_move"))
    if door in ("closed", "both"):
        out.append(door_box(0.0, "door_closed"))
    if door in ("open", "both"):
        out.append(door_box(DOOR_STROKE, "door_open"))
    return out


def door_box(s: float, name: str = "door") -> Box:
    return Box(
        (VX + s, DOOR_Y, (SILL + DOOR_TOP) / 2 + 0.01),
        ((DX1 - DX0) / 2 + 0.02, DOOR_T / 2, (DOOR_TOP - SILL) / 2 + 0.03),
        name,
    )


def in_machine_zone(p) -> bool:
    """點是否在機台管制區（門洞平面前 30 mm 起至機殼內）。"""
    x, y, z = p
    return bool(X0 < x < X1 and Y_BACK < y < FY + 0.03 and z < ROOF)


# ---------------------------------------------------------------------- 加工機（物理模擬內）
@dataclass
class Machine:
    cid: int
    wc: object  # scene.Workcell（加入 bodies 以便執行器碰撞監控）
    arm_id: int
    T_m: float = 10.0
    F_c: float = F_CLAMP
    cut_force: float = F_CUT
    t: float = 0.0
    signals: list = field(default_factory=list)
    events: list = field(default_factory=list)
    swap_cb: object = None  # 加工結束時呼叫 swap_cb(machine) 把胚料換成成品

    def __post_init__(self):
        wc = self.wc
        for b in machine_boxes(door="none"):
            if b.name.startswith("door") or b.name in ("jaw_move", "spindle", "spindle_col", "buffer"):
                continue
            col = (0.80, 0.82, 0.84, 1) if b.name.startswith("m_") else (0.35, 0.37, 0.40, 1)
            if b.name in ("m_table",):
                col = (0.30, 0.32, 0.36, 1)
            fr = MU_FLOOR if b.name == "vise_base" else (MU_JAW if b.name == "jaw_fixed" else 0.8)
            wc._box(b.name, b.center, b.half, col, friction=fr)
        bb = [b for b in machine_boxes(door="none") if b.name == "buffer"][0]
        wc._box("buffer", bb.center, bb.half, (0.45, 0.47, 0.52, 1), friction=1.0)
        # 門（運動學）
        db = door_box(0.0)
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(db.half), physicsClientId=self.cid)
        vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=list(db.half),
            rgbaColor=(0.95, 0.6, 0.12, 0.55),
            physicsClientId=self.cid,
        )
        self.door = pb.createMultiBody(0, col, vis, list(db.center), physicsClientId=self.cid)
        pb.setCollisionFilterGroupMask(
            self.door, -1, S.GROUPS["static"], S.MASKS["static"], physicsClientId=self.cid
        )
        wc.bodies["door"] = self.door
        # 活動鉗口：固定基座（不可見）＋稜柱關節連桿
        jh = [JAW_T / 2, JAW_L / 2, JAW_H / 2]
        jc = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=jh, physicsClientId=self.cid)
        jv = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=jh, rgbaColor=(0.85, 0.3, 0.2, 1), physicsClientId=self.cid
        )
        base_pos = [X_FIX + PART_A + JAW_T / 2, VY, VISE_FLOOR + JAW_H / 2]
        self.jaw = pb.createMultiBody(
            0,
            -1,
            -1,
            base_pos,
            linkMasses=[0.5],
            linkCollisionShapeIndices=[jc],
            linkVisualShapeIndices=[jv],
            linkPositions=[[0, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[pb.JOINT_PRISMATIC],
            linkJointAxis=[[1, 0, 0]],
            physicsClientId=self.cid,
        )
        pb.changeDynamics(self.jaw, 0, lateralFriction=MU_JAW, physicsClientId=self.cid)
        pb.setCollisionFilterGroupMask(
            self.jaw, 0, S.GROUPS["static"], S.MASKS["static"], physicsClientId=self.cid
        )
        pb.resetJointState(self.jaw, 0, JAW_OPEN, physicsClientId=self.cid)
        wc.bodies["jaw_move"] = self.jaw
        # 主軸（運動學、無物理接觸：碰撞遮罩 0，但仍可做距離監控）
        sc = pb.createCollisionShape(
            pb.GEOM_CYLINDER, radius=SPINDLE_R, height=SPINDLE_L, physicsClientId=self.cid
        )
        sv = pb.createVisualShape(
            pb.GEOM_CYLINDER,
            radius=SPINDLE_R,
            length=SPINDLE_L,
            rgbaColor=(0.25, 0.27, 0.32, 1),
            physicsClientId=self.cid,
        )
        tv = pb.createVisualShape(
            pb.GEOM_CYLINDER,
            radius=0.008,
            length=0.065,
            rgbaColor=(0.85, 0.85, 0.9, 1),
            visualFramePosition=[0, 0, -SPINDLE_L / 2 - 0.0325],
            physicsClientId=self.cid,
        )
        self.spindle = pb.createMultiBody(
            0, sc, sv, [VX, VY, SPINDLE_HOME + SPINDLE_L / 2], physicsClientId=self.cid
        )
        self.tool = pb.createMultiBody(
            0, -1, tv, [VX, VY, SPINDLE_HOME + SPINDLE_L / 2], physicsClientId=self.cid
        )
        pb.setCollisionFilterGroupMask(self.spindle, -1, 0, 0, physicsClientId=self.cid)
        wc.bodies["spindle"] = self.spindle
        # 管制區（幾何偵測用的虛擬體）：門洞平面（光幕）與機殼內部
        self.zone_bodies = []
        for c, h in (
            ((VX, FY + 0.005, (SILL + DOOR_TOP) / 2), ((DX1 - DX0) / 2, 0.025, (DOOR_TOP - SILL) / 2)),
            (((X0 + X1) / 2, (FY + Y_BACK) / 2, ROOF / 2), ((X1 - X0) / 2, (FY - Y_BACK) / 2, ROOF / 2)),
        ):
            zc_ = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(h), physicsClientId=self.cid)
            zb = pb.createMultiBody(0, zc_, -1, list(c), physicsClientId=self.cid)
            pb.setCollisionFilterGroupMask(zb, -1, 0, 0, physicsClientId=self.cid)
            self.zone_bodies.append(zb)
        # 狀態
        self.door_s = 0.0  # 0 = 關
        self.door_target = 0.0
        self.door_t0 = 0.0
        self.door_s0 = 0.0
        self.door_moving = False
        self.trip = False
        self.jaw_target = JAW_OPEN
        self.clamp_cmd = False
        self.clamp_t0 = -1.0
        self.cycle = None  # 加工循環開始時間
        self.part = None  # 虎鉗上的零件 body
        self.state = "IDLE"
        self.done = False
        self.cut_log: list = []
        self.start_armed = False
        self._sig = {}
        self._jaw_cmd(JAW_OPEN, 50.0)
        self._set_door(0.0)
        self.set_sig("door_closed", True)
        self.set_sig("door_open", False)
        self.set_sig("clamped", False)
        self.set_sig("cycle_busy", False)
        self.set_sig("cycle_done", False)

    # --- 信號
    def set_sig(self, name, val):
        if self._sig.get(name) != val:
            self._sig[name] = val
            self.signals.append((self.t, name, val))

    def sig(self, name) -> bool:
        return bool(self._sig.get(name, False))

    def event(self, kind, **kw):
        self.events.append({"t": self.t, "event": kind, **kw})

    # --- 管制區偵測
    def arm_in_zone(self, which: str = "any") -> bool:
        zs = {"curtain": self.zone_bodies[:1], "inside": self.zone_bodies[1:], "any": self.zone_bodies}[which]
        for zb in zs:
            for c in pb.getClosestPoints(self.arm_id, zb, 0.0, physicsClientId=self.cid):
                if c[3] >= 1:  # 不含基座（基座固定在區外）
                    return True
        return False

    # --- 門
    def _set_door(self, s):
        self.door_s = float(s)
        b = door_box(s)
        pb.resetBasePositionAndOrientation(self.door, list(b.center), [0, 0, 0, 1], physicsClientId=self.cid)

    def open_door(self) -> bool:
        if self.sig("cycle_busy"):
            self.event("INTERLOCK_DOOR_OPEN", reason="cycle busy")
            return False
        self.event("DOOR_OPEN_CMD")
        self._door_move(DOOR_STROKE)
        return True

    def close_door(self) -> bool:
        if self.arm_in_zone():
            self.event("INTERLOCK_DOOR", reason="arm in machine zone")
            return False
        self.trip = False
        self.event("DOOR_CLOSE_CMD")
        self._door_move(0.0)
        return True

    def request_start(self):
        """PLC：門全關時自動啟動循環（手臂下關門命令後即可離開去做其他工作）。"""
        self.start_armed = True

    def _door_move(self, target):
        self.door_target, self.door_t0, self.door_s0 = target, self.t, self.door_s
        self.door_moving = abs(target - self.door_s) > 1e-9
        self.set_sig("door_closed", False)
        self.set_sig("door_open", False)

    def _door_tick(self):
        if not self.door_moving:
            return
        if self.door_target < self.door_s and self.arm_in_zone("curtain"):  # 關門中光幕遮斷 → 停止
            self.door_moving = False
            self.trip = True
            self.event("CURTAIN_TRIP", door_s=self.door_s)
            return
        dist = abs(self.door_target - self.door_s0)
        p = door_pos(self.t - self.door_t0, dist)
        s = self.door_s0 + np.sign(self.door_target - self.door_s0) * p
        self._set_door(s)
        if p >= dist - 1e-12:
            self.door_moving = False
            self._set_door(self.door_target)
            self.set_sig("door_open", self.door_target > 0)
            self.set_sig("door_closed", self.door_target == 0)

    # --- 虎鉗
    def _jaw_cmd(self, target, force):
        self.jaw_target = target
        pb.setJointMotorControl2(
            self.jaw,
            0,
            pb.POSITION_CONTROL,
            targetPosition=target,
            force=force,
            maxVelocity=JAW_VMAX,
            positionGain=0.3,
            velocityGain=1.0,
            physicsClientId=self.cid,
        )

    def jaw_pos(self) -> float:
        return float(pb.getJointState(self.jaw, 0, physicsClientId=self.cid)[0])

    def jaw_force(self) -> float:
        return float(pb.getJointState(self.jaw, 0, physicsClientId=self.cid)[3])

    def clamp(self, part=None):
        self.part = part if part is not None else self.part
        self._jaw_cmd(-0.010, self.F_c)
        self.clamp_cmd, self.clamp_t0 = True, self.t
        self.set_sig("clamped", False)

    def unclamp(self):
        self._jaw_cmd(JAW_OPEN, 50.0)
        self.clamp_cmd, self.clamp_t0 = False, self.t
        self.set_sig("clamped", False)

    def jaw_settled(self) -> bool:
        v = float(pb.getJointState(self.jaw, 0, physicsClientId=self.cid)[1])
        return abs(v) < 1e-3 and self.t - self.clamp_t0 > 0.15

    def unclamped(self) -> bool:
        return (not self.clamp_cmd) and abs(self.jaw_pos() - JAW_OPEN) < 5e-4

    def seat_gap(self, body=None) -> float:
        """零件底面 4 角離座面的最大高度 [m]（背壓式座面檢知的等效量）。"""
        body = self.part if body is None else body
        if body is None:
            return np.inf
        pos, quat = pb.getBasePositionAndOrientation(body, physicsClientId=self.cid)
        R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
        h = PART_A / 2
        cs = np.array([[sx * h, sy * h, -h] for sx in (-1, 1) for sy in (-1, 1)]) @ R.T + np.array(pos)
        return float(np.max(cs[:, 2]) - VISE_FLOOR)

    def check(self) -> dict:
        """夾緊監控：活動鉗口位置窗（零件寬 50 mm ⇒ 鉗口位置 ≈ 0）＋座面檢知。"""
        jp = self.jaw_pos()
        gap = self.seat_gap()
        ok_w = abs(jp) <= JAW_WIN
        ok_s = gap <= SEAT_TOL
        return {
            "jaw_mm": jp * 1e3,
            "seat_gap_mm": gap * 1e3,
            "jaw_ok": bool(ok_w),
            "seat_ok": bool(ok_s),
            "ok": bool(ok_w and ok_s),
        }

    # --- 循環
    def cycle_start(self) -> bool:
        reasons = []
        if not self.sig("door_closed"):
            reasons.append("door not closed")
        if not self.sig("clamped"):
            reasons.append("not clamped")
        if self.part is None or not self.check()["ok"]:
            reasons.append("part check failed")
        if reasons:
            self.event("INTERLOCK_START", reason="; ".join(reasons))
            return False
        self.cycle = self.t
        self.done = False
        self.state = "CYCLE"
        self.set_sig("cycle_busy", True)
        self.set_sig("cycle_done", False)
        self.event("CYCLE_START")
        self._swapped = False
        self._p_ref = None
        return True

    def _cycle_tick(self):
        if self.cycle is None:
            return
        tc = self.t - self.cycle
        T_cut = max(self.T_m - 2 * T_SPINDLE, 0.0)
        if tc < T_SPINDLE:
            z = SPINDLE_HOME + (SPINDLE_CUT - SPINDLE_HOME) * (0.5 - 0.5 * np.cos(np.pi * tc / T_SPINDLE))
        elif tc < T_SPINDLE + T_cut:
            z = SPINDLE_CUT
            if self.part is not None and self.cut_force > 0:  # 側向切削力，方向每道次交替
                k = int((tc - T_SPINDLE) // PASS_T)
                F = self.cut_force * (1 if k % 2 == 0 else -1)
                pos = pb.getBasePositionAndOrientation(self.part, physicsClientId=self.cid)[0]
                pb.applyExternalForce(self.part, -1, [0, F, 0], pos, pb.WORLD_FRAME, physicsClientId=self.cid)
                if self._p_ref is None:
                    self._p_ref = np.array(pos)
                self.cut_log.append(float(np.linalg.norm(np.array(pos) - self._p_ref)))
        elif tc < self.T_m:
            if not self._swapped:
                self._swapped = True
                if self.swap_cb:
                    self.swap_cb(self)
            u = (tc - T_SPINDLE - T_cut) / T_SPINDLE
            z = SPINDLE_CUT + (SPINDLE_HOME - SPINDLE_CUT) * (0.5 - 0.5 * np.cos(np.pi * u))
        else:
            z = SPINDLE_HOME
            self.cycle = None
            self.done = True
            self.state = "DONE"
            self.set_sig("cycle_busy", False)
            self.set_sig("cycle_done", True)
            self.event("CYCLE_DONE")
        c = [VX, VY, z + SPINDLE_L / 2]
        pb.resetBasePositionAndOrientation(self.spindle, c, [0, 0, 0, 1], physicsClientId=self.cid)
        ang = 2 * np.pi * 5 * (self.t - (self.cycle or 0.0)) if self.cycle is not None else 0.0
        pb.resetBasePositionAndOrientation(
            self.tool, c, [0, 0, np.sin(ang / 2), np.cos(ang / 2)], physicsClientId=self.cid
        )

    def tick(self, dt: float):
        self.t += dt
        self._door_tick()
        self._cycle_tick()
        if self.clamp_cmd and not self.sig("clamped") and self.jaw_settled():
            self.set_sig("clamped", True)
        if self.start_armed and self.sig("door_closed"):
            self.start_armed = False
            self.cycle_start()
        if self.trip:
            self.start_armed = False


# ---------------------------------------------------------------------- 零件
def spawn_finished(cid: int, pos, quat) -> int:
    """成品：外觀＝底板＋四壁（頂面口袋 30×30×15 mm）；質量與質心依移除體積（finished_props）。
    碰撞形狀用完整外形方塊：口袋在頂面內部，不與夾爪/鉗口/輸送帶接觸，對接觸無影響；
    曾用 5 個方塊的複合碰撞形狀，指墊同時壓在底板與側壁兩個共面面上，PyBullet 接觸法向偏斜、
    夾持力異常上升到 80–100 N 且鬆爪後黏住（實測），故改為單一方塊。pos/quat 為外形中心位姿。"""
    a, w, d = PART_A, POCKET_W, POCKET_D
    t = (a - w) / 2
    fp = finished_props()
    halves = [[a / 2, a / 2, (a - d) / 2]]
    poss = [[0, 0, -d / 2]]
    for sx, sy, hx, hy in (
        (1, 0, t / 2, a / 2),
        (-1, 0, t / 2, a / 2),
        (0, 1, w / 2, t / 2),
        (0, -1, w / 2, t / 2),
    ):
        halves.append([hx, hy, d / 2])
        poss.append([sx * (a / 2 - t / 2), sy * (a / 2 - t / 2), a / 2 - d / 2])
    n = len(halves)
    col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[a / 2] * 3, physicsClientId=cid)
    vis = pb.createVisualShapeArray(
        [pb.GEOM_BOX] * n,
        halfExtents=halves,
        visualFramePositions=poss,
        rgbaColors=[[0.72, 0.75, 0.80, 1]] * n,
        physicsClientId=cid,
    )
    b = pb.createMultiBody(
        fp["mass"],
        col,
        vis,
        list(map(float, pos)),
        list(quat),
        baseInertialFramePosition=[0, 0, fp["com_z"]],
        physicsClientId=cid,
    )
    pb.changeDynamics(
        b,
        -1,
        lateralFriction=CUBE.friction,
        frictionAnchor=1,
        linearDamping=0.0,
        angularDamping=0.0,
        physicsClientId=cid,
    )
    return b


def part_center(cid: int, body: int, finished: bool) -> np.ndarray:
    """外形中心（成品的 base 位置是質心，需加回偏移）。"""
    pos, quat = pb.getBasePositionAndOrientation(body, physicsClientId=cid)
    if not finished:
        return np.array(pos)
    R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return np.array(pos) - R @ np.array([0, 0, finished_props()["com_z"]])


@dataclass
class Part:
    body: int
    finished: bool = False
    pocket: int = -1
    history: list = field(default_factory=list)


# ---------------------------------------------------------------------- 工作站（物理執行）
@dataclass
class TendConfig:
    strategy: str = "serial"  # "serial"（S1）/ "buffer"（S2）
    T_m: float = 10.0
    n_parts: int = 4
    F_c: float = F_CLAMP
    cut_force: float = F_CUT
    force: float = 40.0  # 夾爪夾持力
    place_err: tuple = (0.0, 0.0)  # 上料時刻意加入的放置誤差 (dx, dy) [m]
    chip_at: tuple = ()  # 第 k 件下料後在虎鉗座面留下切屑（故障注入）
    empty_pockets: tuple = ()  # 料盤空格
    recover: bool = True
    rrt_time: float = 20.0
    max_retry: int = 2


@dataclass
class TendCell:
    cfg: TendConfig = field(default_factory=TendConfig)
    render: bool = False
    frame_cb: object = None
    cb_every: int = 25

    def __post_init__(self):
        cfg = self.cfg
        self._td = tempfile.TemporaryDirectory()
        self.s = GraspScene(
            CUBE, workcell=True, render=self.render, urdf_path=Path(self._td.name) / "arm_gripper.urdf"
        )
        s = self.s
        pb.removeBody(s.obj, physicsClientId=s.cid)
        s.wc.parts.clear()
        self.m = Machine(
            s.cid, s.wc, s.sim.id, T_m=cfg.T_m, F_c=cfg.F_c, cut_force=cfg.cut_force, swap_cb=self._swap
        )
        wc_pre = s.wc.pre_step

        def pre():
            wc_pre()
            self.m.tick(s.dt)

        s.wc.pre_step = pre
        # 胚料：料盤 4 格
        self.parts: list[Part] = []
        for k, (x, y) in enumerate(S.pocket_centers()[: cfg.n_parts]):
            if k in cfg.empty_pockets:
                continue
            b = s.g.spawn(CUBE, [x, y, Z_POCKET + 1e-3])
            s.wc.add_part(b)
            self.parts.append(Part(b, pocket=k))
        self.n_pockets = min(cfg.n_parts, 4)
        self.next_pocket = 0
        q0 = ik_near(P_OUT, 0.0, Q_REF)
        s.sim.set_state(q0)
        s._hold(q0)
        s.g.obj = self.parts[0].body if self.parts else None
        s.wait(0.3)
        self.ex = Executor(s, frame_cb=self.frame_cb, cb_every=self.cb_every)
        static = [b for b in workcell_boxes() if b.name != "cube"] + machine_boxes("both")
        self.cc = CollisionChecker(static)
        self.chips: list[int] = []
        self.log: list = []  # 手臂活動區段 (tag, t0, t1)
        self.motion_log: list = []
        self.loads: list = []
        self.faults: list = []
        self.buffer = {"A": None, "B": None}
        self.vise_part: Part | None = None
        self.n_unloaded = 0
        self.held: Part | None = None
        self.out_parts: list[Part] = []
        self.door_clear: list = []
        self.belt_drops: list = []

    def close(self):
        self.cc.close()
        self.s.close()
        self._td.cleanup()

    # --- 工具
    @property
    def t(self) -> float:
        return self.s.t

    def _swap(self, m: Machine):
        """加工完成：胚料 → 成品（保留位姿；更新各處參照）。"""
        p = self.vise_part
        if p is None:
            return
        cid = self.s.cid
        pos, quat = pb.getBasePositionAndOrientation(p.body, physicsClientId=cid)
        old = p.body
        pb.removeBody(old, physicsClientId=cid)
        if old in self.s.wc.parts:
            self.s.wc.parts.remove(old)
        p.body = spawn_finished(cid, pos, quat)
        p.finished = True
        self.s.wc.add_part(p.body)
        m.part = p.body
        if self.s.g.obj == old:
            self.s.g.obj = p.body

    def seg(self, tag, t0):
        self.log.append((tag, t0, self.t))

    def wait_until(self, cond, timeout: float = 60.0, tag: str | None = None) -> bool:
        t0 = self.t
        while not cond():
            if self.t - t0 > timeout:
                return False
            self.s.step(10)
        self.ex.t = self.t
        if tag and self.t - t0 > 1e-9:
            self.seg(tag, t0)
        return True

    def goto(self, q_goal, tag: str) -> bool:
        """保證式直線檢查，不通則 Informed RRT*（第一解）＋捷徑；逐段 move_j。"""
        t0 = self.t
        q0 = self.s.sim.get_state()[0]
        q0w = np.asarray(wrap_pi(q0), float)
        off = q0 - q0w
        rec = {"tag": tag, "planned": False, "t_plan": 0.0, "t0": t0}
        path = [q0w, np.asarray(q_goal, float)]
        if not self.cc.edge_free(q0w, q_goal):
            t_plan = 0.0
            res = None
            for k in range(2):
                rrt = RRTStar(self.cc, seed=len(self.motion_log) + 1000 * k)
                res = rrt.plan(q0w, [q_goal], max_iter=5000, time_limit=self.cfg.rrt_time, stop_at_first=True)
                t_plan += res.t_plan
                if res.success:
                    break
            rec.update(planned=True, t_plan=t_plan, rrt_ok=res.success)
            if not res.success:
                rec["ok"] = False
                self.motion_log.append(rec)
                return False
            path = shortcut(self.cc, res.path, 100, seed=0)
            rec.update(raw_len=path_length(res.path), sc_len=path_length(path), n_seg=len(path) - 1)
        prog = [
            {"cmd": "move_j", "q": [float(x) for x in (q + off)], "id": f"{tag}{k}"}
            for k, q in enumerate(path[1:])
        ]
        rep = self.ex.run(prog)
        rec.update(ok=rep.ok, error=None if rep.ok else rep.error.code, results=rep.results)
        self.motion_log.append(rec)
        self.seg(tag, t0)
        return rep.ok

    def goto_tcp(self, xyz, tag: str) -> bool:
        q_now = np.asarray(wrap_pi(self.s.sim.get_state()[0]), float)
        q = ik_near(xyz, 0.0, q_now, sym=False, cc=self.cc)
        if q is None:
            self.motion_log.append({"tag": tag, "ok": False, "error": "NO_FREE_IK"})
            return False
        return self.goto(q, tag)

    def lin(self, xyz, tag: str) -> bool:
        """TCP 直線。先試峰值速度 V_LIN 的時間 T = 1.875 L / V_LIN（五次多項式峰速 = 1.875 L/T）；
        若執行器判定超限（LIMIT，尚未運動即拋出）則退回執行器預設時間縮放。"""
        t0 = self.t
        L = float(np.linalg.norm(np.asarray(xyz, float) - self.tcp()))
        cmd = {"cmd": "move_l", "tcp": [float(v) for v in xyz], "id": tag}
        rep = self.ex.run([{**cmd, "T": max(1.875 * L / V_LIN, 0.3)}]) if L > 1e-4 else None
        if rep is None or (not rep.ok and rep.error.code == "LIMIT"):
            rep = self.ex.run([cmd])
        self.motion_log.append(
            {
                "tag": tag,
                "planned": False,
                "ok": rep.ok,
                "results": rep.results,
                "error": None if rep.ok else rep.error.code,
                "t0": t0,
            }
        )
        self.seg(tag, t0)
        return rep.ok

    def tcp(self) -> np.ndarray:
        return self.ex.tcp_of(self.s.sim.get_state()[0])

    def grip(self, part: Part, tag="grip") -> bool:
        t0 = self.t
        self.s.g.obj = part.body
        rep = self.ex.run([{"cmd": "grip", "force": self.cfg.force, "id": tag}])
        self.seg(tag, t0)
        if rep.ok:
            self.held = part
            self.cc.attach([PART_A / 2] * 3, self._T_tcp_obj(part))
        return rep.ok

    def release(self, tag="release"):
        t0 = self.t
        self.ex.run([{"cmd": "release", "id": tag}])
        self.held = None
        self.cc.detach()
        self.seg(tag, t0)

    def _T_tcp_obj(self, part: Part) -> np.ndarray:
        """夾持件相對 TCP 的位姿（工具朝下 ψ=0：R_tcp = diag(1,−1,−1)）。"""
        R_tcp = np.diag([1.0, -1.0, -1.0])
        c = part_center(self.s.cid, part.body, part.finished)
        T = np.eye(4)
        T[:3, 3] = R_tcp.T @ (c - self.tcp())
        return T

    def part_pos(self, part: Part) -> np.ndarray:
        return part_center(self.s.cid, part.body, part.finished)

    # --- 動作單元
    def pick_tray(self) -> Part | None:
        """依序取料盤胚料；空格（GRIP_EMPTY）→ 記錄並跳下一格。"""
        while self.next_pocket < self.n_pockets:
            k = self.next_pocket
            self.next_pocket += 1
            x, y = S.pocket_centers()[k]
            z_g = Z_POCKET + grasp_offset(PART_A)
            if not self.goto_tcp((x, y, Z_TRAVEL), "to_tray"):
                return None
            self.lin((x, y, z_g), "tray_down")
            part = next((p for p in self.parts if p.pocket == k), None)
            t0 = self.t
            self.s.g.obj = part.body if part else (self.parts[0].body if self.parts else None)
            rep = self.ex.run([{"cmd": "grip", "force": self.cfg.force, "id": "grip_tray"}])
            self.seg("grip_tray", t0)
            if not rep.ok:
                self.faults.append({"t": self.t, "fault": "GRIP_EMPTY", "pocket": k, "action": "skip pocket"})
                self.ex.run([{"cmd": "release", "id": "reopen"}])
                self.lin((x, y, Z_TRAVEL), "tray_up")
                continue
            self.held = part
            self.cc.attach([PART_A / 2] * 3, self._T_tcp_obj(part))
            self.lin((x, y, Z_TRAVEL), "tray_up")
            return part
        return None

    def place_at(self, xy, z_floor, tag) -> bool:
        """放到平面（底面高 z_floor）上，回到搬運高度。"""
        z_g = z_floor + PART_A / 2 + grasp_offset(PART_A)
        z_up = max(Z_TRAVEL, z_floor + 0.12)
        if not self.goto_tcp((*xy, z_up), f"to_{tag}"):
            return False
        ok = self.lin((*xy, z_g), f"{tag}_down")
        self.release(f"{tag}_release")
        ok = self.lin((*xy, z_up), f"{tag}_up") and ok
        return ok

    def pick_at(self, part: Part, xy, z_floor, tag) -> bool:
        z_g = z_floor + PART_A / 2 + grasp_offset(PART_A)
        z_up = max(Z_TRAVEL, z_floor + 0.12)
        if not self.goto_tcp((*xy, z_up), f"to_{tag}"):
            return False
        self.lin((*xy, z_g), f"{tag}_down")
        if not self.grip(part, f"{tag}_grip"):
            return False
        return self.lin((*xy, z_up), f"{tag}_up")

    def to_door(self) -> bool:
        return self.goto_tcp(P_OUT, "to_door")

    def enter(self) -> bool:
        """門未全開不進入（手臂側互鎖）；門外點 → 虎鉗上方（move_l）。"""
        if not self.m.sig("door_open"):
            self.wait_until(lambda: self.m.sig("door_open"), 30.0, "wait_door")
        return self.lin(P_IN, "enter")

    def exit(self) -> bool:
        ok = self.lin(P_OUT, "exit")
        return ok

    def close_and_start(self) -> bool:
        """關門＋預約啟動（門全關時 PLC 自動啟動）；手臂不等關門完成。"""
        if not self.m.close_door():
            return False
        self.m.request_start()
        return True

    def load(self, part: Part) -> dict:
        """手臂夾著胚料在門外 → 上料、夾緊、檢知（失敗時恢復）→ 退出。"""
        m = self.m
        rec = {"t0": self.t, "retries": 0}
        self.enter()
        dx, dy = self.cfg.place_err
        xy = (X_PLACE + dx, VY + dy)
        z_g = Z_PART + grasp_offset(PART_A)
        self.lin((*xy, Z_ENTRY), "over_vise") if (dx or dy) else None
        attempt = 0
        while True:
            self.lin(
                (*xy, z_g + 0.0015), "vise_down"
            )  # 底面離座面 1.5 mm 時鬆爪（避免手臂把零件壓在座面/切屑上）
            p_rel = self.part_pos(part)
            self.release("vise_release")
            m.clamp(part.body)
            self.vise_part = part
            self.lin((*xy, z_g + 0.05), "vise_up")
            self.wait_until(lambda: m.sig("clamped"), 5.0, "wait_clamp")
            chk = m.check()
            pc = self.part_pos(part)
            rec.setdefault("checks", []).append(chk)
            rec.update(
                release_pos=p_rel.tolist(),
                clamped_pos=pc.tolist(),
                err_x_mm=(pc[0] - VX) * 1e3,
                err_y_mm=(pc[1] - VY) * 1e3,
                err_z_mm=(pc[2] - Z_PART) * 1e3,
                shift_mm=(pc[0] - p_rel[0]) * 1e3,
            )
            if chk["ok"] or not self.cfg.recover or attempt >= self.cfg.max_retry:
                break
            # 恢復：鬆鉗 → 夾回 → 抬起 → 吹氣（移除切屑）→ 重放
            attempt += 1
            rec["retries"] = attempt
            self.faults.append(
                {"t": self.t, "fault": "NOT_SEATED", "check": chk, "action": "re-place + air blow"}
            )
            m.unclamp()
            self.wait_until(m.unclamped, 5.0, "wait_unclamp")
            pg = self.part_pos(part)
            xy = (pg[0], pg[1])
            self.lin((*xy, pg[2] + grasp_offset(PART_A)), "regrip_down")
            self.grip(part, "regrip")
            self.lin((*xy, z_g + 0.06), "regrip_up")
            t0 = self.t
            for c in self.chips:
                pb.removeBody(c, physicsClientId=self.s.cid)
            n_chips = len(self.chips)
            self.chips.clear()
            self.s.wait(1.0)  # 吹氣 1 s
            self.ex.t = self.t
            self.seg("air_blow", t0)
            self.faults[-1]["chips_removed"] = n_chips
            xy = (X_PLACE, VY)
            self.lin((*xy, z_g + 0.06), "over_vise")
        rec["ok"] = bool(rec["checks"][-1]["ok"])
        self.lin((*xy, Z_ENTRY), "vise_clear")
        self.exit()
        rec["t1"] = self.t
        self.loads.append(rec)
        return rec

    def unload(self) -> Part | None:
        m = self.m
        part = self.vise_part
        if not m.done:
            self.wait_until(
                lambda: (
                    m.done
                    or m.trip
                    or (not m.start_armed and m.cycle is None and not m.done and m.sig("door_closed"))
                ),
                600.0,
                "wait_machine",
            )
        if not m.done:  # 光幕遮斷或啟動被互鎖拒絕
            return None
        m.open_door()
        self.enter()
        pc = self.part_pos(part)
        self.lin((pc[0], pc[1], Z_ENTRY), "over_part")
        self.lin((pc[0], pc[1], pc[2] + grasp_offset(PART_A)), "unload_down")
        if not self.grip(part, "unload_grip"):
            return None
        m.unclamp()
        self.wait_until(m.unclamped, 5.0, "wait_unclamp")
        m.part = None
        self.vise_part = None
        m.done = False
        m.set_sig("cycle_done", False)
        self.lin((pc[0], pc[1], Z_ENTRY), "unload_up")
        self.n_unloaded += 1
        if self.n_unloaded in self.cfg.chip_at:  # 故障注入：切屑留在座面（零件取走後落下）
            cid = self.s.cid
            h = [0.006, 0.006, 0.0015]
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=h, physicsClientId=cid)
            vis = pb.createVisualShape(
                pb.GEOM_BOX, halfExtents=h, rgbaColor=(0.95, 0.85, 0.3, 1), physicsClientId=cid
            )
            c = pb.createMultiBody(
                0.002, col, vis, [VX - 0.012, VY + 0.010, VISE_FLOOR + 0.0016], physicsClientId=cid
            )
            pb.setCollisionFilterGroupMask(c, -1, S.GROUPS["part"], S.MASKS["part"], physicsClientId=cid)
            self.chips.append(c)
            self.faults.append({"t": self.t, "fault": "CHIP_INJECTED", "after_unload": self.n_unloaded})
        self.exit()
        return part

    def to_belt(self, part: Part) -> bool:
        ok = self.place_at(BELT_DROP, S.BENCH_TOP, "belt")
        self.out_parts.append(part)
        self.belt_drops.append({"t": self.t, "pos": self.part_pos(part).tolist()})
        return ok

    def wait_door_open_cmd(self):
        if not self.m.sig("door_open") and not self.m.door_moving:
            self.m.open_door()

    # --- 策略
    def run(self) -> dict:
        cfg = self.cfg
        m = self.m
        t_start = self.t
        m.open_door()
        part = self.pick_tray()
        if part is None:
            return self.metrics(t_start)
        self.to_door()
        self.load(part)
        self.close_and_start()
        while True:
            if cfg.strategy == "serial":
                fin = self.unload()
                if fin is None:
                    break
                self.to_belt(fin)
                raw = self.pick_tray()
                if raw is None:
                    break
                self.to_door()
                self.load(raw)
                if not self.close_and_start():
                    break
            else:  # buffer
                # 加工期間：緩衝 A → 輸送帶；料盤 → 緩衝 B；回門外
                t0 = self.t
                if self.buffer["A"] is not None:
                    p = self.buffer["A"]
                    self.pick_at(p, BUF_SLOTS["A"], BUF_TOP, "bufA")
                    self.to_belt(p)
                    self.buffer["A"] = None
                raw = self.pick_tray()
                if raw is not None:
                    self.place_at(BUF_SLOTS["B"], BUF_TOP, "bufB")
                    self.buffer["B"] = raw
                self.to_door()
                self.log.append(("side_work", t0, self.t))
                fin = self.unload()
                if fin is None:
                    break
                self.place_at(BUF_SLOTS["A"], BUF_TOP, "bufA")
                self.buffer["A"] = fin
                if self.buffer["B"] is None:  # 無胚料 → 收尾：成品送出
                    self.pick_at(fin, BUF_SLOTS["A"], BUF_TOP, "bufA")
                    self.to_belt(fin)
                    self.buffer["A"] = None
                    break
                raw = self.buffer["B"]
                self.pick_at(raw, BUF_SLOTS["B"], BUF_TOP, "bufB")
                self.buffer["B"] = None
                self.to_door()
                self.load(raw)
                if not self.close_and_start():
                    break
        self.s.wait(2.0)
        return self.metrics(t_start)

    # --- 量測
    def cycle_events(self) -> dict:
        ev = self.m.events
        starts = [e["t"] for e in ev if e["event"] == "CYCLE_START"]
        dones = [e["t"] for e in ev if e["event"] == "CYCLE_DONE"]
        return {"starts": starts, "dones": dones}

    def metrics(self, t_start: float) -> dict:
        ce = self.cycle_events()
        st, dn = ce["starts"], ce["dones"]
        periods = np.diff(st).tolist() if len(st) > 1 else []
        idle = [st[k + 1] - dn[k] for k in range(min(len(dn), len(st) - 1))]  # 循環結束 → 下一次啟動
        T_tot = self.t - t_start
        mv = [
            r
            for m in self.motion_log
            for r in m.get("results", [])
            if r.get("status") == "ok" and r["cmd"].startswith("move")
        ]
        cid = self.s.cid
        outs = []
        for p in self.out_parts:
            pos = self.part_pos(p)
            outs.append(
                {
                    "finished": p.finished,
                    "pos": pos.tolist(),
                    "on_belt": bool(
                        S.BELT_X0 <= pos[0] <= S.BELT_X1 + 0.01
                        and S.BELT_Y0 <= pos[1] <= S.BELT_Y1
                        and pos[2] < S.BENCH_TOP + PART_A
                    ),
                }
            )
        busy = sum(self.cfg.T_m for _ in dn)
        return {
            "strategy": self.cfg.strategy,
            "T_m": self.cfg.T_m,
            "n_cycles": len(dn),
            "n_out": len(self.out_parts),
            "outputs": outs,
            "all_ok": bool(
                len(dn) == len(self.parts)
                and all(o["finished"] and o["on_belt"] for o in outs)
                and len(outs) == len(self.parts)
            ),
            "T_total": T_tot,
            "periods": periods,
            "machine_idle": idle,
            "utilization": busy / (st[-1] + self.cfg.T_m - st[0]) if st else 0.0,
            "starts": st,
            "dones": dn,
            "loads": self.loads,
            "faults": self.faults,
            "events": self.m.events,
            "signals": [(t, n, bool(v)) for t, n, v in self.m.signals],
            "segments": self.log,
            "torque_util": max((r.get("torque_ratio", 0.0) for r in mv), default=0.0),
            "vel_util": max((r.get("vel_ratio", 0.0) for r in mv), default=0.0),
            "n_planned": sum(1 for m in self.motion_log if m.get("planned")),
            "t_plan": sum(m.get("t_plan", 0.0) for m in self.motion_log),
            "errors": [m.get("error") for m in self.motion_log if m.get("error")],
            "cut_slip_mm": max(self.m.cut_log, default=0.0) * 1e3,
            "contacts_arm_machine": _arm_machine_contacts(cid, self.s.sim.id, self.s.wc.bodies),
        }


def _arm_machine_contacts(cid, arm, bodies) -> int:
    return sum(
        len(pb.getContactPoints(arm, b, physicsClientId=cid))
        for n, b in bodies.items()
        if n.startswith(("m_", "door", "jaw", "vise", "spindle"))
    )


# ---------------------------------------------------------------------- 週期模型
def cycle_model(T_m, T_open: float, T_ex: float, T_close: float, T_side: float = 0.0) -> dict:
    """機台照顧週期（穩態，每件）：
        T_c = T_open + T_ex + max(T_close + T_m, T_side)
    T_open/T_close：開/關門時間；T_ex：門全開 → 關門命令（手臂在機台區內外的交換工作，機台必閒置）；
    T_side：關門命令 → 手臂回到門外待命（加工期間的手臂側工作，S1 為 0）。
    T_close + T_m ≥ T_side 為機台瓶頸（手臂等機台），否則為手臂瓶頸。回傳 T_c、稼動率 U = T_m/T_c、瓶頸。"""
    T_m = np.asarray(T_m, float)
    Tc = T_open + T_ex + np.maximum(T_close + T_m, T_side)
    return {"T_c": Tc, "U": T_m / Tc, "robot_bound": T_side > T_close + T_m}


def period_terms(events: list, segments: list) -> list[dict]:
    """由事件/區段紀錄量出每個穩態週期的模型分量（第 k 件 CYCLE_DONE → 第 k+1 件 CYCLE_START）。"""
    ev = lambda kind: [e["t"] for e in events if e["event"] == kind]  # noqa: E731
    done, start = ev("CYCLE_DONE"), ev("CYCLE_START")
    ocmd, ccmd = ev("DOOR_OPEN_CMD"), ev("DOOR_CLOSE_CMD")
    sides = [s for s in segments if s[0] == "side_work"]
    out = []
    for k in range(min(len(done), len(start) - 1)):
        td, ts_next = done[k], start[k + 1]
        to = next(t for t in ocmd if t >= td - 1e-9)
        tc = next(t for t in ccmd if t > to)
        tc_prev = max(t for t in ccmd if t < start[k] + 1e-9)
        side = next((s[2] - s[1] for s in sides if tc_prev - 1e-6 <= s[1] < td), 0.0)
        out.append(
            {
                "k": k + 1,
                "period": start[k + 1] - start[k],
                "T_wait_open": to - td,
                "T_open": door_time(),
                "T_ex": tc - (to + door_time()),
                "T_close": ts_next - tc,
                "T_side": side,
                "machine_idle": ts_next - td,
            }
        )
    return out


# ---------------------------------------------------------------------- 夾持力實驗（無手臂）
def clamp_hold_test(
    F_c: float,
    rate: float = 20.0,
    F_max: float = 400.0,
    slip_mm: float = 0.5,
    dt: float = 1e-3,
    gui_cid: int | None = None,
) -> dict:
    """虎鉗單獨測試：放入胚料 → 以 F_c 夾緊 → 側向力 F_y 以 rate [N/s] 線性增加，記錄零件 y 位移；
    位移超過 slip_mm 時的力 = 實測滑移門檻（與 slip_threshold 比較）。"""
    cid = pb.connect(pb.DIRECT)
    try:
        pb.setGravity(0, 0, -G, physicsClientId=cid)
        pb.setTimeStep(dt, physicsClientId=cid)
        pb.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=cid)

        class _WC:
            bodies: dict = {}

            def _box(self, name, center, half, rgba, group="static", friction=0.8):
                col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(half), physicsClientId=cid)
                b = pb.createMultiBody(0, col, -1, [float(c) for c in center], physicsClientId=cid)
                pb.changeDynamics(b, -1, lateralFriction=friction, physicsClientId=cid)
                self.bodies[name] = b
                return b

        wc = _WC()
        wc.bodies = {}
        dummy = pb.createMultiBody(0, -1, -1, [0, 0, -5], physicsClientId=cid)
        m = Machine(cid, wc, dummy, F_c=F_c)
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[PART_A / 2] * 3, physicsClientId=cid)
        part = pb.createMultiBody(CUBE.mass, col, -1, [X_PLACE, VY, Z_PART + 5e-4], physicsClientId=cid)
        pb.changeDynamics(part, -1, lateralFriction=CUBE.friction, frictionAnchor=1, physicsClientId=cid)

        def step(n=1, Fy=0.0):
            for _ in range(n):
                if Fy:
                    pos = pb.getBasePositionAndOrientation(part, physicsClientId=cid)[0]
                    pb.applyExternalForce(part, -1, [0, Fy, 0], pos, pb.WORLD_FRAME, physicsClientId=cid)
                m.tick(dt)
                pb.stepSimulation(physicsClientId=cid)

        step(200)
        m.clamp(part)
        step(1500)
        p0 = np.array(pb.getBasePositionAndOrientation(part, physicsClientId=cid)[0])
        jaw_f = m.jaw_force()
        chk = m.check()
        F = 0.0
        F_slip = None
        trace = []
        while F < F_max:
            F += rate * dt
            step(1, F)
            p = np.array(pb.getBasePositionAndOrientation(part, physicsClientId=cid)[0])
            dy = (p[1] - p0[1]) * 1e3
            if len(trace) == 0 or abs(F - trace[-1][0]) >= 1.0:
                trace.append((F, dy))
            if abs(dy) > slip_mm:
                F_slip = F
                break
        return {
            "F_c": F_c,
            "F_slip": F_slip,
            "F_pred": slip_threshold(F_c),
            "jaw_force": jaw_f,
            "clamped_x_err_mm": (p0[0] - VX) * 1e3,
            "check": chk,
            "trace": trace,
        }
    finally:
        pb.disconnect(cid)


def vise_capture_test(dx: float, dy: float = 0.0, dt: float = 1e-3, chip: bool = False) -> dict:
    """虎鉗定位測試（無手臂）：胚料以 (X_PLACE+dx, VY+dy) 由鉗口上方下降到鬆爪高度
    （底面離座面 1.5 mm）→ 放開 → 夾緊 → 量測最終位置與監控結果。
    |dx| 超過開口間隙（JAW_OPEN/2 = 6 mm）時零件會落在鉗口頂上。"""
    cid = pb.connect(pb.DIRECT)
    try:
        pb.setGravity(0, 0, -G, physicsClientId=cid)
        pb.setTimeStep(dt, physicsClientId=cid)
        pb.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=cid)

        class _WC:
            def __init__(self):
                self.bodies = {}

            def _box(self, name, center, half, rgba, group="static", friction=0.8):
                col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(half), physicsClientId=cid)
                b = pb.createMultiBody(0, col, -1, [float(c) for c in center], physicsClientId=cid)
                pb.changeDynamics(b, -1, lateralFriction=friction, physicsClientId=cid)
                self.bodies[name] = b
                return b

        wc = _WC()
        dummy = pb.createMultiBody(0, -1, -1, [0, 0, -5], physicsClientId=cid)
        m = Machine(cid, wc, dummy)
        if chip:
            h = [0.006, 0.006, 0.0015]
            c = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=h, physicsClientId=cid)
            pb.createMultiBody(
                0.002, c, -1, [VX - 0.012, VY + 0.010, VISE_FLOOR + 0.0016], physicsClientId=cid
            )
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[PART_A / 2] * 3, physicsClientId=cid)
        z_rel = Z_PART + 0.0015 + (0.003 if chip else 0.0)  # 鬆爪高度（同 TendCell.load）
        z0 = Z_PART + JAW_H + 0.01
        part = pb.createMultiBody(CUBE.mass, col, -1, [X_PLACE + dx, VY + dy, z0], physicsClientId=cid)
        pb.changeDynamics(part, -1, lateralFriction=CUBE.friction, frictionAnchor=1, physicsClientId=cid)
        # 夾爪下降的等效：固定約束（最大力 = 夾持摩擦上限 2μF ≈ 64 N）以 50 mm/s 拉到鬆爪高度後解除
        con = pb.createConstraint(
            part,
            -1,
            -1,
            -1,
            pb.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [X_PLACE + dx, VY + dy, z0],
            physicsClientId=cid,
        )
        n = int(round((z0 - z_rel) / 0.05 / dt))
        for k in range(1, n + 1):
            z = z0 + (z_rel - z0) * k / n
            pb.changeConstraint(con, [X_PLACE + dx, VY + dy, z], maxForce=2 * 0.8 * 40.0, physicsClientId=cid)
            m.tick(dt)
            pb.stepSimulation(physicsClientId=cid)
        pb.removeConstraint(con, physicsClientId=cid)
        for _ in range(400):
            m.tick(dt)
            pb.stepSimulation(physicsClientId=cid)
        m.clamp(part)
        for _ in range(1500):
            m.tick(dt)
            pb.stepSimulation(physicsClientId=cid)
        pos = np.array(pb.getBasePositionAndOrientation(part, physicsClientId=cid)[0])
        chk = m.check()
        return {
            "dx_mm": dx * 1e3,
            "dy_mm": dy * 1e3,
            "x_err_mm": (pos[0] - VX) * 1e3,
            "y_err_mm": (pos[1] - VY) * 1e3,
            "z_err_mm": (pos[2] - Z_PART) * 1e3,
            "clamped": m.sig("clamped"),
            **chk,
        }
    finally:
        pb.disconnect(cid)


__all__ = [
    "F_CUT",
    "Machine",
    "ObjectSpec",
    "P_IN",
    "P_OUT",
    "TendCell",
    "TendConfig",
    "clamp_hold_test",
    "cycle_model",
    "door_pos",
    "period_terms",
    "door_time",
    "finished_props",
    "in_machine_zone",
    "machine_boxes",
    "slip_threshold",
    "spawn_finished",
    "vise_capture_test",
]
