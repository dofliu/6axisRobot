"""平行夾爪與可抓物件模型（T005）。

夾爪（類 Robotiq 2F-85 尺寸）：掌體 + 兩根對向 prismatic 指（沿夾爪 y 軸），夾爪座標系 = 法蘭座標系
（z 為工具方向，與 FK 的 T06 一致）。指位置 s 為單指閉合行程（0 = 全開，間距 80 mm；S_MAX = 全閉）。

抓持力學（庫侖摩擦，兩指對夾）：保持物重需要  2·μ·F_n ≥ m·(g + a)  →  F_min = m(g+a)/(2μ)
（見 Siciliano《Robotics: Modelling, Planning and Control》第 7 章抓持；Murray/Li/Sastry 1994 第 5 章）。
PyBullet 以兩接觸體 lateralFriction 之「乘積」作為有效摩擦係數，因此 μ_eff = μ_pad · μ_obj。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as pb

from .pbsim import FLANGE, PBRobot
from .robot import G, SixAxisRobot
from .urdf import write_urdf

ARM_GRIPPER_URDF = Path(__file__).resolve().parents[1] / "robots" / "ur5_gripper.urdf"

PALM_MASS = 0.45
FINGER_MASS = 0.05
W0 = 0.045  # 全開時指墊中心離中線距離（內表面 0.040 → 開口 80 mm）
PAD_T = 0.010  # 指墊厚度（y）
PAD_W = 0.020  # 指墊寬度（x）
PAD_L = 0.040  # 指墊長度（z）
PAD_Z0 = 0.075  # 指墊起點（離法蘭）
TCP_Z = PAD_Z0 + PAD_L / 2  # 夾持中心（離法蘭）
S_MAX = 0.040  # 單指最大行程
F_MAX = 235.0  # 單指額定夾持力 [N]（2F-85 規格量級）

# 夾持中心相對法蘭的齊次轉換
T_TCP = np.eye(4)
T_TCP[2, 3] = TCP_Z


def gripper_xml(parent: str = "flange") -> str:
    """夾爪連桿/關節 XML 片段：gripper_base 以固定關節接在 parent（法蘭）下，兩指為 prismatic。"""

    def link(name, mass, parts):
        s = f'  <link name="{name}"><inertial><mass value="{mass}"/>'
        s += '<origin xyz="0 0 0.03"/><inertia ixx="1e-4" iyy="1e-4" izz="5e-5" '
        s += 'ixy="0" ixz="0" iyz="0"/></inertial>'
        for sz, xyz, mat, col in parts:
            g = f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}"/>'
            g += f'<geometry><box size="{sz[0]} {sz[1]} {sz[2]}"/></geometry>'
            s += f'<visual>{g}<material name="{mat}"/></visual>'
            if col:
                s += f"<collision>{g}</collision>"
        return s + "</link>\n"

    pz = PAD_Z0 + PAD_L / 2
    U = '  <material name="gbody"><color rgba="0.12 0.12 0.14 1"/></material>\n'
    U += '  <material name="gfinger"><color rgba="0.75 0.77 0.80 1"/></material>\n'
    U += '  <material name="gpad"><color rgba="0.9 0.35 0.1 1"/></material>\n'
    U += link(
        "gripper_base",
        PALM_MASS,
        [
            ((0.075, 0.110, 0.030), (0, 0, 0.015), "gbody", True),
            ((0.060, 0.100, 0.020), (0, 0, 0.040), "gbody", True),
        ],
    )
    for nm, sgn in (("finger_l", +1), ("finger_r", -1)):
        U += link(
            nm,
            FINGER_MASS,
            [
                ((PAD_W, 0.014, 0.060), (0, 0.012 * sgn, 0.055), "gfinger", False),
                ((PAD_W, PAD_T, PAD_L), (0, 0, pz), "gpad", True),
            ],
        )
    U += f'  <joint name="gripper_mount" type="fixed"><parent link="{parent}"/>'
    U += '<child link="gripper_base"/></joint>\n'
    for nm, sgn in (("finger_l", +1), ("finger_r", -1)):
        U += (
            f'  <joint name="{nm}_joint" type="prismatic"><parent link="gripper_base"/><child link="{nm}"/>'
            f'<origin xyz="0 {W0 * sgn} 0"/><axis xyz="0 {-sgn} 0"/>'
            f'<limit lower="0" upper="{S_MAX}" effort="{F_MAX}" velocity="0.15"/></joint>\n'
        )
    return U


def write_arm_with_gripper(robot: SixAxisRobot | None = None, path: str | Path = ARM_GRIPPER_URDF) -> Path:
    """手臂 URDF + 夾爪（同一 multibody，避免約束彈性）。連桿 index：7 = gripper_base，8/9 = 左/右指。"""
    robot = robot or SixAxisRobot()
    write_urdf(robot, path)  # 同時寫出視覺網格
    path = Path(path)
    txt = path.read_text(encoding="utf-8")
    path.write_text(txt.replace("</robot>", gripper_xml() + "</robot>"), encoding="utf-8")
    return path


@dataclass(frozen=True)
class ObjectSpec:
    """可抓物件：形狀 'box'（size=(x,y,z)）或 'cylinder'（size=(半徑, 高)）。"""

    name: str
    shape: str
    size: tuple
    mass: float
    friction: float = 0.8
    rgba: tuple = (0.85, 0.2, 0.15, 1.0)

    @property
    def grasp_width(self) -> float:
        """沿夾爪 y 軸的夾持寬度（物件不旋轉時）。"""
        return self.size[1] if self.shape == "box" else 2 * self.size[0]

    @property
    def height(self) -> float:
        return self.size[2] if self.shape == "box" else self.size[1]


CUBE = ObjectSpec("cube50", "box", (0.05, 0.05, 0.05), 0.20)
BLOCK = ObjectSpec("block", "box", (0.04, 0.06, 0.08), 0.30, rgba=(0.2, 0.45, 0.85, 1.0))
CYLINDER = ObjectSpec("cyl", "cylinder", (0.025, 0.06), 0.15, rgba=(0.2, 0.7, 0.35, 1.0))


def f_min(mass: float, mu: float, accel: float = 0.0) -> float:
    """保持物件所需單指法向力 [N]：F = m(g+a)/(2μ)。"""
    return mass * (G + accel) / (2.0 * mu)


def make_arm_with_gripper(robot: SixAxisRobot | None = None, plane: bool = False) -> PBRobot:
    path = write_arm_with_gripper(robot)
    return PBRobot(robot, urdf=path, plane=plane)


BASE_LINK, FINGER_LINKS = 7, [8, 9]
GEAR_RATIO = -1.0


class Gripper:
    """操作已含夾爪的 PBRobot（見 make_arm_with_gripper）：指位置/力控與物件生成。"""

    PAD_FRICTION = 1.0

    def __init__(self, pbr: PBRobot):
        self.pbr = pbr
        self.cid = pbr.cid
        self.id = pbr.id
        self.fingers = FINGER_LINKS
        for link in (BASE_LINK, *FINGER_LINKS):
            pb.changeDynamics(self.id, link, lateralFriction=self.PAD_FRICTION, frictionAnchor=1,
                              linearDamping=0.0, angularDamping=0.0, physicsClientId=self.cid)
        # 兩指機構連動（齒條/連桿等效）：s_l = s_r，否則等力馬達下物件可被推向一側
        self.gear = pb.createConstraint(self.id, FINGER_LINKS[0], self.id, FINGER_LINKS[1], pb.JOINT_GEAR,
                                        [0, 1, 0], [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        pb.changeConstraint(self.gear, gearRatio=GEAR_RATIO, maxForce=2000.0, erp=0.5,
                            physicsClientId=self.cid)
        self.obj = None
        self.obj_spec = None

    # --- 手指 ---
    def finger_pos(self) -> np.ndarray:
        s = pb.getJointStates(self.id, self.fingers, physicsClientId=self.cid)
        return np.array([x[0] for x in s])

    def command(self, target: float, force: float):
        """兩指位置控制，馬達力上限 = force（接觸後法向力 ≈ force）。"""
        pb.setJointMotorControlArray(
            self.id, self.fingers, pb.POSITION_CONTROL, targetPositions=[target, target],
            forces=[force, force], targetVelocities=[0.0, 0.0],
            positionGains=[0.2, 0.2], velocityGains=[1.0, 1.0], physicsClientId=self.cid,
        )

    def open(self, force: float = 50.0):
        self.command(0.0, force)

    def close(self, force: float):
        self.command(S_MAX, force)

    # --- 物件 ---
    def spawn(self, spec: ObjectSpec, pos, quat=(0, 0, 0, 1)) -> int:
        if spec.shape == "box":
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[s / 2 for s in spec.size],
                                          physicsClientId=self.cid)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[s / 2 for s in spec.size],
                                       rgbaColor=spec.rgba, physicsClientId=self.cid)
        else:
            col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=spec.size[0], height=spec.size[1],
                                          physicsClientId=self.cid)
            vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=spec.size[0], length=spec.size[1],
                                       rgbaColor=spec.rgba, physicsClientId=self.cid)
        self.obj = pb.createMultiBody(spec.mass, col, vis, list(map(float, pos)), list(quat),
                                      physicsClientId=self.cid)
        pb.changeDynamics(self.obj, -1, lateralFriction=spec.friction, frictionAnchor=1,
                          linearDamping=0.0, angularDamping=0.0,
                          physicsClientId=self.cid)
        self.obj_spec = spec
        return self.obj

    def obj_pose(self):
        p, q = pb.getBasePositionAndOrientation(self.obj, physicsClientId=self.cid)
        return np.array(p), np.array(q)

    def tcp_pos(self) -> np.ndarray:
        s = pb.getLinkState(self.pbr.id, FLANGE, computeForwardKinematics=1, physicsClientId=self.cid)
        R = np.array(pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)
        return np.array(s[4]) + R @ T_TCP[:3, 3]

    def contact_forces(self) -> tuple[float, float]:
        """左/右指墊對物件的法向力總和 [N]。"""
        out = []
        for lk in self.fingers:
            c = pb.getContactPoints(self.id, self.obj, lk, -1, physicsClientId=self.cid)
            out.append(float(sum(x[9] for x in c)))
        return out[0], out[1]
