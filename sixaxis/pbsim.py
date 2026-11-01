"""PyBullet 介面：載入 URDF、設定/讀取狀態、逆動力學、自由擺動、離屏多視角渲染。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as pb
import pybullet_data

from .robot import G, SixAxisRobot
from .urdf import write_urdf

URDF_PATH = Path(__file__).resolve().parents[1] / "robots" / "ur5_like.urdf"
FLANGE = 6  # 固定法蘭連桿 index


class PBRobot:
    def __init__(self, robot: SixAxisRobot | None = None, urdf: Path | None = None, plane: bool = False):
        self.robot = robot or SixAxisRobot()
        self.cid = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        pb.setGravity(0, 0, -G, physicsClientId=self.cid)
        urdf = Path(urdf) if urdf else URDF_PATH
        if not urdf.exists():
            write_urdf(self.robot, urdf)
        self.plane = pb.loadURDF("plane.urdf", physicsClientId=self.cid) if plane else None
        self.id = pb.loadURDF(
            str(urdf),
            useFixedBase=True,
            flags=pb.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.cid,
        )
        self.joints = list(range(6))

    def close(self):
        if pb.isConnected(self.cid):
            pb.disconnect(self.cid)

    # --- 狀態 ---
    def set_state(self, q, qd=None):
        qd = np.zeros(6) if qd is None else qd
        for j in self.joints:
            pb.resetJointState(self.id, j, float(q[j]), float(qd[j]), physicsClientId=self.cid)

    def get_state(self):
        s = pb.getJointStates(self.id, self.joints, physicsClientId=self.cid)
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])

    def link_pose(self, link: int = FLANGE):
        """URDF 連桿座標系位姿 (4x4)。link=0..5 為 link1..6，6 為法蘭。"""
        s = pb.getLinkState(self.id, link, computeForwardKinematics=1, physicsClientId=self.cid)
        T = np.eye(4)
        T[:3, :3] = np.array(pb.getMatrixFromQuaternion(s[5])).reshape(3, 3)
        T[:3, 3] = s[4]
        return T

    def inverse_dynamics(self, q, qd, qdd):
        return np.array(
            pb.calculateInverseDynamics(
                self.id,
                list(map(float, q)),
                list(map(float, qd)),
                list(map(float, qdd)),
                physicsClientId=self.cid,
            )
        )

    # --- 無摩擦自由擺動 / 力矩控制 ---
    def make_frictionless(self):
        """零阻尼、零馬達力、放寬速度與限位。"""
        for link in range(-1, 7):
            pb.changeDynamics(
                self.id,
                link,
                linearDamping=0.0,
                angularDamping=0.0,
                jointDamping=0.0,
                maxJointVelocity=1000.0,
                physicsClientId=self.cid,
            )
        # 放寬 PyBullet 關節限位（否則撞限位會損失能量，與動力學模型無關）
        for j in self.joints:
            pb.changeDynamics(
                self.id, j, jointLowerLimit=-1e4, jointUpperLimit=1e4, physicsClientId=self.cid
            )
        pb.setJointMotorControlArray(
            self.id, self.joints, pb.VELOCITY_CONTROL, forces=[0.0] * 6, physicsClientId=self.cid
        )

    def free_swing(self, q0, qd0, dt: float, steps: int, sample_every: int = 1):
        """PyBullet 零力矩模擬，回傳 (t, q, qd)。"""
        self.make_frictionless()
        pb.setTimeStep(dt, physicsClientId=self.cid)
        self.set_state(q0, qd0)
        ts, qs, qds = [0.0], [np.array(q0, float)], [np.array(qd0, float)]
        for k in range(1, steps + 1):
            pb.stepSimulation(physicsClientId=self.cid)
            if k % sample_every == 0:
                q, qd = self.get_state()
                ts.append(k * dt)
                qs.append(q)
                qds.append(qd)
        return np.array(ts), np.array(qs), np.array(qds)

    def apply_torques(self, tau):
        pb.setJointMotorControlArray(
            self.id, self.joints, pb.TORQUE_CONTROL, forces=list(map(float, tau)), physicsClientId=self.cid
        )

    def step(self):
        pb.stepSimulation(physicsClientId=self.cid)

    # --- 渲染 ---
    def add_marker(self, pos, radius=0.01, rgba=(1, 0.1, 0.1, 1)):
        vs = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.cid)
        return pb.createMultiBody(0, -1, vs, list(map(float, pos)), physicsClientId=self.cid)

    def render(self, target=(0.0, 0.0, 0.4), distance=1.6, yaw=45, pitch=-25, size=(400, 300)):
        w, h = size
        view = pb.computeViewMatrixFromYawPitchRoll(
            target, distance, yaw, pitch, 0, 2, physicsClientId=self.cid
        )
        proj = pb.computeProjectionMatrixFOV(50, w / h, 0.05, 10, physicsClientId=self.cid)
        img = pb.getCameraImage(w, h, view, proj, renderer=pb.ER_TINY_RENDERER, physicsClientId=self.cid)
        return np.reshape(img[2], (h, w, 4))[:, :, :3].astype(np.uint8)
