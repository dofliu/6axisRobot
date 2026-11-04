"""抓取/釋放模擬場景（T005）：手臂（PyBullet 位置馬達）+ 夾爪 + 基座上的物件。

手臂以計算力矩控制（ωn = 60 rad/s 臨界阻尼）驅動：逆動力學直接用 PyBullet 的 calculateInverseDynamics
（含夾爪質量，故為「精確模型」），追蹤誤差 ~1e-6 rad。曾試過的替代方案與問題：
* PyBullet 位置馬達：抬升時有 ~0.1 rad 追蹤誤差使夾爪傾斜而把物件擠出；
* 每步 resetJointState（運動學驅動）：手臂關節在步內自由下垂再被重置，造成 1 kHz 鋸齒擾動，
  使物件在 0.9 m/s² 下蠕滑 ~30 mm，與抓持模型無關。
夾爪以位置馬達 + 力上限施加法向力（接觸後 ≈ 力上限）。
"""

from __future__ import annotations

import numpy as np
import pybullet as pb

from .gripper import CUBE, T_TCP, TCP_Z, Gripper, ObjectSpec, make_arm_with_gripper
from .robot import SixAxisRobot
from .scene import Workcell
from .trajectory import joint_ptp

PED_X, PED_TOP = 0.45, 0.10  # 基座（桌面）中心 x 與頂面高度
WN = 60.0  # 手臂計算力矩控制自然頻率 [rad/s]
Q_REF = np.array([0.24, -1.62, -2.14, -0.96, 1.57, -1.33])  # 工具朝下、肘向外構型（IK 選解參考）
R_DOWN = np.diag([1.0, -1.0, -1.0])  # 工具 z 軸朝下


def tool_pose(tcp_xyz) -> np.ndarray:
    """工具朝下的法蘭位姿，使夾持中心（TCP）位於 tcp_xyz。"""
    T = np.eye(4)
    T[:3, :3] = R_DOWN
    T[:3, 3] = np.asarray(tcp_xyz, float) - R_DOWN @ T_TCP[:3, 3]
    return T


class GraspScene:
    def __init__(self, spec: ObjectSpec = CUBE, dt: float = 1e-3, friction: float | None = None,
                 render: bool = False, workcell: bool = False):
        self.r = SixAxisRobot()
        self.sim = make_arm_with_gripper(self.r, plane=render)
        self.cid, self.dt, self.spec = self.sim.cid, dt, spec
        self.sim.make_frictionless()
        pb.setTimeStep(dt, physicsClientId=self.cid)
        pb.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.cid)
        self.g = Gripper(self.sim)
        self.wc = None
        if workcell:  # T006 工作站（工作台頂面高度 = PED_TOP，取料點 = PED_X）
            self.wc = Workcell(self.cid, self.sim.id)
        else:
            he = [0.15, 0.15, PED_TOP / 2]
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.cid)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.15, 0.15, PED_TOP / 2],
                                       rgbaColor=[0.55, 0.57, 0.6, 1], physicsClientId=self.cid)
            self.table = pb.createMultiBody(0, col, vis, [PED_X, 0, PED_TOP / 2], physicsClientId=self.cid)
            pb.changeDynamics(self.table, -1, lateralFriction=1.0, physicsClientId=self.cid)
        self.z_grasp = PED_TOP + spec.height / 2  # 物件中心（= 抓取 TCP 高度）
        self.q_grasp = self._ik([PED_X, 0, self.z_grasp])
        self.sim.set_state(self.q_grasp)
        self.obj = self.g.spawn(spec, [PED_X, 0, self.z_grasp])
        if self.wc:
            self.wc.add_part(self.obj)
        if friction is not None:
            pb.changeDynamics(self.obj, -1, lateralFriction=friction, physicsClientId=self.cid)
        self.rec = {"t": [], "tcp": [], "obj": [], "fL": [], "fR": [], "q": []}
        self.t = 0.0
        self.sim.set_state(self.q_grasp)
        self._hold(self.q_grasp)

    def _ik(self, tcp_xyz):
        sols = self.r.ik(tool_pose(tcp_xyz), Q_REF)
        if not sols:
            raise RuntimeError("IK 無解")
        return sols[0]

    def _hold(self, q, qd=None, qdd=None):
        self._q_cmd = np.asarray(q, float)
        self._qd_cmd = np.zeros(6) if qd is None else np.asarray(qd, float)
        self._qdd_cmd = np.zeros(6) if qdd is None else np.asarray(qdd, float)

    def _arm_torque(self):
        q, qd = self.sim.get_state()
        fp = self.g.finger_pos()
        js = pb.getJointStates(self.sim.id, self.g.fingers, physicsClientId=self.cid)
        fv = np.array([x[1] for x in js])
        a = self._qdd_cmd + 2 * WN * (self._qd_cmd - qd) + WN**2 * (self._q_cmd - q)
        tau = pb.calculateInverseDynamics(
            self.sim.id, list(map(float, [*q, *fp])), list(map(float, [*qd, *fv])),
            list(map(float, [*a, 0.0, 0.0])), physicsClientId=self.cid)
        return np.asarray(tau[:6])

    def step(self, n: int = 1):
        for _ in range(n):
            if self.wc:
                self.wc.pre_step()
            self.sim.apply_torques(self._arm_torque())
            pb.stepSimulation(physicsClientId=self.cid)
            self.t += self.dt
            fl, fr = self.g.contact_forces()
            self.rec["t"].append(self.t)
            self.rec["tcp"].append(self.g.tcp_pos())
            self.rec["obj"].append(self.g.obj_pose()[0])
            self.rec["fL"].append(fl)
            self.rec["fR"].append(fr)
            self.rec["q"].append(self.sim.get_state()[0])

    def wait(self, T: float):
        self.step(int(round(T / self.dt)))

    def move_to(self, q1, T: float, kind: str = "quintic", frame_cb=None, cb_every: int = 25):
        q0 = self.sim.get_state()[0]
        tr = joint_ptp(q0, q1, T, kind)
        n = int(round(T / self.dt))
        for k in range(1, n + 1):
            st = tr.state(k * self.dt)
            self._hold(st[0], st[1], st[2])
            self.step()
            if frame_cb and k % cb_every == 0:
                frame_cb(self)
        self._hold(q1)

    def lift_pose(self, dz: float):
        return self._ik([PED_X, 0, self.z_grasp + dz])

    def peak_tcp_accel(self, q0, q1, T: float) -> float:
        """關節軌跡下 TCP 加速度峰值（FK 數值二次差分）。"""
        tr = joint_ptp(q0, q1, T, "quintic")
        ts = np.arange(0.0, T + 1e-9, 0.002)
        p = np.array([self.r.fk(tr.state(t)[0])[:3, :3] @ T_TCP[:3, 3] + self.r.fk(tr.state(t)[0])[:3, 3]
                      for t in ts])
        a = np.diff(p, 2, axis=0) / 0.002**2
        return float(np.max(np.linalg.norm(a, axis=1)))

    def close(self):
        self.sim.close()


def run_pick_lift(spec: ObjectSpec = CUBE, force: float = 40.0, lift: float = 0.15, T_lift: float = 1.0,
                  friction: float | None = None, frame_cb=None, scene: GraspScene | None = None):
    """開 → 夾 → 抬升 → 保持 → 放回 → 釋放 → 退開。回傳 (scene, metrics)。"""
    s = scene or GraspScene(spec, friction=friction)
    g = s.g
    g.open()
    s.wait(0.3)
    p_obj0 = g.obj_pose()[0].copy()
    g.close(force)
    s.wait(0.6)
    fl, fr = g.contact_forces()
    settle_fingers = g.finger_pos().copy()
    off0 = g.obj_pose()[0] - g.tcp_pos()
    n0 = len(s.rec["t"])
    q_up = s.lift_pose(lift)
    a_peak = s.peak_tcp_accel(s.q_grasp, q_up, T_lift)
    s.move_to(q_up, T_lift, frame_cb=frame_cb)
    s.wait(0.4)
    off_up = g.obj_pose()[0] - g.tcp_pos()
    rel = np.array(s.rec["obj"][n0:]) - np.array(s.rec["tcp"][n0:]) - off0
    slip_lift = float(np.max(np.linalg.norm(rel, axis=1)))
    z_gain = float(g.obj_pose()[0][2] - p_obj0[2])
    held = bool(np.linalg.norm(off_up - off0) < 2e-3)
    s.move_to(s.q_grasp, T_lift, frame_cb=frame_cb)
    s.wait(0.3)
    p_place = g.obj_pose()[0].copy()
    g.open()
    s.wait(0.5)
    s.move_to(q_up, 0.8, frame_cb=frame_cb)
    s.wait(0.3)
    p_end = g.obj_pose()[0]
    m = {
        "force_N": force, "fL_N": fl, "fR_N": fr, "finger_stroke_mm": (settle_fingers * 1e3).tolist(),
        "a_peak_ms2": a_peak, "slip_lift_mm": slip_lift * 1e3, "z_gain_mm": z_gain * 1e3, "held": held,
        "release_drift_mm": float(np.linalg.norm(p_end - p_place) * 1e3),
        "tilt_deg": float(np.degrees(2 * np.arccos(min(1.0, abs(g.obj_pose()[1][3]))))),
    }
    return s, m


def static_slip(spec: ObjectSpec, force: float, hold_T: float = 1.0, friction: float | None = None) -> float:
    """以 60 N 預夾並抬離桌面 5 cm 後，把夾持力降到 force，靜止 hold_T 秒內物件下滑量 [m]（正值＝下滑）。"""
    s = GraspScene(spec, friction=friction)
    try:
        s.g.close(60.0)
        s.wait(0.4)
        s.move_to(s.lift_pose(0.05), 0.8)
        s.wait(0.3)
        s.g.close(force)
        s.wait(0.1)
        z0 = float(s.g.obj_pose()[0][2] - s.g.tcp_pos()[2])
        s.wait(hold_T)
        return z0 - float(s.g.obj_pose()[0][2] - s.g.tcp_pos()[2])
    finally:
        s.close()


__all__ = ["GraspScene", "run_pick_lift", "static_slip", "tool_pose", "TCP_Z"]
