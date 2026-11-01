"""由 RobotParams（DH）產生 URDF。慣性以主軸形式輸出（inertial rpy 對齊圓柱軸），避免匯入器對角化差異。"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import SixAxisRobot

COLORS = [
    "0.25 0.25 0.30 1",
    "0.15 0.45 0.85 1",
    "0.90 0.55 0.10 1",
    "0.20 0.65 0.35 1",
    "0.75 0.25 0.25 1",
    "0.60 0.60 0.65 1",
]


def _rpy(R: np.ndarray) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # gimbal lock：矩陣仍正確，僅第三角取 0
        r = Rotation.from_matrix(R).as_euler("xyz")  # extrinsic xyz = Rz(y)Ry(p)Rx(r) 即 URDF rpy
    return " ".join(repr(float(x)) for x in r)


def _xyz(p) -> str:
    return " ".join(repr(float(x)) for x in p)


def build_urdf(robot: SixAxisRobot, name: str = "sixaxis_ur5like") -> str:
    p = robot.p
    L = [f'<?xml version="1.0"?>\n<robot name="{name}">']
    L.append(
        '  <link name="base_link"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial><visual><origin xyz="0 0 0.02"/>'
        '<geometry><cylinder length="0.04" radius="0.08"/></geometry>'
        '<material name="base"><color rgba="0.2 0.2 0.2 1"/></material></visual>'
        '<collision><origin xyz="0 0 0.02"/><geometry><cylinder length="0.04" radius="0.08"/></geometry></collision></link>'
    )
    for i in range(6):
        seg = robot.seg[i]
        length = float(np.linalg.norm(seg))
        rin = _rpy(robot.R_in[i])
        c = robot.com[i]
        dg = robot.I_diag[i]
        L.append(f'  <link name="link{i + 1}">')
        L.append(
            f'    <inertial><origin xyz="{_xyz(c)}" rpy="{rin}"/><mass value="{p.mass[i]!r}"/>'
            f'<inertia ixx="{float(dg[0])!r}" iyy="{float(dg[1])!r}" izz="{float(dg[2])!r}" ixy="0" ixz="0" iyz="0"/></inertial>'
        )
        L.append(
            f'    <visual><origin xyz="{_xyz(c)}" rpy="{rin}"/><geometry><cylinder length="{float(length)!r}" radius="{p.radius[i]!r}"/></geometry>'
            f'<material name="m{i + 1}"><color rgba="{COLORS[i]}"/></material></visual>'
        )
        L.append(
            f'    <visual><origin xyz="0 0 0"/><geometry><sphere radius="{p.radius[i] * 1.1!r}"/></geometry>'
            f'<material name="j{i + 1}"><color rgba="0.1 0.1 0.1 1"/></material></visual>'
        )
        L.append(
            f'    <collision><origin xyz="{_xyz(c)}" rpy="{rin}"/><geometry><cylinder length="{float(length)!r}" radius="{p.radius[i]!r}"/></geometry></collision>'
        )
        L.append("  </link>")
    L.append(
        f'  <link name="flange"><inertial><mass value="{p.flange_mass!r}"/>'
        '<inertia ixx="1e-12" iyy="1e-12" izz="1e-12" ixy="0" ixz="0" iyz="0"/></inertial></link>'
    )
    parents = ["base_link"] + [f"link{i}" for i in range(1, 6)]
    for i in range(6):
        L.append(
            f'  <joint name="joint{i + 1}" type="revolute"><parent link="{parents[i]}"/><child link="link{i + 1}"/>'
            f'<origin xyz="{_xyz(robot.pF[i])}" rpy="{_rpy(robot.RF[i])}"/><axis xyz="0 0 1"/>'
            f'<limit lower="{p.q_min[i]!r}" upper="{p.q_max[i]!r}" effort="{p.tau_max[i]!r}" velocity="{p.qd_max[i]!r}"/></joint>'
        )
    L.append(
        f'  <joint name="flange_joint" type="fixed"><parent link="link6"/><child link="flange"/>'
        f'<origin xyz="{_xyz(robot.pF[6])}" rpy="{_rpy(robot.RF[6])}"/></joint>'
    )
    L.append("</robot>\n")
    return "\n".join(L)


def write_urdf(robot: SixAxisRobot, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_urdf(robot), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = write_urdf(SixAxisRobot(), Path(__file__).resolve().parents[1] / "robots" / "ur5_like.urdf")
    print("wrote", out)
