"""由 RobotParams（DH）+ appearance 產生 URDF。

* 慣性以主軸形式輸出（inertial rpy 對齊等效圓柱軸），避免匯入器對角化差異。
* 視覺/碰撞幾何來自 `appearance.py`（UR 型外觀），與動力學參數無關。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .appearance import MATERIALS, Prim, base_prims, link_prims
from .meshgen import mesh_name, write_mesh
from .robot import SixAxisRobot, rot_z_to


def _f(x) -> str:
    return repr(float(x))


def _rpy(R: np.ndarray) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # gimbal lock：矩陣仍正確，僅第三角取 0
        r = Rotation.from_matrix(R).as_euler("xyz")  # extrinsic xyz = URDF rpy
    return " ".join(_f(x) for x in r)


def _xyz(p) -> str:
    return " ".join(_f(x) for x in p)


def _geom(p: Prim) -> str:
    if p.kind == "cylinder":
        return f'<cylinder length="{_f(p.size[0])}" radius="{_f(p.size[1])}"/>'
    if p.kind == "box":
        return f'<box size="{_xyz(p.size)}"/>'
    return f'<sphere radius="{_f(p.size[0])}"/>'


def _visual_geom(p: Prim) -> str:
    if p.mesh is not None:
        return f'<mesh filename="meshes/{mesh_name(p.mesh, *p.size)}" scale="1 1 1"/>'
    return _geom(p)


def all_prims(robot: SixAxisRobot) -> list[Prim]:
    out = list(base_prims())
    for i in range(6):
        out += link_prims(robot, i)
    return out


def _prim_xml(p: Prim) -> list[str]:
    origin = f'<origin xyz="{_xyz(p.center)}" rpy="{_rpy(rot_z_to(p.axis))}"/>'
    out = [
        f'    <visual>{origin}<geometry>{_visual_geom(p)}</geometry><material name="{p.material}"/></visual>'
    ]
    if p.collision:
        out.append(f"    <collision>{origin}<geometry>{_geom(p)}</geometry></collision>")
    return out


def build_urdf(robot: SixAxisRobot, name: str = "sixaxis_ur5like") -> str:
    p = robot.p
    L = [f'<?xml version="1.0"?>\n<robot name="{name}">']
    for mname, rgba in MATERIALS.items():
        L.append(f'  <material name="{mname}"><color rgba="{_xyz(rgba)}"/></material>')
    L.append('  <link name="base_link">')
    L.append(
        '    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
    )
    for prim in base_prims():
        L += _prim_xml(prim)
    L.append("  </link>")
    for i in range(6):
        rin = _rpy(robot.R_in[i])
        dg = robot.I_diag[i]
        L.append(f'  <link name="link{i + 1}">')
        L.append(
            f'    <inertial><origin xyz="{_xyz(robot.com[i])}" rpy="{rin}"/><mass value="{_f(p.mass[i])}"/>'
            f'<inertia ixx="{_f(dg[0])}" iyy="{_f(dg[1])}" izz="{_f(dg[2])}" ixy="0" ixz="0" iyz="0"/></inertial>'
        )
        for prim in link_prims(robot, i):
            L += _prim_xml(prim)
        L.append("  </link>")
    L.append(
        f'  <link name="flange"><inertial><mass value="{_f(p.flange_mass)}"/>'
        '<inertia ixx="1e-12" iyy="1e-12" izz="1e-12" ixy="0" ixz="0" iyz="0"/></inertial></link>'
    )
    parents = ["base_link"] + [f"link{i}" for i in range(1, 6)]
    for i in range(6):
        L.append(
            f'  <joint name="joint{i + 1}" type="revolute"><parent link="{parents[i]}"/><child link="link{i + 1}"/>'
            f'<origin xyz="{_xyz(robot.pF[i])}" rpy="{_rpy(robot.RF[i])}"/><axis xyz="0 0 1"/>'
            f'<limit lower="{_f(p.q_min[i])}" upper="{_f(p.q_max[i])}" effort="{_f(p.tau_max[i])}" velocity="{_f(p.qd_max[i])}"/></joint>'
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
    for p in all_prims(robot):  # 視覺網格寫到 URDF 同層的 meshes/
        if p.mesh is not None:
            write_mesh(path.parent / "meshes", p.mesh, *p.size)
    return path


if __name__ == "__main__":
    out = write_urdf(SixAxisRobot(), Path(__file__).resolve().parents[1] / "robots" / "ur5_like.urdf")
    print("wrote", out)
