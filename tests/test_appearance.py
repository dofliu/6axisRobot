import re

import numpy as np
import pybullet as pb

from sixaxis.meshgen import PROFILES, lathe_obj
from sixaxis.pbsim import URDF_PATH

STANDARD_POSES = {
    "zero": np.zeros(6),
    "home": np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]),
    "demo": np.array([0, -1.2, 1.8, -2.1, -1.57, 0.0]),
}


def test_meshes_exist_and_up_to_date():
    text = URDF_PATH.read_text(encoding="utf-8")
    names = set(re.findall(r'filename="meshes/([^"]+)"', text))
    assert len(names) >= 10
    for n in names:
        m = re.match(r"(\w+)_L(\d+)_R(\d+)\.obj", n)
        kind, L, R = m.group(1), int(m.group(2)) / 1e4, int(m.group(3)) / 1e4
        path = URDF_PATH.parent / "meshes" / n
        assert path.exists(), n
        # 名稱以 0.1 mm 量化；重新產生的網格須與檔案一致（尺寸誤差 < 0.05 mm 內的幾何）
        assert path.read_text().count("\nv ") == lathe_obj(PROFILES[kind](L, R)).count("\nv ")


def test_mesh_normals_unit():
    obj = lathe_obj(PROFILES["housing"](0.17, 0.06))
    vn = np.array([[float(x) for x in ln.split()[1:]] for ln in obj.splitlines() if ln.startswith("vn ")])
    assert np.allclose(np.linalg.norm(vn, axis=1), 1.0, atol=1e-4)


def test_visual_detail_loaded(pbr):
    from sixaxis.appearance import MATERIALS

    vs = pb.getVisualShapeData(pbr.id, physicsClientId=pbr.cid)
    per_link = {}
    for v in vs:
        per_link[v[1]] = per_link.get(v[1], 0) + 1
    for link in range(6):
        assert per_link.get(link, 0) >= 3  # 外殼 + 2 密封環（端蓋只畫在外露端）
    caps = [v for v in vs if np.allclose(v[7], MATERIALS["cap"], atol=1e-3)]
    assert len(caps) >= 5


def test_exposed_ends():
    from sixaxis import SixAxisRobot
    from sixaxis.appearance import exposed_ends

    ends = [exposed_ends(SixAxisRobot(), j) for j in range(1, 7)]
    assert ends[0] == (False, False)  # 底座柱：底部埋入底座、頂部被肩外殼包住
    assert ends[5][1]  # 腕 3 外側（法蘭端）必須外露
    assert sum(a + b for a, b in ends) >= 5


def test_standard_poses_no_self_collision(pbr):
    for name, q in STANDARD_POSES.items():
        pbr.set_state(q)
        for a in range(-1, 7):
            for b in range(a + 2, 7):  # 非相鄰連桿
                for c in pb.getClosestPoints(pbr.id, pbr.id, 0.05, a, b, physicsClientId=pbr.cid):
                    assert c[8] >= 0.0, (name, a, b, c[8])
