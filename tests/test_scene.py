"""T006 工作站場景測試：碰撞群組、工作台承載、輸送帶速度與擋料、料盤定位、治具基準、可達性與碰撞查詢。"""

import numpy as np
import pybullet as pb
import pytest

from sixaxis import scene as S
from sixaxis.graspsim import Q_REF, GraspScene, run_pick_lift, tool_pose
from sixaxis.gripper import CUBE

Z_REST = S.BENCH_TOP + CUBE.size[2] / 2


@pytest.fixture
def sc():
    s = GraspScene(CUBE, workcell=True)
    yield s
    s.close()


def pose(s, b):
    return np.array(pb.getBasePositionAndOrientation(b, physicsClientId=s.cid)[0])


def drop(s, xy, z=None, vel=None):
    b = s.g.spawn(CUBE, [xy[0], xy[1], Z_REST + 0.02 if z is None else z])
    s.wc.add_part(b)
    if vel is not None:
        pb.resetBaseVelocity(b, list(vel), physicsClientId=s.cid)
    return b


def test_groups_and_masks():
    # 規則 (gA & mB) && (gB & mA)：arm 與 static/part 碰撞；belt 與 static 不碰撞
    def hit(a, b):
        return bool(S.GROUPS[a] & S.MASKS[b]) and bool(S.GROUPS[b] & S.MASKS[a])

    assert hit("arm", "static") and hit("arm", "belt") and hit("arm", "part")
    assert hit("part", "static") and hit("part", "belt") and hit("part", "part")
    assert not hit("belt", "static") and not hit("belt", "belt") and not hit("arm", "arm")


def test_ghost_passes_through_bench(sc):
    ghost = sc.wc.add_ghost([S.PICK_XY[0], 0.1, 0.3])
    sc.wait(1.0)
    assert pose(sc, ghost)[2] < 0.0  # 穿過工作台持續自由落體
    b = drop(sc, (0.55, -0.1), z=0.3)
    sc.wait(1.0)
    assert pose(sc, b)[2] == pytest.approx(Z_REST, abs=1e-3)  # 同條件的 part 被工作台承住


def test_bench_supports_part(sc):
    assert pose(sc, sc.obj)[2] == pytest.approx(Z_REST, abs=1e-3)
    assert np.linalg.norm(pb.getBaseVelocity(sc.obj, physicsClientId=sc.cid)[0]) < 1e-3
    assert sc.wc.arm_collisions() == []


def test_conveyor_speed_and_stopper(sc):
    b = drop(sc, (0.25, 0.45), z=Z_REST + 1e-3)
    sc.wait(0.5)
    assert pb.getBaseVelocity(b, physicsClientId=sc.cid)[0][0] == pytest.approx(S.BELT_SPEED, rel=0.02)
    x0 = pose(sc, b)[0]
    sc.wait(1.0)
    assert (pose(sc, b)[0] - x0) / 1.0 == pytest.approx(S.BELT_SPEED, rel=0.02)
    assert abs(pose(sc, b)[1] - 0.45) < 1e-3  # 無側向漂移
    sc.wait(3.2)  # 剩 ~0.15 m / 0.1 m/s
    p = pose(sc, b)
    assert abs(p[0] - S.BELT_PICK_XY[0]) < 1e-3 and abs(p[1] - S.BELT_PICK_XY[1]) < 1e-3
    assert np.linalg.norm(pb.getBaseVelocity(b, physicsClientId=sc.cid)[0]) < 1e-3


def test_belt_off_when_not_in_contact(sc):
    b = drop(sc, S.PICK_XY, z=Z_REST + 0.3)  # 工作台上方：不應被帶速驅動
    sc.wait(1.0)
    assert abs(pose(sc, b)[0] - S.PICK_XY[0]) < 1e-3


@pytest.mark.parametrize("idx", range(4))
def test_tray_pocket_locates_part(sc, idx):
    cx, cy = S.pocket_centers()[idx]
    off = (S.POCKET - CUBE.size[0]) / 2  # 單邊間隙 4 mm
    # 含純 x / 純 y 偏移（純軸向偏移取 0.95 倍單邊間隙；無導入倒角，超出間隙者會停在擋牆頂）
    for dx, dy in ((0.75, -0.75), (-0.75, 0.75), (0.95, 0.0), (0.0, -0.95)):
        b = drop(sc, (cx + dx * off, cy + dy * off), z=Z_REST + S.TRAY_PLATE_T + 0.03)
        sc.wait(1.0)
        p = pose(sc, b)
        assert p[2] == pytest.approx(Z_REST + S.TRAY_PLATE_T, abs=1e-3)
        assert abs(p[0] - cx) <= off + 5e-4 and abs(p[1] - cy) <= off + 5e-4  # 留在格內
        pb.removeBody(b, physicsClientId=sc.cid)
        sc.wc.parts.remove(b)


def test_fixture_datum_repeatability(sc):
    rng = np.random.default_rng(6)
    fin = []
    for _ in range(5):
        fx, fy = S.FIX_NOM
        off = rng.uniform(-0.006, 0.006, 2)
        b = drop(sc, (fx - 0.008 + off[0], fy - 0.008 + off[1]), z=Z_REST + 1e-3, vel=(0.6, 0.6, 0))
        sc.wait(1.0)
        fin.append(pose(sc, b)[:2])
        pb.removeBody(b, physicsClientId=sc.cid)
        sc.wc.parts.remove(b)
    fin = np.array(fin)
    assert np.max(np.linalg.norm(fin - np.array(S.FIX_NOM), axis=1)) < 2e-3


STATIONS = [("pick", S.PICK_XY), ("belt", S.BELT_PICK_XY), ("fixture", S.FIX_NOM)] + [
    (f"pocket{i}", c) for i, c in enumerate(S.pocket_centers())
]


@pytest.mark.parametrize("name,xy", STATIONS)
def test_station_reachable_collision_free(sc, name, xy):
    zc = S.BENCH_TOP + (S.TRAY_PLATE_T if name.startswith("pocket") else 0.0) + CUBE.size[2] / 2
    for dz in (0.0, 0.12):  # 抓取高度與上方 12 cm 接近點
        sols = sc.r.ik(tool_pose([xy[0], xy[1], zc + dz]), Q_REF)
        assert sols, f"{name} dz={dz} IK 無解"
        sc.sim.set_state(sols[0])
        assert sc.wc.arm_collisions() == [], (name, dz, sc.wc.arm_collisions())


def test_collision_query_detects_penetration(sc):
    sols = sc.r.ik(tool_pose([S.PICK_XY[0], 0.0, 0.05]), Q_REF)  # 工具中心在桌面下
    sc.sim.set_state(sols[0])
    assert "bench" in sc.wc.arm_collisions()
    assert sc.wc.min_clearance() < 0.0
    sc.sim.set_state(sc.q_grasp)
    assert sc.wc.min_clearance() > 0.0


def test_grasp_lift_still_works_in_workcell():
    s = GraspScene(CUBE, workcell=True)
    s, m = run_pick_lift(CUBE, force=40.0, lift=0.15, T_lift=1.0, scene=s)
    s.close()
    assert m["held"] and m["slip_lift_mm"] < 0.5 and m["release_drift_mm"] < 1.0
