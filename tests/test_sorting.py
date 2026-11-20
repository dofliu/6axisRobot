"""T111 多物件分類揀選：幾何、感知、抓取、序列、夾持工件碰撞與一次完整物理執行。"""

import numpy as np
import pytest

from sixaxis import sorting as SO
from sixaxis.planning import Box, CollisionChecker
from sixaxis.robot import wrap_pi


def test_poly_dist_and_min_rect():
    A = SO.rect_poly((0, 0), 0.0, 0.01, 0.01)
    B = SO.rect_poly((0.05, 0), 0.0, 0.01, 0.01)
    assert SO.poly_dist(A, B) == pytest.approx(0.03, abs=1e-12)
    assert SO.poly_dist(A, SO.rect_poly((0.015, 0.0), 0.7, 0.01, 0.01)) == 0.0  # 相交
    # 旋轉卡尺：任意 yaw 的 40×60 矩形點雲 → 尺寸與方位（mod 90°）還原
    rng = np.random.default_rng(0)
    for yaw in rng.uniform(-np.pi, np.pi, 5):
        pts = SO.rect_poly((0.3, 0.1), yaw, 0.02, 0.03)
        ab = rng.random((200, 2))
        fill = pts[0] + ab[:, :1] * (pts[1] - pts[0]) + ab[:, 1:] * (pts[3] - pts[0])
        a, w, l, area = SO.min_area_rect(np.vstack([pts, fill]))
        assert sorted([w, l]) == pytest.approx([0.04, 0.06], abs=1e-9)
        e = (a - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(e) < 1e-9
        assert area == pytest.approx(0.0024, rel=1e-9)


def test_scatter_constraints():
    for seed in range(5):
        parts = SO.scatter(seed)
        assert len(parts) == 6
        for i, p in enumerate(parts):
            P = p.poly
            assert P[:, 0].min() >= SO.ZONE[0] and P[:, 0].max() <= SO.ZONE[1]
            assert P[:, 1].min() >= SO.ZONE[2] and P[:, 1].max() <= SO.ZONE[3]
            for q in parts[i + 1:]:
                assert SO.poly_dist(P, q.poly) >= SO.GAP - 1e-12


def _det(name, xy, yaw):
    sp = SO.CLASSES[name].spec
    w, l = (sp.size[0], sp.size[1]) if sp.shape == "box" else (2 * sp.size[0],) * 2
    return SO.Detection(name, np.array(xy), yaw, w, l, sp.height, 1.0 if sp.shape == "box" else 0.785, 0,
                        SO.CLASSES[name].chroma)


def test_grasp_candidates_blocked_by_neighbor():
    a = _det("red", (0.45, 0.02), 0.0)
    g = SO.grasp_candidates(a, [])
    assert len(g) == 2 and all(abs(x.width - 0.05) < 1e-12 for x in g)
    # 鄰物緊貼兩個閉合方向（±x 與 ±y 各一個高方柱，間距 8 mm < 指墊厚 10 mm + 餘裕）→ 不可抓
    nb = [_det("blue", (0.45 + 0.025 + 0.008 + 0.02, 0.02), 0.0),
          _det("blue", (0.45, 0.02 + 0.025 + 0.008 + 0.03), 0.0)]
    others = [(d.poly, d.top) for d in nb]
    assert SO.grasp_candidates(a, others) == []
    assert len(SO.grasp_candidates(a, others, check=False)) == 2
    # 只擋 +x 方向 → 剩下沿 y 閉合（ψ = 0）的抓取；兩指墊在 ±y
    g1 = SO.grasp_candidates(a, others[:1])
    assert len(g1) == 1 and abs(np.sin(g1[0].psi)) < 1e-9
    # 若 +x 鄰物高 100 mm（高於掌體下緣 TCP+45 mm = 175 mm），掌體（半寬 37.5 mm）也會壓到 → 不可抓
    tall = _det("blue", (0.45 + 0.025 + 0.008 + 0.02, 0.02), 0.0)
    tall.height = 0.10
    assert SO.grasp_candidates(a, [(tall.poly, tall.top)]) == []
    # 長方柱：60 mm 邊（ψ = yaw）與 40 mm 邊兩個候選
    b = _det("blue", (0.45, 0.02), 0.3)
    assert sorted(round(x.width, 4) for x in SO.grasp_candidates(b, [])) == [0.04, 0.06]


def test_place_pose_rules():
    # 料盤 ψ = 0（兩指沿 y 伸入側銷間）；料箱 ψ = 90°：指墊外緣 |x| 在箱內半寬以內 → 可放到箱底
    assert SO.place_yaw(SO.CLASSES["red"]) == 0.0
    for c in ("blue", "green"):
        psi = SO.place_yaw(SO.CLASSES[c])
        pads, _ = SO.gripper_polys((0.0, 0.0), psi)
        assert max(abs(P[:, 0]).max() for P in pads) < SO.BIN_IN_X - 0.005
        for sl in SO.bin_slots(SO.CLASSES[c].dest):
            cy = SO.BIN_C[SO.CLASSES[c].dest][1]
            sp = SO.CLASSES[c].spec
            half_y = (max(sp.size[:2]) if sp.shape == "box" else 2 * sp.size[0]) / 2
            assert abs(sl[1] - cy) + half_y < SO.BIN_IN_Y  # 最長邊沿 y 也放得下
        # 放置高度：物件底面離箱底 5 mm、指墊下緣高於箱底
        h = SO.CLASSES[c].spec.height
        z = SO.place_height(SO.CLASSES[c])
        assert z - SO.grasp_offset(h) - h / 2 == pytest.approx(SO.BIN_FLOOR + SO.BIN_LIFT)
        assert z - SO.PAD_L / 2 > SO.BIN_FLOOR
    # 夾取高度：指墊上緣與頂面齊平 → 掌體下緣高於 70 mm 方柱頂面 > 餘裕
    from sixaxis.scene import BENCH_TOP
    z_cube = BENCH_TOP + 0.025 + SO.grasp_offset(0.05)
    assert z_cube + SO.PALM_BOTTOM - (BENCH_TOP + 0.07) > SO.GRIP_MARGIN


def test_sequence_optimal_not_worse_and_respects_blocking():
    q0 = np.asarray(wrap_pi(SO.home_q()), float)
    for seed in range(3):
        dets = [_det(p.cls.name, p.xy, p.yaw) for p in SO.scatter(50 + seed, gap=0.004)]
        res = {m: SO.plan_sequence(dets, q0, m) for m in ("scan", "greedy", "optimal")}
        for m, (order, est) in res.items():
            assert sorted(order) == list(range(len(dets))), m
            # 執行順序中每一件在當時的剩餘集合下都必須可抓
            rem = list(range(len(dets)))
            for i in order:
                others = [(dets[j].poly, dets[j].top) for j in rem if j != i]
                assert SO.grasp_candidates(dets[i], others), (m, i)
                rem.remove(i)
        assert res["optimal"][1] <= min(res["scan"][1], res["greedy"][1]) + 1e-9


def test_collision_checker_attached_object():
    wall = Box((0.56, -0.03, 0.20), (0.01, 0.10, 0.20), "wall")  # 牆面 x = 0.55
    cc = CollisionChecker([wall], floor=False)  # 地板距離（肩部外殼 ≈ 28 mm）為常數，排除以便比較
    q = SO.ik_near((0.45, -0.03, 0.30), 0.0, np.asarray(wrap_pi(SO.home_q())))
    d0 = cc.distances(q)[0]
    lever0 = cc.lever.copy()
    cc.attach([0.07, 0.03, 0.03])  # 夾著 140 mm 長工件（沿 x），端面 x = 0.52 → 距牆 30 mm
    d1 = cc.distances(q)[0]
    assert d0 > 0.05
    assert d1 == pytest.approx(0.03, abs=2e-3)
    assert np.all(cc.lever >= lever0 - 1e-12)
    cc.set_movable([Box((0.45, 0.06, 0.15), (0.02, 0.02, 0.05), "m", 0.5)])
    assert cc.distances(q)[0] <= d1 + 1e-12
    cc.detach()
    cc.set_movable([])
    assert cc.distances(q)[0] == pytest.approx(d0, abs=1e-9)
    cc.close()


@pytest.fixture(scope="module")
def cell():
    parts = [SO.Part(SO.CLASSES["red"], (0.40, 0.00), 0.5), SO.Part(SO.CLASSES["blue"], (0.50, 0.04), -0.3),
             SO.Part(SO.CLASSES["green"], (0.52, -0.03), 0.0)]
    c = SO.SortCell(parts)
    yield c
    c.close()


def test_perception_accuracy(cell):
    dets = cell.cam.detect(cell.s.cid, cell.s.sim.id)
    assert len(dets) == 3
    idx = SO.match(dets, cell.parts)
    for d, j in zip(dets, idx, strict=True):
        p = cell.parts[j]
        assert d.cls == p.cls.name and d.consistent
        assert np.hypot(*(d.xy - np.array(p.xy))) < 1.5e-3
        e = SO.yaw_error(d, p)
        assert np.isnan(e) or e < 2.0
        assert abs(d.height - p.cls.spec.height) < 2.5e-3


def test_full_sort_run(cell):
    r = cell.run()
    assert r["success"] and r["n_correct"] == 3
    assert r["pick_fail"] == 0 and not r["errors"]
    tray = [f for f in r["final"] if f["dest"] == "tray"][0]
    assert tray["seated"] and tray["pocket_err_mm"] < 3.0
    assert r["torque_util"] < 1.0
