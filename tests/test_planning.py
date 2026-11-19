"""T110 避障路徑規劃：保證式邊檢查、Lipschitz 上界、RRT/RRT*、捷徑、執行。"""

import numpy as np
import pytest

from sixaxis import planning as P
from sixaxis.graspsim import Q_REF, GraspScene, tool_pose
from sixaxis.gripper import CUBE
from sixaxis.scenarios import wall_scenario


@pytest.fixture(scope="module")
def wall():
    sc = wall_scenario()
    yield sc
    sc.cc.close()


def test_boxes_from_workcell_and_chunking():
    """場景 33 個方塊（> 16，需分批複合體）：分批後的最近距離 = 逐一方塊最近距離的最小值。"""
    s = GraspScene(CUBE, workcell=True)
    boxes = P.boxes_from_bodies(s.cid, s.wc.bodies)
    s.close()
    assert len(boxes) == 33 and len(boxes) > P.CHUNK
    cc = P.CollisionChecker(boxes, floor=False)
    rng = np.random.default_rng(3)
    singles = [P.CollisionChecker([b], floor=False) for b in boxes]
    for _ in range(5):
        q = P.LOWER + (P.UPPER - P.LOWER) * rng.random(6)
        q[1] = rng.uniform(-2.5, -0.6)
        d_all = cc.distances(q)[0]
        d_min = min(c.distances(q)[0] for c in singles)
        assert d_all == pytest.approx(min(d_min, P.D_MAX), abs=1e-9)
    for c in singles + [cc]:
        c.close()


def test_lipschitz_bound(wall):
    """沿直線邊 d_obs(q(λ)) 與自碰撞距離的變化量 ≤ λ·D（D = Σ R_i|Δq_i|，保證式檢查的前提）。"""
    cc = wall.cc
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(25):
        qa = rng.uniform(cc.lower, cc.upper)
        qb = qa + rng.normal(0, 0.15, 6)
        Dab = cc.motion_bound(qa, qb)
        da = np.array(cc.distances(qa))
        for lam in (0.25, 0.5, 1.0):
            d = np.array(cc.distances(qa + lam * (qb - qa)))
            ok = da < P.D_MAX - 1e-9  # 被截斷者不比
            if ok.any():
                worst = max(worst, float(np.max(np.abs(d - da)[ok] / (lam * Dab))))
    assert worst <= 1.0


def test_certified_edge_never_misses_collision(wall):
    """保證式判為無碰撞的邊，以 5 mrad 密集取樣也必須全程 f > 0（零漏判）。"""
    cc = wall.cc
    rng = np.random.default_rng(1)
    n_free = 0
    for _ in range(40):
        qa = wall.q_start + rng.normal(0, 0.6, 6)
        qb = qa + rng.normal(0, 0.4, 6)
        if cc.clearance(qa) <= 0 or cc.clearance(qb) <= 0:
            continue
        if cc.edge_free(qa, qb):
            n_free += 1
            assert cc.edge_dense(qa, qb, 0.005) > 0
    assert n_free >= 5


def test_gamma_star_and_informed_sampling(wall):
    # d=2、單位正方形全自由：γ* = 2(1.5)^{1/2}(1/π)^{1/2}
    g = P.gamma_star(1.0, np.zeros(2), np.ones(2), d=2)
    assert g == pytest.approx(2 * np.sqrt(1.5 / np.pi))
    pl = P.RRTStar(wall.cc, seed=0)
    qs, qg = wall.q_start, wall.goals[0]
    c = 1.3 * np.linalg.norm(qg - qs)
    xs = np.array([pl.sample_informed(qs, qg, c) for _ in range(300)])
    s = np.linalg.norm(xs - qs, axis=1) + np.linalg.norm(xs - qg, axis=1)
    assert np.all(s <= c + 1e-9)


def test_wall_scenario_needs_planning(wall):
    """直接關節直線到任一目標 IK 解皆碰撞；RRT*（第一解即停）找到路徑；
    捷徑後各段仍通過保證式檢查且不變長。"""
    cc = wall.cc
    assert len(wall.goals) >= 1
    assert not any(cc.edge_free(wall.q_start, g) for g in wall.goals)
    res = P.RRTStar(cc, star=True, seed=2).plan(wall.q_start, wall.goals, max_iter=1500, stop_at_first=True)
    assert res.success
    sc = P.shortcut(cc, res.path, 150, seed=0)
    assert P.path_length(sc) <= res.cost + 1e-9
    assert all(cc.edge_free(a, b) for a, b in zip(sc[:-1], sc[1:], strict=True))
    assert np.allclose(sc[0], wall.q_start) and any(np.allclose(sc[-1], g) for g in wall.goals)
    assert P.path_min_clearance(cc, sc, 0.02) > P.M_OBS - 1e-3


def test_rrt_star_cost_monotone(wall):
    res = P.RRTStar(wall.cc, star=True, seed=3).plan(wall.q_start, wall.goals, max_iter=250)
    assert res.success
    c = [h[1] for h in res.history if np.isfinite(h[1])]
    assert all(b <= a + 1e-12 for a, b in zip(c[:-1], c[1:], strict=True))
    assert res.cost == pytest.approx(P.path_length(res.path), rel=1e-9)


def test_time_path_rest_to_rest():
    r = GraspScene.__new__(GraspScene)  # 只需運動學模型
    from sixaxis.robot import SixAxisRobot

    r = SixAxisRobot()
    q0 = r.ik(tool_pose([0.45, 0, 0.3]), Q_REF)[0]
    path = [q0, q0 + 0.3, q0 + np.array([0.6, 0.1, -0.2, 0.3, 0.0, 0.2])]
    segs, smp = P.time_path(r, path)
    assert len(segs) == 2
    assert np.allclose(smp.q[0], path[0]) and np.allclose(smp.q[-1], path[-1])
    assert np.max(np.abs(smp.qd) / r.qd_max) <= 0.5 + 1e-6
    k = int(round(segs[0].duration / 0.01))
    assert np.allclose(smp.qd[k], 0, atol=1e-9)  # 中間路徑點靜止
