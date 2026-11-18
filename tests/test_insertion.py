"""T109 軸孔插入：理論公式、接觸幾何、銷體動力學與完整插入流程。"""

import numpy as np
import pytest

from sixaxis.insertion import (
    RIGID,
    ContactSpec,
    Hole,
    HoleSpec,
    InsertCell,
    InsertParams,
    PegBody,
    PegSpec,
    RCCSpec,
    capture_radius,
    chamfer_forces,
    chamfer_ratio,
    hole_block_obj,
    jam_depth,
    jam_free_tilt,
    tilt_depth_limit,
    two_point_statics,
    wedge_angle,
)

PEG, HOLE = PegSpec(), HoleSpec()
DOWN = np.diag([1.0, -1.0, -1.0])


def test_theory_formulas():
    assert chamfer_ratio(0.0) == pytest.approx(1.0)
    assert chamfer_ratio(0.3) == pytest.approx(1.3 / 0.7)
    assert chamfer_ratio(1.2) == float("inf")
    c = HOLE.R - PEG.r
    assert capture_radius(PEG, HOLE) == pytest.approx(HOLE.w_c + c)
    f = chamfer_forces(0.5e-3, PEG, HOLE, RCCSpec(), 0.3)
    assert f["F_lat"] == pytest.approx(4e3 * (0.5e-3 - c))
    assert wedge_angle(PEG, HOLE, 0.3) == pytest.approx(2 * c / (0.3 * 2 * PEG.r))
    th = np.radians(2.0)
    l = tilt_depth_limit(th, PEG, HOLE)
    assert (l + PEG.r * np.sin(th)) * np.tan(th) + 2 * PEG.r * np.cos(th) == pytest.approx(2 * HOLE.R)
    assert tilt_depth_limit(0.0, PEG, HOLE) == float("inf")


def test_jam_free_tilt_bound():
    """θ_max 為 f(l) = F l/(2μ) + 2cK/l − Kθ0 的最小值等於 0 的傾角；θ0 稍大時 jam_depth 有限且使 f = 0。"""
    F, mu, K = 15.0, 0.3, 15.0
    c = HOLE.R - PEG.r
    tm = jam_free_tilt(F, PEG, HOLE, mu, K)
    l = np.linspace(1e-4, 0.05, 20001)
    f = F * l / (2 * mu) + 2 * c * K / l - K * tm
    assert f.min() == pytest.approx(0.0, abs=1e-5)
    assert jam_depth(tm * 0.99, F, PEG, HOLE, mu, K) == float("inf")
    lj = jam_depth(tm * 1.5, F, PEG, HOLE, mu, K)
    assert F * lj / (2 * mu) + 2 * c * K / lj - K * tm * 1.5 == pytest.approx(0.0, abs=1e-9)
    # 對應兩點接觸靜力：在 l_j、M = K(θ0 − 2c/l)、F_x = 0 時恰在卡死邊界
    s = two_point_statics(0.0, F, -K * (tm * 1.5 - 2 * c / lj), lj, PEG.r, mu)
    assert s["F_req"] == pytest.approx(F, rel=1e-6)


def test_two_point_statics_equilibrium():
    r, mu, l = 0.006, 0.3, 0.005
    for Fx, M in ((2.0, -0.05), (-1.0, -0.02), (0.5, -0.01)):
        s = two_point_statics(Fx, 10.0, M, l, r, mu)
        N1, N2 = s["N1"], s["N2"]
        assert Fx + N1 - N2 == pytest.approx(0.0, abs=1e-12)
        assert M - mu * r * N1 + (mu * r + l) * N2 == pytest.approx(0.0, abs=1e-12)
    s = two_point_statics(0.0, 1.0, -0.05, l, r, mu)  # 大力矩 → 兩點接觸、推力不足 → 卡死
    assert s["two_point"] and s["jam"]
    s2 = two_point_statics(0.0, 100.0, -0.05, l, r, mu)
    assert s2["two_point"] and not s2["jam"]


def test_hole_sdf():
    h = Hole((0, 0, 0), HOLE)
    s = HOLE
    pts = np.array([
        [0.02, 0, -1e-4],  # 頂面下 0.1 mm
        [0.0, 0, -0.005],  # 孔內（無接觸）
        [s.R + 5e-5, 0, -0.005],  # 孔壁內 50 µm
        [0.0, 0, -s.depth - 2e-5],  # 孔底下 20 µm
        [s.R + s.w_c / 2 + 1e-4, 0, -s.w_c / 2],  # 倒角面內
        [0.02, 0, 1e-3],  # 頂面上方
    ])
    pen, nrm = h.sdf_solid(pts)
    assert pen[0] == pytest.approx(1e-4) and nrm[0] == pytest.approx([0, 0, 1])
    assert pen[1] == 0 and pen[5] == 0
    assert pen[2] == pytest.approx(5e-5) and nrm[2] == pytest.approx([-1, 0, 0])
    assert pen[3] == pytest.approx(2e-5) and nrm[3] == pytest.approx([0, 0, 1])
    k = 1 / np.sqrt(2)
    assert pen[4] == pytest.approx(1e-4 * k) and nrm[4] == pytest.approx([-k, 0, k])


def _bench(e, mu=0.3):
    h = Hole((0, 0, 0), HOLE)
    rcc = RCCSpec()
    b = PegBody(PEG, rcc, ContactSpec(mu=mu), h)
    T = np.eye(4)
    T[:3, :3] = DOWN
    T[:3, 3] = [e, 0, 0.0008]
    b.reset(T)
    v = np.array([0, 0, -0.002])
    fl, fa, dd = [], [], []
    for _ in range(2000):
        T[:3, 3] += v * 1e-3
        F, _ = b.advance(T, v, np.zeros(3), 1e-3, 20)
        fl.append(-F[0])
        fa.append(F[2] + PEG.mass * 9.81)
        dd.append(h.depth_of(b.p))
        if dd[-1] > 0.002:
            break
    return np.array(fl), np.array(fa), np.array(dd), b


def test_peg_hangs_statically():
    h = Hole((0, 0, 0), HOLE)
    b = PegBody(PEG, RCCSpec(), ContactSpec(), h)
    T = np.eye(4)
    T[:3, :3] = DOWN
    T[:3, 3] = [0, 0, 0.05]
    b.reset(T)
    for _ in range(100):
        F, M = b.advance(T, np.zeros(3), np.zeros(3), 1e-3, 20)
    assert F == pytest.approx([0, 0, -PEG.mass * 9.81], abs=1e-6)
    assert np.linalg.norm(b.v) < 1e-9


def test_chamfer_bench_matches_quasistatic():
    e = 0.8e-3
    fl, fa, dd, b = _bench(e)
    pred = chamfer_forces(e, PEG, HOLE, RCCSpec(), 0.3)
    assert fl.max() == pytest.approx(pred["F_lat"], rel=0.1)
    sl = (dd > 0.2e-3) & (dd < 0.8e-3) & (fl > 0.3)
    assert np.median(fa[sl] / fl[sl]) == pytest.approx(chamfer_ratio(0.3), rel=0.1)
    assert dd[-1] > 0.002  # 穿過倒角進入孔


def test_tilted_peg_geometric_limit():
    """固定傾角 θ 的銷：在 l < 極限時存在無干涉的側向位置，l > 極限時不存在。"""
    h = Hole((0, 0, 0), HOLE)
    b = PegBody(PEG, RCCSpec(), ContactSpec(), h)
    th = np.radians(2.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    lim = tilt_depth_limit(th, PEG, HOLE)

    def fits(depth):
        for x in np.arange(-3e-4, 3e-4, 2e-6):
            b.p, b.R = np.array([x, 0, -depth]), R
            b.v[:], b.w[:] = 0, 0
            b._contacts(b.com(), 0.0)
            if sum(b.contact_kind) == 0:
                return True
        return False

    assert fits(HOLE.w_c + lim - 1e-4)
    assert not fits(HOLE.w_c + lim + 1e-4)


def test_block_mesh(tmp_path):
    h = Hole((0, 0, 0), HOLE)
    for cut in (False, True):
        p = hole_block_obj(h, tmp_path / f"b{cut}.obj", cut=cut)
        lines = p.read_text().splitlines()
        nv = sum(1 for x in lines if x.startswith("v "))
        idx = [int(t) for x in lines if x.startswith("f ") for t in x.split()[1:]]
        assert nv > 100 and min(idx) >= 1 and max(idx) <= nv


@pytest.fixture
def urdf(tmp_path):
    return tmp_path / "ur5_peg.urdf"


def test_rcc_admittance_insertion(urdf):
    c = InsertCell(hole_err=(0.6e-3, 0.0), urdf_path=urdf)
    m = c.insert(InsertParams())
    c.close()
    assert m["ok"] and m["error"] is None
    assert m["depth_mm"] >= 19.5
    assert m["F_max"] < 20.0
    assert m["lat_final_mm"] < 0.1  # 銷端位於孔內（徑向間隙 80 µm）


def test_rigid_position_fails_overforce(urdf):
    c = InsertCell(hole_err=(0.6e-3, 0.0), rcc=RIGID, urdf_path=urdf)
    m = c.insert(InsertParams(strategy="position"))
    c.close()
    assert not m["ok"] and m["error"] == "OVERFORCE"
    assert m["F_max"] >= 80.0


def test_miss_then_spiral_search(urdf):
    e = 1.4e-3  # > 捕捉半徑 1.08 mm
    c = InsertCell(hole_err=(e, 0.0), urdf_path=urdf)
    m = c.insert(InsertParams())
    c.close()
    assert not m["ok"] and m["error"] == "MISS"
    c = InsertCell(hole_err=(e, 0.0), urdf_path=urdf)
    m = c.insert(InsertParams(search=True))
    c.close()
    assert m["ok"] and m["searched"] and m["depth_mm"] >= 19.5
