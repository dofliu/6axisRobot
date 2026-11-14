"""T106 噴塗表面覆蓋測試：沉積模型（解析/數值一致、質量守恆）、光柵路徑、膜厚量測、PyBullet 噴塗。"""

import json

import numpy as np
import pytest
from scipy.integrate import quad

from sixaxis import spraying as S
from sixaxis.spraying import (
    CC_MIN,
    Panel,
    SprayCell,
    SprayModel,
    SprayParams,
    film_metrics,
    gun_rotation,
    parse_job,
    raster_path,
    transfer_efficiency,
)

PANEL = dict(center=(0.45, 0.0, 0.104), size=(0.40, 0.30))


@pytest.fixture
def model():
    return SprayModel(flow=20 * CC_MIN, h_ref=0.2, half_angle=20.0, beta=2.5)


def test_plane_profile_conserves_flow(model):
    for h in (0.15, 0.20, 0.25):
        tot = quad(lambda r, h=h: model.plane(r, h) * 2 * np.pi * r, 0, model.R * h / model.h_ref)[0]
        assert tot == pytest.approx(model.flow, rel=1e-6)
    assert model.plane(model.R * 1.01) == 0.0


def test_rate_matches_plane_and_geometry(model):
    r = np.linspace(0, 0.09, 50)
    for h in (0.2, 0.25):
        pts = np.column_stack([r, np.zeros_like(r), np.zeros_like(r)])
        q = model.rate(pts, np.tile([0, 0, 1.0], (50, 1)), (0, 0, h), (0, 0, -1.0))
        np.testing.assert_allclose(q, model.plane(r, h), rtol=1e-12, atol=1e-18)
    # 背面（法向背對噴嘴）與噴軸後方不沉積
    assert model.rate([[0, 0, 0]], [[0, 0, -1.0]], (0, 0, 0.2), (0, 0, -1.0))[0] == 0.0
    assert model.rate([[0, 0, 0.4]], [[0, 0, -1.0]], (0, 0, 0.2), (0, 0, -1.0))[0] == 0.0
    # 斜面：同一點、噴軸垂直 → 入射角 γ 使沉積 ∝ cos γ
    a = np.radians(30)
    n = np.array([np.sin(a), 0, np.cos(a)])
    q0 = model.rate([[0, 0, 0]], [[0, 0, 1.0]], (0, 0, 0.2), (0, 0, -1.0))[0]
    q1 = model.rate([[0, 0, 0]], [n], (0, 0, 0.2), (0, 0, -1.0))[0]
    assert q1 == pytest.approx(q0 * np.cos(a), rel=1e-12)


def test_single_stroke_closed_form(model):
    v, dt = 0.3, 2e-4
    xs = np.arange(-0.1, 0.1, v * dt) + v * dt / 2
    noz = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 0.2)])
    y = np.linspace(-0.08, 0.08, 81)
    pts = np.column_stack([np.zeros_like(y), y, np.zeros_like(y)])
    H = model.deposit(pts, np.tile([0, 0, 1.0], (81, 1)), noz, np.tile([0, 0, -1.0], (len(xs), 1)), dt)
    g = model.stroke(y, v)
    assert np.max(np.abs(H - g)) < 2e-3 * g.max()
    tot = quad(lambda yy: model.stroke(yy, v), -model.R, model.R)[0]
    assert tot == pytest.approx(model.flow / v, rel=1e-6)


def test_raster_mean_and_spacing(model):
    v, d = 0.3, 0.05
    y = np.linspace(-d / 2, d / 2, 400, endpoint=False)
    H = model.raster(y, v, d, 41)
    assert H.mean() == pytest.approx(model.flow / (v * d), rel=1e-3)
    cv, rng = model.ripple(d)
    assert H.std() / H.mean() == pytest.approx(cv, rel=1e-3)
    assert rng >= cv
    dm = model.max_spacing(0.02)
    assert model.ripple(dm)[0] <= 0.02 < model.ripple(dm + 0.0025 * model.R)[0]
    assert 0.6 < dm / model.R < 0.9


def test_raster_path_geometry():
    P = Panel(**PANEL, res=0.01)
    seam, ws = raster_path(P, 0.05, standoff=0.2, overrun=0.08)
    assert len(ws) == 7 and ws[-1] - ws[0] == pytest.approx(0.30)
    assert seam.kink_deg < 1e-6
    assert seam.length == pytest.approx(7 * 0.56 + 6 * np.pi * 0.025, rel=1e-9)
    assert np.allclose(seam.p[:, 2], 0.304)
    u, w, _ = P.to_local(seam.p[0])
    assert u[0] == pytest.approx(-0.28) and w[0] == pytest.approx(ws[0])
    seam2, ws2 = raster_path(P, 0.05, edge=0.025)
    assert len(ws2) == 8
    with pytest.raises(ValueError):
        raster_path(P, 0.0)
    R = gun_rotation((0, 0, -1), (0, 1, 0))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_film_metrics_uniform():
    P = Panel(**PANEL, res=0.01)
    H = np.full(P.shape, 20e-6)
    m = film_metrics(H, P)
    assert m["cv_pct"] == pytest.approx(0.0, abs=1e-9) and m["coverage_pct"] == 100.0
    H[0, :] = 10e-6
    m2 = film_metrics(H, P, target=20e-6, border=0.01)
    assert m2["coverage_pct"] == 100.0 and film_metrics(H, P, target=20e-6)["coverage_pct"] < 100.0


def test_parse_job():
    job = {"gun": {"flow_cc_min": 25, "half_angle": 18},
           "panels": [{"name": "a", "center": [0.45, 0, 0.104], "size": [0.4, 0.3], "speed": 0.25}]}
    model, items = parse_job(json.dumps(job))
    assert model.flow == pytest.approx(25 * CC_MIN) and model.half_angle == 18
    assert items[0][0].name == "a" and items[0][1].speed == 0.25
    for bad in ({}, {"panels": []}, {"panels": [{"center": [0, 0, 0]}]},
                {"panels": [{"center": [0, 0, 0], "size": [0.1, 0.1], "foo": 1}]},
                {"panels": [{"center": [0, 0, 0], "size": [0.1, 0.1], "speed": 5}]},
                {"gun": {"beta": 0.5}, "panels": [{"center": [0, 0, 0], "size": [0.1, 0.1]}]}):
        with pytest.raises(ValueError):
            parse_job(bad)


@pytest.fixture
def cell():
    c = SprayCell()
    yield c
    c.close()


def test_spray_nominal(cell):
    P = Panel(**PANEL, res=0.005)
    cell.add_panel(P)
    prm = SprayParams()
    m = cell.spray(P, prm)
    assert m["ok"], m
    assert m["strokes"] == 7 and m["path_dev_on_max_mm"] < 0.05 and m["speed_dev_on_max_pct"] < 0.5
    assert m["axis_err_max_deg"] < 0.01 and m["vel_util"] < 1 and m["torque_util"] < 1
    assert m["clear_mm"] > 5
    H = cell.thickness(P, m["i0"], m["i1"], every=4)
    f = film_metrics(H, P, border=0.01)
    assert f["mean_um"] == pytest.approx(cell.model.flow / (prm.speed * prm.spacing) * 1e6, rel=0.02)
    assert f["cv_pct"] < 3.0 and f["coverage_pct"] > 99
    te = transfer_efficiency(H, P, cell.model, m["on_s"])
    assert 0.5 < te < 1.0
    # 每步積分與每 4 步積分一致
    H1 = cell.thickness(P, m["i0"], m["i1"], every=1)
    assert np.max(np.abs(H1 - H)) < 0.01 * H.mean()


def test_spray_limit_and_tilted_panel(cell):
    P = Panel(**PANEL, res=0.01)
    cell.add_panel(P)
    assert cell.spray(P, SprayParams(speed=0.9))["error"] == "LIMIT"
    a = np.radians(30)
    Pt = Panel((0.55, 0.0, 0.22), (-np.sin(a), 0, np.cos(a)), (0, 1, 0), (0.40, 0.30), res=0.01, name="tilt")
    c2 = SprayCell()
    try:
        c2.add_panel(Pt)
        m = c2.spray(Pt, SprayParams())
        assert m["ok"], m
        f = film_metrics(c2.thickness(Pt, m["i0"], m["i1"], every=4), Pt, border=0.01)
        assert f["mean_um"] == pytest.approx(c2.model.flow / (0.3 * 0.05) * 1e6, rel=0.02)
    finally:
        c2.close()


def test_weldcell_default_tool_unchanged():
    from sixaxis.welding import T_TCP, WeldCell

    c = WeldCell()
    try:
        assert np.allclose(c.T_tcp, T_TCP)
        assert np.allclose(c.tcp_pose()[:3, 3], (0.35, 0.0, 0.35), atol=1e-6)
    finally:
        c.close()
    assert S.T_GUN[2, 3] == pytest.approx(S.GUN_L)
