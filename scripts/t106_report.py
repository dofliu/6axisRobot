"""T106 報告：噴塗表面覆蓋（beta 分佈沉積模型、光柵路徑、膜厚均勻度）→ JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t106_report.py 2026-11-14
注意：與 pytest 不可同時執行（都會重寫 robots/ur5_spray.urdf）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from sixaxis import spraying as S  # noqa: E402
from sixaxis.spraying import (  # noqa: E402
    Panel,
    SprayCell,
    SprayModel,
    SprayParams,
    film_metrics,
    parse_job,
    transfer_efficiency,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T106"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
MODEL = SprayModel()
FLAT = dict(center=(0.45, 0.0, 0.104), size=(0.40, 0.30))
TILT_C = (0.55, 0.0, 0.22)
BORDER = 0.01
CENTRAL = 0.10  # 中央區 |u|, |w| ≤ 0.10 m：與無限多道解析解比較
M: dict = {"date": DATE, "model": {**MODEL.__dict__, "flow_cc_min": MODEL.flow / S.CC_MIN, "R": MODEL.R},
           "gun": {"GUN_L": S.GUN_L, "GUN_MASS": S.GUN_MASS}, "defaults": SprayParams().__dict__,
           "panel_flat": FLAT, "border_m": BORDER}
t_start = time.time()


def tilt_panel(th, res=0.005):
    a = np.radians(th)
    return Panel(TILT_C, (-np.sin(a), 0, np.cos(a)), (0, 1, 0), (0.40, 0.30), res=res, name=f"tilt{th}")


def central(H, P):
    U, Wg = np.meshgrid(P.gu, P.gw, indexing="ij")
    h = H[(np.abs(U) <= CENTRAL) & (np.abs(Wg) <= CENTRAL)]
    return {"central_mean_um": float(h.mean() * 1e6), "central_cv_pct": float(h.std() / h.mean() * 100),
            "central_range_pct": float((h.max() - h.min()) / h.mean() * 100)}


def run(prm=None, panel=None, controller="ctc", res=0.005, keep=False, model=None):
    P = panel or Panel(**FLAT, res=res)
    c = SprayCell(model=model, controller=controller)
    try:
        c.add_panel(P)
        prm = prm or SprayParams()
        m = c.spray(P, prm)
        if m["ok"]:
            H = c.thickness(P, m["i0"], m["i1"], every=2)
            m["film"] = film_metrics(H, P, border=BORDER)
            m["film_full"] = film_metrics(H, P)
            m.update(central(H, P))
            m["te"] = transfer_efficiency(H, P, c.model, m["on_s"])
            m["mean_pred_um"] = c.model.flow / (prm.speed * prm.spacing) * 1e6
            if keep:
                sl = slice(m["i0"], m["i1"])
                m["_H"], m["_P"] = H, P
                m["_ser"] = {k: np.array(c.rec[k][sl]) for k in ("t", "q", "qd", "tau", "tcp", "arc")}
    finally:
        c.close()
    m.pop("q_range_deg", None)
    return m


def clean(m):
    return {k: v for k, v in m.items() if not k.startswith("_") and k not in ("i0", "i1")}


# ---------------------------------------------------- 1. 沉積模型（無手臂）
v0 = 0.3
y = np.linspace(-0.08, 0.08, 161)
dt = 2e-4
xs = np.arange(-0.1, 0.1, v0 * dt) + v0 * dt / 2
noz = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, MODEL.h_ref)])
g_num = MODEL.deposit(np.column_stack([np.zeros_like(y), y, np.zeros_like(y)]), np.tile([0, 0, 1.0], (len(y), 1)),
                      noz, np.tile([0, 0, -1.0], (len(xs), 1)), dt)
g_ana = MODEL.stroke(y, v0)
ratios = np.linspace(0.2, 1.6, 281)
M["model_check"] = {
    "stroke_max_rel_err": float(np.max(np.abs(g_num - g_ana)) / g_ana.max()),
    "stroke_peak_um": float(g_ana.max() * 1e6),
    "max_spacing_mm": {f"{p}%": MODEL.max_spacing(p / 100) * 1e3 for p in (1, 2, 5)},
    "ripple": {str(b): [SprayModel(beta=b).ripple(r * MODEL.R)[0] for r in ratios] for b in (1.5, 2.5, 4.0)},
    "ratios": ratios.tolist(),
}
print("model", M["model_check"]["stroke_max_rel_err"], M["model_check"]["max_spacing_mm"])

# ---------------------------------------------------- 2. 標稱：平板 400×300、v 0.3、Δ 50、h 200
nom = run(res=0.0025, keep=True)
H_nom, P_nom, ser = nom.pop("_H"), nom.pop("_P"), nom.pop("_ser")
M["nominal"] = clean(nom)
print("nominal", nom["ok"], nom["film"], nom["te"], nom["speed_dev_on_max_pct"], nom["cycle_s"])

# ---------------------------------------------------- 3. 道間距掃描
M["spacing"] = []
for d in (0.030, 0.040, 0.050, 0.054, 0.060, 0.070, 0.080, 0.090):
    m = clean(run(SprayParams(spacing=d)))
    m["ripple_cv_pct"], m["ripple_range_pct"] = (x * 100 for x in MODEL.ripple(d))
    M["spacing"].append(m)
    print("spacing", d, m["ok"], m.get("central_cv_pct"), m["ripple_cv_pct"], m.get("spray_s"))

# ---------------------------------------------------- 4. 速度掃描
M["speed"] = []
for v in (0.10, 0.20, 0.30, 0.45, 0.60, 0.70, 0.90):
    m = clean(run(SprayParams(speed=v)))
    M["speed"].append(m)
    print("speed", v, m["ok"], m["error"], m.get("film", {}).get("mean_um"), m.get("vel_util"), m["plan_vel_ratio"])
M["v_max_pred"] = 0.30 / M["nominal"]["plan_vel_ratio"]

# ---------------------------------------------------- 5. 噴距掃描（Δ 固定 50 mm）
M["standoff"] = []
for h in (0.15, 0.175, 0.20, 0.225, 0.25):
    m = clean(run(SprayParams(standoff=h)))
    m["ripple_cv_pct"] = MODEL.ripple(0.05 * MODEL.h_ref / h)[0] * 100  # 圖樣半徑 ∝ h ⇔ 等效間距 Δ·h_ref/h
    M["standoff"].append(m)
    print("standoff", h, m["ok"], m.get("central_cv_pct"), m["ripple_cv_pct"])

# ---------------------------------------------------- 6. 扳機區 × 側邊外延
M["edge"] = []
for trig in (0.0, 0.03, 0.06):
    for edge in (0.0, 0.025):
        m = clean(run(SprayParams(trigger=trig, edge=edge)))
        m["trigger_mm"], m["edge_mm"] = trig * 1e3, edge * 1e3
        M["edge"].append(m)
        print("edge", trig, edge, m["film_full"]["cv_pct"], m["film_full"]["coverage_pct"], m["te"])

# ---------------------------------------------------- 7. 噴槍傾角（平板，繞道方向 u）
M["tilt"] = []
for tl in (0.0, -5.0, -10.0, -15.0, -20.0, -30.0, 10.0, 15.0):
    m = clean(run(SprayParams(tilt_deg=tl)))
    m["tilt_deg"] = tl
    M["tilt"].append(m)
    print("tilt", tl, m["ok"], m["error"], m.get("film", {}).get("cv_pct"), m.get("sigma_min"))

# ---------------------------------------------------- 8. 斜面板（噴槍垂直面板）
M["inclined"] = []
for th in (30, 45):
    m = clean(run(panel=tilt_panel(th)))
    m["panel_tilt_deg"] = th
    M["inclined"].append(m)
    print("inclined", th, m["ok"], m.get("film"), m.get("plan_vel_ratio"))

# ---------------------------------------------------- 9. 控制器比較
M["controllers"] = []
for ctl in ("ctc", "pd"):
    for v in (0.30, 0.60):
        m = clean(run(SprayParams(speed=v), controller=ctl))
        m["controller"] = ctl
        M["controllers"].append(m)
        print("ctrl", ctl, v, m["path_dev_on_max_mm"], m["speed_dev_on_max_pct"], m["film"]["cv_pct"])
M["experiments_s"] = time.time() - t_start
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# ---------------------------------------------------- 圖 1：模型
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
r = np.linspace(-0.1, 0.1, 401)
for h in (0.15, 0.20, 0.25):
    ax[0].plot(r * 1e3, MODEL.plane(np.abs(r), h) * 1e6, label=f"h = {h * 1e3:.0f} mm")
ax[0].set(xlabel="r [mm]", ylabel="film rate [um/s]",
          title=f"Static spray footprint (beta={MODEL.beta}, half-angle {MODEL.half_angle:.0f} deg)")
ax[0].legend(fontsize=8)
ax[1].plot(y * 1e3, g_ana * 1e6, color="k", lw=2, label="closed form g(y)")
ax[1].plot(y[::4] * 1e3, g_num[::4] * 1e6, "o", ms=3, color="#c0553b", label="numerical (point source)")
ax[1].set(xlabel="y across stroke [mm]", ylabel="film [um]",
          title=f"Single stroke, v = 0.3 m/s (max rel. err {M['model_check']['stroke_max_rel_err']:.1e})")
ax[1].legend(fontsize=8)
for b, col in (("1.5", "#8aa"), ("2.5", "#3b6ea8"), ("4.0", "#b58")):
    ax[2].plot(ratios, np.array(M["model_check"]["ripple"][b]) * 100, color=col, label=f"analytic, beta {b}")
ax[2].plot([m["spacing"] / MODEL.R for m in M["spacing"]], [m["central_cv_pct"] for m in M["spacing"]], "o",
           color="#c0553b", label="robot sim (central 200x200)")
ax[2].axhline(2, color="k", lw=0.6, ls=":")
ax[2].axvline(MODEL.max_spacing(0.02) / MODEL.R, color="g", ls="--", lw=0.8, label="max spacing (CV<=2 %)")
ax[2].set(xlabel="stroke spacing / pattern radius R", ylabel="film CV [%]", ylim=(0, 25),
          title="Raster uniformity vs spacing")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_model.png", dpi=105)

# ---------------------------------------------------- 圖 2：標稱膜厚與時間歷程
rb = S.SixAxisRobot()
fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
ext = [P_nom.gw[0] * 1e3, P_nom.gw[-1] * 1e3, P_nom.gu[0] * 1e3, P_nom.gu[-1] * 1e3]
im = ax[0, 0].imshow(H_nom * 1e6, origin="lower", extent=ext, cmap="viridis", aspect="equal")
loc = P_nom.to_local(ser["tcp"])
ax[0, 0].plot(loc[1][::40] * 1e3, loc[0][::40] * 1e3, color="w", lw=0.5, alpha=0.7)
ax[0, 0].set(xlim=ext[:2], ylim=ext[2:], xlabel="w (across strokes) [mm]", ylabel="u (stroke dir.) [mm]",
             title="Film thickness map [um] + nozzle path")
fig.colorbar(im, ax=ax[0, 0], fraction=0.05)
iu = np.argmin(np.abs(P_nom.gu))
ax[0, 1].plot(P_nom.gw * 1e3, H_nom[iu] * 1e6, color="#c0553b", label="robot sim (u = 0)")
ax[0, 1].plot(P_nom.gw * 1e3, MODEL.raster(P_nom.gw, 0.3, 0.05, nom["strokes"]) * 1e6, "--", color="k",
              lw=1, label="analytic (7 ideal strokes)")
ax[0, 1].axhline(nom["mean_pred_um"], color="g", lw=0.6, ls=":", label="Q/(v*spacing)")
ax[0, 1].set(xlabel="w [mm]", ylabel="film [um]", title="Cross-section across strokes")
ax[0, 1].legend(fontsize=8)
iw = np.argmin(np.abs(P_nom.gw))
ax[0, 2].plot(P_nom.gu * 1e3, H_nom[:, iw] * 1e6, color="#3b6ea8")
ax[0, 2].set(xlabel="u [mm]", ylabel="film [um]",
             title=f"Section along strokes (w = 0; trigger +{SprayParams().trigger * 1e3:.0f} mm)")
t = ser["t"] - ser["t"][0]
spd = np.linalg.norm(np.gradient(ser["tcp"], ser["t"], axis=0), axis=1)
ax[1, 0].plot(t, spd, color="#3b6ea8", lw=0.8, label="nozzle speed")
ax[1, 0].fill_between(t, 0, 0.35 * ser["arc"], color="#f2b880", alpha=0.4, label="gun ON")
ax[1, 0].set(xlabel="t [s]", ylabel="[m/s]", ylim=(0, 0.36),
             title=f"Nozzle speed (dev. while ON max {nom['speed_dev_on_max_pct']:.3f} %)")
ax[1, 0].legend(fontsize=8, loc="lower center")
for j in range(6):
    ax[1, 1].plot(t[::10], ser["qd"][::10, j] / rb.qd_max[j], label=f"J{j + 1}")
    ax[1, 2].plot(t[::10], ser["tau"][::10, j] / rb.tau_max[j], label=f"J{j + 1}")
ax[1, 1].set(xlabel="t [s]", ylabel="qd / qd_max", title="Joint velocity utilisation")
ax[1, 2].set(xlabel="t [s]", ylabel="tau / tau_max", title="Joint torque utilisation (applied)")
ax[1, 1].legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_nominal.png", dpi=105)

# ---------------------------------------------------- 圖 3：掃描
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
ok = [m for m in M["speed"] if m["ok"]]
vv = np.array([m["speed_cmd"] for m in ok])
ax[0].plot(vv, [m["film"]["mean_um"] for m in ok], "o", color="#c0553b", label="sim mean film [um]")
vf = np.linspace(0.08, 0.95, 100)
ax[0].plot(vf, MODEL.flow / (vf * 0.05) * 1e6, "--", color="k", lw=0.8, label="Q/(v*spacing)")
ax[0].set(xlabel="nozzle speed [m/s]", ylabel="film [um]", title="Speed sweep")
a2 = ax[0].twinx()
a2.plot([m["speed_cmd"] for m in M["speed"]], [m["plan_vel_ratio"] for m in M["speed"]], "d:", color="#3b6ea8",
        label="planned qd/qd_max")
a2.axhline(1, color="#3b6ea8", lw=0.5)
a2.set_ylabel("planned qd / qd_max")
ax[0].legend(fontsize=7, loc="upper center")
a2.legend(fontsize=7, loc="center right")
hh = [m["standoff"] * 1e3 for m in M["standoff"]]
ax[1].plot(hh, [m["central_cv_pct"] for m in M["standoff"]], "o-", color="#c0553b", label="sim CV (central)")
ax[1].plot(hh, [m["ripple_cv_pct"] for m in M["standoff"]], "x--", color="k", label="analytic CV")
ax[1].plot(hh, [m["film"]["cv_pct"] for m in M["standoff"]], "s:", color="#888", label="sim CV (panel - 10 mm)")
ax[1].set(xlabel="standoff h [mm]", ylabel="CV [%]", title="Standoff sweep (spacing 50 mm)")
ax[1].legend(fontsize=7)
for m in M["edge"]:
    ax[2].plot(m["te"] * 100, m["film_full"]["cv_pct"], "o" if m["edge_mm"] == 0 else "s", color="#3b6ea8")
    ax[2].annotate(f"trig {m['trigger_mm']:.0f} / edge {m['edge_mm']:.0f} mm",
                   (m["te"] * 100, m["film_full"]["cv_pct"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
ax[2].set(xlabel="transfer efficiency to panel [%]", ylabel="full-panel CV [%]",
          title="Trigger zone x side extension")
tl = [m for m in M["tilt"] if m["ok"]]
ax[3].plot([m["tilt_deg"] for m in tl], [m["film"]["cv_pct"] for m in tl], "o-", color="#c0553b",
           label="CV (panel - 10 mm)")
ax[3].plot([m["tilt_deg"] for m in tl], [m["central_cv_pct"] for m in tl], "s:", color="#888", label="CV central")
bad = [m for m in M["tilt"] if not m["ok"]]
for m in bad:
    ax[3].axvline(m["tilt_deg"], color="r", lw=0.6, ls="--")
    ax[3].text(m["tilt_deg"], 0.5, m["error"], rotation=90, fontsize=7, color="r")
ax[3].set(xlabel="gun tilt about stroke dir. [deg]", ylabel="CV [%]", title="Gun tilt (flat panel)")
ax[3].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_sweeps.png", dpi=105)

# ---------------------------------------------------- 動畫：JSON 工作（平板）＋ 45° 斜板
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 330)
JOB = {"gun": {"flow_cc_min": 20, "h_ref": 0.2, "half_angle": 20, "beta": 2.5},
       "panels": [{"name": "flat-panel", "center": list(FLAT["center"]), "size": list(FLAT["size"]),
                   "normal": [0, 0, 1], "u": [0, 1, 0], "speed": 0.3, "spacing": 0.05, "standoff": 0.2}]}
BARE, PAINT = np.array([0.80, 0.81, 0.83]), np.array([0.10, 0.33, 0.78])
rng = np.random.default_rng(106)
frames: list = []


def lab(img, left, right):
    im_ = Image.fromarray(img)
    d_ = ImageDraw.Draw(im_)
    d_.rectangle([0, 0, im_.width, 26], fill=(20, 24, 32))
    d_.text((10, 4), left, fill=(235, 238, 242), font=font)
    d_.text((im_.width - 10 - d_.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im_)


class Demo:
    """面板色塊（依膜厚上色）＋噴霧粒子，於 hook 每 0.1 s 更新並渲染三視角。"""

    def __init__(self, cell, panel, title, views, n_tile=(32, 24), target_um=22.2):
        self.c, self.P, self.title, self.views, self.target = cell, panel, title, views, target_um
        tp = Panel(panel.c + panel.n * 0.0004, panel.n, panel.u, (panel.L, panel.W),
                   res=panel.L / n_tile[0], name="tiles")
        self.tp, self.H = tp, np.zeros(len(tp.pts))
        from scipy.spatial.transform import Rotation

        q = Rotation.from_matrix(panel.rotation()).as_quat()
        he = [panel.L / n_tile[0] / 2 * 0.98, panel.W / n_tile[1] / 2 * 0.98, 0.0003]
        self.tiles = []
        for p in tp.pts:
            vs = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[*BARE, 1], physicsClientId=cell.cid)
            self.tiles.append(pb.createMultiBody(0, -1, vs, list(map(float, p)), list(q), physicsClientId=cell.cid))
        vs = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.0025, rgbaColor=[0.25, 0.5, 0.95, 1],
                                  physicsClientId=cell.cid)
        self.drops = [pb.createMultiBody(0, -1, vs, [0, 0, -1], physicsClientId=cell.cid) for _ in range(45)]
        self.last_t, self.k_last, self.color = -1.0, len(cell.rec["t"]), np.zeros(len(tp.pts))

    def __call__(self, c):
        if c.t - self.last_t < 0.1 - 1e-9:
            return
        self.last_t = c.t
        k1 = len(c.rec["t"])
        on = np.array(c.rec["arc"][self.k_last:k1], bool)
        if on.any():
            P = np.array(c.rec["tcp"][self.k_last:k1])[on][::5]
            A = np.array(c.rec["tool_z"][self.k_last:k1])[on][::5]
            self.H += c.model.deposit(self.tp.pts, self.tp.normals, P, A, 5 * c.dt)
        self.k_last = k1
        lvl = np.clip(self.H * 1e6 / self.target, 0, 1.15)
        for i in np.where(np.abs(lvl - self.color) > 0.02)[0]:
            col = BARE + (PAINT - BARE) * min(lvl[i], 1.0) - 0.15 * max(lvl[i] - 1.0, 0.0)
            pb.changeVisualShape(self.tiles[i], -1, rgbaColor=[*np.clip(col, 0, 1), 1], physicsClientId=c.cid)
            self.color[i] = lvl[i]
        T = c.tcp_pose()
        for b in self.drops:
            if c.arc:
                s = rng.uniform(0.05, 1.0)
                ang = rng.uniform(0, 2 * np.pi)
                rr = np.sqrt(rng.uniform()) * s * MODEL.R * 0.8
                p = T[:3, 3] + T[:3, 2] * s * 0.2 + rr * (np.cos(ang) * T[:3, 0] + np.sin(ang) * T[:3, 1])
            else:
                p = [0, 0, -1]
            pb.resetBasePositionAndOrientation(b, list(map(float, p)), [0, 0, 0, 1], physicsClientId=c.cid)
        mean = float(self.H.mean() * 1e6)
        txt = f"GUN ON  film avg {mean:4.1f} um" if c.arc else f"gun off  film avg {mean:4.1f} um"
        vs = [c.sim.render(target=tg, distance=d_, yaw=y_, pitch=p_, size=SIZE, fov=42) for tg, y_, p_, d_ in self.views]
        frames.append(np.hstack([lab(vs[0], self.title, f"t = {c.t:.1f} s"), lab(vs[1], txt, "side"),
                                 lab(vs[2], "film (blue = target 22 um)", "top")]))


model_j, items = parse_job(JOB)
demo = {"job": JOB, "panels": []}
panel_j, prm_j = items[0]
c = SprayCell(model=model_j, render=True)
c.add_panel(panel_j)
dm = Demo(c, panel_j, "T106 spray: JSON job, flat panel",
          [((0.38, 0.0, 0.22), 50, -24, 1.35), ((0.45, 0.0, 0.20), 0, -12, 0.95), ((0.45, 0.0, 0.10), 90, -89, 0.75)])
c.hook = dm
m = c.spray(panel_j, prm_j)
c.move_j(c.q_home)
for _ in range(6):
    c.wait(0.1)
n_flat = len(frames)
demo["panels"].append({k: v for k, v in m.items() if k not in ("i0", "i1", "q_range_deg")})
demo["panels"][-1]["film"] = film_metrics(c.thickness(panel_j, m["i0"], m["i1"], every=2), panel_j, border=BORDER)
c.close()
Pt = tilt_panel(45, res=0.005)
c = SprayCell(render=True)
c.add_panel(Pt)
dm = Demo(c, Pt, "T106 spray: 45 deg inclined panel",
          [((0.45, 0.0, 0.28), 50, -24, 1.35), ((0.55, 0.0, 0.25), 0, -10, 0.95), ((0.55, 0.0, 0.22), 90, -44, 0.70)])
c.hook = dm
m = c.spray(Pt, SprayParams())
c.move_j(c.q_home)
for _ in range(6):
    c.wait(0.1)
demo["panels"].append({k: v for k, v in m.items() if k not in ("i0", "i1", "q_range_deg")})
demo["panels"][-1]["film"] = film_metrics(c.thickness(Pt, m["i0"], m["i1"], every=2), Pt, border=BORDER)
c.close()
M["demo"] = demo
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(x).resize((720, 198), Image.LANCZOS) for x in frames[::3]]
pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=150, loop=0, optimize=True)
for nm, idx_ in (("key_a", n_flat // 2), ("key_b", n_flat - 1), ("key_c", len(frames) - 1)):
    Image.fromarray(frames[idx_]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = len(frames)
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
M["total_s"] = time.time() - t_start
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("demo", [(p["ok"], p.get("film", {}).get("cv_pct")) for p in demo["panels"]])
print("frames", len(frames), "gif MB", M["gif_MB"], "mp4 MB", M["mp4_MB"], "total s", M["total_s"])
