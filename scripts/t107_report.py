"""T107 報告：鎖螺絲（定位＋姿態＋旋轉）→ JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t107_report.py 2026-11-15
注意：與 pytest 不可同時執行（都會重寫 robots/ur5_driver.urdf）。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from sixaxis.screwing import (  # noqa: E402
    RPM,
    DriverParams,
    FastenParams,
    JointSpec,
    ScrewCell,
    ThreadModel,
    bench_test,
    parse_job,
    run_job,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T107"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
t_start = time.time()
NOM = ThreadModel()
S = NOM.s
DP = DriverParams()
PLATE = dict(center=(0.45, 0.0, 0.10), size=(0.20, 0.14))
TILT_DEG = 20.0
DROP = {"i0", "i1"}


def clean(r):
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items() if k not in DROP}


M: dict = {
    "date": DATE,
    "screw": {**S.__dict__, "d2": S.d2, "d3": S.d3, "As": S.As, "Dkm": S.Dkm, "phi_ct_deg": S.phi_ct_deg},
    "joint": {**JointSpec().__dict__, "c": NOM.c, "K": NOM.K, "k_th": NOM.k_th, "l_e": NOM.l_e,
              "turns_seat": NOM.theta_s / (2 * np.pi), "k_theta": NOM.k_theta, "self_locking": NOM.self_locking(),
              "lead_deg": float(np.degrees(np.arctan(S.P / (np.pi * S.d2)))),
              "rho_deg": float(np.degrees(np.arctan(0.12 / np.cos(np.radians(30)))))},
    "driver": DP.__dict__, "fasten": {k: v for k, v in FastenParams().__dict__.items() if k != "driver"},
}
F_tgt = NOM.torque_to_force(DP.T_target)
M["joint"].update({"F_target": F_tgt, "util_target": NOM.utilization(F_tgt), "T_nu90": 0.9 / NOM.utilization(1) * NOM.K})


def flat_cell(render=False):
    c = ScrewCell(render=render)
    top, _, _ = c.add_plate("A", PLATE["center"], PLATE["size"])
    return c, top


HOLES_XY = [(-0.05, -0.03), (0.0, -0.03), (0.05, -0.03), (-0.05, 0.03), (0.0, 0.03), (0.05, 0.03)]

# ---------------------------------------------------- A. 標稱：6 孔（取料＋二段式扭力控制）
c, top = flat_cell()
nominal = []
for i, (x, y) in enumerate(HOLES_XY):
    h = c.add_hole(f"H{i + 1}", top + np.array([x, y, 0.0]))
    r = c.fasten(h, FastenParams())
    if i == 0:
        ser = c.series(r["i0"], r["i1"])
        nr_log = {k: np.array(v) if k != "phase" else v for k, v in c.nr.log.items()}
        k0 = len(c.rec["t"])
    nominal.append(clean(r))
c.move_j(c.q_home)
rec = {k: np.array(c.rec[k]) for k in ("t", "q", "qd", "tau", "q_ref")}
c.close()
M["nominal"] = nominal
print("A", [(r["hole"], r["ok"], round(r["T_peak"], 4)) for r in nominal], f"{time.time() - t_start:.0f}s")

# ---------------------------------------------------- B. 鎖固策略（接合模擬台 + 手臂）
strat = []
for n_fin in (30, 60, 120, 240, 600):
    for two in (True, False) if n_fin == 600 else (True,):
        prm = replace(DP, n_fin=n_fin, two_stage=two)
        r, nr = bench_test(NOM, prm)
        est = NOM.overshoot_estimate(r["w_cut"], DP.J, r["T_cut"], DP.T_max)
        est_ref = NOM.overshoot_estimate((n_fin if two else DP.n_run) * RPM, DP.J, DP.T_target, DP.T_max)
        strat.append({"n_fin": n_fin, "two_stage": two, **r, "dT_sim": r["T_peak"] - r["T_cut"], "dT_est": est,
                      "dT_est_cmd_speed": est_ref, "rpm_cut": r["w_cut"] / RPM})
M["strategy_bench"] = strat
c, top = flat_cell()
strat_robot = []
for name, prm in (("two-stage 600→60 rpm", DP), ("single-stage 600 rpm", replace(DP, two_stage=False)),
                  ("slow 60 rpm only", replace(DP, n_run=60.0))):
    h = c.add_hole(f"S{len(strat_robot)}", top + np.array([-0.05 + 0.05 * len(strat_robot), 0.0, 0.0]))
    r = c.fasten(h, FastenParams(driver=prm, pick=False))
    strat_robot.append({"name": name, **clean(r)})
c.close()
M["strategy_robot"] = strat_robot
print("B", [(r["name"], r["error"], round(r["T_err_pct"], 2), round(r["t_spin"], 2)) for r in strat_robot])

# ---------------------------------------------------- C. 摩擦離散：扭力控制 vs 扭力–轉角控制
fric = []
for mu in (0.08, 0.10, 0.12, 0.14, 0.16):
    m = ThreadModel(joint=JointSpec(mu_g=mu, mu_k=mu))
    for sname in ("torque", "angle"):
        r, _ = bench_test(m, replace(DP, strategy=sname), nominal=NOM)
        fric.append({"mu": mu, "strategy": sname, **r, "F_theory_torque": m.torque_to_force(DP.T_target)})
M["friction"] = fric
for sname in ("torque", "angle"):
    F = [x["F_clamp"] for x in fric if x["strategy"] == sname]
    M[f"friction_spread_{sname}_pct"] = (max(F) - min(F)) / np.mean(F) * 100
    M[f"friction_halfrange_{sname}_pct"] = (max(F) - min(F)) / 2 / np.mean(F) * 100
c, top = flat_cell()
fric_robot = []
for i, mu in enumerate((0.08, 0.16)):
    for j, sname in enumerate(("torque", "angle")):
        h = c.add_hole(f"M{i}{j}", top + np.array([-0.05 + 0.1 * i, -0.03 + 0.06 * j, 0.0]))
        r = c.fasten(h, FastenParams(driver=replace(DP, strategy=sname), pick=False),
                     ThreadModel(joint=JointSpec(mu_g=mu, mu_k=mu)))
        fric_robot.append({"mu": mu, "strategy": sname, **clean(r)})
c.close()
M["friction_robot"] = fric_robot
print("C", M["friction_spread_torque_pct"], M["friction_spread_angle_pct"])

# ---------------------------------------------------- D. 接合剛性（ISO 5393 硬/軟接合）
stiff, curves = [], {}
for lab, f in (("hard", 1.0), ("medium", 0.2), ("soft", 1 / 30)):
    m = ThreadModel(joint=JointSpec(c=NOM.c * f))
    for two in (True, False):
        r, nr = bench_test(m, replace(DP, two_stage=two), nominal=m)
        stiff.append({"joint": lab, "c": m.c, "k_theta": m.k_theta, "two_stage": two, **r})
        if two:
            th = np.degrees(np.array(nr.log["th"]) - m.theta_s)
            curves[lab] = (th, np.array(nr.log["Tl"]))
M["stiffness"] = stiff
print("D", [(s["joint"], s["two_stage"], round(s["T_err_pct"], 2), round(s["angle_snug_deg"], 1)) for s in stiff])

# ---------------------------------------------------- E. 孔位偏差：搜尋 on/off
c, top = flat_cell()
rng = np.random.default_rng(107)
offs = []
for k, e in enumerate((0.0, 0.25, 0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5)):
    ang = rng.uniform(0, 2 * np.pi)
    err = (e * 1e-3 * np.cos(ang), e * 1e-3 * np.sin(ang), 0.0)
    for search in (False, True):
        h = c.add_hole(f"E{k}{int(search)}", top + np.array([*HOLES_XY[(2 * k + int(search)) % 6], 0.0]),
                       visual=False)
        r = c.fasten(h, FastenParams(pick=False, hole_err=err, search=search))
        offs.append({"offset_mm": e, "search": search, **clean(r)})
c.close()
M["offset"] = offs
M["e_cap_mm"] = ((S.dh - S.d) / 2 + FastenParams().e_cap_extra) * 1e3
print("E", [(o["offset_mm"], o["search"], o["error"], round(o.get("search_s", 0), 3)) for o in offs])

# ---------------------------------------------------- F. 傾角誤差 → 斜牙偵測
c, top = flat_cell()
tilts = []
for k, tdeg in enumerate((0.0, 2.0, 4.0, 4.9, 5.1, 6.0, 8.0)):
    h = c.add_hole(f"T{k}", top + np.array([*HOLES_XY[k % 6], 0.0]), visual=False)
    r = c.fasten(h, FastenParams(pick=False, tilt_err_deg=tdeg))
    tilts.append({"tilt_cmd_deg": tdeg, **clean(r)})
c.close()
M["tilt"] = tilts
print("F", [(t["tilt_cmd_deg"], t["error"], round(t["turns_rundown"] or -1, 2)) for t in tilts])

# ---------------------------------------------------- G. 進給方式；H. 反作用前饋
c, top = flat_cell()
feeds, feed_series = [], {}
for k, (name, kw) in enumerate((("follow (encoder)", {}), ("fixed, c0 = 5 mm", {"feed": "fixed"}),
                                ("fixed, c0 = 9.5 mm", {"feed": "fixed", "c_start": 9.5e-3}))):
    h = c.add_hole(f"G{k}", top + np.array([*HOLES_XY[k], 0.0]))
    r = c.fasten(h, FastenParams(pick=False, **kw))
    feeds.append({"name": name, **clean(r)})
    feed_series[name] = c.series(r["i0"], r["i1"])
react, react_series = [], {}
for k, ff in enumerate((True, False)):
    h = c.add_hole(f"R{k}", top + np.array([*HOLES_XY[3 + k], 0.0]))
    r = c.fasten(h, FastenParams(pick=False, react_ff=ff))
    react.append({"react_ff": ff, **clean(r)})
    react_series[ff] = c.series(r["i0"], r["i1"])
m66 = pb.calculateMassMatrix(c.sim.id, list(map(float, c.sim.get_state()[0])), physicsClientId=c.cid)[5][5]
c.close()
M["feed"], M["reaction"] = feeds, react
M["M66"] = m66
M["roll_static_est_deg"] = float(np.degrees(DP.T_target / (m66 * c.wn**2)))
print("G", [(f["name"], f["error"], round(f["F_ax_min"], 1), round(f["F_ax_max"], 1)) for f in feeds])
print("H", [(r["react_ff"], r["roll_dev_deg"], r["lat_dev_mm"], r["F_clamp"]) for r in react], M["roll_static_est_deg"])

# ---------------------------------------------------- I. 斜面（20°）＋ JSON 工作
c = ScrewCell()
a = np.radians(TILT_DEG)
nt = np.array([-np.sin(a), 0, np.cos(a)])
topt, _, Rt = c.add_plate("B", (0.40, -0.20, 0.13), (0.12, 0.10), n=nt)
tilted = [clean(c.fasten(c.add_hole(f"B{k}", topt + Rt[:, 1] * y, -nt), FastenParams(pick=False)))
          for k, y in enumerate((-0.02, 0.02))]
top2, _, _ = c.add_plate("A", (0.45, 0.10, 0.10), (0.12, 0.10))
JOB = {"holes": [{"name": "J1", "p": [float(v) for v in top2 + [-0.03, 0, 0]], "torque": 2.0},
                 {"name": "J2", "p": [float(v) for v in top2], "torque": 2.0, "strategy": "angle"},
                 {"name": "J3", "p": [float(v) for v in top2 + [0.03, 0, 0]], "torque": 1.5}]}
job_out = run_job(c, JOB)
c.close()
M["tilted"] = tilted
M["job"] = {"job": JOB, "ok": job_out["ok"], "holes": [clean(x) for x in job_out["holes"]],
            "t_total": job_out["t_total"]}
bad = 0
for j in ({"holes": [{"p": [0, 0]}]}, {"holes": [{"p": [0, 0, 0], "torque": 9}]}, {"holes": [{"p": [0, 0, 0],
          "strategy": "yield"}]}):
    try:
        parse_job(j)
    except ValueError:
        bad += 1
M["job"]["invalid_rejected"] = f"{bad}/3"
print("I", [(t["hole"], t["ok"]) for t in tilted], job_out["ok"])

# ---------------------------------------------------- 圖
plt.rcParams.update({"font.size": 9})
# 1. 標稱 H1：時間序列
fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
t = ser["t"]
ax[0, 0].plot(t, ser["w"] / RPM, label="spindle speed [rpm]")
ax[0, 0].set(xlabel="t [s]", ylabel="rpm", title="H1: spindle speed (600 → 60 rpm, brake)")
a2 = ax[0, 0].twinx()
a2.plot(t, ser["th"] / (2 * np.pi), "C1", lw=1, label="screw turns")
a2.set_ylabel("turns")
ax[0, 0].legend(loc="upper left", fontsize=7)
a2.legend(loc="center right", fontsize=7)
ax[0, 1].plot(t, ser["Tl"], label="transducer torque T")
ax[0, 1].plot(t, ser["Tm"], "--", lw=1, label="motor torque T_m")
ax[0, 1].axhline(DP.T_target, color="r", ls=":", label="target 2.5 N·m")
ax[0, 1].set(xlabel="t [s]", ylabel="N·m", title="H1: torque")
ax[0, 1].legend(fontsize=7)
ax[1, 0].plot(t, ser["F_ax"], label="axial bit force F_ax [N]")
ax[1, 0].set(xlabel="t [s]", ylabel="N", title="H1: feed force (robot follows encoder)")
a3 = ax[1, 0].twinx()
a3.plot(t, ser["c"] * 1e3, "C2", lw=1, label="spring compression [mm]")
a3.set_ylabel("mm")
ax[1, 0].legend(loc="lower left", fontsize=7)
a3.legend(loc="lower right", fontsize=7)
sel = ser["th"] > NOM.theta_s - 2 * np.pi * 0.6
ang = np.degrees(ser["th"][sel] - NOM.theta_s)
ax[1, 1].plot(ang, ser["Tl"][sel], label="torque")
ax[1, 1].set(xlabel="angle from seating [deg]", ylabel="N·m", title="H1: torque–angle (seating zone)")
a4 = ax[1, 1].twinx()
a4.plot(ang, ser["F"][sel] / 1e3, "C3", lw=1, label="clamp force [kN]")
a4.set_ylabel("kN")
ax[1, 1].legend(loc="upper left", fontsize=7)
a4.legend(loc="center left", fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_nominal.png", dpi=105)
plt.close(fig)

# 2. 手臂關節（整個取料＋鎖附 H1 循環）
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
tt = rec["t"][:k0]
for j in range(6):
    ax[0].plot(tt, np.degrees(rec["q"][:k0, j]), label=f"J{j + 1}")
    ax[1].plot(tt, rec["qd"][:k0, j])
    ax[2].plot(tt, rec["tau"][:k0, j])
    ax[3].plot(tt, (rec["q"][:k0, j] - rec["q_ref"][:k0, j]) * 1e3)
ax[0].set(title="joint angle [deg]", xlabel="t [s]")
ax[1].set(title="joint velocity [rad/s]", xlabel="t [s]")
ax[2].set(title="joint torque [N·m]", xlabel="t [s]")
ax[3].set(title="tracking error [mrad]", xlabel="t [s]")
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=105)
plt.close(fig)

# 3. 策略 / 摩擦 / 剛性
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
two = [s for s in strat if s["two_stage"]]
x = np.arange(len(two))
ax[0].bar(x - 0.2, [s["T_err_pct"] for s in two], 0.4, label="simulated overshoot")
ax[0].bar(x + 0.2, [s["dT_est"] / DP.T_target * 100 for s in two], 0.4, label="energy estimate (w_cut)")
ax[0].set_xticks(x, [f"{s['n_fin']}" for s in two])
one = next(s for s in strat if not s["two_stage"])
ax[0].set(xlabel="final speed n_fin [rpm] (two-stage)", ylabel="torque overshoot [%]",
          title=f"Overshoot (single-stage 600 rpm: +{one['T_err_pct']:.0f} %)")
ax[0].set_yscale("log")
ax[0].legend(fontsize=7)
for sname, mk in (("torque", "o-"), ("angle", "s-")):
    rows = [f for f in fric if f["strategy"] == sname]
    ax[1].plot([f["mu"] for f in rows], [f["F_clamp"] / 1e3 for f in rows], mk, label=f"{sname} control")
mus = np.linspace(0.08, 0.16, 30)
ax[1].plot(mus, [ThreadModel(joint=JointSpec(mu_g=u, mu_k=u)).torque_to_force(2.5) / 1e3 for u in mus], "k:",
           lw=1, label="F = T/K'(μ) (VDI 2230)")
Fy = [1.0 / ThreadModel(joint=JointSpec(mu_g=u, mu_k=u)).utilization(1.0) / 1e3 for u in mus]
ax[1].plot(mus, Fy, "r--", lw=1, label="yield (ν = 1)")
ax[1].set(xlabel="friction μ_G = μ_K", ylabel="clamp force [kN]", title="Preload scatter vs friction")
ax[1].legend(fontsize=7)
for lab, (th, T) in curves.items():
    s_ = th > -30
    ax[2].plot(th[s_], T[s_], label=lab)
ax[2].set(xlabel="angle from seating [deg]", ylabel="torque [N·m]", title="Joint rate (two-stage)", xscale="symlog")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_strategy.png", dpi=105)
plt.close(fig)

# 4. 定位 / 傾角 / 進給 / 反作用
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for search, mk in ((False, "x"), (True, "o")):
    rows = [o for o in offs if o["search"] == search]
    ax[0].plot([o["offset_mm"] for o in rows], [1 if o["ok"] else 0 for o in rows], mk + "-",
               label=f"search {'on' if search else 'off'}")
rows = [o for o in offs if o["search"] and o.get("search_s")]
a5 = ax[0].twinx()
a5.plot([o["offset_mm"] for o in rows], [o["search_s"] for o in rows], "C2^:", label="search time [s]")
a5.set_ylabel("s")
ax[0].axvline(M["e_cap_mm"], color="gray", ls=":")
ax[0].set(xlabel="hole position error [mm]", ylabel="success", title="Hole offset (capture 0.75 mm)")
ax[0].legend(fontsize=7, loc="center left")
a5.legend(fontsize=7, loc="center right")
ax[1].plot([r["tilt_cmd_deg"] for r in tilts], [r["turns_rundown"] for r in tilts], "o-")
ax[1].axhline(tilts[0]["turns_nom"], color="g", ls=":", label="expected turns")
ax[1].axvline(S.phi_ct_deg, color="r", ls=":", label=f"φ_ct = {S.phi_ct_deg:.1f}°")
for r in tilts:
    ax[1].text(r["tilt_cmd_deg"], r["turns_rundown"] + 0.4, r["error"] or "OK", fontsize=6, rotation=90)
ax[1].set(xlabel="tool tilt error [deg]", ylabel="turns to snug", title="Cross-thread detection", ylim=(-1, 17))
ax[1].legend(fontsize=7)
for name, sr in feed_series.items():
    ax[2].plot(sr["t"], sr["F_ax"], label=name)
ax[2].set(xlabel="t [s]", ylabel="axial force [N]", title="Feed mode")
ax[2].legend(fontsize=7)
for ff, sr in react_series.items():
    ax[3].plot(sr["t"], np.degrees(sr["roll"] - sr["roll"][0]), label=f"reaction feed-forward {'on' if ff else 'off'}")
ax[3].set(xlabel="t [s]", ylabel="tool roll [deg]", title="Reaction torque on wrist")
ax[3].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_robot.png", dpi=105)
plt.close(fig)

# ---------------------------------------------------- 動畫：6 孔（取料）＋ 斜面 2 孔
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 320)
frames: list = []
phases: list = []


def lab(img, left, right):
    im_ = Image.fromarray(img)
    d_ = ImageDraw.Draw(im_)
    d_.rectangle([0, 0, im_.width, 24], fill=(20, 24, 32))
    d_.text((8, 4), left, fill=(235, 238, 242), font=font)
    d_.text((im_.width - 8 - d_.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im_)


class Rec:
    def __init__(self, title, overview):
        self.title, self.overview, self.last, self.hole = title, overview, -1.0, None

    def __call__(self, cc):
        st = cc._sstate
        fast = st is not None and st["mode"] == "thread"
        searching = bool(st and st.get("searching"))
        period = 0.05 if (fast or searching) else 0.12
        if cc.t - self.last < period - 1e-9:
            return
        self.last = cc.t
        T = cc.tcp_pose()
        tg = self.hole.p if (self.hole is not None and np.linalg.norm(T[:3, 3] - self.hole.p) < 0.08) else T[:3, 3]
        v0 = cc.sim.render(target=self.overview[0], distance=self.overview[3], yaw=self.overview[1],
                           pitch=self.overview[2], size=SIZE, fov=42)
        v1 = cc.sim.render(target=tuple(tg), distance=0.16, yaw=35, pitch=-28, size=SIZE, fov=40)
        if fast and cc.nr is not None:
            txt = f"{cc.nr.phase}  {cc.nr.w / RPM:4.0f} rpm  T {cc.nr.Tl:4.2f} N·m"
            txt2 = f"turns {cc.nr.th / (2 * np.pi):5.2f}  F {NOM.clamp(cc.nr.th) / 1e3:4.2f} kN"
        else:
            txt = "spiral search (hole offset 1.34 mm)" if searching else ("positioning" if st else "move")
            txt2 = ""
        frames.append(np.hstack([lab(v0, self.title, f"t = {cc.t:.1f} s"), lab(v1, txt, txt2)]))
        phases.append(cc.nr.phase if fast and cc.nr is not None else
                      ("search" if searching else "move"))


c, top = flat_cell(render=True)
rcd = Rec("T107 screwdriving: 6x M4, 2.5 N·m", ((0.42, 0.06, 0.20), 55, -26, 1.05))
c.hook = rcd
demo = []
for i, (x, y) in enumerate(HOLES_XY):
    h = c.add_hole(f"H{i + 1}", top + np.array([x, y, 0.0]))
    rcd.hole = h
    err = (1.2e-3, -0.6e-3, 0.0) if i == 4 else (0.0, 0.0, 0.0)  # H5：孔位偏 1.34 mm → 螺旋搜尋
    demo.append(clean(c.fasten(h, FastenParams(hole_err=err))))
c.move_j(c.q_home)
n_flat = len(frames)
c.close()
c = ScrewCell(render=True)
topt, _, Rt = c.add_plate("B", (0.40, -0.20, 0.13), (0.12, 0.10), n=nt)
rcd = Rec("T107: 20 deg inclined plate", ((0.38, -0.12, 0.20), 70, -22, 0.95))
c.hook = rcd
for k, y in enumerate((-0.02, 0.02)):
    h = c.add_hole(f"B{k}", topt + Rt[:, 1] * y, -nt)
    rcd.hole = h
    demo.append(clean(c.fasten(h, FastenParams())))
c.move_j(c.q_home)
c.close()
M["demo"] = demo
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(x).resize((600, 240), Image.LANCZOS) for x in frames[::2]]
pal = [g.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)
spin = [i for i, ph in enumerate(phases) if ph == "run"]
search_f = [i for i, ph in enumerate(phases) if ph == "search"]
keys = (("key_a", spin[len(spin) // 12]), ("key_b", search_f[len(search_f) // 2] if search_f else n_flat // 2),
        ("key_c", spin[-5]))
for nm, idx_ in keys:
    Image.fromarray(frames[idx_]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = len(frames)
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
M["total_s"] = time.time() - t_start
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("demo", [(d["hole"], d["ok"], d.get("blocked")) for d in demo])
print("frames", len(frames), "gif MB", M["gif_MB"], "mp4 MB", M["mp4_MB"], "total s", M["total_s"])
