"""T105 報告：銲接路徑（直線＋圓弧焊道、固定銲槍姿態與速度）→ JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t105_report.py 2026-11-13
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
from scipy.spatial.transform import Rotation  # noqa: E402

from sixaxis import welding as W  # noqa: E402
from sixaxis.welding import WeldCell, WeldParams, run_job, turtle_seam  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T105"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
Z = 0.108  # 鋼板頂面
PLATE = ("plate", (0.50, 0.0, 0.104), (0.30, 0.30, 0.008), (0.58, 0.60, 0.64, 1))
STEEL = (0.62, 0.64, 0.68, 1)
VOLT, AMP, D_WIRE, WFS = 24.0, 180.0, 1.2e-3, 0.10
M: dict = {
    "date": DATE,
    "torch": {k: getattr(W, k) for k in ("NECK_DEG", "BODY_L", "NECK_L", "STICKOUT", "TORCH_MASS", "NOZZLE_R")},
    "tcp_offset_mm": (W.T_TCP[:3, 3] * 1e3).tolist(),
    "process": {"volt": VOLT, "amp": AMP, "d_wire_mm": D_WIRE * 1e3, "wfs_m_min": WFS * 60, "eta": 0.8},
    "defaults": WeldParams().__dict__, "wn": W.WN, "plate_top": Z,
}
DROP = ("i0", "i1", "roll_candidates")


def strip(m):
    return {k: v for k, v in m.items() if k not in DROP}


def j_seam(R=0.04, l1=0.12, l2=0.08, name="J"):
    return turtle_seam((0.40, -0.10, Z), (0, 1, 0), [("line", l1), ("arc", R, -90), ("line", l2)], name=name)


def cell(controller="ctc", render=False):
    c = WeldCell(controller=controller, render=render)
    c.add_box(*PLATE)
    return c


def run(seam, prm, auto=False, setup=None, controller="ctc", keep=False):
    c = WeldCell(controller=controller) if setup else cell(controller)
    try:
        if setup:
            setup(c)
        m = (c.weld_auto if auto else c.weld)(seam, prm, volt=VOLT, amp=AMP)
        if keep and m["ok"]:
            m["series"] = c.series(seam, WeldParams(**{**prm.__dict__, "roll_deg": m["roll_deg"]}), m["i0"], m["i1"])
    finally:
        c.close()
    return m


t_start = time.time()
# ---------------------------------------------------- 1. 標稱：J 形焊道（直線 120 + R40 圓弧 90° + 直線 80 mm）
NOM = WeldParams()
nom = run(j_seam(), NOM, keep=True)
ser = nom.pop("series")
M["nominal"] = strip(nom)
M["nominal"]["bead_area_mm2"] = float(W.bead_area(D_WIRE, WFS, NOM.speed))
M["nominal"]["corner_rate_rad_s"] = NOM.speed / 0.04
print("nominal", {k: nom[k] for k in ("ok", "path_max_mm", "speed_dev_max_pct", "work_err_max_deg", "cycle_s")})

# ---------------------------------------------------- 2. 行進速度掃描
M["speeds"] = []
for v in (0.005, 0.010, 0.020, 0.050, 0.100, 0.110, 0.120, 0.150):
    m = strip(run(j_seam(), WeldParams(speed=v)))
    M["speeds"].append(m)
    print("speed", v, m["ok"], m["error"], m.get("path_max_mm"), m.get("vel_util"))
M["v_max_pred"] = NOM.speed / M["nominal"]["plan_vel_ratio"]  # 速度使用率與 v 成正比 → 上限

# ---------------------------------------------------- 3. 圓角半徑 × 航向平滑
M["corner"] = []
for R in (0.005, 0.010, 0.020, 0.040):
    for b in (0.0, 0.002, 0.005):
        m = strip(run(j_seam(R, 0.06, 0.05), WeldParams(blend=b)))
        m.update(R_mm=R * 1e3, heading_theory_deg=float(np.degrees(b / R / 4)), rate_rad_s=0.010 / R)
        M["corner"].append(m)
        print("corner", R, b, m["ok"], m.get("heading_err_max_deg"), m.get("plan_qdd_max"), m.get("speed_dev_max_pct"))

# ---------------------------------------------------- 4. 銲槍角度矩陣（自動選 roll）
M["angles"] = []
for work in (60.0, 90.0, 120.0):
    for travel in (-15.0, 0.0, 15.0):
        m = strip(run(j_seam(0.04, 0.08, 0.06), WeldParams(speed=0.02, work_deg=work, travel_deg=travel), auto=True))
        M["angles"].append(m)
        print("angle", work, travel, m["ok"], m.get("roll_deg"), m.get("work_err_max_deg"), m.get("travel_err_max_deg"))


# ---------------------------------------------------- 5. T 形接頭填角焊（立板）、6. 管–板整圓、7. 斜板
def setup_fillet(c):
    c.add_box(*PLATE)
    c.add_wall(j_seam(), side=-1.0, rgba=STEEL)


PIPE_C, PIPE_R = (0.50, 0.0), 0.03


def circle_seam(c=PIPE_C, r=PIPE_R, name="circle"):
    return turtle_seam((c[0], c[1] - r, Z), (1, 0, 0), [("arc", r, 180), ("arc", r, 180)], name=name)


def setup_pipe(c):
    c.add_box(*PLATE)
    c.add_cylinder("pipe", (*PIPE_C, Z + 0.03), PIPE_R, 0.06, STEEL)


def incline(th_deg):
    a = np.radians(th_deg)
    n = np.array([-np.sin(a), 0.0, np.cos(a)])
    up = np.array([np.cos(a), 0.0, np.sin(a)])
    c0 = np.array([0.50, 0.0, 0.20])
    seam = turtle_seam(c0 - 0.08 * np.array([0, 1.0, 0]) - 0.05 * up, (0, 1, 0),
                       [("line", 0.08), ("arc", 0.03, 90), ("line", 0.05)], normal=n, name=f"incline{th_deg}")

    def setup(c):
        c.add_box("plate", c0 - n * 0.004, (0.20, 0.24, 0.008), PLATE[3],
                  Rotation.from_matrix(np.column_stack([up, [0, 1, 0], n])).as_quat())
    return seam, setup


M["cases"] = {}
m = run(j_seam(name="fillet"), WeldParams(work_deg=45.0), auto=True, setup=setup_fillet)
M["cases"]["fillet"] = {**strip(m), "rolls": m["roll_candidates"]}
m = run(circle_seam(), WeldParams(work_deg=135.0), auto=True, setup=setup_pipe, keep=True)
ser_c = m.pop("series")
M["cases"]["circle"] = {**strip(m), "rolls": m["roll_candidates"]}
for th in (30, 45, 60):
    seam, setup = incline(th)
    m = run(seam, WeldParams(), auto=True, setup=setup)
    M["cases"][f"incline{th}"] = {**strip(m), "rolls": m["roll_candidates"]}
for k, m in M["cases"].items():
    print("case", k, m["ok"], m["error"], m.get("roll_deg"), m.get("path_max_mm"), m.get("clear_mm"), m["roll_tried"])

# ---------------------------------------------------- 8. 控制器比較
M["controllers"] = []
for ctl in ("ctc", "pd"):
    for v in (0.010, 0.050):
        m = strip(run(j_seam(), WeldParams(speed=v), controller=ctl))
        m["controller"] = ctl
        M["controllers"].append(m)
        print("ctrl", ctl, v, m["path_max_mm"], m["speed_dev_max_pct"])
M["experiments_s"] = time.time() - t_start
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# ---------------------------------------------------- 圖 1：標稱時間歷程
r = W.SixAxisRobot()
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
t = ser["t"]
seam = j_seam()
ax[0, 0].plot(seam.p[:, 0] * 1e3, seam.p[:, 1] * 1e3, color="#bbb", lw=6, label="seam (line 120 + arc R40 + line 80)")
ax[0, 0].plot(ser["p"][::50, 0] * 1e3, ser["p"][::50, 1] * 1e3, color="#c0553b", lw=1.2, label="TCP (PyBullet)")
ax[0, 0].plot(*seam.p[0, :2] * 1e3, "o", color="k", ms=5)
ax[0, 0].annotate("start", seam.p[0, :2] * 1e3 + [4, 0], fontsize=8)
ax[0, 0].set(xlabel="x [mm]", ylabel="y [mm]", title="Seam and TCP path (top view)", aspect="equal")
ax[0, 0].legend(fontsize=8, loc="lower right")
ax[0, 1].plot(ser["ts"], ser["speed"] * 1e3, color="#3b6ea8", label="TCP speed (J qd)")
ax[0, 1].plot(t[::20], ser["sd_ref"][::20] * 1e3, "--", color="k", lw=0.8, label="command")
ax[0, 1].set(xlabel="t [s]", ylabel="speed [mm/s]", title=f"Travel speed (steady dev. max {nom['speed_dev_max_pct']:.3f} %)")
ax[0, 1].legend(fontsize=8, loc="lower center")
ins = ax[0, 1].inset_axes([0.25, 0.3, 0.5, 0.35])
ins.plot(ser["ts"], ser["speed"] * 1e3, color="#3b6ea8")
ins.set(xlim=(11.5, 19), ylim=(9.99, 10.01))
ins.tick_params(labelsize=7)
ins.set_title("zoom: arc section", fontsize=7)
ax[0, 2].plot(t[::10], ser["lat"][::10] * 1e6, label="lateral (in plane)")
ax[0, 2].plot(t[::10], ser["hgt"][::10] * 1e6, label="height (along n)")
ax[0, 2].plot(t[::10], (ser["s_ref"] - ser["s"])[::10] * 1e6, label="along-track lag", color="#888", lw=0.8)
ax[0, 2].set(xlabel="t [s]", ylabel="error [um]", title="TCP error w.r.t. seam geometry")
ax[0, 2].legend(fontsize=8)
ax[1, 0].plot(t[::10], (ser["work"] - NOM.work_deg)[::10], label="work angle error")
ax[1, 0].plot(t[::10], (ser["travel"] - NOM.travel_deg)[::10], label="travel angle error")
ax[1, 0].plot(t[::10], ser["heading"][::10], label="heading error (about n)", lw=0.8)
ax[1, 0].set(xlabel="t [s]", ylabel="[deg]", title="Torch attitude error (work 90 / travel +10 deg)")
ax[1, 0].legend(fontsize=8)
for j in range(6):
    ax[1, 1].plot(t[::10], ser["qd"][::10, j] / r.qd_max[j], label=f"J{j + 1}")
    ax[1, 2].plot(t[::10], ser["tau"][::10, j] / r.tau_max[j], label=f"J{j + 1}")
ax[1, 1].set(xlabel="t [s]", ylabel="qd / qd_max", title="Joint velocity utilisation")
ax[1, 2].set(xlabel="t [s]", ylabel="tau / tau_max", title="Joint torque utilisation (applied)")
ax[1, 1].legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_timeseries.png", dpi=105)

# ---------------------------------------------------- 圖 2：速度掃描 / 圓角 / 控制器
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ok = [m for m in M["speeds"] if m["ok"]]
vv = [m["speed_cmd_mm_s"] for m in ok]
ax[0].loglog(vv, [m["path_max_mm"] * 1e3 for m in ok], "o-", label="path error max [um]")
ax[0].loglog(vv, [m["track_max_mm"] * 1e3 for m in ok], "s-", label="tracking error max [um]")
ax[0].loglog(vv, [m["speed_dev_max_pct"] for m in ok], "^-", label="speed deviation max [%]")
ax[0].loglog([m["speed_cmd_mm_s"] for m in M["speeds"]], [m["plan_vel_ratio"] for m in M["speeds"]], "d--",
             color="k", label="planned qd/qd_max")
ax[0].axhline(1.0, color="k", lw=0.6)
ax[0].axvline(M["v_max_pred"] * 1e3, color="r", ls=":", label=f"v_max = {M['v_max_pred'] * 1e3:.0f} mm/s")
ax[0].set(xlabel="travel speed [mm/s]", title="Speed sweep (J seam, R40 corner)")
ax[0].legend(fontsize=7)
for b, mk in ((0.0, "o"), (2.0, "s"), (5.0, "^")):
    rows = [m for m in M["corner"] if m["blend_mm"] == b]
    ax[1].semilogy([m["R_mm"] for m in rows], [m["plan_qdd_max"] for m in rows], mk + "-", label=f"qdd max, blend {b:.0f} mm")
ax[1].set(xlabel="corner radius R [mm]", ylabel="planned max joint accel [rad/s^2]", title="Corner radius x heading blend (10 mm/s)")
a2 = ax[1].twinx()
for b, mk in ((2.0, "s"), (5.0, "^")):
    rows = [m for m in M["corner"] if m["blend_mm"] == b]
    a2.plot([m["R_mm"] for m in rows], [m["heading_err_max_deg"] for m in rows], mk + ":", color="#888",
            label=f"heading dev., blend {b:.0f} mm")
    a2.plot([m["R_mm"] for m in rows], [m["heading_theory_deg"] for m in rows], "x", color="k", ms=7)
a2.set_ylabel("max heading deviation [deg]  (x = theory b/4R)")
ax[1].legend(fontsize=7, loc="upper center")
a2.legend(fontsize=7, loc="center right")
lab = [f"{m['controller'].upper()}\n{m['speed_cmd_mm_s']:.0f} mm/s" for m in M["controllers"]]
x = np.arange(len(lab))
ax[2].bar(x - 0.2, [m["path_max_mm"] * 1e3 for m in M["controllers"]], 0.4, label="path error max [um]")
ax[2].bar(x + 0.2, [m["speed_dev_max_pct"] for m in M["controllers"]], 0.4, label="speed deviation max [%]")
ax[2].set_yscale("log")
ax[2].set_xticks(x, lab)
ax[2].set(title="Computed torque vs PD + gravity")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_sweeps.png", dpi=105)

# ---------------------------------------------------- 圖 3：整圓焊道關節角 / roll 候選 / 角度矩陣
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for j in range(6):
    ax[0].plot(ser_c["t"][::10], np.degrees(ser_c["q"][::10, j] - ser_c["q"][0, j]), label=f"J{j + 1}")
ax[0].set(xlabel="t [s]", ylabel="q - q(0) [deg]",
          title=f"Pipe-to-plate circle (R30): wrist 3 turns 360 deg, roll {M['cases']['circle']['roll_deg']:.0f}")
ax[0].legend(fontsize=7, ncol=3)
names = list(M["cases"])
w = 0.2
for i, roll in enumerate((0.0, 90.0, 180.0, -90.0)):
    vals, cols = [], []
    for k in names:
        row = next((x_ for x_ in M["cases"][k]["rolls"] if x_["roll_deg"] == roll), None)
        vals.append(row["sigma_min"] if row else 0.0)
        cols.append("#3b6ea8" if row and row["ok"] else "#d9a39a")
    ax[1].bar(np.arange(len(names)) + (i - 1.5) * w, vals, w, color=cols, edgecolor="k", lw=0.4)
    for xk, v_ in zip(np.arange(len(names)) + (i - 1.5) * w, vals, strict=True):
        ax[1].text(xk, v_ + 0.004, f"{roll:.0f}", ha="center", fontsize=6)
for i, k in enumerate(names):
    if M["cases"][k].get("roll_deg") is not None and M["cases"][k]["ok"]:
        j = (0.0, 90.0, 180.0, -90.0).index(M["cases"][k]["roll_deg"])
        ax[1].plot(i + (j - 1.5) * w, -0.008, "v", color="g", ms=7, clip_on=False)
ax[1].set_xticks(np.arange(len(names)), names)
ax[1].set(ylabel="sigma_min(J) along seam", title="Roll candidates (blue = feasible, red = limit; green = executed)")
G = np.array([[next(m for m in M["angles"] if m["work_deg"] == wk and m["travel_deg"] == tr)["sigma_min"]
               for tr in (-15.0, 0.0, 15.0)] for wk in (60.0, 90.0, 120.0)])
im = ax[2].imshow(G, cmap="viridis", origin="lower")
for i, wk in enumerate((60.0, 90.0, 120.0)):
    for j, tr in enumerate((-15.0, 0.0, 15.0)):
        m = next(m for m in M["angles"] if m["work_deg"] == wk and m["travel_deg"] == tr)
        ax[2].text(j, i, f"roll {m['roll_deg']:.0f}\n{m['path_max_mm'] * 1e3:.1f} um\n{m['speed_dev_max_pct']:.2f} %",
                   ha="center", va="center", color="w", fontsize=8)
ax[2].set_xticks(range(3), ["-15 (drag)", "0", "+15 (push)"])
ax[2].set_yticks(range(3), ["60", "90", "120"])
ax[2].set(xlabel="travel angle [deg]", ylabel="work angle [deg]", title="Torch angle matrix: sigma_min (colour), path err, speed dev")
fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_cases.png", dpi=105)

# ---------------------------------------------------- 動畫：JSON 工作（填角焊 J 形 + 管–板整圓）
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 330)
DEMO_PIPE = (0.53, -0.05)
JS, CS = j_seam(name="fillet-J"), circle_seam(DEMO_PIPE, name="pipe-circle")


def seg_json(s):
    out = []
    for pc, s1 in zip(s.pieces, s.s_break[1:], strict=True):
        to = [round(float(x), 6) for x in s.pos(float(s1))]
        if pc.kind == "line":
            out.append({"type": "line", "to": to})
        else:
            out.append({"type": "arc", "via": [round(float(x), 6) for x in s.pos(float(s1 - pc.length / 2))], "to": to})
    return out


JOB = {"seams": [
    {"name": JS.name, "start": [0.40, -0.10, Z], "segments": seg_json(JS), "speed": 0.010, "work_deg": 45, "travel_deg": 10},
    {"name": CS.name, "start": [round(float(x), 6) for x in CS.p[0]], "segments": seg_json(CS), "speed": 0.010,
     "work_deg": 135, "travel_deg": 10},
]}
frames: list = []
state = {"last": -1.0, "cool": 0}


def lab(img, left, right):
    im_ = Image.fromarray(img)
    d_ = ImageDraw.Draw(im_)
    d_.rectangle([0, 0, im_.width, 26], fill=(20, 24, 32))
    d_.text((10, 4), left, fill=(235, 238, 242), font=font)
    d_.text((im_.width - 10 - d_.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im_)


def hook(c):
    if c.t - state["last"] < 0.2 - 1e-9:
        return
    state["last"] = c.t
    tcp = c.rec["tcp"][-1]
    pb.resetBasePositionAndOrientation(glow, [float(x) for x in tcp] if c.arc else [0, 0, -1], [0, 0, 0, 1],
                                       physicsClientId=c.cid)
    n_hot = 5  # 最新的幾顆焊珠維持熾熱色，其餘冷卻成暗灰藍
    while state["cool"] < len(c.beads) - n_hot:
        pb.changeVisualShape(c.beads[state["cool"]], -1, rgbaColor=[0.30, 0.29, 0.33, 1], physicsClientId=c.cid)
        state["cool"] += 1
    spd = np.linalg.norm(c.rec["tcp"][-1] - c.rec["tcp"][-21]) / (20 * c.dt) * 1e3 if len(c.rec["t"]) > 21 else 0.0
    txt = f"ARC ON  {spd:4.1f} mm/s" if c.arc else "arc off (air move)"
    views = [c.sim.render(target=tg, distance=d_, yaw=y, pitch=p_, size=SIZE, fov=42)
             for tg, y, p_, d_ in (((0.35, 0.0, 0.20), 55, -22, 1.25), ((0.47, -0.02, 0.13), 115, -32, 0.55),
                                   ((0.47, -0.02, 0.11), 90, -89, 0.62))]
    frames.append(np.hstack([lab(views[0], "T105 welding: fillet J + pipe circle", f"t = {c.t:.1f} s"),
                             lab(views[1], txt, "close-up"), lab(views[2], "work 45/135, travel +10 deg", "top")]))


c = WeldCell(render=True)
c.add_box(*PLATE)
c.add_wall(JS, side=-1.0, rgba=STEEL)
c.add_cylinder("pipe", (*DEMO_PIPE, Z + 0.03), PIPE_R, 0.06, STEEL)
glow = c.sim.add_marker([0, 0, -1], radius=0.006, rgba=(1.0, 1.0, 0.8, 1))
c.hook = hook
demo = run_job(c, JOB, bead_pitch=0.003)
marks = [next(i for i, tt in enumerate(np.arange(len(frames)) * 0.2) if tt >= 15.0), len(frames) - 1]
for _ in range(5):
    c.wait(0.2)
c.close()
M["demo"] = {"ok": demo["ok"], "t_total": demo["t_total"], "job": JOB, "beads": len(c.beads),
             "seams": [{k: v for k, v in strip(s).items() if k != "q_range_deg"} for s in demo["seams"]]}
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(x).resize((660, 182), Image.LANCZOS) for x in frames[::3]]
pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=150, loop=0, optimize=True)
t_f = np.arange(len(frames)) * 0.2
arc_t = np.array(c.rec["t"])[np.array(c.rec["arc"])]
k_mid2 = int(np.argmin(np.abs(t_f - (arc_t[-1] - 8.0))))  # 第二條焊道進行中
for nm, idx_ in (("key_a", marks[0]), ("key_b", k_mid2), ("key_c", len(frames) - 1)):
    Image.fromarray(frames[idx_]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = len(frames)
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
M["total_s"] = time.time() - t_start
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("demo", demo["ok"], [(s["seam"], s["ok"], s.get("roll_deg"), s.get("path_max_mm"), s.get("clear_mm")) for s in demo["seams"]])
print("frames", len(frames), "gif MB", M["gif_MB"], "mp4 MB", M["mp4_MB"], "total s", M["total_s"])
