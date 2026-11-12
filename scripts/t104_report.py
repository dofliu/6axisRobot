"""T104 報告：輸送帶追蹤抓取 → JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t104_report.py 2026-11-12
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from sixaxis import scene as S  # noqa: E402
from sixaxis import tracking as TR  # noqa: E402
from sixaxis.commands import Executor  # noqa: E402
from sixaxis.graspsim import GraspScene  # noqa: E402
from sixaxis.robot import SixAxisRobot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T104"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
SEED = 2026
PC = S.pocket_centers()
V0 = 0.10  # 標稱帶速
M: dict = {
    "date": DATE, "seed": SEED, "v0": V0,
    "params": {k: getattr(TR, k) for k in ("BELT_Y", "X_FEED", "X_DET", "X_WAIT", "Z_HOVER", "X_END_MAX", "T_SYNC",
                                           "T_DESC", "T_LIFT", "T_GRIP_NOM", "T_GRIP_MAX", "CAM_DT", "ALPHA", "BETA",
                                           "K_CLIK", "MAX_DX", "Y_TOL", "GRIP_DWELL")},
    "v_max_window": TR.v_max_window(), "ab_noise_ratio": TR.alpha_beta_noise_ratio(),
}


def cycle(keep=False, k=0, **kw):
    s = GraspScene(workcell=True)
    try:
        m = TR.run_track_cycle(s, Executor(s), PC[k % 4], **kw)
        if keep:
            m["rec"] = {a: np.array(b) for a, b in s.rec.items()}
    finally:
        s.close()
    return m


def strip(m):
    return {k: v for k, v in m.items() if k not in ("est_log", "rec")}


# ---------------------------------------------------- 1. 標稱（0.10 m/s）＋時間歷程
nom = cycle(keep=True, belt_speed=V0)
M["nominal"] = strip(nom)

# ---------------------------------------------------- 2. 帶速掃描
speeds = []
for v in (0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30):
    m = cycle(belt_speed=v)
    speeds.append({**strip(m), "x_end_formula": TR.x_end_pred(v, m.get("grip_s", TR.T_GRIP_NOM)),
                   "x_end_check": TR.x_end_pred(v)})
M["speeds"] = speeds
M["fallback"] = strip(cycle(belt_speed=0.30, fallback=True))

# ---------------------------------------------------- 3. 帶速誤差：開迴路 vs 閉迴路
mis = []
for eps in (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20):
    for mode in ("open", "track"):
        m = cycle(belt_speed=V0 * (1 + eps), v_nom=V0, mode=mode)
        mis.append({"eps": eps, **strip(m),
                    "pred_end_mm": -eps * V0 * (m["t_grip"] - m["t_det"]) * 1e3 if mode == "open" else 0.0})
M["mismatch"] = mis

# ---------------------------------------------------- 4. 量測雜訊（5 個種子）
noise = []
for sg in (0.0, 0.0005, 0.001, 0.002, 0.005):
    for sd in range(5):
        noise.append({"sigma_mm": sg * 1e3, "seed": sd, **strip(cycle(belt_speed=V0, sigma=sg, seed=sd))})
M["noise"] = noise

# ---------------------------------------------------- 5. 橫向偏移
lat = [{"dy_feed_mm": dy * 1e3, **strip(cycle(belt_speed=V0, y=TR.BELT_Y + dy))}
       for dy in (-0.020, -0.012, -0.008, 0.0, 0.008, 0.012, 0.020)]
M["lateral"] = lat

# ---------------------------------------------------- 6. 隨機情境 N=30（帶速、帶速誤差、橫向偏移、雜訊 1 mm）
rng = np.random.default_rng(SEED)
rand = []
for k in range(30):
    v, eps, dy = rng.uniform(0.05, 0.25), rng.uniform(-0.10, 0.10), rng.uniform(-0.010, 0.010)
    for mode in ("track", "open"):
        m = cycle(k=k, belt_speed=v, v_nom=v / (1 + eps), mode=mode, y=TR.BELT_Y + dy, sigma=0.001, seed=k)
        rand.append({"trial": k, "v": v, "eps": eps, "dy_mm": dy * 1e3, **strip(m)})
M["rand"] = rand

# ---------------------------------------------------- 7. 連續來料（4 件 → 4 格）
flows = []
for per in (8.0, 7.0, 6.0, 5.0):
    s = GraspScene(workcell=True)
    f = TR.run_flow(s, Executor(s), 4, per, V0, PC)
    s.close()
    f.pop("bodies")
    flows.append(f)
M["flows"] = flows

# ---------------------------------------------------- 圖 1：標稱時間歷程
rec = nom["rec"]
t = rec["t"]
i0, i1 = np.searchsorted(t, nom["t_det"] - 0.1), np.searchsorted(t, nom["t_end"])
sl = slice(i0, i1)
tt = t[sl] - nom["t_det"]
ph = [("sync", nom["t_sync"], nom["t_sync"] + TR.T_SYNC, "#dbe7f5"),
      ("descend", nom["t_sync"] + TR.T_SYNC, nom["t_grip"], "#e3f1df"),
      ("grip", nom["t_grip"], nom["t_grip"] + nom["grip_s"], "#f8e3c6"),
      ("lift", nom["t_grip"] + nom["grip_s"], nom["t_end"], "#eedff2")]
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a in ax.flat:
    for nm, a0, a1, c in ph:
        a.axvspan(a0 - nom["t_det"], a1 - nom["t_det"], color=c, lw=0)
for nm, a0, a1, c in ph:
    ax[0, 0].text((a0 + a1) / 2 - nom["t_det"], 0.335, nm, ha="center", fontsize=8)
ax[0, 0].plot(tt, rec["obj"][sl, 0], label="part x (actual)", lw=2)
ax[0, 0].plot(tt, rec["tcp"][sl, 0], "--", label="TCP x", lw=1.5)
el = np.array(nom["est_log"])
ax[0, 0].plot(el[::3, 0] - nom["t_det"], el[::3, 1], ".", ms=3, color="k", label="estimate x̂ (alpha-beta)")
ax[0, 0].set(xlabel="t - t_det [s]", ylabel="x [m]", title=f"Belt-direction position, v = {V0} m/s", ylim=(0.19, 0.345))
ax[0, 0].legend(fontsize=8, loc="lower right")
d = (rec["tcp"][sl, :2] - rec["obj"][sl, :2]) * 1e3
ax[0, 1].plot(tt, d[:, 0], label="x (belt direction)")
ax[0, 1].plot(tt, d[:, 1], label="y (lateral)")
ax[0, 1].set(xlabel="t - t_det [s]", ylabel="TCP - part [mm]", title="Tracking error", ylim=(-6, 6))
ax[0, 1].legend(fontsize=8)
ax[0, 2].plot(tt, rec["tcp"][sl, 2] * 1e3, color="#555", label="TCP z [mm]")
ax[0, 2].set(xlabel="t - t_det [s]", ylabel="TCP z [mm]", title="TCP height / finger normal force")
a2 = ax[0, 2].twinx()
a2.plot(tt, rec["fL"][sl], color="#c0553b", lw=1, label="finger L [N]")
a2.plot(tt, rec["fR"][sl], color="#3b6ea8", lw=1, label="finger R [N]")
a2.set_ylabel("force [N]")
a2.legend(fontsize=8, loc="center right")
q = rec["q"][sl]
dt = t[1] - t[0]
qd = np.gradient(q, dt, axis=0)
ker = np.ones(15) / 15
qdd = np.apply_along_axis(lambda x: np.convolve(x, ker, "same"), 0, np.gradient(qd, dt, axis=0))
r = SixAxisRobot()
idx = np.arange(0, len(q), 5)
tau = np.array([r.rne(q[i], qd[i], qdd[i]) for i in idx])
for j in range(6):
    ax[1, 0].plot(tt, np.degrees(q[:, j] - q[0, j]), label=f"J{j + 1}")
    ax[1, 1].plot(tt, qd[:, j] / r.qd_max[j], label=f"J{j + 1}")
    ax[1, 2].plot(tt[idx], tau[:, j] / r.tau_max[j], label=f"J{j + 1}")
ax[1, 0].set(xlabel="t - t_det [s]", ylabel="q - q(0) [deg]", title="Joint angles (relative)")
ax[1, 1].set(xlabel="t - t_det [s]", ylabel="qd / qd_max", title="Joint velocity utilisation")
ax[1, 2].set(xlabel="t - t_det [s]", ylabel="tau / tau_max", title="Joint torque utilisation (RNE model)")
ax[1, 0].legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_timeseries.png", dpi=105)

# ---------------------------------------------------- 圖 2：帶速掃描 / 帶速誤差
fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
okv = [s for s in speeds if s["success"]]
ax[0].plot([s["belt_speed"] for s in okv], [s["track_rms_mm"] for s in okv], "o-", label="tracking RMS (descent)")
ax[0].plot([s["belt_speed"] for s in okv], [s["track_max_mm"] for s in okv], "s-", label="tracking max")
ax[0].plot([s["belt_speed"] for s in okv], [abs(s["dx_mm"]) for s in okv], "^-", label="|in-hand dx|")
ax[0].plot([s["belt_speed"] for s in okv], [s["place"]["xy_err_mm"] for s in okv], "d-", label="place XY error")
ax[0].axvline(M["v_max_window"], color="#c0553b", ls="--", lw=1)
ax[0].text(M["v_max_window"] - 0.004, 0.05, "window limit", rotation=90, ha="right", color="#c0553b", fontsize=8)
for s in speeds:
    if not s["success"]:
        ax[0].plot(s["belt_speed"], 0.0, "x", color="#c0553b", ms=9, mew=2)
ax[0].set(xlabel="belt speed [m/s]", ylabel="[mm]", title="Belt speed sweep (x = rejected: WINDOW)")
ax[0].legend(fontsize=8)
vv = np.linspace(0.02, 0.32, 100)
ax[1].plot(vv, [TR.x_end_pred(v) for v in vv], color="#888", label="x_end model (t_grip nominal)")
ax[1].plot([s["belt_speed"] for s in okv], [s["x_end"] for s in okv], "o", color="#3b6ea8", label="simulation")
ax[1].axhline(TR.X_END_MAX, color="#c0553b", ls="--", lw=1, label="X_END_MAX")
ax[1].set(xlabel="belt speed [m/s]", ylabel="TCP x at end of lift [m]", title="Tracking window")
ax[1].legend(fontsize=8)
for mode, c, mk in (("open", "#c0553b", "o"), ("track", "#3b6ea8", "s")):
    mm = [x for x in mis if x["mode"] == mode]
    ax[2].plot([x["eps"] * 100 for x in mm], [x["track_end_mm"] for x in mm], mk, color=c, ms=7,
               label=f"{mode} loop (simulation)")
ee = np.linspace(-0.2, 0.2, 2)
dtm = np.mean([x["t_grip"] - x["t_det"] for x in mis if x["mode"] == "open"])
ax[2].plot(ee * 100, -ee * V0 * dtm * 1e3, "-", color="#c0553b", lw=1, label="open loop: -eps*v*(t_grip - t_det)")
ax[2].axhspan(-TR.MAX_DX * 1e3, TR.MAX_DX * 1e3, color="#e3f1df", lw=0, zorder=0, label="compensable in-hand offset")
ax[2].set(xlabel="belt speed error eps [%]", ylabel="TCP - part at grip [mm]", title="Speed mismatch: open vs closed loop")
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_speed.png", dpi=105)

# ---------------------------------------------------- 圖 3：雜訊 / 隨機情境 / 連續來料
fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
sgs = sorted({n["sigma_mm"] for n in noise})
rms = [[n["track_rms_mm"] for n in noise if n["sigma_mm"] == s] for s in sgs]
ax[0].errorbar(sgs, [np.mean(x) for x in rms], yerr=[np.std(x) for x in rms], fmt="o-", capsize=3,
               label="tracking RMS (5 seeds)")
dxs = [[abs(n["dx_mm"]) for n in noise if n["sigma_mm"] == s] for s in sgs]
ax[0].plot(sgs, [np.max(x) for x in dxs], "s--", label="max |in-hand dx|")
ax[0].plot(sgs, np.sqrt(M["ab_noise_ratio"]) * np.array(sgs), ":", color="#888", label="alpha-beta steady state: 0.46 sigma")
ax[0].set(xlabel="measurement noise sigma [mm]", ylabel="[mm]", title="Camera noise (v = 0.10 m/s)")
ax[0].legend(fontsize=8)
for mode, c, mk in (("open", "#c0553b", "o"), ("track", "#3b6ea8", "s")):
    rr = [x for x in rand if x["mode"] == mode and "track_end_mm" in x]
    ax[1].scatter([x["eps"] * 100 for x in rr], [x["track_end_mm"] for x in rr], color=c, marker=mk, s=28,
                  label=f"{mode}: {sum(x['success'] for x in rand if x['mode'] == mode)}/30 placed")
ax[1].axhspan(-TR.MAX_DX * 1e3, TR.MAX_DX * 1e3, color="#e3f1df", lw=0, zorder=0)
ax[1].set(xlabel="belt speed error eps [%]", ylabel="TCP - part at grip [mm]",
          title="Random scenarios N=30 (v 0.05-0.25, sigma 1 mm)")
ax[1].legend(fontsize=8)
col = {"track": "#3b6ea8", "static": "#d39a2a"}
for r_, f in enumerate(flows):
    for k, it in enumerate(f["items"]):
        ax[2].barh(r_ + (k - 1.5) * 0.19, it["t_done"] - it["t_feed"], left=it["t_feed"], height=0.17,
                   color=col[it["method"]], edgecolor="k", lw=0.5)
ax[2].set_yticks(range(len(flows)), [f"period {f['period_s']:.0f} s" for f in flows])
ax[2].barh(0, 0, color=col["track"], label="tracked pick")
ax[2].barh(0, 0, color=col["static"], label="static pick at stopper")
ax[2].set(xlabel="time [s]", title="Continuous feed: part residence (feed -> placed)")
ax[2].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_stats.png", dpi=105)

# ---------------------------------------------------- 動畫：連續來料 4 件 → 高速 0.25 → 0.30 超窗改靜態抓取
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 330)
frames: list = []
state = {"txt": "", "sub": "", "last": -1.0}
RGBA = [(0.85, 0.25, 0.2, 1), (0.95, 0.6, 0.15, 1), (0.3, 0.65, 0.35, 1), (0.2, 0.5, 0.85, 1)]


def lab(img, left, right):
    im = Image.fromarray(img)
    d_ = ImageDraw.Draw(im)
    d_.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d_.text((10, 4), left, fill=(235, 238, 242), font=font)
    d_.text((im.width - 10 - d_.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im)


def cb(sc):
    if sc.t - state["last"] < 0.1 - 1e-9:
        return
    state["last"] = sc.t
    views = [sc.sim.render(target=tg, distance=d_, yaw=y, pitch=p_, size=SIZE, fov=42)
             for tg, y, p_, d_ in (((0.38, 0.2, 0.15), 60, -25, 1.25), ((0.35, 0.45, 0.15), 0, -12, 0.85),
                                   ((0.4, 0.2, 0.1), 90, -75, 1.15))]
    frames.append(np.hstack([lab(views[0], state["txt"], f"t = {sc.t:.1f} s"),
                             lab(views[1], state["sub"], "belt side"),
                             lab(views[2], "belt -> tray", "top")]))


demo, marks = {}, []
state.update(txt="T104 conveyor tracking: 0.10 m/s", sub="4 parts, 7 s period", last=-1.0)
sc = GraspScene(workcell=True, render=True)
ex = Executor(sc, frame_cb=cb, cb_every=25)
f = TR.run_flow(sc, ex, 4, 7.0, V0, PC, hook=cb, rgbas=RGBA)
for _ in range(8):
    sc.wait(0.1)
    cb(sc)
sc.close()
f.pop("bodies")
demo["flow"] = f
marks.append(len(frames))
for key, v, fb, sub in (("fast", 0.25, False, "tracked pick near the window limit"),
                        ("too_fast", 0.30, True, "outside window -> static pick")):
    state.update(txt=f"T104 belt speed {v:.2f} m/s", sub=sub, last=-1.0)
    sc = GraspScene(workcell=True, render=True)
    ex = Executor(sc, frame_cb=cb, cb_every=25)
    m = TR.run_track_cycle(sc, ex, PC[0], v, fallback=fb, hook=cb)
    for _ in range(8):
        sc.wait(0.1)
        cb(sc)
    sc.close()
    demo[key] = strip(m)
    marks.append(len(frames))
M["demo"] = demo
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(x).resize((660, 182), Image.LANCZOS) for x in frames[::3]]  # GIF 每 3 幀取 1 以控制在 8 MB 內
pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=150, loop=0, optimize=True)
# 關鍵畫面：第一件追蹤夾取中、連續來料結束、超窗後靜態抓取結束
for nm, idx_ in (("key_a", min(22, marks[0] - 1)), ("key_b", marks[0] - 1), ("key_c", marks[2] - 1)):
    Image.fromarray(frames[idx_]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = len(frames)
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

KEYS = ("success", "error", "track_rms_mm", "track_max_mm", "track_end_mm", "dx_mm", "grip_s", "x_end", "cycle_s")


def show(tag, m):
    print(tag, {k: (round(float(m[k]), 3) if isinstance(m.get(k), float) else m.get(k)) for k in KEYS},
          round(m["place"]["xy_err_mm"], 3) if "place" in m else None)


show("nominal", nom)
for s in speeds:
    show(f"v={s['belt_speed']}", s)
show("fallback", M["fallback"])
for x in mis:
    show(f"eps={x['eps']} {x['mode']} pred={x['pred_end_mm']:.2f}", x)
for sg in sgs:
    nn = [n for n in noise if n["sigma_mm"] == sg]
    print("noise", sg, sum(n["success"] for n in nn), np.mean([n["track_rms_mm"] for n in nn]),
          max(abs(n["dx_mm"]) for n in nn))
for x in lat:
    show(f"dy={x['dy_feed_mm']}", x)
for mode in ("track", "open"):
    rr = [x for x in rand if x["mode"] == mode]
    print("rand", mode, sum(x["success"] for x in rr), [x["error"] for x in rr if not x["success"]])
for f in flows + [demo["flow"]]:
    print("flow", f["period_s"], f["placed"], f["tracked"], f["static"], round(f["total_s"], 1), round(f["throughput_ppm"], 2))
print("demo", {k: v.get("success") for k, v in demo.items()}, M["frames"], M["gif_MB"], M["mp4_MB"])
