"""T101 報告：點到點搬運 → JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t101_report.py 2026-11-06
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

from sixaxis.commands import Executor  # noqa: E402
from sixaxis.graspsim import GraspScene  # noqa: E402
from sixaxis.tasks import random_pairs, run_transfer, transfer_program  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T101"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
N, SEED = 30, 2026
FIX = ((0.36, 0.0), (0.44, 0.015))  # 速度/曲線掃描用固定起訖點（距離 81 mm）
M: dict = {"date": DATE, "n": N, "seed": SEED, "fixed_pair": FIX}


def one(src, dst, **kw):
    s = GraspScene(workcell=True)
    try:
        m = run_transfer(s, Executor(s), src, dst, **kw)
    finally:
        s.close()
    m["dry_cycle_s"] = Executor().run(transfer_program(src, dst, **kw)).t_total
    return m


def strip(m):
    return {k: v for k, v in m.items() if k != "results"}


# ---------------------------------------------------- 1. 隨機起訖點批次
batch = [one(a, b) for a, b in random_pairs(N, SEED)]
ok = [m for m in batch if m["success"]]
arr = lambda k: np.array([m[k] for m in batch])  # noqa: E731
M["batch"] = {
    "success": len(ok), "rate": len(ok) / N,
    "xy_mean": float(arr("xy_err_mm").mean()), "xy_max": float(arr("xy_err_mm").max()),
    "xy_rms": float(np.sqrt((arr("xy_err_mm") ** 2).mean())),
    "z_abs_max": float(np.abs(arr("z_err_mm")).max()), "tilt_max": float(arr("tilt_deg").max()),
    "cycle_mean": float(arr("cycle_s").mean()), "cycle_min": float(arr("cycle_s").min()),
    "cycle_max": float(arr("cycle_s").max()), "dry_cycle_mean": float(arr("dry_cycle_s").mean()),
    "vel_util_max": float(arr("vel_util").max()), "torque_util_max": float(arr("torque_util").max()),
    "tcp_err_max": float(arr("tcp_err_mm").max()),
    "dist_mean": float(arr("dist_mm").mean()),
    "trials": [strip(m) for m in batch],
}

# ---------------------------------------------------- 2. 速度 × 曲線掃描
sweep = []
for prof in ("quintic", "trapezoid"):
    for spd in (0.4, 0.6, 0.8, 1.0):
        m = one(*FIX, speed=spd, profile=prof)
        sweep.append({"profile": prof, "speed": spd, **strip(m)})
M["sweep"] = sweep

# ---------------------------------------------------- 3. 夾持力掃描（搬運中是否掉落）
force = []
for f in (5.0, 10.0, 20.0, 40.0, 80.0):
    m = one(*FIX, force=f)
    force.append({"force": f, "ok": m["ok"], "error": m["error"], "success": m["success"],
                  "xy_err_mm": m["xy_err_mm"], "cycle_s": m["cycle_s"]})
M["force"] = force

# ---------------------------------------------------- 圖
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].hist(arr("xy_err_mm"), bins=12, color="#3b6ea8")
ax[0].set(xlabel="place XY error [mm]", ylabel="count", title=f"Placement error (N={N})")
ax[1].hist(arr("cycle_s"), bins=12, color="#4a9a6a")
ax[1].set(xlabel="cycle time [s]", title="Cycle time")
ax[2].scatter(arr("dist_mm"), arr("cycle_s"), color="#a86a3b")
ax[2].set(xlabel="transfer distance [mm]", ylabel="cycle time [s]", title="Cycle vs distance")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_batch.png", dpi=110)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for prof, c in (("quintic", "#3b6ea8"), ("trapezoid", "#c0553b")):
    r = [x for x in sweep if x["profile"] == prof]
    sp = [x["speed"] for x in r]
    ax[0].plot(sp, [x["cycle_s"] for x in r], "o-", color=c, label=prof)
    ax[1].plot(sp, [x["torque_util"] for x in r], "o-", color=c, label=prof)
    ax[2].plot(sp, [x["xy_err_mm"] for x in r], "o-", color=c, label=prof)
for a, y in zip(ax, ("cycle time [s]", "peak torque utilisation", "place XY error [mm]"), strict=True):
    a.set(xlabel="speed factor", ylabel=y)
    a.legend()
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_sweep.png", dpi=110)

# ---------------------------------------------------- 動畫（單次示範）
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 330)
frames: list = []


def lab(img, left, right):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d.text((10, 4), left, fill=(235, 238, 242), font=font)
    d.text((im.width - 10 - d.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im)


def cb(sc):
    views = [sc.sim.render(target=(0.42, 0.0, 0.14), distance=1.0, yaw=y, pitch=p_, size=SIZE, fov=42)
             for y, p_ in ((35, -28), (90, -14), (180, -78))]
    frames.append(np.hstack([lab(views[0], "T101 point-to-point transfer", f"t = {sc.t:.1f} s"),
                             lab(views[1], f"holding={ex.holding}", "side"),
                             lab(views[2], "src (0.36,0) -> dst (0.44,0.015)", "top")]))


sc = GraspScene(workcell=True, render=True)
ex = Executor(sc, frame_cb=cb, cb_every=50)
demo = run_transfer(sc, ex, *FIX)
rec = {k: np.array(v) for k, v in sc.rec.items()}
sc.close()
M["demo"] = strip(demo)
M["demo_results"] = demo["results"]

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for k, n in enumerate("xyz"):
    ax[0].plot(rec["t"], rec["tcp"][:, k] * 1e3, label=f"TCP {n}")
for r in demo["results"]:
    if r["cmd"] in ("grip", "release"):
        ax[0].axvline(r["t_end"] if r["cmd"] == "grip" else r["t_start"], color="k", ls=":", lw=0.8)
ax[0].set(ylabel="TCP [mm]", title="Demo transfer: TCP position (dotted: grip / release)")
ax[0].legend(ncol=3)
ax[1].plot(rec["t"], rec["fL"], label="finger L")
ax[1].plot(rec["t"], rec["fR"], label="finger R")
ax[1].set(xlabel="t [s]", ylabel="contact force [N]")
ax[1].legend()
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_demo_tcp_forces.png", dpi=110)

imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((720, 200), Image.LANCZOS) for f in frames]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=50, loop=0, optimize=True)
n = len(frames)
for nm, idx in (("key_a", n // 6), ("key_b", int(n * 0.5)), ("key_c", int(n * 0.95))):
    Image.fromarray(frames[idx]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = n
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in M.items() if k not in ("batch", "demo_results")}, indent=1, ensure_ascii=False)[:3500])
print({k: v for k, v in M["batch"].items() if k != "trials"})
