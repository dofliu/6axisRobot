"""T102 報告：取放 pick & place（料盤格）→ JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t102_report.py 2026-11-10
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
from sixaxis.commands import Executor  # noqa: E402
from sixaxis.graspsim import GraspScene  # noqa: E402
from sixaxis.tasks import ZONE_X, ZONE_Y, run_pick_place  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T102"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
PC = S.pocket_centers()
N, SEED = 24, 2026
M: dict = {"date": DATE, "n": N, "seed": SEED, "pockets": PC}
BASE = (0.40, 0.0)


def one(true, place, **kw):
    s = GraspScene(workcell=True)
    try:
        m = run_pick_place(s, Executor(s), true, place, **kw)
    finally:
        s.close()
    return m


def strip(m):
    return {k: v for k, v in m.items() if k != "results"}


# ---------------------------------------------------- 1. 標稱批次：隨機取點 → 4 個料盤格（無偏差）
rng = np.random.default_rng(SEED)
batch = []
for i in range(N):
    true = (float(rng.uniform(*ZONE_X)), float(rng.uniform(*ZONE_Y)))
    m = one(true, PC[i % 4])
    batch.append({"pocket": i % 4, "pick": list(true), **strip(m)})
arr = lambda k: np.array([m[k] for m in batch])  # noqa: E731
M["batch"] = {
    "success": int(sum(m["success"] for m in batch)), "n": N,
    "xy_mean": float(arr("xy_err_mm").mean()), "xy_max": float(arr("xy_err_mm").max()),
    "z_abs_max": float(np.abs(arr("z_err_mm")).max()), "tilt_max": float(arr("tilt_deg").max()),
    "yaw_max": float(arr("yaw_deg").max()),
    "cycle_mean": float(arr("cycle_s").mean()), "torque_util_max": float(arr("torque_util").max()),
    "vel_util_max": float(arr("vel_util").max()),
    "per_pocket": [{"pocket": k, "success": int(sum(m["success"] for m in batch if m["pocket"] == k)),
                    "n": int(sum(m["pocket"] == k for m in batch))} for k in range(4)],
    "trials": batch,
}

# ---------------------------------------------------- 2. 取點偏差掃描（x / y 軸，無恢復 vs 有恢復）
OFFS = [-20, -18, -16, -14, -12, -8, -4, 0, 4, 8, 12, 14, 16, 18, 20]
sweep = []
for ax in (0, 1):
    for rec in (False, True):
        for d in OFFS:
            cmd = (BASE[0] + (d * 1e-3 if ax == 0 else 0), BASE[1] + (d * 1e-3 if ax == 1 else 0))
            m = one(BASE, PC[1], cmd_pick=cmd, recover=rec)
            sweep.append({"axis": "xy"[ax], "cmd_offset_mm": d, "recover": rec, **strip(m)})
M["sweep"] = sweep

# ---------------------------------------------------- 3. 隨機感測誤差批次（均勻 ±12 mm，兩軸）
rng = np.random.default_rng(SEED + 1)
NE = 30
errs = [(float(rng.uniform(-12, 12)), float(rng.uniform(-12, 12))) for _ in range(NE)]
rand = {"raw": [], "recover": []}
for i, (ex_, ey_) in enumerate(errs):
    true = (float(rng.uniform(*ZONE_X)), float(rng.uniform(*ZONE_Y)))
    cmd = (true[0] + ex_ * 1e-3, true[1] + ey_ * 1e-3)
    for key, rec in (("raw", False), ("recover", True)):
        m = one(true, PC[i % 4], cmd_pick=cmd, recover=rec)
        rand[key].append({"err_mm": [ex_, ey_], "pocket": i % 4, **strip(m)})
M["rand"] = {"n": NE, "range_mm": 12}
for key in rand:
    r = rand[key]
    M["rand"][key] = {
        "success": int(sum(m["success"] for m in r)), "xy_mean": float(np.mean([m["xy_err_mm"] for m in r])),
        "xy_max": float(np.max([m["xy_err_mm"] for m in r])), "cycle_mean": float(np.mean([m["cycle_s"] for m in r])),
        "attempts_mean": float(np.mean([m["attempts"] for m in r])),
        "retry_cases": int(sum(m["attempts"] > 1 for m in r)), "trials": r,
    }

# ---------------------------------------------------- 圖
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
ax[0].hist(arr("xy_err_mm"), bins=10, color="#3b6ea8")
ax[0].set(xlabel="place XY error [mm]", ylabel="count", title=f"Nominal batch (N={N})")
ax[1].hist(arr("cycle_s"), bins=10, color="#4a9a6a")
ax[1].set(xlabel="cycle time [s]", title="Cycle time")
ax[2].scatter([p[0] for p in PC], [p[1] for p in PC], marker="s", s=500, facecolor="none", edgecolor="#888")
for k in range(4):
    sub = [m for m in batch if m["pocket"] == k]
    ax[2].scatter([PC[k][0]] * len(sub), [PC[k][1]] * len(sub), alpha=0)
    ax[2].annotate(f"pocket {k}\n{M['batch']['per_pocket'][k]['success']}/{M['batch']['per_pocket'][k]['n']}", PC[k],
                   ha="center", va="center", fontsize=8)
ax[2].set(xlabel="x [m]", ylabel="y [m]", title="Tray pockets (success)", aspect="equal")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_batch.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, axn in zip(ax, "xy", strict=True):
    for rec, c, lab_ in ((False, "#c0553b", "no recovery"), (True, "#3b6ea8", "with regrasp")):
        r = [x for x in sweep if x["axis"] == axn and x["recover"] == rec]
        a.plot([x["cmd_offset_mm"] for x in r], [1.0 if x["success"] else 0.0 for x in r], "o-", color=c, label=lab_,
               alpha=0.85)
    a.set(xlabel=f"commanded pick offset along {axn} [mm]", title=f"Pick offset tolerance ({axn})")
    a.set_yticks([0, 1], ["fail", "success"])
    a.legend(loc="center right")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_offset_sweep.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
for a, key, ttl in zip(ax, ("raw", "recover"), ("no recovery", "with regrasp"), strict=True):
    for m in rand[key]:
        a.scatter(m["err_mm"][0], m["err_mm"][1], c="#3b6ea8" if m["success"] else "#c0553b", s=36,
                  marker="o" if m["attempts"] == 1 else "^")
    a.set(xlabel="sensing error x [mm]", title=f"{ttl}: {M['rand'][key]['success']}/{NE} success")
    a.axhspan(-14, 14, color="#4a9a6a", alpha=0.08)
ax[0].set_ylabel("sensing error y [mm]")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_random_error.png", dpi=110)

# ---------------------------------------------------- 動畫：x 偏差 12 mm → 偵測偏置 → 放回重取 → 放入料盤格 2
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
    views = [sc.sim.render(target=(0.43, -0.08, 0.14), distance=1.05, yaw=y, pitch=p_, size=SIZE, fov=42)
             for y, p_ in ((35, -28), (90, -14), (180, -78))]
    frames.append(np.hstack([lab(views[0], "T102 pick & place + regrasp", f"t = {sc.t:.1f} s"),
                             lab(views[1], f"holding={ex.holding}", "side"),
                             lab(views[2], "pick 12 mm off in x -> pocket 2", "top")]))


sc = GraspScene(workcell=True, render=True)
ex = Executor(sc, frame_cb=cb, cb_every=50)
cmd_demo = (BASE[0] + 0.012, BASE[1])
demo = run_pick_place(sc, ex, BASE, PC[2], cmd_pick=cmd_demo, recover=True)
sc.close()
M["demo"] = strip(demo)
M["demo_results"] = demo["results"]
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((720, 200), Image.LANCZOS) for f in frames]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=50, loop=0, optimize=True)
n = len(frames)
for nm, idx in (("key_a", n // 6), ("key_b", int(n * 0.45)), ("key_c", int(n * 0.97))):
    Image.fromarray(frames[idx]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = n
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print({k: v for k, v in M["batch"].items() if k not in ("trials",)})
print(M["rand"]["raw"]["success"], M["rand"]["recover"]["success"], M["demo"]["success"], M["demo"]["log"])
for x in sweep:
    print(x["axis"], x["cmd_offset_mm"], x["recover"], x["success"], x["error"], x["attempts"], round(x["xy_err_mm"], 1))
