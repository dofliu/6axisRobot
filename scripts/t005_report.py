"""T005 報告：夾爪抓取/抬升/釋放實驗 → JSON/圖/動畫/報告 md。
用法：PYTHONPATH=. python3 scripts/t005_report.py 2026-11-03
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

from sixaxis.graspsim import GraspScene, run_pick_lift, static_slip  # noqa: E402
from sixaxis.gripper import BLOCK, CUBE, CYLINDER, f_min  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T005"
MEDIA, DATA, REP = ROOT / "docs/media", ROOT / "docs/test-reports/data", ROOT / "docs/test-reports"
DATA.mkdir(exist_ok=True)
M: dict = {"date": DATE}

# ---------------------------------------------------------------- 1. 三種物件取放
objs = {}
traces = {}
for spec, lift in ((CUBE, 0.15), (BLOCK, 0.15), (CYLINDER, 0.15)):
    s, m = run_pick_lift(spec, force=40.0, lift=lift, T_lift=1.0)
    m["f_min_N"] = f_min(spec.mass, spec.friction, m["a_peak_ms2"])
    objs[spec.name] = m
    traces[spec.name] = {k: np.array(v) for k, v in s.rec.items()}
    s.close()
M["objects"] = objs

# ---------------------------------------------------------------- 2. 靜態夾持力下限掃描
F_STAT = [0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0, 3.0]
stat = [{"F_N": f, "drop_mm": static_slip(CUBE, f) * 1e3} for f in F_STAT]
M["static_sweep"] = {"f_min_analytic_N": f_min(CUBE.mass, CUBE.friction), "rows": stat,
                     "threshold_sim_N": min(r["F_N"] for r in stat if r["drop_mm"] < 0.1)}

# ---------------------------------------------------------------- 3. 動態（快速抬升 T=0.5 s）掃描
dyn = []
for f in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0):
    s, m = run_pick_lift(CUBE, force=f, lift=0.15, T_lift=0.5)
    dyn.append({"F_N": f, "a_peak": m["a_peak_ms2"], "slip_mm": m["slip_lift_mm"], "held": m["held"],
                "f_min_N": f_min(CUBE.mass, CUBE.friction, m["a_peak_ms2"])})
    s.close()
M["dynamic_sweep"] = dyn

# ---------------------------------------------------------------- 圖
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for nm, tr in traces.items():
    t = tr["t"]
    ax[0].plot(t, (tr["obj"][:, 2] - tr["obj"][0, 2]) * 1e3, label=nm)
    ax[1].plot(t, tr["fL"], label=f"{nm} L")
ax[0].set(xlabel="t [s]", ylabel="object z rise [mm]", title="Pick-lift-place-release")
ax[0].legend(fontsize=8)
ax[1].set(xlabel="t [s]", ylabel="left pad normal force [N]", title="Grip force (command 40 N)")
ax[1].legend(fontsize=7)
tr = traces["cube50"]
rel = (tr["obj"] - tr["tcp"]) * 1e3
ax[2].plot(tr["t"], rel[:, 0], label="x")
ax[2].plot(tr["t"], rel[:, 1], label="y")
ax[2].plot(tr["t"], rel[:, 2], label="z")
ax[2].set(xlabel="t [s]", ylabel="cube - TCP [mm]", title="Cube offset relative to gripper")
ax[2].legend()
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_grasp.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
fs = [r["F_N"] for r in stat]
ax[0].semilogy(fs, [max(r["drop_mm"], 1e-3) for r in stat], "o-")
ax[0].axvline(M["static_sweep"]["f_min_analytic_N"], color="r", ls="--", label="analytic m g/(2μ)")
ax[0].set(xlabel="grip force per finger [N]", ylabel="drop in 1 s [mm]", title="Static hold: slip vs force")
ax[0].legend()
ax[1].semilogy([r["F_N"] for r in dyn], [max(r["slip_mm"], 1e-3) for r in dyn], "o-")
ax[1].axvline(dyn[0]["f_min_N"], color="r", ls="--", label="analytic m(g+a)/(2μ)")
ax[1].set(xlabel="grip force per finger [N]", ylabel="max slip during lift [mm]",
          title=f"Fast lift (a_peak={dyn[0]['a_peak']:.1f} m/s²)")
ax[1].legend()
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_threshold.png", dpi=110)

# ---------------------------------------------------------------- 動畫
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


def cb_factory(name):
    def cb(sc):
        views = [
            sc.sim.render(target=(0.45, 0.0, 0.2), distance=0.75, yaw=y, pitch=p, size=SIZE, fov=40)
            for y, p in ((20, -20), (90, -12), (150, -35))
        ]
        fl, fr = sc.g.contact_forces()
        frames.append(np.hstack([lab(views[0], f"T005 {name}", f"t = {sc.t:.2f} s"),
                                 lab(views[1], f"grip F = {fl:.0f}/{fr:.0f} N", "side"),
                                 lab(views[2], f"obj z = {sc.g.obj_pose()[0][2] * 1e3:.0f} mm", "top")]))
    return cb


for spec in (CUBE, CYLINDER):
    sc = GraspScene(spec, render=True)
    cb = cb_factory(spec.name)
    sc, _ = run_pick_lift(spec, force=40.0, lift=0.15, T_lift=1.0, frame_cb=cb, scene=sc)
    sc.close()
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((720, 200), Image.LANCZOS) for f in frames]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=50, loop=0, optimize=True)
n = len(frames)
for nm, idx in (("key_a", n // 8), ("key_b", n // 3), ("key_c", int(n * 0.9))):
    Image.fromarray(frames[idx]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = n
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(M, indent=2, ensure_ascii=False))
