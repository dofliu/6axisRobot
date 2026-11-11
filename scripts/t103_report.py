"""T103 報告：疊棧/碼垛 → JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t103_report.py 2026-11-11
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
from sixaxis.stacking import (  # noqa: E402
    FEED_XY,
    PITCH,
    PYR_XY,
    STACK_XY,
    A,
    column_slots,
    pyramid_slots,
    run_stack,
    stable_height,
    stack_margins,
    tower_slots,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T103"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
SEED = 2026
M: dict = {"date": DATE, "seed": SEED, "feed_xy": FEED_XY, "stack_xy": STACK_XY, "pyr_xy": PYR_XY, "pitch": PITCH,
           "a": A}


def one(slots, **kw):
    s = GraspScene(workcell=True)
    try:
        m = run_stack(s, Executor(s), slots, **kw)
    finally:
        s.close()
    return m


def strip(m, keep_cubes=True):
    out = {k: v for k, v in m.items() if k not in ("results", "bodies")}
    if not keep_cubes:
        out.pop("placed"), out.pop("final")
    return out


def xy(slots, off=None):
    c = np.array([[s.x, s.y] for s in slots])
    return c if off is None else c + off


# ---------------------------------------------------- 1. 垂直疊棧 10 層（標稱）
NT = 10
tower = one(tower_slots(NT))
M["tower"] = strip(tower)
M["tower_steps"] = tower["results"][-8:]

# ---------------------------------------------------- 2. 釋放高度 / 速度掃描（6 層塔）
sweep = []
for dz in (0.0, 0.001, 0.002, 0.005, 0.010):
    sweep.append({"dz_mm": dz * 1e3, "speed": 1.0, **strip(one(tower_slots(6), dz_release=dz), False)})
for sp in (0.5, 0.75):
    sweep.append({"dz_mm": 0.0, "speed": sp, **strip(one(tower_slots(6), speed=sp), False)})
M["sweep"] = sweep

# ---------------------------------------------------- 3. 碼垛樣式：金字塔 3-2-1、柱式 2×3
M["pyramid"] = strip(one(pyramid_slots(3)))
M["column"] = strip(one(column_slots(2, 3)))

# ---------------------------------------------------- 4. 錯位塔：倒塌層數 vs 解析 d < a/n
stag = []
for d in (0.006, 0.008, 0.0095, 0.0105, 0.012, 0.013, 0.016, 0.0175, 0.022, 0.027):
    slots = tower_slots(8, stagger=d)
    m = one(slots, settle=0.5)
    stag.append({"d_mm": d * 1e3, "pred": stable_height(xy(slots)), "built": m["built"], "error": m["error"],
                 "n_crit": A / d})
M["stagger"] = stag

# ---------------------------------------------------- 5. 隨機放置誤差：6 層塔，兩軸均勻 ±e
rng = np.random.default_rng(SEED)
rand = []
for e in (0.005, 0.010, 0.015, 0.020):
    for k in range(6):
        off = rng.uniform(-e, e, size=(6, 2))
        slots = tower_slots(6)
        m = one(slots, offsets=off, settle=0.5)
        pred = stable_height(xy(slots, off))
        rand.append({"e_mm": e * 1e3, "trial": k, "pred": pred, "built": m["built"], "error": m["error"],
                     "min_margin_mm": float(stack_margins(xy(slots, off)).min() * 1e3),
                     "offsets_mm": (off * 1e3).tolist()})
M["rand"] = rand

# ---------------------------------------------------- 圖
fin = tower["final"]
L = [f["layer"] + 1 for f in fin]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].plot(L, [f["dx_mm"] for f in fin], "o-", label="dx")
ax[0].plot(L, [f["dy_mm"] for f in fin], "s-", label="dy")
ax[0].set(xlabel="layer", ylabel="XY error [mm]", title=f"Tower N={NT}: per-layer XY error")
ax[0].legend()
ax[1].plot(L, [f["z_err_mm"] for f in fin], "o-", color="#c0553b")
ax[1].set(xlabel="layer", ylabel="z error [mm]", title="Height error (cumulative contact compliance)")
ax[2].bar(L, tower["cycles_s"], color="#4a9a6a")
ax[2].set(xlabel="layer", ylabel="cycle time [s]", title="Cycle time per cube")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_tower.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
dd = np.linspace(5.5, 28, 200)
ax[0].step(dd, np.ceil(A * 1e3 / dd) - 1, where="post", color="#888", label="analytic: max n with d < a/n")
ax[0].plot([s["d_mm"] for s in stag], [s["built"] for s in stag], "o", color="#c0553b", ms=8,
           label="simulation (layers standing)")
ax[0].set(xlabel="stagger per layer d [mm]", ylabel="max stable layers", title="Staggered tower: collapse height")
ax[0].legend()
for e, c in zip(sorted({r["e_mm"] for r in rand}), ("#3b6ea8", "#4a9a6a", "#d39a2a", "#c0553b"), strict=True):
    r = [x for x in rand if x["e_mm"] == e]
    jit = (np.arange(len(r)) - len(r) / 2) * 0.03
    ax[1].scatter([x["pred"] + j for x, j in zip(r, jit, strict=True)], [x["built"] for x in r], color=c,
                  label=f"±{e:.0f} mm", s=40, alpha=0.85)
ax[1].plot([0, 6.5], [0, 6.5], "k--", lw=0.8)
ax[1].set(xlabel="predicted stable layers (CoM criterion)", ylabel="simulated standing layers",
          title="Random placement error, 6-layer tower")
ax[1].legend(title="uniform error")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_stability.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
s1 = [s for s in sweep if s["speed"] == 1.0]
ax[0].plot([s["dz_mm"] for s in s1], [s["xy_max_mm"] for s in s1], "o-", label="max XY error")
ax[0].plot([s["dz_mm"] for s in s1], [s["disturb_max_mm"] for s in s1], "s-", label="max disturbance")
ax[0].set(xlabel="release height above nominal [mm]", ylabel="[mm]", title="6-layer tower: release height")
ax[0].legend()
for nm, pat, c, x0 in (("pyramid 3-2-1", M["pyramid"], "#3b6ea8", 0.0), ("column 2x3", M["column"], "#c0553b", 200.0)):
    for f in pat["final"]:
        ax[1].add_patch(plt.Rectangle((x0 + f["x"] * 1e3 - 25, f["z"] * 1e3 - 25), 50, 50, fill=False, ec=c, lw=1.5))
    ax[1].plot([], [], "s", mfc="none", mec=c, label=f"{nm} (x+{x0:.0f})" if x0 else nm)
ax[1].axhline(100, color="#888", lw=1)
ax[1].set(xlabel="x [mm]", ylabel="z [mm]", title="Palletizing patterns (measured cube outlines)", aspect="equal",
          xlim=(350, 760), ylim=(90, 270))
ax[1].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_patterns.png", dpi=110)

# ---------------------------------------------------- 動畫：5 層塔 → 金字塔 3-2-1 → 錯位 13 mm 塔（第 4 塊倒塌）
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 330)
frames: list = []
state = {"txt": "", "sub": ""}


def lab(img, left, right):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d.text((10, 4), left, fill=(235, 238, 242), font=font)
    d.text((im.width - 10 - d.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im)


def cb(sc):
    views = [sc.sim.render(target=(0.42, 0.0, 0.20), distance=d, yaw=y, pitch=p_, size=SIZE, fov=42)
             for y, p_, d in ((35, -22, 1.05), (90, -8, 0.95), (180, -70, 1.0))]
    frames.append(np.hstack([lab(views[0], state["txt"], f"t = {sc.t:.1f} s"),
                             lab(views[1], state["sub"], "side"),
                             lab(views[2], "feeder -> stack", "top")]))


demo = {}
marks = []
for key, txt, sub, slots in (
        ("tower5", "T103 stacking: tower x5", "aligned", tower_slots(5)),
        ("pyramid", "T103 palletizing: pyramid 3-2-1", "interlocked", pyramid_slots(3)),
        ("stagger13", "T103 staggered tower d=13 mm", "predicted: 3 layers max", tower_slots(5, stagger=0.013))):
    state.update(txt=txt, sub=sub)
    sc = GraspScene(workcell=True, render=True)
    ex = Executor(sc, frame_cb=cb, cb_every=100)
    m = run_stack(sc, ex, slots, settle=0.3)
    for _ in range(12):  # 結尾停留/倒塌過程
        sc.wait(0.1)
        cb(sc)
    sc.close()
    demo[key] = strip(m, False)
    marks.append(len(frames))
M["demo"] = demo
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((660, 182), Image.LANCZOS) for f in frames[::3]]  # GIF 每 3 幀取 1 以控制在 8 MB 內
pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=150, loop=0, optimize=True)
for nm, idx in (("key_a", marks[0] - 1), ("key_b", marks[1] - 1), ("key_c", marks[2] - 1)):
    Image.fromarray(frames[idx]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = len(frames)
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
for k in ("tower", "pyramid", "column"):
    print(k, {a: b for a, b in M[k].items() if a not in ("placed", "final", "cycles_s")})
for s in sweep:
    print("sweep", s["dz_mm"], s["speed"], s["success"], round(s["xy_max_mm"], 2), round(s["disturb_max_mm"], 2),
          round(s["cycle_mean_s"], 2))
for s in stag:
    print("stag", s)
for r in rand:
    print("rand", r["e_mm"], r["pred"], r["built"], r["error"], round(r["min_margin_mm"], 1))
print("demo", demo, M["gif_MB"], M["mp4_MB"])
