"""T006 報告：工作站場景量測 → JSON/圖/動畫/報告 md。
用法：PYTHONPATH=. python3 scripts/t006_report.py 2026-11-04
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from sixaxis import scene as S  # noqa: E402
from sixaxis.graspsim import Q_REF, GraspScene, tool_pose  # noqa: E402
from sixaxis.gripper import CUBE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T006"
MEDIA, DATA, REP = ROOT / "docs/media", ROOT / "docs/test-reports/data", ROOT / "docs/test-reports"
DATA.mkdir(exist_ok=True)
M: dict = {"date": DATE}
ZR = S.BENCH_TOP + CUBE.size[2] / 2


def pos(s, b):
    return np.array(pb.getBasePositionAndOrientation(b, physicsClientId=s.cid)[0])


def vel(s, b):
    return np.array(pb.getBaseVelocity(b, physicsClientId=s.cid)[0])


def newpart(s, xy, z, v=None):
    b = s.g.spawn(CUBE, [xy[0], xy[1], z])
    s.wc.add_part(b)
    if v is not None:
        pb.resetBaseVelocity(b, list(v), physicsClientId=s.cid)
    return b


# ------------------------------------------------------------ 1. 靜態量測（無渲染）
s = GraspScene(CUBE, workcell=True)
# 輸送帶
b = newpart(s, (0.20, 0.45), ZR + 1e-3)
tt, xx, vv = [], [], []
for k in range(5000):
    s.step()
    if k % 5 == 0:
        tt.append(s.t)
        xx.append(pos(s, b)[0])
        vv.append(vel(s, b)[0])
    if np.linalg.norm(vel(s, b)) < 1e-4 and pos(s, b)[0] > 0.4 and k > 1000:
        break
tt, xx, vv = map(np.array, (tt, xx, vv))
m_ = (tt > 0.3) & (xx < S.STOP_X - 0.08)
v_ss = float(np.mean(vv[m_]))
p = pos(s, b)
t_ramp = float(tt[np.argmax(vv > 0.98 * S.BELT_SPEED)])
M["belt"] = {"v_cmd": S.BELT_SPEED, "v_ss": v_ss, "v_err_pct": 100 * (v_ss - S.BELT_SPEED) / S.BELT_SPEED,
             "ramp_98pct_s": t_ramp, "stop_xy": p[:2].tolist(), "stop_err_mm": float(np.linalg.norm(p[:2] - S.BELT_PICK_XY) * 1e3),
             "y_drift_mm": float(abs(p[1] - 0.45) * 1e3), "t_to_stop_s": float(tt[-1])}
s.close()

# 料盤
s = GraspScene(CUBE, workcell=True)
off = (S.POCKET - CUBE.size[0]) / 2
rows = []
for i, (cx, cy) in enumerate(S.pocket_centers()):
    for sx, sy in ((0.75, -0.75), (-0.75, 0.75), (0.95, 0.0), (0.0, -0.95)):
        bb = newpart(s, (cx + off * sx, cy + off * sy), ZR + S.TRAY_PLATE_T + 0.03)
        s.wait(1.0)
        q = pos(s, bb)
        rows.append({"pocket": i, "drop_off_mm": [off * sx * 1e3, off * sy * 1e3],
                     "err_xy_mm": [(q[0] - cx) * 1e3, (q[1] - cy) * 1e3], "z_err_mm": float((q[2] - ZR - S.TRAY_PLATE_T) * 1e3)})
        pb.removeBody(bb, physicsClientId=s.cid)
        s.wc.parts.remove(bb)
M["tray"] = {"clearance_per_side_mm": off * 1e3, "rows": rows,
             "all_inside": bool(all(max(abs(x) for x in r["err_xy_mm"]) <= off * 1e3 + 0.5 for r in rows))}

# 治具
rng = np.random.default_rng(6)
fin = []
for _ in range(10):
    o = rng.uniform(-0.006, 0.006, 2)
    bb = newpart(s, (S.FIX_NOM[0] - 0.008 + o[0], S.FIX_NOM[1] - 0.008 + o[1]), ZR + 1e-3, (0.6, 0.6, 0))
    s.wait(1.0)
    fin.append(pos(s, bb)[:2] - np.array(S.FIX_NOM))
    pb.removeBody(bb, physicsClientId=s.cid)
    s.wc.parts.remove(bb)
fin = np.array(fin) * 1e3
M["fixture"] = {"n": len(fin), "err_mm": fin.tolist(), "max_mm": float(np.max(np.linalg.norm(fin, axis=1))),
                "rms_mm": float(np.sqrt(np.mean(np.sum(fin**2, axis=1)))), "start_offset_max_mm": 14.0}

# 可達性與碰撞
reach = []
for nm, xy in [("pick", S.PICK_XY), ("belt", S.BELT_PICK_XY), ("fixture", S.FIX_NOM)] + [
        (f"pocket{i}", c) for i, c in enumerate(S.pocket_centers())]:
    zc = S.BENCH_TOP + (S.TRAY_PLATE_T if nm.startswith("pocket") else 0.0) + CUBE.size[2] / 2
    r_ = {"station": nm, "xy": list(xy), "radius_m": float(np.hypot(*xy))}
    for lab, dz in (("grasp", 0.0), ("hover", 0.12)):
        sols = s.r.ik(tool_pose([xy[0], xy[1], zc + dz]), Q_REF)
        free = []
        for q in sols:
            s.sim.set_state(q)
            free.append(not s.wc.arm_collisions())
        s.sim.set_state(sols[0])
        r_[lab] = {"n_ik": len(sols), "n_free": int(sum(free)), "best_free": bool(free[0]),
                   "clearance_mm": float(s.wc.min_clearance() * 1e3)}
    reach.append(r_)
M["reach"] = reach
# 碰撞查詢驗證
sol = s.r.ik(tool_pose([S.PICK_XY[0], 0.0, 0.05]), Q_REF)[0]
s.sim.set_state(sol)
M["collision_query"] = {"below_bench_hits": s.wc.arm_collisions(), "penetration_mm": float(s.wc.min_clearance() * 1e3)}
s.close()

# ------------------------------------------------------------ 2. 圖
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(tt, vv * 1e3)
ax[0].axhline(S.BELT_SPEED * 1e3, color="r", ls="--", label="belt speed")
ax[0].set(xlabel="t [s]", ylabel="part vx [mm/s]", title="Conveyor: velocity tracking & stop")
ax[0].legend()
ax[1].plot(tt, xx * 1e3)
ax[1].axhline(S.BELT_PICK_XY[0] * 1e3, color="r", ls="--", label="stopper rest x")
ax[1].set(xlabel="t [s]", ylabel="part x [mm]", title="Part position")
ax[1].legend()
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_belt.png", dpi=110)

fig, ax = plt.subplots(figsize=(7.5, 6.2))


def rect(cx, cy, hx, hy, **kw):
    ax.add_patch(mp.Rectangle((cy - hy, cx - hx), 2 * hy, 2 * hx, **kw))


rect(S.BENCH_X, 0, S.BENCH_HX, S.BENCH_HY, fc="#c9ccd2", ec="k", label="bench")
rect((S.BELT_X0 + S.BELT_X1) / 2, (S.BELT_Y0 + S.BELT_Y1) / 2, (S.BELT_X1 - S.BELT_X0) / 2,
     (S.BELT_Y1 - S.BELT_Y0) / 2, fc="#3a3d44", ec="k")
rect(S.STOP_X + 0.005, 0.45, 0.005, 0.1, fc="#d93b2b", ec="k")
tw = S.TRAY_PITCH + S.POCKET + 2 * S.POST
rect(S.TRAY_C[0], S.TRAY_C[1], tw / 2, tw / 2, fc="#46506a", ec="k")
for cx, cy in S.pocket_centers():
    rect(cx, cy, S.POCKET / 2, S.POCKET / 2, fc="#dfe6f5", ec="#888")
    for sx in (-1, 1):
        rect(cx + sx * (S.POCKET / 2 + S.POST / 2), cy, S.POST / 2, S.POCKET / 2, fc="#f2c12e", ec="k")
        for sy in (-1, 1):
            rect(cx + sx * S.PIN_X, cy + sy * (S.POCKET / 2 + S.POST / 2), S.POST / 4, S.POST / 2, fc="#f2c12e", ec="k")
rect(S.FIX_NOM[0] + 0.03, S.FIX_NOM[1], 0.005, 0.05, fc="#e6801a", ec="k")
rect(S.FIX_NOM[0] + 0.019, S.FIX_NOM[1] + 0.03, 0.006, 0.005, fc="#e6801a", ec="k")
rect(*S.FIX_NOM, 0.025, 0.025, fc="none", ec="#d93b2b", ls="--")
ax.plot(0, 0, "ko", ms=12)
ax.annotate("robot base", (0, 0.02))
for nm, xy in (("pick", S.PICK_XY), ("belt pick", S.BELT_PICK_XY)):
    ax.plot(xy[1], xy[0], "g^")
    ax.annotate(nm, (xy[1] + 0.01, xy[0]))
ax.set(xlim=(-0.45, 0.65), ylim=(-0.05, 0.75), xlabel="y [m]", ylabel="x [m]", title="Workcell layout (top view)")
ax.set_aspect("equal")
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_layout.png", dpi=110)

# ------------------------------------------------------------ 3. 示範：輸送帶 → 料盤、桌面 → 料盤
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 330)
frames: list = []
clr: list = []


def lab(img, left, right):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d.text((10, 4), left, fill=(235, 238, 242), font=font)
    d.text((im.width - 10 - d.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im)


state = {"txt": ""}


def cb(sc):
    clr.append((sc.t, sc.wc.min_clearance() * 1e3, len(sc.wc.arm_collisions())))
    views = [sc.sim.render(target=(0.40, 0.12, 0.12), distance=1.15, yaw=y, pitch=p_, size=SIZE, fov=42)
             for y, p_ in ((35, -28), (90, -14), (180, -78))]
    frames.append(np.hstack([lab(views[0], f"T006 {state['txt']}", f"t = {sc.t:.1f} s"),
                             lab(views[1], f"clearance {clr[-1][1]:.0f} mm", "side"),
                             lab(views[2], "belt →", "top")]))


def move(sc, q, T):
    sc.move_to(q, T, frame_cb=cb, cb_every=50)


def ik(xy, zc):
    sols = sc.r.ik(tool_pose([xy[0], xy[1], zc]), Q_REF)
    return sols[0]


def pick_place(sc, body, src, zsrc, dst, zdst, label):
    sc.g.obj = body
    n0 = len(sc.rec["t"])
    t0 = sc.t
    state["txt"] = f"pick {label}"
    sc.g.open()
    t_ = sc.g.tcp_pos()
    move(sc, ik(t_[:2], t_[2] + 0.12), 0.6)  # 先垂直抬起（指間有物件時不可橫掃）
    move(sc, ik((0.45, 0.2), 0.30), 1.0)  # 經高處中繼點，避開治具擋板
    move(sc, ik(src, zsrc + 0.12), 1.3)
    move(sc, ik(src, zsrc), 0.8)
    sc.wait(0.2)
    sc.g.close(40.0)
    sc.wait(0.5)
    state["txt"] = f"carry {label}"
    move(sc, ik(src, zsrc + 0.12), 0.8)
    move(sc, ik(dst, zdst + 0.12), 1.6)
    state["txt"] = f"place {label}"
    move(sc, ik(dst, zdst), 0.8)
    sc.wait(0.2)
    sc.g.open()
    sc.wait(0.4)
    move(sc, ik(dst, zdst + 0.12), 0.6)
    return {"cycle_s": sc.t - t0, "place_err_mm": [(pos(sc, body)[0] - dst[0]) * 1e3, (pos(sc, body)[1] - dst[1]) * 1e3],
            "z": float(pos(sc, body)[2]), "n": len(sc.rec["t"]) - n0}


sc = GraspScene(CUBE, workcell=True, render=True)
bench_cube = sc.obj
state["txt"] = "conveyor feed"
belt_cube = newpart(sc, (0.34, 0.45), ZR + 1e-3)
sc.g.open()
for _ in range(14):  # 約 1.4 s：零件隨帶前進並被擋料板擋住
    sc.step(100)
    cb(sc)
sc.wait(0.2)
M["demo_belt_part_stop_err_mm"] = float(np.linalg.norm(pos(sc, belt_cube)[:2] - S.BELT_PICK_XY) * 1e3)
pcs = S.pocket_centers()
zt = ZR + S.TRAY_PLATE_T
demo = [pick_place(sc, belt_cube, S.BELT_PICK_XY, ZR, pcs[0], zt, "belt→pocket0"),
        pick_place(sc, bench_cube, S.PICK_XY, ZR, pcs[3], zt, "bench→pocket3")]
M["demo"] = demo
clr_a = np.array(clr)
M["demo_clearance"] = {"min_mm": float(clr_a[:, 1].min()), "collision_samples": int(clr_a[:, 2].sum()), "samples": len(clr_a)}
sc.close()

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(clr_a[:, 0], clr_a[:, 1])
ax.axhline(0, color="r", ls="--")
ax.set(xlabel="t [s]", ylabel="min arm–scene clearance [mm]", title="Arm clearance during demo (search 300 mm)")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_clearance.png", dpi=110)

imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((720, 200), Image.LANCZOS) for f in frames]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=50, loop=0, optimize=True)
n = len(frames)
for nm, idx in (("key_a", n // 10), ("key_b", int(n * 0.4)), ("key_c", int(n * 0.92))):
    Image.fromarray(frames[idx]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = n
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in M.items() if k not in ("reach", "tray", "fixture")}, indent=2, ensure_ascii=False))
