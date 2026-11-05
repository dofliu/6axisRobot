"""T007 報告：指令介面 → JSON/圖/動畫/報告 md。
用法：PYTHONPATH=. python3 scripts/t007_report.py 2026-11-05
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

from sixaxis import scene as S  # noqa: E402
from sixaxis.commands import Executor, parse  # noqa: E402
from sixaxis.graspsim import GraspScene  # noqa: E402
from sixaxis.gripper import CUBE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T007"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
ZR = S.BENCH_TOP + CUBE.size[2] / 2
PC = S.pocket_centers()
ZP = ZR + S.TRAY_PLATE_T
BX, BY = S.BELT_PICK_XY

PROG_A = f"""# 程式 A（DSL）：桌面 (0.45,0) → 料盤格 3
move_j tcp=0.45,0,0.30
move_j tcp=0.45,0,0.22
move_l tcp=0.45,0,{ZR}
grip force=40
move_l tcp=0.45,0,0.25
move_c via=0.52,-0.0625,0.25 tcp={PC[3][0]},{PC[3][1]},0.25
move_l tcp={PC[3][0]},{PC[3][1]},{ZP}
release
move_l tcp={PC[3][0]},{PC[3][1]},0.25
"""
PROG_B = [  # 程式 B（JSON）：輸送帶擋料位 → 料盤格 0
    {"cmd": "move_j", "tcp": [0.45, 0.2, 0.30], "id": "relay"},
    {"cmd": "move_j", "tcp": [BX, BY, 0.245], "id": "above_belt"},
    {"cmd": "move_l", "tcp": [BX, BY, ZR], "id": "descend"},
    {"cmd": "grip", "force": 40, "id": "grip_belt_part"},
    {"cmd": "move_l", "tcp": [BX, BY, 0.245], "id": "lift"},
    {"cmd": "move_j", "tcp": [0.45, 0.2, 0.30], "id": "relay2"},
    {"cmd": "move_j", "tcp": [PC[0][0], PC[0][1], 0.25], "id": "above_pocket0"},
    {"cmd": "move_l", "tcp": [PC[0][0], PC[0][1], ZP], "id": "place_down"},
    {"cmd": "release", "id": "release"},
    {"cmd": "move_l", "tcp": [PC[0][0], PC[0][1], 0.25], "id": "retreat"},
    {"cmd": "wait", "T": 0.3},
]
ERR_CASES = [
    ("未知指令", [{"cmd": "fly"}], None),
    ("缺少必要參數", [{"cmd": "move_l"}], None),
    ("參數個數錯誤", [{"cmd": "move_l", "tcp": [0.4, 0]}], None),
    ("參數範圍錯誤", [{"cmd": "grip", "force": 1e4}], None),
    ("JSON 格式錯誤", "[{bad json", None),
    ("目標不可達", "move_j tcp=2,0,0.3", None),
    ("圓弧三點共線", "move_j tcp=0.45,0,0.3\nmove_c via=0.45,0,0.35 tcp=0.45,0,0.4", None),
    ("T 過短 → 超速", "move_j tcp=0.45,0,0.3\nmove_l tcp=0.45,0.3,0.3 T=0.1", None),
    ("關節超限位", [{"cmd": "move_j", "q": [10, 0, 0, 0, 0, 0]}], None),
    ("空夾（無物件）", "move_j tcp=0.45,0.1,0.30\ngrip force=30", "sim"),
    ("壓進工作台", "move_j tcp=0.45,0,0.3\nmove_l tcp=0.45,0,0.07", "sim"),
]
M: dict = {"date": DATE}

# ------------------------------------------------------------ 1. 錯誤回報
rows = []
for name, prog, mode in ERR_CASES:
    s = GraspScene(CUBE, workcell=True) if mode == "sim" else None
    rep = Executor(s).run(prog)
    rows.append({"case": name, "code": rep.error.code, "index": rep.error.index, "message": rep.error.message,
                 "executed_ok": sum(r["status"] == "ok" for r in rep.results)})
    if s:
        s.close()
M["errors"] = rows

# ------------------------------------------------------------ 2. dry-run vs 物理
dry = Executor().run(PROG_A)
M["dry_total_s"] = dry.t_total

# ------------------------------------------------------------ 3. 示範（物理 + 渲染）
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (400, 330)
frames: list = []
state = {"txt": ""}


def lab(img, left, right):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d.text((10, 4), left, fill=(235, 238, 242), font=font)
    d.text((im.width - 10 - d.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im)


def cb(sc):
    views = [sc.sim.render(target=(0.40, 0.10, 0.12), distance=1.15, yaw=y, pitch=p_, size=SIZE, fov=42)
             for y, p_ in ((35, -28), (90, -14), (180, -78))]
    frames.append(np.hstack([lab(views[0], f"T007 {state['txt']}", f"t = {sc.t:.1f} s"),
                             lab(views[1], f"holding={ex.holding}", "side"),
                             lab(views[2], "cmd interface", "top")]))


sc = GraspScene(CUBE, workcell=True, render=True)
bench_cube = sc.obj
belt_cube = sc.g.spawn(CUBE, [0.34, 0.45, ZR + 1e-3])
sc.wc.add_part(belt_cube)
sc.g.obj = bench_cube
sc.g.open()
ex = Executor(sc, frame_cb=cb, cb_every=50)
state["txt"] = "program A: bench→pocket3"
repA = ex.run(PROG_A)
posA = np.array(pb.getBasePositionAndOrientation(bench_cube, physicsClientId=sc.cid)[0])
sc.g.obj = belt_cube
state["txt"] = "program B: belt→pocket0"
repB = ex.run(PROG_B)
posB = np.array(pb.getBasePositionAndOrientation(belt_cube, physicsClientId=sc.cid)[0])
M["programs"] = {
    "A": {"ok": repA.ok, "t_s": repA.t_total, "place_xy_err_mm": float(np.linalg.norm(posA[:2] - np.array(PC[3])) * 1e3),
          "place_z_err_mm": float((posA[2] - ZP) * 1e3), "results": repA.results},
    "B": {"ok": repB.ok, "t_s": repB.t_total - repA.t_total,
          "place_xy_err_mm": float(np.linalg.norm(posB[:2] - np.array(PC[0])) * 1e3),
          "place_z_err_mm": float((posB[2] - ZP) * 1e3), "results": repB.results},
}
clr = sc.wc.min_clearance()
rec = {k: np.array(v) for k, v in sc.rec.items()}
sc.close()

# 圖：TCP 軌跡與指力
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for k, n in enumerate("xyz"):
    ax[0].plot(rec["t"], rec["tcp"][:, k] * 1e3, label=f"TCP {n}")
for r in [*repA.results, *repB.results]:
    if r["cmd"] in ("grip", "release"):
        ax[0].axvline(r["t_start"] if r["cmd"] == "release" else r["t_end"], color="k", ls=":", lw=0.8)
ax[0].set(ylabel="TCP [mm]", title="Program A + B：TCP 位置（虛線：grip/release）")
ax[0].legend(ncol=3)
ax[1].plot(rec["t"], rec["fL"], label="finger L")
ax[1].plot(rec["t"], rec["fR"], label="finger R")
ax[1].set(xlabel="t [s]", ylabel="contact force [N]")
ax[1].legend()
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_tcp_forces.png", dpi=110)

mv = [r for r in [*repA.results, *repB.results] if r["cmd"].startswith("move")]
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.bar(range(len(mv)), [r["tcp_err_mm"] for r in mv])
ax.set_xticks(range(len(mv)))
ax.set_xticklabels([r["id"] for r in mv], rotation=60, ha="right", fontsize=7)
ax.set(ylabel="TCP error at end of motion [mm]", title="Per-command tracking error")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_cmd_errors.png", dpi=110)
M["max_tcp_err_mm"] = float(max(r["tcp_err_mm"] for r in mv))
M["max_vel_ratio"] = float(max(r["vel_ratio"] for r in mv if "vel_ratio" in r))
M["max_torque_ratio"] = float(max(r["torque_ratio"] for r in mv if "torque_ratio" in r))
M["final_clearance_mm"] = clr * 1e3
M["parse_roundtrip"] = len(parse(PROG_A)) == len(parse(json.dumps(parse(PROG_A))))

imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((720, 200), Image.LANCZOS) for f in frames]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=50, loop=0, optimize=True)
n = len(frames)
for nm, idx in (("key_a", n // 6), ("key_b", int(n * 0.45)), ("key_c", int(n * 0.95))):
    Image.fromarray(frames[idx]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = n
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in M.items() if k != "programs"}, indent=1, ensure_ascii=False)[:3000])
print({k: (v["ok"], v["t_s"], v["place_xy_err_mm"], v["place_z_err_mm"]) for k, v in M["programs"].items()})
