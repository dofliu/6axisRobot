"""T003 示範動畫 v2：以 T002a 的新外觀（UR 型網格、光影、陰影、地坪）重新渲染同一條示範軌跡。

不覆寫原 T003 動畫/報告（依規則保留），輸出 `<DATE>_T003_v2.*`。軌跡建構程式碼與
scripts/t003_report.py「示範序列」段相同，並核對點數/時長與原 metrics JSON 一致。

用法：PYTHONPATH=. python3 scripts/t003_render_v2.py 2026-11-01
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from sixaxis import SixAxisRobot
from sixaxis.pbsim import PBRobot
from sixaxis.trajectory import (
    ArcPath,
    LinePath,
    Quintic,
    Trapezoid,
    concat,
    joint_ptp,
    min_duration,
    sample_cartesian,
    sample_joint,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T003_v2"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    font = ImageFont.truetype(FONT, 15)
except OSError:
    font = ImageFont.load_default()

# ------------------------------------------------------------ 示範序列（同 t003_report.py）
robot = SixAxisRobot()
Q_HOME = np.array([0.3, -1.1, 1.3, -0.7, 0.9, 0.2])
Q_FAR = Q_HOME + np.array([0.5, 0.3, -0.3, 0.4, -0.5, 0.6])
DT = 0.01
T_pt = min_duration(Q_HOME - Q_FAR, robot.qd_max * 0.5)
S1 = sample_joint(robot, joint_ptp(Q_FAR, Q_HOME, T_pt, "quintic"), DT)
T0p = robot.fk(Q_HOME)
P0 = T0p[:3, 3]
Rz = lambda a: Rotation.from_rotvec([0, 0, a]).as_matrix()  # noqa: E731
T_b = T0p.copy()
T_b[:3, 3] = P0 + [0.0, 0.25, 0.0]
T_b[:3, :3] = T0p[:3, :3] @ Rz(0.6)
S2 = sample_cartesian(robot, LinePath(T0p, T_b), Trapezoid(2.0, 0.25), Q_HOME, DT)
T_c = T_b.copy()
T_c[:3, 3] = T_b[:3, 3] + [0.0, 0.2, 0.0]
T_c[:3, :3] = T_b[:3, :3] @ Rz(-0.4)
S3 = sample_cartesian(robot, ArcPath(T_b, T_b[:3, 3] + [0.0, 0.1, 0.1], T_c), Quintic(3.0), S2.q[-1], DT)
S4 = sample_cartesian(robot, LinePath(T_c, T0p), Trapezoid(2.5, 0.2), S3.q[-1], DT)
parts = [S1, S2, S3, S4]
demo = concat(parts)

orig = json.loads((DATA / f"{DATE}_T003_metrics.json").read_text(encoding="utf-8"))["demo"]
assert len(demo) == orig["n_points"] and abs(demo.t[-1] - orig["duration_s"]) < 1e-9, "軌跡與原 T003 不一致"

# ------------------------------------------------------------ 渲染
seg_names = [
    "move_j (quintic)",
    "move_l + SLERP (trapezoid)",
    "move_c arc + SLERP (quintic)",
    "move_l return (trapezoid)",
]
vis = PBRobot(robot, plane=True)
SIZE = (480, 360)


def lab(img, left, right):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d.text((10, 4), left, fill=(235, 238, 242), font=font)
    d.text((im.width - 10 - d.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im)


bounds = np.cumsum([len(p) - (1 if i else 0) for i, p in enumerate(parts)])
step, frames, t_r = 4, [], []
for k in range(0, len(demo), step):
    seg_i = int(np.searchsorted(bounds, k, side="right"))
    vis.set_state(demo.q[k])
    if seg_i >= 1 and (k // step) % 3 == 0:  # 與原版相同的軌跡標記（圓弧段綠色）
        vis.add_marker(demo.pos[k], 0.007, (0.95, 0.35, 0.1, 1) if seg_i != 2 else (0.1, 0.7, 0.25, 1))
    t0 = time.perf_counter()
    views = [
        vis.render(target=(-0.3, 0.1, 0.4), distance=1.5, yaw=y, pitch=-20, size=SIZE) for y in (30, 120)
    ]
    t_r.append((time.perf_counter() - t0) / 2)
    name = seg_names[min(seg_i, 3)]
    frames.append(
        np.hstack(
            [lab(views[0], f"T003  {name}", "yaw 30"), lab(views[1], f"t = {demo.t[k]:.2f} s", "yaw 120")]
        )
    )
vis.close()

fps = 25
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=fps, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((640, 240), Image.LANCZOS) for f in frames[::2]]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(
    MEDIA / f"{TAG}.gif",
    save_all=True,
    append_images=pal[1:],
    duration=int(2000 / fps),
    loop=0,
    optimize=True,
)
for nm, idx in (("a", len(frames) // 6), ("b", len(frames) // 2), ("c", len(frames) - 2)):
    Image.fromarray(frames[idx]).save(MEDIA / f"{TAG}_key_{nm}.png")

M = {
    "date": DATE,
    "source_trajectory": {
        "n_points": len(demo),
        "duration_s": float(demo.t[-1]),
        "matches_T003_metrics": True,
    },
    "frames": len(frames),
    "render_ms_per_view": float(np.mean(t_r) * 1e3),
    "gif_MB": (MEDIA / f"{TAG}.gif").stat().st_size / 1e6,
    "mp4_MB": (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6,
}
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(M, indent=2, ensure_ascii=False))
