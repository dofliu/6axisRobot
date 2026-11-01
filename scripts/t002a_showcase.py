"""T002a：手臂外觀擬真化展示（環繞鏡頭 + IK 圓軌跡）、前後對照圖、量化資料 JSON。

用法：PYTHONPATH=. python3 scripts/t002a_showcase.py YYYY-MM-DD
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as pb
from PIL import Image, ImageDraw, ImageFont

from sixaxis import SixAxisRobot
from sixaxis.pbsim import URDF_PATH, PBRobot

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T002a"
MEDIA = ROOT / "docs/media"
DATA = ROOT / "docs/test-reports/data"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    font = ImageFont.truetype(FONT, 15)
    font_big = ImageFont.truetype(FONT, 20)
except OSError:
    font = font_big = ImageFont.load_default()

robot = SixAxisRobot()
pbr = PBRobot(robot, plane=True)
M: dict = {"date": DATE}

# ---------------------------------------------------------------- 幾何統計
urdf = URDF_PATH.read_text(encoding="utf-8")
meshes = sorted(set(re.findall(r'filename="meshes/([^"]+)"', urdf)))
tri = sum(Path(URDF_PATH.parent / "meshes" / n).read_text().count("\nf ") for n in meshes)
vis = pb.getVisualShapeData(pbr.id, physicsClientId=pbr.cid)
M["geometry"] = {
    "mesh_files": len(meshes),
    "mesh_triangles_total": tri,
    "mesh_dir_MB": sum((URDF_PATH.parent / "meshes" / n).stat().st_size for n in meshes) / 1e6,
    "visual_shapes": len(vis),
    "collision_shapes": urdf.count("<collision>"),
}


# ---------------------------------------------------------------- 自我碰撞（非相鄰連桿）
def min_clearance(q):
    pbr.set_state(q)
    best, pair = 9.0, None
    for a in range(-1, 7):
        for b in range(a + 2, 7):
            for c in pb.getClosestPoints(pbr.id, pbr.id, 1.0, a, b, physicsClientId=pbr.cid):
                if c[8] < best:
                    best, pair = c[8], (a, b)
    return best, pair


poses = {
    "zero": np.zeros(6),
    "home": np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]),
    "demo": np.array([0, -1.2, 1.8, -2.1, -1.57, 0.0]),
}
M["self_clearance_m"] = {
    k: {"min": min_clearance(q)[0], "pair": min_clearance(q)[1]} for k, q in poses.items()
}
rng = np.random.default_rng(0)
rand = [min_clearance(q)[0] for q in rng.uniform(-np.pi, np.pi, (300, 6))]
M["random_pose_self_collision_fraction"] = float(np.mean(np.array(rand) < 0))

# ---------------------------------------------------------------- 動作序列
fps = 20
home = poses["home"]
demo = poses["demo"]
# 段 1：home → demo，五次多項式（0 初末速度/加速度）時間律，3 s；鏡頭環繞
n1 = 3 * fps
s = np.linspace(0, 1, n1)
blend = 10 * s**3 - 15 * s**4 + 6 * s**5
seg1 = home + blend[:, None] * (demo - home)
# 段 2：IK 水平圓（同 T002），5 s
R_down = np.diag([1.0, -1.0, -1.0])
center, rad = np.array([0.45, 0.0, 0.35]), 0.12
n2 = 5 * fps
s2 = np.linspace(0, 1, n2 + 1)
phi = 2 * np.pi * (10 * s2**3 - 15 * s2**4 + 6 * s2**5)
path = np.stack(
    [center[0] + rad * np.cos(phi), center[1] + rad * np.sin(phi), np.full_like(phi, center[2])], 1
)
q_prev, seg2 = seg1[-1].copy(), []
T0 = np.eye(4)
T0[:3, :3], T0[:3, 3] = R_down, path[0]
q_prev = robot.ik(T0, q_prev)[0]
# 從 demo 平滑接到圓起點（1 s）
n_lead = fps
lead = seg1[-1] + (
    10 * np.linspace(0, 1, n_lead) ** 3
    - 15 * np.linspace(0, 1, n_lead) ** 4
    + 6 * np.linspace(0, 1, n_lead) ** 5
)[:, None] * (q_prev - seg1[-1])
for p in path:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_down, p
    q_prev = robot.ik(T, q_prev)[0]
    seg2.append(q_prev)
seg2 = np.array(seg2)
for k in range(0, len(path), 5):
    pbr.add_marker(path[k], 0.006, rgba=(0.95, 0.35, 0.1, 1))

# ---------------------------------------------------------------- 渲染
size = (480, 360)


def label(img, text, sub=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d.text((10, 4), text, fill=(235, 238, 242), font=font)
    if sub:
        d.text((im.width - 10 - d.textlength(sub, font=font), 4), sub, fill=(150, 190, 255), font=font)
    return np.asarray(im)


frames, t_render = [], []
seq = (
    [(q, "Pose transition (quintic)") for q in seg1]
    + [(q, "Lead-in") for q in lead]
    + [(q, "Analytic-IK circle, tool down") for q in seg2]
)
yaw0 = 20.0
for k, (q, title) in enumerate(seq):
    pbr.set_state(q)
    t0 = time.perf_counter()
    orbit_yaw = yaw0 + 360.0 * k / len(seq)
    left = pbr.render(target=(0.15, 0.0, 0.35), distance=1.25, yaw=orbit_yaw, pitch=-20, size=size)
    right = pbr.render(target=(0.25, 0.0, 0.32), distance=0.95, yaw=90, pitch=-8, size=size)
    t_render.append((time.perf_counter() - t0) / 2)
    f = np.hstack([label(left, title, "orbit camera"), label(right, f"t = {k / fps:4.2f} s", "side view")])
    frames.append(f)
M["render_ms_per_view"] = float(np.mean(t_render) * 1e3)
M["frames"] = len(frames)

imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=fps, macro_block_size=None, quality=6)
# GIF 控制在 ~8MB 內：每隔一幀（10 fps）、縮成 640x240、96 色
gif_frames = [Image.fromarray(f).resize((640, 240), Image.LANCZOS) for f in frames[::2]]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif_frames]
pal[0].save(
    MEDIA / f"{TAG}.gif",
    save_all=True,
    append_images=pal[1:],
    duration=int(2000 / fps),
    loop=0,
    optimize=True,
)
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6

# 關鍵畫面（高解析）
for name, q, cam in (
    ("hero", demo, dict(target=(0.1, 0.0, 0.32), distance=1.1, yaw=35, pitch=-18)),
    ("home", home, dict(target=(0.1, 0.0, 0.42), distance=1.2, yaw=120, pitch=-12)),
    (
        "wrist_closeup",
        seg2[n2 // 3],
        dict(target=tuple(robot.link_frames(seg2[n2 // 3])[4][:3, 3]), distance=0.42, yaw=60, pitch=-22),
    ),
):
    pbr.set_state(q)
    Image.fromarray(pbr.render(size=(960, 720), **cam)).save(MEDIA / f"{TAG}_{name}.png")

# 前後對照：舊版畫面（T002 動畫中段）vs 新版同姿態同相機
old = Image.open(MEDIA / f"{DATE}_T002_circle_mid.png").convert("RGB")
pbr.set_state(seg2[n2 // 2])  # 與 T002 中段畫面同一圓軌跡中點
new = np.hstack(
    [
        pbr.render(target=(0.2, 0.0, 0.35), distance=1.5, yaw=50, pitch=-25, size=(400, 300)),
        pbr.render(target=(0.2, 0.0, 0.3), distance=1.3, yaw=0, pitch=-89, size=(400, 300)),
    ]
)
cmp_img = Image.new("RGB", (800, 640), (255, 255, 255))
cmp_img.paste(old.resize((800, 300)), (0, 20))
cmp_img.paste(Image.fromarray(new), (0, 340))
d = ImageDraw.Draw(cmp_img)
d.text((8, 2), "Before (T002): primitive cylinders, no shading/shadow", fill=(0, 0, 0), font=font)
d.text(
    (8, 322),
    "After (T002a): UR-style lathe meshes, offsets, caps, lighting + shadow + AA",
    fill=(0, 0, 0),
    font=font,
)
cmp_img.save(MEDIA / f"{TAG}_before_after.png")
pbr.close()

(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(M, indent=2, ensure_ascii=False))
