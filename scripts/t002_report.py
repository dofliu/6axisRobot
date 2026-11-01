"""T002：收集量化指標、產生圖表/動畫，並輸出 JSON（報告 Markdown 由 scripts/t002_write_md.py 產生）。

用法：python3 scripts/t002_report.py [YYYY-MM-DD]
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
from PIL import Image, ImageDraw  # noqa: E402

from sixaxis import SixAxisRobot  # noqa: E402
from sixaxis.pbsim import PBRobot  # noqa: E402
from sixaxis.robot import rot_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TASK = "T002"
MEDIA = ROOT / "docs" / "media"
DATA = ROOT / "docs" / "test-reports" / "data"
MEDIA.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
tag = f"{DATE}_{TASK}"

rng = np.random.default_rng(20261101)
robot = SixAxisRobot()
pbr = PBRobot(robot)  # 數值比對用：無地面，避免底座與地面接觸干擾自由擺動
M: dict = {"date": DATE}


def stats(x):
    x = np.asarray(x)
    return {"max": float(x.max()), "mean": float(x.mean()), "rms": float(np.sqrt((x**2).mean()))}


# ---------------------------------------------------------------- FK vs PyBullet
pos_e, rot_e = [], []
for q in rng.uniform(-np.pi, np.pi, (1000, 6)):
    pbr.set_state(q)
    Tp, Tm = pbr.link_pose(6), robot.fk(q)
    pos_e.append(np.linalg.norm(Tp[:3, 3] - Tm[:3, 3]))
    rot_e.append(np.linalg.norm(rot_log(Tp[:3, :3] @ Tm[:3, :3].T)))
M["fk_pos_err_m"], M["fk_rot_err_rad"] = stats(pos_e), stats(rot_e)

# ---------------------------------------------------------------- IK 往返
res, nsol, found, near_ok = [], [], 0, 0
for q in rng.uniform(-np.pi, np.pi, (1000, 6)):
    T = robot.fk(q)
    sols = robot.ik_all(T, q)
    nsol.append(len(sols))
    for s in sols:
        Te = robot.fk(s)
        res.append(np.linalg.norm(Te[:3, 3] - T[:3, 3]) + np.linalg.norm(Te[:3, :3] - T[:3, :3]))
    found += any(
        np.allclose(np.sin(s), np.sin(q), atol=1e-6) and np.allclose(np.cos(s), np.cos(q), atol=1e-6)
        for s in sols
    )
    b = robot.ik(T, q)
    near_ok += bool(b) and np.abs(np.sin(b[0] - q)).max() < 1e-6
M["ik_residual"] = stats(res)
M["ik_original_found"] = found
M["ik_nearest_ok"] = int(near_ok)
M["ik_solution_count_hist"] = {int(k): int((np.array(nsol) == k).sum()) for k in sorted(set(nsol))}

# ---------------------------------------------------------------- Jacobian
h, jerr = 1e-6, []
for q in rng.uniform(-np.pi, np.pi, (200, 6)):
    J, Jn = robot.jacobian(q), np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        Tp, Tm = robot.fk(q + dq), robot.fk(q - dq)
        Jn[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        Jn[3:, i] = rot_log(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
    jerr.append(np.abs(J - Jn).max())
M["jacobian_err"] = stats(jerr)

# ---------------------------------------------------------------- 逆動力學 vs PyBullet
id_e = []
for _ in range(500):
    q, qd, qdd = rng.uniform(-np.pi, np.pi, 6), rng.uniform(-2, 2, 6), rng.uniform(-5, 5, 6)
    id_e.append(np.abs(pbr.inverse_dynamics(q, qd, qdd) - robot.rne(q, qd, qdd)).max())
M["id_err_Nm"] = stats(id_e)

# ---------------------------------------------------------------- 自由擺動：能量守恆 + PyBullet 比對
q0 = np.array([0.3, -0.4, 0.8, -0.5, 0.6, 0.2])
qd0 = np.array([0.5, -0.3, 0.4, 1.0, -0.8, 0.6])
dt_sw, T_sw = 1e-3, 2.0
t0 = time.perf_counter()
ts, qs, qds = robot.simulate(q0, qd0, dt_sw, int(T_sw / dt_sw))
M["swing_sim_wall_s"] = time.perf_counter() - t0
E = np.array([robot.energy(a, b) for a, b in zip(qs[::10], qds[::10])])  # (N,2)
Etot = E.sum(axis=1)
M["energy"] = {
    "E0_J": float(Etot[0]),
    "drift_J": float(Etot.max() - Etot.min()),
    "rel_drift": float((Etot.max() - Etot.min()) / abs(Etot[0])),
    "KE_range_J": [float(E[:, 0].min()), float(E[:, 0].max())],
    "dt_s": dt_sw,
    "T_s": T_sw,
}
tp, qp, qdp = pbr.free_swing(q0, qd0, 1e-4, int(T_sw / 1e-4), sample_every=10)
Ep = np.array([sum(robot.energy(a, b)) for a, b in zip(qp[::10], qdp[::10])])
n = min(len(qs), len(qp))
dq_t = np.abs(qs[:n] - qp[:n]).max(axis=1)
M["pybullet_swing"] = {
    "dq_at_0.1s": float(dq_t[100]),
    "dq_at_0.2s": float(dq_t[200]),
    "dq_at_0.5s": float(dq_t[500]),
    "dq_at_1s": float(dq_t[1000]),
    "dq_at_2s": float(dq_t[-1]),
    "pb_energy_drift_J": float(Ep.max() - Ep.min()),
    "pb_dt_s": 1e-4,
}


# ---------------------------------------------------------------- 效能
def timeit(f, n):
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


qq = rng.uniform(-1, 1, 6)
Tt = robot.fk(qq)
M["perf"] = {
    "fk_us": timeit(lambda: robot.fk(qq), 2000) * 1e6,
    "jacobian_us": timeit(lambda: robot.jacobian(qq), 1000) * 1e6,
    "ik_all_ms": timeit(lambda: robot.ik_all(Tt, qq), 200) * 1e3,
    "ik_select_ms": timeit(lambda: robot.ik(Tt, qq), 200) * 1e3,
    "rne_us": timeit(lambda: robot.rne(qq, qq, qq), 500) * 1e6,
    "mass_matrix_us": timeit(lambda: robot.mass_matrix(qq), 200) * 1e6,
    "forward_dynamics_us": timeit(lambda: robot.forward_dynamics(qq, qq, qq), 200) * 1e6,
    "pybullet_id_us": timeit(lambda: pbr.inverse_dynamics(qq, qq, qq), 500) * 1e6,
}

# ---------------------------------------------------------------- 奇異點
gen = np.array([0.3, -1.1, 1.4, -0.6, 1.0, 0.2])
w, e_ = gen.copy(), gen.copy()
w[4], e_[2] = 0.0, 0.0
M["singularity"] = {
    "generic": {
        "flags": robot.singularities(gen),
        "sigma_min": robot.sigma_min(gen),
        "manip": robot.manipulability(gen),
    },
    "wrist(q5=0)": {"flags": robot.singularities(w), "sigma_min": robot.sigma_min(w)},
    "elbow(q3=0)": {"flags": robot.singularities(e_), "sigma_min": robot.sigma_min(e_)},
}

# ---------------------------------------------------------------- 示範 A：笛卡兒圓（解析 IK 追蹤）
R_down = np.diag([1.0, -1.0, -1.0])
center, rad, T_c, fps = np.array([0.45, 0.0, 0.35]), 0.12, 5.0, 20
nA = int(T_c * fps)
sA = np.linspace(0, 1, nA + 1)
phi = 2 * np.pi * (10 * sA**3 - 15 * sA**4 + 6 * sA**5)  # 最小急動度
path = np.stack(
    [center[0] + rad * np.cos(phi), center[1] + rad * np.sin(phi), np.full_like(phi, center[2])], 1
)
q_prev = np.array([0.0, -1.2, 1.8, -2.1, -1.57, 0.0])
qA = []
for p in path:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_down, p
    sols = robot.ik(T, q_prev)
    assert sols, "圓軌跡不可達"
    q_prev = sols[0]
    qA.append(q_prev)
qA = np.array(qA)
dtA = T_c / nA
qdA = np.gradient(qA, dtA, axis=0)
qddA = np.gradient(qdA, dtA, axis=0)
tauA = np.array([robot.rne(a, b, c) for a, b, c in zip(qA, qdA, qddA)])
tauA_pb = np.array([pbr.inverse_dynamics(a, b, c) for a, b, c in zip(qA, qdA, qddA)])
pos_track = np.array([np.linalg.norm(robot.fk(q)[:3, 3] - p) for q, p in zip(qA, path)])
M["circle_demo"] = {
    "radius_m": rad,
    "duration_s": T_c,
    "tracking_err_m_max": float(pos_track.max()),
    "joint_jump_max_rad": float(np.abs(np.diff(qA, axis=0)).max()),
    "max_joint_vel_rad_s": float(np.abs(qdA).max()),
    "max_torque_Nm": np.abs(tauA).max(axis=0).tolist(),
    "torque_limits_Nm": robot.tau_max.tolist(),
    "limits_violation": robot.check_limits(qA[0], qdA.max(axis=0), np.abs(tauA).max(axis=0)),
    "id_vs_pybullet_max_Nm": float(np.abs(tauA - tauA_pb).max()),
}

# ---------------------------------------------------------------- 圖表
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].hist(np.array(pos_e) * 1e9, bins=40, color="#2a6fdb")
ax[0].set(title="FK vs PyBullet position error (1000 poses)", xlabel="error [nm]", ylabel="count")
ax[1].hist(np.array(id_e), bins=40, color="#d9822b")
ax[1].set(title="Inverse dynamics vs PyBullet (500 samples)", xlabel="max |dtau| [N·m]")
fig.tight_layout()
fig.savefig(MEDIA / f"{tag}_errors.png", dpi=110)
plt.close(fig)

tt = ts[::10]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(tt, E[:, 0], label="KE")
ax[0].plot(tt, E[:, 1], label="PE")
ax[0].plot(tt, Etot, "k", lw=2, label="Total (this model, RK4 dt=1ms)")
ax[0].plot(tp[::10], Ep, "r--", label="Total (PyBullet, dt=0.1ms)")
ax[0].set(title="Frictionless free swing: energy", xlabel="t [s]", ylabel="J")
ax[0].legend(fontsize=7)
ax[1].plot(tt, (Etot - Etot[0]) * 1e6)
ax[1].set(title="Total energy drift (this model)", xlabel="t [s]", ylabel="E - E0 [µJ]")
fig.tight_layout()
fig.savefig(MEDIA / f"{tag}_energy.png", dpi=110)
plt.close(fig)

tA = np.linspace(0, T_c, nA + 1)
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
ax[0, 0].plot(tA, qA)
ax[0, 0].set(title="Joint angles (IK circle)", ylabel="rad")
ax[0, 1].plot(tA, qdA)
ax[0, 1].set(title="Joint velocities", ylabel="rad/s")
for j in range(6):
    ax[1, 0].plot(tA, tauA[:, j], label=f"J{j + 1}")
    ax[1, 0].plot(tA, tauA_pb[:, j], "k:", lw=0.7)
ax[1, 0].set(title="Joint torques (solid: RNE, dotted: PyBullet)", xlabel="t [s]", ylabel="N·m")
ax[1, 0].legend(fontsize=7, ncol=3)
ax[1, 1].plot(tA, pos_track * 1e9)
ax[1, 1].set(title="Tool position tracking error", xlabel="t [s]", ylabel="nm")
fig.tight_layout()
fig.savefig(MEDIA / f"{tag}_circle_curves.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- 動畫
pbr.close()
pbr = PBRobot(robot, plane=True)  # 渲染用（含地面）
for k in range(0, nA + 1, 5):
    pbr.add_marker(path[k], 0.008)
views = [
    dict(target=(0.2, 0.0, 0.35), distance=1.5, yaw=50, pitch=-25),
    dict(target=(0.2, 0.0, 0.3), distance=1.3, yaw=0, pitch=-89),
]


def frame(q, label):
    pbr.set_state(q)
    ims = [pbr.render(size=(400, 300), **v) for v in views]
    img = Image.fromarray(np.hstack(ims))
    ImageDraw.Draw(img).text((8, 6), label, fill=(0, 0, 0))
    return np.array(img)


frames = [frame(q, f"A: analytic-IK circle  t={t:4.2f}s") for q, t in zip(qA, tA)]
sw_step = int(round(1 / fps / dt_sw))
for k in range(0, len(qs), sw_step):
    frames.append(frame(qs[k], f"B: frictionless free swing (RNE/FD)  t={k * dt_sw:4.2f}s"))
pbr.close()

imageio.mimsave(MEDIA / f"{tag}.mp4", frames, fps=fps, macro_block_size=None, quality=7)
pal = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
pal[0].save(
    MEDIA / f"{tag}.gif",
    save_all=True,
    append_images=pal[1:],
    duration=int(1000 / fps),
    loop=0,
    optimize=True,
)
for name, idx in (("start", 0), ("circle_mid", nA // 2), ("swing", nA + 1 + 20)):
    Image.fromarray(frames[idx]).save(MEDIA / f"{tag}_{name}.png")
M["media"] = {
    "frames": len(frames),
    "gif_MB": (MEDIA / f"{tag}.gif").stat().st_size / 1e6,
    "mp4_MB": (MEDIA / f"{tag}.mp4").stat().st_size / 1e6,
}
(DATA / f"{tag}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(M, indent=2, ensure_ascii=False))
