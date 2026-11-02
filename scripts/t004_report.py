"""T004 報告：跑控制器實驗（PyBullet TORQUE_CONTROL）→ JSON/圖/動畫/報告 md。
用法：PYTHONPATH=. python3 scripts/t004_report.py 2026-11-02
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

from sixaxis import SixAxisRobot  # noqa: E402
from sixaxis.control import (  # noqa: E402
    Admittance,
    CartesianImpedance,
    CartesianPositionTracker,
    ComputedTorque,
    PDGravity,
    critical_gains,
)
from sixaxis.ctrlsim import prepare, run  # noqa: E402
from sixaxis.pbsim import PBRobot  # noqa: E402
from sixaxis.robot import RobotParams  # noqa: E402
from sixaxis.trajectory import joint_ptp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T004"
MEDIA, DATA, REP = ROOT / "docs/media", ROOT / "docs/test-reports/data", ROOT / "docs/test-reports"
Q0 = np.array([0.3, -1.1, 1.3, -0.7, 0.9, 0.2])
Q1 = Q0 + np.array([0.3, 0.2, -0.2, 0.3, -0.3, 0.4])
r = SixAxisRobot()
sim = PBRobot(r)
mdiag = np.diag(r.mass_matrix(Q0))
WN_PD, WN_CT, T_TR = 30.0, 40.0, 1.5
pd = PDGravity(r, mdiag * WN_PD**2, mdiag * 2 * WN_PD)
ct = ComputedTorque(r, *critical_gains(WN_CT))
wrong = SixAxisRobot(RobotParams(mass=tuple(1.2 * m for m in RobotParams().mass)))
ct_bad = ComputedTorque(r, *critical_gains(WN_CT), model=wrong)
M = {"date": DATE}


def step_info(rec, target, tol=0.02):
    e = np.abs(rec["q"] - target)
    d = np.abs(target - Q0)
    j = int(np.argmax(d))
    over = float(max(0.0, np.max((rec["q"][:, j] - target[j]) * np.sign(target[j] - Q0[j]))) / d[j])
    nrm = e / d
    out = np.where(np.any(nrm > tol, axis=1))[0]
    ts = float(rec["t"][out[-1] + 1]) if len(out) and out[-1] + 1 < len(rec["t"]) else float("nan")
    return {"settle_2pct_s": ts, "overshoot_ratio": over, "final_err_rad": float(e[-1].max()),
            "tau_peak_ratio": float(np.max(np.abs(rec["tau"]) / r.tau_max))}


# A) 設定點階躍（PD+g、CT）
prepare(sim, Q0)
recA_pd = run(sim, lambda t, q, qd: pd(q, qd, Q1), lambda t: (Q1, np.zeros(6)), 3.0)
prepare(sim, Q0)
recA_ct = run(sim, lambda t, q, qd: ct(q, qd, Q1), lambda t: (Q1, np.zeros(6)), 3.0)
M["step"] = {"PD+g": step_info(recA_pd, Q1), "CT": step_info(recA_ct, Q1)}

# B) 五次多項式軌跡追蹤
traj = joint_ptp(Q0, Q1, T_TR, "quintic")


def track(ctrl):
    prepare(sim, Q0)

    def ref(t):
        q, qd, _ = traj.state(min(t, T_TR))
        return q, qd

    def c(t, q, qd):
        qr, qdr, qddr = traj.state(min(t, T_TR))
        return ctrl(q, qd, qr, qdr, qddr)

    return run(sim, c, ref, T_TR + 0.5)


recB = {"PD+g": track(pd), "CT": track(ct), "CT (+20% mass model error)": track(ct_bad)}
M["tracking"] = {}
for k, rec in recB.items():
    e = rec["q"] - rec["q_ref"]
    M["tracking"][k] = {"rms_rad": float(np.sqrt(np.mean(e**2))), "max_rad": float(np.abs(e).max()),
                        "tau_peak_ratio": float(np.max(np.abs(rec["tau"]) / r.tau_max))}

# C) 笛卡兒阻抗：20 N 向下推力（t>0.2 s），不同剛性
T0 = r.fk(Q0)
F = np.array([0.0, 0.0, -20.0])
fz = lambda t: F if t > 0.2 else np.zeros(3)  # noqa: E731
M["impedance"], recC = [], {}
for k in (1000.0, 2000.0, 4000.0):
    imp = CartesianImpedance(r, [k] * 3 + [30.0] * 3, [80.0] * 3 + [0.3] * 3)
    prepare(sim, Q0)
    rec = run(sim, lambda t, q, qd, imp=imp: imp(q, qd, T0), lambda t: (Q0, np.zeros(6)), 2.0, ext_force=fz)
    dz = float(r.fk(rec["q"][-1])[2, 3] - T0[2, 3])
    zt = np.array([r.fk(q)[2, 3] - T0[2, 3] for q in rec["q"]])
    recC[k] = (rec["t"], zt)
    M["impedance"].append({"K_N_per_m": k, "expected_dz_mm": F[2] / k * 1e3, "measured_dz_mm": dz * 1e3,
                           "rel_err": abs(dz - F[2] / k) / abs(F[2] / k)})

# D) 導納（5 N，Ka=200 N/m）
FA, KA = np.array([0.0, 0.0, -5.0]), 200.0
adm = Admittance(2.0, 30.0, KA)
tr = CartesianPositionTracker(r, ComputedTorque(r, *critical_gains(60.0)), Q0)
fa = lambda t: FA if t > 0.2 else np.zeros(3)  # noqa: E731
prepare(sim, Q0)


def cadm(t, q, qd):
    x, v = adm.step(fa(t), 1e-3)
    T = T0.copy()
    T[:3, 3] += x
    return tr.step(q, qd, T, v, 1e-3)


recD = run(sim, cadm, lambda t: (Q0, np.zeros(6)), 3.0, ext_force=fa)
zD = np.array([r.fk(q)[2, 3] - T0[2, 3] for q in recD["q"]])
M["admittance"] = {"F_N": float(FA[2]), "Ka": KA, "expected_dz_mm": FA[2] / KA * 1e3,
                   "measured_dz_mm": float(zD[-1] * 1e3)}

# ------------------------------------------------------------ 圖
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for rec, nm in ((recA_pd, "PD+g"), (recA_ct, "CT")):
    ax[0].plot(rec["t"], np.abs(rec["q"] - Q1).max(1), label=nm)
ax[0].set(yscale="log", xlabel="t [s]", ylabel="max |q-q_d| [rad]", title="Step response")
ax[0].legend()
for k, rec in recB.items():
    ax[1].plot(rec["t"], np.abs(rec["q"] - rec["q_ref"]).max(1), label=k)
ax[1].set(yscale="log", xlabel="t [s]", ylabel="max tracking err [rad]", title="Quintic tracking (1.5 s)")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_joint_control.png", dpi=110)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for k, (t, z) in recC.items():
    ax[0].plot(t, z * 1e3, label=f"K={k:.0f} N/m")
    ax[0].axhline(F[2] / k * 1e3, color="gray", lw=0.6, ls=":")
ax[0].set(xlabel="t [s]", ylabel="z offset [mm]", title="Impedance: 20 N push")
ax[0].legend()
ax[1].plot(recD["t"], zD * 1e3)
ax[1].axhline(FA[2] / KA * 1e3, color="gray", ls=":")
ax[1].set(xlabel="t [s]", ylabel="z offset [mm]", title="Admittance: 5 N, Ka=200 N/m")
for i in range(6):
    ax[2].plot(recB["CT"]["t"], recB["CT"]["tau"][:, i], label=f"J{i+1}")
ax[2].set(xlabel="t [s]", ylabel="torque [N·m]", title="CT tracking torque")
ax[2].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_cartesian_control.png", dpi=110)

# ------------------------------------------------------------ 動畫：CT 追蹤 → 阻抗被推（2000 N/m）
sim.close()
vis = PBRobot(r, plane=True)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
SIZE = (480, 360)


def lab(img, left, right):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(20, 24, 32))
    d.text((10, 4), left, fill=(235, 238, 242), font=font)
    d.text((im.width - 10 - d.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
    return np.asarray(im)


recI = None
sim2 = PBRobot(r)
imp = CartesianImpedance(r, [2000.0] * 3 + [30.0] * 3, [80.0] * 3 + [0.3] * 3)
prepare(sim2, Q0)
recI = run(sim2, lambda t, q, qd: imp(q, qd, T0), lambda t: (Q0, np.zeros(6)), 2.0, ext_force=fz)
sim2.close()
segs = [("CT quintic tracking", recB["CT"], 1.0), ("Impedance K=2000 N/m, 20 N push @0.2 s", recI, 1.0)]
frames = []
for name, rec, _ in segs:
    for k in range(0, len(rec["t"]), 25):
        vis.set_state(rec["q"][k])
        views = [vis.render(target=(-0.3, 0.1, 0.4), distance=1.5, yaw=y, pitch=-20, size=SIZE) for y in (30, 120)]
        frames.append(np.hstack([lab(views[0], f"T004  {name}", "yaw 30"), lab(views[1], f"t = {rec['t'][k]:.2f} s", "yaw 120")]))
vis.close()
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(f).resize((640, 240), Image.LANCZOS) for f in frames]
pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=50, loop=0, optimize=True)
Image.fromarray(frames[len(frames) // 4]).save(MEDIA / f"{TAG}_key_a.png")
Image.fromarray(frames[-1]).save(MEDIA / f"{TAG}_key_b.png")
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(M, indent=2, ensure_ascii=False))
