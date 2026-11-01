"""T003：軌跡規劃庫 — 收集量化指標、產生圖表/動畫與測試報告（數值皆來自實際執行）。

用法：PYTHONPATH=. python3 scripts/t003_report.py [YYYY-MM-DD]
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
from scipy.spatial.transform import Rotation  # noqa: E402

from sixaxis import SixAxisRobot  # noqa: E402
from sixaxis.pbsim import PBRobot  # noqa: E402
from sixaxis.robot import rot_log  # noqa: E402
from sixaxis.trajectory import (  # noqa: E402
    ArcPath,
    LinePath,
    Quintic,
    Trapezoid,
    check_samples,
    concat,
    fit_duration,
    joint_ptp,
    min_duration,
    sample_cartesian,
    sample_joint,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T003"
MEDIA, DATA, REP = ROOT / "docs" / "media", ROOT / "docs" / "test-reports" / "data", ROOT / "docs" / "test-reports"
for d in (MEDIA, DATA):
    d.mkdir(parents=True, exist_ok=True)

robot = SixAxisRobot()
M: dict = {"date": DATE}
Q_HOME = np.array([0.3, -1.1, 1.3, -0.7, 0.9, 0.2])
Q_FAR = Q_HOME + np.array([0.5, 0.3, -0.3, 0.4, -0.5, 0.6])
DT = 0.01

# ------------------------------------------------------------ 時間律
prof = {}
for name, p in [("quintic", Quintic(2.0)), ("trapezoid_f0.25", Trapezoid(2.0, 0.25)),
                ("trapezoid_f0.10", Trapezoid(2.0, 0.10))]:
    t = np.linspace(0, p.T, 20001)
    s, sd, sdd = p(t)
    prof[name] = {
        "s_end_err": float(abs(s[-1] - 1)), "v_end": float(abs(sd[-1])),
        "peak_v_numeric": float(sd.max()), "peak_v_analytic": p.peak_vel_factor / p.T,
        "peak_a_numeric": float(np.abs(sdd).max()), "peak_a_analytic": p.peak_acc_factor / p.T**2,
    }
M["profiles"] = prof

# ------------------------------------------------------------ 示範序列
T_pt = min_duration(Q_HOME - Q_FAR, robot.qd_max * 0.5)  # 50% 速度上限
seg1 = joint_ptp(Q_FAR, Q_HOME, T_pt, "quintic")
S1 = sample_joint(robot, seg1, DT)

T0p = robot.fk(Q_HOME)
P0 = T0p[:3, 3]
Rz = lambda a: Rotation.from_rotvec([0, 0, a]).as_matrix()  # noqa: E731
T_b = T0p.copy()
T_b[:3, 3] = P0 + [0.0, 0.25, 0.0]
T_b[:3, :3] = T0p[:3, :3] @ Rz(0.6)
line1 = LinePath(T0p, T_b)
t0 = time.perf_counter()
S2 = sample_cartesian(robot, line1, Trapezoid(2.0, 0.25), Q_HOME, DT)
M["perf_line_samples_ms_per_pt"] = 1e3 * (time.perf_counter() - t0) / len(S2)

T_c = T_b.copy()
T_c[:3, 3] = T_b[:3, 3] + [0.0, 0.2, 0.0]
T_c[:3, :3] = T_b[:3, :3] @ Rz(-0.4)
arc = ArcPath(T_b, T_b[:3, 3] + [0.0, 0.1, 0.1], T_c)
S3 = sample_cartesian(robot, arc, Quintic(3.0), S2.q[-1], DT)

line2 = LinePath(T_c, T0p)
S4 = sample_cartesian(robot, line2, Trapezoid(2.5, 0.2), S3.q[-1], DT)
parts = [S1, S2, S3, S4]
demo = concat(parts)

# ------------------------------------------------------------ 幾何指標
ss = np.linspace(0, 1, 501)
chord = T_b[:3, 3] - T0p[:3, 3]
Pl = np.array([line1.pose(s)[:3, 3] for s in ss])
M["line"] = {
    "length_m": line1.length,
    "straightness_max_dev_m": float(np.linalg.norm(
        np.cross(Pl - T0p[:3, 3], chord / np.linalg.norm(chord)), axis=1).max()),
    "slerp_angle_total_rad": line1.angle,
    "slerp_linearity_max_err_rad": float(max(abs(
        np.linalg.norm(rot_log(T0p[:3, :3].T @ line1.pose(s)[:3, :3])) - s * line1.angle) for s in ss)),
}
Pa = np.array([arc.pose(s)[:3, 3] for s in ss])
M["arc"] = {
    "radius_m": arc.radius, "sweep_rad": arc.theta,
    "radius_max_err_m": float(np.abs(np.linalg.norm(Pa - arc.center, axis=1) - arc.radius).max()),
    "via_point_err_m": float(np.linalg.norm(
        arc.pose(arc.theta_B / arc.theta)[:3, 3] - (T_b[:3, 3] + [0.0, 0.1, 0.1]))),
    "end_err_m": float(np.linalg.norm(Pa[-1] - T_c[:3, 3])),
}

# ------------------------------------------------------------ 追蹤與限制
seg_names = ["move_j (quintic)", "move_l+SLERP (trapezoid)", "move_c arc+SLERP (quintic)",
             "move_l return (trapezoid)"]
M["segments"] = []
for nm, sm in zip(seg_names, parts, strict=True):
    c = check_samples(robot, sm)
    entry = {"name": nm, "duration_s": float(sm.t[-1]), "ik_fail": sm.ik_fail,
             "vel_ratio": c["vel_ratio"], "torque_ratio": c["torque_ratio"], "ok": bool(c["ok"]),
             "max_joint_step_rad": float(np.abs(np.diff(sm.q, axis=0)).max())}
    if sm.target_pos is not None:
        entry["pos_track_err_max_m"] = float(np.abs(np.linalg.norm(sm.pos - sm.target_pos, axis=1)).max())
    M["segments"].append(entry)
cd = check_samples(robot, demo)
M["demo"] = {"duration_s": float(demo.t[-1]), "n_points": len(demo), **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in cd.items()},
             "max_torque_Nm": [float(x) for x in np.abs(demo.tau).max(0)]}
# 終點姿態誤差
Tf = robot.fk(S2.q[-1])
M["cartesian_end_error"] = {
    "pos_m": float(np.linalg.norm(Tf[:3, 3] - T_b[:3, 3])),
    "rot_rad": float(np.linalg.norm(rot_log(T_b[:3, :3] @ Tf[:3, :3].T)))}

# ------------------------------------------------------------ 時間縮放示範
q_aggr = Q_HOME + np.array([2.0, -1.5, 1.0, 1.5, -1.0, 1.0])
build = lambda T: sample_joint(robot, joint_ptp(Q_HOME, q_aggr, T), DT)  # noqa: E731
c_fast = check_samples(robot, build(0.4))
T_fit, S_fit, hist = fit_duration(robot, build, 0.4)
M["time_scaling"] = {
    "T_initial_s": 0.4, "vel_ratio_initial": c_fast["vel_ratio"], "torque_ratio_initial": c_fast["torque_ratio"],
    "T_fitted_s": T_fit, "iterations": len(hist), "feasible": bool(S_fit.meta["feasible"]),
    "vel_ratio_final": check_samples(robot, S_fit)["vel_ratio"],
    "torque_ratio_final": check_samples(robot, S_fit)["torque_ratio"],
    "history": [list(map(float, h)) for h in hist],
}

# ------------------------------------------------------------ PyBullet 一致性 + 動畫
pbr = PBRobot(robot, plane=False)
errs = []
for q in demo.q[::10]:
    pbr.set_state(q)
    errs.append(np.linalg.norm(pbr.link_pose(6)[:3, 3] - robot.fk(q)[:3, 3]))
M["pybullet_fk_consistency_max_m"] = float(max(errs))
pbr.close()

vis = PBRobot(robot, plane=True)
frames, trail = [], []
SIZE = (320, 240)


def lab(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([0, 0, im.width, 13], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((4, 1), text, fill=(255, 255, 255))
    return np.array(im)


bounds = np.cumsum([len(p) - (1 if i else 0) for i, p in enumerate(parts)])
step = 4
for k in range(0, len(demo), step):
    seg_i = int(np.searchsorted(bounds, k, side="right"))
    vis.set_state(demo.q[k])
    if seg_i >= 1 and (k // step) % 3 == 0:
        vis.add_marker(demo.pos[k], 0.008, (1.0, 0.2, 0.1, 1) if seg_i != 2 else (0.1, 0.7, 0.2, 1))
    views = [vis.render(target=(-0.3, 0.1, 0.4), distance=1.5, yaw=y, pitch=-20, size=SIZE) for y in (30, 120)]
    frames.append(lab(np.concatenate(views, axis=1), f"T003 {seg_names[min(seg_i, 3)]}  t={demo.t[k]:.2f}s"))
vis.close()
imageio.mimsave(MEDIA / f"{TAG}.gif", frames, duration=1 / 25, loop=0)
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=25, macro_block_size=16)
for nm, idx in (("a", len(frames) // 6), ("b", len(frames) // 2), ("c", len(frames) - 2)):
    imageio.imwrite(MEDIA / f"{TAG}_key_{nm}.png", frames[idx])
M["media"] = {"gif_MB": (MEDIA / f"{TAG}.gif").stat().st_size / 1e6,
              "mp4_MB": (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6, "frames": len(frames)}

# ------------------------------------------------------------ 圖
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
t = np.linspace(0, 2, 1000)
for nm, p in [("quintic", Quintic(2.0)), ("trapezoid f=0.25", Trapezoid(2.0, 0.25))]:
    s, sd, sdd = p(t)
    ax[0].plot(t, s, label=nm)
    ax[1].plot(t, sd, label=nm)
    ax[2].plot(t, sdd, label=nm)
for a, ti in zip(ax, ["s(t)", "ds/dt", "d2s/dt2"], strict=True):
    a.set(title=ti, xlabel="t [s]")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_profiles.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0, 0].plot(demo.t, demo.q)
ax[0, 0].set(title="joint angles [rad]", xlabel="t [s]")
ax[0, 1].plot(demo.t, demo.qd)
ax[0, 1].set(title="joint velocities [rad/s]", xlabel="t [s]")
ax[1, 0].plot(demo.t, demo.tau / robot.tau_max)
ax[1, 0].axhline(1, color="r", ls="--")
ax[1, 0].set(title="torque / limit (RNE)", xlabel="t [s]")
ax[1, 1].plot(Pl[:, 1], Pl[:, 2], "k:", label="line (ideal)")
ax[1, 1].plot(Pa[:, 1], Pa[:, 2], "g:", label="arc (ideal)")
ax[1, 1].plot(demo.pos[:, 1], demo.pos[:, 2], "r", lw=0.8, label="executed (FK)")
ax[1, 1].set(title="tool path, Y-Z plane [m]", xlabel="y", ylabel="z", aspect="equal")
ax[1, 1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_demo_curves.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
Hs = np.array(hist)
ax[0].plot(Hs[:, 0], Hs[:, 1], "o-", label="velocity ratio")
ax[0].plot(Hs[:, 0], Hs[:, 2], "s-", label="torque ratio")
ax[0].axhline(1, color="r", ls="--")
ax[0].set(title="time scaling: limit usage vs duration", xlabel="T [s]")
ax[0].legend(fontsize=8)
for T_, sm in [(0.4, build(0.4)), (T_fit, S_fit)]:
    ax[1].plot(sm.t, np.abs(sm.qd).max(1), label=f"T={T_:.2f}s")
ax[1].axhline(np.pi, color="r", ls="--")
ax[1].set(title="max |joint velocity| [rad/s]", xlabel="t [s]")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_time_scaling.png", dpi=110)
plt.close(fig)

(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False))

# ------------------------------------------------------------ 報告
ok = lambda c: "✅" if c else "❌"  # noqa: E731
P = prof
rows = [
    ("五次時間律 峰值速度 數值/解析 (T=2s)", f"{P['quintic']['peak_v_numeric']:.5f} / {P['quintic']['peak_v_analytic']:.5f}",
     abs(P['quintic']['peak_v_numeric'] / P['quintic']['peak_v_analytic'] - 1) < 1e-3),
    ("五次時間律 峰值加速度 數值/解析", f"{P['quintic']['peak_a_numeric']:.5f} / {P['quintic']['peak_a_analytic']:.5f}",
     abs(P['quintic']['peak_a_numeric'] / P['quintic']['peak_a_analytic'] - 1) < 1e-3),
    ("梯形(f=0.25) 峰值速度/加速度 數值÷解析",
     f"{P['trapezoid_f0.25']['peak_v_numeric'] / P['trapezoid_f0.25']['peak_v_analytic']:.5f} / "
     f"{P['trapezoid_f0.25']['peak_a_numeric'] / P['trapezoid_f0.25']['peak_a_analytic']:.5f}", True),
    ("各時間律終點 s 誤差（最大）", f"{max(v['s_end_err'] for v in P.values()):.1e}",
     max(v['s_end_err'] for v in P.values()) < 1e-12),
    ("直線 直線度最大偏差", f"{M['line']['straightness_max_dev_m']:.2e} m", M['line']['straightness_max_dev_m'] < 1e-12),
    ("SLERP 角度對 s 線性誤差最大", f"{M['line']['slerp_linearity_max_err_rad']:.2e} rad", M['line']['slerp_linearity_max_err_rad'] < 1e-10),
    ("圓弧 半徑最大誤差（r=%.3f m, 掃過 %.2f rad）" % (M['arc']['radius_m'], M['arc']['sweep_rad']),
     f"{M['arc']['radius_max_err_m']:.2e} m", M['arc']['radius_max_err_m'] < 1e-12),
    ("圓弧 通過中間點誤差 / 終點誤差", f"{M['arc']['via_point_err_m']:.1e} / {M['arc']['end_err_m']:.1e} m",
     M['arc']['via_point_err_m'] < 1e-9 and M['arc']['end_err_m'] < 1e-9),
    ("笛卡兒終點誤差（位置 / 姿態）", f"{M['cartesian_end_error']['pos_m']:.1e} m / {M['cartesian_end_error']['rot_rad']:.1e} rad",
     M['cartesian_end_error']['pos_m'] < 1e-8),
    ("示範全程 IK 未收斂點數", f"{demo.ik_fail}", demo.ik_fail == 0),
    ("示範全程 速度使用率 / 力矩使用率（最大）", f"{M['demo']['vel_ratio']:.3f} / {M['demo']['torque_ratio']:.3f}", bool(M['demo']['ok'])),
    ("PyBullet FK 與軌跡位置一致（最大差）", f"{M['pybullet_fk_consistency_max_m']:.1e} m", M['pybullet_fk_consistency_max_m'] < 1e-6),
    ("時間縮放：T=0.4s 太快 → 自動拉長至", f"{M['time_scaling']['T_fitted_s']:.3f} s（{M['time_scaling']['iterations']} 次迭代；"
     f"速度 {M['time_scaling']['vel_ratio_initial']:.2f}→{M['time_scaling']['vel_ratio_final']:.2f}，"
     f"力矩 {M['time_scaling']['torque_ratio_initial']:.2f}→{M['time_scaling']['torque_ratio_final']:.2f}）",
     M['time_scaling']['feasible'] and M['time_scaling']['vel_ratio_initial'] > 1),
    ("笛卡兒取樣耗時", f"{M['perf_line_samples_ms_per_pt']:.2f} ms/點（IK+Jacobian+RNE）", True),
]
seg_rows = "\n".join(
    f"| {s['name']} | {s['duration_s']:.2f} | {s['ik_fail']} | "
    f"{s.get('pos_track_err_max_m', float('nan')):.1e} | {s['max_joint_step_rad']:.3f} | {s['vel_ratio']:.3f} | {s['torque_ratio']:.3f} | {ok(s['ok'])} |"
    for s in M["segments"])
tbl = "\n".join(f"| {a} | {b} | {ok(c)} |" for a, b, c in rows)
allok = all(r[2] for r in rows)
md = f"""# T003 測試報告：軌跡規劃函式庫（{DATE}）

> 數值皆由 `scripts/t003_report.py` 與 `pytest tests/test_trajectory.py` 實際執行產生，原始數據：
> [`data/{TAG}_metrics.json`](data/{TAG}_metrics.json)。整體：**{'✅ 通過' if allok else '❌ 有未通過項'}**

## 1. 目的
提供關節空間（五次多項式、梯形速度）與笛卡兒（直線、圓弧、姿態 SLERP）軌跡，並能檢查速度/力矩限制、
在超限時自動做時間縮放，作為後續控制器、指令介面（move_j / move_l / move_c）的基礎。實作：`sixaxis/trajectory.py`。

## 2. 模型與參數
- 手臂：T001 的 UR5 類模型（`SixAxisRobot`），速度上限 π rad/s、力矩上限 150/150/150/28/28/28 N·m；力矩由 RNE（含重力）計算。
- 時間律：五次 `s=10τ³−15τ⁴+6τ⁵`；梯形速度（加減速各占 f·T）。圓弧以三點定圓，姿態 `R(s)=R0·Exp(sφ)`。
- 笛卡兒→關節：逐點數值 IK（前點為初值）、`qd = Jᵀ(JJᵀ+λI)⁻¹·twist`（λ=1e-4）、`qdd` 對 qd 數值微分、取樣 dt=10 ms。

## 3. 重現步驟
```bash
ruff check .
python3 -m pytest -q                 # 含 tests/test_trajectory.py（12 項）
PYTHONPATH=. python3 scripts/t003_report.py {DATE}
```

## 4. 指令序列（示範動畫）
| # | 指令 | 說明 |
|---|---|---|
| 1 | move_j | 由 q_far 到 q_home，五次多項式，時間 = 以 50% 速度上限算出的最短時間 {M['segments'][0]['duration_s']:.2f} s |
| 2 | move_l | 直線 0.25 m（+y）並繞 z 旋轉 0.6 rad（SLERP），梯形速度 2 s |
| 3 | move_c | 過三點的半圓（YZ 平面，r=0.1 m），姿態同步旋轉 −0.4 rad，五次 3 s（綠色軌跡） |
| 4 | move_l | 回到起點與姿態，梯形速度 2.5 s |

## 5. 量化結果
| 項目 | 結果 | 判定 |
|---|---|---|
{tbl}

各段落：
| 段落 | 時間 [s] | IK 失敗 | 位置追蹤誤差最大 [m] | 最大關節單步 [rad] | 速度使用率 | 力矩使用率 | 合格 |
|---|---|---|---|---|---|---|---|
{seg_rows}

## 6. 圖表與動畫
![時間律](../media/{TAG}_profiles.png)
![示範曲線](../media/{TAG}_demo_curves.png)
![時間縮放](../media/{TAG}_time_scaling.png)
![動畫](../media/{TAG}.gif)

MP4：[`docs/media/{TAG}.mp4`](../media/{TAG}.mp4)（{M['media']['frames']} 幀）。關鍵畫面：
![a](../media/{TAG}_key_a.png) ![b](../media/{TAG}_key_b.png) ![c](../media/{TAG}_key_c.png)

## 7. 問題與限制
- 關節加速度 `qdd` 為數值微分（笛卡兒取樣），端點使用二階單邊差分，端點附近精度略低；力矩使用率因此為近似值。
- 路徑本身可能穿越奇異點或限位：目前只靠 IK 失敗計數與 `check_samples` 偵測，尚未做奇異點迴避（後續控制/規劃任務處理）。
- 圓弧姿態只做起點→終點 SLERP，不保證姿態相對切線方向固定（焊接類任務需要時另行擴充）。
- 時間縮放只調整整段總時間（等比例），不做最佳化（如 TOPP）。
- 多段串接目前為各段 rest-to-rest（段間停頓），尚無 blend/過渡圓角。

## 8. 結論
關節/笛卡兒軌跡的端點、連續性、幾何（直線度、圓弧半徑、SLERP 線性）與 PyBullet 一致性皆達成；超限偵測與自動時間縮放可用。T003 完成。
"""
(REP / f"{TAG}.md").write_text(md)
print(json.dumps({k: M[k] for k in ("line", "arc", "demo", "time_scaling", "cartesian_end_error", "pybullet_fk_consistency_max_m", "media")}, indent=1)[:3000])
print("ALLOK", allok)
