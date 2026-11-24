"""T203 報告資料：pick 學習曲線、最佳模型 vs 腳本式基準（同 200 物件位置）、長時保持、起點擾動、
未見物件（重物件/長方塊）、夾持力–摩擦極限掃描、推論時間、圖與動畫。

先訓練：PYTHONPATH=. python3 scripts/t203_train.py
PYTHONPATH=. python3 scripts/t203_report.py          # 評估＋圖＋動畫（4 行程）
PYTHONPATH=. python3 scripts/t203_report.py --anim   # 只重做圖/動畫（讀既有 JSON）
PYTHONPATH=. python3 scripts/t203_report.py --plots  # 只重畫統計圖
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from t201_report import label  # noqa: E402

from sixaxis.gripper import BLOCK, CUBE, f_min  # noqa: E402
from sixaxis.rlpick import PickEnv, RandomPickPolicy, ScriptedPickPolicy, run_pick_episode  # noqa: E402
from sixaxis.rltrain import EVAL_SEEDS, MODELS, SB3Policy, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-24_T203"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
SEEDS = list(range(1000, 1200))  # 200 個測試物件位置（與學習曲線評估集 5000–5049 不重疊）
RL = ["sac_pick_s0", "sac_pick_s1", "ppo_pick_s0", "ppo_pick_s1"]
BASE = ["scripted", "random"]
LABEL = {"sac_pick_s0": "SAC seed0", "sac_pick_s1": "SAC seed1", "ppo_pick_s0": "PPO seed0",
         "ppo_pick_s1": "PPO seed1", "scripted": "Scripted FSM", "random": "Random"}
COL = {"sac_pick_s0": "#c0392b", "sac_pick_s1": "#e8a33d", "ppo_pick_s0": "#1f6fb4", "ppo_pick_s1": "#6aa9de",
       "scripted": "#333333", "random": "#999999"}
HEAVY = replace(CUBE, name="cube_1kg", mass=1.0)
VARIANTS = {"q0wide": dict(q0_noise=0.3), "heavy": dict(obj=HEAVY), "block": dict(obj=BLOCK),
            "long": dict(max_steps=200)}


def make(name, **kw):
    env = PickEnv(**kw)
    if name == "scripted":
        return env, ScriptedPickPolicy(env)
    if name == "random":
        return env, RandomPickPolicy(env)
    return env, SB3Policy(load(MODELS / f"{name}_best.zip"))


KEYS = ("seed", "success", "collision", "drop", "t_success", "t_grasp", "steps", "return", "h_final", "h_max",
        "hold_frac", "regrasp", "tau_ratio", "obj_xy")


def keep(r):
    return {k: r[k] for k in KEYS}


def summarize(res):
    ts = [r["t_success"] for r in res if r["t_success"] is not None]
    tg = [r["t_grasp"] for r in res if r["t_grasp"] is not None]
    return {
        "n": len(res), "success_rate": float(np.mean([r["success"] for r in res])),
        "ever_success": float(np.mean([r["t_success"] is not None for r in res])),
        "grasp_rate": float(np.mean([r["t_grasp"] is not None for r in res])),
        "collision_rate": float(np.mean([r["collision"] for r in res])),
        "drop_rate": float(np.mean([r["drop"] for r in res])),
        "t_success_mean": float(np.mean(ts)) if ts else None,
        "t_success_p95": float(np.percentile(ts, 95)) if ts else None,
        "t_grasp_mean": float(np.mean(tg)) if tg else None,
        "hold_frac_mean": float(np.mean([r["hold_frac"] for r in res])),
        "regrasp_mean": float(np.mean([r["regrasp"] for r in res])),
        "h_final_mean_mm": 1e3 * float(np.mean([r["h_final"] for r in res])),
        "return_mean": float(np.mean([r["return"] for r in res])),
        "tau_ratio_max": float(np.max([r["tau_ratio"] for r in res])),
    }


def job(args):
    name, kind = args
    import torch

    torch.set_num_threads(1)
    if kind in ("test", "evalset"):
        env, pol = make(name)
        res = [run_pick_episode(env, pol, s) for s in (SEEDS if kind == "test" else EVAL_SEEDS)]
        env.close()
        return name, kind, [keep(r) for r in res]
    if kind in VARIANTS:
        env, pol = make(name, **VARIANTS[kind])
        n = 50 if kind == "long" else 100
        res = []
        for s in SEEDS[:n]:
            r = run_pick_episode(env, pol, s, record=kind == "long")
            k = keep(r)
            if kind == "long":  # 訓練視界 5 s 之後（5–10 s）是否持續抓住並保持高度
                lg = r["log"]
                late = lg["t"] > 5.0 + 1e-9
                k["hold_late"] = float(np.mean((lg["h"][late] >= env.h_lift) & lg["grasped"][late])) if late.any() else 0.0
                k["h_late_std_mm"] = 1e3 * float(np.std(lg["h"][late])) if late.any() else None
            res.append(k)
        env.close()
        return name, kind, res
    if kind == "fsweep":  # 夾持力掃描（腳本基準）：摩擦極限 F_min = m(g + a)/(2μ)
        out = []
        fm = f_min(CUBE.mass, 0.8)
        for ratio in (0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0):
            env, pol = make("scripted", f_grip=ratio * fm)
            res = [run_pick_episode(env, pol, s) for s in SEEDS[:10]]
            env.close()
            out.append({"ratio": ratio, "f_grip": ratio * fm, "success_rate": float(np.mean([r["success"] for r in res])),
                        "h_final_mm": 1e3 * float(np.mean([r["h_final"] for r in res]))})
        return name, kind, {"f_min": fm, "sweep": out, **lift_accel()}
    if kind == "timing":
        env, pol = make(name)
        obs, _ = env.reset(seed=SEEDS[0])
        pol.reset()
        ts, tenv = [], []
        for _ in range(100):
            t0 = time.perf_counter()
            a = pol(obs)
            t1 = time.perf_counter()
            obs, _, term, trunc, _ = env.step(a)
            ts.append(t1 - t0)
            tenv.append(time.perf_counter() - t1)
            if term or trunc:
                obs, _ = env.reset(seed=SEEDS[1])
                pol.reset()
        env.close()
        return name, kind, {"mean_ms": 1e3 * float(np.mean(ts[5:])), "p95_ms": 1e3 * float(np.percentile(ts[5:], 95)),
                            "env_step_ms": 1e3 * float(np.mean(tenv[5:]))}
    raise ValueError(kind)


def lift_accel(seeds=SEEDS[:10]):
    """腳本基準抬升起步的 TCP 鉛直加速度峰值（2 ms 子步取樣、Savitzky–Golay 21 點三次二階導數平滑），
    用於含慣性的摩擦極限 F_min(a) = m(g + a)/(2μ) ⇒ F/F_min(0) ≥ 1 + a/g。"""
    from scipy.signal import savgol_filter

    env = PickEnv()
    zs, peaks = [], []
    env.substep_cb = lambda e: zs.append(e.tcp()[0][2])
    for s in seeds:
        zs.clear()
        r = run_pick_episode(env, ScriptedPickPolicy(env), s, record=True)
        i0 = int(np.argmax(r["log"]["grasped"])) * env.n_sub
        a = savgol_filter(np.array(zs), 21, 3, deriv=2, delta=env.dt)
        peaks.append(float(a[i0:].max()))
    env.close()
    a = float(np.mean(peaks))
    return {"a_peak": a, "a_peak_max": float(np.max(peaks)), "ratio_needed": 1.0 + a / 9.81}


def run_all():
    t0 = time.time()
    names = RL + BASE
    jobs = [(n, "test") for n in names] + [(n, "evalset") for n in RL + ["scripted"]]
    jobs += [(n, k) for k in VARIANTS for n in RL + ["scripted"]]
    jobs += [("scripted", "fsweep")] + [(n, "timing") for n in RL + ["scripted"]]
    with Pool(4) as pool:
        out = pool.map(job, jobs)
    M = {"test": {}, "per_episode": {}, "evalset": {}, "timing": {}, "train": {}}
    M.update({k: {} for k in VARIANTS})
    for name, kind, res in out:
        if kind == "test":
            M["test"][name] = summarize(res)
            M["per_episode"][name] = res
        elif kind == "evalset":
            M["evalset"][name] = summarize(res)
        elif kind in VARIANTS:
            M[kind][name] = summarize(res)
            if kind == "long":
                M[kind][name]["hold_late_mean"] = float(np.mean([r["hold_late"] for r in res]))
                M[kind][name]["hold_late_all"] = float(np.mean([r["hold_late"] == 1.0 for r in res]))
                sd = [r["h_late_std_mm"] for r in res if r["h_late_std_mm"] is not None]
                M[kind][name]["h_late_std_mm_mean"] = float(np.mean(sd)) if sd else None
        elif kind == "fsweep":
            M["fsweep"] = res
        elif kind == "timing":
            M["timing"][name] = res
    # 同物件位置的成功時間比（相對腳本基準）
    ts = {r["seed"]: r["t_success"] for r in M["per_episode"]["scripted"]}
    M["time_ratio_vs_scripted"] = {}
    for n in RL:
        rr = [r["t_success"] / ts[r["seed"]] for r in M["per_episode"][n]
              if r["t_success"] is not None and ts[r["seed"]] is not None]
        M["time_ratio_vs_scripted"][n] = {"mean": float(np.mean(rr)) if rr else None,
                                          "faster_frac": float(np.mean(np.array(rr) < 1)) if rr else None}
    for n in RL:
        lg = json.loads((MODELS / f"{n}_log.json").read_text())
        curve = lg["curve"]
        first = {p: next((c for c in curve if c["success_rate"] >= p), None) for p in (0.5, 0.9)}
        M["train"][n] = {
            "timesteps": lg["timesteps"], "wall_s": lg["wall_s"], "completed": lg["completed"], "best": lg["best"],
            "steps_to_50": first[0.5]["timesteps"] if first[0.5] else None,
            "steps_to_90": first[0.9]["timesteps"] if first[0.9] else None,
            "wall_to_90": first[0.9]["wall_s"] if first[0.9] else None,
            "final": curve[-1], "n_evals": len(curve), "hparams": lg["hparams"],
            "model_kb": (MODELS / f"{n}_best.zip").stat().st_size / 1024, "fps": lg["timesteps"] / lg["wall_s"],
        }
    best = max(RL, key=lambda n: (M["train"][n]["best"]["success_rate"], M["train"][n]["best"]["return_mean"]))
    M["best_model"] = best
    (MODELS / "T203_best.txt").write_text(f"{best}\n", encoding="utf-8")
    M["runtime_s"] = time.time() - t0
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    return M


# ------------------------------------------------------------------ 圖
def plots(M):
    logs = {n: json.loads((MODELS / f"{n}_log.json").read_text())["curve"] for n in RL}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for n in RL:
        cv = logs[n]
        t = np.array([x["timesteps"] for x in cv]) / 1e3
        ax[0].plot(t, 100 * np.array([x["success_rate"] for x in cv]), color=COL[n], lw=1.7, label=LABEL[n])
        ax[0].plot(t, 100 * np.array([x["grasp_rate"] for x in cv]), color=COL[n], lw=0.9, ls=":")
        ax[1].plot(np.array([x["wall_s"] for x in cv]) / 60, 100 * np.array([x["success_rate"] for x in cv]),
                   color=COL[n], lw=1.7, label=LABEL[n])
        ax[2].plot(t, [x["return_mean"] for x in cv], color=COL[n], lw=1.7, label=LABEL[n])
    ax[2].axhline(M["evalset"]["scripted"]["return_mean"], color="k", ls="--", lw=1, label="Scripted FSM")
    ax[0].axhline(100 * M["evalset"]["scripted"]["success_rate"], color="k", ls="--", lw=1, label="Scripted FSM")
    for a in (ax[0], ax[2]):
        a.set_xlim(left=0)
    ax[0].set_title("eval success (solid) / ever-grasped (dotted), 50 fixed objects")
    ax[0].set_xlabel("environment steps [×1000]")
    ax[0].set_ylabel("[%]")
    ax[1].set_title("success vs wall-clock (1 CPU core each)")
    ax[1].set_xlabel("wall-clock [min]")
    ax[2].set_title("eval mean return")
    ax[2].set_xlabel("environment steps [×1000]")
    for a in ax[:2]:
        a.set_ylim(-3, 103)
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_learning.png", dpi=110)
    plt.close(fig)

    names = RL + ["scripted"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for n in names:
        tsu = [r["t_success"] for r in M["per_episode"][n] if r["t_success"] is not None]
        if tsu:
            ax[0].hist(tsu, bins=np.arange(0, 5.05, 0.1), histtype="step", lw=1.5, color=COL[n], label=LABEL[n])
    ax[0].set_xlabel("time to first success (grasped & h ≥ 10 cm) [s]")
    ax[0].set_ylabel("episodes")
    ax[0].set_title("200 test objects")
    ax[0].legend(fontsize=7)
    kinds = [("test", "nominal"), ("q0wide", "q0 ±0.3 rad"), ("heavy", "1 kg cube"), ("block", "40×60×80 block"),
             ("long", "10 s episode")]
    x = np.arange(len(kinds))
    w = 0.16
    for i, n in enumerate(names):
        v = [100 * M[k][n]["success_rate"] for k, _ in kinds]
        ax[1].bar(x + (i - 2) * w, v, w, color=COL[n], label=LABEL[n])
    ax[1].set_xticks(x, [lab for _, lab in kinds], fontsize=8)
    ax[1].set_ylim(0, 128)
    ax[1].set_ylabel("success at episode end [%]")
    ax[1].set_title("robustness / generalization")
    ax[1].legend(fontsize=7, ncol=5, loc="upper center")
    fs = M["fsweep"]
    ax[2].plot([s["ratio"] for s in fs["sweep"]], [100 * s["success_rate"] for s in fs["sweep"]], "ko-", label="success")
    ax2 = ax[2].twinx()
    ax2.plot([s["ratio"] for s in fs["sweep"]], [s["h_final_mm"] for s in fs["sweep"]], "s--", color="#1f6fb4",
             label="final lift h")
    ax2.set_ylabel("mean final h [mm]", color="#1f6fb4")
    ax[2].axvline(1.0, color="r", ls=":", lw=1.2, label="F_min = mg/(2μ)")
    ax[2].axvline(fs["ratio_needed"], color="m", ls="--", lw=1.2, label=f"1 + a_peak/g = {fs['ratio_needed']:.2f}")
    ax[2].set_xscale("log")
    ax[2].set_xticks([0.25, 0.5, 1, 2, 5], ["0.25", "0.5", "1", "2", "5"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("F_grip / F_min")
    ax[2].set_ylabel("success [%]")
    ax[2].set_title(f"grip force sweep (scripted, F_min = {fs['f_min']:.2f} N)")
    ax[2].legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_compare.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.6), sharey=True)
    for c, n in enumerate(names):
        for r in M["per_episode"][n]:
            x_, y_ = r["obj_xy"]
            ax[c].scatter(y_, x_, s=10, marker="o" if r["success"] else "x",
                          color=COL[n] if r["success"] else "red")
        ax[c].set_title(f"{LABEL[n]}: {100 * M['test'][n]['success_rate']:.0f} %", fontsize=9)
        ax[c].set_xlabel("object y [m]")
        ax[c].set_aspect("equal")
    ax[0].set_ylabel("object x [m]")
    fig.suptitle("test object positions: ○ success, × failure (red)", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_map.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 動畫與曲線
def anim(M):
    best = M["best_model"]
    other = next(n for n in RL if n.startswith("ppo" if best.startswith("sac") else "sac")
                 and M["test"][n]["success_rate"] == max(M["test"][m]["success_rate"] for m in RL
                                                       if m[:3] == n[:3]))
    seeds = [1003, 1011, 1027]
    W, H = 400, 300
    logs, rows = {}, []
    for n in [best, other, "scripted"]:
        env, pol = make(n, render_mode="rgb_array", render_size=(W, H))
        frames_a, frames_b = [], []
        for s in seeds:
            res = run_pick_episode(env, pol, s, record=True)
            if s == seeds[0]:
                logs[n] = res["log"]
            # 同一回合重播兩視角（決定性環境：重新執行同種子同策略）
            obs, _ = env.reset(seed=s)
            pol.reset()
            for k in range(env.max_steps):
                a = pol(obs)
                obs, _, term, trunc, info = env.step(a)
                if k % 2 == 1 or term or trunc:
                    st = " SUCCESS" if info["success"] else (" grasped" if info["grasped"] else "")
                    txt = f"{LABEL[n]} | seed {s} | t={env.k * env.dt_c:.1f}s h={1e3 * info['h']:.0f}mm{st}"
                    f1 = env.pbr.render(target=(0.42, 0.0, 0.28), distance=1.45, yaw=55, pitch=-24, size=(W, H),
                                        supersample=1)
                    po = info["obj"]
                    f2 = env.pbr.render(target=(po[0], po[1], po[2] + 0.03), distance=0.6, yaw=0, pitch=-15, size=(W, H),
                                        supersample=1)
                    frames_a.append(label(f1, txt))
                    frames_b.append(label(f2, "close-up tracking the object (pads on both sides)"))
                if term or trunc:
                    break
            frames_a += [frames_a[-1]] * 5
            frames_b += [frames_b[-1]] * 5
        env.close()
        rows.append([np.concatenate([a, b], axis=1) for a, b in zip(frames_a, frames_b, strict=True)])
    L = max(len(r) for r in rows)
    rows = [r + [r[-1]] * (L - len(r)) for r in rows]
    grid = [np.concatenate([r[k] for r in rows], axis=0) for k in range(L)]
    imageio.mimsave(MEDIA / f"{TAG}.mp4", grid, fps=10, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((416, 468), Image.LANCZOS) for x in grid]
    pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)
    for tag, k in (("start", 0), ("grasp", 14), ("lift", 30)):
        Image.fromarray(grid[min(k, L - 1)]).save(MEDIA / f"{TAG}_key_{tag}.png")

    show = [best, other, "scripted"]
    fig, ax = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    for c, n in enumerate(show):
        lg = logs[n]
        t = lg["t"]
        ax[0, c].plot(t, lg["tcp"][:, 2] * 1e3, label="TCP z")
        ax[0, c].plot(t, lg["obj"][:, 2] * 1e3, label="object z")
        ax[0, c].plot(t, 1e3 * lg["d"], label="|p_obj − p_tcp|")
        ax[0, c].axhline(1e3 * (0.125 + 0.10), color="r", ls=":", lw=1, label="lift goal (obj z)")
        ax[0, c].set_title(f"{LABEL[n]}: heights / distance [mm]")
        ax[0, c].legend(fontsize=7)
        ax[1, c].plot(t, lg["fc"][:, 0], label="left pad")
        ax[1, c].plot(t, lg["fc"][:, 1], "--", label="right pad")
        ax[1, c].plot(t, 1e3 * lg["s"], label="finger stroke [mm]")
        ax[1, c].set_title("pad normal force [N] / stroke")
        ax[1, c].legend(fontsize=7)
        ax[2, c].plot(t, lg["a"][:, :3])
        ax[2, c].step(t, lg["a"][:, 3], "k", lw=1, where="post")
        ax[2, c].set_title("action: v_x, v_y, v_z (color), grip (black)")
        ax[3, c].plot(t, lg["tau"])
        ax[3, c].set_title("joint torque τ [N·m]")
        ax[3, c].set_xlabel("t [s]")
    ax[2, 0].legend(["vx", "vy", "vz", "grip"], fontsize=7)
    ax[3, 0].legend([f"τ{i + 1}" for i in range(6)], fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_episode.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    for c, n in enumerate(show):
        ax[0, c].plot(logs[n]["t"], logs[n]["q"])
        ax[0, c].set_title(f"{LABEL[n]}: q [rad]")
        ax[1, c].plot(logs[n]["t"], logs[n]["qd"])
        ax[1, c].set_title("q̇ [rad/s]")
        ax[1, c].set_xlabel("t [s]")
    ax[0, 0].legend([f"q{i + 1}" for i in range(6)], fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    if "--plots" in sys.argv:  # 只重畫統計圖
        plots(json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8")))
        sys.exit()
    if "--anim" in sys.argv:
        M = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
        if "a_peak" not in M["fsweep"]:  # 舊 JSON 補量測抬升加速度
            M["fsweep"].update(lift_accel())
            (DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        M = run_all()
    plots(M)
    anim(M)
    print(json.dumps({k: M[k] for k in ("test", "q0wide", "heavy", "block", "long", "fsweep", "timing",
                                         "time_ratio_vs_scripted", "best_model", "runtime_s")}, indent=1))
    print(json.dumps({n: {k: v for k, v in M["train"][n].items() if k != "hparams"} for n in RL}, indent=1))
