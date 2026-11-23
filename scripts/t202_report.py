"""T202 報告資料：PPO/SAC 學習曲線、最佳模型 vs T201 傳統基準（同一 200 目標）、停留/起點擾動測試、推論時間、動畫。

先訓練：PYTHONPATH=. python3 scripts/t202_train.py
PYTHONPATH=. python3 scripts/t202_report.py          # 評估＋圖＋動畫（4 行程）
PYTHONPATH=. python3 scripts/t202_report.py --anim   # 只重做圖/動畫（讀既有 JSON）
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from t201_report import label, summarize  # noqa: E402

from sixaxis.rlenv import R_DOWN, PlannerPolicy, ReachEnv, ResolvedRatePolicy, pose, run_episode  # noqa: E402
from sixaxis.rltrain import EVAL_SEEDS, MODELS, SB3Policy, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-23_T202"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
SEEDS = list(range(1000, 1200))  # 與 T201 基準相同的 200 個測試目標
RL = ["sac_ee", "ppo_ee", "sac_joint", "ppo_joint"]
BASE = {"planner_joint": ("joint", PlannerPolicy), "rr_ee": ("ee", ResolvedRatePolicy)}
LABEL = {"sac_ee": "SAC (ee)", "ppo_ee": "PPO (ee)", "sac_joint": "SAC (joint)", "ppo_joint": "PPO (joint)",
         "planner_joint": "Planner IK+quintic", "rr_ee": "Resolved-rate (ee)"}
COL = {"sac_ee": "#c0392b", "ppo_ee": "#1f6fb4", "sac_joint": "#e8a33d", "ppo_joint": "#6aa9de",
       "planner_joint": "#333333", "rr_ee": "#2a9d5c"}


def make(name, **kw):
    if name in BASE:
        mode, cls = BASE[name]
        env = ReachEnv(action_mode=mode, **kw)
        return env, cls(env)
    env = ReachEnv(action_mode=name.split("_")[1], **kw)
    return env, SB3Policy(load(MODELS / f"{name}_best.zip"))


def keep(r):
    return {k: r[k] for k in ("seed", "d0", "success", "t_success", "d_final", "return", "path_ratio", "tau_ratio",
                              "collision", "steps")}


def job(args):
    name, kind = args
    import torch

    torch.set_num_threads(1)
    if kind == "test":  # 主測試：200 目標、成功即終止（同 T201）
        env, pol = make(name)
        res = [run_episode(env, pol, s) for s in SEEDS]
        env.close()
        return name, kind, [keep(r) for r in res]
    if kind == "evalset":  # 學習曲線評估集上的基準回報（畫水平參考線）
        env, pol = make(name)
        res = [run_episode(env, pol, s) for s in EVAL_SEEDS]
        env.close()
        return name, kind, [keep(r) for r in res]
    if kind == "hold":  # 不終止、跑滿 5 s：到達後能否停住
        env, pol = make(name, terminate_on_success=False)
        out = []
        for s in SEEDS[:50]:
            r = run_episode(env, pol, s, record=True)
            d = r["log"]["d"]
            last = d[-40:]  # 最後 2 s
            out.append({"seed": s, "d_final": float(d[-1]), "in_tol_last2s": float(np.mean(last < env.tol)),
                        "d_last2s_max": float(last.max()), "t_success": r["t_success"],
                        "qd_final": float(np.linalg.norm(r["log"]["qd"][-1]))})
        env.close()
        return name, kind, out
    if kind == "q0wide":  # 起點擾動 0.1 → 0.3 rad（訓練分佈外）
        env, pol = make(name, q0_noise=0.3)
        res = [run_episode(env, pol, s) for s in SEEDS[:100]]
        env.close()
        return name, kind, [keep(r) for r in res]
    if kind == "timing":  # 每步決策時間（不含物理）
        env, pol = make(name)
        obs, _ = env.reset(seed=SEEDS[0])
        pol.reset()
        ts = []
        for _ in range(60):
            t0 = time.perf_counter()
            a = pol(obs)
            ts.append(time.perf_counter() - t0)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset(seed=SEEDS[1])
                pol.reset()
        env.close()
        return name, kind, {"mean_ms": 1e3 * float(np.mean(ts[5:])), "p95_ms": 1e3 * float(np.percentile(ts[5:], 95))}
    raise ValueError(kind)


def run_all():
    t0 = time.time()
    names = RL + list(BASE)
    jobs = [(n, k) for k in ("test", "hold", "q0wide", "timing", "evalset") for n in names]
    with Pool(4) as pool:
        out = pool.map(job, jobs)
    M = {"test": {}, "per_episode": {}, "hold": {}, "q0wide": {}, "timing": {}, "evalset": {}, "train": {}}
    for name, kind, res in out:
        if kind == "test":
            M["test"][name] = summarize(res)
            M["per_episode"][name] = res
        elif kind == "hold":
            M["hold"][name] = {
                "n": len(res), "d_final_mean_mm": 1e3 * float(np.mean([r["d_final"] for r in res])),
                "d_final_max_mm": 1e3 * float(np.max([r["d_final"] for r in res])),
                "in_tol_last2s": float(np.mean([r["in_tol_last2s"] for r in res])),
                "all_in_tol_frac": float(np.mean([r["in_tol_last2s"] == 1.0 for r in res])),
                "d_last2s_max_mm": 1e3 * float(np.max([r["d_last2s_max"] for r in res])),
                "qd_final_mean": float(np.mean([r["qd_final"] for r in res])),
            }
        elif kind == "q0wide":
            M["q0wide"][name] = summarize(res)
        elif kind == "timing":
            M["timing"][name] = res
        elif kind == "evalset":
            M["evalset"][name] = {"return_mean": float(np.mean([r["return"] for r in res])),
                                  "success_rate": float(np.mean([r["success"] for r in res]))}
    # 同目標成功時間比（相對規劃基準）
    tp = np.array([r["t_success"] for r in M["per_episode"]["planner_joint"]], float)
    M["time_ratio_vs_planner"] = {}
    for n in RL + ["rr_ee"]:
        t = np.array([r["t_success"] if r["t_success"] is not None else np.nan for r in M["per_episode"][n]], float)
        M["time_ratio_vs_planner"][n] = {"mean": float(np.nanmean(t / tp)), "faster_frac": float(np.mean(t < tp))}
    for n in RL:
        lg = json.loads((MODELS / f"{n}_log.json").read_text())
        curve = lg["curve"]
        first90 = next((c for c in curve if c["success_rate"] >= 0.9), None)
        first100 = next((c for c in curve if c["success_rate"] >= 1.0), None)
        M["train"][n] = {
            "timesteps": lg["timesteps"], "wall_s": lg["wall_s"], "completed": lg["completed"], "best": lg["best"],
            "steps_to_90": first90["timesteps"] if first90 else None,
            "wall_to_90": first90["wall_s"] if first90 else None,
            "steps_to_100": first100["timesteps"] if first100 else None,
            "final": curve[-1], "n_evals": len(curve), "hparams": lg["hparams"],
            "model_kb": (MODELS / f"{n}_best.zip").stat().st_size / 1024,
            "fps": lg["timesteps"] / lg["wall_s"],
        }
    M["bounds"] = time_bounds(M)
    M["runtime_s"] = time.time() - t0
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    return M


def time_bounds(M):
    """最短時間下界（每個測試目標）：
    * ee 動作：法蘭線速度 |v| ≤ v_max·√3（各軸 ≤ v_max）但姿態保持與 ω_max 縮放只會更慢；取直線 + 單軸飽和的
      保守下界 t_ee = (d0 − tol)/(√3·v_max)，及「沿直線且速率 v_max」的參考時間 t_line = (d0 − tol)/v_max。
    * joint 動作：|q̇_i| ≤ ω_max ⇒ 到達某一工具朝下 IK 解至少 t_j = min_sol max_i |Δq_i|/ω_max
      （無限加速度；joint 模式不要求工具朝下，故此值為參考而非嚴格下界）。"""
    env = ReachEnv()
    out = {}
    for s in SEEDS:
        _, info = env.reset(seed=s)
        q0, g, d0 = env.q_ref.copy(), info["goal"], info["distance"]
        sols = env.robot.ik(pose(R_DOWN, g), q0)
        tj = min(float(np.max(np.abs(q - q0))) for q in sols) / env.omega_max
        out[s] = {"t_line": max(d0 - env.tol, 0) / env.v_max, "t_ee_lb": max(d0 - env.tol, 0) / (np.sqrt(3) * env.v_max),
                  "t_joint_ik": tj}
    env.close()
    res = {}
    for n in RL + list(BASE):
        ref = "t_line" if n.endswith("ee") else "t_joint_ik"
        r = np.array([rr["t_success"] / max(out[rr["seed"]][ref], 0.05) for rr in M["per_episode"][n]
                      if rr["t_success"] is not None and out[rr["seed"]][ref] > 0.1])
        res[n] = {"ref": ref, "ratio_mean": float(r.mean()), "ratio_min": float(r.min()),
                  "ratio_p95": float(np.percentile(r, 95))}
    res["t_line_mean"] = float(np.mean([v["t_line"] for v in out.values()]))
    res["t_joint_ik_mean"] = float(np.mean([v["t_joint_ik"] for v in out.values()]))
    return res


# ------------------------------------------------------------------ 圖
def smooth(y, k=3):
    y = np.asarray(y, float)
    if y.size < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def plots(M):
    logs = {n: json.loads((MODELS / f"{n}_log.json").read_text())["curve"] for n in RL}
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    for c, mode in enumerate(("ee", "joint")):
        for n in RL:
            if not n.endswith(mode):
                continue
            cv = logs[n]
            t = np.array([x["timesteps"] for x in cv]) / 1e3
            ax[0, c].plot(t, 100 * np.array([x["success_rate"] for x in cv]), color=COL[n], lw=1.6, label=LABEL[n])
            ax[1, c].plot(t, [x["return_mean"] for x in cv], color=COL[n], lw=1.6, label=LABEL[n])
            tr = [(x["timesteps"] / 1e3, x["train_success"]) for x in cv if x["train_success"] is not None]
            if tr:
                a = np.array(tr)
                ax[0, c].plot(a[:, 0], 100 * a[:, 1], color=COL[n], lw=0.8, ls=":", alpha=0.8)
        for b in BASE:
            ax[1, c].axhline(M["evalset"][b]["return_mean"], color=COL[b], ls="--", lw=1, label=LABEL[b])
        ax[0, c].set_title(f"action = {mode}: eval success rate (solid) / train (dotted)")
        ax[0, c].set_ylabel("success [%]")
        ax[0, c].set_ylim(-3, 103)
        ax[1, c].set_title(f"action = {mode}: eval mean return (50 fixed goals)")
        ax[1, c].set_ylabel("return (dense)")
        ax[1, c].set_xlabel("environment steps [×1000]")
        for a in (ax[0, c], ax[1, c]):
            a.set_xscale("symlog", linthresh=10)
            a.set_xlim(left=0)
        ax[1, c].set_ylim(-40, 5)
        ax[0, c].legend(fontsize=8)
        ax[1, c].legend(fontsize=7, loc="lower right")
    for n in RL:
        cv = logs[n]
        ax[0, 2].plot(np.array([x["wall_s"] for x in cv]) / 60, 100 * np.array([x["success_rate"] for x in cv]),
                      color=COL[n], lw=1.6, label=LABEL[n])
        ax[1, 2].plot(np.array([x["wall_s"] for x in cv]) / 60, [x["d_final_mean_mm"] for x in cv], color=COL[n],
                      lw=1.6, label=LABEL[n])
    ax[0, 2].set_title("success vs wall-clock (1 CPU core each)")
    ax[0, 2].set_ylabel("success [%]")
    ax[1, 2].set_title("mean final distance vs wall-clock")
    ax[1, 2].set_ylabel("final distance [mm]")
    ax[1, 2].set_yscale("log")
    ax[1, 2].axhline(20, color="k", ls=":", lw=1)
    ax[1, 2].set_xlabel("wall-clock [min]")
    ax[0, 2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_learning.png", dpi=110)
    plt.close(fig)

    names = RL + list(BASE)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for n in names:
        ts = [r["t_success"] for r in M["per_episode"][n] if r["t_success"] is not None]
        ax[0].hist(ts, bins=np.arange(0, 5.05, 0.1), histtype="step", lw=1.5, color=COL[n], label=LABEL[n])
    ax[0].set_xlabel("time to success [s]")
    ax[0].set_ylabel("episodes")
    ax[0].set_title("success time on 200 test goals")
    ax[0].legend(fontsize=7)
    tp = {r["seed"]: r["t_success"] for r in M["per_episode"]["planner_joint"]}
    for n in RL + ["rr_ee"]:
        pts = [(tp[r["seed"]], r["t_success"]) for r in M["per_episode"][n] if r["t_success"] is not None]
        if pts:
            a = np.array(pts)
            ax[1].scatter(a[:, 0], a[:, 1], s=6, color=COL[n], alpha=0.6, label=LABEL[n])
    ax[1].plot([0, 3], [0, 3], "k--", lw=0.8)
    ax[1].set_xlabel("planner time to success [s]")
    ax[1].set_ylabel("policy time to success [s]")
    ax[1].set_title("same goal: policy vs planner")
    ax[1].legend(fontsize=7)
    x = np.arange(len(names))
    w = 0.38
    sr = [100 * M["test"][n]["success_rate"] for n in names]
    hd = [100 * M["hold"][n]["all_in_tol_frac"] for n in names]
    ax[2].bar(x - w / 2, sr, w, color=[COL[n] for n in names], label="success (terminate)")
    ax[2].bar(x + w / 2, hd, w, color=[COL[n] for n in names], alpha=0.45, hatch="//",
              label="hold: in tol whole last 2 s")
    for i, (a, b) in enumerate(zip(sr, hd, strict=True)):
        ax[2].text(i - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=7)
        ax[2].text(i + w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=7)
    ax[2].set_xticks(x, [LABEL[n].replace(" (", "\n(").replace(" IK", "\nIK") for n in names], fontsize=7)
    ax[2].set_ylim(0, 112)
    ax[2].set_ylabel("[%]")
    ax[2].legend(fontsize=7, loc="lower left")
    ax[2].set_title("success / hold-at-goal")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_compare.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 動畫與曲線
def anim(M):
    goal_seeds = [1003, 1011, 1027, 1042]
    W, H = 400, 300
    cols, logs = [], {}
    for n in ["sac_ee", "ppo_ee", "sac_joint", "ppo_joint"]:
        env, pol = make(n, render_mode="rgb_array", render_size=(W, H))
        env.reset(seed=0)
        env.render()
        frames = []
        for s in goal_seeds:
            res = run_episode(env, pol, s, record=True, frames=True)
            tag = "SUCCESS" if res["success"] else ("COLLISION" if res["collision"] else "timeout")
            k = len(res["frames"])
            for i, f in enumerate(res["frames"]):
                d = res["log"]["d"][i - 1] if i else res["d0"]
                st = f" {tag}" if i == k - 1 else ""
                frames.append(label(f, f"{LABEL[n]} | seed {s} | t={i * env.dt_c:.2f}s d={1e3 * d:.0f}mm{st}"))
            frames += [frames[-1]] * 6
            if s == goal_seeds[0]:
                logs[n] = res["log"]
        env.close()
        cols.append(frames)
    env, pol = make("planner_joint")
    logs["planner_joint"] = run_episode(env, pol, goal_seeds[0], record=True)["log"]
    env.close()
    L = max(len(c) for c in cols)
    cols = [c + [c[-1]] * (L - len(c)) for c in cols]
    grid = [np.concatenate([np.concatenate([cols[0][k], cols[1][k]], axis=1),
                            np.concatenate([cols[2][k], cols[3][k]], axis=1)], axis=0) for k in range(L)]
    imageio.mimsave(MEDIA / f"{TAG}.mp4", grid, fps=20, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((640, 480), Image.LANCZOS) for x in grid[::2]]
    pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)
    Image.fromarray(grid[0]).save(MEDIA / f"{TAG}_key_start.png")
    Image.fromarray(grid[min(25, L - 1)]).save(MEDIA / f"{TAG}_key_mid.png")

    show = ["planner_joint", "sac_ee", "ppo_ee", "sac_joint"]
    fig, ax = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
    for c, n in enumerate(show):
        lg = logs[n]
        t = lg["t"]
        ax[0, c].plot(t, lg["q"])
        ax[0, c].set_title(f"{LABEL[n]}: q [rad]")
        ax[1, c].plot(t, lg["qd"])
        ax[1, c].set_title("q̇ [rad/s]")
        ax[2, c].plot(t, lg["tau"])
        ax[2, c].set_title("τ [N·m]")
        ax[3, c].semilogy(t, 1e3 * lg["d"], "k")
        ax[3, c].axhline(20, color="r", ls=":", lw=1)
        ax[3, c].set_title("distance to goal [mm]")
        ax[3, c].set_xlabel("t [s]")
    ax[0, 0].legend([f"q{i + 1}" for i in range(6)], fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.2), sharey=True)
    for c, n in enumerate(show):
        ax[c].plot(logs[n]["t"], logs[n]["a"])
        ax[c].set_title(f"{LABEL[n]}: action a ∈ [−1, 1]")
        ax[c].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_actions.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    if "--anim" in sys.argv:
        M = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
    else:
        M = run_all()
    plots(M)
    anim(M)
    print(json.dumps({k: M[k] for k in ("test", "hold", "q0wide", "timing", "evalset", "time_ratio_vs_planner", "bounds",
                                         "runtime_s")}, indent=1))
    print(json.dumps({n: {k: v for k, v in M["train"][n].items() if k not in ("hparams",)} for n in RL}, indent=1))
