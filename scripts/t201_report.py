"""T201 報告資料：reach 環境的基準策略比較、低階控制驗證、目標分佈、吞吐量、動畫。

PYTHONPATH=. python3 scripts/t201_report.py          # 全部（約 5 分鐘，4 行程）
PYTHONPATH=. python3 scripts/t201_report.py --anim   # 只重做動畫/圖（讀既有 JSON）
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
from PIL import Image, ImageDraw  # noqa: E402

from sixaxis.rlenv import (  # noqa: E402
    PlannerPolicy,
    RandomPolicy,
    ReachEnv,
    ResolvedRatePolicy,
    run_episode,
    throughput,
)

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-22_T201"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
N_EP = 200
SEEDS = list(range(1000, 1000 + N_EP))
CONFIGS = {
    "planner_joint": ("joint", PlannerPolicy),
    "rr_joint": ("joint", ResolvedRatePolicy),
    "rr_ee": ("ee", ResolvedRatePolicy),
    "random_joint": ("joint", RandomPolicy),
    "random_ee": ("ee", RandomPolicy),
}


def make_policy(cls, env):
    return cls(env, 0) if cls is RandomPolicy else cls(env)


def eval_config(args):
    name, term = args
    mode, cls = CONFIGS[name]
    env = ReachEnv(action_mode=mode, terminate_on_success=term)
    pol = make_policy(cls, env)
    seeds = SEEDS if term else SEEDS[:50]
    res = [run_episode(env, pol, s) for s in seeds]
    env.close()
    return name, term, res


def summarize(res):
    succ = np.array([r["success"] for r in res])
    ts = np.array([r["t_success"] for r in res if r["t_success"] is not None])
    return {
        "n": len(res),
        "success_rate": float(succ.mean()),
        "collision_rate": float(np.mean([r["collision"] for r in res])),
        "t_success_mean": float(ts.mean()) if ts.size else None,
        "t_success_p95": float(np.percentile(ts, 95)) if ts.size else None,
        "d_final_mean_mm": 1e3 * float(np.mean([r["d_final"] for r in res])),
        "d_final_max_mm": 1e3 * float(np.max([r["d_final"] for r in res])),
        "return_mean": float(np.mean([r["return"] for r in res])),
        "return_std": float(np.std([r["return"] for r in res])),
        "path_ratio_mean": float(np.mean([r["path_ratio"] for r in res if r["d0"] > 0.05])),
        "tau_ratio_max": float(np.max([r["tau_ratio"] for r in res])),
        "d0_mean_mm": 1e3 * float(np.mean([r["d0"] for r in res])),
    }


def discrete_error(us, dt, kp, kd):
    """離散時間理論：PyBullet 半隱式 Euler（q̇⁺ = q̇ + Δt·q̈，q⁺ = q + Δt·q̇⁺）+ 精確計算力矩 q̈ = a_cmd
    （模型完全抵消 M、C、g），參考 r_n = r_{n−1} + u_n Δt（控制器用子步末端的參考）。回傳 r_n − q_{n+1}。"""
    q = qd = r = 0.0
    out = []
    for u in us:
        r += u * dt
        a = kp * (r - q) + kd * (u - qd)
        qd += dt * a
        q += dt * qd
        out.append(r - q)
    return np.array(out)


def velocity_step_test():
    """關節速度指令步階：從靜止給 u = 0.5 rad/s（0.5 s）再反向 −0.5（0.5 s）。
    理論（速度前饋 + 臨界阻尼計算力矩，忽略飽和）：誤差 e = q_d − q 滿足 ë + 2ω_n ė + ω_n² e = 0，
    速度步階 Δu 的初值 ė(0) = Δu → e(t) = Δu t e^{−ω_n t}，峰值 Δu/(e·ω_n) 於 t = 1/ω_n。"""
    out = []
    for j in range(6):
        env = ReachEnv(max_steps=1000)
        env.reset(seed=0, options={"q0": env.q_home, "goal": [0.4, 0.3, 0.3]})
        rec = []

        def cb(q_d, qd_d, tau, env=env, rec=rec, j=j):
            q, qd = env.pbr.get_state()
            rec.append((q_d[j] - q[j], qd_d[j], qd[j], tau[j]))

        env.substep_cb = cb
        a = np.zeros(6)
        for k in range(20):
            a[j] = 0.5 if k < 10 else -0.5
            env.step(a)
        env.close()
        r = np.array(rec)
        wn = np.sqrt(env.kp)
        n1 = 10 * env.n_sub
        us = np.r_[np.full(n1, 0.5), np.full(n1, -0.5)]
        ed = discrete_error(us, env.dt, env.kp, env.kd)
        out.append({
            "joint": j + 1,
            "disc_peak1": float(np.max(np.abs(ed[:n1]))),
            "disc_peak2": float(np.max(np.abs(ed[n1:]))),
            "disc_maxdev": float(np.max(np.abs(ed - r[:, 0]))),
            "peak1": float(np.max(np.abs(r[:n1, 0]))),
            "peak2": float(np.max(np.abs(r[n1:, 0]))),
            "theory1": 0.5 / (np.e * wn),
            "theory2": 1.0 / (np.e * wn),
            "t_peak1": float(np.argmax(np.abs(r[:n1, 0])) + 1) * env.dt,
            "tau_max": float(np.max(np.abs(r[:, 3]))),
            "tau_limit": float(env.robot.tau_max[j]),
            "trace": r[:, 0].tolist() if j in (1, 5) else None,
            "disc_trace": ed.tolist() if j == 1 else None,
            "qd_trace": r[:, 2].tolist() if j in (1, 5) else None,
        })
    return out, env.dt, wn


def goal_stats(n=3000):
    env = ReachEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    pts = rng.uniform(env.goal_low, env.goal_high, (n, 3))
    ok = np.array([env.goal_reachable(p) for p in pts])
    sampled = np.array([env.sample_goal() for _ in range(600)])
    home = env.robot.fk(env.q_home)[:3, 3]
    env.close()
    return {"box_reachable_frac": float(ok.mean()), "pts": pts.tolist(), "ok": ok.tolist(),
            "sampled": sampled.tolist(), "home": home.tolist()}


def bench_throughput():
    out = {}
    for mode in ("joint", "ee"):
        env = ReachEnv(action_mode=mode)
        throughput(env, 50)
        out[mode] = throughput(env, 2000)
        env.close()
    return out


def run_all():
    t0 = time.time()
    jobs = [(n, True) for n in CONFIGS] + [("planner_joint", False), ("rr_joint", False), ("rr_ee", False)]
    with Pool(4) as pool:
        results = pool.map(eval_config, jobs)
        vs = pool.apply_async(velocity_step_test)
        gs = pool.apply_async(goal_stats)
        vstep, dt, wn = vs.get()
        goals = gs.get()
    thr = bench_throughput()
    M = {"baselines": {}, "static": {}, "per_episode": {}}
    for name, term, res in results:
        key = "baselines" if term else "static"
        M[key][name] = summarize(res)
        if term:
            M["per_episode"][name] = [
                {k: r[k] for k in ("seed", "d0", "success", "t_success", "d_final", "return", "path_ratio",
                                   "tau_ratio", "collision")} for r in res
            ]
    # 規劃 vs 解析率：同一目標的成功時間比
    tp = np.array([r["t_success"] for r in M["per_episode"]["planner_joint"]], float)
    tr = np.array([r["t_success"] for r in M["per_episode"]["rr_joint"]], float)
    M["planner_vs_rr_time_ratio"] = float(np.nanmean(tp / tr))
    M["planner_faster_frac"] = float(np.nanmean(tp < tr))
    M["vstep"], M["dt"], M["wn"] = vstep, dt, wn
    M["goals"] = goals
    M["throughput"] = thr
    M["runtime_s"] = time.time() - t0
    M["env"] = env_params()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    return M


def env_params():
    e = ReachEnv()
    p = {
        "control_hz": 1 / e.dt_c, "sim_hz": 1 / e.dt, "n_sub": e.n_sub, "max_steps": e.max_steps, "tol": e.tol,
        "v_tol": e.v_tol, "omega_max": e.omega_max, "v_max": e.v_max, "wn": float(np.sqrt(e.kp)), "e_max": e.e_max,
        "q0_noise": e.q0_noise, "c_action": e.c_action, "r_success": e.r_success, "r_collision": e.r_collision,
        "goal_low": e.goal_low.tolist(), "goal_high": e.goal_high.tolist(), "q_home": e.q_home.tolist(),
        "obs_dim": int(e.observation_space.shape[0]),
    }
    e.close()
    return p


# ------------------------------------------------------------------ 圖
LABEL = {"planner_joint": "Planner (IK+quintic, joint)", "rr_joint": "Resolved-rate (joint)",
         "rr_ee": "Resolved-rate (ee)", "random_joint": "Random (joint)", "random_ee": "Random (ee)"}
COL = {"planner_joint": "#1f6fb4", "rr_joint": "#e07b20", "rr_ee": "#2a9d5c", "random_joint": "#888888",
       "random_ee": "#bbbbbb"}


def plots(M):
    # 1) 基準比較：成功時間分佈、最終距離、回報
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for n in ("planner_joint", "rr_joint", "rr_ee"):
        ts = [r["t_success"] for r in M["per_episode"][n] if r["t_success"] is not None]
        ax[0].hist(ts, bins=np.arange(0, 5.05, 0.1), histtype="step", lw=1.6, color=COL[n], label=LABEL[n])
    ax[0].set_xlabel("time to success [s]")
    ax[0].set_ylabel("episodes")
    ax[0].legend(fontsize=8)
    ax[0].set_title("success time (N=200)")
    for n in CONFIGS:
        d0 = [r["d0"] for r in M["per_episode"][n]]
        ret = [r["return"] for r in M["per_episode"][n]]
        ax[1].scatter(d0, ret, s=7, color=COL[n], label=LABEL[n], alpha=0.8)
    ax[1].set_xlabel("initial distance d0 [m]")
    ax[1].set_ylabel("episode return (dense)")
    ax[1].set_title("return vs initial distance")
    ax[1].legend(fontsize=7)
    names = list(CONFIGS)
    sr = [M["baselines"][n]["success_rate"] * 100 for n in names]
    ax[2].bar(range(len(names)), sr, color=[COL[n] for n in names])
    for i, v in enumerate(sr):
        ax[2].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax[2].set_xticks(range(len(names)), [LABEL[n].replace(" (", "\n(") for n in names], fontsize=7)
    ax[2].set_ylabel("success rate [%]")
    ax[2].set_ylim(0, 110)
    ax[2].set_title("success rate")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_baselines.png", dpi=110)
    plt.close(fig)

    # 2) 速度步階驗證
    dt, wn = M["dt"], M["wn"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for v in M["vstep"]:
        if v["trace"] is None:
            continue
        e = np.array(v["trace"])
        t = (np.arange(e.size) + 1) * dt
        ax[0].plot(t, 1e3 * e, label=f"joint {v['joint']} sim")
        ax[1].plot(t, v["qd_trace"], label=f"joint {v['joint']} q̇")
    t = np.arange(0, 1.0, dt)
    th = np.where(t < 0.5, 0.5 * t * np.exp(-wn * t), 0.5 * t * np.exp(-wn * t)
                  - 1.0 * (t - 0.5) * np.exp(-wn * (t - 0.5)))
    ax[0].plot(t, 1e3 * th, "k--", lw=1, label="continuous theory Δu·t·e^{-ω_n t}")
    dtr = next(v["disc_trace"] for v in M["vstep"] if v["disc_trace"] is not None)
    ax[0].plot((np.arange(len(dtr)) + 1) * dt, 1e3 * np.array(dtr), "r:", lw=1.5, label="discrete theory (semi-implicit Euler)")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("tracking error q_d − q [mrad]")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"velocity step 0→0.5→−0.5 rad/s (ω_n={wn:.0f})")
    ax[1].axhline(0.5, color="k", lw=0.5, ls=":")
    ax[1].axhline(-0.5, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("q̇ [rad/s]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_vstep.png", dpi=110)
    plt.close(fig)

    # 3) 目標分佈
    g = M["goals"]
    pts, ok, sp, home = np.array(g["pts"]), np.array(g["ok"]), np.array(g["sampled"]), np.array(g["home"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, (i, j), lab in ((ax[0], (0, 1), ("x", "y")), (ax[1], (0, 2), ("x", "z"))):
        a.scatter(pts[~ok, i], pts[~ok, j], s=3, c="#d33", alpha=0.4, label="box, unreachable (tool-down)")
        a.scatter(sp[:, i], sp[:, j], s=4, c="#1f6fb4", alpha=0.6, label="sampled goals")
        a.plot(home[i], home[j], "k*", ms=12, label="home flange")
        a.plot(0, 0, "ks", ms=8, label="base")
        a.set_xlabel(f"{lab[0]} [m]")
        a.set_ylabel(f"{lab[1]} [m]")
        a.set_aspect("equal")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_title(f"top view (reachable {100 * g['box_reachable_frac']:.1f}% of box)")
    ax[1].set_title("side view")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_goals.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 動畫
def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(20, 24, 30))
    d.text((6, 3), text, fill=(240, 240, 240))
    return np.asarray(im)


def anim(M):
    goal_seeds = [1003, 1011, 1027, 1042]
    pols = [("joint", PlannerPolicy, "Planner IK+quintic"), ("ee", ResolvedRatePolicy, "Resolved-rate (ee)"),
            ("joint", RandomPolicy, "Random policy")]
    W, H = 400, 300
    cols, logs = [], {}
    for mode, cls, name in pols:
        env = ReachEnv(action_mode=mode, render_mode="rgb_array", render_size=(W, H))
        env.reset(seed=0)
        env.render()
        pol = make_policy(cls, env)
        frames = []
        for s in goal_seeds:
            res = run_episode(env, pol, s, record=True, frames=True)
            tag = "SUCCESS" if res["success"] else "timeout"
            n = len(res["frames"])
            for i, f in enumerate(res["frames"]):
                d = res["log"]["d"][max(i - 1, 0)] if i else res["d0"]
                st = f" {tag}" if i == n - 1 else ""
                frames.append(label(f, f"{name} | seed {s} | t={i * env.dt_c:.2f}s d={1e3 * d:.0f}mm{st}"))
            frames += [frames[-1]] * 6
            if cls is PlannerPolicy and s == goal_seeds[0]:
                logs["planner"] = res["log"]
            if cls is ResolvedRatePolicy and s == goal_seeds[0]:
                logs["rr"] = res["log"]
        env.close()
        cols.append(frames)
    L = max(len(c) for c in cols)
    cols = [c + [c[-1]] * (L - len(c)) for c in cols]
    frames = [np.concatenate([c[k] for c in cols], axis=1) for k in range(L)]
    imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((900, 225), Image.LANCZOS) for x in frames[::2]]
    pal = [g.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)
    Image.fromarray(frames[0]).save(MEDIA / f"{TAG}_key_start.png")
    Image.fromarray(frames[min(30, L - 1)]).save(MEDIA / f"{TAG}_key_mid.png")

    # 關節曲線（規劃 vs 解析率，同一目標）
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for c, (k, nm) in enumerate((("planner", "Planner (joint)"), ("rr", "Resolved-rate (ee)"))):
        lg = logs[k]
        t = lg["t"]
        ax[0, c].plot(t, lg["q"])
        ax[0, c].set_title(f"{nm}: joint angles q [rad]")
        ax2 = ax[0, c].twinx()
        ax2.plot(t, 1e3 * lg["d"], "k--", lw=1.2)
        ax2.set_ylabel("distance d [mm] (dashed)")
        ax[1, c].plot(t, lg["tau"])
        ax[1, c].set_title("joint torque τ [N·m] (last substep)")
        ax[1, c].set_xlabel("t [s]")
    ax[0, 0].legend([f"q{i + 1}" for i in range(6)], fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
    for c, k in enumerate(("planner", "rr")):
        ax[c].plot(logs[k]["t"], logs[k]["qd"])
        ax[c].set_title(f"{k}: joint velocity q̇ [rad/s]")
        ax[c].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_velocity.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if "--anim" in sys.argv:
        M = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
    else:
        M = run_all()
    plots(M)
    anim(M)
    print(json.dumps({k: M[k] for k in ("baselines", "static", "throughput", "runtime_s",
                                         "planner_vs_rr_time_ratio", "planner_faster_frac")}, indent=1))
    print([{k: v[k] for k in ("joint", "peak1", "theory1", "disc_peak1", "peak2", "theory2", "disc_peak2",
                              "disc_maxdev", "t_peak1", "tau_max")}
           for v in M["vstep"]])
    print("box reachable", M["goals"]["box_reachable_frac"])
