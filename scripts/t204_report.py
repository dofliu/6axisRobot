"""T204 報告資料：place／完整 pick-and-place 學習曲線、示範預填消融、最佳模型 vs 腳本基準（同 200 組物件/目標）、
長時保持、起點擾動、未見物件、鬆爪落下高度掃描（自由落體 v = √(v₀² + 2gh) 驗證）、推論時間、圖與動畫。

先訓練：PYTHONPATH=. python3 scripts/t204_train.py
PYTHONPATH=. python3 scripts/t204_report.py          # 評估＋圖＋動畫（4 行程）
PYTHONPATH=. python3 scripts/t204_report.py --anim   # 只重做圖/動畫（讀既有 JSON）
PYTHONPATH=. python3 scripts/t204_report.py --interim  # 階段報告：腳本/隨機基準、落下掃描、學習曲線、各組最佳模型評估集、動畫
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
from t201_report import label  # noqa: E402

from sixaxis.gripper import BLOCK  # noqa: E402
from sixaxis.rlplace import PlaceEnv, RandomPlacePolicy, ScriptedPlacePolicy, run_place_episode  # noqa: E402
from sixaxis.rltrain import EVAL_SEEDS, MODELS, SB3Policy, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-25_T204"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
SEEDS = list(range(1000, 1200))  # 200 組測試（物件起點＋目標），與學習曲線評估集 5000–5049 不重疊
RL = ["sacbc_place_s0", "sacbc_place_s1", "sac_place_demo_s0", "sac_place_demo_s1", "sac_place_s0", "ppo_place_s0"]
BASE = ["scripted", "random"]
LABEL = {"sacbc_place_s0": "SAC+BC s0", "sacbc_place_s1": "SAC+BC s1", "sac_place_demo_s0": "SAC+demo s0", "sac_place_demo_s1": "SAC+demo s1", "sac_place_s0": "SAC s0",
         "ppo_place_s0": "PPO s0", "scripted": "Scripted FSM", "random": "Random"}
COL = {"sacbc_place_s0": "#1b7f3b", "sacbc_place_s1": "#6cc07f", "sac_place_demo_s0": "#c0392b", "sac_place_demo_s1": "#e8a33d", "sac_place_s0": "#8e44ad",
       "ppo_place_s0": "#1f6fb4", "scripted": "#333333", "random": "#999999"}
VARIANTS = {"q0wide": dict(q0_noise=0.3), "block": dict(obj=BLOCK), "long": dict(max_steps=300)}
H_REL = (0.002, 0.01, 0.02, 0.04, 0.06, 0.08)
G = 9.81


def make(name, **kw):
    env = PlaceEnv(**kw)
    if name == "scripted":
        return env, ScriptedPlacePolicy(env)
    if name == "random":
        return env, RandomPlacePolicy(env)
    return env, SB3Policy(load(MODELS / f"{name}_best.zip"))


KEYS = ("seed", "success", "collision", "drop", "tipped", "t_grasp", "t_lift", "t_place", "t_success", "steps",
        "return", "e_xy_final", "e_z_final", "tilt_final", "hold_frac", "tau_ratio", "obj_xy", "goal_xy", "dist")


def keep(r):
    return {k: r[k] for k in KEYS}


def _m(v):
    return float(np.mean(v)) if len(v) else None


def summarize(res):
    ts = [r["t_success"] for r in res if r["t_success"] is not None]
    ex = [r["e_xy_final"] for r in res if r["success"]]
    return {
        "n": len(res), "success_rate": _m([r["success"] for r in res]),
        "ever_success": _m([r["t_success"] is not None for r in res]),
        "grasp_rate": _m([r["t_grasp"] is not None for r in res]),
        "place_rate": _m([r["t_place"] is not None for r in res]),
        "collision_rate": _m([r["collision"] for r in res]),
        "drop_rate": _m([r["drop"] for r in res]),
        "tipped_rate": _m([r["tipped"] for r in res]),
        "t_success_mean": _m(ts), "t_success_p95": float(np.percentile(ts, 95)) if ts else None,
        "t_grasp_mean": _m([r["t_grasp"] for r in res if r["t_grasp"] is not None]),
        "e_xy_succ_mean_mm": 1e3 * _m(ex) if ex else None,
        "e_xy_succ_p95_mm": 1e3 * float(np.percentile(ex, 95)) if ex else None,
        "e_xy_all_median_mm": 1e3 * float(np.median([r["e_xy_final"] for r in res])),
        "tilt_succ_max_deg": float(np.degrees(max(r["tilt_final"] for r in res if r["success"]))) if ex else None,
        "hold_frac_mean": _m([r["hold_frac"] for r in res]),
        "return_mean": _m([r["return"] for r in res]),
        "tau_ratio_max": float(np.max([r["tau_ratio"] for r in res])),
    }


def release_sweep(seeds=SEEDS[:10]):
    """腳本基準：放下時物件底面離台面 h_rel 再鬆爪。以 2 ms 子步記錄物件 z、v_z 與兩指接觸力，
    兩指皆失去接觸時刻 k₀ 起為自由落體：理論著地速度 v = √(v₀² + 2 g h₀)（h₀ = z(k₀) − z_rest），
    與量得的最大下降速度比較；並量最終放置誤差與傾角。"""
    out = []
    for h in H_REL:
        env = PlaceEnv()
        pol = ScriptedPlacePolicy(env, h_rel=h)
        buf = []

        def cb(e):
            p, _, v = e.obj_state()
            buf.append((p[2], v[2], sum(e.finger_contacts()), e.grip_cmd))

        env.substep_cb = cb
        rows = []
        for s in seeds:
            buf.clear()
            r = run_place_episode(env, pol, s)
            z, vz, fc, gc = (np.array(x) for x in zip(*buf, strict=True))
            i_open = int(np.argmax((gc == 0) & (np.arange(len(gc)) > np.argmax(gc > 0))))
            i0 = i_open + int(np.argmax(fc[i_open:] <= 0.0))
            h0 = float(z[i0] - env.z_obj0)
            j = i0 + int(np.argmax(z[i0:] <= env.z_obj0 + 1e-3)) + 1
            v_meas = float(-vz[i0:j + 1].min()) if j > i0 else 0.0
            v0 = float(-vz[i0])
            v_th = float(np.sqrt(max(v0, 0.0) ** 2 + 2 * G * max(h0, 0.0)))
            rows.append({"h0": h0, "v0": v0, "v_meas": v_meas, "v_theory": v_th,
                         "t_open_to_free_ms": 2.0 * (i0 - i_open), "success": r["success"],
                         "e_xy_mm": 1e3 * r["e_xy_final"], "tilt_deg": float(np.degrees(r["tilt_final"]))})
        env.close()
        agg = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k != "success"}
        agg.update({"h_rel": h, "success_rate": float(np.mean([x["success"] for x in rows])),
                    "e_xy_max_mm": float(max(x["e_xy_mm"] for x in rows)),
                    "v_err_max": float(max(abs(x["v_meas"] - x["v_theory"]) for x in rows if x["h0"] > 0.003)
                                       if any(x["h0"] > 0.003 for x in rows) else 0.0)})
        out.append(agg)
    return out


def job(args):
    name, kind = args
    import torch

    torch.set_num_threads(1)
    if kind in ("test", "evalset"):
        env, pol = make(name)
        res = [run_place_episode(env, pol, s) for s in (SEEDS if kind == "test" else EVAL_SEEDS)]
        env.close()
        return name, kind, [keep(r) for r in res]
    if kind in VARIANTS:
        env, pol = make(name, **VARIANTS[kind])
        n = 50 if kind == "long" else 100
        res = []
        for s in SEEDS[:n]:
            r = run_place_episode(env, pol, s)
            k = keep(r)
            if kind == "long":
                k["held_after_8s"] = bool(r["success"] and r["t_success"] is not None
                                          and r["hold_frac"] * env.k >= env.k - r["t_success"] / env.dt_c - 0.5)
            res.append(k)
        env.close()
        return name, kind, res
    if kind == "release":
        return name, kind, release_sweep()
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
        env.close()
        return name, kind, {"mean_ms": 1e3 * float(np.mean(ts[5:])), "p95_ms": 1e3 * float(np.percentile(ts[5:], 95)),
                            "env_step_ms": 1e3 * float(np.mean(tenv[5:]))}
    raise ValueError(kind)


def run_all():
    t0 = time.time()
    jobs = [("scripted", "release")] + [(n, "test") for n in RL + BASE] + [(n, "evalset") for n in RL + ["scripted"]]
    jobs += [(n, k) for k in VARIANTS for n in RL + ["scripted"]]
    jobs += [(n, "timing") for n in RL + ["scripted"]]
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
                M[kind][name]["held_after_8s"] = float(np.mean([r["held_after_8s"] for r in res]))
        elif kind == "release":
            M["release"] = res
        elif kind == "timing":
            M["timing"][name] = res
    ts = {r["seed"]: r["t_success"] for r in M["per_episode"]["scripted"]}
    M["time_ratio_vs_scripted"] = {}
    for n in RL:
        rr = [r["t_success"] / ts[r["seed"]] for r in M["per_episode"][n]
              if r["t_success"] is not None and ts[r["seed"]] is not None]
        M["time_ratio_vs_scripted"][n] = {"mean": _m(rr), "faster_frac": _m(np.array(rr) < 1) if rr else None}
    # 成功率 vs 起訖距離
    bins = [0.12, 0.2, 0.28, 0.36, 0.6]
    M["by_dist"] = {"bins": bins}
    for n in RL + ["scripted"]:
        pe = M["per_episode"][n]
        M["by_dist"][n] = [_m([r["success"] for r in pe if lo <= r["dist"] < hi]) for lo, hi in zip(bins, bins[1:])]
    M["by_dist"]["count"] = [sum(lo <= r["dist"] < hi for r in M["per_episode"]["scripted"])
                             for lo, hi in zip(bins, bins[1:])]
    for n in RL:
        lg = json.loads((MODELS / f"{n}_log.json").read_text())
        curve = lg["curve"]
        first = {p: next((c for c in curve if c["success_rate"] >= p), None) for p in (0.5, 0.9)}
        M["train"][n] = {
            "timesteps": lg["timesteps"], "wall_s": lg["wall_s"], "completed": lg["completed"], "best": lg["best"],
            "steps_to_50": first[0.5]["timesteps"] if first[0.5] else None,
            "steps_to_90": first[0.9]["timesteps"] if first[0.9] else None,
            "wall_to_50": first[0.5]["wall_s"] if first[0.5] else None,
            "final": curve[-1], "n_evals": len(curve), "hparams": lg["hparams"], "demo": lg.get("demo"),
            "model_kb": (MODELS / f"{n}_best.zip").stat().st_size / 1024, "fps": lg["timesteps"] / lg["wall_s"],
        }
    best = max(RL, key=lambda n: (M["train"][n]["best"]["success_rate"], M["train"][n]["best"]["return_mean"]))
    M["best_model"] = best
    thr = max(0.0, np.floor(10 * M["test"][best]["success_rate"] - 1.5) / 10)  # 回歸測試門檻（留餘裕）
    (MODELS / "T204_best.txt").write_text(f"{best} {thr:.1f}\n", encoding="utf-8")
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
    ax[0].axhline(100 * M["evalset"]["scripted"]["success_rate"], color="k", ls="--", lw=1, label="Scripted FSM")
    ax[2].axhline(M["evalset"]["scripted"]["return_mean"], color="k", ls="--", lw=1, label="Scripted FSM")
    ax[0].set_xscale("symlog", linthresh=50)
    ax[2].set_xscale("symlog", linthresh=50)
    ax[0].set_title("eval success (solid) / ever-grasped (dotted), 50 fixed tasks")
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
    kinds = [("test", "nominal"), ("q0wide", "q0 ±0.3 rad"), ("block", "40×60×80 block"), ("long", "15 s episode")]
    x = np.arange(len(kinds))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        v = [100 * M[k][n]["success_rate"] for k, _ in kinds]
        ax[0].bar(x + (i - (len(names) - 1) / 2) * w, v, w, color=COL[n], label=LABEL[n])
    ax[0].set_xticks(x, [lab for _, lab in kinds], fontsize=8)
    ax[0].set_ylim(0, 128)
    ax[0].set_ylabel("success at episode end [%]")
    ax[0].set_title("robustness / generalization")
    ax[0].legend(fontsize=7, ncol=3, loc="upper center")
    bd = M["by_dist"]
    xc = [(lo + hi) / 2 for lo, hi in zip(bd["bins"], bd["bins"][1:])]
    for n in names:
        ax[1].plot(xc, [100 * v if v is not None else np.nan for v in bd[n]], "o-", color=COL[n], label=LABEL[n])
    ax[1].set_xlabel("object → goal distance [m] (bin centers)")
    ax[1].set_ylabel("success [%]")
    ax[1].set_title("success vs transport distance (200 test tasks)")
    ax[1].legend(fontsize=7)
    rs = M["release"]
    h0 = [1e3 * r["h0"] for r in rs]
    ax[2].plot(h0, [r["v_meas"] for r in rs], "ko", label="measured peak fall speed")
    hh = np.linspace(0, max(h0) / 1e3 * 1.05, 50)
    ax[2].plot(1e3 * hh, np.sqrt(2 * G * hh), "r--", label="√(2 g h₀)")
    ax[2].plot(h0, [r["v_theory"] for r in rs], "r+", ms=10, label="√(v₀² + 2 g h₀)")
    ax[2].set_xlabel("free-fall height at contact loss h₀ [mm]")
    ax[2].set_ylabel("speed [m/s]")
    ax2 = ax[2].twinx()
    ax2.plot(h0, [r["e_xy_mm"] for r in rs], "s:", color="#1f6fb4", label="placement error")
    ax2.set_ylabel("mean final |e_xy| [mm]", color="#1f6fb4")
    ax[2].set_title("release-height sweep (scripted, 10 tasks each)")
    ax[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_compare.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, len(names), figsize=(2.9 * len(names), 3.6), sharey=True)
    for c, n in enumerate(names):
        for r in M["per_episode"][n]:
            (ox, oy), (gx, gy) = r["obj_xy"], r["goal_xy"]
            ok = r["success"]
            ax[c].plot([oy, gy], [ox, gx], color=COL[n] if ok else "red", lw=0.4 if ok else 0.9, alpha=0.6)
            ax[c].scatter(gy, gx, s=8, marker="o" if ok else "x", color=COL[n] if ok else "red")
        ax[c].set_title(f"{LABEL[n]}: {100 * M['test'][n]['success_rate']:.1f} %", fontsize=9)
        ax[c].set_xlabel("y [m]")
        ax[c].set_aspect("equal")
    ax[0].set_ylabel("x [m]")
    fig.suptitle("test tasks: line = object start → goal (dot); red × = failure", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_map.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 動畫與曲線
def anim(M):
    best = M["best_model"]
    cands = [n for n in RL if n != best]
    other = max(cands, key=lambda n: M["test"][n]["success_rate"]) if cands else "random"
    seeds = [1003, 1011]
    W, H = 400, 300
    logs, rows = {}, []
    for n in [best, other, "scripted"]:
        env, pol = make(n, render_mode="rgb_array", render_size=(W, H))
        frames_a, frames_b = [], []
        for s in seeds:
            res = run_place_episode(env, pol, s, record=True)
            if s == seeds[0]:
                logs[n] = res["log"]
            obs, _ = env.reset(seed=s)
            pol.reset()
            for k in range(env.max_steps):
                a = pol(obs)
                obs, _, term, trunc, info = env.step(a)
                if k % 2 == 1 or term or trunc:
                    st = (" PLACED" if info["success"] else (" at goal" if info["at_goal"] else
                          (" grasped" if info["grasped"] else "")))
                    txt = (f"{LABEL[n]} | seed {s} | t={env.k * env.dt_c:.1f}s "
                           f"e={1e3 * info['e_goal']:.0f}mm{st}")
                    f1 = env.pbr.render(target=(0.45, 0.0, 0.25), distance=1.4, yaw=60, pitch=-30, size=(W, H),
                                        supersample=1)
                    f2 = env.pbr.render(target=(0.475, 0.0, 0.1), distance=1.0, yaw=90, pitch=-75, size=(W, H),
                                        supersample=1)
                    frames_a.append(label(f1, txt))
                    frames_b.append(label(f2, "top view: green square = goal"))
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
    for tag, k in (("start", 0), ("carry", 20), ("placed", 45)):
        Image.fromarray(grid[min(k, L - 1)]).save(MEDIA / f"{TAG}_key_{tag}.png")

    show = [best, other, "scripted"]
    fig, ax = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    for c, n in enumerate(show):
        lg = logs[n]
        t = lg["t"]
        ax[0, c].plot(t, lg["tcp"][:, 2] * 1e3, label="TCP z")
        ax[0, c].plot(t, lg["obj"][:, 2] * 1e3, label="object z")
        ax[0, c].plot(t, 1e3 * lg["e_goal"], label="|p_obj − goal|")
        ax[0, c].axhline(20, color="r", ls=":", lw=1, label="tol_xy 20 mm")
        ax[0, c].set_title(f"{LABEL[n]}: heights / goal error [mm]")
        ax[0, c].legend(fontsize=7)
        ax[1, c].plot(t, lg["fc"][:, 0], label="left pad")
        ax[1, c].plot(t, lg["fc"][:, 1], "--", label="right pad")
        ax[1, c].plot(t, 1e3 * lg["s"], label="finger stroke [mm]")
        ax[1, c].plot(t, lg["r"], "k", lw=0.8, label="reward")
        ax[1, c].set_title("pad force [N] / stroke / reward")
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


def interim():
    """訓練未收斂時的階段報告資料（不寫 models/T204_best.txt）。"""
    t0 = time.time()
    jobs = [("scripted", "release"), ("scripted", "test"), ("random", "test"), ("scripted", "evalset")]
    jobs += [(n, "evalset") for n in RL if (MODELS / f"{n}_best.zip").exists()]
    jobs += [("scripted", "long"), ("scripted", "q0wide"), ("scripted", "block"), ("scripted", "timing")]
    with Pool(4) as pool:
        out = pool.map(job, jobs)
    M = {"test": {}, "per_episode": {}, "evalset": {}, "train": {}, "timing": {}}
    for name, kind, res in out:
        if kind == "release":
            M["release"] = res
        elif kind == "timing":
            M["timing"][name] = res
        elif kind == "test":
            M["test"][name] = summarize(res)
            M["per_episode"][name] = res
        else:
            M.setdefault(kind, {})[name] = summarize(res)
    for n in RL:
        lp = MODELS / f"{n}_log.json"
        if not lp.exists():
            continue
        lg = json.loads(lp.read_text())
        cv = lg["curve"]
        M["train"][n] = {"timesteps": lg["timesteps"], "wall_s": lg["wall_s"], "best": lg["best"], "final": cv[-1],
                         "max_place_rate": max(c["place_rate"] for c in cv),
                         "max_grasp_rate": max(c["grasp_rate"] for c in cv),
                         "fps": lg["timesteps"] / lg["wall_s"], "demo": lg.get("demo"),
                         "bc_stats": lg.get("bc_stats"), "hparams": lg["hparams"]}
    M["best_model"] = max(M["train"], key=lambda n: (M["train"][n]["best"]["success_rate"],
                                                     M["train"][n]["best"]["return_mean"]))
    M["runtime_s"] = time.time() - t0
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{TAG}_interim.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    return M


def interim_plots(M):
    names = [n for n in RL if n in M["train"]]
    logs = {n: json.loads((MODELS / f"{n}_log.json").read_text())["curve"] for n in names}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for n in names:
        cv = logs[n]
        t = np.array([x["timesteps"] for x in cv]) / 1e3
        ax[0].plot(t, 100 * np.array([x["success_rate"] for x in cv]), color=COL[n], lw=1.7, label=LABEL[n])
        ax[1].plot(t, 100 * np.array([x["place_rate"] for x in cv]), color=COL[n], lw=1.5, label=LABEL[n])
        ax[1].plot(t, 100 * np.array([x["grasp_rate"] for x in cv]), color=COL[n], lw=0.8, ls=":")
        ax[2].plot(t, [x["return_mean"] for x in cv], color=COL[n], lw=1.7, label=LABEL[n])
    ax[2].axhline(M["evalset"]["scripted"]["return_mean"], color="k", ls="--", lw=1, label="Scripted FSM")
    for a in ax:
        a.set_xlim(0, 120)  # SAC 系列 ≈ 9–10 萬步；PPO 38.6 萬步只顯示前段
        a.set_xlabel("environment steps [×1000]")
        a.legend(fontsize=7)
    ax[0].set_title("eval success at episode end (50 fixed tasks) [%]\n(PPO: 386k steps, axis clipped)")
    ax[1].set_title("ever placed at goal (solid) / ever grasped (dotted) [%]")
    ax[2].set_title("eval mean return")
    ax[0].set_ylim(-3, 103)
    ax[1].set_ylim(-3, 103)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_learning.png", dpi=110)
    plt.close(fig)

    rs = M["release"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    h0 = np.array([1e3 * r["h0"] for r in rs])
    ax[0].plot(h0, [r["v_meas"] for r in rs], "ko", label="measured (last 2 ms sample before touchdown)")
    hh = np.linspace(0, h0.max() / 1e3 * 1.05, 50)
    ax[0].plot(1e3 * hh, np.sqrt(2 * G * hh), "r--", label="√(2 g h₀)")
    ax[0].plot(h0, [r["v_theory"] for r in rs], "r+", ms=10, label="√(v₀² + 2 g h₀)")
    ax[0].set_xlabel("free-fall height at contact loss h₀ [mm]")
    ax[0].set_ylabel("fall speed [m/s]")
    ax[0].set_title("release-height sweep: touchdown speed (scripted, 10 tasks each)")
    ax[0].legend(fontsize=7)
    ax[1].plot([1e3 * r["h_rel"] for r in rs], [r["e_xy_mm"] for r in rs], "s-", label="mean")
    ax[1].plot([1e3 * r["h_rel"] for r in rs], [r["e_xy_max_mm"] for r in rs], "^--", label="max")
    ax[1].axhline(20, color="r", ls=":", label="tol_xy = 20 mm")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("commanded release height h_rel [mm]")
    ax[1].set_ylabel("final placement error |e_xy| [mm]")
    ax[1].set_title("placement error vs release height")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_release.png", dpi=110)
    plt.close(fig)

    pe = M["per_episode"]["scripted"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in pe:
        (ox, oy), (gx, gy) = r["obj_xy"], r["goal_xy"]
        ax[0].plot([oy, gy], [ox, gx], color="#333" if r["success"] else "red", lw=0.4, alpha=0.6)
        ax[0].scatter(gy, gx, s=6, color="#1b7f3b" if r["success"] else "red")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("y [m]")
    ax[0].set_ylabel("x [m]")
    ax[0].set_title(f"scripted FSM, 200 test tasks: {100 * M['test']['scripted']['success_rate']:.1f} % success")
    ax[1].scatter([r["dist"] for r in pe], [r["t_success"] or np.nan for r in pe], s=8, color="#333")
    ax[1].set_xlabel("object → goal distance [m]")
    ax[1].set_ylabel("time to success [s]")
    ax[1].set_title("scripted cycle time vs transport distance")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_map.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if "--interim" in sys.argv:
        if "--anim" in sys.argv:
            M = json.loads((DATA / f"{TAG}_interim.json").read_text(encoding="utf-8"))
        else:
            M = interim()
        interim_plots(M)
        RL[:] = [M["best_model"]]
        best = M["best_model"]
        M["test"].setdefault(best, {"success_rate": 0.0})
        anim(M)
        print(json.dumps({k: M[k] for k in ("test", "evalset", "release", "timing", "best_model", "runtime_s")},
                         indent=1))
        print(json.dumps({n: {k: v for k, v in M["train"][n].items() if k != "hparams"} for n in M["train"]},
                         indent=1))
        sys.exit()
    if "--anim" in sys.argv:
        M = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
    else:
        M = run_all()
    plots(M)
    anim(M)
    print(json.dumps({k: M[k] for k in ("test", "q0wide", "block", "long", "release", "timing", "by_dist",
                                         "time_ratio_vs_scripted", "best_model", "runtime_s")}, indent=1))
    print(json.dumps({n: {k: v for k, v in M["train"][n].items() if k != "hparams"} for n in RL}, indent=1))
