"""T205 報告資料：隨機姿態＋domain randomization 取放。

內容：腳本基準（含/不含方位對齊）與隨機策略（200 組 DR 測試）、標稱環境腳本基準、
夾持幾何驗證（固定方位誤差 Δ 不轉夾爪抓取 vs 理論 w_eff(Δ) = w|cosΔ| + L|sinΔ| ≤ 80 mm 與 Δ_max）、
學習曲線、各組最佳模型於 DR／標稱測試集交叉評估（DR 效果）、依方位/長度/質量/摩擦分組成功率、推論時間、圖與動畫。

先訓練：PYTHONPATH=. python3 scripts/t205_train.py
PYTHONPATH=. python3 scripts/t205_report.py --interim        # 階段報告（不寫 models/T205_best.txt）
PYTHONPATH=. python3 scripts/t205_report.py --interim --anim # 只重做圖/動畫（讀既有 JSON）
PYTHONPATH=. python3 scripts/t205_report.py                  # 最終報告（寫 models/T205_best.txt）
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

from sixaxis.gripper import PAD_L  # noqa: E402
from sixaxis.rlplace import run_place_episode  # noqa: E402
from sixaxis.rlrand import (  # noqa: E402
    OPENING,
    RandomRandPlacePolicy,
    RandPlaceEnv,
    ScriptedRandPlacePolicy,
    clear_misalignment,
    effective_width,
    finger_axis_yaw,
    max_misalignment,
    pads_hit_box,
    short_axis_yaw,
    strip_extent,
    wrap_half_pi,
)
from sixaxis.rltrain import EVAL_SEEDS, MODELS, SB3Policy, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-27_T205"  # 階段報告
TAG_FINAL = "2026-11-27_T205"  # 最終報告日期於收斂後更新
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
SEEDS = list(range(1000, 1200))
RL = ["sac_rand_demo_s0", "sac_rand_demo_s1", "sacbc_rand_s0", "sac_rand_nodr_demo_s0"]
BASE = ["scripted", "noturn", "random"]
LABEL = {"sac_rand_demo_s0": "SAC+demo DR s0", "sac_rand_demo_s1": "SAC+demo DR s1", "sacbc_rand_s0": "SAC+BC DR s0",
         "sac_rand_nodr_demo_s0": "SAC+demo noDR s0", "scripted": "Scripted FSM (yaw align)",
         "noturn": "Scripted, no yaw align", "random": "Random"}
COL = {"sac_rand_demo_s0": "#c0392b", "sac_rand_demo_s1": "#e8a33d", "sacbc_rand_s0": "#1b7f3b",
       "sac_rand_nodr_demo_s0": "#8e44ad", "scripted": "#333333", "noturn": "#1f6fb4", "random": "#999999"}
GEOM_L = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
GEOM_W = 0.040
GEOM_DEG = tuple(range(0, 91, 5))


def make(name, dr=True, **kw):
    env = RandPlaceEnv(dr=dr, **kw)
    if name == "scripted":
        return env, ScriptedRandPlacePolicy(env)
    if name == "noturn":
        return env, ScriptedRandPlacePolicy(env, k_psi=0.0, tol_yaw=10.0)
    if name == "random":
        return env, RandomRandPlacePolicy(env)
    return env, SB3Policy(load(MODELS / f"{name}_best.zip"))


KEYS = ("seed", "success", "collision", "drop", "tipped", "t_grasp", "t_lift", "t_place", "t_success", "steps",
        "return", "e_xy_final", "e_z_final", "tilt_final", "hold_frac", "tau_ratio", "obj_xy", "goal_xy", "dist")


def keep(r, env):
    k = {key: r[key] for key in KEYS}
    k.update({"params": dict(env.params), "yaw_rel0": float(env.yaw_rel0)})
    return k


def _m(v):
    return float(np.mean(v)) if len(v) else None


def summarize(res):
    ts = [r["t_success"] for r in res if r["t_success"] is not None]
    ex = [r["e_xy_final"] for r in res if r["success"]]
    return {
        "n": len(res), "success_rate": _m([r["success"] for r in res]),
        "ever_success": _m([r["t_success"] is not None for r in res]),
        "grasp_rate": _m([r["t_grasp"] is not None for r in res]),
        "lift_rate": _m([r["t_lift"] is not None for r in res]),
        "place_rate": _m([r["t_place"] is not None for r in res]),
        "collision_rate": _m([r["collision"] for r in res]),
        "drop_rate": _m([r["drop"] for r in res]),
        "tipped_rate": _m([r["tipped"] for r in res]),
        "t_success_mean": _m(ts), "t_success_p95": float(np.percentile(ts, 95)) if ts else None,
        "t_grasp_mean": _m([r["t_grasp"] for r in res if r["t_grasp"] is not None]),
        "e_xy_succ_mean_mm": 1e3 * _m(ex) if ex else None,
        "e_xy_succ_p95_mm": 1e3 * float(np.percentile(ex, 95)) if ex else None,
        "hold_frac_mean": _m([r["hold_frac"] for r in res]),
        "return_mean": _m([r["return"] for r in res]),
        "tau_ratio_max": float(np.max([r["tau_ratio"] for r in res])),
    }


BINS = {
    "yaw": ("|Δ₀| [deg]", lambda r: abs(np.degrees(r["yaw_rel0"])), [0, 15, 30, 45, 60, 75, 90.01]),
    "L": ("L [mm]", lambda r: 1e3 * r["params"]["L"], [35, 60, 70, 80, 90, 100.01]),
    "mass": ("m [kg]", lambda r: r["params"]["mass"], [0.1, 0.2, 0.3, 0.4001]),
    "mu": ("μ_obj", lambda r: r["params"]["mu"], [0.5, 0.625, 0.75, 0.875, 1.0001]),
}


def breakdown(res):
    out = {}
    for key, (_, f, b) in BINS.items():
        vals = [f(r) for r in res]
        out[key] = {"bins": b, "count": [int(sum(lo <= v < hi for v in vals)) for lo, hi in zip(b, b[1:])],
                    "success": [_m([r["success"] for r, v in zip(res, vals) if lo <= v < hi])
                                for lo, hi in zip(b, b[1:])]}
    return out


def geom_trial(env, L, rel_deg):
    """不轉夾爪的腳本抓取：L × 40 × 50 mm 長方塊，短邊軸與指軸相對 yaw = rel。
    回傳：閉合前（APPROACH/DESCEND）指墊是否碰到物件與最大接觸力、閉合前物件被推轉的角度、
    是否抬起（抓住且抬升 ≥ 5 cm）、閉合後的相對 yaw（量自我對正）。"""
    params = {"L": L, "w": GEOM_W, "h": 0.050, "mass": 0.2, "mu": 0.8, "f_grip": 40.0, "wn": 25.0}
    opts = {"q0": env.q_home, "obj_xy": [0.45, 0.1], "goal_xy": [0.45, -0.1], "params": params}
    env.reset(seed=0, options={**opts, "obj_yaw": 0.0})
    rel = np.radians(rel_deg) - (1e-3 if rel_deg == 90 else 0.0)
    yaw = rel - wrap_half_pi(short_axis_yaw(env.last["qo"]) - finger_axis_yaw(env.last["R"]))
    pol = ScriptedRandPlacePolicy(env, k_psi=0.0, tol_yaw=10.0)
    obs, _ = env.reset(seed=0, options={**opts, "obj_yaw": float(yaw)})
    pol.reset()
    rel0 = env.yaw_rel0
    lifted, rel_grasp, fc_max, top = False, None, 0.0, False
    f_pre, yaw_obj0, rel_pre = 0.0, short_axis_yaw(env.last["qo"]), None
    z_top = env.z_obj0 + 0.025
    pose = None  # 指墊下緣降到物件頂面（＋2 mm）時的實際位姿 → 以此位姿預測是否碰撞
    for _ in range(90):
        act = pol(obs)
        pre = pol.phase in ("approach", "descend")  # 本步指令仍為閉合前階段
        m0 = env.last
        if pre and pose is None and (m0["p"][2] - PAD_L / 2 <= z_top + 0.002):
            pose = {"tcp": m0["p"][:2].tolist(), "finger_yaw": finger_axis_yaw(m0["R"]),
                    "obj": m0["po"][:2].tolist(), "short_yaw": short_axis_yaw(m0["qo"])}
        obs, _, term, trunc, info = env.step(act)
        fc_max = max(fc_max, min(info["fc"]))
        if pre:
            f_pre = max(f_pre, max(info["fc"]))
            rel_pre = env.yaw_rel(env.last)
        if pol.phase == "lift" and rel_grasp is None:
            rel_grasp = env.yaw_rel(env.last)
        if info["lifted"]:
            lifted = True
            break
        if term or trunc:
            break
    top = bool(env.last["po"][2] < env.z_obj0 - 0.004)  # 指墊壓在物件上 → 物件被壓/推倒
    return {"L": L, "rel_deg": rel_deg, "rel0_deg": float(np.degrees(rel0)), "lifted": lifted,
            "rel_after_close_deg": None if rel_grasp is None else float(np.degrees(rel_grasp)),
            "w_eff_mm": 1e3 * effective_width(GEOM_W, L, rel), "strip_mm": 1e3 * strip_extent(GEOM_W, L, rel),
            "f_pre_close_max": f_pre, "touched_before_close": bool(f_pre > 0.5),
            "pred_hit_measured_pose": None if pose is None else pads_hit_box(
                pose["tcp"], pose["finger_yaw"], pose["obj"], pose["short_yaw"], L, GEOM_W),
            "pose_at_top": None if pose is None else {
                "dyaw_deg": float(np.degrees(wrap_half_pi(pose["short_yaw"] - pose["finger_yaw"]))),
                "off_mm": (1e3 * (np.array(pose["tcp"]) - pose["obj"])).tolist()},
            "rel_before_close_deg": None if rel_pre is None else float(np.degrees(rel_pre)),
            "obj_turn_before_close_deg": float(np.degrees(wrap_half_pi(short_axis_yaw(env.last["qo"]) - yaw_obj0)))
            if rel_grasp is None else None,
            "fc_min_max": fc_max, "pushed_down": top}


def job(args):
    name, kind = args
    import torch

    torch.set_num_threads(1)
    if kind in ("test", "test_nodr", "evalset", "evalset_nodr"):
        env, pol = make(name, dr=not kind.endswith("nodr"))
        res = []
        for s in (SEEDS if kind.startswith("test") else EVAL_SEEDS):
            res.append(keep(run_place_episode(env, pol, s), env))
        env.close()
        return name, kind, res
    if kind.startswith("geom"):
        L = float(kind.split(":")[1])
        env = RandPlaceEnv(dr=False, obs_noise=False)
        out = [geom_trial(env, L, d) for d in GEOM_DEG]
        env.close()
        return name, kind, out
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


def collect(final: bool):
    t0 = time.time()
    rl = [n for n in RL if (MODELS / f"{n}_best.zip").exists()]
    jobs = [(n, "test") for n in rl + BASE] + [(n, "test_nodr") for n in rl + ["scripted", "noturn"]]
    jobs += [("scripted", "evalset"), ("scripted", "evalset_nodr")]
    jobs += [("geom", f"geom:{L}") for L in GEOM_L]
    jobs += [(n, "timing") for n in rl[:1] + ["scripted"]]
    with Pool(4) as pool:
        out = pool.map(job, jobs)
    M = {"test": {}, "test_nodr": {}, "per_episode": {}, "evalset": {}, "evalset_nodr": {}, "timing": {},
         "train": {}, "geom": [], "by": {}}
    for name, kind, res in out:
        if kind.startswith("geom"):
            M["geom"] += res
        elif kind == "timing":
            M["timing"][name] = res
        else:
            M[kind][name] = summarize(res)
            if kind == "test":
                M["per_episode"][name] = res
                M["by"][name] = breakdown(res)
    M["theory"] = {f"{L:.2f}": {"delta_max_deg": float(np.degrees(max_misalignment(GEOM_W, L))),
                                "delta_clear_deg": float(np.degrees(clear_misalignment(GEOM_W, L))),
                                "w_eff_90_mm": 1e3 * effective_width(GEOM_W, L, np.pi / 2)} for L in GEOM_L}
    M["opening_mm"] = 1e3 * OPENING
    # DR 參數實際分布（200 組測試）
    pe = M["per_episode"]["scripted"]
    M["dr_stats"] = {k: {"min": float(min(r["params"][k] for r in pe)), "max": float(max(r["params"][k] for r in pe)),
                         "mean": float(np.mean([r["params"][k] for r in pe]))}
                     for k in ("L", "w", "h", "mass", "mu", "f_grip", "wn")}
    M["dr_stats"]["frac_L_gt_opening"] = float(np.mean([r["params"]["L"] > OPENING for r in pe]))
    M["dr_stats"]["frac_yaw_gt_45"] = float(np.mean([abs(r["yaw_rel0"]) > np.pi / 4 for r in pe]))
    for n in rl:
        lg = json.loads((MODELS / f"{n}_log.json").read_text())
        cv = lg["curve"]
        first = {p: next((c for c in cv if c["success_rate"] >= p), None) for p in (0.5, 0.8)}
        M["train"][n] = {"timesteps": lg["timesteps"], "wall_s": lg["wall_s"], "best": lg["best"], "final": cv[-1],
                         "max_place_rate": max(c["place_rate"] for c in cv),
                         "max_grasp_rate": max(c["grasp_rate"] for c in cv),
                         "steps_to_50": first[0.5]["timesteps"] if first[0.5] else None,
                         "steps_to_80": first[0.8]["timesteps"] if first[0.8] else None,
                         "fps": lg["timesteps"] / lg["wall_s"], "demo": lg.get("demo"), "n_evals": len(cv),
                         "env_kwargs": lg.get("env_kwargs"), "hparams": lg["hparams"],
                         "model_kb": (MODELS / f"{n}_best.zip").stat().st_size / 1024}
    ts = {r["seed"]: r["t_success"] for r in pe}
    M["time_ratio_vs_scripted"] = {}
    for n in rl:
        rr = [r["t_success"] / ts[r["seed"]] for r in M["per_episode"][n]
              if r["t_success"] is not None and ts[r["seed"]] is not None]
        M["time_ratio_vs_scripted"][n] = {"mean": _m(rr), "faster_frac": _m(np.array(rr) < 1) if rr else None}
    dr_models = [n for n in rl if "nodr" not in n]
    M["best_model"] = max(dr_models, key=lambda n: (M["test"][n]["success_rate"], M["test"][n]["return_mean"]))
    if final:
        best = M["best_model"]
        thr = max(0.0, np.floor(10 * M["test"][best]["success_rate"] - 1.5) / 10)
        (MODELS / "T205_best.txt").write_text(f"{best} {thr:.1f}\n", encoding="utf-8")
    M["rl"] = rl
    M["runtime_s"] = time.time() - t0
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    return M


# ------------------------------------------------------------------ 圖
def plots(M):
    rl = M["rl"]
    logs = {n: json.loads((MODELS / f"{n}_log.json").read_text())["curve"] for n in rl}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for n in rl:
        cv = logs[n]
        t = np.array([x["timesteps"] for x in cv]) / 1e3
        ax[0].plot(t, 100 * np.array([x["success_rate"] for x in cv]), color=COL[n], lw=1.7, label=LABEL[n])
        ax[1].plot(t, 100 * np.array([x["place_rate"] for x in cv]), color=COL[n], lw=1.5, label=LABEL[n])
        ax[1].plot(t, 100 * np.array([x["grasp_rate"] for x in cv]), color=COL[n], lw=0.8, ls=":")
        ax[2].plot(t, [x["return_mean"] for x in cv], color=COL[n], lw=1.7, label=LABEL[n])
    ax[0].axhline(100 * M["evalset"]["scripted"]["success_rate"], color="k", ls="--", lw=1, label="Scripted (DR)")
    ax[2].axhline(M["evalset"]["scripted"]["return_mean"], color="k", ls="--", lw=1, label="Scripted (DR)")
    ax[0].set_title("eval success at episode end, 50 fixed tasks [%]\n(noDR model evaluated on nominal env)")
    ax[1].set_title("ever placed at goal (solid) / ever grasped (dotted) [%]")
    ax[2].set_title("eval mean return")
    for a in ax:
        a.set_xlabel("environment steps [×1000]")
        a.legend(fontsize=7)
    ax[0].set_ylim(-3, 103)
    ax[1].set_ylim(-3, 103)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_learning.png", dpi=110)
    plt.close(fig)

    # 夾持幾何
    G = M["geom"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    Ls = np.linspace(0.04, 0.10, 100)
    ax[0].plot(1e3 * Ls, [np.degrees(max_misalignment(GEOM_W, L)) for L in Ls], "r--",
               label="full-width jaw: w cosΔ + L sinΔ = 80 mm")
    ax[0].plot(1e3 * Ls, [np.degrees(clear_misalignment(GEOM_W, L)) for L in Ls], "r-",
               label="20 mm pads: strip extent = 80 mm (clear descent)")
    for g in G:
        ax[0].scatter(1e3 * g["L"] + (1.2 if g["touched_before_close"] else -1.2), g["rel_deg"],
                      marker="x" if g["touched_before_close"] else "o",
                      color="#1b7f3b" if g["lifted"] else "#c0392b", s=26)
    ax[0].scatter([], [], marker="o", color="k", label="pads cleared the box (left)")
    ax[0].scatter([], [], marker="x", color="k", label="pads hit box before closing (right)")
    ax[0].scatter([], [], marker="s", color="#1b7f3b", label="green: lifted / red: failed")
    ax[0].set_xlabel("box length L [mm] (w = 40 mm)")
    ax[0].set_ylabel("initial misalignment |Δ| [deg]")
    ax[0].set_title("parallel-jaw grasp feasibility vs misalignment")
    ax[0].legend(fontsize=7, loc="lower left")
    for L in GEOM_L:
        gg = [g for g in G if g["L"] == L and g["rel_after_close_deg"] is not None]
        ax[1].plot([g["rel_deg"] for g in gg], [abs(g["rel_after_close_deg"]) for g in gg], "o-",
                   label=f"L = {1e3 * L:.0f} mm")
    ax[1].plot([0, 90], [0, 90], "k:", lw=1, label="no rotation")
    ax[1].set_xlabel("misalignment before closing |Δ₀| [deg]")
    ax[1].set_ylabel("|Δ| after closing [deg]")
    ax[1].set_title("self-alignment of the box by the closing jaws")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_geometry.png", dpi=110)
    plt.close(fig)

    # 依分組成功率
    names = rl + ["scripted", "noturn"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for i, (key, (xl, _, b)) in enumerate(BINS.items()):
        xc = [(lo + hi) / 2 for lo, hi in zip(b, b[1:])]
        for n in names:
            v = M["by"][n][key]["success"]
            ax[i].plot(xc, [100 * x if x is not None else np.nan for x in v], "o-", color=COL[n], label=LABEL[n])
        ax[i].set_xlabel(xl + "  (n per bin: " + ",".join(map(str, M["by"]["scripted"][key]["count"])) + ")")
        ax[i].set_ylim(-3, 103)
    ax[0].set_ylabel("success [%] (200 DR test tasks)")
    ax[0].legend(fontsize=7)
    fig.suptitle("success rate grouped by initial yaw misalignment / box length / mass / friction", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_breakdown.png", dpi=110)
    plt.close(fig)

    # DR 交叉評估
    fig, ax = plt.subplots(figsize=(9, 4))
    names = rl + ["scripted", "noturn"]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [100 * M["test"][n]["success_rate"] for n in names], 0.4, label="DR test env (200)")
    ax.bar(x + 0.2, [100 * M["test_nodr"][n]["success_rate"] for n in names], 0.4, label="nominal env (200)")
    ax.set_xticks(x, [LABEL[n] for n in names], fontsize=7, rotation=15)
    ax.set_ylabel("success [%]")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("cross evaluation: trained with/without domain randomization")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_crosseval.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 動畫與曲線
def pick_seeds():
    """選 2 組長物件（L > 80 mm）且初始方位差大（|Δ₀| > 50°）的測試任務，展示方位對齊的必要性。"""
    env = RandPlaceEnv()
    out = []
    for s in SEEDS:
        env.reset(seed=s)
        if env.params["L"] > 0.082 and abs(env.yaw_rel0) > np.radians(50):
            out.append(s)
        if len(out) == 2:
            break
    env.close()
    return out


def anim(M):
    show = [M["best_model"], "scripted", "noturn"]
    seeds = pick_seeds()
    M["anim_seeds"] = seeds
    W, H = 400, 300
    logs, rows = {}, []
    for n in show:
        env, pol = make(n, render_mode="rgb_array", render_size=(W, H))
        logs[n] = record_episode(env, pol, seeds[0])
        frames_a, frames_b = [], []
        for s in seeds:
            obs, _ = env.reset(seed=s)
            pol.reset()
            for k in range(env.max_steps):
                obs, _, term, trunc, info = env.step(pol(obs))
                if k % 2 == 1 or term or trunc:
                    st = (" PLACED" if info["success"] else (" at goal" if info["at_goal"] else
                          (" grasped" if info["grasped"] else "")))
                    txt = (f"{LABEL[n]} | seed {s} | t={env.k * env.dt_c:.1f}s "
                           f"dyaw={np.degrees(info['yaw_rel']):+.0f}deg{st}")
                    f1 = env.pbr.render(target=(0.45, 0.0, 0.25), distance=1.4, yaw=60, pitch=-30, size=(W, H),
                                        supersample=1)
                    f2 = env.pbr.render(target=(0.475, 0.0, 0.1), distance=0.9, yaw=90, pitch=-80, size=(W, H),
                                        supersample=1)
                    frames_a.append(label(f1, txt))
                    p = env.params
                    frames_b.append(label(f2, f"top | box {1e3 * p['L']:.0f}x{1e3 * p['w']:.0f}x{1e3 * p['h']:.0f} mm "
                                              f"{p['mass']:.2f} kg mu {p['mu']:.2f}"))
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
    for tag, k in (("start", 0), ("align", 10), ("carry", 25), ("placed", 45)):
        Image.fromarray(grid[min(k, L - 1)]).save(MEDIA / f"{TAG}_key_{tag}.png")

    fig, ax = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    for c, n in enumerate(show):
        lg = logs[n]
        t = lg["t"]
        ax[0, c].plot(t, lg["tcp"][:, 2] * 1e3, label="TCP z")
        ax[0, c].plot(t, lg["obj"][:, 2] * 1e3, label="object z")
        ax[0, c].plot(t, 1e3 * lg["e_goal"], label="|p_obj − goal|")
        ax[0, c].axhline(20, color="r", ls=":", lw=1, label="tol_xy 20 mm")
        ax[0, c].set_title(f"{LABEL[n]} (seed {seeds[0]}) [mm]", fontsize=9)
        ax[0, c].legend(fontsize=7)
        ax[1, c].plot(t, np.degrees(lg["yaw_rel"]), label="Δ (box short axis − finger axis)")
        ax[1, c].plot(t, np.degrees(lg["psi"]), label="ψ_ref")
        ax[1, c].axhline(0, color="k", lw=0.5)
        ax[1, c].set_title("yaw [deg]")
        ax[1, c].legend(fontsize=7)
        ax[2, c].plot(t, lg["fc"][:, 0], label="left pad [N]")
        ax[2, c].plot(t, lg["fc"][:, 1], "--", label="right pad [N]")
        ax[2, c].step(t, 10 * lg["a"][:, 4], "k", lw=0.8, where="post", label="grip cmd ×10")
        ax[2, c].plot(t, 10 * lg["a"][:, 3], color="#8e44ad", lw=0.8, label="ψ̇ cmd ×10")
        ax[2, c].set_title("pad force / commands")
        ax[2, c].legend(fontsize=7)
        ax[3, c].plot(t, lg["tau"])
        ax[3, c].set_title("joint torque τ [N·m]")
        ax[3, c].set_xlabel("t [s]")
    ax[3, 0].legend([f"τ{i + 1}" for i in range(6)], fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_episode.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    for c, n in enumerate(show):
        ax[0, c].plot(logs[n]["t"], logs[n]["q"])
        ax[0, c].set_title(f"{LABEL[n]}: q [rad]", fontsize=9)
        ax[1, c].plot(logs[n]["t"], logs[n]["qd"])
        ax[1, c].set_title("q̇ [rad/s]")
        ax[1, c].set_xlabel("t [s]")
    ax[0, 0].legend([f"q{i + 1}" for i in range(6)], fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=100)
    plt.close(fig)


def record_episode(env, pol, seed):
    """一回合的時間序列（含相對 yaw Δ 與 ψ_ref）。"""
    obs, _ = env.reset(seed=seed)
    pol.reset()
    keys = ("t", "q", "qd", "tau", "tcp", "obj", "e_goal", "fc", "a", "yaw_rel", "psi")
    lg = {k: [] for k in keys}
    term = trunc = False
    while not (term or trunc):
        a = np.asarray(pol(obs), float)
        obs, _, term, trunc, info = env.step(a)
        q, qd = env.pbr.get_state()
        vals = (env.k * env.dt_c, q, qd, info["tau"], info["tcp"], info["obj"], info["e_goal"], info["fc"], a,
                info["yaw_rel"], info["psi_ref"])
        for k, v in zip(keys, vals, strict=True):
            lg[k].append(v)
    return {k: np.array(v) for k, v in lg.items()}


if __name__ == "__main__":
    final = "--interim" not in sys.argv
    if final:
        TAG = TAG_FINAL
    if "--anim" in sys.argv:
        M = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
        if "--geom" in sys.argv:  # 只重跑夾持幾何實驗
            with Pool(4) as pool:
                M["geom"] = sum((r for _, _, r in pool.map(job, [("geom", f"geom:{L}") for L in GEOM_L])), [])
    else:
        M = collect(final)
    plots(M)
    anim(M)
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: M[k] for k in ("test", "test_nodr", "theory", "dr_stats", "timing", "time_ratio_vs_scripted",
                                         "best_model", "runtime_s", "anim_seeds")}, indent=1))
    print(json.dumps({n: {k: v for k, v in M["train"][n].items() if k != "hparams"} for n in M["train"]}, indent=1))
    print(json.dumps(M["geom"], indent=0))
