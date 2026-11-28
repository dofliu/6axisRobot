"""T205 第二輪：DR／無 DR 各 3 種子（等訓練預算 ≈ 35 分鐘/組）的交叉評估與統計。

每個模型在 200 組 DR 測試任務（種子 1000–1199，DR 開）與 200 組標稱任務（同種子、DR 關：標稱 70×45×50 mm、無觀測雜訊）
各評估一次，回報：
* 單一模型：成功率與 Wilson 95 % 區間（sixaxis/stats.py）、到目標率、碰撞/掉落、放置誤差、首次成功時間；
* 種子間：平均 ± 標準差與 Student t 95 % 區間；DR vs 無 DR 的 Welch t 檢定；泛化落差（標稱 − DR）；
* 學習曲線：各組在共同步數網格上的平均 ± 標準差（評估集 50 組；DR 模型評估於 DR 環境、無 DR 模型於標稱環境）；
* 依初始方位差／長度／質量／摩擦分組成功率（DR 測試，種子平均）；
* 動畫：同一 DR 測試任務上 DR 最佳模型 vs 無 DR 最佳模型（兩視角），與該回合時間序列。

PYTHONPATH=. python3 scripts/t205_seeds.py           # 評估＋圖＋動畫
PYTHONPATH=. python3 scripts/t205_seeds.py --anim    # 讀既有 JSON，只重做圖與動畫
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import t205_report as R  # noqa: E402
from PIL import Image  # noqa: E402
from t201_report import label  # noqa: E402

from sixaxis.rltrain import MODELS  # noqa: E402
from sixaxis.stats import seed_summary, welch, wilson  # noqa: E402

TAG = "2026-11-28_T205"
GROUPS = {"DR": ["sac_rand_demo_s0", "sac_rand_demo_s1", "sac_rand_demo_s2"],
          "noDR": ["sac_rand_nodr_demo_s0", "sac_rand_nodr_demo_s1", "sac_rand_nodr_demo_s2"]}
ENVS = {"test": "DR test env", "test_nodr": "nominal env"}
GCOL = {"DR": "#c0392b", "noDR": "#8e44ad"}
LAB = {n: f"SAC+demo {g} s{n[-1]}" for g, ns in GROUPS.items() for n in ns}
LAB.update(R.LABEL)


def model_stats(res):
    s = R.summarize(res)
    k = int(sum(r["success"] for r in res))
    s["k"] = k
    s["wilson"] = wilson(k, len(res))
    return s


def collect():
    t0 = time.time()
    models = [n for ns in GROUPS.values() for n in ns]
    jobs = [(n, kind) for n in models + ["scripted"] for kind in ENVS]
    with Pool(4) as pool:
        out = pool.map(R.job, jobs)
    M = {"models": {}, "by": {}, "per_episode": {}, "groups": {}, "train": {}}
    for name, kind, res in out:
        M["models"].setdefault(name, {})[kind] = model_stats(res)
        M["per_episode"].setdefault(name, {})[kind] = [{k: r[k] for k in ("seed", "success", "t_success")} for r in res]
        if kind == "test":
            M["by"][name] = R.breakdown(res)
    for g, ns in GROUPS.items():
        G = {}
        for kind in ENVS:
            for key in ("success_rate", "ever_success", "place_rate", "grasp_rate", "collision_rate", "drop_rate",
                        "e_xy_succ_mean_mm", "t_success_mean", "tau_ratio_max"):
                G[f"{kind}:{key}"] = seed_summary([M["models"][n][kind][key] for n in ns])
        G["gap_nominal_minus_dr"] = seed_summary([M["models"][n]["test_nodr"]["success_rate"]
                                                  - M["models"][n]["test"]["success_rate"] for n in ns])
        V = {key: np.array([[np.nan if v is None else v for v in M["by"][n][key]["success"]] for n in ns])
             for key in R.BINS}
        G["by"] = {key: np.nanmean(v, axis=0).tolist() for key, v in V.items()}
        G["by_std"] = {key: np.nanstd(v, axis=0, ddof=1).tolist() for key, v in V.items()}
        M["groups"][g] = G
    M["welch"] = {kind: welch([M["models"][n][kind]["success_rate"] for n in GROUPS["DR"]],
                              [M["models"][n][kind]["success_rate"] for n in GROUPS["noDR"]]) for kind in ENVS}
    for n in models:
        lg = json.loads((MODELS / f"{n}_log.json").read_text())
        cv = lg["curve"]
        M["train"][n] = {"timesteps": lg["timesteps"], "wall_s": lg["wall_s"], "best": lg["best"],
                         "final": cv[-1], "fps": lg["timesteps"] / lg["wall_s"], "n_evals": len(cv),
                         "curve": [{k: c[k] for k in ("timesteps", "success_rate", "place_rate", "return_mean")}
                                   for c in cv]}
    M["bins"] = {k: v[2] for k, v in R.BINS.items()}
    M["bin_count"] = M["by"]["scripted"]
    M["runtime_s"] = time.time() - t0
    return M


def plots(M):
    # 交叉評估：種子平均 ± 標準差＋各種子點
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    xs = {"test": 0, "test_nodr": 1}
    for j, (g, ns) in enumerate(GROUPS.items()):
        for kind, x0 in xs.items():
            S = M["groups"][g][f"{kind}:success_rate"]
            x = x0 + (j - 0.5) * 0.36
            ax[0].bar(x, 100 * S["mean"], 0.34, color=GCOL[g], alpha=0.55,
                      label=f"trained {g}" if kind == "test" else None)
            ax[0].errorbar(x, 100 * S["mean"], yerr=100 * S["std"], color="k", capsize=4)
            for n in ns:
                st = M["models"][n][kind]
                lo, hi = st["wilson"]
                xx = x + 0.1 * (int(n[-1]) - 1)
                ax[0].errorbar(xx, 100 * st["success_rate"], yerr=[[100 * (st["success_rate"] - lo)],
                                                                   [100 * (hi - st["success_rate"])]],
                               fmt="o", ms=4, color=GCOL[g], ecolor="0.4", lw=0.8)
    for kind, x0 in xs.items():
        ax[0].hlines(100 * M["models"]["scripted"][kind]["success_rate"], x0 - 0.45, x0 + 0.45, colors="k",
                     linestyles="--", label="scripted FSM" if kind == "test" else None)
    ax[0].set_xticks([0, 1], ["evaluated: DR test env (200)", "evaluated: nominal env (200)"])
    ax[0].set_ylabel("success [%]")
    ax[0].set_ylim(0, 108)
    ax[0].set_title("cross evaluation, 3 seeds each (bar: mean ± std; dots: per seed with Wilson 95 % CI)",
                    fontsize=9)
    ax[0].legend(fontsize=8, loc="upper left")
    # 學習曲線（共同步數網格）
    for g, ns in GROUPS.items():
        tmax = min(M["train"][n]["curve"][-1]["timesteps"] for n in ns)
        grid = np.arange(10_000, tmax + 1, 10_000)
        Y = np.array([np.interp(grid, [c["timesteps"] for c in M["train"][n]["curve"]],
                                [c["success_rate"] for c in M["train"][n]["curve"]]) for n in ns])
        m, s = 100 * Y.mean(0), 100 * Y.std(0, ddof=1)
        ax[1].plot(grid / 1e3, m, color=GCOL[g], lw=2, label=f"trained {g} (mean of {len(ns)} seeds)")
        ax[1].fill_between(grid / 1e3, m - s, m + s, color=GCOL[g], alpha=0.18)
        for n in ns:
            cv = M["train"][n]["curve"]
            ax[1].plot([c["timesteps"] / 1e3 for c in cv], [100 * c["success_rate"] for c in cv], color=GCOL[g],
                       lw=0.6, alpha=0.5)
    ax[1].set_xlabel("environment steps [×1000]")
    ax[1].set_ylabel("eval success, 50 fixed tasks [%]")
    ax[1].set_title("learning curves (DR models on DR env, noDR models on nominal env); band = ±1 std", fontsize=9)
    ax[1].set_ylim(-3, 103)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R.MEDIA / f"{TAG}_seeds.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for i, (key, (xl, _, b)) in enumerate(R.BINS.items()):
        xc = np.array([(lo + hi) / 2 for lo, hi in zip(b, b[1:])])
        for g in GROUPS:
            m = 100 * np.array(M["groups"][g]["by"][key])
            s = 100 * np.array(M["groups"][g]["by_std"][key])
            ax[i].errorbar(xc, m, yerr=s, fmt="o-", color=GCOL[g], capsize=3, label=f"trained {g} (3 seeds)")
        ax[i].plot(xc, [100 * v for v in M["by"]["scripted"][key]["success"]], "ks--", ms=4, label="scripted")
        ax[i].set_xlabel(xl + "  (n per bin: " + ",".join(map(str, M["by"]["scripted"][key]["count"])) + ")")
        ax[i].set_ylim(-3, 108)
    ax[0].set_ylabel("success [%] on 200 DR test tasks")
    ax[0].legend(fontsize=7)
    fig.suptitle("DR test success grouped by initial yaw misalignment / box length / mass / friction (mean ± std)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(R.MEDIA / f"{TAG}_seeds_breakdown.png", dpi=110)
    plt.close(fig)


def best_of(M, g):
    return max(GROUPS[g], key=lambda n: (M["models"][n]["test"]["success_rate"], M["models"][n]["test"]["return_mean"]))


def anim(M):
    """DR 最佳 vs 無 DR 最佳，在「DR 模型成功、無 DR 模型失敗」的 DR 測試任務上（若無則任選兩者皆成功者）。"""
    a, b = best_of(M, "DR"), best_of(M, "noDR")
    ea = {r["seed"]: r["success"] for r in M["per_episode"][a]["test"]}
    eb = {r["seed"]: r["success"] for r in M["per_episode"][b]["test"]}
    env = R.RandPlaceEnv()
    cands = []
    for s in R.SEEDS:
        if ea[s] and not eb[s]:
            env.reset(seed=s)
            cands.append((abs(env.yaw_rel0), s))
    env.close()
    seed = max(cands)[1] if cands else next(s for s in R.SEEDS if ea[s] and eb[s])
    M["anim"] = {"seed": seed, "models": [a, b], "dr_ok_nodr_fail_count": len(cands)}
    W, H = 400, 300
    rows, logs = [], {}
    for n in (a, b):
        env, pol = R.make(n, render_mode="rgb_array", render_size=(W, H))
        logs[n] = R.record_episode(env, pol, seed)
        obs, _ = env.reset(seed=seed)
        pol.reset()
        fa, fb = [], []
        for k in range(env.max_steps):
            obs, _, term, trunc, info = env.step(pol(obs))
            if k % 2 == 1 or term or trunc:
                st = (" PLACED" if info["success"] else (" at goal" if info["at_goal"] else
                      (" grasped" if info["grasped"] else "")))
                f1 = env.pbr.render(target=(0.45, 0.0, 0.25), distance=1.4, yaw=60, pitch=-30, size=(W, H),
                                    supersample=1)
                f2 = env.pbr.render(target=(0.475, 0.0, 0.1), distance=0.9, yaw=90, pitch=-80, size=(W, H),
                                    supersample=1)
                fa.append(label(f1, f"{LAB[n]} | DR task {seed} | t={env.k * env.dt_c:.1f}s "
                                    f"dyaw={np.degrees(info['yaw_rel']):+.0f}deg{st}"))
                p = env.params
                fb.append(label(f2, f"top | box {1e3 * p['L']:.0f}x{1e3 * p['w']:.0f}x{1e3 * p['h']:.0f} mm "
                                    f"{p['mass']:.2f} kg mu {p['mu']:.2f} F {p['f_grip']:.0f} N"))
            if term or trunc:
                break
        env.close()
        rows.append([np.concatenate([x, y], axis=1) for x, y in zip(fa, fb, strict=True)])
    L = max(len(r) for r in rows)
    rows = [r + [r[-1]] * (L - len(r) + 5) for r in rows]
    grid = [np.concatenate([r[k] for r in rows], axis=0) for k in range(L + 5)]
    imageio.mimsave(R.MEDIA / f"{TAG}.mp4", grid, fps=10, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((480, 360), Image.LANCZOS) for x in grid]
    pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(R.MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)
    for tag, k in (("start", 0), ("grasp", 20), ("end", L - 1)):
        Image.fromarray(grid[min(k, L - 1)]).save(R.MEDIA / f"{TAG}_key_{tag}.png", optimize=True)

    fig, ax = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    for c, n in enumerate((a, b)):
        lg = logs[n]
        t = lg["t"]
        ax[0, c].plot(t, lg["q"])
        ax[0, c].set_title(f"{LAB[n]} on DR task {seed}: q [rad]", fontsize=9)
        ax[1, c].plot(t, lg["qd"])
        ax[1, c].set_title("q̇ [rad/s]")
        ax[2, c].plot(t, lg["tau"])
        ax[2, c].set_title("τ [N·m]")
        ax[3, c].plot(t, 1e3 * lg["e_goal"], label="|p_obj − goal| [mm]")
        ax[3, c].plot(t, 1e3 * lg["obj"][:, 2], label="object z [mm]")
        ax[3, c].axhline(20, color="r", ls=":", lw=1, label="tol 20 mm")
        ax[3, c].set_title("placement error / object height")
        ax[3, c].legend(fontsize=7)
        ax[4, c].plot(t, np.degrees(lg["yaw_rel"]), label="Δ yaw [deg]")
        ax[4, c].plot(t, lg["fc"][:, 0], label="left pad [N]")
        ax[4, c].plot(t, lg["fc"][:, 1], "--", label="right pad [N]")
        ax[4, c].set_title("yaw misalignment / pad forces")
        ax[4, c].legend(fontsize=7)
        ax[4, c].set_xlabel("t [s]")
    ax[0, 0].legend([f"q{i + 1}" for i in range(6)], fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(R.MEDIA / f"{TAG}_episode.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    path = R.DATA / f"{TAG}_seeds.json"
    M = json.loads(path.read_text(encoding="utf-8")) if "--anim" in sys.argv else collect()
    plots(M)
    anim(M)
    path.write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, v in M["models"].items():
        print(n, {k: (round(v[k]["success_rate"], 3), [round(x, 3) for x in v[k]["wilson"]]) for k in ENVS})
    for g, G in M["groups"].items():
        print(g, {k: (round(G[k]["mean"], 3), G[k]["std"] and round(G[k]["std"], 3)) for k in G
                  if k.endswith("success_rate") or k.startswith("gap")})
    print("welch", M["welch"], "anim", M["anim"], "runtime", M["runtime_s"])
