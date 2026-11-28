"""T205 訓練：隨機姿態＋domain randomization 取放（SixAxisRandPlace-v0），四組各用 1 核平行（wall-time 預算到時存檔停止）。

組態（沿用 T204 最佳做法：SAC＋加噪腳本示範預填回放緩衝；並比較 SACBC 與「無 DR」）：
  sac_rand_demo_s0／s1：SAC＋100 個示範預填，DR 開（尺寸/質量/摩擦/夾持力/伺服頻寬＋觀測雜訊），種子 0、1
  sacbc_rand_s0：SAC＋示範專屬緩衝＋Q-filter BC（sixaxis/sacbc.py），DR 開
  sac_rand_nodr_demo_s0：SAC＋示範預填，DR 關（只隨機化位置與 yaw，標稱 70×45×50 mm 長方塊、無觀測雜訊）
  第二輪新增 sac_rand_demo_s2、sac_rand_nodr_demo_s1／s2（DR／無 DR 各湊滿 3 種子）

不指定名稱時只跑第一輪四組（ROUND1）；第二輪：
PYTHONPATH=. python3 scripts/t205_train.py sac_rand_demo_s2 sac_rand_nodr_demo_s1 sac_rand_nodr_demo_s2

PYTHONPATH=. python3 scripts/t205_train.py                  # 從頭訓練（預設每組 2100 s）
PYTHONPATH=. python3 scripts/t205_train.py --resume         # 由 models/<name>_last.zip 接續
PYTHONPATH=. python3 scripts/t205_train.py --budget 600 sac_rand_demo_s0
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

COMMON = dict(total_timesteps=800_000, eval_freq=10_000, demo_episodes=100)
CONFIGS = {
    "sac_rand_demo_s0": dict(algo="sac", seed=0, **COMMON),
    "sac_rand_demo_s1": dict(algo="sac", seed=1, **COMMON),
    "sacbc_rand_s0": dict(algo="sacbc", seed=0, **COMMON),
    "sac_rand_nodr_demo_s0": dict(algo="sac", seed=0, env_kwargs={"dr": False}, **COMMON),
    # 第二輪（2026-11-28）：補第 3 個 DR 種子與無 DR 種子 1、2，使 DR／無 DR 各 3 種子可做平均 ± 標準差
    "sac_rand_demo_s2": dict(algo="sac", seed=2, **COMMON),
    "sac_rand_nodr_demo_s1": dict(algo="sac", seed=1, env_kwargs={"dr": False}, **COMMON),
    "sac_rand_nodr_demo_s2": dict(algo="sac", seed=2, env_kwargs={"dr": False}, **COMMON),
}
ROUND1 = ["sac_rand_demo_s0", "sac_rand_demo_s1", "sacbc_rand_s0", "sac_rand_nodr_demo_s0"]


def run(args):
    name, budget, resume = args
    from sixaxis.rltrain import train

    log = train(name=name, task="randplace", action_mode="randplace", wall_budget_s=budget, resume=resume,
                **CONFIGS[name])
    return name, log["timesteps"], log["wall_s"], log["best"]


if __name__ == "__main__":
    argv = sys.argv[1:]
    resume = "--resume" in argv
    budget = float(argv[argv.index("--budget") + 1]) if "--budget" in argv else 2100.0
    names = [a for a in argv if a in CONFIGS] or ROUND1
    with Pool(len(names)) as pool:
        for name, steps, wall, best in pool.imap_unordered(run, [(n, budget, resume) for n in names]):
            print(f"== {name}: {steps} steps, {wall:.0f} s, best succ {best['success_rate']:.2f} "
                  f"ret {best['return_mean']:.2f}", flush=True)
