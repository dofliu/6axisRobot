"""T204 訓練：place／完整 pick-and-place 環境，四組各用 1 核平行（wall-time 預算到時存檔停止）。

組態（消融：示範預填回放緩衝的效果）：
  sac_place_demo_s0／s1：SAC＋100 個加噪腳本示範（成功回合）預填回放緩衝，種子 0、1
  sac_place_s0：SAC 無示範（同 T203 做法）
  ppo_place_s0：PPO 無示範
  sacbc_place_s0／s1：SAC＋示範專屬緩衝＋Q-filter 行為複製損失（sixaxis/sacbc.py），種子 0、1
  （實際執行：前四組先啟動；sacbc 兩組於約 13 分鐘後加入、預算 1800 s，期間 6 行程共用 4 核）

PYTHONPATH=. python3 scripts/t204_train.py                  # 從頭訓練（預設每組 2100 s）
PYTHONPATH=. python3 scripts/t204_train.py --resume         # 由 models/<name>_last.zip 接續
PYTHONPATH=. python3 scripts/t204_train.py --budget 600 sac_place_demo_s0
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

CONFIGS = {
    "sac_place_demo_s0": dict(algo="sac", seed=0, total_timesteps=600_000, eval_freq=10_000, demo_episodes=100),
    "sac_place_demo_s1": dict(algo="sac", seed=1, total_timesteps=600_000, eval_freq=10_000, demo_episodes=100),
    "sac_place_s0": dict(algo="sac", seed=0, total_timesteps=600_000, eval_freq=10_000),
    "ppo_place_s0": dict(algo="ppo", seed=0, total_timesteps=5_000_000, eval_freq=50_000),
    "sacbc_place_s0": dict(algo="sacbc", seed=0, total_timesteps=600_000, eval_freq=10_000, demo_episodes=100),
    "sacbc_place_s1": dict(algo="sacbc", seed=1, total_timesteps=600_000, eval_freq=10_000, demo_episodes=100),
}


def run(args):
    name, budget, resume = args
    from sixaxis.rltrain import train

    log = train(name=name, task="place", action_mode="place", wall_budget_s=budget, resume=resume,
                **CONFIGS[name])
    return name, log["timesteps"], log["wall_s"], log["best"]


if __name__ == "__main__":
    argv = sys.argv[1:]
    resume = "--resume" in argv
    budget = float(argv[argv.index("--budget") + 1]) if "--budget" in argv else 2100.0
    names = [a for a in argv if a in CONFIGS] or list(CONFIGS)[:4]
    with Pool(len(names)) as pool:
        for name, steps, wall, best in pool.imap_unordered(run, [(n, budget, resume) for n in names]):
            print(f"== {name}: {steps} steps, {wall:.0f} s, best succ {best['success_rate']:.2f} "
                  f"ret {best['return_mean']:.2f}", flush=True)
