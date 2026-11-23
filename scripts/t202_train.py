"""T202 訓練：PPO / SAC × 動作空間 ee / joint，四組各用 1 核平行訓練（wall-time 預算到時存檔停止）。

PYTHONPATH=. python3 scripts/t202_train.py                 # 從頭訓練（預設每組 1800 s）
PYTHONPATH=. python3 scripts/t202_train.py --resume        # 由 models/<name>_last.zip 接續
PYTHONPATH=. python3 scripts/t202_train.py --budget 600 sac_ee   # 只訓練指定組態
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

CONFIGS = {
    "sac_ee": dict(algo="sac", action_mode="ee", total_timesteps=400_000, eval_freq=10_000),
    "sac_joint": dict(algo="sac", action_mode="joint", total_timesteps=400_000, eval_freq=10_000),
    "ppo_ee": dict(algo="ppo", action_mode="ee", total_timesteps=3_000_000, eval_freq=40_000),
    "ppo_joint": dict(algo="ppo", action_mode="joint", total_timesteps=3_000_000, eval_freq=40_000),
}


def run(args):
    name, budget, resume = args
    from sixaxis.rltrain import train

    log = train(name=name, wall_budget_s=budget, resume=resume, seed=0, **CONFIGS[name])
    return name, log["timesteps"], log["wall_s"], log["best"]


if __name__ == "__main__":
    argv = sys.argv[1:]
    resume = "--resume" in argv
    budget = float(argv[argv.index("--budget") + 1]) if "--budget" in argv else 1800.0
    names = [a for a in argv if a in CONFIGS] or list(CONFIGS)
    with Pool(len(names)) as pool:
        for name, steps, wall, best in pool.imap_unordered(run, [(n, budget, resume) for n in names]):
            print(f"== {name}: {steps} steps, {wall:.0f} s, best succ {best['success_rate']:.2f} "
                  f"ret {best['return_mean']:.2f}", flush=True)
