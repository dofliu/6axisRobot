"""T203 訓練：pick 環境（TCP 速度＋夾爪），SAC / PPO × 種子 0、1，四組各用 1 核平行（wall-time 預算到時存檔停止）。

PYTHONPATH=. python3 scripts/t203_train.py                  # 從頭訓練（預設每組 2100 s）
PYTHONPATH=. python3 scripts/t203_train.py --resume         # 由 models/<name>_last.zip 接續
PYTHONPATH=. python3 scripts/t203_train.py --budget 600 sac_pick_s0
"""

from __future__ import annotations

import sys
from multiprocessing import Pool

CONFIGS = {
    "sac_pick_s0": dict(algo="sac", seed=0, total_timesteps=500_000, eval_freq=10_000),
    "sac_pick_s1": dict(algo="sac", seed=1, total_timesteps=500_000, eval_freq=10_000),
    "ppo_pick_s0": dict(algo="ppo", seed=0, total_timesteps=5_000_000, eval_freq=50_000),
    "ppo_pick_s1": dict(algo="ppo", seed=1, total_timesteps=5_000_000, eval_freq=50_000),
}


def run(args):
    name, budget, resume = args
    from sixaxis.rltrain import train

    log = train(name=name, task="pick", action_mode="pick", wall_budget_s=budget, resume=resume, **CONFIGS[name])
    return name, log["timesteps"], log["wall_s"], log["best"]


if __name__ == "__main__":
    argv = sys.argv[1:]
    resume = "--resume" in argv
    budget = float(argv[argv.index("--budget") + 1]) if "--budget" in argv else 2100.0
    names = [a for a in argv if a in CONFIGS] or list(CONFIGS)
    with Pool(len(names)) as pool:
        for name, steps, wall, best in pool.imap_unordered(run, [(n, budget, resume) for n in names]):
            print(f"== {name}: {steps} steps, {wall:.0f} s, best succ {best['success_rate']:.2f} "
                  f"ret {best['return_mean']:.2f}", flush=True)
