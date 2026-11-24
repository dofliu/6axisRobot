"""T202 強化學習訓練：以 Stable-Baselines3 的 PPO / SAC 訓練 T201 reach 環境，記錄學習曲線並存 checkpoint。

演算法（皆為 SB3 2.x 實作，本模組只負責組態、評估、時間預算、存檔與續訓）：
* PPO（Schulman et al. 2017, "Proximal Policy Optimization Algorithms"）：on-policy、截斷比率目標
  L = E[min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t)]，優勢以 GAE(γ, λ)（Schulman et al. 2016）估計。
* SAC（Haarnoja et al. 2018, "Soft Actor-Critic ..."；自動溫度 2018b）：off-policy 最大熵
  J = Σ E[r_t + α H(π(·|s_t))]，雙 Q 取最小值、目標網路 Polyak 平均 τ、溫度 α 自動調整到目標熵 −dim(A)。

學習曲線：每 eval_freq 個環境步以「固定」評估目標集（eval_seeds，與 T201/T202 最終測試集不重疊）
做確定性（取均值動作）評估，記錄成功率、平均回報、末距離、成功時間；以（成功率, 平均回報）字典序選最佳模型存
models/<name>_best.zip，並在結束時存 models/<name>_last.zip（含續訓所需的網路與最佳化器狀態，
SAC 的回放緩衝不存：檔案會 > 50MB，續訓時重新收集）。
時間預算：wall_budget_s 到時由回呼中止 learn()（CLAUDE.md：單次執行訓練 ≲ 40 分鐘）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .rlenv import ReachEnv, run_episode

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
EVAL_SEEDS = list(range(5000, 5050))  # 學習曲線用固定評估集（T201 測試集為 1000–1199）

# 預設超參數（SB3 / rl-baselines3-zoo 對 PyBullet 機械手臂 reach 類任務常用值，γ 依 100 步視界取 0.98）
HPARAMS = {
    "ppo": dict(learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10, gamma=0.98, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0)),
    "sac": dict(learning_rate=3e-4, buffer_size=300_000, batch_size=256, gamma=0.98, tau=0.005,
                learning_starts=2_000, train_freq=1, gradient_steps=1, ent_coef="auto",
                policy_kwargs=dict(net_arch=[256, 256])),
}


def _algo(algo: str):
    from stable_baselines3 import PPO, SAC

    return {"ppo": PPO, "sac": SAC}[algo]


class SB3Policy:
    """把 SB3 模型包成 T201 `run_episode` 的策略介面（確定性：取高斯均值 / tanh 均值）。"""

    def __init__(self, model, deterministic: bool = True):
        self.model, self.deterministic = model, deterministic

    def reset(self):
        pass

    def __call__(self, obs):
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return a


def evaluate(model, env, seeds) -> dict:
    """在固定種子集上確定性評估，回傳彙總指標（reach：T201 指標；pick：另加抓取/抬升/保持指標）。"""
    pol = SB3Policy(model)
    if isinstance(env, ReachEnv):
        res = [run_episode(env, pol, s) for s in seeds]
    else:
        from .rlpick import run_pick_episode

        res = [run_pick_episode(env, pol, s) for s in seeds]
        ts = [r["t_success"] for r in res if r["t_success"] is not None]
        return {
            "success_rate": float(np.mean([r["success"] for r in res])),
            "collision_rate": float(np.mean([r["collision"] for r in res])),
            "drop_rate": float(np.mean([r["drop"] for r in res])),
            "grasp_rate": float(np.mean([r["t_grasp"] is not None for r in res])),
            "return_mean": float(np.mean([r["return"] for r in res])),
            "hold_frac_mean": float(np.mean([r["hold_frac"] for r in res])),
            "h_final_mean_mm": 1e3 * float(np.mean([r["h_final"] for r in res])),
            "d_final_mean_mm": float("nan"),
            "t_success_mean": float(np.mean(ts)) if ts else None,
            "tau_ratio_max": float(np.max([r["tau_ratio"] for r in res])),
        }
    ts = [r["t_success"] for r in res if r["t_success"] is not None]
    return {
        "success_rate": float(np.mean([r["success"] for r in res])),
        "collision_rate": float(np.mean([r["collision"] for r in res])),
        "return_mean": float(np.mean([r["return"] for r in res])),
        "d_final_mean_mm": 1e3 * float(np.mean([r["d_final"] for r in res])),
        "t_success_mean": float(np.mean(ts)) if ts else None,
        "tau_ratio_max": float(np.max([r["tau_ratio"] for r in res])),
    }


def _make_callback(eval_env, eval_seeds, eval_freq, budget_s, name, models_dir, log, t_start):
    from stable_baselines3.common.callbacks import BaseCallback

    class EvalBudget(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0
            self.best = (-1.0, -np.inf)

        def _eval(self):
            t0 = time.perf_counter()
            m = evaluate(self.model, eval_env, eval_seeds)
            eps = list(self.model.ep_info_buffer)
            m.update({
                "timesteps": int(self.model.num_timesteps),
                "wall_s": time.perf_counter() - t_start,
                "eval_s": time.perf_counter() - t0,
                "train_success": float(np.mean([e.get("success", 0.0) for e in eps])) if eps else None,
                "train_return": float(np.mean([e["r"] for e in eps])) if eps else None,
            })
            log["curve"].append(m)
            key = (m["success_rate"], m["return_mean"])
            if key > self.best:
                self.best = key
                self.model.save(models_dir / f"{name}_best")
                log["best"] = dict(m)
            print(f"[{name}] t={m['timesteps']:>8d} wall={m['wall_s']:7.1f}s succ={m['success_rate']:.2f} "
                  f"ret={m['return_mean']:8.2f} d={m['d_final_mean_mm']:6.1f}mm "
                  f"h={m.get('h_final_mean_mm', 0):6.1f}mm",
                  flush=True)

        def _on_step(self) -> bool:
            if self.model.num_timesteps >= self.next_eval:
                self._eval()
                self.next_eval += eval_freq
            return time.perf_counter() - t_start < budget_s

        def _on_training_end(self):
            if not log["curve"] or log["curve"][-1]["timesteps"] != self.model.num_timesteps:
                self._eval()

    return EvalBudget()


def train(
    algo: str = "sac",
    action_mode: str = "ee",
    total_timesteps: int = 300_000,
    wall_budget_s: float = 1200.0,
    eval_freq: int = 10_000,
    eval_seeds=EVAL_SEEDS,
    seed: int = 0,
    name: str | None = None,
    models_dir: Path = MODELS,
    resume: bool = False,
    hparams: dict | None = None,
    env_kwargs: dict | None = None,
    torch_threads: int = 1,
    task: str = "reach",
) -> dict:
    """訓練一個組態並回傳紀錄（學習曲線、最佳評估、耗時、超參數）。

    task="pick" 時用 T203 `PickEnv`（action_mode 只作名稱用，動作固定為 TCP 速度＋夾爪）。
    resume=True 時讀 models/<name>_last.zip 接續（num_timesteps 累加，學習曲線附加到既有 <name>_log.json）。
    """
    import torch
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(torch_threads)
    name = name or f"{algo}_{action_mode}"
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(env_kwargs or {})
    hp = dict(HPARAMS[algo])
    hp.update(hparams or {})

    if task == "pick":
        from .rlpick import PickEnv

        def make_env():
            return PickEnv(**env_kwargs)
    else:
        def make_env():
            return ReachEnv(action_mode=action_mode, **env_kwargs)
    env = Monitor(make_env(), info_keywords=("success",))
    env.reset(seed=seed)
    eval_env = make_env()
    log_path = models_dir / f"{name}_log.json"
    cls = _algo(algo)
    last = models_dir / f"{name}_last.zip"
    if resume and last.exists():
        model = cls.load(last, env=env)
        log = json.loads(log_path.read_text()) if log_path.exists() else {"curve": []}
        wall0 = log["curve"][-1]["wall_s"] if log["curve"] else 0.0
    else:
        model = cls("MlpPolicy", env, seed=seed, verbose=0, **hp)
        log, wall0 = {"curve": []}, 0.0
    log.update({"task": task, "algo": algo, "action_mode": action_mode, "seed": seed,
                "hparams": _jsonable(hp), "env_kwargs": env_kwargs,
                "eval_seeds": [int(s) for s in eval_seeds]})
    t_start = time.perf_counter() - wall0
    budget = wall0 + wall_budget_s
    cb = _make_callback(eval_env, eval_seeds, eval_freq, budget, name, models_dir, log, t_start)
    if resume and log["curve"]:
        cb.best = (log["best"]["success_rate"], log["best"]["return_mean"])
        cb.next_eval = log["curve"][-1]["timesteps"] + eval_freq
    remaining = max(total_timesteps - model.num_timesteps, 1)
    model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=not resume)
    model.save(last)
    log["timesteps"] = int(model.num_timesteps)
    log["wall_s"] = time.perf_counter() - t_start
    log["completed"] = bool(model.num_timesteps >= total_timesteps)
    log_path.write_text(json.dumps(log, indent=1))
    env.close()
    eval_env.close()
    return log


def load(path, algo: str | None = None):
    """讀取存檔模型（不需環境即可 predict）。algo 未給時由檔名前綴判斷。"""
    path = Path(path)
    algo = algo or path.name.split("_")[0]
    return _algo(algo).load(path, device="cpu")


def _jsonable(d):
    out = {}
    for k, v in d.items():
        plain = isinstance(v, (int, float, str, list, type(None)))
        out[k] = _jsonable(v) if isinstance(v, dict) else (v if plain else str(v))
    return out
