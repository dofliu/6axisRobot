"""T202：SB3 PPO/SAC 訓練流程（短跑）、存檔/續訓、已訓練模型回歸。"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from sixaxis.rlenv import ReachEnv, run_episode  # noqa: E402
from sixaxis.rltrain import MODELS, SB3Policy, evaluate, load, train  # noqa: E402


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_short_train_save_resume(tmp_path, algo):
    hp = {"n_steps": 64, "batch_size": 32, "n_epochs": 1} if algo == "ppo" else {"learning_starts": 50,
                                                                                    "buffer_size": 1000}
    log = train(algo, "ee", total_timesteps=128, eval_freq=64, eval_seeds=[5000, 5001], models_dir=tmp_path,
                hparams=hp)
    assert log["timesteps"] >= 128 and log["completed"]
    assert len(log["curve"]) >= 2 and log["curve"][0]["timesteps"] <= 1
    assert (tmp_path / f"{algo}_ee_best.zip").exists() and (tmp_path / f"{algo}_ee_last.zip").exists()
    saved = json.loads((tmp_path / f"{algo}_ee_log.json").read_text())
    assert saved["algo"] == algo and saved["hparams"]["gamma"] == 0.98
    # 續訓：步數累加、學習曲線附加
    log2 = train(algo, "ee", total_timesteps=256, eval_freq=64, eval_seeds=[5000], models_dir=tmp_path,
                 hparams=hp, resume=True)
    assert log2["timesteps"] >= 256
    assert len(log2["curve"]) > len(log["curve"])
    ts = [c["timesteps"] for c in log2["curve"]]
    assert ts == sorted(ts)


def test_wall_budget_stops_training(tmp_path):
    log = train("ppo", "ee", total_timesteps=10**7, wall_budget_s=3.0, eval_freq=10**6, eval_seeds=[5000],
                models_dir=tmp_path, hparams={"n_steps": 64, "batch_size": 32, "n_epochs": 1})
    assert not log["completed"] and log["timesteps"] < 10**6


def test_policy_adapter_deterministic(tmp_path):
    train("sac", "ee", total_timesteps=60, eval_freq=10**6, eval_seeds=[5000], models_dir=tmp_path,
          hparams={"learning_starts": 30, "buffer_size": 500})
    m = load(tmp_path / "sac_ee_last.zip")
    env = ReachEnv(action_mode="ee")
    r1 = run_episode(env, SB3Policy(m), 7)
    r2 = run_episode(env, SB3Policy(m), 7)
    env.close()
    assert r1["return"] == r2["return"] and r1["steps"] == r2["steps"]


BEST = sorted(Path(MODELS).glob("*_best.zip"))


@pytest.mark.skipif(not (Path(MODELS) / "sac_ee_best.zip").exists(), reason="尚未訓練")
def test_trained_sac_ee_reaches():
    """已存的 SAC（ee 動作）最佳模型：在與訓練/評估不同的目標上確定性執行應幾乎全成功且無碰撞。"""
    m = load(Path(MODELS) / "sac_ee_best.zip")
    env = ReachEnv(action_mode="ee")
    out = evaluate(m, env, range(1000, 1020))
    env.close()
    assert out["success_rate"] >= 0.9 and out["collision_rate"] == 0.0
    assert out["tau_ratio_max"] <= 1.0


@pytest.mark.parametrize("path", BEST, ids=[p.stem for p in BEST])
def test_saved_models_load_and_small(path):
    assert path.stat().st_size < 50e6
    m = load(path)
    if "_pick" in path.stem:  # T203 pick 模型
        from sixaxis.rlpick import PickEnv

        env = PickEnv()
    else:
        env = ReachEnv(action_mode="ee" if "_ee" in path.stem else "joint")
    obs, _ = env.reset(seed=1)
    a = SB3Policy(m)(obs)
    env.close()
    assert a.shape == env.action_space.shape and np.all(np.abs(a) <= 1.0)
