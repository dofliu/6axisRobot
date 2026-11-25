"""T204：place／完整 pick-and-place 環境、分階段獎勵、腳本基準、示範收集、訓練整合與已訓練模型回歸。"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from sixaxis.rlplace import PlaceEnv, RandomPlacePolicy, ScriptedPlacePolicy, collect_demos, run_place_episode


@pytest.fixture(scope="module")
def env():
    e = PlaceEnv()
    yield e
    e.close()


def test_check_env_and_make():
    from gymnasium.utils.env_checker import check_env

    e = gym.make("SixAxisPlace-v0")
    check_env(e.unwrapped, skip_render_check=True)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (36,) and e.action_space.shape == (4,)
    e.close()


def test_goal_sampling_and_obs(env):
    for s in range(20):
        obs, info = env.reset(seed=s)
        g = info["goal"]
        assert np.linalg.norm(g[:2] - info["obj_xy"]) >= env.d_min
        assert np.all(g[:2] >= env.goal_low) and np.all(g[:2] <= env.goal_high)
        assert np.allclose(obs[30:33], g, atol=1e-6)
        assert np.allclose(obs[33:36], obs[6:9] - g, atol=1e-6)
    obs, info = env.reset(seed=0, options={"obj_xy": [0.4, 0.1], "goal_xy": [0.55, -0.1]})
    assert np.allclose(info["goal"][:2], [0.55, -0.1]) and np.allclose(obs[6:8], [0.4, 0.1], atol=1e-6)


def test_deterministic(env):
    a = np.array([0.3, -0.2, -0.5, 1.0])
    out = []
    for _ in range(2):
        env.reset(seed=11)
        for _ in range(10):
            obs, r, *_ = env.step(a)
        out.append((obs.copy(), r))
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]


def test_scripted_place_succeeds(env):
    pol = ScriptedPlacePolicy(env)
    for s in range(6):
        r = run_place_episode(env, pol, s)
        assert r["success"] and not r["collision"] and not r["drop"] and not r["tipped"]
        assert r["e_xy_final"] < 5e-3 and r["tilt_final"] < np.radians(1)
        assert r["t_grasp"] < r["t_lift"] < r["t_place"] <= r["t_success"]


def test_staged_reward_monotone(env):
    """腳本執行中各階段獎勵落在設計區間：抓住前 ≤ 1、搬運 [1.5, 4]、成功 = R_s；成功後保持到回合結束。"""
    r = run_place_episode(env, ScriptedPlacePolicy(env), 3, record=True)
    lg = r["log"]
    g = lg["grasped"].astype(bool)
    rew = lg["r"] + env.c_action * np.sum(lg["a"][:, :3] ** 2, axis=1)  # 去掉動作懲罰
    k_s = int(round(r["t_success"] / env.dt_c)) - 1
    assert np.all(rew[: np.argmax(g)] <= 1.0 + 1e-9)
    assert np.all((rew[g] >= 1.5 - 1e-9) & (rew[g] <= 4.0 + 1e-9))
    assert np.allclose(rew[k_s:], env.r_success)
    assert rew[g].max() > 3.9  # 搬到目標時接近上界
    assert r["hold_frac"] > 0.2


def test_holding_at_goal_is_not_success(env):
    """放到目標但不鬆爪：at_goal 成立、success 不成立（獎勵 ≤ 4 < 鬆爪後 4.5／6）。"""
    pol = ScriptedPlacePolicy(env, n_open=10**6)
    pol.reset()
    obs, _ = env.reset(seed=4)
    info = {}
    for _ in range(env.max_steps):
        a = pol(obs)
        if pol.phase == "open":
            a[3] = 1.0  # 維持閉合
        obs, r, term, trunc, info = env.step(a)
    assert info["at_goal"] and info["grasped"] and not info["success"] and r <= 4.0


def test_release_off_goal_fails(env):
    """在起點直接放回（未搬運）不算成功；目標誤差 ≈ 起訖距離。"""
    pol = ScriptedPlacePolicy(env)
    pol.reset()
    obs, info = env.reset(seed=8)
    goal = env.goal.copy()
    env.goal = np.array([*info["obj_xy"], env.z_obj0])  # 讓腳本把物件放回原地
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(pol(obs))
        env.goal = np.array([*env.obj_xy0, env.z_obj0]) if pol.phase != "retreat" else goal
    env.goal = goal
    assert not info["success"] and info["e_xy"] >= env.d_min - 0.01


def test_random_policy_fails(env):
    res = [run_place_episode(env, RandomPlacePolicy(env, s), s) for s in range(3)]
    assert not any(r["success"] for r in res)


def test_collect_demos(env):
    trans, n_ok, n_try = collect_demos(env, 2, noise=0.15)
    assert n_ok == 2 and n_try >= 2 and len(trans) == 2 * env.max_steps
    o, a, r, o2, term, trunc = trans[-1]
    assert o.shape == (36,) and a.shape == (4,) and trunc and not term and r > 5.0


def test_short_place_train_with_demos(tmp_path):
    pytest.importorskip("stable_baselines3")
    from sixaxis.rltrain import train

    log = train("sac", "place", task="place", total_timesteps=64, eval_freq=64, eval_seeds=[5000],
                models_dir=tmp_path, demo_episodes=1, hparams={"batch_size": 32})
    assert log["task"] == "place" and log["demo"]["transitions"] == 160
    assert log["hparams"]["learning_starts"] == 0
    assert "place_rate" in log["curve"][0] and (tmp_path / "sac_place_best.zip").exists()


MODELS = Path(__file__).resolve().parents[1] / "models"
SEL = MODELS / "T204_best.txt"


@pytest.mark.skipif(not SEL.exists(), reason="尚未訓練")
def test_trained_place_model():
    """已存最佳 place 模型（models/T204_best.txt）在訓練/評估外的位置上確定性執行。"""
    from sixaxis.rltrain import evaluate, load

    name = SEL.read_text().split()[0]
    m = load(MODELS / f"{name}_best.zip")
    e = PlaceEnv()
    out = evaluate(m, e, range(1000, 1020))
    e.close()
    assert out["success_rate"] >= float(SEL.read_text().split()[1]) and out["collision_rate"] == 0.0
