"""T201 reach 環境測試。"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sixaxis.rlenv import PlannerPolicy, RandomPolicy, ReachEnv, ResolvedRatePolicy, run_episode


@pytest.fixture(scope="module")
def env():
    e = ReachEnv()
    yield e
    e.close()


@pytest.mark.parametrize("mode", ["joint", "ee"])
def test_check_env(mode):
    e = ReachEnv(action_mode=mode)
    check_env(e, skip_render_check=True)
    assert e.action_space.shape == ((6,) if mode == "joint" else (3,))
    e.close()


def test_registered():
    e = gym.make("SixAxisReach-v0", action_mode="ee")
    obs, _ = e.reset(seed=1)
    assert obs.shape == (27,)
    e.close()


def test_reset_obs_and_goal(env):
    obs, info = env.reset(seed=3)
    p, _ = env.ee()
    assert np.allclose(obs[12:15], p, atol=1e-6)
    assert np.allclose(obs[15:18], env.goal)
    assert np.allclose(obs[18:21], env.goal - p, atol=1e-6)
    assert np.all(env.goal >= env.goal_low) and np.all(env.goal <= env.goal_high)
    assert info["distance"] == pytest.approx(np.linalg.norm(env.goal - p))
    # 法蘭位置與自寫 FK 一致
    q, _ = env.pbr.get_state()
    assert np.allclose(env.robot.fk(q)[:3, 3], p, atol=1e-6)


def test_deterministic(env):
    def roll():
        env.reset(seed=7)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(10):
            o, r, *_ = env.step(rng.uniform(-1, 1, 6))
            out.append((o, r))
        return out

    a, b = roll(), roll()
    for (o1, r1), (o2, r2) in zip(a, b, strict=True):
        assert np.array_equal(o1, o2) and r1 == r2


def test_zero_action_holds(env):
    """零動作：計算力矩內環保持靜止（重力補償），漂移 < 0.5 mm。"""
    env.reset(seed=5)
    p0, _ = env.ee()
    for _ in range(20):
        _, _, term, _, info = env.step(np.zeros(6))
        assert not term
    assert np.linalg.norm(info["ee_pos"] - p0) < 5e-4


def test_velocity_command_tracking(env):
    """固定關節速度指令 1 s：q 位移 ≈ u·t（速度前饋 → 斜坡無穩態誤差）。"""
    env.reset(seed=2, options={"q0": env.q_home, "goal": [0.4, 0.3, 0.3]})
    a = np.array([0.5, 0, 0, 0, 0, -0.5])
    q0, _ = env.pbr.get_state()
    for _ in range(20):
        env.step(a)
    q, qd = env.pbr.get_state()
    assert np.allclose((q - q0)[[0, 5]], [0.5, -0.5], atol=0.02)
    assert np.allclose(qd[[0, 5]], [0.5, -0.5], atol=0.02)


def test_success_terminates_and_reward(env):
    p_goal = env.robot.fk(env.q_home)[:3, 3]
    env.reset(seed=0, options={"q0": env.q_home, "goal": p_goal})
    _, r, term, trunc, info = env.step(np.zeros(6))
    assert info["success"] and term and not trunc
    assert r == pytest.approx(-info["distance"] + env.r_success, abs=1e-9)


def test_truncation(env):
    e = ReachEnv(max_steps=5)
    e.reset(seed=1)
    for k in range(5):
        _, _, term, trunc, _ = e.step(np.zeros(6))
    assert trunc and not term
    e.close()


def test_floor_collision_terminates():
    e = ReachEnv(action_mode="ee", max_steps=200)
    e.reset(seed=0, options={"q0": e.q_home, "goal": [0.4, 0.0, 0.1]})
    hit = False
    for _ in range(200):
        _, r, term, trunc, info = e.step(np.array([0, 0, -1.0]))
        if term:
            hit = info["collision"]
            break
    assert hit and r < -e.r_collision + 1e-9
    e.close()


@pytest.mark.parametrize("policy", [PlannerPolicy, ResolvedRatePolicy])
def test_baselines_succeed(env, policy):
    pol = policy(env)
    res = [run_episode(env, pol, s) for s in range(100, 110)]
    assert all(r["success"] and not r["collision"] for r in res)
    assert max(r["tau_ratio"] for r in res) < 1.0


def test_random_policy_fails(env):
    res = [run_episode(env, RandomPolicy(env, 1), s) for s in range(3)]
    assert not any(r["success"] for r in res)
