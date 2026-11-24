"""T203：pick 環境（夾爪抓取＋抬升）、腳本基準、訓練整合與已訓練模型回歸。"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from sixaxis.gripper import T_TCP, f_min
from sixaxis.rlpick import TABLE_TOP, PickEnv, RandomPickPolicy, ScriptedPickPolicy, run_pick_episode


@pytest.fixture(scope="module")
def env():
    e = PickEnv()
    yield e
    e.close()


def test_check_env_and_make():
    from gymnasium.utils.env_checker import check_env

    e = gym.make("SixAxisPick-v0")
    check_env(e.unwrapped, skip_render_check=True)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (30,) and e.action_space.shape == (4,)
    e.close()


def test_obs_matches_fk(env):
    obs, info = env.reset(seed=3)
    q, _ = env.pbr.get_state()
    T = env.robot.fk(q) @ T_TCP
    assert np.allclose(obs[:3], T[:3, 3], atol=1e-6)
    assert np.allclose(obs[6:9], [*info["obj_xy"], env.z_obj0], atol=1e-6)
    assert np.allclose(obs[9:12], obs[6:9] - obs[:3], atol=1e-6)


def test_deterministic(env):
    a = np.array([0.3, -0.2, -0.5, 1.0])
    out = []
    for _ in range(2):
        env.reset(seed=11)
        for _ in range(10):
            obs, r, *_ = env.step(a)
        out.append((obs.copy(), r))
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]


def test_zero_action_holds_and_orientation(env):
    """零速度指令：TCP 不動、物件靜置；q0 擾動造成的工具傾角依姿態保持律衰減
    θ_n ≈ θ_0(1 − k_R Δt)^n（離散一階系統）。"""
    obs0, _ = env.reset(seed=5)

    def tilt():
        return float(np.arccos(np.clip(-env.tcp()[2][2, 2], -1, 1)))

    th0 = tilt()
    for _ in range(40):
        obs, _, term, trunc, info = env.step(np.array([0, 0, 0, -1.0]))
    assert not term
    assert np.linalg.norm(obs[:3] - obs0[:3]) < 2e-3
    assert abs(info["h"]) < 1e-3
    th_theory = th0 * (1 - env.k_rot * env.dt_c) ** 40
    assert th0 > np.radians(2) and tilt() < np.radians(0.3)
    assert abs(tilt() - th_theory) < 0.3 * th_theory


def test_tcp_velocity_tracking(env):
    env.reset(seed=2)
    for _ in range(20):
        _, _, _, _, info = env.step(np.array([0.0, 0.5, 0.0, -1.0]))
    v = env.tcp()[1]
    assert abs(v[1] - 0.15) < 5e-3 and abs(v[0]) < 5e-3 and abs(v[2]) < 5e-3


def test_scripted_pick_succeeds(env):
    pol = ScriptedPickPolicy(env)
    for s in range(6):
        r = run_pick_episode(env, pol, s)
        assert r["success"] and not r["collision"] and not r["drop"]
        assert r["h_final"] >= env.h_lift and r["regrasp"] == 0


def test_grasp_force_and_no_slip(env):
    """閉合後兩指法向力 ≈ F_grip（T005 力上限模型），抬升中物件相對 TCP 不滑動。"""
    pol = ScriptedPickPolicy(env)
    r = run_pick_episode(env, pol, 7, record=True)
    lg = r["log"]
    g = lg["grasped"].astype(bool)
    assert g.sum() > 20
    fc = lg["fc"][g][-10:]
    assert np.all(np.abs(fc - env.f_grip) < 0.1 * env.f_grip)
    rel = (lg["obj"] - lg["tcp"])[g]
    assert np.max(np.linalg.norm(rel - rel[0], axis=1)) < 3e-3
    assert env.f_grip > 10 * f_min(env.part.mass, 0.8)


def test_weak_grip_slips():
    """夾持力低於 f_min = mg/(2μ) 時物件無法抬起（摩擦極限）。"""
    e = PickEnv(f_grip=0.5 * f_min(0.2, 0.8))
    r = run_pick_episode(e, ScriptedPickPolicy(e), 0)
    e.close()
    assert not r["success"] and r["h_final"] < 0.02


def test_drop_terminates(env):
    obs, _ = env.reset(seed=0, options={"obj_xy": [0.85, 0.0]})  # 台面外 → 掉到地面
    for k in range(30):
        _, r, term, _, info = env.step(np.array([0, 0, 0, -1.0]))
        if term:
            break
    assert term and info["drop"] and r < -5
    assert info["obj"][2] < TABLE_TOP


def test_random_policy_fails(env):
    res = [run_pick_episode(env, RandomPickPolicy(env, s), s) for s in range(3)]
    assert not any(r["success"] for r in res)


def test_short_pick_train(tmp_path):
    pytest.importorskip("stable_baselines3")
    from sixaxis.rltrain import train

    log = train("ppo", "pick", task="pick", total_timesteps=128, eval_freq=64, eval_seeds=[5000],
                models_dir=tmp_path, hparams={"n_steps": 64, "batch_size": 32, "n_epochs": 1})
    assert log["task"] == "pick" and log["timesteps"] >= 128
    assert "grasp_rate" in log["curve"][0] and (tmp_path / "ppo_pick_best.zip").exists()


MODELS = Path(__file__).resolve().parents[1] / "models"
TRAINED = sorted(MODELS.glob("*_pick_*_best.zip"))


@pytest.mark.skipif(not TRAINED, reason="尚未訓練")
def test_trained_pick_model():
    """已存最佳 pick 模型（評估集最高者，見 models/T203_best.txt）在訓練/評估外的物件位置上確定性執行。"""
    from sixaxis.rltrain import evaluate, load

    sel = MODELS / "T203_best.txt"
    name = sel.read_text().split()[0] if sel.exists() else TRAINED[0].name.replace("_best.zip", "")
    m = load(MODELS / f"{name}_best.zip")
    e = PickEnv()
    out = evaluate(m, e, range(1000, 1020))
    e.close()
    assert out["success_rate"] >= 0.8 and out["collision_rate"] == 0.0
