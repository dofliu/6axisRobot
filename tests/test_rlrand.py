"""T205：隨機姿態＋domain randomization 取放環境、夾爪方位控制、夾持幾何、腳本基準與訓練整合。"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as pb
import pytest

from sixaxis.rlplace import run_place_episode
from sixaxis.rlrand import (
    DR_RANGES,
    NOMINAL,
    OPENING,
    SIGMA_P,
    RandPlaceEnv,
    ScriptedRandPlacePolicy,
    clear_misalignment,
    effective_width,
    finger_axis_yaw,
    max_misalignment,
    short_axis_yaw,
    strip_extent,
    wrap_half_pi,
)


@pytest.fixture(scope="module")
def env():
    e = RandPlaceEnv()
    yield e
    e.close()


def test_check_env_and_make():
    from gymnasium.utils.env_checker import check_env

    import sixaxis.rlrand  # noqa: F401

    e = gym.make("SixAxisRandPlace-v0")
    check_env(e.unwrapped, skip_render_check=True)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (42,) and e.action_space.shape == (5,)
    e.close()


def test_geometry_formulas():
    assert OPENING == pytest.approx(0.080)
    for w, L in [(0.045, 0.07), (0.035, 0.1), (0.055, 0.09)]:
        dm = max_misalignment(w, L)
        assert effective_width(w, L, dm) == pytest.approx(OPENING, abs=1e-12)
        assert effective_width(w, L, 0.9 * dm) < OPENING < effective_width(w, L, 1.1 * dm)
    assert max_misalignment(0.04, 0.05) == pytest.approx(np.pi / 2)  # 對角線 64 mm < 80 mm：任何方位可抓
    assert max_misalignment(0.09, 0.1) == 0.0
    assert wrap_half_pi(np.pi / 2) == pytest.approx(-np.pi / 2)
    assert wrap_half_pi(3.0) == pytest.approx(3.0 - np.pi)


def test_strip_extent():
    """帶狀區長度：Δ = 0 → w、Δ = 90° → L、b → ∞ 退化為投影寬度、b = 0 為中心弦長；並與蒙地卡羅取樣比對。"""
    w, L = 0.04, 0.1
    assert strip_extent(w, L, 0.0) == pytest.approx(w)
    assert strip_extent(w, L, np.pi / 2 - 1e-9) == pytest.approx(L)
    for d in np.radians([10, 30, 50, 70]):
        assert strip_extent(w, L, d, b=10.0) == pytest.approx(effective_width(w, L, d))
        assert strip_extent(w, L, d, b=0.0) == pytest.approx(min(w / np.cos(d), L / np.sin(d)))
        rng = np.random.default_rng(0)
        t, s_ = rng.uniform(-0.08, 0.08, 400_000), rng.uniform(-0.01, 0.01, 400_000)
        x, y = t * np.sin(d) + s_ * np.cos(d), t * np.cos(d) - s_ * np.sin(d)
        inside = (np.abs(x) <= L / 2) & (np.abs(y) <= w / 2)
        assert 2 * np.abs(t[inside]).max() == pytest.approx(strip_extent(w, L, d), abs=1e-3)
    dc = clear_misalignment(w, L)
    assert strip_extent(w, L, dc) == pytest.approx(OPENING, abs=1e-9)
    assert dc > max_misalignment(w, L)  # 窄指墊比全寬夾爪容許更大方位誤差
    assert clear_misalignment(0.04, 0.06) == pytest.approx(np.pi / 2)


def test_domain_randomization(env):
    ps, yaws = [], []
    for s in range(40):
        _, info = env.reset(seed=s)
        p = info["params"]
        ps.append(p)
        yaws.append(info["obj_yaw"])
        for k, (lo, hi) in DR_RANGES.items():
            assert lo <= p[k] <= hi
        assert p["L"] >= p["w"]
        # 物件確實以抽樣尺寸/質量/摩擦/姿態生成
        shape = pb.getCollisionShapeData(env.obj, -1, physicsClientId=env.cid)[0]
        assert np.allclose(shape[3], [p["L"], p["w"], p["h"]], atol=1e-6)
        dyn = pb.getDynamicsInfo(env.obj, -1, physicsClientId=env.cid)
        assert dyn[0] == pytest.approx(p["mass"]) and dyn[1] == pytest.approx(p["mu"])
        _, q = pb.getBasePositionAndOrientation(env.obj, physicsClientId=env.cid)
        assert wrap_half_pi(pb.getEulerFromQuaternion(q)[2] - info["obj_yaw"]) == pytest.approx(0, abs=1e-6)
        assert env.kp == pytest.approx(p["wn"] ** 2) and env.f_grip == p["f_grip"]
    for k in DR_RANGES:
        assert np.std([p[k] for p in ps]) > 0
    assert min(yaws) < -1.0 and max(yaws) > 1.0
    nominal = RandPlaceEnv(dr=False)
    _, info = nominal.reset(seed=3)
    assert (info["params"]["L"], info["params"]["w"], info["params"]["h"]) == NOMINAL.size
    assert not nominal.obs_noise
    nominal.close()


def test_observation_noise(env):
    errs = []
    for s in range(150):
        obs, _ = env.reset(seed=s)
        errs.append(obs[6:9] - env.last["po"])
        L, w, h = env.part.size
        assert np.allclose(obs[38:41], [L / 0.1, w / 0.1, h / 0.1], atol=1e-6)
    errs = np.array(errs)
    assert np.all(np.abs(errs.mean(0)) < 0.6e-3)
    assert np.allclose(errs.std(0), SIGMA_P, rtol=0.2)
    clean = RandPlaceEnv(dr=False)
    obs, _ = clean.reset(seed=0)
    assert np.allclose(obs[6:9], clean.last["po"], atol=1e-6)
    d = clean.yaw_rel(clean.last)
    assert np.allclose(obs[36:38], [np.cos(2 * d), np.sin(2 * d)], atol=1e-6)
    clean.close()


def test_deterministic(env):
    a = np.array([0.3, -0.2, -0.5, 0.7, 1.0])
    out = []
    for _ in range(2):
        env.reset(seed=11)
        for _ in range(10):
            obs, r, *_ = env.step(a)
        out.append((obs.copy(), r))
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]


def test_yaw_rate_tracking(env):
    """ψ̇ 指令：ψ_ref 等速積分（ω_ψ Δt），實際指軸 yaw 跟隨（落後 ≈ ψ̇/k_R）、TCP 位置幾乎不動。"""
    env.reset(seed=2, options={"q0": env.q_home})
    p0 = env.last["p"].copy()
    f0 = finger_axis_yaw(env.last["R"])
    psi0 = env.psi_ref
    n = 10
    for _ in range(n):
        env.step([0, 0, 0, 1.0, -1])
    assert env.psi_ref - psi0 == pytest.approx(n * env.omega_psi * env.dt_c, rel=1e-9)
    for _ in range(15):
        env.step([0, 0, 0, 0.0, -1])
    turned = wrap_half_pi(finger_axis_yaw(env.last["R"]) - f0)
    assert turned == pytest.approx(wrap_half_pi(n * env.omega_psi * env.dt_c), abs=np.deg2rad(1.5))
    assert np.linalg.norm(env.last["p"] - p0) < 4e-3
    assert np.max(np.abs(env.last["R"][:, 2] - [0, 0, -1])) < 0.02  # 仍朝下


def test_reach_reward_alignment(env):
    env.reset(seed=5)
    m = dict(env.last)
    d = env.yaw_rel(m)
    assert env._reach_scale(m) == pytest.approx(0.6 + 0.4 * np.cos(d) ** 2)


def test_scripted_randomized_succeeds(env):
    pol = ScriptedRandPlacePolicy(env)
    for s in range(5):
        r = run_place_episode(env, pol, s)
        assert r["success"] and not r["collision"] and not r["drop"] and not r["tipped"]
        assert r["e_xy_final"] < 8e-3


def _run_fixed(env, pol, L, rel):
    """長方塊 L × 40 × 50 mm，物件短邊軸與夾爪指軸相對 yaw = rel（手臂在 home）。"""
    params = {"L": L, "w": 0.040, "h": 0.050, "mass": 0.2, "mu": 0.8, "f_grip": 40.0, "wn": 25.0}
    opts = {"q0": env.q_home, "obj_xy": [0.45, 0.1], "goal_xy": [0.45, -0.1], "params": params}
    env.reset(seed=0, options={**opts, "obj_yaw": 0.0})
    yaw = 0.0 + rel - wrap_half_pi(short_axis_yaw(env.last["qo"]) - finger_axis_yaw(env.last["R"]))
    return run_place_episode(env, pol, 0, options={**opts, "obj_yaw": float(yaw)})


def test_orientation_needed_for_long_object():
    """L = 95 mm > 開口 80 mm：不轉夾爪、物件短邊與指軸差 90° → 夾不住；有方位對齊的腳本 → 成功。"""
    e = RandPlaceEnv(dr=False, obs_noise=False)
    no_turn = ScriptedRandPlacePolicy(e, k_psi=0.0, tol_yaw=10.0)
    r = _run_fixed(e, no_turn, 0.095, np.pi / 2 - 1e-3)
    assert not r["success"] and r["t_lift"] is None
    r = _run_fixed(e, ScriptedRandPlacePolicy(e), 0.095, np.pi / 2 - 1e-3)
    assert r["success"]
    e.close()


def test_train_smoke(tmp_path: Path):
    from sixaxis.rltrain import train

    log = train(algo="sac", task="randplace", name="t", total_timesteps=300, eval_freq=10_000,
                eval_seeds=[5000], demo_episodes=1, models_dir=tmp_path)
    assert log["timesteps"] >= 300 and log["demo"]["episodes"] == 1
    assert (tmp_path / "t_best.zip").exists() and (tmp_path / "t_last.zip").exists()


BEST = Path(__file__).resolve().parents[1] / "models" / "T205_best.txt"


@pytest.mark.skipif(not BEST.exists(), reason="T205 最終模型尚未選定")
def test_trained_randplace_model():
    from sixaxis.rltrain import SB3Policy, load

    name, thr = BEST.read_text().split()
    e = RandPlaceEnv(**({"dr": False} if "nodr" in name else {}))
    pol = SB3Policy(load(BEST.parent / f"{name}_best.zip"))
    rate = np.mean([run_place_episode(e, pol, s)["success"] for s in range(1000, 1020)])
    e.close()
    assert rate >= float(thr)
