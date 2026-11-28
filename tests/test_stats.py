import numpy as np
import pytest
from scipy import stats

from sixaxis.stats import seed_summary, welch, wilson


def test_wilson_known_values():
    # k = 0, n = 10：上界 z²/(n + z²) = 3.8415/13.8415 = 0.2775（Brown et al. 2001 表列 0.278）
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(3.8415 / 13.8415, abs=1e-4)
    # 對稱性：k ↔ n−k
    lo1, hi1 = wilson(3, 20)
    lo2, hi2 = wilson(17, 20)
    assert lo1 == pytest.approx(1 - hi2) and hi1 == pytest.approx(1 - lo2)
    # 含 p̂ 且在 [0, 1] 內；n 增加區間變窄
    for k, n in ((1, 7), (50, 100), (199, 200)):
        lo, hi = wilson(k, n)
        assert 0 <= lo <= k / n <= hi <= 1
    assert np.diff(wilson(100, 200)) < np.diff(wilson(10, 20))


def test_wilson_coverage_monte_carlo():
    # 涵蓋率檢驗：p = 0.9、n = 200，Wilson 95 % 區間實際涵蓋率應接近 0.95（容許 ±0.02）
    rng = np.random.default_rng(0)
    p, n = 0.9, 200
    ks = rng.binomial(n, p, size=4000)
    cover = np.mean([lo <= p <= hi for lo, hi in (wilson(int(k), n) for k in ks)])
    assert abs(cover - 0.95) < 0.02


def test_seed_summary_t_interval():
    x = [0.40, 0.34, 0.55]
    s = seed_summary(x)
    assert s["mean"] == pytest.approx(np.mean(x))
    assert s["std"] == pytest.approx(np.std(x, ddof=1))
    h = stats.t.ppf(0.975, 2) * np.std(x, ddof=1) / np.sqrt(3)
    assert s["ci"][1] - s["mean"] == pytest.approx(h)
    assert stats.t.ppf(0.975, 2) == pytest.approx(4.303, abs=1e-3)
    one = seed_summary([0.5])
    assert one["std"] is None and one["ci"] is None


def test_welch_detects_difference():
    assert welch([0.9, 0.92, 0.88], [0.3, 0.35, 0.32])["p"] < 0.01
    assert welch([0.5, 0.6, 0.4], [0.5, 0.45, 0.55])["p"] > 0.5
