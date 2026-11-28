"""RL 多種子評估用的統計工具（T205 起：成功率 Wilson 區間、種子間平均 ± 標準差與 t 信賴區間）。

* Wilson score 區間（Wilson 1927；Brown, Cai & DasGupta 2001
  "Interval Estimation for a Binomial Proportion"）：
  p̂ = k/n，z = Φ⁻¹(1 − α/2)，
  center = (p̂ + z²/(2n)) / (1 + z²/n)，half = z √(p̂(1−p̂)/n + z²/(4n²)) / (1 + z²/n)。
  比常態近似（Wald）在 p̂ 接近 0/1 或 n 小時涵蓋率正確，且區間不會超出 [0, 1]。
* 種子間：把每個種子的成功率視為一個樣本 x_i（i = 1..m），平均 x̄、樣本標準差 s（ddof = 1），
  95 % 信賴區間 x̄ ± t_{m−1, 0.975} s/√m（Student t；m = 3 時 t = 4.303，區間很寬——種子數少是主要不確定來源，
  Henderson et al. 2018 "Deep RL that Matters"；
  Agarwal et al. 2021 "Deep RL at the Edge of the Statistical Precipice"）。
* Welch t 檢定（不假設等變異）比較兩組種子平均。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """成功 k 次 / n 次的 Wilson score 信賴區間 (lo, hi)。"""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def seed_summary(x, conf: float = 0.95) -> dict:
    """種子間彙總：平均、樣本標準差（ddof=1）、Student t 信賴區間。m = 1 時標準差與區間為 None。"""
    x = np.asarray(x, float)
    m = len(x)
    out = {"n_seeds": m, "values": x.tolist(), "mean": float(x.mean()), "std": None, "ci": None}
    if m >= 2:
        s = float(x.std(ddof=1))
        h = float(stats.t.ppf(0.5 + conf / 2, m - 1) * s / np.sqrt(m))
        out.update(std=s, ci=(float(x.mean() - h), float(x.mean() + h)))
    return out


def welch(a, b) -> dict:
    """兩組種子平均的 Welch t 檢定（雙尾）。"""
    r = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False)
    return {"t": float(r.statistic), "p": float(r.pvalue), "diff": float(np.mean(a) - np.mean(b))}
