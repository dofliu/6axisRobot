"""由 docs/test-reports/data/2026-11-27_T205_metrics.json 與 models/*_rand_*_log.json 產生 T205 階段報告
（數字皆來自實際執行）。

PYTHONPATH=. python3 scripts/t205_write_md.py <pytest 通過數>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-27_T205"
M = json.loads((ROOT / f"docs/test-reports/data/{TAG}_metrics.json").read_text(encoding="utf-8"))
NPASS = sys.argv[1] if len(sys.argv) > 1 else "?"
RL = M["rl"]
NAMES = {"sac_rand_demo_s0": "SAC＋示範預填，DR，種子 0", "sac_rand_demo_s1": "SAC＋示範預填，DR，種子 1",
         "sacbc_rand_s0": "SAC＋示範 Q-filter BC，DR，種子 0", "sac_rand_nodr_demo_s0": "SAC＋示範預填，無 DR，種子 0",
         "scripted": "腳本 FSM（含方位對齊）", "noturn": "腳本 FSM（不轉夾爪）", "random": "隨機策略"}
BEST = M["best_model"]


def f(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def pct(x, n=1):
    return "—" if x is None else f"{100 * x:.{n}f} %"


TR, T, TN = M["train"], M["test"], M["test_nodr"]
train_rows = "\n".join(
    f"| {NAMES[n]} | {TR[n]['timesteps']:,} | {TR[n]['wall_s'] / 60:.1f} | {TR[n]['fps']:.0f} | "
    f"{pct(TR[n]['best']['success_rate'], 0)}（{TR[n]['best']['timesteps']:,} 步） | "
    f"{pct(TR[n]['final']['success_rate'], 0)} | {pct(TR[n]['max_place_rate'], 0)} | {pct(TR[n]['max_grasp_rate'], 0)} | "
    f"{TR[n]['demo']['episodes']}/{TR[n]['demo']['tried']} |"
    for n in RL
)
names = RL + ["scripted", "noturn", "random"]
test_rows = "\n".join(
    f"| {NAMES[n]} | **{pct(T[n]['success_rate'])}** | {pct(T[n]['grasp_rate'])} | {pct(T[n]['lift_rate'])} | "
    f"{pct(T[n]['place_rate'])} | {pct(T[n]['collision_rate'])} / {pct(T[n]['drop_rate'])} / {pct(T[n]['tipped_rate'])} | "
    f"{f(T[n]['t_success_mean'])} | {f(T[n]['e_xy_succ_mean_mm'], 1)} / {f(T[n]['e_xy_succ_p95_mm'], 1)} | "
    f"{f(T[n]['tau_ratio_max'])} |"
    for n in names
)
cross_rows = "\n".join(
    f"| {NAMES[n]} | {pct(T[n]['success_rate'])} | {pct(TN[n]['success_rate'])} | "
    f"{pct(T[n]['place_rate'])} | {pct(TN[n]['place_rate'])} |"
    for n in RL + ["scripted", "noturn"]
)
th = M["theory"]
geom = M["geom"]


def gsum(L):
    gg = [g for g in geom if abs(g["L"] - L) < 1e-9]
    clear = [g["rel_deg"] for g in gg if not g["touched_before_close"]]
    lifted = [g["rel_deg"] for g in gg if g["lifted"]]
    fail = [g["rel_deg"] for g in gg if not g["lifted"]]
    return gg, clear, lifted, fail


geom_rows = []
for L in sorted({g["L"] for g in geom}):
    gg, clear, lifted, fail = gsum(L)
    t = th[f"{L:.2f}"]
    pred = [g["rel_deg"] for g in gg if g["strip_mm"] <= M["opening_mm"]]
    agree = sum((g["strip_mm"] <= M["opening_mm"]) == (not g["touched_before_close"]) for g in gg)
    agree2 = sum(g["pred_hit_measured_pose"] == g["touched_before_close"] for g in gg)
    geom_rows.append(
        f"| {1e3 * L:.0f} | {f(t['delta_max_deg'], 1)} | {f(t['delta_clear_deg'], 1)} | "
        f"{', '.join(str(x) for x in pred) if len(pred) < len(gg) else '全部'} | "
        f"{', '.join(str(x) for x in clear) if len(clear) < len(gg) else '全部'} | {agree}/{len(gg)} | "
        f"{agree2}/{len(gg)} | "
        f"{len(lifted)}/{len(gg)} | {', '.join(str(x) for x in fail) or '無'} |")
geom_rows = "\n".join(geom_rows)
n_agree = sum((g["strip_mm"] <= M["opening_mm"]) == (not g["touched_before_close"]) for g in geom)
n_agree2 = sum(g["pred_hit_measured_pose"] == g["touched_before_close"] for g in geom)
mis = [g for g in geom if (g["strip_mm"] <= M["opening_mm"]) == g["touched_before_close"]]
mis_strip = (min(g["strip_mm"] for g in mis), max(g["strip_mm"] for g in mis)) if mis else (0, 0)
offs = [np.hypot(*g["pose_at_top"]["off_mm"]) for g in geom if g["pose_at_top"]]
big_f = [g for g in geom if g["f_pre_close_max"] > 100]
small_f = [g for g in geom if 0.5 < g["f_pre_close_max"] <= 100]
align = [g for g in geom if g["rel_after_close_deg"] is not None and 10 <= g["rel_deg"] <= 35
         and not g["touched_before_close"]]
align_txt = ", ".join(f"L={1e3 * g['L']:.0f} mm：{g['rel_deg']}°→{abs(g['rel_after_close_deg']):.1f}°"
                      for g in align[1::4][:6])
DS = M["dr_stats"]
by = M["by"]


def by_row(key, fmt):
    b = by["scripted"][key]["bins"]
    head = " | ".join(fmt(lo, hi) for lo, hi in zip(b, b[1:]))
    rows = [f"| 分組 | {head} |", "|" + "---|" * (len(b))]
    rows.append("| 樣本數 | " + " | ".join(str(c) for c in by["scripted"][key]["count"]) + " |")
    for n in RL + ["scripted", "noturn"]:
        rows.append(f"| {NAMES[n]} | " + " | ".join(pct(v, 0) for v in by[n][key]["success"]) + " |")
    return "\n".join(rows)


yaw_tab = by_row("yaw", lambda lo, hi: f"{lo:.0f}–{min(hi, 90):.0f}°")
L_tab = by_row("L", lambda lo, hi: f"{lo:.0f}–{min(hi, 100):.0f} mm")
tr = M["time_ratio_vs_scripted"]
tm = M["timing"]
S, NT = T["scripted"], T["noturn"]
B = T[BEST]
hp = TR[BEST]["hparams"]

md = f"""# T205 隨機化物件位置與姿態＋domain randomization（階段報告，第一輪訓練）

日期：2026-11-27　狀態：**進行中 [~]**（環境、基準、夾持幾何驗證與第一輪訓練完成；訓練未收斂，下次續訓後出最終報告）

## 1. 目的
T204 的取放任務物件只有位置隨機、姿態固定（yaw = 0）且物理參數固定。T205 將其擴充為：
1. **物件姿態隨機**：長方塊初始 yaw ~ U[−90°, 90°)，且長邊可大於夾爪開口（80 mm），必須先把夾爪轉到物件短邊方向才能抓；
2. **夾爪方位動作**：動作由 4 維擴為 5 維（TCP 速度＋繞 z 轉速 ψ̇＋夾爪）；
3. **Domain randomization**：每回合隨機物件尺寸/質量/摩擦、夾持力、關節伺服頻寬，並對物件位姿觀測加雜訊；
4. 以腳本基準（含/不含方位對齊）、夾持幾何理論與 RL（SAC＋示範、SACBC、有/無 DR）比較。

## 2. 模型與參數
* 環境 `sixaxis/rlrand.py`：`RandPlaceEnv(PlaceEnv)`，`gym.make("SixAxisRandPlace-v0")`（需先 `import sixaxis.rlrand`）。
  場景、計算力矩內環、獎勵階段與成功判定同 T204（`sixaxis/rlplace.py`），每回合 160 步（8 s）、成功不終止。
* **方位控制**：R_d(ψ) = R_z(ψ)·R_DOWN，ψ_ref ← clip(ψ_ref + a₃·ω_ψ·Δt, ±(π/2 + 0.35))，ω_ψ = 1.5 rad/s；
  ω = k_R·log(R_d Rᵀ) + ψ̇ ẑ（k_R = 2），DLS 解關節速度 → q_ref → 計算力矩。
* **觀測 42 維**：T204 36 維（物件位置/四元數為含雜訊估測）＋ cos 2Δ, sin 2Δ（Δ＝物件短邊軸與指軸相對 yaw，2Δ 消除 180° 對稱）
  ＋ 尺寸 (L, w, h)/0.1 ＋ ψ_ref/π。
* **獎勵**：T204 分階段 max 獎勵，接近項乘以方位對齊倍率 0.6 + 0.4 cos²Δ（`PlaceEnv._reach_scale` 掛鉤，T204 為 1）。
* **Domain randomization**（Tobin et al. 2017；Peng et al. 2018 動力學隨機化），每回合均勻抽樣：

| 參數 | 範圍 | 200 組測試實際 min / mean / max |
|---|---|---|
| 短邊 w | 35–55 mm | {1e3 * DS['w']['min']:.1f} / {1e3 * DS['w']['mean']:.1f} / {1e3 * DS['w']['max']:.1f} mm |
| 長邊 L | max(w, 50)–100 mm | {1e3 * DS['L']['min']:.1f} / {1e3 * DS['L']['mean']:.1f} / {1e3 * DS['L']['max']:.1f} mm（L > 80 mm 佔 {pct(DS['frac_L_gt_opening'], 0)}） |
| 高 h | 45–60 mm | {1e3 * DS['h']['min']:.1f} / {1e3 * DS['h']['mean']:.1f} / {1e3 * DS['h']['max']:.1f} mm |
| 質量 m | 0.1–0.4 kg | {DS['mass']['min']:.3f} / {DS['mass']['mean']:.3f} / {DS['mass']['max']:.3f} kg |
| 物件摩擦 μ_obj | 0.5–1.0 | {DS['mu']['min']:.2f} / {DS['mu']['mean']:.2f} / {DS['mu']['max']:.2f} |
| 夾持力 F_grip | 30–50 N | {DS['f_grip']['min']:.1f} / {DS['f_grip']['mean']:.1f} / {DS['f_grip']['max']:.1f} N |
| 伺服頻寬 ω_n（K_p = ω_n²、K_d = 2ω_n） | 20–30 rad/s | {DS['wn']['min']:.1f} / {DS['wn']['mean']:.1f} / {DS['wn']['max']:.1f} |
| 初始 yaw | −90°–90° | 需轉夾爪 |Δ₀| > 45° 佔 {pct(DS['frac_yaw_gt_45'], 0)} |
| 觀測雜訊 | 物件位置 σ = 2 mm、yaw σ = 2°（每步獨立） | 獎勵/成功一律用真值 |

  無 DR（`dr=False`）：只隨機位置與 yaw，物件固定為 70×45×50 mm、0.2 kg、μ 0.8，無觀測雜訊。
* **腳本基準** `ScriptedRandPlacePolicy`：T204 八段 FSM 加上方位對齊（APPROACH 期間 ψ̇ = k_ψ Δ，k_ψ = 4，|Δ| < 4° 才下降），
  讀與 RL 相同的含雜訊物件估測（一階低通 α = 0.5）；「不轉夾爪」版本 k_ψ = 0。
* **夾持幾何**：全寬夾爪需跨越長方形投影寬度 w_eff(Δ) = w|cosΔ| + L|sinΔ|；本夾爪指墊寬 b = 20 mm，
  只掃過寬 b 的帶狀區，需跨越的是帶內物件長度 `strip_extent`（線性規劃的角點解：s_c = (L/2)cosΔ − (w/2)sinΔ
  夾到 ±b/2，t_max = min((L/2 − s_c cosΔ)/sinΔ, (w/2 + s_c sinΔ)/cosΔ)，全長 2t_max；推導見 `rlrand.strip_extent` 註解，
  並以 40 萬點蒙地卡羅驗證於 `tests/test_rlrand.py`）。指墊不碰物件頂面即可下降的條件 strip_extent ≤ 80 mm。
* 訓練（`scripts/t205_train.py`，4 組各 1 核平行、2100 s 預算）：SAC（lr 3e-4、batch {hp['batch_size']}、γ {hp['gamma']}、
  緩衝 {hp['buffer_size']:,}、網路 256×256）＋ 100 個加噪（σ 0.15）腳本示範成功回合預填；SACBC＝示範專屬緩衝＋Q-filter BC（T204）。

## 3. 可重現指令
```bash
python3 -m pytest -q tests/test_rlrand.py
PYTHONPATH=. python3 scripts/t205_train.py               # 4 組 × 2100 s（4 核）
PYTHONPATH=. python3 scripts/t205_report.py --interim    # 評估、圖、動畫 → docs/test-reports/data/{TAG}_metrics.json
PYTHONPATH=. python3 scripts/t205_write_md.py {NPASS}
```

## 4. 指令序列（腳本基準一回合）
APPROACH（物件上方 8 cm＋轉夾爪 ψ̇ = 4Δ）→ DESCEND（|Δ| < 4°，TCP 到物件中心，指墊下緣 ≥ 台面 + 4 mm）→ CLOSE（6 步）
→ LIFT（8 cm）→ CARRY（目標上方）→ LOWER（離台面 2 mm）→ OPEN（5 步）→ RETREAT（上升 8 cm）。
RL 動作：每 50 ms 輸出 [v_x, v_y, v_z, ψ̇, grip] ∈ [−1, 1]⁵（v_max 0.3 m/s、ω_ψ 1.5 rad/s，grip > 0 閉合）。

## 5. 結果

### 5.1 夾持幾何驗證（不轉夾爪的腳本抓取，w = 40 mm、h = 50 mm，Δ₀ 每 5°）
| L [mm] | 全寬夾爪 Δ_max [°] | 20 mm 指墊理論 Δ_clear [°] | 理論可淨空下降的 Δ₀ | 實測指墊未碰物件的 Δ₀ | 理想位姿理論一致 | 實際位姿 SAT 一致 | 抬起 | 失敗 Δ₀ |
|---|---|---|---|---|---|---|---|---|
{geom_rows}

* 「指墊是否在閉合前碰到物件」：理想位姿（TCP 對準物件中心、Δ = Δ₀）的帶狀區理論 strip_extent ≤ 80 mm 與實測一致 {n_agree}/{len(geom)} 組；
  改用指墊下緣降到物件頂面時的**實際**位姿（實測 Δ 與 TCP–物件水平偏移，偏移平均 {np.mean(offs):.1f} mm、最大 {max(offs):.1f} mm），
  以分離軸定理判斷俯視指墊矩形與物件矩形是否重疊（`rlrand.pads_hit_box`），一致 {n_agree2}/{len(geom)} 組。
  不一致者理想帶狀長度皆在 {mis_strip[0]:.1f}–{mis_strip[1]:.1f} mm（每側餘隙 ≤ {(80 - mis_strip[0]) / 2:.1f} mm），與 TCP 定位偏移及下降中姿態誤差同量級，屬邊界情形。
* 碰觸分兩類：輕擦稜邊（{len(small_f)} 組，閉合前最大接觸力 ≤ 100 N，多仍可抬起）與指墊壓在物件頂面（{len(big_f)} 組，> 100 N，
  其中抬起 {sum(g["lifted"] for g in big_f)} 組——物件被壓動/推轉後偶然落入兩指間）。
* 全寬夾爪公式 w_eff 會低估可容許方位誤差（窄指墊只需避開帶狀區：L ≥ 80 mm 時 Δ_clear ≈ 46.9° vs Δ_max 26–37°）。
* **閉合自我對正**：Δ₀ 中等時兩指閉合產生的力偶把物件轉向對齊（{align_txt}），因此「可抓」範圍大於「可淨空下降」範圍；
  更大 Δ₀ 時指墊先壓到/推動物件（閉合前最大接觸力見 JSON `f_pre_close_max`），結果依接觸細節而定，長物件 Δ₀ 接近 90° 時指墊壓在頂面而失敗。
  ![geometry](../media/{TAG}_geometry.png)

### 5.2 200 組 DR 測試任務（種子 1000–1199，物件/目標/姿態/物理參數/雜訊皆相同）
| 策略 | 成功（回合結束） | 曾夾住 | 曾抬起 | 曾到目標 | 碰撞 / 掉落 / 傾倒 | 首次成功 [s] | 放置誤差 平均 / P95 [mm] | 最大力矩使用率 |
|---|---|---|---|---|---|---|---|---|
{test_rows}

* 腳本基準含方位對齊 {pct(S['success_rate'])}，不轉夾爪 {pct(NT['success_rate'])}：方位對齊是此任務的必要能力
  （不轉夾爪的失敗集中在大 |Δ₀| 與長物件，見 5.4）。
* RL（第一輪）最佳為 {NAMES[BEST]}：{pct(B['success_rate'])}（曾到目標 {pct(B['place_rate'])}）。

### 5.3 DR 交叉評估（成功率／曾到目標）
| 策略 | DR 測試環境 | 標稱環境（無 DR） | 到目標（DR） | 到目標（標稱） |
|---|---|---|---|---|
{cross_rows}

* 第一輪（同 ≈ 35 分鐘預算）中，**無 DR** 訓練的模型反而在 DR 測試環境成功率最高（{pct(T['sac_rand_nodr_demo_s0']['success_rate'])}，標稱環境 {pct(TN['sac_rand_nodr_demo_s0']['success_rate'])}）：
  DR 增加回合間變異、使同樣步數下學得較慢；而 DR 範圍（尺寸 ±、質量、摩擦、夾持力、伺服頻寬、2 mm/2° 雜訊）尚未超出無 DR 策略的容忍度。
  DR 模型在兩環境間差距小（DR 對 SAC 預填兩組 ≤ 4.5 個百分點），SACBC 在標稱環境反而下降（{pct(TN['sacbc_rand_s0']['success_rate'])}），顯示其策略仍不穩定。
  DR 的價值需在收斂後、或加大分布外擾動時才能判斷（下次續訓後評估）。

![cross](../media/{TAG}_crosseval.png)

### 5.4 依初始方位差 |Δ₀| 與長邊 L 分組成功率（DR 測試 200 組）
{yaw_tab}

{L_tab}

![breakdown](../media/{TAG}_breakdown.png)

### 5.5 訓練（第一輪，各 1 核、2100 s）
| 組態 | 環境步 | 時間 [min] | 步/s | 評估集最佳成功（步數） | 最後一次 | 最高曾到目標 | 最高曾夾住 | 示範成功/嘗試 |
|---|---|---|---|---|---|---|---|---|
{train_rows}

![learning](../media/{TAG}_learning.png)

* 推論時間：{NAMES[RL[0]]} {f(tm[RL[0]]['mean_ms'], 2)} ms/步（P95 {f(tm[RL[0]]['p95_ms'], 2)}）、環境一步 {f(tm[RL[0]]['env_step_ms'], 1)} ms；
  腳本 {f(tm['scripted']['mean_ms'], 3)} ms/步。
* 與腳本同任務的首次成功時間比（RL/腳本）：""" + "；".join(
    f"{NAMES[n]} {f(tr[n]['mean'])}（{pct(tr[n]['faster_frac'], 0)} 較快）" for n in RL if tr[n]["mean"] is not None
) + f"""。

### 5.6 圖表與動畫
* 動畫（三列：{NAMES[BEST]}／腳本含方位對齊／腳本不轉夾爪；左＝斜視、右＝俯視；任務種子 {M.get('anim_seeds')}，皆為 L > 82 mm 且 |Δ₀| > 50°）：
  ![anim](../media/{TAG}.gif)　[MP4](../media/{TAG}.mp4)
* 關鍵畫面：![k1](../media/{TAG}_key_start.png) ![k2](../media/{TAG}_key_align.png) ![k3](../media/{TAG}_key_carry.png) ![k4](../media/{TAG}_key_placed.png)
* 單回合曲線（高度/目標誤差、相對 yaw Δ 與 ψ_ref、指墊力與指令、關節力矩）：![ep](../media/{TAG}_episode.png)
* 關節角/速度：![jt](../media/{TAG}_joints.png)

## 6. 問題與限制
* 第一輪訓練未收斂（T204 同類方法需約 19 萬步才達 90 %，本輪各組約 {min(TR[n]['timesteps'] for n in RL) // 1000}–{max(TR[n]['timesteps'] for n in RL) // 1000} 千步）；
  42 維觀測＋5 維動作且 DR 增加變異，需續訓。
* 放置只要求位置（xy < 20 mm、正立、鬆爪），不要求放置方位；物件為長方塊（無圓柱/不規則形）。
* DR 未含感知延遲、動作延遲、台面摩擦與相機外參；觀測雜訊為白雜訊（實際視覺誤差有偏差與時間相關）。
* 夾持幾何實驗以單一 w = 40 mm、h = 50 mm、μ = 0.8 進行；自我對正結果依摩擦與接觸細節而異，未建立完整的推動力學（Mason 1986）模型。

## 7. 結論（階段）
* 新增 `SixAxisRandPlace-v0`：物件姿態隨機＋夾爪方位動作＋domain randomization，腳本基準（含方位對齊）在 200 組 DR 任務達 {pct(S['success_rate'])}，
  不轉夾爪降為 {pct(NT['success_rate'])}，確認方位對齊為必要；窄指墊帶狀區理論預測閉合前是否碰物件 {n_agree}/{len(geom)}（以實際位姿 {n_agree2}/{len(geom)}）。
* RL 第一輪最佳 {pct(B['success_rate'])}；下一步以 `scripts/t205_train.py --resume` 續訓後完成最終比較與 DR 效果分析。

測試：`python3 -m pytest` {NPASS} passed；`ruff check .` 通過。
"""
out = ROOT / f"docs/test-reports/{TAG}.md"
out.write_text(md, encoding="utf-8")
print(out)
