"""由 docs/test-reports/data/2026-11-28_T205_seeds.json 產生 T205 第二輪階段報告（數字皆來自實際執行）。

PYTHONPATH=. python3 scripts/t205_seeds_md.py <pytest 結果字串>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-28_T205"
M = json.loads((ROOT / f"docs/test-reports/data/{TAG}_seeds.json").read_text(encoding="utf-8"))
NPASS = sys.argv[1] if len(sys.argv) > 1 else "?"
GROUPS = {"DR": ["sac_rand_demo_s0", "sac_rand_demo_s1", "sac_rand_demo_s2"],
          "noDR": ["sac_rand_nodr_demo_s0", "sac_rand_nodr_demo_s1", "sac_rand_nodr_demo_s2"]}
GN = {"DR": "DR 訓練", "noDR": "無 DR 訓練"}
NEW = {"sac_rand_demo_s2", "sac_rand_nodr_demo_s1", "sac_rand_nodr_demo_s2"}


def pct(x, n=1):
    return "—" if x is None else f"{100 * x:.{n}f} %"


def ci(c, rate=True):
    if c is None:
        return "—"
    lo, hi = (max(0.0, c[0]), min(1.0, c[1])) if rate else c
    return f"[{100 * lo:.1f}, {100 * hi:.1f}] %" if rate else f"[{lo:.2f}, {hi:.2f}]"


def ms(S, rate=True, n=1):
    if rate:
        return f"{100 * S['mean']:.{n}f} ± {100 * S['std']:.{n}f} %"
    return f"{S['mean']:.2f} ± {S['std']:.2f}"


Mo, Tr, G = M["models"], M["train"], M["groups"]
model_rows = "\n".join(
    f"| {GN[g]} s{n[-1]}{'（本輪新訓）' if n in NEW else '（第一輪）'} | {Tr[n]['timesteps']:,} | {Tr[n]['wall_s'] / 60:.1f} | "
    f"{pct(Tr[n]['best']['success_rate'], 0)}（{Tr[n]['best']['timesteps']:,}） | {pct(Tr[n]['final']['success_rate'], 0)} | "
    f"**{pct(Mo[n]['test']['success_rate'])}** {ci(Mo[n]['test']['wilson'])} | "
    f"{pct(Mo[n]['test_nodr']['success_rate'])} {ci(Mo[n]['test_nodr']['wilson'])} | "
    f"{pct(Mo[n]['test']['place_rate'])} | {pct(Mo[n]['test']['grasp_rate'])} |"
    for g, ns in GROUPS.items() for n in ns)
S = Mo["scripted"]
model_rows += (f"\n| 腳本 FSM（含方位對齊） | — | — | — | — | **{pct(S['test']['success_rate'])}** {ci(S['test']['wilson'])} | "
               f"{pct(S['test_nodr']['success_rate'])} {ci(S['test_nodr']['wilson'])} | {pct(S['test']['place_rate'])} | "
               f"{pct(S['test']['grasp_rate'])} |")

cross = "\n".join(
    f"| {GN[g]} | {ms(G[g]['test:success_rate'])} {ci(G[g]['test:success_rate']['ci'])} | "
    f"{ms(G[g]['test_nodr:success_rate'])} {ci(G[g]['test_nodr:success_rate']['ci'])} | "
    f"{100 * G[g]['gap_nominal_minus_dr']['mean']:+.1f} ± {100 * G[g]['gap_nominal_minus_dr']['std']:.1f} pp |"
    for g in GROUPS)
W = M["welch"]

metric_keys = [("ever_success", "曾成功率", True), ("place_rate", "曾到目標率", True), ("grasp_rate", "夾住率", True),
               ("collision_rate", "碰撞終止率", True), ("drop_rate", "掉落率", True)]
metric_rows = "\n".join(
    f"| {lab} | {ms(G['DR'][f'test:{k}'])} | {ms(G['noDR'][f'test:{k}'])} |" for k, lab, _ in metric_keys)
metric_rows += (f"\n| 成功者放置誤差 [mm] | {ms(G['DR']['test:e_xy_succ_mean_mm'], False)} | "
                f"{ms(G['noDR']['test:e_xy_succ_mean_mm'], False)} |")
metric_rows += (f"\n| 首次成功時間 [s] | {ms(G['DR']['test:t_success_mean'], False)} | "
                f"{ms(G['noDR']['test:t_success_mean'], False)} |")
metric_rows += (f"\n| 最大力矩使用率 | {ms(G['DR']['test:tau_ratio_max'], False)} | "
                f"{ms(G['noDR']['test:tau_ratio_max'], False)} |")

b = M["bins"]["yaw"]
cnt = M["bin_count"]["yaw"]["count"]
yaw_rows = "\n".join(
    f"| {b[i]:.0f}–{min(b[i + 1], 90):.0f}° | {cnt[i]} | {pct(G['DR']['by']['yaw'][i], 0)} ± {pct(G['DR']['by_std']['yaw'][i], 0)} | "
    f"{pct(G['noDR']['by']['yaw'][i], 0)} ± {pct(G['noDR']['by_std']['yaw'][i], 0)} | "
    f"{pct(M['by']['scripted']['yaw']['success'][i], 0)} |" for i in range(len(cnt)))
A = M["anim"]

md = f"""# T205 隨機化物件位置與姿態＋domain randomization — 第二輪階段報告（3 種子交叉評估）

- 日期：2026-11-28（台北 18:37 排程；當日前兩次排程無開發紀錄，本次改為開發執行）
- 狀態：**進行中 [~]**（本輪補齊 DR／無 DR 各 3 個種子並以等訓練預算做交叉評估；各組學習曲線仍在上升、尚未收斂，下輪續訓）
- 第一輪報告：[2026-11-27_T205.md](2026-11-27_T205.md)（環境、獎勵、DR 範圍、腳本基準、夾持幾何驗證）

## 1. 目的
依 ROADMAP T205 完成條件，「DR 訓練 vs 無 DR 訓練」須在標稱環境與 DR 環境交叉評估，並以 ≥ 3 個種子報告平均 ± 標準差與 95 % 信賴區間。
第一輪只有 DR 2 種子（＋SACBC 1）與無 DR 1 種子，本輪新訓 `sac_rand_demo_s2`、`sac_rand_nodr_demo_s1`、`sac_rand_nodr_demo_s2`，
使兩組都有 3 個種子、每組皆約 35 分鐘（單核）的同等訓練預算，先做「等預算」比較。

## 2. 模型與參數
- 環境 `SixAxisRandPlace-v0`（`sixaxis/rlrand.py`，第一輪報告第 2 節）：物件 yaw ~ U[−90°, 90°)、5 維動作（TCP 速度＋ψ̇＋夾爪）、42 維觀測、160 步（8 s）。
  - DR 開：尺寸 w 35–55／L ≤ 100／h 45–60 mm、質量 0.1–0.4 kg、摩擦 0.5–1.0、夾持力 30–50 N、伺服頻寬 20–30 rad/s，物件位置 σ 2 mm／yaw σ 2° 觀測雜訊。
  - DR 關（標稱）：70 × 45 × 50 mm、0.2 kg、μ 0.8、40 N、25 rad/s、無觀測雜訊；位置與 yaw 仍隨機。
- 演算法：SB3 SAC＋100 個加噪腳本示範預填回放緩衝（T204 最佳做法），超參數同第一輪；種子 0/1/2。
- 測試：200 組固定測試種子 1000–1199，各模型分別在 DR 環境與標稱環境（同種子 → 同位置/目標/yaw，僅 DR 參數與雜訊不同）評估，確定性策略。
- 統計（`sixaxis/stats.py`，新增）：單模型成功率 Wilson score 95 % 區間（Brown, Cai & DasGupta 2001）；
  種子間平均 ± 樣本標準差、Student t 95 % 區間 x̄ ± t₂,₀.₉₇₅ s/√3（t = 4.303；超出 [0, 1] 的部分截斷顯示）；兩組平均差 Welch t 檢定。

## 3. 可重現指令
```bash
PYTHONPATH=. python3 scripts/t205_train.py sac_rand_demo_s2 sac_rand_nodr_demo_s1 sac_rand_nodr_demo_s2  # 3 組各 1 核、2100 s
PYTHONPATH=. python3 scripts/t205_seeds.py        # 6 模型 × 2 環境 × 200 回合＋腳本基準、圖、動畫（約 {M['runtime_s'] / 60:.0f} 分鐘，4 核）
PYTHONPATH=. python3 scripts/t205_seeds_md.py "<pytest 結果>"
python3 -m pytest tests/test_stats.py tests/test_rlrand.py
```

## 4. 指令序列（每回合）
reset(seed) → 物件（DR 參數）與目標標記生成 → 策略每 50 ms 輸出 (v_x, v_y, v_z, ψ̇, grip) → TCP 速度＋姿態保持 DLS 積分 q_ref → 2 ms 計算力矩內環 →
160 步後判定 success（物件 xy 距目標 < 20 mm、靜置、傾角 < 10°、夾爪張開且指墊無接觸）。

## 5. 量化結果

### 5.1 各模型（200 組測試；括號為 Wilson 95 % 區間）
| 模型 | 訓練步數 | 訓練時間 [min] | 評估集最佳（步數） | 評估集最後 | DR 測試成功 | 標稱環境成功 | DR 曾到目標 | DR 夾住 |
|---|---|---|---|---|---|---|---|---|
{model_rows}

### 5.2 交叉評估（3 種子平均 ± 標準差，[t 95 % 區間]）
| 訓練方式 | 在 DR 測試環境 | 在標稱環境 | 標稱 − DR 落差 |
|---|---|---|---|
{cross}

Welch t 檢定（DR − 無 DR，種子平均差）：DR 環境 {100 * W['test']['diff']:+.1f} pp（t = {W['test']['t']:.2f}, p = {W['test']['p']:.3f}）；
標稱環境 {100 * W['test_nodr']['diff']:+.1f} pp（t = {W['test_nodr']['t']:.2f}, p = {W['test_nodr']['p']:.3f}）。

### 5.3 其他指標（DR 測試環境，3 種子平均 ± 標準差）
| 指標 | DR 訓練 | 無 DR 訓練 |
|---|---|---|
{metric_rows}

腳本基準：成功 {pct(S['test']['success_rate'])}、放置誤差 {S['test']['e_xy_succ_mean_mm']:.1f} mm、首次成功 {S['test']['t_success_mean']:.2f} s。

### 5.4 依初始方位差 |Δ₀| 分組（DR 測試，種子平均 ± 標準差）
| \\|Δ₀\\| | 任務數 | DR 訓練 | 無 DR 訓練 | 腳本 |
|---|---|---|---|---|
{yaw_rows}

## 6. 圖表與動畫
- 交叉評估與學習曲線（3 種子平均 ± 標準差）：![](../media/{TAG}_seeds.png)
- 依方位差／長度／質量／摩擦分組：![](../media/{TAG}_seeds_breakdown.png)
- 動畫（DR 測試任務 {A['seed']}：上 = DR 最佳 `{A['models'][0]}`、下 = 無 DR 最佳 `{A['models'][1]}`；左斜視、右俯視）：
  [MP4](../media/{TAG}.mp4)／[GIF](../media/{TAG}.gif)；此任務選自「DR 模型成功、無 DR 模型失敗」的 {A['dr_ok_nodr_fail_count']} 組中 |Δ₀| 最大者（非代表性，整體上無 DR 模型較佳）。
- 關鍵畫面：![](../media/{TAG}_key_start.png) ![](../media/{TAG}_key_end.png)
- 該回合關節角/速度/力矩、放置誤差、方位差與指墊力：![](../media/{TAG}_episode.png)

## 7. 問題與限制
- **未收斂**：6 組在 ≈ 35 分鐘（12.9–14.4 萬步）時評估集成功率多數仍在上升（無 DR s1/s2 最佳即最後一次評估），本輪結果只代表「等短預算」比較，不能作為 DR 效果的最終結論。
- 種子數 3：t 區間寬（無 DR 組 DR 環境 {ci(G['noDR']['test:success_rate']['ci'])}），兩組差 p ≈ {W['test']['p']:.2f}／{W['test_nodr']['p']:.2f}，未達 0.05 顯著。
- 在此預算下 **無 DR 訓練在 DR 測試環境反而較佳**；DR 模型在兩環境間落差小但整體成功率低，符合「DR 增加任務變異 → 同預算學得較慢」的預期（Tobin 2017、Peng 2018 皆以更長訓練換取穩健度）。
  DR 模型對方位差較不敏感（各分組約 32–44 %），無 DR 模型隨 |Δ₀| 增大由 ≈ 71–74 % 降至 43 %。
- 兩組在標稱環境的成功率皆略低於 DR 環境（落差 −2 ～ −4 pp，小於單模型 Wilson 區間寬度）；DR 測試含較短物件（L < 70 mm）可能較易抓取，但本輪未分離此因素。
- 力矩使用率皆達 1.00（策略頻繁打滿速度命令），放置誤差 ≈ 12 mm 遠大於腳本 1.6 mm（同 T204 觀察）。
- SACBC（第一輪 1 種子）本輪未納入多種子比較。

## 8. 結論
本輪補齊 DR／無 DR 各 3 種子並完成等預算交叉評估：DR 訓練 {ms(G['DR']['test:success_rate'])}（DR 環境）、{ms(G['DR']['test_nodr:success_rate'])}（標稱）；
無 DR 訓練 {ms(G['noDR']['test:success_rate'])}、{ms(G['noDR']['test_nodr:success_rate'])}；腳本 100 %。短預算下 DR 尚未顯示優勢且組間差未達顯著。
下一步：6 組續訓（`--resume`，兩輪 3 組並行）至學習曲線平穩，再以 `t205_report.py`（最終）＋本腳本產生最終 DR 效果結論並標 [x]。

測試：{NPASS}；`ruff check .` 通過。
"""
(ROOT / f"docs/test-reports/{TAG}.md").write_text(md, encoding="utf-8")
print(md[:400])
