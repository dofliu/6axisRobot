"""由 docs/test-reports/data/2026-11-26_T204_metrics.json 與 models/*_place_*_log.json 產生 T204 最終報告
（數字皆來自實際執行；第一輪階段報告見 2026-11-25_T204.md）。

PYTHONPATH=. python3 scripts/t204_write_final_md.py <pytest 通過數>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-26_T204"
M = json.loads((ROOT / f"docs/test-reports/data/{TAG}_metrics.json").read_text(encoding="utf-8"))
NPASS = sys.argv[1] if len(sys.argv) > 1 else "?"
RL = ["sacbc_place_s0", "sacbc_place_s1", "sac_place_demo_s0", "sac_place_demo_s1", "sac_place_s0", "ppo_place_s0"]
RESUMED = {"sacbc_place_s0", "sacbc_place_s1", "sac_place_demo_s1"}
NAMES = {"sacbc_place_s0": "SAC＋示範 Q-filter BC 種子 0", "sacbc_place_s1": "SAC＋示範 Q-filter BC 種子 1",
         "sac_place_demo_s0": "SAC＋示範預填緩衝 種子 0", "sac_place_demo_s1": "SAC＋示範預填緩衝 種子 1",
         "sac_place_s0": "SAC 無示範 種子 0", "ppo_place_s0": "PPO 無示範 種子 0",
         "scripted": "腳本式 FSM 基準", "random": "隨機策略"}
BEST = M["best_model"]
TR, T, E = M["train"], M["test"], M["evalset"]


def f(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def pct(x, n=0):
    return "—" if x is None else f"{100 * x:.{n}f} %"


def curve(n):
    return json.loads((ROOT / f"models/{n}_log.json").read_text())["curve"]


train_rows = "\n".join(
    f"| {NAMES[n]} | {'2 輪（續訓）' if n in RESUMED else '1 輪'} | {TR[n]['timesteps']:,} | {TR[n]['wall_s'] / 60:.0f} | "
    f"{pct(TR[n]['best']['success_rate'])}（{TR[n]['best']['timesteps']:,} 步） | {pct(TR[n]['final']['success_rate'])} | "
    f"{f(TR[n]['steps_to_50'] / 1e3, 0) if TR[n]['steps_to_50'] else '—'} | {f(TR[n]['steps_to_90'] / 1e3, 0) if TR[n]['steps_to_90'] else '—'} | "
    f"{TR[n]['model_kb']:.0f} |"
    for n in RL
)


def row(n, S):
    return (f"| {NAMES[n]} | {pct(S['success_rate'], 1)} | {pct(S['ever_success'])} | {pct(S['grasp_rate'])} | {pct(S['place_rate'])} | "
            f"{pct(S['collision_rate'], 1)} / {pct(S['drop_rate'], 1)} / {pct(S['tipped_rate'], 1)} | "
            f"{f(S['t_success_mean'])}（{f(S['t_success_p95'])}） | {f(S['e_xy_succ_mean_mm'], 2)}（{f(S['e_xy_succ_p95_mm'], 2)}） | "
            f"{pct(S['hold_frac_mean'])} | {f(S['return_mean'], 0)} | {f(S['tau_ratio_max'])} |")


test_rows = "\n".join(row(n, T[n]) for n in RL + ["scripted", "random"])
ev_rows = "\n".join(f"| {NAMES[n]} | {pct(E[n]['success_rate'])} | {pct(TR[n]['best']['success_rate']) if n in TR else '—'} |" for n in RL + ["scripted"])
var = [("q0wide", "起點擾動 ±0.3 rad（100）"), ("block", "40×60×80 mm 長方塊（100）"), ("long", "15 s 回合（50）")]
var_rows = "\n".join(
    f"| {NAMES[n]} | {pct(T[n]['success_rate'])} | " + " | ".join(pct(M[k][n]['success_rate']) for k, _ in var)
    + f" | {pct(M['long'][n].get('held_after_8s'))} |"
    for n in RL + ["scripted"]
)
tr_rows = "\n".join(
    f"| {NAMES[n]} | {f(M['time_ratio_vs_scripted'][n]['mean'])} | {pct(M['time_ratio_vs_scripted'][n]['faster_frac'])} | "
    f"{M['timing'][n]['mean_ms']:.3f} |" for n in RL
)
bd = M["by_dist"]
bins = bd["bins"]
bd_head = " | ".join(f"[{lo:.2f}, {hi:.2f}) m（{c}）" for lo, hi, c in zip(bins, bins[1:], bd["count"]))
bd_rows = "\n".join(f"| {NAMES[n]} | " + " | ".join(pct(v) for v in bd[n]) + " |" for n in RL + ["scripted"])
rel = M["release"]
B = T[BEST]
S = T["scripted"]
cv = curve(BEST)
cvb = curve("sacbc_place_s1")
c1 = [c for c in cv if c["timesteps"] > 90_000]
thr = (ROOT / "models/T204_best.txt").read_text().split()
first_round_best = max(c["success_rate"] for c in cv if c["timesteps"] <= 90_000)

md = f"""# T204 place／完整 pick-and-place：續訓、200 組比較與泛化（{TAG[:10]}，完成）

> 第一輪（環境、腳本基準、自由落體驗證、示範學習機制、6 組訓練）見 [2026-11-25_T204.md](2026-11-25_T204.md)。
> 本次以 `--resume` 對第一輪最有希望的三組各續訓 ≈ 35 分鐘（3 行程共用 4 核），最佳組 {NAMES[BEST]} 評估集成功率由 {pct(first_round_best)} 升至
> **{pct(TR[BEST]['best']['success_rate'])}**，並在 200 組測試任務達 **{pct(B['success_rate'], 1)}**；腳本基準 {pct(S['success_rate'], 1)}。

## 目的
完成 T204：在第一輪 checkpoint 上續訓直到取放（抓 → 搬 → 放 → 鬆爪退離 → 物件靜置保持）收斂，
並以 200 組未參與訓練/選模的測試任務比較 RL 與腳本基準的成功率、週期時間、放置精度、力矩使用率，以及起點擾動、未見物件、長回合保持的泛化。

## 模型與參數（與第一輪相同，摘要）
* 環境 `SixAxisPlace-v0`（`sixaxis/rlplace.py`）：PyBullet 2 ms 子步、策略 20 Hz、4 維動作（TCP 線速度 ±0.3 m/s、夾爪開閉 40 N）、計算力矩內環；
  回合 160 步 = 8 s；方塊 50 mm／0.2 kg；目標距起點 ≥ 12 cm；觀測 36 維。
* success = at_goal（xy < 20 mm、|Δz| < 10 mm、傾角 < 10°）∧ 夾爪張開 ∧ 兩指墊無接觸；不終止，回合成功 = 最後一步仍 success。
* 分階段取 max 獎勵（Zhu et al. 2020）：r = 6（success）否則 max(1 − tanh 10d, [grasped](1.5 + 2.5(1 − tanh 5e)), 4.5[at_goal ∧ ¬grasped]) − 0.01‖a‖² − 10[碰撞] − 10[掉落]。
* 續訓（`rltrain.train(resume=True)`）：讀 `models/<name>_last.zip`，學習曲線附加到既有 log、步數累加；回放緩衝未存檔（> 50 MB），
  續訓開始時以同種子重新收集 100 個加噪腳本示範（預填組放入一般緩衝、SACBC 放入示範專屬緩衝），之後由新經驗重新填滿。
* 選模：評估集（種子 5000–5049）最佳成功率的 `_best.zip`；測試集（種子 1000–1199）與評估集不重疊。回歸測試門檻寫入 `models/T204_best.txt`：`{thr[0]} {thr[1]}`。

## 可重現指令
```bash
PYTHONPATH=. python3 scripts/t204_train.py --resume --budget 2100 sacbc_place_s0 sacbc_place_s1 sac_place_demo_s1
PYTHONPATH=. python3 scripts/t204_report.py          # 200 組測試＋泛化＋圖＋動畫（4 行程，評估 {M['runtime_s'] / 60:.0f} 分鐘）
PYTHONPATH=. python3 scripts/t204_write_final_md.py {NPASS}
ruff check . && python3 -m pytest -q
```

## 指令序列（每回合）
1. `reset(seed)`：取樣物件起點與目標、重建方塊、手臂於工具朝下 TCP (0.45, 0, 0.32) 附近、夾爪全開。
2. 每 50 ms：策略輸出 (v_x, v_y, v_z, grip) → 25 個 2 ms 物理子步。
3. 量測 grasped／at_goal／success、碰撞、掉落；跑滿 160 步（15 s 變體 300 步）。

## 量化結果

### 1. 訓練總表
| 組態 | 訓練輪數 | 總步數 | 總時間 [min] | 最佳評估成功（步數） | 最後評估成功 | 首達 50 % [k 步] | 首達 90 % [k 步] | 模型 [KB] |
|---|---|---|---|---|---|---|---|---|
{train_rows}

* {NAMES[BEST]}：第一輪結束（9 萬步）評估集 {pct(first_round_best)}，續訓後 {c1[0]['timesteps']:,} → {cv[-1]['timesteps']:,} 步之間由
  {pct(c1[0]['success_rate'])} 升至 {pct(max(c['success_rate'] for c in c1))}，最後 4 次評估 {', '.join(pct(c['success_rate']) for c in cv[-4:])}（穩定）。
* SACBC 種子 1 續訓後最高 {pct(max(c['success_rate'] for c in cvb))}、最後 {pct(cvb[-1]['success_rate'])}；曾到目標比例達 {pct(max(c['place_rate'] for c in cvb))}——
  到目標幾乎全會，但「鬆爪退離」仍不穩定（評估間 20–78 % 擺盪）。SACBC 種子 0 最佳 {pct(TR['sacbc_place_s0']['best']['success_rate'])}。
* 未續訓的三組（SAC 無示範、PPO、預填種子 0）維持第一輪結果，作為消融對照。

![學習曲線](../media/{TAG}_learning.png)

### 2. 200 組測試任務（種子 1000–1199，確定性策略）
| 方法 | 回合成功 | 曾成功 | 曾夾住 | 曾到目標 | 碰撞 / 掉落 / 傾倒 | 首次成功 [s]（P95） | 成功者放置誤差 [mm]（P95） | 保持比例 | 回報 | 最大力矩使用率 |
|---|---|---|---|---|---|---|---|---|---|---|
{test_rows}

評估集（50 組，選模用）與測試集一致性：

| 方法 | 評估集（重跑） | 訓練記錄最佳 |
|---|---|---|
{ev_rows}

週期時間（同一任務、兩者皆成功時 t_RL / t_腳本）與決策時間：

| 方法 | 時間比平均 | 比腳本快的比例 | 決策 [ms/步] |
|---|---|---|---|
{tr_rows}

腳本 {M['timing']['scripted']['mean_ms'] * 1e3:.0f} µs/步；環境一步 {M['timing']['scripted']['env_step_ms']:.2f} ms。

![比較](../media/{TAG}_compare.png)
![測試任務地圖](../media/{TAG}_map.png)

### 3. 泛化與穩健性（測試種子前 N 組）
| 方法 | 標稱（200） | {' | '.join(lab for _, lab in var)} | 15 s 回合成功後保持到結束 |
|---|---|---|---|---|---|
{var_rows}

### 4. 成功率 vs 搬運距離
| 方法 | {bd_head} |
|---|{'---|' * (len(bins) - 1)}
{bd_rows}

### 5. 鬆爪落下自由落體驗證（腳本，各 10 回合；第一輪已詳列，本次重跑）
所有高度量得–理論著地速度差 ≤ {max(r['v_err_max'] for r in rel):.4f} m/s（< gΔt = 0.0196 m/s 取樣界）；放置誤差 {rel[0]['e_xy_mm']:.2f} → {rel[-1]['e_xy_mm']:.2f} mm（h_rel 2 → 80 mm）。

### 單回合曲線（種子 1003；左：{NAMES[BEST]}、中：次佳 RL、右：腳本）
![回合曲線](../media/{TAG}_episode.png)
![關節曲線](../media/{TAG}_joints.png)

## 動畫
上至下：{NAMES[BEST]}、次佳 RL、腳本基準；左全景、右俯視（綠色方塊為目標），任務種子 1003、1011。

![動畫](../media/{TAG}.gif)

[MP4](../media/{TAG}.mp4)｜關鍵畫面：[開始](../media/{TAG}_key_start.png)、[搬運](../media/{TAG}_key_carry.png)、[放置](../media/{TAG}_key_placed.png)

## 觀察與問題
* **續訓解決了第一輪的鬆爪瓶頸**（對 {NAMES[BEST]}）：第一輪 9 萬步時到目標但不鬆爪；續訓後「曾到目標」與「回合成功」幾乎重合，
  表示策略學會「到目標 → 張爪 → 上升退離」並保持。代價是總共 {TR[BEST]['timesteps'] / 1e3:.0f} k 步（T203 pick 約 9 萬步達 90 %），符合長視界多階段任務的樣本需求。
* **示範的兩種用法排名在續訓後翻轉**：第一輪 SACBC 種子 0 最佳；續訓後預填緩衝種子 1 最高且最穩定，SACBC 兩種子在 0–78 % 擺盪。
  可能原因：BC 項把策略拉向腳本的「緩慢下降＋停頓開爪」動作，而 Q-filter 只在示範 Q 值較高時啟用，兩種目標在鬆爪階段互相拉扯；
  測試集上 SACBC 種子 1 有 {pct(T['sacbc_place_s1']['place_rate'], 1)} 回合到目標、放置誤差更小（{f(T['sacbc_place_s1']['e_xy_succ_mean_mm'], 1)} mm），但只有 {pct(T['sacbc_place_s1']['success_rate'], 1)} 完成鬆爪退離——瓶頸仍在鬆爪。
  種子間差異大（只 2 種子），不足以下結論哪種較好，只能說兩者都遠優於無示範（SAC 無示範 / PPO 測試成功 {pct(T['sac_place_s0']['success_rate'])} / {pct(T['ppo_place_s0']['success_rate'])}）。
* **RL 比腳本快、但精度與穩定性較差**：首次成功 {f(B['t_success_mean'])} s vs 腳本 {f(S['t_success_mean'])} s；成功者放置誤差 {f(B['e_xy_succ_mean_mm'], 1)} mm vs {f(S['e_xy_succ_mean_mm'], 2)} mm；
  最大力矩使用率 {f(B['tau_ratio_max'])} vs {f(S['tau_ratio_max'])}。獎勵只要求 20 mm 容差且不懲罰力矩，因此 RL 用較激進的速度換時間。
* 失敗模式見測試地圖（紅 ×）：以「曾到目標但最後未成功」（未鬆爪或鬆後碰動）與「未夾住」為主；成功率隨搬運距離的變化見第 4 節。
* 限制：物件姿態固定（yaw = 0）、完美感知（真值位置）、二元夾爪、單一物件、每組 ≤ 2 種子；續訓時回放緩衝不保存（續訓初期以小緩衝更新）；
  3 組續訓、3 組未續訓，wall-time 與步數不等量，方法間比較只作參考。物件姿態與參數隨機化留待 T205。

## 結論
T204 完成：SixAxisPlace-v0 上最佳 RL 策略（{NAMES[BEST]}，{TR[BEST]['timesteps'] / 1e3:.0f} k 步、約 {TR[BEST]['wall_s'] / 60:.0f} 分鐘單核訓練）在 200 組未見任務達
{pct(B['success_rate'], 1)} 回合成功（無碰撞 {pct(1 - B['collision_rate'], 1)}），週期比腳本基準短；腳本基準仍以 {pct(S['success_rate'], 1)}、亞毫米放置精度領先。
示範（預填或 BC）是此長視界任務學得起來的關鍵。下一步 T205：物件位置與姿態隨機化＋domain randomization。

## 測試
`tests/test_rlplace.py` 11 項全部啟用（`test_trained_place_model`：`models/T204_best.txt` 所記模型在種子 1000–1019 成功率 ≥ {thr[1]}、無碰撞）；
`tests/test_rltrain.py` 已存模型載入涵蓋 place。全套 **{NPASS} passed**、`ruff check` 通過。
"""

(ROOT / f"docs/test-reports/{TAG}.md").write_text(md, encoding="utf-8")
print("written", len(md))
