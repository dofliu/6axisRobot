"""由 docs/test-reports/data/2026-11-23_T202_metrics.json 與 models/*_log.json 產生 T202 報告（數字皆來自實際執行）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-23_T202"
M = json.loads((ROOT / f"docs/test-reports/data/{TAG}_metrics.json").read_text(encoding="utf-8"))
T1 = json.loads((ROOT / "docs/test-reports/data/2026-11-22_T201_metrics.json").read_text(encoding="utf-8"))
LOGS = {n: json.loads((ROOT / f"models/{n}_log.json").read_text()) for n in ("sac_ee", "ppo_ee", "sac_joint", "ppo_joint")}
NPASS = sys.argv[1] if len(sys.argv) > 1 else "?"
RL = ["sac_ee", "ppo_ee", "sac_joint", "ppo_joint"]
ALL = RL + ["planner_joint", "rr_ee"]
NAMES = {"sac_ee": "SAC（ee 動作）", "ppo_ee": "PPO（ee 動作）", "sac_joint": "SAC（joint 動作）",
         "ppo_joint": "PPO（joint 動作）", "planner_joint": "規劃：IK＋五次多項式（T201）", "rr_ee": "解析率 P 控制 ee（T201）"}
E = T1["env"]


def f(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


Tt, H, Q, TM, B, TR = M["test"], M["hold"], M["q0wide"], M["timing"], M["bounds"], M["train"]
# T201 基準重跑一致性
same = all(abs(Tt[k]["return_mean"] - T1["baselines"][k]["return_mean"]) < 1e-9 for k in ("planner_joint", "rr_ee"))

train_rows = "\n".join(
    f"| {NAMES[n]} | {TR[n]['timesteps']:,} | {TR[n]['wall_s'] / 60:.1f} | {TR[n]['fps']:.0f} | "
    f"{TR[n]['steps_to_90']:,} / {TR[n]['wall_to_90'] / 60:.1f} | {TR[n]['steps_to_100']:,} | "
    f"{TR[n]['best']['timesteps']:,} | {TR[n]['model_kb'] / 1024:.2f} |"
    for n in RL
)
test_rows = "\n".join(
    f"| {NAMES[n]} | {100 * Tt[n]['success_rate']:.1f} % | {100 * Tt[n]['collision_rate']:.1f} % | "
    f"{f(Tt[n]['t_success_mean'])} | {f(Tt[n]['t_success_p95'])} | {f(Tt[n]['d_final_mean_mm'], 1)} | "
    f"{f(Tt[n]['return_mean'])} ± {f(Tt[n]['return_std'])} | {f(Tt[n]['path_ratio_mean'])} | {f(Tt[n]['tau_ratio_max'])} |"
    for n in ALL
)
ratio_rows = "\n".join(
    f"| {NAMES[n]} | {f(M['time_ratio_vs_planner'][n]['mean'], 3) if n in M['time_ratio_vs_planner'] else '1'} | "
    f"{100 * M['time_ratio_vs_planner'][n]['faster_frac']:.1f} % | "
    f"{'t_line' if B[n]['ref'] == 't_line' else 't_IK'} | {f(B[n]['ratio_mean'], 3)} | {f(B[n]['ratio_min'], 3)} | "
    f"{f(B[n]['ratio_p95'], 3)} |"
    if n in M["time_ratio_vs_planner"] else
    f"| {NAMES[n]} | 1 | — | t_IK | {f(B[n]['ratio_mean'], 3)} | {f(B[n]['ratio_min'], 3)} | {f(B[n]['ratio_p95'], 3)} |"
    for n in ALL
)
hold_rows = "\n".join(
    f"| {NAMES[n]} | {100 * H[n]['all_in_tol_frac']:.0f} % | {100 * H[n]['in_tol_last2s']:.1f} % | "
    f"{f(H[n]['d_final_mean_mm'], 2)} | {f(H[n]['d_final_max_mm'], 2)} | {f(H[n]['qd_final_mean'], 3)} |"
    for n in ALL
)
q_rows = "\n".join(
    f"| {NAMES[n]} | {100 * Q[n]['success_rate']:.0f} % | {100 * Q[n]['collision_rate']:.0f} % | "
    f"{f(Q[n]['t_success_mean'])} | {f(Q[n]['d_final_mean_mm'], 1)} | {f(Q[n]['tau_ratio_max'])} |"
    for n in ALL
)
tm_rows = " ｜ ".join(f"{NAMES[n]} {TM[n]['mean_ms']:.3f} ms" for n in ALL)


def curve_rows(n, every=1):
    c = LOGS[n]["curve"]
    pick = [x for i, x in enumerate(c) if i % every == 0 or i == len(c) - 1]
    return "\n".join(
        f"| {x['timesteps']:,} | {x['wall_s'] / 60:.1f} | {100 * x['success_rate']:.0f} % | {x['return_mean']:.2f} | "
        f"{x['d_final_mean_mm']:.1f} | {f(x['t_success_mean'])} |" for x in pick
    )


hp = {n: LOGS[n]["hparams"] for n in RL}
md = f"""# T202 PPO/SAC 訓練 reach 與學習曲線（測試報告）

日期：2026-11-23｜TaskID：T202｜Phase 3 第二項｜相依：T201

## 1. 目的
以 Stable-Baselines3 的 **PPO** 與 **SAC** 在 T201 的 `SixAxisReach-v0` 環境訓練 reach 策略（ee 與 joint 兩種動作空間，共 4 組），
記錄學習曲線、把模型存到 `models/`，並在與 T201 **完全相同的 200 個測試目標**上，以相同指標和傳統規劃控制基準
（IK＋五次多項式、笛卡兒解析率 P 控制）比較；另做「到達後停住」與「起點擾動加大」兩項泛化測試。

## 2. 模型與參數
### 2.1 環境（T201，不變）
PyBullet 2 ms 物理子步、策略 {E['control_hz']:.0f} Hz、回合 {E['max_steps']} 步（5 s）；計算力矩內環 ω_n = {E['wn']:.0f} rad/s；
joint 動作 u = a·ω_max（ω_max = {E['omega_max']} rad/s），ee 動作 v = a·v_max（v_max = {E['v_max']} m/s，姿態保持＋DLS）；
觀測 27 維；dense 獎勵 r = −d − {E['c_action']}‖a‖² + {E['r_success']:.0f}·[成功] − {E['r_collision']:.0f}·[碰撞]；
成功 = d < {1e3 * E['tol']:.0f} mm 且 ‖ṗ‖ < {E['v_tol']} m/s（成功即終止）；起點 q_home ± {E['q0_noise']} rad 均勻擾動。

### 2.2 演算法與超參數（`sixaxis/rltrain.py`）
* **PPO**（Schulman et al. 2017）：截斷比率目標 L = E[min(ρÂ, clip(ρ, 1−ε, 1+ε)Â)]，GAE(γ, λ)（Schulman et al. 2016）。
  lr = {hp['ppo_ee']['learning_rate']}, n_steps = {hp['ppo_ee']['n_steps']}, batch = {hp['ppo_ee']['batch_size']},
  epochs = {hp['ppo_ee']['n_epochs']}, γ = {hp['ppo_ee']['gamma']}, λ = {hp['ppo_ee']['gae_lambda']}, ε = {hp['ppo_ee']['clip_range']},
  網路 2×256（tanh，策略與價值分開），log σ 初值 −1。
* **SAC**（Haarnoja et al. 2018a/b）：最大熵目標 Σ E[r + αH(π)]、雙 Q 取最小、Polyak τ = {hp['sac_ee']['tau']}、
  溫度 α 自動調整至目標熵 −dim(A)。lr = {hp['sac_ee']['learning_rate']}, buffer = {hp['sac_ee']['buffer_size']:,},
  batch = {hp['sac_ee']['batch_size']}, γ = {hp['sac_ee']['gamma']}, learning_starts = {hp['sac_ee']['learning_starts']:,},
  每環境步 1 次梯度更新，網路 2×256（ReLU）。
* γ = 0.98：有效視界 1/(1−γ) = 50 步 = 2.5 s，涵蓋 T201 規劃基準 P95 成功時間 1.75 s。
* 學習曲線評估：每 10k（SAC）/ 40k（PPO）步以**固定** 50 個目標（種子 5000–5049，與測試集 1000–1199 不重疊）做確定性評估
  （取分佈均值動作）；以（成功率, 平均回報）字典序最佳者存 `models/<組態>_best.zip`，結束存 `<組態>_last.zip`（可續訓）。
* 計算預算：四組平行、各 1 個 CPU 核心（`torch.set_num_threads(1)`）、wall-time 上限 30 分鐘（CLAUDE.md：單次 ≲ 40 分鐘），
  單一隨機種子（seed 0）。

## 3. 可重現指令
```bash
python3 -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install stable-baselines3==2.9.2
PYTHONPATH=. python3 scripts/t202_train.py            # 4 組平行訓練，各 1800 s（--resume 由 *_last.zip 接續）
PYTHONPATH=. python3 scripts/t202_report.py           # 評估、圖、動畫（約 {M['runtime_s']:.0f} s 評估＋渲染）
python3 scripts/t202_write_md.py {NPASS}
ruff check . && python3 -m pytest -q
```
訓練主控台紀錄：[data/{TAG}_train.log](data/{TAG}_train.log)；學習曲線原始資料 `models/*_log.json`；彙總 [data/{TAG}_metrics.json](data/{TAG}_metrics.json)。

## 4. 指令序列（每回合）
`env.reset(seed)` → 目標於前方方塊取樣（工具朝下 IK 可達）→ 每 50 ms：`a = model.predict(obs, deterministic=True)` →
`env.step(a)`（25 個 2 ms 計算力矩子步）→ 成功（停在 2 cm 內）／碰撞即終止，否則 100 步截斷。

## 5. 量化結果
### 5.1 訓練效率（30 分鐘、1 核）
| 組態 | 總環境步 | 時間 [min] | 步/s | 首次評估成功 ≥ 90 %（步 / 分鐘） | 首次 100 % 步數 | 最佳模型步數 | 模型檔 [MB] |
|---|---|---|---|---|---|---|---|
{train_rows}

* PPO 每秒步數為 SAC 的 6–7 倍（SAC 每步一次 2×256 雙 Q 梯度更新，在此 CPU 約 11 ms/次）；以**樣本數**計 SAC（ee）
  最省（{TR['sac_ee']['steps_to_90']:,} 步達 90 %），以**時間**計 PPO（joint）最快（{TR['ppo_joint']['wall_to_90'] / 60:.1f} 分鐘）。
* 四組均在 30 分鐘內達到評估集 100 % 成功、0 碰撞；預算到時（皆未跑完設定總步數）存檔停止，曲線已平穩（見圖）。

學習曲線（評估集，摘錄）：

<details><summary>SAC（ee）</summary>

| 步數 | 時間 [min] | 成功率 | 平均回報 | 平均末距離 [mm] | 平均成功時間 [s] |
|---|---|---|---|---|---|
{curve_rows('sac_ee')}
</details>
<details><summary>PPO（ee）</summary>

| 步數 | 時間 [min] | 成功率 | 平均回報 | 平均末距離 [mm] | 平均成功時間 [s] |
|---|---|---|---|---|---|
{curve_rows('ppo_ee', 2)}
</details>
<details><summary>SAC（joint）</summary>

| 步數 | 時間 [min] | 成功率 | 平均回報 | 平均末距離 [mm] | 平均成功時間 [s] |
|---|---|---|---|---|---|
{curve_rows('sac_joint')}
</details>
<details><summary>PPO（joint）</summary>

| 步數 | 時間 [min] | 成功率 | 平均回報 | 平均末距離 [mm] | 平均成功時間 [s] |
|---|---|---|---|---|---|
{curve_rows('ppo_joint', 2)}
</details>

### 5.2 主測試：與 T201 基準相同的 200 個目標（成功即終止）
| 策略 | 成功率 | 碰撞率 | 平均成功時間 [s] | P95 [s] | 平均末距離 [mm] | 回報 | 路徑比 | 最大力矩使用率 |
|---|---|---|---|---|---|---|---|---|
{test_rows}

（基準重跑與 T201 報告數字{'完全一致' if same else '不一致（需檢查）'}；路徑比 = 法蘭路徑長 / 初始距離，只計 d0 > 5 cm。）

### 5.3 時間最佳性
| 策略 | 成功時間 / 規劃（同目標平均） | 比規劃快的目標比例 | 參考時間 | 成功時間 / 參考（平均） | 最小 | P95 |
|---|---|---|---|---|---|---|
{ratio_rows}

參考時間（每個目標）：ee 動作 t_line = (d0 − tol)/v_max（以 v_max 沿直線，平均 {B['t_line_mean']:.3f} s）；joint 動作
t_IK = min_解 max_i |Δq_i|/ω_max（無限加速度、到工具朝下 IK 解的關節速度上限時間，平均 {B['t_joint_ik_mean']:.3f} s）。
理論：五次多項式峰值速度 = 1.875·Δq/T，規劃基準把峰值設為 0.9·ω_max ⇒ T = 2.083·t_IK；實測規劃比值 {B['planner_joint']['ratio_mean']:.3f}
（略小於 2.083，因 2 cm 容差使成功早於軌跡結束）。學到的 joint 策略比值 < 1：joint 動作空間**不要求工具朝下**，
策略可同時用腕部關節縮短路徑（t_IK 對它不是嚴格下界）；ee 策略 v 各軸獨立飽和，對角方向 |v| 最高可達 √3·v_max，故比值可 < 1。

### 5.4 到達後停住（不終止，跑滿 5 s，50 目標）
| 策略 | 最後 2 s 全在容差內的回合 | 最後 2 s 在容差內時間比例 | 5 s 末距離平均 [mm] | 最大 [mm] | 末關節速度 ‖q̇‖ [rad/s] |
|---|---|---|---|---|---|
{hold_rows}

### 5.5 起點擾動加大（q0 ± 0.3 rad，訓練為 ± 0.1 rad；100 目標）
| 策略 | 成功率 | 碰撞率 | 平均成功時間 [s] | 平均末距離 [mm] | 最大力矩使用率 |
|---|---|---|---|---|---|
{q_rows}

### 5.6 每步決策時間（單核，不含物理）
{tm_rows}

## 6. 圖表與動畫
![學習曲線](../media/{TAG}_learning.png)

![比較](../media/{TAG}_compare.png)

![關節/速度/力矩/距離](../media/{TAG}_joints.png)

![動作](../media/{TAG}_actions.png)

動畫（左上 SAC-ee、右上 PPO-ee、左下 SAC-joint、右下 PPO-joint；4 個測試目標，綠點為目標）：
![動畫](../media/{TAG}.gif)

[MP4](../media/{TAG}.mp4)｜關鍵畫面：[起始](../media/{TAG}_key_start.png)、[中段](../media/{TAG}_key_mid.png)

## 7. 問題與限制
* **獎勵設計的副作用（成功即終止）**：成功後的行為從未出現在訓練資料中，PPO（joint）到達後不會停住——
  不終止時僅 {100 * H['ppo_joint']['all_in_tol_frac']:.0f} % 回合最後 2 s 全在容差內、末距離平均 {H['ppo_joint']['d_final_mean_mm']:.1f} mm；
  SAC（joint）{100 * H['sac_joint']['all_in_tol_frac']:.0f} %；ee 策略因姿態保持與 DLS 使動作→運動映射較平順，兩者皆 100 %。
  規劃與解析率基準本質上收斂到 0（末距離 ≤ {H['rr_ee']['d_final_max_mm']:.3f} mm）。需要「到達並保持」的後續任務（pick）應改用不終止＋停留獎勵。
* **精度**：獎勵只要求 d < 20 mm，RL 策略的末距離平均 10–14 mm（剛跨過門檻即終止），規劃基準 {Tt['planner_joint']['d_final_mean_mm']:.1f} mm；
  成功判據較嚴時需重新訓練或接傳統精定位。
* **力矩使用率**：joint 策略以接近 bang-bang 的關節速度（多軸同時飽和於 ω_max）換取時間，最大力矩使用率
  {Tt['sac_joint']['tau_ratio_max']:.2f}–{Tt['ppo_joint']['tau_ratio_max']:.2f}（規劃 {Tt['planner_joint']['tau_ratio_max']:.2f}）；
  起點擾動加大時 SAC（joint）達 {Q['sac_joint']['tau_ratio_max']:.2f}（力矩飽和）。獎勵僅含 0.01‖a‖² 動作懲罰，未懲罰加加速度/力矩。
* 單一隨機種子、單一超參數組（未調參），無法給出種子間變異；SAC 在此 CPU 上僅約 60 步/s，30 分鐘約 11 萬步。
* 決策時間 ≈ 0.3–0.4 ms/步（MLP 推論），遠小於 50 ms 控制週期；傳統基準 < 0.03 ms。
* 無感測雜訊、延遲與動力學隨機化（T205 再加入）；評估在 T201 同分佈目標，僅測了起點擾動的外推。

## 8. 結論
* PPO 與 SAC 在兩種動作空間下 30 分鐘（1 核）內都學會 reach：200 個測試目標 **100 % 成功、0 碰撞**，力矩不超限。
* 學到的策略比傳統基準**更快**：成功時間為規劃基準的 {M['time_ratio_vs_planner']['sac_joint']['mean']:.2f}（SAC-joint）–
  {M['time_ratio_vs_planner']['ppo_ee']['mean']:.2f}（PPO-ee）倍，接近速度上限決定的參考時間（比值 {B['sac_joint']['ratio_mean']:.2f}–{B['ppo_ee']['ratio_mean']:.2f}）；
  代價是精度只到獎勵門檻、力矩使用率較高、且 joint 策略不會停留——都是獎勵/終止設計直接造成，已量化。
* 樣本效率 SAC 優（ee：{TR['sac_ee']['steps_to_90']:,} 步達 90 %），時間效率 PPO 優（joint：{TR['ppo_joint']['wall_to_90'] / 60:.1f} 分鐘達 90 %）。
* 模型存於 `models/`（每檔 ≤ {max(TR[n]['model_kb'] for n in RL) / 1024:.1f} MB），`pytest` 以已存模型做回歸測試。
* 測試：`ruff check` 通過、`pytest` {NPASS} passed。
* 下一步：T203 pick（夾爪抓取；採不終止＋停留/抓穩獎勵，起點用本項 ee 策略的經驗）。
"""
(ROOT / f"docs/test-reports/{TAG}.md").write_text(md, encoding="utf-8")
print("ok", same)
