"""由 docs/test-reports/data/2026-11-24_T203_metrics.json 與 models/*_pick_*_log.json 產生 T203 報告（數字皆來自實際執行）。

PYTHONPATH=. python3 scripts/t203_write_md.py <pytest 通過數>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "2026-11-24_T203"
M = json.loads((ROOT / f"docs/test-reports/data/{TAG}_metrics.json").read_text(encoding="utf-8"))
NPASS = sys.argv[1] if len(sys.argv) > 1 else "?"
RL = ["sac_pick_s0", "sac_pick_s1", "ppo_pick_s0", "ppo_pick_s1"]
ALL = RL + ["scripted", "random"]
NAMES = {"sac_pick_s0": "SAC 種子 0", "sac_pick_s1": "SAC 種子 1", "ppo_pick_s0": "PPO 種子 0",
         "ppo_pick_s1": "PPO 種子 1", "scripted": "腳本式 FSM 基準", "random": "隨機策略"}
BEST = M["best_model"]


def f(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def pct(x, n=0):
    return "—" if x is None else f"{100 * x:.{n}f} %"


def steps(x):
    return "—" if x is None else f"{x:,}"


TR, Tt = M["train"], M["test"]
train_rows = "\n".join(
    f"| {NAMES[n]} | {TR[n]['timesteps']:,} | {TR[n]['wall_s'] / 60:.1f} | {TR[n]['fps']:.0f} | "
    f"{steps(TR[n]['steps_to_50'])} | {steps(TR[n]['steps_to_90'])} | "
    f"{pct(TR[n]['best']['success_rate'])}（{TR[n]['best']['timesteps']:,} 步） | {pct(TR[n]['final']['success_rate'])} | "
    f"{TR[n]['model_kb'] / 1024:.2f} |"
    for n in RL
)
test_rows = "\n".join(
    f"| {NAMES[n]} | {pct(Tt[n]['success_rate'], 1)} | {pct(Tt[n]['ever_success'], 1)} | {pct(Tt[n]['grasp_rate'], 1)} | "
    f"{pct(Tt[n]['collision_rate'], 1)} / {pct(Tt[n]['drop_rate'], 1)} | {f(Tt[n]['t_grasp_mean'])} | "
    f"{f(Tt[n]['t_success_mean'])} | {f(Tt[n]['t_success_p95'])} | {pct(Tt[n]['hold_frac_mean'], 1)} | "
    f"{f(Tt[n]['regrasp_mean'])} | {f(Tt[n]['h_final_mean_mm'], 1)} | {f(Tt[n]['return_mean'], 1)} | "
    f"{f(Tt[n]['tau_ratio_max'])} |"
    for n in ALL
)
VAR = [("q0wide", "起點擾動 ±0.3 rad"), ("heavy", "1 kg 方塊"), ("block", "40×60×80 mm 長方塊"), ("long", "10 s 回合")]
var_rows = "\n".join(
    f"| {NAMES[n]} | {pct(Tt[n]['success_rate'])} | "
    + " | ".join(f"{pct(M[k][n]['success_rate'])}（掉落 {pct(M[k][n]['drop_rate'])}）" for k, _ in VAR)
    + f" | {pct(M['long'][n]['hold_late_all'])} | {f(M['long'][n]['h_late_std_mm_mean'], 2)} |"
    for n in RL + ["scripted"]
)
fs = M["fsweep"]
fs_rows = "\n".join(f"| {s['ratio']:.2f} | {s['f_grip']:.2f} | {pct(s['success_rate'])} | {s['h_final_mm']:.1f} |"
                    for s in fs["sweep"])
tm_rows = "\n".join(f"| {NAMES[n]} | {M['timing'][n]['mean_ms']:.3f} | {M['timing'][n]['p95_ms']:.3f} | "
                    f"{M['timing'][n]['env_step_ms']:.2f} |" for n in RL + ["scripted"])
ratio_rows = "\n".join(f"| {NAMES[n]} | {f(M['time_ratio_vs_scripted'][n]['mean'], 3)} | "
                       f"{pct(M['time_ratio_vs_scripted'][n]['faster_frac'])} |" for n in RL)
hp_s, hp_p = TR["sac_pick_s0"]["hparams"], TR["ppo_pick_s0"]["hparams"]
EXTRA = (ROOT / "docs/test-reports/data" / f"{TAG}_notes.md")
notes = EXTRA.read_text(encoding="utf-8") if EXTRA.exists() else ""

md = f"""# T203 pick：夾爪抓取環境與 PPO/SAC 訓練（{TAG[:10]}）

## 目的
Phase 3 第三項：把 T201/T202 的 reach 擴充為「抓起物件」——手臂＋T005 平行夾爪須移到桌上隨機位置的方塊、對中、閉合夾爪、
抬高 10 cm 並**保持到回合結束**。依 T202 的發現（成功即終止使「到達後」行為未受訓練），本環境成功不終止，以回合最後一步
仍抓住並維持高度作為成功判據。比較 SB3 PPO/SAC（各 2 個種子）與傳統教導式有限狀態機（FSM）基準，並驗證夾持物理
（法向力 ≈ 設定力、摩擦極限 F_min = mg/(2μ)）。

## 模型與參數
* 環境 `sixaxis/rlpick.py`：`PickEnv`（`gym.make("SixAxisPick-v0")`，需先 `import sixaxis.rlpick`）。
  PyBullet DIRECT，物理步長 2 ms、50 次求解迭代，策略 20 Hz（25 子步），回合 100 步 = 5 s。
* 機構：UR5 類手臂（T001）＋T005 平行夾爪（同一 multibody、私有暫存 URDF，可多行程平行）；工作台頂面 z = 0.10 m；
  方塊 50 mm、0.2 kg、μ_obj = 0.8（指墊 1.0，PyBullet 有效 μ = 0.8）；物件中心於 x ∈ [0.35, 0.60]、y ∈ [−0.20, 0.20] m 均勻隨機（yaw = 0）。
* 動作 4 維：TCP 線速度 v = a·0.3 m/s（工具朝下姿態保持 ω = 2·log(R_d Rᵀ)、阻尼最小平方 → 關節速度，|q̇| ≤ 1.5 rad/s 等比縮放）
  → q_ref 積分 → 計算力矩內環（含夾爪/手指 8 自由度 PyBullet 逆動力學，ω_n = 25 rad/s 臨界阻尼）；
  a₄ > 0 → 夾爪以力上限 F_grip = 40 N 閉合，否則張開。
* 觀測 30 維：p_tcp、ṗ_tcp、p_obj、p_obj − p_tcp、物件四元數、ṗ_obj、指開度、左右指墊接觸旗標、夾爪指令、q/π、抬升高度 h。
* 獎勵（分段 dense，依 robosuite Lift，Zhu et al. 2020）：
  r = (1 − tanh 10d) + 0.5·[grasped] + 2·[grasped]·min(h/0.10, 1) + 2·[success] − 0.01‖a_xyz‖² − 10·[碰撞] − 10·[掉落]；
  grasped = 兩指墊法向力皆 > 0.5 N 且夾爪指令閉合；success = grasped 且 h ≥ 0.10 m。
* 終止：手臂連桿 link1–6 撞地/撞台、物件掉出台面（z < 台面 − 3 cm）。成功**不**終止。
* 演算法（`sixaxis/rltrain.py` 加入 `task="pick"`）：SAC（lr {hp_s['learning_rate']}、batch {hp_s['batch_size']}、γ {hp_s['gamma']}、
  緩衝 {hp_s['buffer_size']:,}、自動熵溫度、[256, 256]）與 PPO（n_steps {hp_p['n_steps']}、batch {hp_p['batch_size']}、
  epochs {hp_p['n_epochs']}、γ {hp_p['gamma']}、GAE λ {hp_p['gae_lambda']}、clip {hp_p['clip_range']}、log σ₀ = −1、[256, 256]），
  種子 0/1；每組 1 核、wall-time 預算 35 分鐘；學習曲線評估集 = 物件種子 5000–5049（確定性）。
* 傳統基準 `ScriptedPickPolicy`：APPROACH（物件上方 8 cm）→ DESCEND（到物件中心）→ CLOSE（停住 6 步 = 0.3 s）→ LIFT（抬到 h = 12 cm），
  各段 resolved-rate 比例控制 v = 4(p_t − p)、|v| ≤ 0.3 m/s，讀真實物件位置（完美感知）。

## 可重現指令
```bash
PYTHONPATH=. python3 scripts/t203_train.py            # 4 組各 2100 s（SAC/PPO × 種子 0/1）
PYTHONPATH=. python3 scripts/t203_train.py --resume   # 由 models/<name>_last.zip 接續
PYTHONPATH=. python3 scripts/t203_report.py           # 評估＋圖＋動畫（4 行程，約 {M['runtime_s'] / 60:.0f} 分鐘＋動畫）
PYTHONPATH=. python3 scripts/t203_write_md.py {NPASS}
ruff check . && python3 -m pytest -q
```

## 指令序列（每回合）
1. `reset(seed)`：重建方塊於隨機位置（清除接觸快取，相同種子＋動作逐位元重現）、手臂起點 = 工具朝下 TCP (0.45, 0, 0.32) 的 IK 解 ± 0.1 rad、夾爪全開。
2. 每 50 ms：策略輸出 (v_x, v_y, v_z, grip) → 25 個 2 ms 物理子步（計算力矩＋夾爪馬達）。
3. 量測 grasped／h／success／碰撞／掉落，跑滿 100 步；回合成功 = 最後一步 success。

## 量化結果

### 訓練
| 組態 | 總步數 | 時間 [min] | 步/s | 評估成功 ≥ 50 % 步數 | ≥ 90 % 步數 | 最佳評估成功（步數） | 最後評估成功 | 最佳模型 [MB] |
|---|---|---|---|---|---|---|---|---|
{train_rows}

最佳模型（評估集成功率→回報字典序）：**{NAMES[BEST]}**（`models/{BEST}_best.zip`，記錄於 `models/T203_best.txt`）。

![學習曲線](../media/{TAG}_learning.png)

### 200 個測試物件位置（種子 1000–1199，確定性策略）
| 方法 | 回合成功 | 曾成功 | 曾夾住 | 碰撞 / 掉落 | 首次夾住 [s] | 首次成功 [s] | 成功 P95 [s] | 保持比例 | 重夾次數 | 末抬升 h [mm] | 回報 | 最大力矩使用率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{test_rows}

同物件位置成功時間比（策略 / 腳本基準，兩者皆成功者）：

| 方法 | 平均時間比 | 比基準快的比例 |
|---|---|---|
{ratio_rows}

![比較](../media/{TAG}_compare.png)
![物件位置成功圖](../media/{TAG}_map.png)

### 穩健性與泛化（各 100 回合；10 s 回合 50 回合）
| 方法 | 名目 | {' | '.join(lab for _, lab in VAR)} | 10 s 回合 5–10 s 全程保持 | 5–10 s 高度標準差 [mm] |
|---|---|---|---|---|---|---|---|
{var_rows}

### 夾持力–摩擦極限（腳本基準，各 10 回合）
理論：兩指 Coulomb 摩擦保持 2μF ≥ m(g + a) ⇒ F_min = mg/(2μ) = {fs['f_min']:.3f} N（m = 0.2 kg、μ = 0.8；T005）。

| F_grip / F_min | F_grip [N] | 成功 | 平均末抬升 h [mm] |
|---|---|---|---|
{fs_rows}

### 決策時間（每步，單核）
| 方法 | 策略 [ms] | P95 [ms] | 環境一步（25 子步）[ms] |
|---|---|---|---|
{tm_rows}

### 單回合曲線（種子 1003）
![回合曲線](../media/{TAG}_episode.png)
![關節曲線](../media/{TAG}_joints.png)

## 動畫
上至下：{NAMES[BEST]}、另一演算法最佳種子、腳本基準；左為全景、右為沿 x 方向的近景側視（物件種子 1003、1011、1027）。

![動畫](../media/{TAG}.gif)

[MP4](../media/{TAG}.mp4)｜關鍵畫面：[開始](../media/{TAG}_key_start.png)、[夾取](../media/{TAG}_key_grasp.png)、[抬升](../media/{TAG}_key_lift.png)

{notes}

## 測試
`tests/test_rlpick.py` 12 項：`check_env`／`gym.make`、觀測與 FK 一致、決定性、零動作保持與姿態保持衰減 θ_n = θ₀(1 − k_RΔt)ⁿ、
TCP 速度追蹤、腳本基準成功、夾持法向力 ≈ F_grip 且抬升不滑、夾持力 < F_min 抬不起、物件掉落終止、隨機策略失敗、
`train(task="pick")` 短跑存檔、已存最佳 pick 模型在 20 個新位置 ≥ 80 % 成功且無碰撞。全套 **{NPASS} passed**、`ruff check` 通過。
"""

(ROOT / f"docs/test-reports/{TAG}.md").write_text(md, encoding="utf-8")
print("written", len(md))
