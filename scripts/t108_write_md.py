"""T108 報告 md：由 docs/test-reports/data/<date>_T108_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t108_write_md.py 2026-11-16 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T108"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
PAD, FP, FIN, RA = M["pad"], M["force"], M["finish"], M["raster"]
LM = M["loop_nominal"]


def f(x, fmt=".2f"):
    return "—" if x is None else format(x, fmt)


def res(r):
    return "OK" if r.get("ok") else f"NG：{r.get('error')}"


MODE = {"admittance": "導納力控", "position": "位置控制"}
P = M["polish"]
rowsP = "\n".join(
    f"| {r['label']} | {MODE[r['mode']]} | {res(r)} | {f(r.get('F_mean'))} | {f(r.get('F_rms_err'))} | "
    f"{f(r.get('F_max_err'))} | {f(r.get('F_min'))}–{f(r.get('F_max'), '.1f')} | {f(r.get('loss_frac', 0) * 100, '.1f')} | "
    f"{f(r.get('land_overshoot', 0) * 100, '.1f')} | {f(r['rem_mean_um'])} | {f(r['rem_cv'] * 100, '.1f')} | "
    f"{f(r['rem_min_um'])}–{f(r['rem_max_um'])} | {f(r.get('lat_max_mm'), '.3f')} | {f(r.get('torque_util'), '.3f')} | "
    f"{f(r.get('cycle_s'), '.1f')} |" for r in P)
nomA = next(r for r in P if r["scenario"] == "nominal" and r["mode"] == "admittance")
adm = [r for r in P if r["mode"] == "admittance"]
pos = [r for r in P if r["mode"] == "position"]
RP = M["raster_pred"]

TD = M["touchdown"]
rowsT = "\n".join(
    f"| {r['k_e']:.0e} | {r['D']:.0f} | {r['zeta_ideal']:.2f} | {r['routh_kcrit']:.2e} | "
    f"{r['pred_rigid_os'] * 100:.1f}{'（振盪）' if r['pred_rigid_osc'] > 0.5 else ''} | "
    f"{r['pred_servo_os'] * 100:.1f}{'（振盪）' if r['pred_servo_osc'] > 0.5 else ''} | "
    f"{r['pred_servo1ms_os'] * 100:.1f}{'（振盪）' if r['pred_servo1ms_osc'] > 0.5 else ''} | "
    f"{r['sim_os'] * 100:.1f} | {r['sim_tail_std']:.2f} | {f(r['sim_land_s'])} | {r['error'] or 'OK'} |" for r in TD)
stable = [r for r in TD if r["error"] is None]
dev = max(abs(r["sim_os"] - r["pred_servo_os"]) for r in stable)

SL = M["slope"]
rowsS = "\n".join(
    f"| {r['slope_deg']:.1f} | {r['K_i']:.0f} | {r['e_pred']:.3f} | {r['e_sim']:+.3f} | "
    f"{(r['e_sim'] / r['e_pred'] - 1) * 100 if r['K_i'] == 0 else r['e_sim'] / r['e_pred'] * 100:+.1f} % | "
    f"{r['F_rms_err']:.3f} |" for r in SL)

SP = M["speed"]
rowsV = "\n".join(
    f"| {r['v'] * 1e3:.0f} | {r['F_mean']:.2f} | {r['h0_sim_um']:.3f} | {r['h0_pred_um']:.3f} | "
    f"{(r['h0_sim_um'] / r['h0_pred_um'] - 1) * 100:+.2f} | {r['hmax_sim_um']:.3f} | {r['hmax_pred_um']:.3f} | "
    f"{r['prof_max_err_pct']:.2f} |" for r in SP)

FF = M["ff_noise"]
rowsF = "\n".join(
    f"| {'有' if r['force_ff'] else '無'} | {r['sigma_F']:.1f} | {f(r['F_rms_err'], '.3f')} | {f(r['F_max_err'], '.3f')} | "
    f"{f(r['land_overshoot'] * 100, '.1f')} | {f(r['lat_max_mm'], '.4f')} | {r['error'] or 'OK'} |" for r in FF)

DB = M["deburr"]
rowsD = []
for d in DB:
    for p in d["passes"]:
        rowsD.append(
            f"| {d['label']} | {p['pass']} | {p['burr_max_um']:.1f} | {p['burr_mean_um']:.1f} | {p['gouge_um']:.1f} | "
            f"{f(p.get('F_rms_err'), '.3f')} | {f(p.get('F_max'), '.1f')} | {f(p.get('torque_util'), '.3f')} |")
rowsD = "\n".join(rowsD)
sumD = "\n".join(
    f"| {d['label']} | {'達成' if d['done'] else '未達成'} | {len(d['passes']) - 1} | {d['passes'][0]['burr_max_um']:.0f} → "
    f"{d['passes'][-1]['burr_max_um']:.1f} | {d['passes'][-1]['gouge_um']:.1f} | "
    f"{(d['passes'][0]['burr_max_um'] - d['passes'][-1]['burr_max_um']) / max(d['passes'][-1]['gouge_um'], 1e-9):.2f} | "
    f"{d['time_s']:.1f} |" for d in DB)
# 硬墊毛邊衰減解析：δ_base ≪ h_b 時毛邊承受 p ≈ k_w·h_b → 每道 h ← h·exp(−K_p·k_w·ω/v·I(y))，
# I(y) = ∫√(x²+y²)dx（弦長 ±c）= c·R + y²·ln((R+c)/y)，y = 墊心偏離稜線距離
import math  # noqa: E402

R_, Y_, V_ = PAD["radius"], 0.0075, 0.010
c_ = math.sqrt(R_**2 - Y_**2)
I_ = c_ * R_ + Y_**2 * math.log((R_ + c_) / Y_)


def rho_pred(ke):
    return math.exp(-PAD["k_p"] * ke / (math.pi * R_**2) * PAD["omega"] / V_ * I_)


rowsR = []
for d in DB:
    hs = [p["burr_max_um"] for p in d["passes"]]
    rat = [hs[k + 1] / hs[k] for k in range(1, len(hs) - 1)]
    rp = rho_pred(d["k_e"])
    n_need = math.ceil(math.log(M["deburr_tol_um"] / hs[1]) / math.log(rp)) + 1 if rp < 1 else None
    rowsR.append(f"| {d['label']} | {rp:.4f} | {min(rat):.4f}–{max(rat):.4f} | {n_need if n_need else '—'} | "
                 f"{len(d['passes']) - 1} |")
rowsR = "\n".join(rowsR)
DM = M["demo"]
G = "https://github.com/dofliu/6axisRobot/blob/main/docs/media"

md = f"""# T108 去毛邊／拋光（力控、接觸力維持）測試報告

日期：{D}　狀態：完成　模組：`sixaxis/finishing.py`　測試：`tests/test_finishing.py`

## 1. 目的
以六軸手臂＋力控研磨主軸執行**拋光**（光柵掃描、材料均勻移除）與**去毛邊**（沿稜邊多道次移除毛邊），
驗證：(1) 法向接觸力在工件形狀未知誤差下的維持能力（導納力控 vs 純位置控制）；(2) 力控外環的觸地暫態與
穩定極限，並以含伺服柔度的集中參數模型預測；(3) 斜面的穩態力誤差 D·v·tanα 與積分項；
(4) Preston 移除模型的單道剖面、深度 ∝ 1/v 與多道均勻度；(5) 硬墊／軟墊去毛邊的選擇性與母材過切。

## 2. 模型與參數
### 2.1 工具
`robots/ur5_polisher.urdf`：法蘭 → 六軸力感測器 → 主軸馬達 → 主軸鼻 → 背板 → 研磨墊（半徑 {PAD['radius'] * 1e3:.1f} mm），
法蘭到墊面 0.195 m、質量 2.2 kg。TCP = 未壓縮墊面中心，z 軸朝工件。研磨墊不設碰撞幾何，接觸力由解析模型計算後以外力/外力矩
施加於工具連桿（避免 PyBullet 剛體接觸重複計力），其餘工具外殼仍參與碰撞檢查。

### 2.2 接觸、摩擦與材料移除（公式見 `sixaxis/finishing.py` 模組說明）
* **Winkler 彈性基礎**（Johnson, *Contact Mechanics* §5.9）：工件頂面以高度場 H(u,w) 表示（拋光 0.5 mm、去毛邊 0.25 mm 格距），
  墊面足印內 p = k_w·max(H − h_p, 0)，k_w = k_e/(πR²)；名目 k_e = {PAD['k_e']:.0e} N/m（k_w = {PAD['k_w']:.3e} N/m³）。
* **Coulomb 摩擦**：μ = {PAD['mu']}，每格依墊上該點相對速度 v_TCP + ω_s·a × r 計算；平面均壓時合力 ≈ 0、合力矩 = (2/3)μF_nR。
* **Preston 方程**（Preston 1927）：∂H/∂t = −K_p·p·|v|，K_p = {PAD['k_p']:.0e} m²/N（代表值），主軸 {PAD['rpm']:.0f} rpm
  （ω = {PAD['omega']:.1f} rad/s）。平面單道剖面解析式 h(w) = K_p·p0·ω/v·(cR + w²·ln((R+c)/|w|))，c = √(R²−w²)；
  中心 h(0) = K_p·F·ω/(πv)。
* **毛邊**：工件頂面沿 u 的窄稜（寬 1 mm、高 0.13–0.37 mm 的雙正弦剖面、長 100 mm）；殘留毛邊高 = 稜頂 − 兩側 1 mm 內母材高。

### 2.3 力控
* **導納外環**（Hogan 1985；Siciliano §9.4）：M·z̈ + D·ż = (F_d − F_m) + K_i∫(F_d − F_m)，|ż| ≤ v_app；
  F_d = {FP['F_d']:.0f} N、M = {FP['M']:.0f} kg、D = {FP['D']:.0f} N·s/m、v_app = {FP['v_app'] * 1e3:.0f} mm/s、起始高 {FP['z_app'] * 1e3:.0f} mm。
* **感測**：F/T 讀值 = 真實接觸力螺旋＋法向高斯雜訊 σ = {FP['sigma_F']} N，一階低通 f_c = {FP['f_c']:.0f} Hz。
* **內環**：名目路徑（plan_spray 等速時間律、姿態固定）關節軌跡 + Δq = J⁻¹[−n·z; 0]（含 ż、z̈ 前饋），計算力矩控制 ω_n = 60 rad/s；
  力前饋 τ_ff = −J_TCPᵀ·w_m。
* **理論**：理想內環時 ω_a = √(k_e/M) = {LM['omega_a']:.0f} rad/s、ζ = {LM['zeta']:.3f}；斜坡穩態誤差 e = D·ż_s；
  Routh（剛性內環＋感測低通）臨界剛性 k_crit = D(M + Dτ)/(Mτ) = {M['routh_kcrit_nominal']:.2e} N/m。
  **伺服柔度模型**：計算力矩對外力的誤差動態 Λ_n(ë + 2ω_nė + ω_n²e) = −F + F_m，Λ_n = 1/(nᵀJ_vM_q⁻¹J_vᵀn) = {M['lambda_n']:.2f} kg
  （伺服剛性 Λ_nω_n² = {M['lambda_n'] * 3600:.0f} N/m），實際位置 x = z + e。
* **位置控制基準**：沿名目面固定壓入 δ_0 = F_d/k_e = {FP['F_d'] / PAD['k_e'] * 1e3:.2f} mm，無力回授。

### 2.4 工作與路徑
工件 160×100 mm（頂面 z = 0.15 m，中心 (0.45, 0, 0.15)），拋光光柵 {RA['n']} 道 × {RA['length'] * 1e3:.0f} mm、道距 {RA['spacing'] * 1e3:.0f} mm、
180° 圓弧換道（全程接觸）、進給 {FIN['speed'] * 1e3:.0f} mm/s；量測區 |u| ≤ {RA['u_in'] * 1e3:.0f} mm、|w| ≤ {RA['w_in'] * 1e3:.0f} mm。
流程：關節 PTP 到起點上方 {FIN['approach'] * 1e3:.0f} mm → 下降到 z_app → 導納觸地（|F − F_d| ≤ 5 % 持續 0.15 s）→ 沿路徑力控 → 抬起退刀。
錯誤碼：UNREACHABLE、LIMIT、NO_CONTACT（觸地 2 s 內未穩定）、OVERFORCE（> 3F_d 或 > 150 N）。

## 3. 可重現指令
```bash
python3 -m pip install -r requirements.txt   # pybullet 需 --no-build-isolation，見 CLAUDE.md
ruff check . && python3 -m pytest -q          # 全套測試
PYTHONPATH=. python3 scripts/t108_report.py {D}     # 實驗、圖、動畫 → docs/test-reports/data/{T}_metrics.json
python3 scripts/t108_write_md.py {D} "<pytest 摘要>"  # 本報告
```
報告腳本與 pytest 不可同時執行（都會重寫 `robots/ur5_polisher.urdf`）。

## 4. 指令序列（拋光一個工件）
```python
c = FinishCell()
s = Surface((0.45, 0, 0.15), size=(0.16, 0.10)).set_shape(wave_amp=3e-4, wave_len=0.04)  # 形狀誤差規劃端未知
c.add_surface(s)
seam, _ = raster_on(s, 0.010, 6, 0.100)
m = c.finish(seam, FinishParams(speed=0.025, force=ForceParams(F_d=20, M=2, D=600)))
removal_stats(s, 0.035, 0.020)
```

## 5. 結果
### 5.1 拋光：表面形狀誤差 × 控制模式（光柵 6 道）
| 情境 | 控制 | 結果 | F 平均 [N] | F RMS 誤差 [N] | 最大誤差 [N] | F 範圍 [N] | 脫離 % | 觸地超越 % | 移除平均 [µm] | 移除 CV % | 移除範圍 [µm] | 橫向偏差 [mm] | 力矩使用率 | 週期 [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{rowsP}

解析預測（單道剖面疊加，忽略換道圓弧）：內部區平均 {RP['mean_um']:.2f} µm、CV {RP['cv'] * 100:.1f} %；名目面導納模擬 {nomA['rem_mean_um']:.2f} µm、
CV {nomA['rem_cv'] * 100:.1f} %。導納力控 5 種情境 F RMS 誤差 {min(r['F_rms_err'] for r in adm):.2f}–{max(r['F_rms_err'] for r in adm):.2f} N、
移除 CV {min(r['rem_cv'] for r in adm) * 100:.1f}–{max(r['rem_cv'] for r in adm) * 100:.1f} %；位置控制 F RMS 誤差
{min(r['F_rms_err'] for r in pos):.2f}–{max(r['F_rms_err'] for r in pos):.2f} N。

![移除深度](../media/{T}_removal.png)
![情境力曲線](../media/{T}_scenarios.png)

### 5.2 觸地暫態與穩定極限（40 mm 單道）
| k_e [N/m] | D [N·s/m] | 理想 ζ | Routh k_crit | 剛性內環＋低通 預測超越 % | 伺服柔度模型 % | 伺服柔度＋1 ms 離散 % | 模擬超越 % | 模擬末段 σ_F [N] | 觸地時間 [s] | 結果 |
|---|---|---|---|---|---|---|---|---|---|---|
{rowsT}

穩定的情境中，模擬超越量與伺服柔度模型（連續時間）最大差 {dev * 100:.1f} 個百分點。

![觸地](../media/{T}_touchdown.png)

### 5.3 斜面穩態力誤差（60 mm 單道上坡、v = {FIN['speed'] * 1e3:.0f} mm/s）
| 坡度 [°] | K_i [1/s] | 預測 D·v·tanα [N] | 模擬 F − F_d [N] | 偏差（K_i=0：相對預測；K_i>0：殘留佔預測） | F RMS 誤差 [N] |
|---|---|---|---|---|---|
{rowsS}

### 5.4 單道移除剖面與進給速度（Preston 解析比對）
| v [mm/s] | F 平均 [N] | 模擬 h(0) [µm] | 解析 h(0) [µm] | 差 % | 模擬最大 [µm] | 解析最大 [µm] | 剖面最大誤差（佔峰值）% |
|---|---|---|---|---|---|---|---|
{rowsV}

![斜面與速度](../media/{T}_slope_speed.png)

### 5.5 力前饋與感測雜訊（起伏面 0.3 mm／λ 40 mm、80 mm 單道）
| 力前饋 | σ_F [N] | F RMS 誤差 [N] | 最大誤差 [N] | 觸地超越 % | 橫向偏差 [mm] | 結果 |
|---|---|---|---|---|---|---|
{rowsF}

### 5.6 去毛邊（沿稜邊、墊心偏離稜線 7.5 mm、v = 10 mm/s，容許殘留 {M['deburr_tol_um']:.0f} µm）
| 設定 | 結果 | 道次 | 毛邊最大高 [µm] | 母材最大過切 [µm] | 選擇性（毛邊降幅／過切） | 總時間 [s] |
|---|---|---|---|---|---|---|
{sumD}

**毛邊衰減解析**：硬墊下母材壓縮 δ_b ≪ 毛邊高 h_b，毛邊壓力 ≈ k_w·h_b 與設定力無關（位移控制式移除），
每道衰減比 ρ = exp(−K_p·k_w·ω/v·I(y))，I(y) = c·R + y²·ln((R+c)/y)（y = 墊心偏離稜線 7.5 mm、c = √(R²−y²)）；
母材則承受其餘的力，過切 ∝ (F_d − 毛邊承載)。故**較小的設定力去毛邊速度相同、過切較少**。第 1 道含毛邊兩端（墊未全覆蓋）效應，
比值自第 2 道起算；「預測道次」= 由第 1 道後高度依 ρ 推到容許值所需道次。

| 設定 | 預測 ρ | 模擬逐道比 | 預測道次 | 模擬道次 |
|---|---|---|---|---|
{rowsR}

逐道：
| 設定 | 道次 | 毛邊最大 [µm] | 毛邊平均 [µm] | 母材過切 [µm] | F RMS 誤差 [N] | F 最大 [N] | 力矩使用率 |
|---|---|---|---|---|---|---|---|
{rowsD}

![去毛邊](../media/{T}_deburr.png)

### 5.7 關節軌跡（名目拋光）
![關節](../media/{T}_joints.png)

## 6. 動畫
左：全景；右：工具近景（相位、法向力、導納位移 z）。前段為起伏工件拋光（淺色軌跡＝已拋光區），後段為硬墊 20 N 去毛邊 3 道（黃色毛邊隨移除下沉，外觀高度放大 6 倍）。
示範結果：{', '.join(res(d) for d in DM)}。

![動畫]({G}/{T}.gif?raw=true)

[MP4](../media/{T}.mp4)（{M['frames']} 幀，GIF {M['gif_MB']:.1f} MB、MP4 {M['mp4_MB']:.1f} MB）

| 關鍵畫面 | |
|---|---|
| ![a](../media/{T}_key_a.png) | ![b](../media/{T}_key_b.png) |
| ![c](../media/{T}_key_c.png) | |

## 7. 問題與限制
* 接觸為 2.5D Winkler 高度場模型（非 PyBullet 剛體接觸）：忽略墊的剪切耦合、Hertz 非線性、墊磨耗與溫度；
  工具傾斜對接觸的影響未計（姿態誤差 < 0.01°）。
* K_p、μ、k_e 為代表值，非量測校正；Preston 係數依研磨材/工件材料差異可達數量級。
* 毛邊以頂面窄稜近似（2.5D），不含側向翻邊/根部撕裂；殘留量測假設完美雷射剖面量測。
* 力感測為模擬讀值＋高斯雜訊，未含溫漂、重力/慣性補償誤差；導納參數未自動依 k_e 估測調整（觸地表顯示固定 D 在硬接觸時超越大）。
* 去毛邊路徑偏心量（7.5 mm）與道次為固定設定，未最佳化；硬墊下母材仍有過切，實務需配合更硬的工具或浮動主軸。

## 8. 結論
* 導納力控在 ±0.6 mm 偏移、0.5° 傾斜、0.3 mm 起伏等規劃端未知的形狀誤差下皆維持 20 N 法向力（偏移/傾斜情境移除分佈與名目面相同）；
  純位置控制在 +0.6 mm 偏移時力增為約 50 N、−0.6 mm 完全脫離、傾斜/起伏時力在數 N 到 30 N 間變動。
  起伏面即使力控仍有較大移除 CV：墊足印（25 mm）內壓力集中在波峰，波峰移除較多（拋光的整平效果），並非力控誤差。
* 觸地暫態必須計入計算力矩伺服柔度（Λ_n ≈ {M['lambda_n']:.1f} kg、伺服剛性約 {M['lambda_n'] * 3600 / 1e3:.0f} kN/m）才能預測超越量
  （已穩定情境最大差 {dev * 100:.1f} 個百分點）；剛性內環＋低通的模型（含 Routh k_crit）在 k_e ≥ 2e5 N/m 高估超越量、
  並把 5e5／1e6 N/m、D = 600 判為振盪，而模擬仍收斂。k_e = 2e6 N/m 兩種阻尼皆 NO_CONTACT（2 s 內未穩定）：
  D = 2√(k_e·M) 時連續時間模型預測穩定、加入 1 ms 離散更新後才預測振盪，顯示此時 1 kHz 更新率是限制；
  1e6 N/m、D = 600 則相反（連續模型預測振盪、模擬收斂），模型在穩定邊界附近不可靠。
* 力前饋對法向力誤差無明顯影響（RMS 0.295 vs 0.284 N），但抑制墊摩擦對伺服的擾動：無前饋時橫向路徑偏差由 0.010 增為 1.08 mm。
* 斜面穩態誤差符合 D·v·tanα，加入積分項 K_i = 20 /s 後消除。
* Preston 單道剖面、深度 ∝ 1/v 與多道疊加皆與解析一致。
* 去毛邊：硬墊下毛邊衰減比與解析 ρ 一致且與設定力無關，10／20／40 N 道次相同而母材過切 ∝ 力 → 應以「維持接觸的最小力」去毛邊；
  軟墊壓力分散，每道毛邊僅降約 4 %，同時母材過切與毛邊降幅同量級，不適合去毛邊。

測試：{PYT}
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written", T)
