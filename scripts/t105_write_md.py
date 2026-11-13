"""T105 報告 md：由 docs/test-reports/data/<date>_T105_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t105_write_md.py 2026-11-13 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T105"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
N0, TO, PR, DF = M["nominal"], M["torch"], M["process"], M["defaults"]


def ok(b):
    return "✓" if b else "✗"


def f(x, fmt=".2f"):
    return "—" if x is None else format(x, fmt)


def um(m, key):
    return f(m[key] * 1e3, ".2f") if m.get(key) is not None else "—"


def row(tag, m):
    """共用列：結果、路徑誤差、速度、角度、使用率、間隙。"""
    if m.get("path_max_mm") is None:
        return (f"| {tag} | ✗ {m.get('error')} | — | — | — | — | — | — | — | "
                f"{f(m.get('plan_vel_ratio'))} | — | {f(m.get('precheck_clear_mm'), '.1f')} |")
    return (f"| {tag} | {ok(m['ok'])} {m.get('error') or ''} | {um(m, 'path_rms_mm')}／{um(m, 'path_max_mm')} | "
            f"{um(m, 'track_max_mm')} | {f(m['speed_mean_mm_s'], '.3f')} | {f(m['speed_dev_max_pct'], '.3f')} | "
            f"{f(m['work_err_max_deg'], '.3f')}／{f(m['travel_err_max_deg'], '.3f')} | {f(m['heading_err_max_deg'], '.2f')} | "
            f"{f(m['joint_err_max_mrad'])} | {f(m['vel_util'])} | {f(m['torque_util'])} | {f(m['clear_mm'], '.1f')} |")


HEAD = ("| {} | 結果 | 路徑誤差 RMS／最大 [µm] | 追蹤誤差最大 [µm] | 平均速度 [mm/s] | 速度偏差最大 [%] | "
        "work／travel 角誤差最大 [°] | 航向偏差最大 [°] | 關節誤差最大 [mrad] | 速度使用率 | 力矩使用率 | 最小間隙 [mm] |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|")

speeds = "\n".join(row(f"{m['speed_cmd_mm_s']:.0f}", m) for m in M["speeds"])
sp_ok = [m for m in M["speeds"] if m["ok"]]
sp_no = [m for m in M["speeds"] if not m["ok"]]
corner = "\n".join(
    f"| {m['R_mm']:.0f} | {m['blend_mm']:.0f} | {ok(m['ok'])} | {m['rate_rad_s']:.2f} | {f(m['plan_qdd_max'], '.1f')} | "
    f"{f(m['joint_err_max_mrad'])} | {f(m['speed_dev_max_pct'], '.3f')} | {um(m, 'path_max_mm')} | "
    f"{f(m['heading_err_max_deg'], '.2f')} | {f(m['heading_theory_deg'], '.2f')} | "
    f"{f(m['work_err_max_deg'], '.2f')}／{f(m['travel_err_max_deg'], '.2f')} | {f(m['vel_util'])} | {f(m['torque_util'])} |"
    for m in M["corner"])
angles = "\n".join(
    f"| {m['work_deg']:.0f} | {m['travel_deg']:+.0f} | {ok(m['ok'])} | {f(m.get('roll_deg'), '.0f')} | {f(m['sigma_min'], '.3f')} | "
    f"{um(m, 'path_max_mm')} | {f(m['speed_dev_max_pct'], '.3f')} | {f(m['work_err_max_deg'], '.3f')}／"
    f"{f(m['travel_err_max_deg'], '.3f')} | {f(m['vel_util'])} | {f(m['torque_util'])} | {f(m['clear_mm'], '.1f')} |"
    for m in M["angles"])
CASE_NAME = {"fillet": "T 形接頭填角焊（J 形、立板高 40 mm）work 45°", "circle": "管–板整圓填角焊（管徑 60 mm）work 135°",
             "incline30": "斜板 30°", "incline45": "斜板 45°", "incline60": "斜板 60°"}
cases = "\n".join(row(f"{CASE_NAME[k]}；roll {f(m.get('roll_deg'), '.0f')}°", m) for k, m in M["cases"].items())


def roll_row(k, m):
    cells = []
    for x in m["rolls"]:
        why = "可行" if x["ok"] else ("超限位" if not x["limit_ok"] else ("IK 失敗" if x["ik_fail"] else "超速"))
        cells.append(f"{x['roll_deg']:.0f}°：σ_min {x['sigma_min']:.3f}、J6 ∈ [{x['q6_min']:.2f}, {x['q6_max']:.2f}] rad、{why}")
    tried = "；".join(f"{x['roll_deg']:.0f}° → {x['error'] or '成功'}" for x in m["roll_tried"])
    return f"| {CASE_NAME[k]} | " + "<br>".join(cells) + f" | {tried} |"


rolls = "\n".join(roll_row(k, m) for k, m in M["cases"].items())
ctrl = "\n".join(row(f"{'計算力矩' if m['controller'] == 'ctc' else 'PD＋重力補償'}／{m['speed_cmd_mm_s']:.0f} mm/s", m)
                 for m in M["controllers"])
demo = "\n".join(row(f"{m['seam']}（roll {f(m.get('roll_deg'), '.0f')}°、{m['length_mm']:.0f} mm、{m['weld_s']:.1f} s）", m)
                 for m in M["demo"]["seams"])
CI, C30, C60 = M["cases"]["circle"], M["cases"]["incline30"], M["cases"]["incline60"]
FI = M["cases"]["fillet"]
c_no = [m for m in M["corner"] if m["blend_mm"] == 0]
c_b2 = [m for m in M["corner"] if m["blend_mm"] == 2]
r5 = {m["blend_mm"]: m for m in M["corner"] if m["R_mm"] == 5}
pd = {m["speed_cmd_mm_s"]: m for m in M["controllers"] if m["controller"] == "pd"}
ctc = {m["speed_cmd_mm_s"]: m for m in M["controllers"] if m["controller"] == "ctc"}
ang_ok = sum(m["ok"] for m in M["angles"])
over = [m for m in sp_ok if m["vel_util"] > 1.0]
job = '{"seams": [\n' + ",\n".join(" " + json.dumps(x, ensure_ascii=False) for x in M["demo"]["job"]["seams"]) + "\n]}"
inc60_txt = (f"60° 斜板：roll 候選依序嘗試 {'、'.join(str(int(x['roll_deg'])) + '° → ' + (x['error'] or '成功') for x in C60['roll_tried'])}"
             f"，最終{'成功' if C60['ok'] else '失敗（' + str(C60['error']) + '）'}。")

md = f"""# T105 銲接路徑（直線＋圓弧焊道、固定銲槍姿態與速度）測試報告

- 日期：{D}
- 測試：{PYT}
- 程式：`sixaxis/welding.py`、`tests/test_welding.py`、`scripts/t105_report.py`、`scripts/t105_write_md.py`
- 數據：`docs/test-reports/data/{T}_metrics.json`（本報告所有數字由該檔產生）

## 目的
讓手臂帶著銲槍沿「直線＋圓弧」組成的焊道，以**固定的銲槍姿態（work／travel 角）與固定的行進速度**完成銲接，並量化：
TCP 對焊道幾何的路徑誤差、行進速度均勻度（決定入熱量與焊道截面均勻度）、銲槍角度誤差、關節速度／力矩使用率、噴嘴與工件間隙。
另外找出這項工作的實際限制：圓角處的腕部角速度上限、直線–圓弧接點的角加速度衝擊、繞銲絲 roll 的選擇、碰撞。

## 模型與參數
### 銲槍（`robots/ur5_torch.urdf`，由 `write_arm_with_torch()` 產生）
| 項目 | 數值 |
|---|---|
| 型式 | 彎頸 MIG 銲槍，固定於法蘭；質量 {TO['TORCH_MASS']} kg |
| 槍身（沿法蘭 z）／槍頸＋噴嘴／銲絲伸出 | {TO['BODY_L'] * 1e3:.0f}／{TO['NECK_L'] * 1e3:.0f}／{TO['STICKOUT'] * 1e3:.0f} mm |
| 彎頸角 β | {TO['NECK_DEG']:.0f}°（繞法蘭 y） |
| TCP（銲絲尖端）相對法蘭 | T_TCP = Tz({TO['BODY_L'] * 1e3:.0f} mm)·Ry(β)·Tz({(TO['NECK_L'] + TO['STICKOUT']) * 1e3:.0f} mm) → 位置 ({M['tcp_offset_mm'][0]:.2f}, {M['tcp_offset_mm'][1]:.2f}, {M['tcp_offset_mm'][2]:.2f}) mm；z 軸 = 銲絲方向 |
| 噴嘴半徑 | {TO['NOZZLE_R'] * 1e3:.0f} mm（銲絲不含碰撞幾何） |

TCP 位姿以 URDF 內的 `tcp` 連桿由 PyBullet 讀回（與 `fk(q)·T_TCP` 相差 < 1e-6，見測試）。

### 焊道座標與銲槍角度（AWS A3.0 定義）
焊道在法向量 n 的平面上，以弧長 s 參數化；t = 切線（行進方向）、b = n × t（行進方向左側）。
- 工作角 work（由工件表面量起；90° = 垂直表面，< 90° 槍身倒向 +b）：u = sin(work)·n + cos(work)·b
- 行進角 travel（+ = 前進法 push、− = 後退法 drag）：銲絲方向 a = −cos(travel)·u + sin(travel)·t
- 工具 x 軸 = t 在 ⟂a 平面的投影，再繞 a 轉 roll。**銲絲為軸對稱，roll 不影響製程**，是可用來避開限位／奇異／碰撞的自由度。
- 固定姿態 = work／travel 相對 (t, n, b) 固定：直線段為世界座標固定姿態；圓弧段銲槍繞 n 以 ω = κ·v 旋轉（κ = ±1/R）。
  平面焊道 R_tcp(s) = Rot(n, ψ(s))·R_tcp(0)，ψ(s) = ∫κ ds。

### 等速時間律與接點平滑
- 弧長速度：起／收弧各以 smoothstep（3τ² − 2τ³）在 t_ramp = {DF['t_ramp']} s 內加／減速，中間等速 v；T = L/v + t_ramp，峰值加速度 1.5·v/t_ramp。
- 直線–圓弧接點曲率不連續 → 銲槍角速度階躍 κ·v。可選「航向平滑」：ψ_o(s) = (1/2b)∫ψ dσ（半寬 b 移動平均），角速度變連續，
  代價是接點處航向偏離切線 |κ|·b/4 = b/(4R)（由積分式推得）。位置路徑不變，只有姿態被平滑。
- 規劃：沿時間律每 10 ms 取樣 → 數值 IK（前一點為初值）→ q̇ = J⁻¹·(法蘭 twist) → 三次 Hermite 樣條 → 1 kHz 參考。
- 進／退刀：沿銲絲反方向 {DF['approach'] * 1e3:.0f} mm；起弧停留 {DF['t_pre']} s、收弧（填弧坑）停留 {DF['t_crater']} s。

### 製程換算（不模擬電弧物理）
- 線入熱量 Q = η·U·I/v（EN 1011-1，GMAW η = {PR['eta']}）：U = {PR['volt']:.0f} V、I = {PR['amp']:.0f} A、v = 10 mm/s → **{N0['heat_J_mm']:.1f} J/mm**
- 焊道截面積 A = (πd²/4)·WFS/v：線徑 {PR['d_wire_mm']} mm、送絲 {PR['wfs_m_min']:.0f} m/min → **{N0['bead_area_mm2']:.2f} mm²**
- 兩者皆 ∝ 1/v，所以「速度偏差 %」≈「入熱量／焊道截面偏差 %」。

### 控制與量測
- 計算力矩控制 ωn = {M['wn']:.0f} rad/s 臨界阻尼，逆動力學用 PyBullet `calculateInverseDynamics`（含銲槍質量）；對照組為 PD＋重力補償
  （ωn = 30 rad/s，增益依各軸等效慣量縮放）。力矩以 τ_max 限幅；步長 1 ms。
- 路徑誤差：實際 TCP 到**焊道幾何**最近點的偏差（側向＋高度），與時間無關；追蹤誤差：對同一時刻的參考點；速度：v = J_v q̇ + ω × r。
- 銲槍角度：由實際銲絲方向對最近焊道點的 (t, n, b) 量回 work／travel；航向偏差：工具 x 軸繞 n 相對切線航向的偏角。
- 間隙：手臂＋銲槍（不含銲絲）到工作台／工件的最小距離；規劃後先做運動學碰撞預檢（間隙 < 0.5 mm → COLLISION，不執行）。

## 可重現指令
```bash
python3 -m pip install "setuptools<70" wheel && python3 -m pip install --no-build-isolation pybullet==3.2.7
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/t105_report.py {D}
python3 scripts/t105_write_md.py {D} "<pytest 摘要>"
ruff check . && python3 -m pytest
```
注意：報告腳本與 pytest 不可同時執行（兩者都會重寫 `robots/ur5_torch.urdf`）。

## 指令序列
每條焊道（`WeldCell.weld`）：
1. `move_j` 到進刀點（起點沿銲絲反方向 {DF['approach'] * 1e3:.0f} mm）
2. `move_l` 直線進刀到起點 → 起弧、停留 {DF['t_pre']} s
3. 沿焊道等速銲接（固定 work／travel 角）
4. 停留 {DF['t_crater']} s 收弧 → `move_l` 沿銲絲反方向退刀

示範動畫使用的 JSON 工作（`run_job`；未指定 `roll_deg` → 自動選 roll）：
```json
{job}
```

## 量化結果
### 1. 標稱焊道（鋼板平銲，J 形：直線 120 mm + R40 圓弧 90° + 直線 80 mm，總長 {N0['length_mm']:.1f} mm；10 mm/s、work 90°、travel +10°）
| 項目 | 數值 |
|---|---|
| 結果 | {ok(N0['ok'])} |
| 路徑誤差（對焊道幾何）RMS／最大 | {um(N0, 'path_rms_mm')}／{um(N0, 'path_max_mm')} µm（側向最大 {um(N0, 'lat_max_mm')}、高度最大 {um(N0, 'hgt_max_mm')} µm） |
| 追蹤誤差（對同時刻參考點）RMS／最大 | {um(N0, 'track_rms_mm')}／{um(N0, 'track_max_mm')} µm（≈ v × 1 個控制週期 = 10 µm 的沿焊道延遲） |
| 行進速度（等速段）平均／標準差／最大偏差 | {N0['speed_mean_mm_s']:.4f} mm/s／{N0['speed_std_mm_s'] * 1e3:.2f} µm/s／{N0['speed_dev_max_pct']:.3f} % |
| 入熱量偏差最大 | {N0['heat_dev_max_pct']:.3f} %（標稱 {N0['heat_J_mm']:.1f} J/mm） |
| work／travel 角誤差最大 | {N0['work_err_max_deg']:.4f}°／{N0['travel_err_max_deg']:.5f}° |
| 航向偏差最大（接點暫態） | {N0['heading_err_max_deg']:.3f}° |
| 關節追蹤誤差最大 | {N0['joint_err_max_mrad']:.2f} mrad |
| 圓弧段銲槍角速度 v/R | {N0['corner_rate_rad_s']:.2f} rad/s；腕 3 行程 {N0['q_range_deg'][5]:.0f}° |
| 速度／力矩使用率 | {N0['vel_util']:.3f}／{N0['torque_util']:.3f}；路徑上 σ_min(J) = {N0['sigma_min']:.3f} |
| 噴嘴–鋼板最小間隙 | {N0['clear_mm']:.1f} mm（伸出長度 {TO['STICKOUT'] * 1e3:.0f} mm） |
| 銲接時間／含進退刀週期 | {N0['weld_s']:.2f} s／{N0['cycle_s']:.2f} s |

![時間歷程](../media/{T}_timeseries.png)

### 2. 行進速度掃描（同一 J 形焊道）
{HEAD.format('速度 [mm/s]')}
{speeds}

- 圓弧段需要腕部以 v/R 旋轉，規劃的關節速度使用率與 v 成正比（10 mm/s 時 {N0['plan_vel_ratio']:.4f}）→ 預測上限 **v_max = {M['v_max_pred'] * 1e3:.1f} mm/s**；
  實測：規劃可行到 {max(m['speed_cmd_mm_s'] for m in sp_ok):.0f} mm/s，{'、'.join(f"{m['speed_cmd_mm_s']:.0f}" for m in sp_no)} mm/s 回報 LIMIT（不執行）。
- {('可行但接點暫態使實際關節速度超過額定：' + '、'.join(f"{m['speed_cmd_mm_s']:.0f} mm/s（{m['vel_util']:.2f}）" for m in over) + '——未做航向平滑時，高速下接點的速度階躍會造成超調。') if over else '可行速度下實際關節速度皆未超過額定。'}

### 3. 圓角半徑 × 航向平滑（10 mm/s；直線 60 + 圓弧 90° + 直線 50 mm）
| R [mm] | 平滑半寬 b [mm] | 結果 | v/R [rad/s] | 規劃關節加速度最大 [rad/s²] | 關節誤差最大 [mrad] | 速度偏差最大 [%] | 路徑誤差最大 [µm] | 航向偏差最大 [°] | 理論 b/(4R) [°] | work／travel 誤差最大 [°] | 速度使用率 | 力矩使用率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{corner}

- 無平滑時接點角速度階躍，規劃關節加速度達 {min(m['plan_qdd_max'] for m in c_no):.0f}–{max(m['plan_qdd_max'] for m in c_no):.0f} rad/s²（10 ms 取樣內完成階躍）；
  b = 2 mm 降到 {min(m['plan_qdd_max'] for m in c_b2):.2f}–{max(m['plan_qdd_max'] for m in c_b2):.2f} rad/s²。
- R5 小圓角：速度偏差由 {r5[0]['speed_dev_max_pct']:.2f} %（無平滑）降到 {r5[2]['speed_dev_max_pct']:.2f} %（b = 2 mm）／{r5[5]['speed_dev_max_pct']:.2f} %（b = 5 mm），
  代價是接點航向偏差 {r5[2]['heading_err_max_deg']:.1f}°／{r5[5]['heading_err_max_deg']:.1f}°（理論 {r5[2]['heading_theory_deg']:.1f}°／{r5[5]['heading_theory_deg']:.1f}°）。
  因標稱銲槍接近垂直（travel 10°），航向偏差只反映成 work 角 ≤ {max(m['work_err_max_deg'] for m in M['corner']):.2f}° 的變化。
- 位置路徑誤差不受平滑影響（只平滑姿態）。

![掃描](../media/{T}_sweeps.png)

### 4. 銲槍角度矩陣（20 mm/s；直線 80 + R40 圓弧 + 直線 60 mm；自動選 roll）
| work [°] | travel [°] | 結果 | 選用 roll [°] | σ_min | 路徑誤差最大 [µm] | 速度偏差最大 [%] | work／travel 誤差最大 [°] | 速度使用率 | 力矩使用率 | 最小間隙 [mm] |
|---|---|---|---|---|---|---|---|---|---|---|
{angles}

{ang_ok}/{len(M['angles'])} 組成功。

### 5. 接頭／工件形式（10 mm/s、travel +10°、自動選 roll）
{HEAD.format('案例')}
{cases}

繞銲絲 roll 候選（規劃摘要）與實際嘗試順序（依 σ_min 由大到小，碰撞預檢不過就換下一個）：
| 案例 | roll 候選 | 嘗試結果 |
|---|---|---|
{rolls}

- 整圓焊道：銲槍繞管一圈，腕 3 需轉 {CI['q_range_deg'][5]:.0f}°；不可行的 roll 候選（J6 超出 ±2π 限位或路徑經過奇異點，見上表）被排除，自動選用 roll {CI['roll_deg']:.0f}°。
- 填角焊噴嘴離立板／管壁最小間隙僅 {FI['clear_mm']:.1f}／{CI['clear_mm']:.1f} mm（work 45° 時噴嘴邊緣本來就貼近立板；幾何估算 15·sin45° − 11·cos45° ≈ 2.8 mm）。
- {inc60_txt}

![案例](../media/{T}_cases.png)

### 6. 控制器比較（標稱 J 形焊道）
{HEAD.format('控制器／速度')}
{ctrl}

PD＋重力補償沒有慣性／科氏前饋：10 mm/s 時路徑誤差最大 {um(pd[10.0], 'path_max_mm')} µm、速度偏差 {pd[10.0]['speed_dev_max_pct']:.1f} %
（計算力矩：{um(ctc[10.0], 'path_max_mm')} µm、{ctc[10.0]['speed_dev_max_pct']:.3f} %）；50 mm/s 時 {um(pd[50.0], 'path_max_mm')} µm、{pd[50.0]['speed_dev_max_pct']:.1f} %。
速度偏差直接等於入熱量偏差，因此銲接這類「等速」製程需要有前饋的軌跡追蹤控制。

### 7. 示範工作（JSON → `run_job`：填角焊 J 形 + 管–板整圓，連續執行）
{HEAD.format('焊道')}
{demo}

整個工作（含進退刀、空移、回原點）{M['demo']['t_total']:.1f} s，{'全部成功' if M['demo']['ok'] else '有焊道失敗'}；沿實際 TCP 軌跡每 3 mm 放一顆焊珠（共 {M['demo']['beads']} 顆，僅視覺）。

## 圖表與動畫
- 動畫（{M['frames']} 幀；左：全景、中：近拍、右：俯視；橘色＝剛銲的焊珠，灰色＝已冷卻）：
  [GIF](../media/{T}.gif)（{M['gif_MB']:.1f} MB）／[MP4](../media/{T}.mp4)（{M['mp4_MB']:.1f} MB）

![動畫](../media/{T}.gif)

| 第一條焊道（填角焊 J 形）進行中 | 第二條焊道（管–板整圓）進行中 | 完成回原點 |
|---|---|---|
| ![a](../media/{T}_key_a.png) | ![b](../media/{T}_key_b.png) | ![c](../media/{T}_key_c.png) |

## 問題與限制
- 追蹤誤差是「理想模型」下的數值：計算力矩控制用的逆動力學與模擬器完全相同，沒有關節摩擦、齒隙、撓性、量測雜訊與延遲，
  所以 µm 級誤差只代表軌跡規劃與控制流程正確，不代表實機精度（實機一般 0.05–0.1 mm 重複精度）。
- 不模擬電弧、熔池、熱變形與銲絲接觸；入熱量與焊道截面只是由速度換算的指標；焊珠為視覺標記。
- 焊道限單一平面內的直線＋圓弧且切線連續（折角 > 1° 會拒絕）；不支援空間曲線、擺動（weaving）、多層多道。
- 圓角處速度上限由腕部角速度決定（v ≤ q̇_max·R 量級）；目前不會自動在圓角降速，超限直接回報 LIMIT。
- 碰撞只做「規劃後預檢＋執行中監測」，不會自動繞開；進／退刀的 PTP 空移未做碰撞預檢（T110 避障規劃處理）。
- roll 只在 0／90／180／−90° 四個候選中選，且整條焊道固定；未做連續最佳化。
- 填角焊噴嘴間隙只有 1–2 mm，實務上需依噴嘴尺寸調整 work 角或伸出長度。

## 結論
- 直線＋圓弧焊道在固定 work／travel 角與等速下完成：標稱 10 mm/s 路徑誤差最大 {um(N0, 'path_max_mm')} µm、速度偏差最大 {N0['speed_dev_max_pct']:.3f} %、
  銲槍角度誤差 ≤ {max(N0['work_err_max_deg'], N0['travel_err_max_deg']):.3f}°、力矩使用率 {N0['torque_util']:.2f}。
- 速度上限由圓角處腕部角速度決定：R40 圓角 v_max ≈ {M['v_max_pred'] * 1e3:.0f} mm/s，超過即回報 LIMIT。
- 航向平滑可把接點關節加速度需求降低兩個數量級以上，代價為接點航向偏差 b/(4R)，與理論一致。
- 銲絲軸對稱提供的 roll 自由度必須利用：整圓焊道在 roll 0° 下會超出 J6 限位（J6 需到 {CI['rolls'][0]['q6_min']:.2f} rad），自動選 roll 後 360° 一次銲完。
- 平銲、T 形填角、管–板整圓、30°／45° 斜板皆成功；{inc60_txt}
- 下一步：T106 噴塗表面覆蓋（可沿用本模組的焊道座標、等速時間律與工具姿態定義）。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("wrote", f"docs/test-reports/{T}.md")
