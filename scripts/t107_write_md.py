"""T107 報告 md：由 docs/test-reports/data/<date>_T107_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t107_write_md.py 2026-11-15 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T107"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
S, J, DP, FP = M["screw"], M["joint"], M["driver"], M["fasten"]
NOM = M["nominal"]


def f(x, fmt=".2f"):
    return "—" if x is None else format(x, fmt)


def res(r):
    return "OK" if r["ok"] else f"NG：{r['error']}"


def mx(rows, k):
    return max(r[k] for r in rows)


def mn(rows, k):
    return min(r[k] for r in rows)


# --- A 標稱
rowsA = "\n".join(
    f"| {r['hole']} | {res(r)} | {r['pos_err_mm'] * 1e3:.3f} | {r['tilt_deg']:.1e} | {r['turns_rundown']:.3f} | "
    f"{r['angle_snug_deg']:.2f} | {r['T_peak']:.4f} | {r['T_err_pct']:+.3f} | {r['F_clamp']:.0f} | {r['util']:.3f} | "
    f"{r['F_ax_min']:.1f}–{r['F_ax_max']:.1f} | {r['roll_dev_deg']:.1e} | {r['torque_util']:.3f} | "
    f"{r['t_spin']:.2f} | {r['cycle_s']:.2f} |" for r in NOM)
okA = sum(r["ok"] for r in NOM)

# --- B 策略
SB = [s for s in M["strategy_bench"] if not (s["two_stage"] and s["n_fin"] == DP["n_run"])]
rowsB = "\n".join(
    f"| {'二段' if s['two_stage'] else '單段'} | {s['n_fin'] if s['two_stage'] else DP['n_run']:.0f} | {s['rpm_cut']:.1f} | "
    f"{s['T_peak']:.4f} | {s['T_err_pct']:+.2f} | {s['T_peak'] - DP['T_target']:.4f} | {s['dT_est']:.4f} | {s['t_spindle']:.3f} | "
    f"{res(s)} |" for s in SB)
rowsBr = "\n".join(
    f"| {r['name']} | {res(r)} | {r['T_peak']:.4f} | {r['T_err_pct']:+.2f} | {r['F_clamp']:.0f} | {r['util']:.3f} | "
    f"{r['t_spin']:.2f} | {r['cycle_s']:.2f} |" for r in M["strategy_robot"])
one = next(s for s in SB if not s["two_stage"])
two60 = next(s for s in SB if s["two_stage"] and s["n_fin"] == 60)

# --- C 摩擦
FR = M["friction"]
rowsC = "\n".join(
    f"| {x['mu']:.2f} | {'扭力' if x['strategy'] == 'torque' else '扭力–轉角'} | {x['T_peak']:.3f} | {x['F_clamp']:.0f} | "
    f"{x['F_theory_torque']:.0f} | {x['angle_snug_deg']:.2f} | {x['util']:.3f} | {res(x)} |" for x in FR)
rowsCr = "\n".join(
    f"| {x['mu']:.2f} | {x['strategy']} | {res(x)} | {x['T_peak']:.3f} | {x['F_clamp']:.0f} | {x['util']:.3f} |"
    for x in M["friction_robot"])
ft = [x for x in FR if x["strategy"] == "torque"]
fa = [x for x in FR if x["strategy"] == "angle"]

# --- D 剛性
ST = M["stiffness"]
rowsD = "\n".join(
    f"| {s['joint']} | {s['c']:.3g} | {s['k_theta']:.3f} | {'二段' if s['two_stage'] else '單段'} | {s['angle_snug_deg']:.1f} | "
    f"{s['T_peak']:.4f} | {s['T_err_pct']:+.2f} | {s['t_spindle']:.2f} | {res(s)} |" for s in ST)

# --- E 孔位
OF = M["offset"]
rowsE = "\n".join(
    f"| {o['offset_mm']:.2f} | {'有' if o['search'] else '無'} | {res(o)} | {'是' if o.get('blocked') else '否'} | "
    f"{f(o.get('search_s'), '.3f')} | {f(o.get('search_r_mm'), '.2f')} | {f(o.get('lat_cap_mm'), '.3f')} | "
    f"{o['cycle_s']:.2f} |" for o in OF)
srch = [o for o in OF if o["search"] and o.get("search_s") is not None and o["ok"]]

# --- F 傾角
TL = M["tilt"]
rowsF = "\n".join(
    f"| {t['tilt_cmd_deg']:.1f} | {t['tilt_deg']:.3f} | {'是' if t['cross'] else '否'} | {f(t['turns_rundown'], '.3f')} | "
    f"{t['turns_nom']:.3f} | {t['T_peak']:.3f} | {t['F_clamp']:.0f} | {res(t)} |" for t in TL)

# --- G/H
rowsG = "\n".join(
    f"| {g['name']} | {res(g)} | {g['F_ax_min']:.1f} | {g['F_ax_max']:.1f} | {g['c_min_mm']:.2f} | {g['c_max_mm']:.2f} | "
    f"{g['T_peak']:.3f} |" for g in M["feed"])
rowsH = "\n".join(
    f"| {'有' if r['react_ff'] else '無'} | {res(r)} | {r['roll_dev_deg']:.4f} | {r['lat_dev_mm']:.4f} | {r['T_peak']:.4f} | "
    f"{r['F_clamp']:.0f} | {r['angle_snug_deg']:.2f} | {r['torque_util']:.3f} |" for r in M["reaction"])
RH = {r["react_ff"]: r for r in M["reaction"]}

# --- I
rowsI = "\n".join(
    f"| {r['hole']} | {res(r)} | {r['pos_err_mm']:.4f} | {r['tilt_deg']:.4f} | {r['T_peak']:.4f} | {r['F_clamp']:.0f} | "
    f"{r['torque_util']:.3f} | {r['cycle_s']:.2f} |" for r in M["tilted"] + M["job"]["holes"])
DM = M["demo"]

md = f"""# T107 鎖螺絲（定位＋姿態＋旋轉）測試報告

* 日期：{D}
* 對應 Roadmap：Phase 2 / T107（相依 T102）
* 程式：`sixaxis/screwing.py`；報告腳本 `scripts/t107_report.py`、`scripts/t107_write_md.py`；測試 `tests/test_screwing.py`
* 原始數據：[`data/{T}_metrics.json`](data/{T}_metrics.json)

## 1. 目的
建立「電動起子鎖螺絲」的完整模型與工作流程：手臂把螺絲**定位**到孔上方、**姿態**對準孔軸（含斜面），
起子主軸**旋轉**旋入並鎖到目標扭力，並以扭力/轉角監控判定品質。量化：定位/姿態誤差、最終扭力誤差、
夾緊力與其對摩擦的離散、過衝、進給力、反作用對手腕的影響、斜牙與找不到孔的偵測、週期時間。

## 2. 模型與參數（公式出處見 `sixaxis/screwing.py` 模組說明）
### 2.1 螺絲與接合
| 項目 | 值 | 依據 |
|---|---|---|
| 螺絲 | M{S['d'] * 1e3:.0f}×{S['P'] * 1e3:.1f}×{S['L'] * 1e3:.0f} 內六角（ISO 4762），8.8 級 R_p0.2 = {S['Rp'] / 1e6:.0f} MPa | ISO 898-1 |
| d2 / d3 / A_s | {S['d2'] * 1e3:.3f} / {S['d3'] * 1e3:.3f} mm / {S['As'] * 1e6:.2f} mm² | ISO 724、ISO 898-1 |
| 頭部 d_k / 通孔 d_h / D_Km | {S['dk'] * 1e3:.1f} / {S['dh'] * 1e3:.1f} / {S['Dkm'] * 1e3:.2f} mm | ISO 4762、ISO 273 |
| 夾緊板厚 t_b、嚙合長 l_e | {J['t_b'] * 1e3:.1f} mm、{J['l_e'] * 1e3:.1f} mm（座合 {J['turns_seat']:.3f} 圈） | — |
| 摩擦 μ_G = μ_K（標稱） | {J['mu_g']:.2f} | 代表值 |
| K' = 0.16P + 0.58μ_G·d2 + μ_K·D_Km/2 | {J['K'] * 1e3:.4f} mm | VDI 2230 (R13) |
| 接合剛性 c（柔度法） | {J['c']:.3e} N/m → 扭力–轉角斜率 k_θ = {J['k_theta']:.2f} N·m/rad | VDI 2230 |
| 目標扭力 T_A → 夾緊力 F_M、利用率 ν | {DP['T_target']:.2f} N·m → {J['F_target']:.0f} N、ν = {J['util_target']:.3f}（ν = 0.9 對應 {J['T_nu90']:.2f} N·m） | VDI 2230 (R14) |
| 自鎖：導程角 vs 螺紋摩擦角 | {J['lead_deg']:.2f}° < {J['rho_deg']:.2f}°（μ = 0.12） | — |
| 斜牙臨界傾角 φ_ct = atan(P/2d) | {S['phi_ct_deg']:.2f}°（本專案簡化幾何判據） | — |
| 殘留摩擦扭力 T_pv、斜牙扭力斜率 k_ct | {J['T_pv']:.2f} N·m、{J['k_ct']:.2f} N·m/rad | 代表值 |

### 2.2 電動起子與手臂
| 項目 | 值 |
|---|---|
| 起子工具 | 法蘭→起子頭尖端 {0.270 * 1e3:.0f} mm、1.6 kg（`robots/ur5_driver.urdf`，連桿 7 = 起子、8 = TCP） |
| 主軸 | J = {DP['J']:.1e} kg·m²、T_max = {DP['T_max']:.1f} N·m、速度 PI 頻寬 {DP['w_bw']:.0f} rad/s、控制/積分週期 {DP['h'] * 1e3:.1f} ms |
| 鎖固策略（標稱） | 二段式扭力控制：旋入 {DP['n_run']:.0f} rpm → 座合前 {DP['n_slow']:.1f} 圈降至 {DP['n_fin']:.0f} rpm → 到 {DP['T_target']:.1f} N·m 切斷並以 −T_max 煞車 |
| 監控窗 | 旋入圈數 ±{DP['turns_tol']:.1f} 圈、峰值扭力 {DP['torque_win'][0]:.2f}–{DP['torque_win'][1]:.2f}×目標、snug（{DP['snug_frac']:.1f}·T_A）後轉角 {DP['angle_win'][0]:.1f}–{DP['angle_win'][1]:.1f}×標稱 |
| 浮動起子頭 | 標稱壓縮 {FP['c_nom'] * 1e3:.0f} mm、行程 {FP['stroke'] * 1e3:.0f} mm、k_s = {FP['k_s']:.0f} N/m、預壓 {FP['F0']:.0f} N（標稱 F_ax = {FP['F0'] + FP['k_s'] * FP['c_nom']:.0f} N） |
| 入孔捕捉半徑 e_cap | (d_h − d)/2 + 倒角 0.5 = {M['e_cap_mm']:.2f} mm；螺旋搜尋間距 {FP['search_pitch'] * 1e3:.1f} mm、{FP['search_v'] * 1e3:.0f} mm/s、最大半徑 {FP['search_rmax'] * 1e3:.1f} mm |
| 手臂控制 | 計算力矩控制 ωn = 60 rad/s（PyBullet 1 kHz）；起子反作用（−T_m 繞孔軸、−F_ax 沿孔軸）以外力施加於工具連桿；前饋 τ_ff = −J_tcpᵀ·w_ext |

## 3. 測試步驟（可重現）
```bash
python3 -m pip install -r requirements.txt   # pybullet 安裝方式見 CLAUDE.md
ruff check .
python3 -m pytest -q                         # {PYT}
PYTHONPATH=. python3 scripts/t107_report.py {D}     # 實驗 → JSON、圖、GIF/MP4（不可與 pytest 同時跑）
python3 scripts/t107_write_md.py {D} "<pytest 摘要>"
```

## 4. 指令序列（每顆螺絲）
1. `move_j` 到供料器上方 → `move_l` 下降 40 mm 到螺絲頭 → 真空吸附 0.1 s → `move_l` 上升。
2. `move_j` 到孔軸上方 {FP['approach'] * 1e3:.0f} mm（TCP z 軸 = 孔軸；斜面時姿態依法向量）。
3. `move_l` 沿孔軸下降 {FP['t_descent']:.1f} s，螺絲端到螺紋口；若彈簧壓縮 > c_nom + 1.5 mm（頂到板面）→ 停止，
   螺旋搜尋（或回報 NO_ENGAGE）。
4. 起子旋入：手臂依主軸編碼器（深度 = P·θ/2π）沿孔軸跟隨；扭力/轉角監控；到目標切斷＋煞車。
5. `move_l` 沿孔軸退回 {FP['approach'] * 1e3:.0f} mm，判定 OK/NG。
JSON 工作格式：`{{"holes": [{{"name", "p", "a", "torque", "strategy", "angle_deg", "search"}}]}}`（`parse_job`／`run_job`）。

## 5. 量化結果
### A. 標稱：平板 6 孔（含取料、二段式扭力控制）— {okA}/6 OK
| 孔 | 結果 | 入孔位置誤差 [µm] | 傾角 [°] | 旋入圈數 | snug 後轉角 [°] | 峰值扭力 [N·m] | 扭力誤差 [%] | 夾緊力 [N] | ν | 進給力 [N] | 工具 roll 偏差 [°] | 力矩使用率 | 旋轉時間 [s] | 週期 [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{rowsA}

標稱旋入圈數 {NOM[0]['turns_nom']:.3f}、snug 後標稱轉角 {NOM[0]['angle_nom_deg']:.2f}°（硬接合，ISO 5393：snug 後 ≤ 30° 達目標）。

![nominal](../media/{T}_nominal.png)
![joints](../media/{T}_joints.png)

### B. 鎖固策略與過衝
接合模擬台（不含手臂，F_ax = 25 N；ΔT_est 為能量法 ½Jω² = ½k_θΔθ² + (T_A+T_max)Δθ 以「實際切斷轉速」計算＋取樣延遲）：

| 段數 | 最終轉速 [rpm] | 切斷時轉速 [rpm] | 峰值扭力 [N·m] | 過衝 [%] | ΔT = T_peak − T_A [N·m] | ΔT 估計 [N·m] | 主軸時間 [s] | 判定 |
|---|---|---|---|---|---|---|---|---|
{rowsB}

切斷時轉速低於指令（座合段扭力斜率 k_θ·ω ≈ {J['k_theta'] * DP['n_fin'] * 6.2832 / 60:.0f} N·m/s 的斜坡負載使速度 PI 迴路落後），
故以實際切斷轉速估計過衝。估計把取樣延遲取滿一個週期 h（實際平均 h/2），故為保守上界；與模擬同量級
（單段 600 rpm：模擬 {one['T_peak'] - DP['T_target']:.3f} vs 估計 {one['dT_est']:.3f} N·m）。

手臂上實鎖：

| 策略 | 結果 | 峰值扭力 [N·m] | 誤差 [%] | 夾緊力 [N] | ν | 旋轉時間 [s] | 週期 [s] |
|---|---|---|---|---|---|---|---|
{rowsBr}

### C. 摩擦離散：扭力控制 vs 扭力–轉角控制（模擬台；控制器只知道標稱 μ = 0.12）
| μ | 策略 | 峰值扭力 [N·m] | 夾緊力 [N] | F = T_A/K'(μ) [N] | snug 後轉角 [°] | ν | 判定 |
|---|---|---|---|---|---|---|---|
{rowsC}

μ = 0.08–0.16 時夾緊力離散（(max−min)/mean）：扭力控制 {M['friction_spread_torque_pct']:.1f} %（±{M['friction_halfrange_torque_pct']:.1f} %），
扭力–轉角控制 {M['friction_spread_angle_pct']:.1f} %（±{M['friction_halfrange_angle_pct']:.1f} %）。扭力控制在 μ = 0.08 時 ν = {ft[0]['util']:.3f}（{'超過降伏' if ft[0]['util'] > 1 else '接近降伏'}）。
手臂實鎖驗證：

| μ | 策略 | 結果 | 峰值扭力 [N·m] | 夾緊力 [N] | ν |
|---|---|---|---|---|---|
{rowsCr}

### D. 接合剛性（硬 / 中 / 軟接合）
| 接合 | c [N/m] | k_θ [N·m/rad] | 段數 | snug 後轉角 [°] | 峰值扭力 [N·m] | 誤差 [%] | 主軸時間 [s] | 判定 |
|---|---|---|---|---|---|---|---|---|
{rowsD}

![strategy](../media/{T}_strategy.png)

### E. 孔位偏差（手臂不知道）與螺旋搜尋
| 偏差 [mm] | 搜尋 | 結果 | 頂到板面 | 搜尋時間 [s] | 搜尋半徑 [mm] | 入孔時徑向誤差 [mm] | 週期 [s] |
|---|---|---|---|---|---|---|---|
{rowsE}

偏差 ≤ e_cap = {M['e_cap_mm']:.2f} mm 直接入孔；超過時無搜尋即 NO_ENGAGE，有搜尋時
{len(srch)} 例成功（搜尋時間 {f(min((o['search_s'] for o in srch), default=None), '.3f')}–{f(max((o['search_s'] for o in srch), default=None), '.3f')} s）。

### F. 工具傾角誤差 → 斜牙偵測
| 指令傾角 [°] | 實測傾角 [°] | 斜牙 | 旋入圈數 | 標稱圈數 | 峰值扭力 [N·m] | 夾緊力 [N] | 判定 |
|---|---|---|---|---|---|---|---|
{rowsF}

### G. 進給方式（手臂是否跟隨旋入深度）
| 方式 | 結果 | F_ax 最小 [N] | F_ax 最大 [N] | 壓縮最小 [mm] | 壓縮最大 [mm] | 峰值扭力 [N·m] |
|---|---|---|---|---|---|---|
{rowsG}

### H. 反作用力矩前饋
| 前饋 | 結果 | 工具 roll 偏差 [°] | TCP 徑向偏移 [mm] | 峰值扭力 [N·m] | 夾緊力 [N] | snug 後轉角 [°] | 力矩使用率 |
|---|---|---|---|---|---|---|---|
{rowsH}

無前饋時，計算力矩控制在 J6 的等效剛性只有 M66·ωn² = {M['M66']:.2e} × 60² = {M['M66'] * 3600:.2f} N·m/rad，
2.5 N·m 反作用的靜態估計 roll ≈ {M['roll_static_est_deg']:.0f}°（模擬 {RH[False]['roll_dev_deg']:.1f}°）；
工具反轉吃掉部分轉角，使夾緊力 {RH[False]['F_clamp']:.0f} N（有前饋 {RH[True]['F_clamp']:.0f} N）。

![robot](../media/{T}_robot.png)

### I. 斜面（{20:.0f}° 傾斜板）與 JSON 工作
| 孔 | 結果 | 入孔位置誤差 [mm] | 傾角 [°] | 峰值扭力 [N·m] | 夾緊力 [N] | 力矩使用率 | 週期 [s] |
|---|---|---|---|---|---|---|---|
{rowsI}

JSON 工作（J1 2.0 N·m 扭力、J2 2.0 N·m 扭力–轉角、J3 1.5 N·m）：{'全部 OK' if M['job']['ok'] else 'NG'}；格式錯誤拒絕 {M['job']['invalid_rejected']}。

## 6. 動畫與關鍵畫面
左：工作站全景；右：孔位特寫（紅色條紋為螺絲頭轉向標記），標頭顯示主軸狀態、轉速、扭力、圈數與夾緊力。
示範：平板 6 孔（H5 孔位偏 1.34 mm → 螺旋搜尋）＋ 20° 斜板 2 孔，共 {len(DM)} 顆，{sum(d['ok'] for d in DM)}/{len(DM)} OK。

![T107](../media/{T}.gif)

[MP4](../media/{T}.mp4)｜關鍵畫面：[旋入中](../media/{T}_key_a.png)、[螺旋搜尋](../media/{T}_key_b.png)、[斜板鎖緊](../media/{T}_key_c.png)

## 7. 問題與限制
* 螺紋接觸以解析模型（VDI 2230 扭力式＋線性接合剛性）處理，未以 PyBullet 接觸模擬螺紋；螺絲為運動學顯示物件。
  不含自攻/成形扭力、潤滑隨轉速變化、接合嵌入鬆弛（embedding）與降伏後塑性區；ν > 1 時只標示、不模擬降伏。
* 監控盲點：μ = 0.08 扭力控制（手臂實鎖）ν = {next(x for x in M['friction_robot'] if x['mu'] == 0.08 and x['strategy'] == 'torque')['util']:.4f}
  已達降伏，但 snug 後轉角 {next(x for x in M['friction_robot'] if x['mu'] == 0.08 and x['strategy'] == 'torque')['angle_snug_deg']:.2f}° 仍在窗內
  （上限 {DP['angle_win'][1]:.1f}×{NOM[0]['angle_nom_deg']:.2f}°）而判 OK；目標扭力 2.5 N·m 對低摩擦批次偏高，
  或須收窄轉角上限（≈ 1.3×）才能攔截。
* 斜牙判據 φ_ct = atan(P/2d) 與斜牙扭力斜率 k_ct 為簡化假設（非標準值），用於驗證「旋入圈數窗」偵測邏輯。
* 入孔捕捉只看徑向誤差 ≤ e_cap（假設浮動起子頭的徑向柔順可吸收）；螺旋搜尋以真實孔位判斷「掉入」，未模擬板面摩擦與起子頭打滑。
* 手臂採計算力矩控制（無積分項），對未建模的外力矩剛性低，故需反作用前饋（實際上起子內建扭力感測器可提供）；無前饋時 roll 偏差大。
* 主軸速度迴路在座合的斜坡負載下明顯落後，使實際過衝小於以指令轉速估計者；實機起子廠商會調整煞車/切斷補償，參數為代表值。

## 8. 結論
* 標稱 6 孔 {okA}/6 OK：入孔位置誤差 ≤ {mx(NOM, 'pos_err_mm') * 1e3:.3f} µm、傾角 ≤ {mx(NOM, 'tilt_deg'):.1e}°，峰值扭力誤差
  {mn(NOM, 'T_err_pct'):+.3f}～{mx(NOM, 'T_err_pct'):+.3f} %，夾緊力 {mn(NOM, 'F_clamp'):.0f} N（ν = {NOM[0]['util']:.3f}），
  進給力 {mn(NOM, 'F_ax_min'):.1f}–{mx(NOM, 'F_ax_max'):.1f} N，力矩使用率 ≤ {mx(NOM, 'torque_util'):.3f}，週期 {mn(NOM, 'cycle_s'):.2f}–{mx(NOM, 'cycle_s'):.2f} s。
* 硬接合必須二段式：單段 600 rpm 過衝 {one['T_err_pct']:+.1f} %（TORQUE_NG），二段 60 rpm 僅 {two60['T_err_pct']:+.3f} %；軟接合單段誤差小。
* 扭力控制的夾緊力受摩擦影響 ±{M['friction_halfrange_torque_pct']:.0f} %（與 F = T/K'(μ) 一致），扭力–轉角控制降至 ±{M['friction_halfrange_angle_pct']:.0f} %。
* 監控可偵測找不到孔（NO_ENGAGE，搜尋可恢復）、斜牙（傾角 > φ_ct 時旋入圈數過少）、起子頭脫離（BIT_LIFT，手臂不跟隨時）。
* 下一步：T108 去毛邊/拋光（力控、接觸力維持）。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written", T)
