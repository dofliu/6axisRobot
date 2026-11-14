"""T106 報告 md：由 docs/test-reports/data/<date>_T106_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t106_write_md.py 2026-11-14 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T106"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
N0, MO, DF, MC = M["nominal"], M["model"], M["defaults"], M["model_check"]
F0 = N0["film"]


def ok(b):
    return "✓" if b else "✗"


def f(x, fmt=".2f"):
    return "—" if x is None else format(x, fmt)


def g(m, *ks, fmt=".2f"):
    x = m
    for k in ks:
        if not isinstance(x, dict) or k not in x:
            return "—"
        x = x[k]
    return f(x, fmt)


HEAD = ("| {} | 結果 | 道數 | 噴塗時間 [s] | 平均膜厚 [µm]（理論 Q/vΔ） | CV [%] | 極差 [%] | ±10 % 面積 [%] | "
        "中央區 CV [%] | 轉移效率 [%] | 路徑偏差（開槍）最大 [mm] | 速度偏差（開槍）最大 [%] | 速度使用率 | 力矩使用率 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def row(tag, m):
    if not m["ok"]:
        return (f"| {tag} | ✗ {m['error']} | {m.get('strokes', '—')} | — | — | — | — | — | — | — | — | — | "
                f"{g(m, 'plan_vel_ratio')}（規劃） | {g(m, 'plan_torque_ratio')}（規劃） |")
    fm = m["film"]
    return (f"| {tag} | {ok(m['ok'])} | {m['strokes']} | {m['spray_s']:.1f} | {fm['mean_um']:.2f}（{m['mean_pred_um']:.2f}） | "
            f"{fm['cv_pct']:.2f} | {fm['range_pct']:.1f} | {fm['within_tol_pct']:.1f} | {m['central_cv_pct']:.2f} | "
            f"{m['te'] * 100:.1f} | {m['path_dev_on_max_mm']:.4f} | {m['speed_dev_on_max_pct']:.3f} | "
            f"{m['vel_util']:.2f} | {m['torque_util']:.2f} |")


spacing = "\n".join(
    f"| {m['spacing'] * 1e3:.0f} | {m['spacing'] / MO['R']:.3f} | {ok(m['ok'])} | {m['strokes']} | {g(m, 'spray_s', fmt='.1f')} | "
    f"{g(m, 'film', 'mean_um')}（{g(m, 'mean_pred_um')}） | {g(m, 'central_cv_pct')} | {m['ripple_cv_pct']:.2f} | "
    f"{g(m, 'central_range_pct', fmt='.1f')} | {m['ripple_range_pct']:.1f} | {g(m, 'film', 'cv_pct')} | {f(m.get('te', 0) * 100, '.1f')} |"
    for m in M["spacing"])
speed = "\n".join(row(f"{m['speed_cmd']:.2f} m/s", m) for m in M["speed"])
standoff = "\n".join(
    f"| {m['standoff'] * 1e3:.0f} | {MO['R'] * m['standoff'] / MO['h_ref'] * 1e3:.1f} | {ok(m['ok'])} | "
    f"{g(m, 'film', 'mean_um')} | {g(m, 'central_cv_pct')} | {m['ripple_cv_pct']:.2f} | {g(m, 'film', 'cv_pct')} | "
    f"{g(m, 'film', 'within_tol_pct', fmt='.1f')} | {f(m.get('te', 0) * 100, '.1f')} | {g(m, 'clear_mm', fmt='.1f')} |"
    for m in M["standoff"])
edge = "\n".join(
    f"| {m['trigger_mm']:.0f} | {m['edge_mm']:.0f} | {m['strokes']} | {m['on_s']:.2f} | {m['film_full']['cv_pct']:.2f} | "
    f"{m['film_full']['min_um']:.1f} | {m['film_full']['coverage_pct']:.1f} | {m['film']['cv_pct']:.2f} | {m['te'] * 100:.1f} | "
    f"{m['spray_s']:.1f} |"
    for m in M["edge"])
tilt = "\n".join(row(f"{m['tilt_deg']:+.0f}°（σ_min {m['sigma_min']:.3f}）", m) for m in M["tilt"])
incl = "\n".join(row(f"斜板 {m['panel_tilt_deg']}°（噴槍垂直板面）", m) for m in M["inclined"])
ctrl = "\n".join(row(f"{'計算力矩' if m['controller'] == 'ctc' else 'PD＋重力補償'}／{m['speed_cmd']:.2f} m/s", m)
                 for m in M["controllers"])
DEMO = M["demo"]
demo_rows = "\n".join(
    f"| {p['panel']} | {ok(p['ok'])} | {p['strokes']} | {p['cycle_s']:.1f} | {p['film']['mean_um']:.2f} | {p['film']['cv_pct']:.2f} | "
    f"{p['film']['within_tol_pct']:.1f} | {p['path_dev_on_max_mm']:.4f} | {p['speed_dev_on_max_pct']:.3f} | {p['clear_mm']:.1f} |"
    for p in DEMO["panels"])
sp_bad = [m for m in M["speed"] if not m["ok"]]
sp_ok = [m for m in M["speed"] if m["ok"]]
mean_err = max(abs(m["film"]["mean_um"] / m["mean_pred_um"] - 1) for m in sp_ok) * 100
sp_cmp = max(abs(m["central_cv_pct"] - m["ripple_cv_pct"]) for m in M["spacing"] if m["ok"])
st_cmp = max(abs(m["central_cv_pct"] - m["ripple_cv_pct"]) for m in M["standoff"] if m["ok"])
tilt_bad = [m for m in M["tilt"] if not m["ok"]]
tilt_ok = [m for m in M["tilt"] if m["ok"]]
I30, I45 = M["inclined"]
CT, PD = M["controllers"][0], M["controllers"][2]
E0 = next(m for m in M["edge"] if m["trigger_mm"] == 0 and m["edge_mm"] == 0)
EB = max(M["edge"], key=lambda m: m["film_full"]["coverage_pct"] * 1000 + m["te"])
ms2 = MC["max_spacing_mm"]

md = f"""# T106 噴塗表面覆蓋 測試報告（{D}）

## 1. 目的
建立噴塗製程的「塗料沉積模型 → 光柵（zig-zag）噴塗路徑 → 手臂力矩控制執行 → 由實際噴嘴位姿積分膜厚」整條鏈，
並以量化指標（平均膜厚、膜厚變異係數 CV、極差、目標 ±10 % 面積比、覆蓋率、轉移效率、路徑/速度偏差、力矩/速度使用率）
回答：道間距、噴塗速度、噴距、扳機區、噴槍傾角與控制器對塗層均勻度與可行性的影響。

## 2. 模型與參數
### 2.1 噴槍
直式自動噴槍以固定關節接在法蘭：法蘭 → 噴嘴口 {M['gun']['GUN_L'] * 1e3:.0f} mm、質量 {M['gun']['GUN_MASS']} kg（含於 PyBullet
逆動力學）。TCP = 噴嘴口（URDF 連桿 8），z 軸 = 噴霧中心軸。工作站沿用 T105 `WeldCell`（本次把工具 URDF 與 TCP 變換參數化，
銲槍為預設，既有 T105 測試不受影響）。

### 2.2 沉積模型（`sixaxis/spraying.py: SprayModel`）
* 參考平面（垂直噴軸、距離 h_ref）膜厚速率為 **beta 分佈**（Balkan & Arikan 1999；Conner et al., *IEEE T-ASE* 2(4), 2005）：
  f(r) = Q·β/(πR²)·(1 − r²/R²)^(β−1)，R = h_ref·tan φ，∫f dA = Q。
* 一般曲面推廣（點源）：角通量 F(θ) = f(h_ref·tan θ)·h_ref²/cos³θ，任意點膜厚速率 ḣ = F(θ)·cos γ/ρ²
  （θ：與噴軸夾角、γ：入射角、ρ：距離；背面不沉積、不計遮蔽）。垂直平面距離 h 時圖樣等比放大、峰值 ∝ 1/h²、總量守恆。
* 單道解析解：g(y) = Q·β/(πR²v)·B(½,β)·R^(2−2β)·(R² − y²)^(β−½)，∫g dy = Q/v；
  間距 Δ 的多道內部平均 = **Q/(v·Δ)**，均勻度 CV 只與 Δ/R、β 有關。

| 參數 | 值 |
|---|---|
| 沉積體積流量 Q（膜體積，已含轉移率） | {MO['flow_cc_min']:.1f} cm³/min |
| 參考噴距 h_ref、錐半角 φ、圖樣半徑 R | {MO['h_ref'] * 1e3:.0f} mm、{MO['half_angle']:.0f}°、{MO['R'] * 1e3:.1f} mm |
| beta 指數 β | {MO['beta']} |
| 標稱噴塗：速度、道間距、噴距 | {DF['speed']} m/s、{DF['spacing'] * 1e3:.0f} mm、{DF['standoff'] * 1e3:.0f} mm |
| 每道超出面板 overrun、扳機區（\\|u\\| ≤ L/2 + trigger）、換道半徑 | {DF['overrun'] * 1e3:.0f} mm、+{DF['trigger'] * 1e3:.0f} mm、Δ/2 |
| 起/停加減速（smoothstep） | {DF['t_ramp']} s |
| 平板 | 中心 {tuple(M['panel_flat']['center'])}、{M['panel_flat']['size'][0] * 1e3:.0f}×{M['panel_flat']['size'][1] * 1e3:.0f} mm（膜厚格點 2.5 mm；掃描 5 mm） |
| 膜厚統計區 | 「面板 − 10 mm 邊」為主；另列全面板與中央 200×200 mm |
| 控制 | 計算力矩（ωn = 60 rad/s）、1 kHz；對照 PD＋重力補償 |

### 2.3 路徑與時間律
噴嘴路徑位於與面板平行、距離 h 的平面：每道長 L + 2·overrun，道間以半徑 Δ/2 的 180° 圓弧轉向（切線連續，沿用 T105 `Seam`／
`SpeedProfile` 等速時間律）；道數 n = ⌈(W + 2·edge)/Δ⌉ + 1、對稱排列。噴槍姿態全程固定（若隨航向轉，換道時腕部角速度 = v/(Δ/2)
= {0.3 / 0.025:.0f} rad/s，遠超 π rad/s）。噴軸傾斜時整條路徑沿噴軸平移，使噴霧中心仍落在道中心線。逐點數值 IK、
q̇ = J⁻¹[v; 0]、三次 Hermite 插值、RNE 檢查力矩，再以碰撞預檢（間隙 ≥ 5 mm）把關。

## 3. 可重現指令
```bash
python3 -m pip install "setuptools<70" wheel && python3 -m pip install --no-build-isolation pybullet==3.2.7
python3 -m pip install -r requirements.txt
ruff check . && python3 -m pytest -q          # 不可與報告腳本同時執行（共用 robots/ur5_spray.urdf）
PYTHONPATH=. python3 scripts/t106_report.py {D}       # 實驗 → docs/test-reports/data/{T}_metrics.json、圖、動畫
python3 scripts/t106_write_md.py {D} "<pytest 摘要>"
```
測試：{PYT}

## 4. 指令序列（動畫用 JSON 工作）
```json
{json.dumps(DEMO['job'], ensure_ascii=False)}
```
`parse_job` → `SprayCell.spray`：move_j 到第一道起點上方 {DF['approach'] * 1e3:.0f} mm → move_l 下降到噴距 → 等速光柵（扳機依噴霧中心是否在
\\|u\\| ≤ L/2 + trigger 自動開/關）→ 沿噴軸退槍 → 回原點。動畫第二段為 45° 斜板（噴槍垂直板面）。

## 5. 量化結果
### 5.1 模型驗證（無手臂）
* 單道：點源數值積分 vs 解析 g(y) 最大相對誤差 **{MC['stroke_max_rel_err']:.1e}**（峰值 {MC['stroke_peak_um']:.2f} µm @ 0.3 m/s）。
* CV 隨 Δ/R **非單調**（beta 分佈支撐邊緣的 (R² − y²)^(β−½) 造成），β = 2.5 時 CV ≤ 1／2／5 % 的「連續區」最大道距為
  **{ms2['1%']:.1f}／{ms2['2%']:.1f}／{ms2['5%']:.1f} mm**（從小間距往上掃到首次超標；之後雖有孤立甜蜜點但對噴距變化敏感，不採用）。
  標稱 Δ = 50 mm（Δ/R = {0.05 / MO['R']:.3f}）。

### 5.2 標稱噴塗（平板 400×300 mm）
{HEAD.format('情境')}
{row('標稱', N0)}

* 全面板（含邊）：CV {N0['film_full']['cv_pct']:.2f} %、最小 {N0['film_full']['min_um']:.1f} µm、±10 % 面積 {N0['film_full']['within_tol_pct']:.1f} %；
  週期（含進退槍）{N0['cycle_s']:.1f} s、最小間隙 {N0['clear_mm']:.1f} mm、噴軸誤差最大 {N0['axis_err_max_deg']:.4f}°。

### 5.3 道間距掃描（v = 0.3 m/s、h = 200 mm）
| Δ [mm] | Δ/R | 結果 | 道數 | 噴塗時間 [s] | 平均膜厚 [µm]（理論） | 中央 CV 模擬 [%] | CV 解析 [%] | 中央極差 模擬 [%] | 極差 解析 [%] | CV（面板−10 mm）[%] | 轉移效率 [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
{spacing}

中央區模擬 CV 與無限多道解析 CV 最大差 {sp_cmp:.2f} 個百分點（中央區 200 mm 不是 Δ 的整數倍、端道影響，造成小差異）。

### 5.4 速度掃描（Δ = 50 mm）
{HEAD.format('速度')}
{speed}

* 平均膜厚與 Q/(vΔ) 最大相對差 {mean_err:.2f} %（膜厚 ∝ 1/v）；規劃速度使用率 ∝ v，預測上限 v_max = 0.3/{N0['plan_vel_ratio']:.3f} =
  **{M['v_max_pred']:.2f} m/s**；{'、'.join(f"{m['speed_cmd']:.2f} m/s" for m in sp_bad)} 回報 {'／'.join(sorted({m['error'] for m in sp_bad}))}（不執行）。

### 5.5 噴距掃描（Δ = 50 mm 固定）
| 噴距 h [mm] | 圖樣半徑 [mm] | 結果 | 平均膜厚 [µm] | 中央 CV 模擬 [%] | CV 解析（等效 Δ·h_ref/h）[%] | CV（面板−10 mm）[%] | ±10 % 面積 [%] | 轉移效率 [%] | 最小間隙 [mm] |
|---|---|---|---|---|---|---|---|---|---|
{standoff}

最大差（模擬 − 解析）{st_cmp:.2f} 個百分點。

### 5.6 扳機區 × 側邊外延（全面板含邊的指標）
| 扳機 trigger [mm] | 側邊外延 edge [mm] | 道數 | 開槍時間 [s] | CV 全面板 [%] | 最小膜厚 [µm] | 覆蓋率（≥ 0.9·平均）[%] | CV 面板−10 mm [%] | 轉移效率 [%] | 噴塗時間 [s] |
|---|---|---|---|---|---|---|---|---|---|
{edge}

### 5.7 噴槍傾角（平板、繞道方向 u 傾斜）
{HEAD.format('傾角')}
{tilt}

### 5.8 斜面板（噴槍垂直板面）
{HEAD.format('情境')}
{incl}

### 5.9 控制器比較
{HEAD.format('控制器／速度')}
{ctrl}

### 5.10 動畫 JSON 工作
| 面板 | 結果 | 道數 | 週期 [s] | 平均膜厚 [µm] | CV [%] | ±10 % 面積 [%] | 路徑偏差最大 [mm] | 速度偏差最大 [%] | 最小間隙 [mm] |
|---|---|---|---|---|---|---|---|---|---|
{demo_rows}

## 6. 圖表與動畫
![模型](../media/{T}_model.png)
![標稱](../media/{T}_nominal.png)
![掃描](../media/{T}_sweeps.png)

動畫（左：全景、中：側視、右：俯視膜厚色塊，藍色 = 目標 22 µm；藍點為噴霧粒子示意）：
![T106](../media/{T}.gif)

[MP4](../media/{T}.mp4)｜關鍵畫面：[平板噴塗中](../media/{T}_key_a.png)、[平板完成](../media/{T}_key_b.png)、[45° 斜板完成](../media/{T}_key_c.png)

## 7. 問題與限制
* 沉積模型為理想點源（beta 分佈、Q 為已含轉移率的膜體積流量）：不模擬霧化、氣流、靜電、反彈、流掛與溶劑揮發；未做遮蔽（凹面/遮蔽區不適用）；
  參數 β、φ 為代表值，實機須以單點/單道噴塗板量測擬合。
* CV(Δ) 非單調是理想 beta 分佈銳利邊界的結果；實際噴霧邊界較平滑，甜蜜點會較不明顯，故以「連續合格區」的保守準則選道距。
* 平板工作空間很緊：直式噴槍＋腕部朝下，面板中心 x = 0.50 m 時遠角 IK 無解，故標稱面板放在 x = 0.45 m；
  噴槍傾角 {', '.join(f"{m['tilt_deg']:+.0f}°" for m in tilt_bad)} 失敗（{'／'.join(sorted({m['error'] for m in tilt_bad}))}；
  正傾角＝噴軸偏向手臂，遠側道 IK 無解；−30° 時沿路徑 σ_min 降到 {next(m for m in M['tilt'] if m['tilt_deg'] == -30)['sigma_min']:.1e}，
  腕部奇異使數值 IK 失敗）。直式噴槍繞噴軸的 roll 只轉 J6，無法像 T105 銲槍那樣用 roll 迴避。
* 只處理平面面板；曲面（例如圓柱外殼）需沿面法向改變姿態與噴距，留待後續（T107 以後若需要）。
* 膜厚以實際噴嘴位姿每 2 ms 矩形積分（單元測試確認每 1 ms 與每 4 ms 差 < 1 %）。

## 8. 結論
* 模型：單道解析解與點源數值積分一致（{MC['stroke_max_rel_err']:.0e}），平均膜厚 Q/(vΔ) 由速度掃描驗證（最大差 {mean_err:.2f} %）。
* 標稱噴塗 {N0['strokes']} 道、{N0['spray_s']:.1f} s：平均 {F0['mean_um']:.2f} µm（理論 {N0['mean_pred_um']:.2f}）、CV {F0['cv_pct']:.2f} %、
  ±10 % 面積 {F0['within_tol_pct']:.1f} %、轉移效率 {N0['te'] * 100:.1f} %；開槍段路徑偏差 ≤ {N0['path_dev_on_max_mm']:.4f} mm、速度偏差 ≤ {N0['speed_dev_on_max_pct']:.3f} %、
  力矩使用率 {N0['torque_util']:.2f}。
* 道間距決定均勻度（CV 與解析解一致，最大差 {sp_cmp:.2f} pp）；速度決定膜厚（∝ 1/v）並受關節速度限制（v_max ≈ {M['v_max_pred']:.2f} m/s）；
  噴距改變圖樣半徑而改變等效 Δ/R。邊緣以扳機區與側邊外延改善：全面板 CV {E0['film_full']['cv_pct']:.2f} % → {EB['film_full']['cv_pct']:.2f} %、
  覆蓋率 {E0['film_full']['coverage_pct']:.1f} % → {EB['film_full']['coverage_pct']:.1f} %，代價是轉移效率 {E0['te'] * 100:.1f} % → {EB['te'] * 100:.1f} %。
* 斜面板 30°／45° 噴槍垂直板面時膜厚與平板一致（CV {I30['film']['cv_pct']:.2f}／{I45['film']['cv_pct']:.2f} %），但 45° 的規劃速度使用率
  {I45['plan_vel_ratio']:.2f}（平板 {N0['plan_vel_ratio']:.2f}）。PD＋重力補償於 0.3 m/s 的開槍段速度偏差 {PD['speed_dev_on_max_pct']:.2f} %
  （計算力矩 {CT['speed_dev_on_max_pct']:.3f} %），CV {PD['film']['cv_pct']:.2f} % vs {CT['film']['cv_pct']:.2f} %。
* 下一步：T107 鎖螺絲（定位＋姿態＋旋轉）。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written", T)
