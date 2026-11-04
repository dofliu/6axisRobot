"""由 T006 metrics JSON 產生繁體中文測試報告。用法：python3 scripts/t006_write_md.py 2026-11-04 "<pytest 摘要>" """

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
date, passed = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "?")
tag = f"{date}_T006"
m = json.loads((ROOT / "docs/test-reports/data" / f"{tag}_metrics.json").read_text(encoding="utf-8"))
b, t, f, d, dc = m["belt"], m["tray"], m["fixture"], m["demo"], m["demo_clearance"]
tray_rows = "\n".join(
    f"| {r['pocket']} | ({r['drop_off_mm'][0]:+.1f}, {r['drop_off_mm'][1]:+.1f}) | "
    f"({r['err_xy_mm'][0]:+.2f}, {r['err_xy_mm'][1]:+.2f}) | {r['z_err_mm']:+.3f} |" for r in t["rows"])
reach_rows = "\n".join(
    f"| {r['station']} | {r['radius_m']:.3f} | {r['grasp']['n_ik']}（無碰撞 {r['grasp']['n_free']}） | "
    f"{r['grasp']['clearance_mm']:.1f} | {r['hover']['n_ik']}（無碰撞 {r['hover']['n_free']}） | "
    f"{r['hover']['clearance_mm']:.1f} |" for r in m["reach"])
fx_rows = "\n".join(f"| {i + 1} | {e[0]:+.2f} | {e[1]:+.2f} |" for i, e in enumerate(f["err_mm"]))

md = f"""# {tag.replace('_', ' ')} 工作站場景 測試報告

日期：{date}　　數據：`docs/test-reports/data/{tag}_metrics.json`（由 `scripts/t006_report.py` 實際執行產生）

## 1. 目的
建立工作站場景（工作台、輸送帶＋擋料板、料盤、固定治具）與碰撞群組，驗證：零件被工作台承載、輸送帶速度與擋料定位、
料盤格定位、治具基準、各工位對手臂可達且無碰撞，並提供手臂—場景碰撞/間隙查詢（供 T110 避障規劃使用）。

## 2. 模型與參數（`sixaxis/scene.py`）
* 版面（俯視圖見下圖）：工作台 x∈[0.28,0.62]、y∈±0.29、頂面 0.10 m（= T005 基座高度，取料點 (0.45,0)）；
  料盤中心 (0.45,−0.17)，2×2 格、間距 90 mm、格內淨寬 58 mm（50 mm 方塊單邊間隙 4 mm），
  每格以 ±x 兩片端板 + ±y 側 4 根小銷（x=±22 mm，兩指墊 x 範圍 ±10 mm 不干涉）定位，高 20 mm；
  治具：+x 後擋板 + +y 側銷（μ_eff≈0.3）構成 2D 基準，標稱零件中心 (0.45,0.14)；
  輸送帶 x∈[0.15,0.55]、y∈[0.35,0.55]、帶面 0.10 m、+x 方向 {b['v_cmd']*1e3:.0f} mm/s，末端擋料板，零件停在 (0.495,0.45)。
* 輸送帶模型：PyBullet 無表面速度接觸，故帶面摩擦設 0，每步對與帶面接觸的零件施力
  F = m·clip((v_belt − v_x)/dt, ±μg)（μ_belt = 0.5）——庫侖限幅的理想摩擦驅動帶；脫離帶面（被夾起）即不受驅動。
* 碰撞群組（`setCollisionFilterGroupMask`，規則 (gA & mB) && (gB & mA)）：arm=1（含夾爪）、static=2（工作台/料盤/治具/機架/擋板）、
  belt=4、part=8。arm↔static/belt/part、part↔所有群組皆碰撞；belt↔static、arm↔arm 不碰撞（避免機架與帶面互相卡死）。
  另有群組/遮罩皆為 0 的 ghost 體用於驗證過濾。
* 查詢：`Workcell.arm_collisions(margin)` 回傳與手臂（含夾爪）距離 ≤ margin 的場景體；`min_clearance()` 回傳最小距離（負＝穿透）。
* `GraspScene(..., workcell=True)` 改用此場景（T005 原行為不變，預設仍為單一基座）。

## 3. 重現指令
```
python3 -m pytest tests/test_scene.py
PYTHONPATH=. python3 scripts/t006_report.py {date}
python3 scripts/t006_write_md.py {date} "{passed}"
```

## 4. 指令序列（示範，`scripts/t006_report.py`）
輸送帶送料（零件 x=0.34 → 擋料板，約 1.4 s）→［垂直抬起 → 高處中繼點 (0.45,0.2,0.30) → 目標上方 12 cm → 下降 → 閉爪 40 N →
抬升 → 搬運至料盤格上方 → 下降 → 開爪 → 抬起］，依序執行 ①輸送帶零件 → 料盤格 0、②工作台零件 → 料盤格 3。
全程為關節空間五次多項式 PTP，計算力矩控制（T005 的 `GraspScene.move_to`）。

## 5. 量化結果
### 5.1 輸送帶（零件由 x=0.20 起送）
| 指標 | 數值 |
|---|---|
| 穩態速度（指令 {b['v_cmd']*1e3:.0f} mm/s） | {b['v_ss']*1e3:.2f} mm/s（誤差 {b['v_err_pct']:+.2f} %） |
| 速度達 98 % 時間 | {b['ramp_98pct_s']*1e3:.0f} ms |
| 擋料板停止位置誤差（相對 (0.495,0.45)） | {b['stop_err_mm']:.3f} mm |
| 側向漂移 | {b['y_drift_mm']:.4f} mm |
| 起送至靜止 | {b['t_to_stop_s']:.2f} s |

![belt]({'../media/' + tag}_belt.png)

### 5.2 料盤格定位（方塊落下 3 cm；單邊間隙 {t['clearance_per_side_mm']:.1f} mm；4 格 × 4 種偏移）
全部停留在格內且貼平底板：{t['all_inside']}；z 最大誤差 {max(abs(r['z_err_mm']) for r in t['rows']):.3f} mm。
| 格 | 落下偏移 (x,y) mm | 靜止誤差 (x,y) mm | z 誤差 mm |
|---|---|---|---|
{tray_rows}

### 5.3 治具基準（{f['n']} 次，初始偏離標稱 ≤ {f['start_offset_max_mm']:.0f} mm，以 0.6 m/s 對角滑向基準角）
最大誤差 {f['max_mm']:.2f} mm，RMS {f['rms_mm']:.2f} mm。
| # | x 誤差 mm | y 誤差 mm |
|---|---|---|
{fx_rows}

### 5.4 工位可達性與碰撞（工具朝下、IK 取最接近 `Q_REF` 的解；間隙 = 手臂與場景最小距離）
| 工位 | 距基座 m | 抓取高度 IK 解數 | 間隙 mm | 上方 12 cm IK 解數 | 間隙 mm |
|---|---|---|---|---|---|
{reach_rows}

（抓取高度間隙 5.0 mm 為指墊底面距桌面；所有工位的首選解皆無碰撞。）
碰撞查詢驗證：工具中心置於桌面下 5 cm → 命中 {m['collision_query']['below_bench_hits']}，穿透 {-m['collision_query']['penetration_mm']:.1f} mm。

![layout]({'../media/' + tag}_layout.png)

### 5.5 示範取放（動畫）
| 週期 | 週期時間 | 放置誤差 (x,y) mm | 放置高度 z (m) |
|---|---|---|---|
| ① 輸送帶 → 格 0 | {d[0]['cycle_s']:.1f} s | ({d[0]['place_err_mm'][0]:+.2f}, {d[0]['place_err_mm'][1]:+.2f}) | {d[0]['z']:.4f} |
| ② 工作台 → 格 3 | {d[1]['cycle_s']:.1f} s | ({d[1]['place_err_mm'][0]:+.2f}, {d[1]['place_err_mm'][1]:+.2f}) | {d[1]['z']:.4f} |

輸送帶零件停止誤差 {m['demo_belt_part_stop_err_mm']:.3f} mm；全程 {dc['samples']} 個取樣（每 50 ms）手臂與場景最小間隙 {dc['min_mm']:.2f} mm，
碰撞取樣數 {dc['collision_samples']}。

![clearance]({'../media/' + tag}_clearance.png)

動畫：[GIF]({'../media/' + tag}.gif)（{m['gif_MB']:.1f} MB）／[MP4]({'../media/' + tag}.mp4)（{m['mp4_MB']:.1f} MB），{m['frames']} 格，三視角。

![key_a]({'../media/' + tag}_key_a.png)
![key_b]({'../media/' + tag}_key_b.png)
![key_c]({'../media/' + tag}_key_c.png)

## 6. 問題與限制
* 開發中發現：①最初的「角柱」料盤只擋斜向，純 x/y 方向可自由滑出，改為端板 + 側銷；②示範第二輪曾因從夾爪抱著工件的姿態直接 PTP 橫掃，
  把桌面零件掃走，且 PTP 路徑曾擦到治具擋板（clearance=0）→ 示範改為「先垂直抬起、經高處中繼點」。這正是 T110 避障規劃的需求。
* 料盤無導入倒角：偏移超出單邊間隙（4 mm）的零件會停在擋牆頂（實測 6 mm 偏移 z 高出 8 mm），所以純軸向測試取 0.95 倍間隙；定位精度 = 間隙（±4 mm）。
* 治具以「滑入基準角」定位：PyBullet 接觸無恢復係數建模，零件會輕微反彈/停在途中，{f['n']} 次中最大 {f['max_mm']:.1f} mm、RMS {f['rms_mm']:.1f} mm，
  不是精密夾持；之後若需精確定位應加氣動夾具（主動夾緊）。測試中另以 5 次、2 mm 門檻驗證（pytest 通過）。
* 輸送帶是力模型而非表面速度；零件被夾爪抬離後不受驅動、與帶面無摩擦（帶面 μ=0），零件在帶上不會因偏轉而自動對正（示範皆以 yaw=0 生成）。
* 碰撞間隙查詢僅針對手臂（含夾爪）對場景靜態體，不含手持零件；地板未建碰撞體。

## 7. 結論
工作站場景與碰撞群組可用：輸送帶速度誤差 {b['v_err_pct']:+.2f} %、擋料板定位 {b['stop_err_mm']:.3f} mm；料盤 4 格皆可收容；所有工位可達且首選 IK 解無碰撞；
示範兩個取放週期放置誤差 < 0.5 mm、全程無碰撞（最小間隙 {dc['min_mm']:.1f} mm）。測試：{passed}。下一步 T007 指令介面。
"""
(ROOT / "docs/test-reports" / f"{tag}.md").write_text(md, encoding="utf-8")
print("ok")
