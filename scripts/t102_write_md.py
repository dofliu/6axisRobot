"""T102 報告 md：由 docs/test-reports/data/<date>_T102_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t102_write_md.py 2026-11-10 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T102"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
B, R = M["batch"], M["rand"]
pk = "\n".join(f"| {p['pocket']} | ({M['pockets'][p['pocket']][0]:.4f}, {M['pockets'][p['pocket']][1]:.4f}) | "
               f"{p['success']}/{p['n']} |" for p in B["per_pocket"])


def rows(axis):
    out = []
    raw = {x["cmd_offset_mm"]: x for x in M["sweep"] if x["axis"] == axis and not x["recover"]}
    rec = {x["cmd_offset_mm"]: x for x in M["sweep"] if x["axis"] == axis and x["recover"]}
    for d in sorted(raw):
        a, b = raw[d], rec[d]
        out.append(f"| {d:+d} | {'✓' if a['success'] else '✗'} {a['error'] or ''} | {a['xy_err_mm']:.1f} | "
                   f"{'✓' if b['success'] else '✗'} {b['error'] or ''} | {b['attempts']} | {b['xy_err_mm']:.1f} |")
    return "\n".join(out)


steps = "\n".join(f"| {r['index']} | {r['id']} | {r['cmd']} | {r.get('duration', float('nan')):.2f} | "
                  f"{r.get('torque_ratio', float('nan')):.2f} | {r.get('target_err_mm', float('nan')):.3f} |"
                  for r in M["demo_results"]).replace("nan", "—")
lg = "\n".join(
    f"| {x['attempt']} | {x['stage']} | {x.get('cmd_pick', ['—'])[0] if 'cmd_pick' in x else '—'} | {x['error'] or '成功'} | "
    f"{x['dx_mm']:.2f} |" if "dx_mm" in x else f"| {x['attempt']} | {x['stage']} | — | {x['error'] or '成功'} | — |"
    for x in M["demo"]["log"])
md = f"""# {D} T102 取放（pick & place）

## 目的
延續 T101，把「桌面取料 → 放入料盤格」做成完整取放任務：量化標稱取放的精度與週期，並測試**取點定位/感測誤差**對取放的影響
（夾爪自動對中範圍、x 向殘留偏差造成的放置失敗），再加入最簡單的**失敗偵測與重取（regrasp）**恢復，比較有/無恢復的成功率。

## 模型與參數
* 實作：`sixaxis/tasks.py` 新增 `pick_place_program`、`run_pick_place`、`inhand_ok`（沿用 T007 指令、T005 夾爪、T006 料盤）。
* 放置目標：T006 料盤 4 格（格內淨寬 58 mm、方塊 50 mm → 單邊間隙 4 mm；格中心高度 = 桌面 + 方塊半高 + 板厚 4 mm）。
* 取點區：x∈[0.36, 0.44]、y∈[−0.01, 0.02] m（同 T101）。成功判定：指令全部成功、放置 XY ≤ 3 mm、Z ≤ 1.5 mm、傾角 ≤ 2°、偏航 ≤ 5°
  （傾角 = 方塊 z 軸與世界 z 軸夾角；偏航由料盤格約束，單獨記錄）。
* 夾爪沿 y 軸閉合（開口 80 mm、方塊 50 mm）→ y 向偏差由兩指自動對中；x 向沒有任何對中作用，指墊寬 ±10 mm。
* 恢復策略（`recover=True`）：抬升後檢查方塊中心相對 TCP 的偏差（XY ≤ 1.5 mm、Z ≤ 3 mm、傾角 ≤ 3°，模擬視覺/觸覺回饋）；
  不合格（GRIP_EMPTY／COLLISION／INHAND_OFFSET）→ 若仍持有則原位放回 → 鬆開並垂直退到安全高度 → 讀回方塊實際位置（模擬重新感測）→ 重取，最多 3 次；
  方塊傾倒（>10°）或被撞離桌面則回報 UNRECOVERABLE。
* 每次試驗皆使用全新場景。亂數種子 {M['seed']}。

## 可重現指令
```bash
PYTHONPATH=. python3 scripts/t102_report.py {D}
python3 scripts/t102_write_md.py {D} "<pytest 摘要>"
ruff check . && python3 -m pytest        # {PYT}
```

## 指令序列
`pick_place_program(pick, pocket)` = `transfer_program` 的 8 道指令（上方定位 → 直線下降 → 夾 → 直線抬升 → PTP 搬運 → 直線下降至格內 → 放 → 抬升）。
夾持後先垂直抬升再 PTP（T006 發現：抱著工件不可橫掃）。示範（取點指令偏 +12 mm x，放入格 2）最後一次完整執行的指令時序：

| # | id | 指令 | 時長 [s] | 力矩使用率 | 終點 TCP 誤差 [mm] |
|---|---|---|---|---|---|
{steps}

示範的取放嘗試記錄（偵測偏置 → 重取 → 放置）：

| 嘗試 | 階段 | 取點 x 指令 [m] | 結果 | 抬升後 dx [mm] |
|---|---|---|---|---|
{lg}

## 量化結果
### 1. 標稱批次（隨機取點 N={B['n']}，輪流放入 4 格，無定位誤差，無恢復）
| 指標 | 數值 |
|---|---|
| 成功率 | {B['success']}/{B['n']} |
| 放置 XY 誤差 平均 / 最大 [mm] | {B['xy_mean']:.3f} / {B['xy_max']:.3f} |
| 放置 Z 誤差 絕對最大 [mm] | {B['z_abs_max']:.3f} |
| 傾角 最大 / 偏航 最大 [deg] | {B['tilt_max']:.3f} / {B['yaw_max']:.2f} |
| 週期時間 平均 [s] | {B['cycle_mean']:.2f} |
| 速度使用率 最大 / 力矩使用率 最大 | {B['vel_util_max']:.2f} / {B['torque_util_max']:.2f} |

各格成功數：

| 格 | 中心 (x, y) [m] | 成功 |
|---|---|---|
{pk}

### 2. 取點偏差掃描（真值取點 (0.40, 0.0)，放入格 1）
沿單軸偏移「指令取點」（偏移量 = 感測/定位誤差）。左：整段一次執行（無恢復）；右：偵測 + 重取。

**x 軸偏差**
| 偏移 [mm] | 無恢復 | 放置 XY 誤差 [mm] | 有恢復 | 取點次數 | 放置 XY 誤差 [mm] |
|---|---|---|---|---|---|
{rows('x')}

**y 軸偏差**
| 偏移 [mm] | 無恢復 | 放置 XY 誤差 [mm] | 有恢復 | 取點次數 | 放置 XY 誤差 [mm] |
|---|---|---|---|---|---|
{rows('y')}

### 3. 隨機感測誤差批次（兩軸各均勻 ±{R['range_mm']} mm，N={R['n']}，相同誤差兩種策略）
| 指標 | 無恢復 | 有恢復 |
|---|---|---|
| 成功率 | {R['raw']['success']}/{R['n']} | {R['recover']['success']}/{R['n']} |
| 放置 XY 誤差 平均 / 最大 [mm] | {R['raw']['xy_mean']:.2f} / {R['raw']['xy_max']:.2f} | {R['recover']['xy_mean']:.2f} / {R['recover']['xy_max']:.2f} |
| 週期時間 平均 [s] | {R['raw']['cycle_mean']:.2f} | {R['recover']['cycle_mean']:.2f} |
| 平均取點次數 / 需重取案例 | {R['raw']['attempts_mean']:.2f} / {R['raw']['retry_cases']} | {R['recover']['attempts_mean']:.2f} / {R['recover']['retry_cases']} |

## 圖表與動畫
* 標稱批次：![batch](../media/{T}_batch.png)
* 取點偏差掃描：![sweep](../media/{T}_offset_sweep.png)
* 隨機感測誤差（▲ = 需重取）：![rand](../media/{T}_random_error.png)
* 動畫：[GIF](../media/{T}.gif)（{M['gif_MB']:.2f} MB）／[MP4](../media/{T}.mp4)（{M['mp4_MB']:.2f} MB）
* 關鍵畫面：[A](../media/{T}_key_a.png)、[B](../media/{T}_key_b.png)、[C](../media/{T}_key_c.png)
* 原始數據：`docs/test-reports/data/{T}_metrics.json`

## 問題與限制
* 無恢復時 x 向偏差在本次掃描的 ±4 mm 起即失敗（方塊偏離夾爪中心，撞到料盤側銷或放置偏位；±2 mm 未列入掃描，開發中探測 +2 mm 仍成功）；y 向在 ±14 mm 內由夾爪對中吸收；
  |y 偏差| ≥ 16 mm 時指墊壓在方塊頂面把方塊撞倒，簡單恢復無法處理（UNRECOVERABLE），需要更完整的感測與主動判斷（T208）。
* 「感測」是直接讀取模擬器的方塊位置（理想、無雜訊），並非視覺；真實系統須加入相機/力覺雜訊模型，恢復成功率會較低。
* 工具姿態固定（T007 move_l/move_c 無姿態目標），故不處理方塊 yaw 偏差；偏航僅記錄。
* 放置目標只測料盤 4 格，未測治具與疊放（T103）；工件僅單一方塊。
* 第 2 節為單一真值取點的單軸掃描，第 3 節 N=30，統計信賴區間有限。

## 結論
標稱取放 {B['success']}/{B['n']} 成功（放置 XY 誤差最大 {B['xy_max']:.2f} mm、週期平均 {B['cycle_mean']:.2f} s、力矩使用率 ≤ {B['torque_util_max']:.2f}）。
取點存在 ±12 mm 隨機誤差時，無恢復僅 {R['raw']['success']}/{R['n']} 成功，加入「抬升後檢查 → 放回 → 重新感測 → 重取」後達 {R['recover']['success']}/{R['n']}
（平均週期 {R['raw']['cycle_mean']:.2f} → {R['recover']['cycle_mean']:.2f} s）。此恢復機制與量測方法可作為 T103 疊棧、T104 輸送帶追蹤與 T208 主動判斷層的基礎。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written")
