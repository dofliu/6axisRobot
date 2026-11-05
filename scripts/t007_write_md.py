"""T007 報告 md：由 docs/test-reports/data/<date>_T007_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t007_write_md.py 2026-11-05 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T007"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))


def prog_table(res):
    out = ["| # | id | 指令 | 時長 [s] | 速度使用率 | 力矩使用率 | 終點 TCP 誤差 [mm] |", "|---|---|---|---|---|---|---|"]
    for r in res:
        f = lambda k, p=2: f"{r[k]:.{p}f}" if k in r else "—"  # noqa: E731
        out.append(f"| {r['index']} | {r['id']} | {r['cmd']} | {f('duration')} | {f('vel_ratio')} | "
                   f"{f('torque_ratio')} | {f('tcp_err_mm', 3)} |")
    return "\n".join(out)


A, B = M["programs"]["A"], M["programs"]["B"]
err = "\n".join(f"| {e['case']} | {e['code']} | #{e['index']} | {e['message']} | {e['executed_ok']} |" for e in M["errors"])
md = f"""# {D} T007 指令介面（JSON / DSL）

## 目的
提供以文字下達的指令介面：解析與驗證（JSON 或 DSL）→ 執行器（運動學 dry-run 或 PyBullet 物理）→ 結構化錯誤回報，
作為 Phase 2 工廠動作與 Phase 3 RL 主動判斷層的共同動作語彙。

## 模型與參數
* 實作：`sixaxis/commands.py`；建構於 T003 軌跡（`joint_ptp`、`LinePath`、`ArcPath`、`sample_cartesian`、`fit_duration`）、
  T005 `GraspScene`（計算力矩手臂 + 夾爪）、T006 `Workcell`（碰撞查詢）。
* 指令：`move_j`（`q` / `q_deg` / `tcp`）、`move_l`、`move_c`（`via`+`tcp`）、`grip`（`force`，預設 40 N）、`release`、`wait`、`set_speed`；
  共用可選 `T`、`speed`、`profile`（quintic/trapezoid）、`id`。TCP 目標一律為夾持中心，`move_j tcp` 為工具朝下，
  `move_l/move_c` 保持當下姿態。
* 時間：未給 `T` → `fit_duration` 求速度與力矩使用率 ≤ 0.5·speed 的最短時間（0.5 為負載/抓持餘裕）；
  給了 `T` 但超限 → `LIMIT` 錯誤，不默默改時間。
* 錯誤碼：SYNTAX / RANGE / UNREACHABLE / LIMIT / COLLISION / GRIP_EMPTY；程式先整體驗證，驗證失敗不執行任何指令；
  執行中遇錯即停（`stop_on_error=False` 可繼續）。
* 碰撞：每 50 ms 查詢手臂與場景靜態體（工作台/料盤/治具/輸送帶…）距離 ≤ 0 → COLLISION。
* 空夾偵測：夾持 0.5 s 後兩指行程已到底或兩指接觸力 < 20 % 設定力 → GRIP_EMPTY。

## 可重現指令
```bash
PYTHONPATH=. python3 scripts/t007_report.py {D}
python3 scripts/t007_write_md.py {D} "<pytest 摘要>"
ruff check . && python3 -m pytest        # {PYT}
```

## 指令序列
程式 A（DSL，桌面方塊 → 料盤格 3）與程式 B（JSON，輸送帶擋料位方塊 → 料盤格 0）見 `scripts/t007_report.py` 的 `PROG_A` / `PROG_B`。
DSL 範例：
```
move_j tcp=0.45,0,0.30
move_l tcp=0.45,0,0.125
grip force=40
move_c via=0.52,-0.0625,0.25 tcp=0.495,-0.125,0.25
release
```

## 量化結果
### 物理執行
| 程式 | 成功 | 時間 [s] | 放置 XY 誤差 [mm] | 放置 Z 誤差 [mm] |
|---|---|---|---|---|
| A（DSL） | {A['ok']} | {A['t_s']:.2f} | {A['place_xy_err_mm']:.3f} | {A['place_z_err_mm']:.3f} |
| B（JSON） | {B['ok']} | {B['t_s']:.2f} | {B['place_xy_err_mm']:.3f} | {B['place_z_err_mm']:.3f} |

* 全部運動指令終點 TCP 誤差最大 {M['max_tcp_err_mm']:.3f} mm；速度使用率最大 {M['max_vel_ratio']:.2f}、力矩使用率最大 {M['max_torque_ratio']:.2f}（上限 1.0）。
* 程式 A 的 dry-run 預估總時間 {M['dry_total_s']:.2f} s，物理執行 {A['t_s']:.2f} s（差在夾持/釋放等待與取整）。
* 結束時手臂與場景最小間隙 {M['final_clearance_mm']:.1f} mm；JSON↔DSL 往返解析一致：{M['parse_roundtrip']}。

程式 A 逐指令：
{prog_table(A['results'])}

程式 B 逐指令：
{prog_table(B['results'])}

### 錯誤回報（皆為實際執行結果）
| 案例 | 錯誤碼 | 指令序號 | 訊息 | 錯誤前已完成指令數 |
|---|---|---|---|---|
{err}

## 圖表與動畫
* 動畫：[GIF](../media/{T}.gif)（{M['gif_MB']:.1f} MB）／[MP4](../media/{T}.mp4)（{M['mp4_MB']:.2f} MB），{M['frames']} 幀
* ![TCP 與指力](../media/{T}_tcp_forces.png)
* ![逐指令誤差](../media/{T}_cmd_errors.png)
* 關鍵畫面：![a](../media/{T}_key_a.png) ![b](../media/{T}_key_b.png) ![c](../media/{T}_key_c.png)

## 問題與限制
* `move_l/move_c` 保持當下姿態，尚無姿態目標（yaw）參數；之後視需要擴充。
* 力矩檢查使用 T001 RNE 模型（未含夾爪與工件質量），因此 0.5 的名目使用率是為此預留的餘裕。
* 碰撞僅檢查手臂對靜態場景，不含夾持中工件對場景（T110 避障規劃處理）；取放需自行加入抬升/中繼點。
* `grip` 以接觸力判斷落空，無法區分「夾到錯誤物件」；後續主動判斷層（T208）再處理。
* 執行器遇錯即停，不含自動恢復（T208）。

## 結論
文字指令（JSON/DSL）可完整驅動取放與輸送帶取料，錯誤以序號與錯誤碼回報，並可在不啟動物理的 dry-run 下預檢可達性/限位/超速。
下一步可進入 Phase 2（T101 點到點搬運）與 T201（Gymnasium reach 環境）。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
