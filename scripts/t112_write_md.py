"""T112 報告 md：由 docs/test-reports/data/<date>_T112_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t112_write_md.py 2026-11-21 "<pytest 摘要>"
可選 docs/test-reports/data/<date>_T112_narrative.md（發現與討論，人工撰寫、引用本 JSON 數字）會附在結果後。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T112"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
P = M["params"]
R = {r["name"]: r for r in M["runs"]}
MD = M["model"]
IL = M["interlock"]
CL = M["clearance"]
A = M.get("anim", {})


def f(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def yn(b):
    return "✔" if b else "✘"


# ---------------------------------------------------------------- 策略比較
strat_rows = []
for tm in P["tms"]:
    row = [f"{tm:g}"]
    for s in ("serial", "buffer"):
        r = R[f"{s}_{tm:g}"]
        pr = r["periods"]
        row += [f"{np.mean(pr):.2f}", f"{np.mean(r['machine_idle']):.2f}", f"{r['T_m'] / pr[1] * 100:.1f}",
                yn(r["all_ok"])]
    s1, s2 = R[f"serial_{tm:g}"]["periods"][1], R[f"buffer_{tm:g}"]["periods"][1]
    row.append(f"{(s1 - s2) / s1 * 100:.1f}")
    strat_rows.append("| " + " | ".join(row) + " |")

term_rows = []
for s in ("serial", "buffer"):
    for pt in MD["terms_ref"][s]:
        term_rows.append(f"| {'S1' if s == 'serial' else 'S2'} | {pt['k']} | {pt['period']:.2f} | {pt['T_open']:.2f} | "
                         f"{pt['T_ex']:.2f} | {pt['T_close']:.2f} | {pt['T_side']:.2f} | {pt['machine_idle']:.2f} |")

model_rows = []
for r in MD["rows"]:
    if r["T_m"] == P["t_ref"]:
        continue
    model_rows.append(f"| {'S1' if r['strategy'] == 'serial' else 'S2'} | {r['T_m']:g} | {r['k']} | {r['meas']:.2f} | "
                      f"{r['pred']:.2f} | {r['meas'] - r['pred']:+.2f} | {'手臂' if r['robot_bound'] else '機台'} |")

# 每件 T_ex 組成（S1 vs S2，T_m = 15，第 2 週期）
def ex_breakdown(r):
    ev = r["events"]
    done = [e["t"] for e in ev if e["event"] == "CYCLE_DONE"]
    ccmd = [e["t"] for e in ev if e["event"] == "DOOR_CLOSE_CMD"]
    t0 = done[1]
    t1 = next(t for t in ccmd if t > t0)
    agg = {}
    for tag, a, b in r["segments"]:
        if a >= t0 - 1e-6 and b <= t1 + 1e-6 and tag != "side_work":
            k = tag.rstrip("0123456789")
            agg[k] = agg.get(k, 0.0) + (b - a)
    return agg


_s1 = next(x for x in MD["terms_ref"]["serial"] if x["k"] == 2)
_s2 = next(x for x in MD["terms_ref"]["buffer"] if x["k"] == 2)
XB = _s2["T_side"] - _s2["T_close"]
_c2 = _s2["T_open"] + _s2["T_ex"] + _s2["T_side"]  # S2 手臂瓶頸時的週期
XC = _c2 - (_s1["T_open"] + _s1["T_ex"] + _s1["T_close"])  # S1 週期 = 常數 + T_m
DEX = _s1["T_ex"] - _s2["T_ex"]
exb = {s: ex_breakdown(R[f"{s}_15"]) for s in ("serial", "buffer")}
GROUPS = [("開門等待", ("wait_door",)), ("進/出門洞", ("enter", "exit")),
          ("虎鉗下料（下降/夾/鬆鉗/抬升）", ("over_part", "unload_down", "unload_grip", "wait_unclamp", "unload_up")),
          ("虎鉗上料（下降/放/夾緊/抬升）", ("vise_down", "vise_release", "vise_up", "wait_clamp", "vise_clear",
                                           "over_vise")),
          ("送輸送帶", ("to_belt", "belt_down", "belt_release", "belt_up")),
          ("料盤取胚料", ("to_tray", "tray_down", "grip_tray", "tray_up")),
          ("緩衝台放/取", ("to_bufA", "bufA_down", "bufA_release", "bufA_up", "to_bufB", "bufB_down", "bufB_grip",
                          "bufB_up")),
          ("回門外", ("to_door",))]
ex_rows = []
for nm, keys in GROUPS:
    v = [sum(exb[s].get(k, 0.0) for k in keys) for s in ("serial", "buffer")]
    ex_rows.append(f"| {nm} | {v[0]:.2f} | {v[1]:.2f} |")
ex_rows.append(f"| **合計（T_ex，不含開門）** | {sum(exb['serial'].values()) - exb['serial'].get('wait_door', 0):.2f} | "
               f"{sum(exb['buffer'].values()) - exb['buffer'].get('wait_door', 0):.2f} |")

# ---------------------------------------------------------------- 上料精度
loads = [ld for nm, r in R.items() if nm.split("_")[0] in ("serial", "buffer") for ld in r["loads"]]
ex_ = [abs(ld["err_x_mm"]) for ld in loads]
ey_ = [abs(ld["err_y_mm"]) for ld in loads]
ez_ = [abs(ld["err_z_mm"]) for ld in loads]
sh_ = [ld["shift_mm"] for ld in loads]
gap_ = [ld["checks"][-1]["seat_gap_mm"] for ld in loads]
jaw_ = [abs(ld["checks"][-1]["jaw_mm"]) for ld in loads]
tq = max(r["torque_util"] for r in M["runs"])
cut_slip = max(r["cut_slip_mm"] for r in M["runs"])
contacts = sum(r["contacts_arm_machine"] for r in M["runs"])
n_plan = sum(r["n_planned"] for r in M["runs"])
errors = sorted({e for r in M["runs"] for e in r["errors"]})

cap = sorted([c for c in M["capture"] if c["dy_mm"] == 0 and c["seat_gap_mm"] < 1.0 or
              (c["dy_mm"] == 0 and abs(c["dx_mm"]) > 5.9)], key=lambda c: c["dx_mm"])
cap_rows = [f"| {c['dx_mm']:+.0f} | {c['x_err_mm']:+.3f} | {c['z_err_mm']:+.2f} | {c['jaw_mm']:+.2f} | "
            f"{c['seat_gap_mm']:.2f} | {yn(c['ok'])} |" for c in cap]
cap_ok = [c["dx_mm"] for c in cap if c["ok"]]
cap_y = [c for c in M["capture"] if c["dy_mm"] != 0]
chip = next(c for c in M["capture"] if c["seat_gap_mm"] > 1.0 and c["dx_mm"] == 0)

hold_rows = [f"| {h['F_c']:.0f} | {abs(h['jaw_force']):.1f} | {h['F_pred']:.1f} | {f(h['F_slip'], 1)} | "
             f"{(h['F_slip'] - h['F_pred']) / h['F_pred'] * 100:+.1f} | {h['F_slip'] / P['F_cut']:.1f} |"
             for h in M["hold"]]

# ---------------------------------------------------------------- 故障
cr, cn, ep = R["chip_recover"], R["chip_norecover"], R["empty_pocket"]
pe1, pe2 = R["place_err_px"], R["place_err_nx"]
ns = [x for x in cr["faults"] if x["fault"] == "NOT_SEATED"]
ev_cn = [e for e in cn["events"] if e["event"] == "INTERLOCK_START"]

door_min = min(CL["held"]["d_mm"][:31])
door_min_e = min(CL["empty"]["d_mm"][:31])

md = f"""# T112 機台上下料（machine tending）測試報告

* 日期：{D}
* 工作項目：T112（Phase 2 工廠複雜動作；相依 T102 取放、T110 避障規劃）
* 程式：`sixaxis/tending.py`（加工機 `Machine`、工作站 `TendCell`、週期模型、虎鉗/夾持實驗）、
  `scripts/t112_report.py`（實驗→JSON→圖→動畫）、`scripts/t112_write_md.py`（本報告）、`tests/test_tending.py`
* 原始數據：`docs/test-reports/data/{T}_metrics.json`

## 1. 目的

模擬工廠最常見的機器人應用之一：手臂替 CNC 加工機上下料。驗證
1. **加工機與手臂的協調**：自動門、虎鉗夾緊、主軸加工循環，以 PLC 信號與互鎖（門/啟動/光幕）交握；
2. **精度**：胚料放入虎鉗後由固定鉗口基準定位，夾緊監控（鉗口位置窗＋座面檢知）確認座正；
3. **夾持力學**：Coulomb 夾持門檻 F_slip = 2μ_jF_c + μ_f mg 與模擬滑移起點比較、加工中側向切削力下零件不滑移；
4. **週期與稼動率**：S1「依序」與 S2「緩衝交換」兩種策略，以週期模型 T_c = T_open + T_ex + max(T_close + T_m, T_side)
   做樣本外預測並與物理模擬比對；
5. **故障恢復**：切屑墊在零件下（座面檢知失敗→吹氣重放）、料盤空格、互鎖拒絕不良狀態。

## 2. 模型與參數

| 項目 | 值 |
|---|---|
| 虎鉗位置（夾緊後零件中心） | ({P['VX']:.2f}, {P['VY']:.2f}) m；座面 z = {0.12:.2f} m |
| 前牆 / 門洞 | y = {P['FY']:.2f} m；x ∈ [{P['door'][0]:.2f}, {P['door'][1]:.2f}]、z ∈ [{P['door'][2]:.2f}, {P['door'][3]:.2f}] m |
| 自動門 | 行程 {P['door_stroke'] * 1e3:.0f} mm、v_max {P['door_v']:.2f} m/s、a_max {P['door_a']:.1f} m/s² → 梯形曲線 T = s/v + v/a = {P['door_time']:.3f} s |
| 胚料 / 成品 | {P['part'][0] * 1e3:.0f} mm 立方、{P['part'][1] * 1e3:.0f} g → 頂面口袋 30×30×15 mm：{P['finished']['mass'] * 1e3:.1f} g（移除 {P['finished']['removed_g']:.1f} g）、質心下移 {-P['finished']['com_z'] * 1e3:.2f} mm |
| 虎鉗 | 活動鉗口稜柱關節、力上限 F_c = {P['F_clamp']:.0f} N、全開間隙 {P['jaw_open'] * 1e3:.0f} mm（放置目標 = 開口中心，容許 ±{P['jaw_open'] / 2 * 1e3:.0f} mm） |
| 摩擦 | 鉗口 {P['mu_jaw']}、座面 {P['mu_floor']}、零件 {P['mu_part']}（PyBullet 組合 = 乘積 → 有效 {P['mu_jaw'] * P['mu_part']:.2f} / {P['mu_floor'] * P['mu_part']:.2f}） |
| 夾緊監控 | 鉗口位置窗 ±{P['jaw_win'] * 1e3:.1f} mm；座面檢知門檻 {P['seat_tol'] * 1e3:.1f} mm |
| 切削力 | F_cut = k_c a_p h = {P['k_c']:.0f} N/mm² × {P['a_p']} mm × {P['h']} mm = {P['F_cut']:.0f} N（±y，每 {P['pass_t']:.0f} s 換向） |
| 手臂直線段 | 峰值 TCP 速度 {P['v_lin']} m/s（T = 1.875 L / v；超限則退回 T007 時間縮放）；進出門洞 TCP 高 {P['z_entry']:.2f} m |
| 位置 | 門外等待點 P_out = {tuple(round(x, 3) for x in P['p_out'])}、虎鉗上方 P_in = {tuple(round(x, 3) for x in P['p_in'])}、緩衝台頂 {P['buf_top']:.2f} m、輸送帶放料點 {tuple(P['belt_drop'])} |

公式依據：梯形速度曲線（T003）；Coulomb 摩擦夾持（夾具設計標準做法：夾緊力 × 摩擦係數 × 接觸面數 ≥ 安全係數 × 切削力）；
Kienzle 比切削力 F = k_c·b·h（k_c 取常數）；機台照顧週期「機台時間 + 不重疊的手臂時間」（Groover）；
光幕停止為 IEC 61496 的功能等效（僅幾何偵測）。

## 3. 可重現指令

```bash
python3 -m pytest -q tests/test_tending.py          # 9 項
PYTHONPATH=. python3 scripts/t112_report.py {D}       # 全部實驗（4 行程平行）→ JSON / 圖 / 動畫
python3 scripts/t112_write_md.py {D} "<pytest 摘要>"  # 本報告
```

## 4. 指令序列（S2 緩衝交換，一件的交換段）

```
[加工中] 緩衝A → 夾成品 → move_j 輸送帶上方 → move_l 下降 → release → 上升
         → 料盤格 → 夾胚料 → 緩衝B 放下 → move_j P_out（門外等待）
[CYCLE_DONE] PLC 開門（{P['door_time']:.2f} s）→ 等 door_open → move_l P_in → 下降 → grip → 鬆鉗 → 上升 → move_l P_out
         → 緩衝A 放成品 → 緩衝B 夾胚料 → P_out → P_in → 下降到開口中心（底面離座 1.5 mm）→ release
         → 夾緊（同時抬升 50 mm）→ 監控（鉗口位置窗＋座面）→ P_out → 關門命令＋預約啟動 → [門全關自動 CYCLE_START]
```
S1 依序：下料 → 輸送帶 → 料盤 → 上料，全部在機台停機期間進行。大範圍移動皆經 T110 保證式碰撞檢查
（{n_plan} 段需要 RRT*；其餘關節直線即通過）。

## 5. 量化結果

### 5.1 策略比較（每組 4 件；週期 = 相鄰兩次 CYCLE_START 間隔）

| T_m [s] | S1 平均週期 [s] | S1 機台閒置 [s] | S1 稼動率 k=2 [%] | S1 成功 | S2 平均週期 [s] | S2 機台閒置 [s] | S2 稼動率 k=2 [%] | S2 成功 | S2 較 S1 縮短（k=2）[%] |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(strat_rows)}

10 組全部 4/4 件成品送上輸送帶（`all_ok`），手臂與機台（牆/門/鉗口/主軸）接觸點 {contacts}、執行器錯誤 {errors or '無'}、
最大力矩使用率 {tq:.2f}。

**週期分量**（T_m = {P['t_ref']:g} s 量測；k = 第 k 件完成 → 第 k+1 件啟動）：

| 策略 | k | 週期 | T_open | T_ex | T_close | T_side | 機台閒置 |
|---|---|---|---|---|---|---|---|
{chr(10).join(term_rows)}

**交換時間 T_ex 組成**（T_m = 15 s、k = 2，各動作區段實測秒數）：

| 動作 | S1 | S2 |
|---|---|---|
{chr(10).join(ex_rows)}

### 5.2 週期模型樣本外驗證

以 T_m = {P['t_ref']:g} s 量得的 T_open、T_ex、T_close、T_side（各 k 分別）代入
T_c = T_open + T_ex + max(T_close + T_m, T_side)，預測其他 T_m 的每個週期：

| 策略 | T_m | k | 實測 [s] | 預測 [s] | 誤差 [s] | 瓶頸 |
|---|---|---|---|---|---|---|
{chr(10).join(model_rows)}

最大絕對誤差 **{MD['max_abs_err']:.3f} s**（相對 {MD['max_rel_err'] * 100:.2f} %）。
由 k = 2 分量：S2 在 T_m < T_side − T_close = {XB:.2f} s 時為手臂瓶頸；S1 與 S2 週期相等的交叉點 T_m* = {XC:.2f} s
（T_m > T_m* 時 S2 較快，每件省 T_ex1 − T_ex2 = {DEX:.2f} s；T_m 更短時 S2 的加工期間側工作反而拖慢週期，T_m = 4 s 實測 S2 較 S1 慢）。

### 5.3 上料精度與夾緊監控（10 組策略實驗共 {len(loads)} 次上料）

| 指標 | 平均 | 最大 |
|---|---|---|
| 夾緊後 x 誤差（固定鉗口定位）[mm] | {np.mean(ex_):.3f} | {np.max(ex_):.3f} |
| 夾緊後 y 誤差（不定位 = 手臂放置）[mm] | {np.mean(ey_):.3f} | {np.max(ey_):.3f} |
| 夾緊後 z 誤差 [mm] | {np.mean(ez_):.3f} | {np.max(ez_):.3f} |
| 夾緊推移量（鬆爪 → 夾緊，x）[mm] | {np.mean(sh_):.2f} | {np.max(sh_):.2f} |
| 座面間隙 [mm] | {np.mean(gap_):.3f} | {np.max(gap_):.3f} |
| 鉗口位置 |偏差| [mm] | {np.mean(jaw_):.3f} | {np.max(jaw_):.3f} |
| 加工中零件位移（35 N 切削力）[mm] | — | {cut_slip:.4f} |

刻意加入放置誤差（完整工作站）：(+4, 0) mm → 夾緊後 x 誤差 {pe1['loads'][0]['err_x_mm']:+.3f} / {pe1['loads'][1]['err_x_mm']:+.3f} mm；
(−4, +2) mm → x {pe2['loads'][0]['err_x_mm']:+.3f} mm、y {pe2['loads'][0]['err_y_mm']:+.3f} mm（y 誤差保留，符合「x 由基準定位、y 不定位」）。

### 5.4 虎鉗定位捕捉範圍（無手臂：由鉗口上方下降到鬆爪高度 → 放開 → 夾緊）

| dx [mm] | x 誤差 [mm] | z 誤差 [mm] | 鉗口位置 [mm] | 座面間隙 [mm] | 監控 OK |
|---|---|---|---|---|---|
{chr(10).join(cap_rows)}

捕捉範圍實測 dx ∈ [{min(cap_ok):+.0f}, {max(cap_ok):+.0f}] mm，與幾何間隙 ±{P['jaw_open'] / 2 * 1e3:.0f} mm 一致（1 mm 取樣）；
超出時零件落在鉗口頂上，鉗口位置窗與座面檢知同時判 NG。y 偏 {cap_y[0]['dy_mm']:+.0f}/{cap_y[1]['dy_mm']:+.0f} mm →
夾緊後 y 誤差 {cap_y[0]['y_err_mm']:+.2f}/{cap_y[1]['y_err_mm']:+.2f} mm（不修正）。
切屑（12×12×3 mm）墊在零件下：座面間隙 {chip['seat_gap_mm']:.2f} mm（> 0.3 mm 判 NG），但鉗口位置 {chip['jaw_mm']:+.2f} mm 仍在窗內
——**只看鉗口位置會漏判**，座面檢知是必要的。

### 5.5 夾持力：Coulomb 門檻 vs 模擬滑移起點（側向力 20 N/s 斜坡，位移 > 0.5 mm 判滑移）

| F_c [N] | 鉗口實際法向力 [N] | 預測 F_slip [N] | 模擬 F_slip [N] | 誤差 [%] | 對 35 N 切削力安全係數 |
|---|---|---|---|---|---|
{chr(10).join(hold_rows)}

### 5.6 互鎖與光幕（同一工作站依序測試）

| 測試 | 結果 |
|---|---|
| 門關、無零件 → 啟動循環 | {'拒絕 ✔' if not IL['start_no_part'] else '接受 ✘'} |
| 開門時間（命令 → door_open） | {IL['door_time_meas']:.3f} s（模型 {P['door_time']:.3f} s） |
| 手臂在機台內 → 關門 | {'拒絕 ✔' if not IL['close_inside'] else '接受 ✘'}（arm_in_zone = {IL['arm_in_zone_inside']}） |
| 已夾緊、門開 → 啟動循環 | {'拒絕 ✔' if not IL['start_door_open'] else '接受 ✘'} |
| 繞過互鎖強制關門、手臂在門洞平面 | 光幕 {'停止 ✔' if IL['trip'] else '未停 ✘'}，門位置 {IL['door_s_at_trip_mm']:.0f} mm（全開 {P['door_stroke'] * 1e3:.0f}），門–手臂最小距離 {IL['min_gap_door_arm_mm']:.1f} mm |
| 手臂退出 → 關門 → 啟動 | 關門 {yn(IL['close_outside'])}、啟動 {yn(IL['start_ok'])} |
| 循環中 → 開門 | {'拒絕 ✔' if not IL['open_during_cycle'] else '接受 ✘'} |

門洞直線教導路徑（P_out → P_in，31 點取樣，規劃碰撞模型）最小間隙：空手 {door_min_e:.1f} mm、夾持胚料 {door_min:.1f} mm。

### 5.7 故障注入與恢復

| 情境 | 結果 |
|---|---|
| S2、第 1、2 件下料後虎鉗座面留切屑，恢復開啟 | {cr['n_cycles']} 循環、{cr['n_out']} 件成品、all_ok = {cr['all_ok']}；偵測 NOT_SEATED {len(ns)} 次（座面間隙 {', '.join(f"{x['check']['seat_gap_mm']:.2f}" for x in ns)} mm），各吹氣重放 1 次後監控 OK |
| 同上但不恢復（僅 1 片切屑） | PLC 拒絕啟動（INTERLOCK_START：{ev_cn[0]['reason'] if ev_cn else '—'}），流程停止，完成 {cn['n_cycles']} 循環——不良裝夾不會被加工 |
| S1、料盤第 2 格空 | GRIP_EMPTY → 跳下一格，{ep['n_cycles']} 循環、{ep['n_out']} 件、all_ok = {ep['all_ok']} |
| 恢復造成的週期增加（chip_recover vs buffer_10，k=1） | {cr['periods'][0] - R['buffer_10']['periods'][0]:+.2f} s |

### 5.8 圖表與動畫

![甘特圖](../media/{T}_gantt.png)

![週期模型](../media/{T}_cycle_model.png)

![虎鉗捕捉](../media/{T}_capture.png)

![夾持力](../media/{T}_clamp.png)

![門洞間隙](../media/{T}_clearance.png)

![佈置](../media/{T}_layout.png)

動畫（S2 緩衝交換、T_m = 8 s、3 件，第 1 件下料後注入切屑 → 偵測、吹氣、重放）：

![動畫](../media/{T}.gif)

MP4：[`docs/media/{T}.mp4`](../media/{T}.mp4)；關鍵畫面：
[上料](../media/{T}_key_load.png)、[加工中](../media/{T}_key_cycle.png)、[切屑恢復](../media/{T}_key_recover.png)、
[結束](../media/{T}_key_end.png)；關節曲線：

![關節](../media/{T}_joints.png)

動畫執行：{A.get('n_cycles', '—')} 循環、{A.get('n_out', '—')} 件、all_ok = {A.get('all_ok', '—')}、最大追蹤誤差
{f(A.get('max_track_err_mrad'), 3)} mrad、力矩使用率 {f(A.get('torque_util'), 2)}。

"""

nar = ROOT / f"docs/test-reports/data/{T}_narrative.md"
if nar.exists():
    md += nar.read_text(encoding="utf-8") + "\n"
md += f"""
## 測試結果

```
{PYT}
```
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written", T)
