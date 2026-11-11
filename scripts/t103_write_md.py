"""T103 報告 md：由 docs/test-reports/data/<date>_T103_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t103_write_md.py 2026-11-11 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T103"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
TW, PY, CO = M["tower"], M["pyramid"], M["column"]
a_mm = M["a"] * 1e3


def ok(b):
    return "✓" if b else "✗"


layers = "\n".join(
    f"| {f['layer'] + 1} | {f['dx_mm']:+.2f} | {f['dy_mm']:+.2f} | {f['z_err_mm']:+.3f} | {f['tilt_deg']:.3f} | "
    f"{f['disturb_mm']:.2f} | {c:.2f} |" for f, c in zip(TW["final"], TW["cycles_s"], strict=True))
steps = "\n".join(f"| {r['index']} | {r['id']} | {r['cmd']} | {r.get('duration', float('nan')):.2f} | "
                  f"{r.get('torque_ratio', float('nan')):.2f} | {r.get('target_err_mm', float('nan')):.3f} |"
                  for r in M["tower_steps"]).replace("nan", "—")
sweep = "\n".join(
    f"| {s['dz_mm']:.0f} | {s['speed']:.2f} | {ok(s['success'])} {s['built']}/{s['n']} | {s['xy_max_mm']:.2f} | "
    f"{s['z_abs_max_mm']:.3f} | {s['tilt_max_deg']:.3f} | {s['disturb_max_mm']:.2f} | {s['cycle_mean_s']:.2f} |"
    for s in M["sweep"])


def pat(name, p):
    return (f"| {name} | {ok(p['success'])} {p['built']}/{p['n']} | {p['xy_max_mm']:.2f} | {p['z_abs_max_mm']:.3f} | "
            f"{p['tilt_max_deg']:.3f} | {p['disturb_max_mm']:.2f} | {p['cycle_mean_s']:.2f} | {p['total_s']:.1f} | "
            f"{p['torque_util']:.2f} |")


pats = "\n".join([pat(f"垂直疊棧 {TW['n']} 層", TW), pat("金字塔 3-2-1（壓縫）", PY), pat("柱式 2×3（對齊）", CO)])
stag = "\n".join(f"| {s['d_mm']:.1f} | {s['n_crit']:.2f} | {s['pred']} | {s['built']} | {s['error'] or '—'} | "
                 f"{ok(s['pred'] == s['built'])} |" for s in M["stagger"])
stag_match = sum(s["pred"] == s["built"] for s in M["stagger"])
es = sorted({r["e_mm"] for r in M["rand"]})
rand_rows = []
for e in es:
    r = [x for x in M["rand"] if x["e_mm"] == e]
    rand_rows.append(
        f"| ±{e:.0f} | {len(r)} | {sum(x['built'] == 6 for x in r)}/{len(r)} | {sum(x['pred'] == 6 for x in r)}/{len(r)} | "
        f"{sum(x['built'] == x['pred'] for x in r)}/{len(r)} | "
        f"{', '.join(str(x['built']) for x in r)} | {', '.join(str(x['pred']) for x in r)} |")
rand_rows = "\n".join(rand_rows)
S0, S1 = M["sweep"][0], M["sweep"][1]
prefix_n = sum(x["min_margin_mm"] > 0 and x["pred"] < 6 for x in M["rand"])
rand_match = sum(x["built"] == x["pred"] for x in M["rand"])
mism = [x for x in M["rand"] if x["built"] != x["pred"]]
mism_txt = ("全部一致。" if not mism else "不一致案例：" + "；".join(
    f"±{x['e_mm']:.0f} mm #{x['trial']}（預測 {x['pred']}、模擬 {x['built']}、最小餘裕 {x['min_margin_mm']:.1f} mm）"
    for x in mism) + "。")
demo = "\n".join(f"| {k} | {v['built']}/{v['n']} | {v['error'] or '—'} | {v['total_s']:.1f} |"
                 for k, v in M["demo"].items())

md = f"""# {D} T103 疊棧／碼垛

## 目的
延續 T102 取放，把方塊**逐層堆疊**：垂直疊棧（單柱多層）、壓縫金字塔與柱式碼垛。量化每層放置精度、累積誤差、後續放置對已放方塊的擾動、
週期時間與力矩使用率；並以**靜力學質心判據**預測「放歪的堆疊何時倒塌」，與物理模擬逐案比對，做為之後主動判斷（T208）「這一塊還能不能往上放」的依據。

## 模型與參數
* 實作：`sixaxis/stacking.py`（樣式 `tower_slots`／`pyramid_slots`／`column_slots`、`stack_program`、`run_stack`、判據 `stack_margins`／`stable_height`）；
  只組合 T007 指令（`tasks.transfer_program`），手臂計算力矩控制、夾持力 40 N（T005）。
* 工件：50 mm 方塊、0.20 kg；層高 50 mm；第 k 層（k 從 0）中心高度 = 桌面 0.10 + 0.025 + 0.05·k m。
* 供料：每放完一塊，在供料點 ({M['feed_xy'][0]}, {M['feed_xy'][1]}) 生成下一塊（模擬供料器）。疊棧位置 ({M['stack_xy'][0]}, {M['stack_xy'][1]})；
  碼垛中心 ({M['pyr_xy'][0]}, {M['pyr_xy'][1]})，同層中心距 {M['pitch'] * 1e3:.0f} mm（間隙 2 mm）。
* 夾爪沿 y 閉合且工具姿態固定 → 同層方塊只沿 x 排列（指墊不需伸入相鄰方塊之間）。
* 搬運高度隨層數升高：TCP = max(0.25, 放置高度 + 0.07) m。抬升後以 `inhand_ok` 檢查方塊在夾爪內置中（T102）。
* 每放一塊並退回後檢查所有已放方塊是否在位（|z 誤差| ≤ 5 mm、傾角 ≤ 5°、XY ≤ 15 mm），否則記 `COLLAPSE@k` 並停止；
  全部放完靜置後再量一次。成功判定：全部在位且 XY ≤ 2 mm、|Z| ≤ 1.5 mm、傾角 ≤ 2°。

### 靜力穩定判據（依據）
剛體方塊單柱堆疊：對每個介面，其上方所有方塊的合質心水平投影必須落在該介面的接觸區（上下兩方塊底面的交集）內；
否則重力對接觸區邊緣的力矩無法由只能受壓的法向接觸力平衡而傾倒（靜力學零力矩條件，Meriam & Kraige《Statics》；
積木懸伸問題，Paterson & Zwick, "Overhang", Amer. Math. Monthly 116(1), 2009）。
等質量、每層等距錯位 d 的 n 層塔：介面上方有 m 塊時合質心相對下方方塊偏 d(m+1)/2，接觸區邊界在 a/2，
最嚴格為最底介面（m = n−1）→ **d < a/n**（a = {a_mm:.0f} mm）。`stable_height` 對逐層堆疊的每個前綴都檢查所有介面。

## 可重現指令
```bash
PYTHONPATH=. python3 scripts/t103_report.py {D}
python3 scripts/t103_write_md.py {D} "<pytest 摘要>"
ruff check . && python3 -m pytest        # {PYT}
```

## 指令序列
每塊 8 道指令（`stack_program`）：供料點上方定位 → 直線下降 → 夾 → 直線抬升 → PTP 搬運到堆疊上方 → 直線下降到層高 → 放 → 直線退回。
{TW['n']} 層塔最後一塊（第 {TW['n']} 層）的指令時序：

| # | id | 指令 | 時長 [s] | 力矩使用率 | 終點 TCP 誤差 [mm] |
|---|---|---|---|---|---|
{steps}

## 量化結果
### 1. 垂直疊棧 {TW['n']} 層（標稱，無放置誤差）
結果：{ok(TW['success'])} {TW['built']}/{TW['n']} 層；總時間 {TW['total_s']:.1f} s；速度／力矩使用率最大 {TW['vel_util']:.2f}／{TW['torque_util']:.2f}；
指令終點 TCP 誤差最大 {TW['tcp_err_mm']:.3f} mm。各層最終狀態（靜置後）：

| 層 | dx [mm] | dy [mm] | z 誤差 [mm] | 傾角 [deg] | 被後續放置擾動 [mm] | 週期 [s] |
|---|---|---|---|---|---|---|
{layers}

### 2. 樣式比較
| 樣式 | 結果 | XY 最大 [mm] | \\|Z\\| 最大 [mm] | 傾角最大 [deg] | 擾動最大 [mm] | 平均週期 [s] | 總時間 [s] | 力矩使用率 |
|---|---|---|---|---|---|---|---|---|
{pats}

### 3. 釋放高度與速度（6 層塔）
釋放高度 = 放開時方塊底面高於標稱接觸面的距離（>0 即讓方塊自由落下該距離）。

| 釋放高度 [mm] | speed | 結果 | XY 最大 [mm] | \\|Z\\| 最大 [mm] | 傾角最大 [deg] | 擾動最大 [mm] | 平均週期 [s] |
|---|---|---|---|---|---|---|---|
{sweep}

標稱高度釋放（0 mm）時夾爪把方塊壓在堆疊上再放開，下層被擠動最多 {S0['disturb_max_mm']:.2f} mm；
抬高 1 mm 釋放即降到 {S1['disturb_max_mm']:.2f} mm，且 XY 最大誤差 {S0['xy_max_mm']:.2f} → {S1['xy_max_mm']:.2f} mm。

### 4. 錯位塔：倒塌層數 vs 解析判據（每層沿 +x 錯位 d，目標 8 層）
| d [mm] | a/d | 解析可堆層數 | 模擬可堆層數 | 模擬結果 | 一致 |
|---|---|---|---|---|---|
{stag}

一致 {stag_match}/{len(M['stagger'])}。

### 5. 隨機放置誤差（6 層塔，每塊兩軸各均勻 ±e，種子 {M['seed']}）
| e [mm] | N | 模擬堆滿 6 層 | 判據預測堆滿 | 層數一致 | 模擬層數 | 預測層數 |
|---|---|---|---|---|---|---|
{rand_rows}

判據與模擬層數一致 {rand_match}/{len(M['rand'])}。{mism_txt}
其中 {prefix_n} 案「6 層全部放上後」的整體判據餘裕 > 0，但堆到一半的前綴已不穩（上層尚未放上、缺少配重）而在中途倒塌——
逐層堆疊必須對**每個中間狀態**檢查，而不是只檢查最終樣式。

## 圖表與動畫
* {TW['n']} 層塔各層誤差與週期：![tower](../media/{T}_tower.png)
* 穩定性（左：錯位塔；右：隨機誤差，預測 vs 模擬）：![stability](../media/{T}_stability.png)
* 釋放高度與碼垛樣式（量測外框）：![patterns](../media/{T}_patterns.png)
* 動畫（2× 速，GIF 每 3 幀取 1；5 層塔 → 金字塔 3-2-1 → 錯位 13 mm 塔倒塌）：![anim](../media/{T}.gif)
  [GIF](../media/{T}.gif)（{M['gif_MB']:.2f} MB）／[MP4](../media/{T}.mp4)（{M['mp4_MB']:.2f} MB）
* 關鍵畫面：[A 5 層塔](../media/{T}_key_a.png)、[B 金字塔](../media/{T}_key_b.png)、[C 錯位塔倒塌](../media/{T}_key_c.png)
* 原始數據：`docs/test-reports/data/{T}_metrics.json`

動畫三段的實際結果：

| 段落 | 堆成層數 | 結果 | 模擬時間 [s] |
|---|---|---|---|
{demo}

## 問題與限制
* 供料為「在供料點直接生成方塊」的理想供料器，位置無誤差；未接輸送帶（留給 T104）。
* 工具姿態固定、夾爪沿 y 閉合，同層只能沿 x 排列；無法做逐層旋轉 90° 的交錯碼垛或 y 向並排（需 move_l 姿態目標與更大指間隙）。
* 穩定判據為剛體、等質量、無偏航、單柱的靜力條件，不含放開瞬間的動態衝擊與接觸柔度；餘裕接近 0 的案例模擬結果可能與判據差一層。
* 倒塌偵測直接讀模擬器方塊位姿（理想感測）；倒塌後不做清理/重建。
* 週期隨層數增加（{TW['cycles_s'][0]:.2f} → {TW['cycles_s'][-1]:.2f} s）：夾持後以直線垂直抬升到「放置高度 + 0.07 m」，高層的抬升行程長；未做路徑/時間最佳化。
* 高度只測到 {TW['n']} 層（0.5 m）；更高需檢查可達範圍與手腕碰撞。第 5 節每個誤差等級 N=6，統計量有限。

## 結論
垂直疊棧 {TW['built']}/{TW['n']} 層成功：XY 誤差最大 {TW['xy_max_mm']:.2f} mm、|Z| 最大 {TW['z_abs_max_mm']:.3f} mm、傾角最大 {TW['tilt_max_deg']:.3f}°、
平均週期 {TW['cycle_mean_s']:.2f} s、力矩使用率 ≤ {TW['torque_util']:.2f}；金字塔 3-2-1 與柱式 2×3 碼垛皆 {PY['built']}/{PY['n']}、{CO['built']}/{CO['n']} 完成。
錯位塔的倒塌層數與解析判據 d < a/n 一致 {stag_match}/{len(M['stagger'])}，隨機放置誤差下一致 {rand_match}/{len(M['rand'])}，
表示可用 `stable_height` 在放置前預判堆疊是否會倒。下一步 T104 輸送帶追蹤抓取。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written")
