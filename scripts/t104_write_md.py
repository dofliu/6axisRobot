"""T104 報告 md：由 docs/test-reports/data/<date>_T104_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t104_write_md.py 2026-11-12 "<pytest 摘要>"
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T104"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
P, N0 = M["params"], M["nominal"]
V0 = M["v0"]


def ok(b):
    return "✓" if b else "✗"


def f(x, fmt=".2f"):
    return "—" if x is None else format(x, fmt)


def row_cycle(tag, m):
    pl = m.get("place") or {}
    return (f"| {tag} | {ok(m['success'])} {m.get('method', '')} | {m.get('error') or m.get('track_error') or '—'} | "
            f"{f(m.get('track_rms_mm'))} | {f(m.get('track_max_mm'))} | {f(m.get('dx_mm'), '+.2f')} | "
            f"{f(m.get('x_end'), '.3f')} | {f(pl.get('xy_err_mm'))} | {f(m.get('cycle_s'))} | "
            f"{f(m.get('vel_util'))} | {f(m.get('torque_util'))} |")


speeds = "\n".join(row_cycle(f"{s['belt_speed']:.2f}", s) for s in M["speeds"])
spd_ok = [s for s in M["speeds"] if s["success"]]
spd_no = [s for s in M["speeds"] if not s["success"]]
FB = M["fallback"]

mis_rows = []
for eps in sorted({x["eps"] for x in M["mismatch"]}):
    o = next(x for x in M["mismatch"] if x["eps"] == eps and x["mode"] == "open")
    c = next(x for x in M["mismatch"] if x["eps"] == eps and x["mode"] == "track")
    mis_rows.append(
        f"| {eps * 100:+.0f} | {o['pred_end_mm']:+.2f} | {o['track_end_mm']:+.2f} | {o['dx_mm']:+.2f} | "
        f"{ok(o['success'])} {o['error'] or ''} | {c['track_end_mm']:+.2f} | {c['dx_mm']:+.2f} | "
        f"{c['v_est_grip']:.4f} | {ok(c['success'])} |")
mis_rows = "\n".join(mis_rows)
op = [x for x in M["mismatch"] if x["mode"] == "open"]
cl = [x for x in M["mismatch"] if x["mode"] == "track"]
pred_dev = max(abs(x["track_end_mm"] - x["pred_end_mm"]) for x in op)
dt_grip = float(np.mean([x["t_grip"] - x["t_det"] for x in op]))

noise_rows = []
for sg in sorted({n["sigma_mm"] for n in M["noise"]}):
    nn = [n for n in M["noise"] if n["sigma_mm"] == sg]
    noise_rows.append(
        f"| {sg:.1f} | {sum(n['success'] for n in nn)}/{len(nn)} | {np.mean([n['track_rms_mm'] for n in nn]):.2f} | "
        f"{max(n['track_max_mm'] for n in nn):.2f} | {max(abs(n['dx_mm']) for n in nn):.2f} | "
        f"{max(n['place']['xy_err_mm'] for n in nn if n['success']):.2f} | {max(n['torque_util'] for n in nn):.2f} | "
        f"{np.sqrt(M['ab_noise_ratio']) * sg:.2f} |")
noise_rows = "\n".join(noise_rows)
n_ok = sum(n["success"] for n in M["noise"])

lat_rows = "\n".join(
    f"| {x['dy_feed_mm']:+.0f} | {ok(x['success'])} | {x.get('error') or '—'} | {f(x.get('dy_mm'), '+.3f')} | "
    f"{f(x.get('track_rms_mm'))} | {f((x.get('place') or {}).get('xy_err_mm'))} |" for x in M["lateral"])

rt = [x for x in M["rand"] if x["mode"] == "track"]
ro = [x for x in M["rand"] if x["mode"] == "open"]


def rstat(rr):
    g = [x for x in rr if x["success"]]
    errs = {}
    for x in rr:
        if not x["success"]:
            errs[x["error"]] = errs.get(x["error"], 0) + 1
    e = [abs(x["track_end_mm"]) for x in rr if "track_end_mm" in x]
    return (f"{len(g)}/{len(rr)} | {np.mean(e):.2f} | {np.max(e):.2f} | "
            f"{(max(x['place']['xy_err_mm'] for x in g) if g else float('nan')):.2f} | "
            f"{(np.mean([x['cycle_s'] for x in g]) if g else float('nan')):.2f} | "
            f"{'、'.join(f'{k}×{v}' for k, v in errs.items()) or '—'}")


def flow_row(fl):
    its = fl["items"]
    return (f"| {fl['period_s']:.0f} | {fl['placed']}/{fl['n']} | {fl['tracked']} | {fl['static']} | {fl['total_s']:.1f} | "
            f"{fl['throughput_ppm']:.2f} | {max(i['xy_err_mm'] for i in its):.2f} | "
            f"{', '.join(format(i['t_done'] - i['t_feed'], '.1f') for i in its)} |")


flows = "\n".join(flow_row(fl) for fl in M["flows"])
F_all = [fl for fl in M["flows"] if fl["tracked"] == fl["n"]]
F_mix = [fl for fl in M["flows"] if fl["tracked"] < fl["n"]]
DF = M["demo"]["flow"]
demo = "\n".join([
    f"| 連續來料 {V0:.2f} m/s、週期 {DF['period_s']:.0f} s | {DF['placed']}/{DF['n']}（追蹤 {DF['tracked']}、靜態 {DF['static']}） | {DF['total_s']:.1f} |",
    f"| 帶速 0.25 m/s 單件 | {ok(M['demo']['fast']['success'])} {M['demo']['fast']['method']} | {M['demo']['fast']['cycle_s']:.1f} |",
    f"| 帶速 0.30 m/s 單件（超窗） | {ok(M['demo']['too_fast']['success'])} {M['demo']['too_fast']['method']} | {M['demo']['too_fast']['cycle_s']:.1f} |",
])
PL = N0["place"]
sumT = P["T_SYNC"] / 2 + P["T_DESC"] + P["T_GRIP_NOM"] + P["T_LIFT"] / 2

md = f"""# {D} T104 輸送帶追蹤抓取

## 目的
T102/T103 的工件都是靜止的。本項讓手臂**在輸送帶不停機的情況下抓取移動中的工件**（conveyor tracking / on-the-fly picking）並放入料盤：
估測工件位置與帶速 → TCP 與工件同步 → 追蹤中下降、夾持、抬升 → 放置。量化追蹤誤差、手內偏移、放置誤差、週期、速度/力矩使用率，
找出可追蹤的帶速上限（追蹤窗），並比較「只靠標稱帶速的開迴路預測」與「相機回授的閉迴路估測」在帶速有誤差時的差異；
最後做連續來料，觀察來料週期短於手臂週期時的行為（改在擋料板前靜態抓取）。

## 模型與參數
* 實作：`sixaxis/tracking.py`（`BeltEstimator`、`Tracker`、`track_pick`、`place_held`、`static_pick`、`run_track_cycle`、`run_flow`、
  判據 `x_end_pred`／`v_max_window`）。場景為 T006 工作站：輸送帶 x∈[0.15, 0.55]、y∈[0.35, 0.55] m，沿 +x 前進，x = 0.52 有擋料板；
  帶面為庫侖限幅的速度追隨模型（μ = 0.5）。工件：50 mm 方塊、0.20 kg；夾持力 40 N；手臂為計算力矩控制（T005）。
* 幾何：上料 x = {P['X_FEED']}、光電開關 x = {P['X_DET']}、等待點 TCP = ({P['X_WAIT']}, {P['BELT_Y']:.2f}, {P['Z_HOVER']:.3f}) m（指尖離工件頂面 15 mm）、
  抬升結束 TCP 的 x 上限 X_END_MAX = {P['X_END_MAX']} m（工件前緣距擋料板 35 mm）。
* 時間：同步 T_SYNC = {P['T_SYNC']} s、追蹤下降 T_DESC = {P['T_DESC']} s、夾持「兩指法向力 ≥ 0.5·F 持續 {P['GRIP_DWELL']} 步（0.1 s）」即完成（上限 {P['T_GRIP_MAX']} s，
  可行性判斷用標稱 {P['T_GRIP_NOM']} s）、抬升 T_LIFT = {P['T_LIFT']} s（同時 x 向由帶速減到 0）。
* 感測：光電開關給第一筆位置；相機每 {P['CAM_DT'] * 1e3:.0f} ms 量測工件 xy（讀模擬器位置 + 高斯雜訊 σ，預設 0.5 mm）。

### 依據
1. **α-β 濾波器**（等速目標的穩態卡爾曼濾波；Kalata 1984, IEEE Trans. AES 20(2)；Bar-Shalom, Li & Kirubarajan 2001, §6.5）：
   預測 x⁻ = x + v·T；殘差 r = z − x⁻；x = x⁻ + α·r；v ← v + (β/T)·r，α = {P['ALPHA']}、β = {P['BETA']}、T = {P['CAM_DT']} s。
   對等速目標穩態誤差為 0；位置估測變異數／量測變異數 = (2α² + 2β − 3αβ)/(α(4 − 2α − β)) = {M['ab_noise_ratio']:.3f}
   （標準差 ×{np.sqrt(M['ab_noise_ratio']):.2f}；單元測試以 4 萬筆蒙地卡羅驗證，誤差 < 8 %）。
2. **開迴路預測誤差**：帶速相對誤差 ε（實際 = 標稱·(1+ε)）時，TCP − 工件 = −ε·v_nom·(t − t_det)（等速積分）。
3. **同步**：p = p_w + s(τ)·(p̂ − p_w)，s 為五次時間律（s、ṡ、s̈ 端點為 0）→ 結束時 p = p̂、ṗ = ṡ(p̂ − p_w) + s·v̂ = v̂。
   觸發條件 x̂ ≥ x_w − v̂·T_SYNC/2（TCP 幾乎不需後退）。
4. **CLIK**（Siciliano 等 2009, §3.7）：q̇ = J⁻¹(q)·[ṗ_d + K(p_d − p(q)); K·log(R_d Rᵀ)]，K = {P['K_CLIK']:.0f} 1/s，每 1 ms 積分得關節指令，
   q̈ 前饋取 q̇ 一階差分，交給計算力矩控制。工具姿態固定朝下。
5. **追蹤窗**：自同步觸發到抬升結束，TCP 沿帶向行進到
   x_end(v) = max(x_w − v·T_SYNC/2, X_DET) + v·(T_SYNC + T_DESC + t_grip + T_LIFT/2)，須 ≤ X_END_MAX。
   低速時 x_end = x_w + v·{sumT:.2f}；v > {2 * (P['X_WAIT'] - P['X_DET']) / P['T_SYNC']:.2f} m/s 時觸發點落在光電上游，只能在光電處觸發。
   解得 **v_max = {M['v_max_window']:.3f} m/s**（`v_max_window`）。估測超窗 → 不追（WINDOW），可改用後備的擋料板前靜態抓取。
6. **手內偏移補償**：抬升後量測 d = 工件中心 − TCP，放置 TCP = 料盤格中心 − d。|dx| > {P['MAX_DX'] * 1e3:.0f} mm 視為失敗（INHAND_OFFSET：
   指墊 x 向 ±10 mm、料盤側銷內緣 17 mm，餘裕 7 mm）；橫向 |ŷ − 帶中心| > {P['Y_TOL'] * 1e3:.0f} mm 不追（LATERAL；指開口單邊 15 mm）。

成功判定：放入料盤格後 XY ≤ 3 mm、|Z| ≤ 1.5 mm、傾角 ≤ 2°、偏航 ≤ 5°（同 T102）。

## 可重現指令
```bash
PYTHONPATH=. python3 scripts/t104_report.py {D}
python3 scripts/t104_write_md.py {D} "<pytest 摘要>"
ruff check . && python3 -m pytest        # {PYT}
```
注意：報告腳本與 pytest 不可同時執行（兩者都會重寫 `robots/ur5_gripper.urdf`）。

## 指令序列
單一週期（`run_track_cycle`）：
1. `move_l`（必要時先垂直抬到 0.25 m）→ `move_j tcp=({P['X_WAIT']}, {P['BELT_Y']:.2f}, {P['Z_HOVER']:.3f})` 到等待點（T007 指令）
2. 上料（工件以帶速出現在 x = {P['X_FEED']}）→ 光電觸發 → 等待同步觸發
3. 同步 {P['T_SYNC']} s → 追蹤下降 {P['T_DESC']} s → 追蹤中夾持 → 抬升 {P['T_LIFT']} s（線上 CLIK，非 T007 指令）
4. 量測手內偏移 → `move_j`（料盤格上方）→ `move_l`（下降到格內高度）→ `release` → `move_l`（退回）（T007 指令）

## 量化結果
### 1. 標稱（帶速 {V0:.2f} m/s、σ = 0.5 mm）
| 項目 | 數值 |
|---|---|
| 結果 | {ok(N0['success'])}（{N0['method']}） |
| 光電觸發 → 開始同步 → 開始夾持 → 抬升結束 [s] | 0 → {N0['t_sync'] - N0['t_det']:.3f} → {N0['t_grip'] - N0['t_det']:.3f} → {N0['t_end'] - N0['t_det']:.3f} |
| 追蹤誤差（下降段，TCP − 工件，帶向）RMS／最大／夾持瞬間 [mm] | {N0['track_rms_mm']:.2f}／{N0['track_max_mm']:.2f}／{N0['track_end_mm']:+.2f} |
| 夾持瞬間橫向誤差 [mm] | {N0['track_y_end_mm']:+.2f} |
| 夾持瞬間帶速估測 [m/s] | {N0['v_est_grip']:.4f} |
| 夾持完成時間 [s]；兩指法向力 L／R [N] | {N0['grip_s']:.3f}；{N0['fL_N']:.1f}／{N0['fR_N']:.1f} |
| 手內偏移 dx／dy／dz [mm]、傾角 [deg] | {N0['dx_mm']:+.2f}／{N0['dy_mm']:+.3f}／{N0['dz_mm']:+.3f}、{N0['inhand_tilt_deg']:.2f} |
| 抬升結束 TCP x [m]（上限 {P['X_END_MAX']}） | {N0['x_end']:.3f} |
| 放置 XY／Z 誤差 [mm]、傾角／偏航 [deg] | {PL['xy_err_mm']:.2f}／{PL['z_err_mm']:+.3f}、{PL['tilt_deg']:.3f}／{PL['yaw_deg']:.2f} |
| 週期（上料 → 放好退回）[s] | {N0['cycle_s']:.2f} |
| 速度／力矩使用率（追蹤段；全程力矩） | {N0['vel_util']:.2f}／{N0['torque_util']:.2f} |

### 2. 帶速掃描（σ = 0.5 mm）
| 帶速 [m/s] | 結果 | 錯誤 | 追蹤 RMS [mm] | 追蹤最大 [mm] | 手內 dx [mm] | x_end [m] | 放置 XY [mm] | 週期 [s] | 速度使用率 | 力矩使用率 |
|---|---|---|---|---|---|---|---|---|---|---|
{speeds}

追蹤窗判據 v_max = {M['v_max_window']:.3f} m/s：{', '.join(format(s['belt_speed'], '.2f') for s in spd_ok)} m/s 成功，
{', '.join(format(s['belt_speed'], '.2f') for s in spd_no)} m/s 被判定超窗（WINDOW）而不追，與判據一致。
後備：帶速 0.30 m/s 加上 `fallback=True` → 等工件被擋料板擋停後靜態抓取：{ok(FB['success'])}（{FB['method']}），
放置 XY {FB['place']['xy_err_mm']:.2f} mm、週期 {FB['cycle_s']:.2f} s。

### 3. 帶速誤差：開迴路 vs 閉迴路（控制器假設 {V0:.2f} m/s，實際 = {V0:.2f}·(1+ε)）
「夾持瞬間誤差」= TCP − 工件（帶向）。開迴路解析值 = −ε·v_nom·(t_grip − t_det)，t_grip − t_det 平均 {dt_grip:.3f} s。

| ε [%] | 開迴路解析 [mm] | 開迴路模擬 [mm] | 開迴路手內 dx [mm] | 開迴路結果 | 閉迴路模擬 [mm] | 閉迴路手內 dx [mm] | 閉迴路 v̂ [m/s] | 閉迴路結果 |
|---|---|---|---|---|---|---|---|---|
{mis_rows}

開迴路模擬與解析式最大相差 {pred_dev:.2f} mm；開迴路成功 {sum(x['success'] for x in op)}/{len(op)}、閉迴路成功 {sum(x['success'] for x in cl)}/{len(cl)}。

### 4. 相機量測雜訊（帶速 {V0:.2f} m/s，每級 5 個種子）
| σ [mm] | 成功 | 追蹤 RMS 平均 [mm] | 追蹤最大 [mm] | 手內 \\|dx\\| 最大 [mm] | 放置 XY 最大 [mm] | 力矩使用率最大 | α-β 穩態理論 {np.sqrt(M['ab_noise_ratio']):.2f}σ [mm] |
|---|---|---|---|---|---|---|---|
{noise_rows}

合計 {n_ok}/{len(M['noise'])} 成功。

### 5. 工件橫向偏移（帶速 {V0:.2f} m/s）
| 上料 y 偏移 [mm] | 結果 | 錯誤 | 手內 dy [mm] | 追蹤 RMS [mm] | 放置 XY [mm] |
|---|---|---|---|---|---|
{lat_rows}

### 6. 隨機情境 N = {len(rt)}（帶速 U(0.05, 0.25) m/s、帶速誤差 ε U(±10 %)、橫向 U(±10 mm)、σ = 1 mm；種子 {M['seed']}；同一組情境兩種模式）
| 模式 | 成功 | 夾持瞬間 \\|誤差\\| 平均 [mm] | 最大 [mm] | 放置 XY 最大 [mm] | 平均週期 [s] | 失敗原因 |
|---|---|---|---|---|---|---|
| 閉迴路 track | {rstat(rt)} |
| 開迴路 open | {rstat(ro)} |

### 7. 連續來料（帶速 {V0:.2f} m/s，4 件依序放入料盤 4 格）
策略：目標 = 最下游未取工件；仍在追蹤窗內就追蹤夾取，否則等它被擋料板擋停後靜態抓取。

| 來料週期 [s] | 放好 | 追蹤夾取 | 靜態抓取 | 總時間 [s] | 產能 [件/分] | 放置 XY 最大 [mm] | 各件滯留時間（上料 → 放好）[s] |
|---|---|---|---|---|---|---|---|
{flows}

## 圖表與動畫
* 標稱追蹤時間歷程（位置、追蹤誤差、TCP 高度與指力、關節角/速度/力矩使用率；底色為 同步／下降／夾持／抬升）：
  ![timeseries](../media/{T}_timeseries.png)
* 帶速掃描、追蹤窗、帶速誤差（開/閉迴路）：![speed](../media/{T}_speed.png)
* 雜訊、隨機情境、連續來料：![stats](../media/{T}_stats.png)
* 動畫（2× 速，GIF 每 3 幀取 1；連續來料 4 件 → 0.25 m/s 追蹤 → 0.30 m/s 超窗改靜態抓取）：![anim](../media/{T}.gif)
  [GIF](../media/{T}.gif)（{M['gif_MB']:.2f} MB）／[MP4](../media/{T}.mp4)（{M['mp4_MB']:.2f} MB）
* 關鍵畫面：[A 追蹤夾取中](../media/{T}_key_a.png)、[B 4 件放滿料盤](../media/{T}_key_b.png)、[C 超窗後靜態抓取完成](../media/{T}_key_c.png)
* 原始數據：`docs/test-reports/data/{T}_metrics.json`

動畫三段的實際結果：

| 段落 | 結果 | 模擬時間 [s] |
|---|---|---|
{demo}

## 問題與限制
* 「相機」是讀模擬器位置加高斯雜訊，無延遲、無遮擋；夾爪下降後實際相機可能被遮住。光電開關位置視為精確。
* 夾爪模型（T005）為無速度上限的位置馬達，閉合只需約 {N0['grip_s']:.2f} s（含 0.1 s 力穩定判定）；實機氣動/電動夾爪更慢，追蹤窗會更小
  （判據已用標稱 {P['T_GRIP_NOM']} s，可直接代入實際值）。
* 帶面為「庫侖限幅速度追隨」的等效模型（非真實帶面接觸），工件在帶上不會自轉；工具姿態固定，不處理工件偏航，橫向只容許 ±{P['Y_TOL'] * 1e3:.0f} mm。
* 追蹤段以 CLIK 線上產生指令，未做事前的速度/力矩可行性規劃，只在執行中記錄使用率；力矩使用率由手臂 RNE 模型估算（不含夾爪與工件質量）。
  雜訊大時 q̈ 前饋（一階差分）會放大雜訊，力矩使用率隨 σ 上升（見第 4 節）。
* 追蹤窗受輸送帶長度（0.40 m）與擋料板限制，v_max 只有 {M['v_max_window']:.2f} m/s；等待點與各段時間未隨帶速最佳化。
* 連續來料只測 4 件（料盤容量）；來料週期短於手臂週期時工件在擋料板前排隊，未處理料盤滿載/換盤與帶上工件間距過小的情況。
* 放入料盤後的偏航約 {PL['yaw_deg']:.1f}°（靜態抓取也有，來自料盤側銷的導正，非追蹤造成）。

## 結論
帶速 {spd_ok[0]['belt_speed']:.2f}–{spd_ok[-1]['belt_speed']:.2f} m/s 皆能在不停機下追蹤夾取並放入料盤：追蹤誤差 RMS ≤ {max(s['track_rms_mm'] for s in spd_ok):.2f} mm、
手內偏移 |dx| ≤ {max(abs(s['dx_mm']) for s in spd_ok):.2f} mm、放置 XY ≤ {max(s['place']['xy_err_mm'] for s in spd_ok):.2f} mm、
週期 {min(s['cycle_s'] for s in spd_ok):.2f}–{max(s['cycle_s'] for s in spd_ok):.2f} s、力矩使用率 ≤ {max(s['torque_util'] for s in spd_ok):.2f}；
超過追蹤窗判據 v_max = {M['v_max_window']:.3f} m/s 的帶速被正確拒絕並可由靜態抓取後備完成。
帶速有 ±20 % 誤差時，開迴路夾持誤差與解析式 −ε·v·Δt 相符（最大差 {pred_dev:.2f} mm）且只在 |ε| 小時成功（{sum(x['success'] for x in op)}/{len(op)}），
α-β 閉迴路則 {sum(x['success'] for x in cl)}/{len(cl)} 成功；隨機情境 N = {len(rt)} 閉迴路 {sum(x['success'] for x in rt)}/{len(rt)}、開迴路 {sum(x['success'] for x in ro)}/{len(ro)}。
連續來料週期 ≥ {min(fl['period_s'] for fl in F_all) if F_all else float('nan'):.0f} s 時 4 件全部追蹤夾取；{('週期 ' + '、'.join(format(fl['period_s'], '.0f') for fl in F_mix) + ' s 時部分改為靜態抓取但仍全部放好。') if F_mix else ''}
下一步 T105 銲接路徑。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written")
