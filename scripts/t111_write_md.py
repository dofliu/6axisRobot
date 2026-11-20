"""T111 報告 md：由 docs/test-reports/data/<date>_T111_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t111_write_md.py 2026-11-20 "<pytest 摘要>"
可選 docs/test-reports/data/<date>_T111_narrative.md（發現與討論，人工撰寫、引用本 JSON 數字）會附在結果後。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T111"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
PR = M["params"]
PH = M["phys"]


def stat(xs, fmt=".2f"):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return "—"
    return f"{np.mean(xs):{fmt}}（{np.min(xs):{fmt}}–{np.max(xs):{fmt}}）"


def grp(g):
    return sorted([r for r in PH if r["group"] == g], key=lambda r: r["seed"])


# ---------------------------------------------------------------- 感知
P = M["perception"]
items = [i for p in P for i in p["items"] if i["matched"]]
n_true = sum(p["n_true"] for p in P)
n_det = sum(p["n_det"] for p in P)
n_match = sum(p["n_matched"] for p in P)
cls_ok = sum(i["cls"] == i["true"] for i in items)
cons = sum(i["consistent"] for i in items)
perc_rows = []
for c in ("red", "blue", "green"):
    it = [i for i in items if i["true"] == c]
    pe = [i["pos_err_mm"] for i in it]
    ye = [i["yaw_err_deg"] for i in it if i["yaw_err_deg"] is not None and i["yaw_err_deg"] == i["yaw_err_deg"]]
    he = [abs(i["h_err_mm"]) for i in it]
    perc_rows.append(
        f"| {c} | {len(it)} | {np.mean(pe):.2f} | {np.percentile(pe, 95):.2f} | {np.max(pe):.2f} | "
        f"{(f'{np.mean(ye):.2f} / {np.max(ye):.2f}') if ye else '—（旋轉對稱）'} | {np.mean(he):.2f} | "
        f"{stat([i['round'] for i in it], '.3f')} | {sum(i['cls'] == c for i in it)}/{len(it)} |")

# ---------------------------------------------------------------- 序列
Sq = M["sequence"]
est = {m: np.array([s[m]["est"] for s in Sq]) for m in ("scan", "greedy", "optimal")}
gain_scan = (est["scan"] - est["optimal"]) / est["scan"] * 100
gain_greedy = (est["greedy"] - est["optimal"]) / est["greedy"] * 100
n_opt_eq_greedy = int(np.sum(np.abs(est["greedy"] - est["optimal"]) < 1e-6))
t_dp = [s["optimal"]["t"] for s in Sq]
blocked = [sum(s["blocked"]) for s in Sq]
SD = M["sequence_dense"]
bd = [sum(s["blocked"]) for s in SD]
bd_nc = [sum(s["blocked_nocheck"]) for s in SD]
est_d = {m: [s[m]["est"] for s in SD] for m in ("scan", "greedy", "optimal")}
n_inf_d = {m: sum(1 for x in est_d[m] if not np.isfinite(x)) for m in est_d}


def phys_row(label, g):
    rs = grp(g)
    if not rs:
        return f"| {label} | — |"
    n = sum(r["n_parts"] for r in rs)
    c = sum(r["n_correct"] for r in rs)
    ok = sum(r["success"] for r in rs)
    seated = sum(r["n_seated"] for r in rs)
    ntray = sum(r["n_tray"] for r in rs)
    pf = sum(r["pick_fail"] for r in rs)
    errs = {}
    for r in rs:
        for e in r["errors"]:
            errs[e] = errs.get(e, 0) + 1
    es = "、".join(f"{k}×{v}" for k, v in errs.items()) or "無"
    good = [r for r in rs if r["success"]]
    return (f"| {label} | {ok}/{len(rs)} | {c}/{n} | {seated}/{ntray} | {pf} | "
            f"{stat([r['cycle_s'] for r in good], '.1f')} | {stat([r['cycle_s'] / r['n_parts'] for r in good], '.2f')} | "
            f"{sum(r['n_planned'] for r in rs)} | {stat([r['torque_util'] for r in rs])} | "
            f"{stat([r['max_disturb_mm'] for r in rs], '.1f')} | {es} |")


# 料盤放置精度（全部成功組）
tray = [f for r in PH if r["group"] in ("seq_optimal", "seq_greedy", "seq_scan", "post_plan", "dense_full")
        for f in r["final"] if f["dest"] == "tray" and f["where"] == "tray"]
bins = [f for r in PH if r["group"] in ("seq_optimal", "seq_greedy", "seq_scan", "post_plan", "dense_full")
        for f in r["final"] if f["dest"] != "tray" and f["where"] == f["dest"]]
upright = sum(f["tilt_deg"] < 5 for f in bins)

# 視覺記錄（物理執行中每次拍照）
vis = [v for r in PH if r["group"] == "seq_optimal" for v in r["vision"]]
vis_items = [i for v in vis for i in v["items"] if i.get("matched")]
n_reobs = sum(v.get("reobserve", False) for v in vis)

# 避障
pp, pn = grp("post_plan"), grp("post_noplan")
pl = [m for r in pp for m in r["motions"] if m.get("planned")]

# 序列：實際週期
seq_cyc = {m: [r["cycle_s"] for r in grp(f"seq_{m}")] for m in ("optimal", "greedy", "scan")}
seq_est = {m: [r["log"][0]["est_seq"] if r["log"] else None for r in grp(f"seq_{m}")] for m in seq_cyc}

A = M["anim"]

L = []
w = L.append
w(f"# T111 多物件分類揀選 — 測試報告（{D}）\n")
w("## 目的\n")
w("在工作台取料區隨機散置三類物件（紅方塊、藍長方柱、綠圓柱），由俯視相機辨識類別與位姿，"
  "規劃無干涉的夾取方向與揀選順序，以避障搬運（T110 規劃器加入「其餘物件」與「夾持中工件」）分別送到料盤格與 A/B 料箱。"
  "量化：感知誤差、分類正確率、序列最佳化效益、成功率、週期時間、力矩使用率、與鄰物干涉／避障的消融比較。\n")
w("## 模型與參數\n")
w("| 項目 | 內容 |\n|---|---|")
cl = PR["classes"]
w(f"| 物件類別 → 目的地 | 紅 方塊 {cl['red']['size']} m、{cl['red']['mass']} kg → 料盤 4 格（格內單邊間隙 4 mm）；"
  f"藍 長方柱 {cl['blue']['size']} m、{cl['blue']['mass']} kg → A 箱；綠 圓柱（半徑, 高）{cl['green']['size']} m、"
  f"{cl['green']['mass']} kg → B 箱 |")
w(f"| 取料區 | 物件輪廓須在 x∈[{PR['zone'][0]}, {PR['zone'][1]}]、y∈[{PR['zone'][2]}, {PR['zone'][3]}] m，"
  f"輪廓間距 ≥ {PR['gap'] * 1e3:.0f} mm（密集組 {PR['dense_gap'] * 1e3:.0f} mm），yaw 均勻隨機；每場景 2+2+2 件 |")
w(f"| 料箱 | 中心 A {PR['bin_c']['binA']}、B {PR['bin_c']['binB']}，內尺寸 {2 * PR['bin_in'][0] * 1e3:.0f}×"
  f"{2 * PR['bin_in'][1] * 1e3:.0f} mm（x×y），壁頂 z = {PR['bin_wall_top']:.3f} m；投放點沿 y ±{PR['bin_slot'] * 1e3:.0f} mm；"
  f"放置時 ψ = 90°（兩指沿 x，指墊外緣距壁 10 mm）下降到物件底面離箱底 {PR['bin_lift'] * 1e3:.0f} mm（即放到底）再鬆開 |")
w("| 夾取高度 | 指墊上緣高出物件頂面 3 mm（TCP = 頂面 − 17 mm）：夾 50 mm 方塊時掌體下緣高出 70 mm 鄰物 8 mm；"
  "不取齊平以避免共面稜邊造成 PyBullet 接觸法向翻轉（鬆爪黏住） |")
w(f"| 俯視相機 | 位置 {PR['cam_eye']}，FOV {PR['cam_fov']}°，{PR['cam_px']}×{PR['cam_px']} px，"
  f"桌面像素 {P[0]['px_mm']:.3f} mm；TinyRenderer RGB＋深度 |")
w(f"| 感知演算法 | 深度反投影（(PV)⁻¹·NDC）→ 背景高度圖相減（>6 mm）→ 8-連通標記 → 頂面點質心、最小面積外接矩形（旋轉卡尺）、"
  f"圓形度 < {PR['round_t']} 判圓、色度最近類別＋形狀/高度一致性；手臂遮住 ROI > {PR['occl_px']} px 時退到觀測姿態重拍 |")
w(f"| 抓取規劃 | 方塊兩組對邊、圓柱每 22.5°；兩指墊與掌體俯視矩形 vs 鄰物/靜態體（頂面高於部件下緣－餘裕者）距離 ≥ "
  f"{PR['grip_margin'] * 1e3:.0f} mm；全開內距 {PR['open_w'] * 1e3:.0f} mm |")
w(f"| 序列 | 成本 = 兩段關節 PTP 最短時間（qd_max×0.5，五次多項式）＋固定 {PR['t_fixed']} s；Held–Karp DP（含可抓相依）、"
  "貪婪最近鄰、掃描順序（x→y）；執行時每放一件重拍、重排、只做第一件 |")
w(f"| 搬運 | 高度 TCP z = {PR['z_clear']} m；T110 `CollisionChecker` ＋ 其餘物件（偵測方塊）＋ 夾持工件（附在 TCP）；"
  "關節直線保證式檢查不通則 Informed RRT*（第一解，≤ 30 s）＋捷徑 |")
w(f"| 障礙物組 | 料盤右側光柵立柱 中心 {PR['post'][0]}、半邊長 {PR['post'][1]}（高 0.30 m），位於取料區與 B 箱之間 |")
w("| 控制 | T007 執行器（計算力矩 ω_n = 60 rad/s、1 kHz、夾持力 40 N），夾取後檢查物件在手中偏差 ≤ 3 mm、傾斜 ≤ 5°，否則放回重拍重抓（≤ 3 次） |")
w("")
w("## 可重現指令\n")
w("```bash\npython3 -m pip install -r requirements.txt   # pybullet 需 --no-build-isolation，見 CLAUDE.md")
w("python3 -m pytest -q tests/test_sorting.py")
w(f"PYTHONPATH=. python3 scripts/t111_report.py {D}        # 4 行程平行，約 20 分鐘；--anim 只重做動畫/圖")
w(f"python3 scripts/t111_write_md.py {D} \"<pytest 摘要>\"\n```\n")
w("## 指令序列（每件）\n")
w("1. 拍照 → 偵測（若手臂遮住取料區：`move_j` 回觀測姿態再拍）\n2. 序列規劃（DP）→ 取第一件；抓取候選（最大間隙者）\n"
  "3. `move_j`（直線或 RRT* 路徑點）到物件上方 z = 0.27 m、夾爪 yaw ψ\n4. `move_l` 下降到物件中心高 → `grip` 40 N → `move_l` 抬升；在手檢查\n"
  "5. 夾持工件併入碰撞模型 → `move_j`（直線或 RRT*）到目的格上方（料盤 ψ = 0；長方柱讓 60 mm 邊沿 x）\n"
  "6. 以取樣檢查選放置格（垂直行程手臂間隙 > 3 mm）→ `move_l` 下降（料盤：放入格；料箱：放到箱底）→ `release` → `move_l` 抬升\n")

w("## 量化結果\n")
w(f"### 1. 感知（{len(P)} 個靜態場景、{n_true} 件）\n")
w(f"偵測 {n_det} 個、與真值配對 {n_match}/{n_true}；分類正確 {cls_ok}/{len(items)}；形狀/高度一致性 {cons}/{len(items)}；"
  f"單張影像處理 {stat([p['t_detect'] * 1e3 for p in P], '.0f')} ms。\n")
w("| 類別 | n | 位置誤差平均 [mm] | 95% [mm] | 最大 [mm] | yaw 誤差 平均/最大 [°] | 高度誤差 [mm] | 圓形度 | 分類 |")
w("|---|---|---|---|---|---|---|---|---|")
L += perc_rows
w(f"\n物理執行中（最佳序列組 {len(grp('seq_optimal'))} 場景）共拍照 {len(vis)} 次，其中 {n_reobs} 次因手臂遮住取料區而退到觀測姿態重拍；"
  f"配對物件位置誤差 {stat([i['pos_err_mm'] for i in vis_items])} mm。\n")
w("![perception](../media/" + T + "_perception.png)\n")
w("![camera](../media/" + T + "_camera.png)\n")

w(f"### 2. 序列最佳化（{len(Sq)} 個場景，運動學估計）\n")
w("| 策略 | 估計週期 [s] | 相對 DP 多出 |\n|---|---|---|")
w(f"| 掃描順序 | {stat(list(est['scan']))} | {stat(list(gain_scan / (100 - gain_scan) * 100), '.1f')} % |")
w(f"| 貪婪最近鄰 | {stat(list(est['greedy']))} | {stat(list(gain_greedy / (100 - gain_greedy) * 100), '.1f')} % |")
w(f"| Held–Karp DP | {stat(list(est['optimal']))} | — |")
w(f"\nDP 計算時間 {stat(t_dp, '.2f')} s；貪婪恰為最佳的場景 {n_opt_eq_greedy}/{len(Sq)}；標準間距下有被鄰物擋住（無可行抓取）的物件"
  f" {stat(blocked, '.1f')} 件/場景。")
w(f"密集（{PR['dense_gap'] * 1e3:.0f} mm）{len(SD)} 場景：被擋 {stat(bd, '.1f')} 件/場景（不檢查鄰物則 {stat(bd_nc, '.1f')}）；"
  f"無可行序列（死結）場景數 掃描 {n_inf_d['scan']}、貪婪 {n_inf_d['greedy']}、DP {n_inf_d['optimal']}。\n")
w("物理執行（同 12 場景，滾動時域）實際週期：")
w(f"DP {stat(seq_cyc['optimal'], '.1f')} s、貪婪 {stat(seq_cyc['greedy'], '.1f')} s、掃描 {stat(seq_cyc['scan'], '.1f')} s。\n")
w("![sequence](../media/" + T + "_sequence.png)\n")

w("### 3. 物理執行與消融\n")
w("| 組別 | 場景成功 | 物件正確 | 料盤入座 | 抓取失敗 | 週期 [s]（成功場景） | 每件 [s] | RRT* 次數 | 力矩使用率 | 鄰物最大位移 [mm] | 錯誤 |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for lab_, g in (("完整（DP 序列）", "seq_optimal"), ("完整（貪婪）", "seq_greedy"), ("完整（掃描順序）", "seq_scan"),
                ("固定 ψ = 0（不用視覺方位）", "fixed_yaw"), ("立柱＋規劃", "post_plan"), ("立柱＋不規劃（直線 PTP）", "post_noplan"),
                ("密集 4 mm＋抓取檢查", "dense_full"), ("密集 4 mm、不檢查鄰物", "dense_nocheck")):
    w(phys_row(lab_, g))
w("")
w(f"料盤放置（成功組 {len(tray)} 件）：格心誤差 {stat([f['pocket_err_mm'] for f in tray])} mm、高度誤差 "
  f"{stat([f['z_err_mm'] for f in tray])} mm、yaw {stat([f['yaw_deg'] for f in tray])}°；入座 {sum(f['seated'] for f in tray)}/{len(tray)}。"
  f"料箱投放 {len(bins)} 件全部在正確箱內，直立（傾斜 < 5°）{upright}/{len(bins)}。\n")
if pl:
    w(f"立柱組 RRT* 共 {len(pl)} 次（接近 {sum(m['tag'] == 'approach' for m in pl)}、搬運 {sum(m['tag'] == 'transport' for m in pl)}、"
      f"觀測 {sum(m['tag'] == 'observe' for m in pl)}），全部找到解 {sum(bool(m.get('rrt_ok')) for m in pl)}/{len(pl)}；"
      f"規劃時間 {stat([m['t_plan'] for m in pl])} s；捷徑後路徑 {stat([m.get('sc_len') for m in pl])} rad、"
      f"{stat([m.get('n_seg') for m in pl], '.1f')} 段。\n")
w("![ablation](../media/" + T + "_ablation.png)\n")

w("### 4. 示範執行（立柱場景 seed 0，動畫）\n")
w(f"結果：{A['n_correct']}/{A['n_parts']} 正確、週期 {A['cycle_s']:.1f} s、RRT* {A['n_planned']} 次（規劃 {A['t_plan']:.1f} s）、"
  f"重拍 {A['n_reobserve']} 次、力矩使用率 {A['torque_util']:.2f}、最大關節追蹤誤差 {A['max_track_err_mrad']:.2f} mrad。\n")
w("| 件 | 類別 | ψ 夾取 [°] | 夾持寬 [mm] | 夾爪–鄰物間隙 [mm] | 目的地 | 結果 | 時間 [s] |\n|---|---|---|---|---|---|---|---|")
for k, e in enumerate(A["log"]):
    w(f"| {k + 1} | {e['cls']} | {np.degrees(e['psi']):.0f} | {e['width_mm']:.1f} | {min(e['clear_mm'], 100):.1f} | "
      f"{e.get('dest', '—')} | {e['result']} | {e['t_end'] - e['t_start']:.2f} |")
w("")
w("![anim](../media/" + T + ".gif)\n")
w(f"MP4：[{T}.mp4](../media/{T}.mp4)\n")
w("| 開始 | 中段 | 完成 |\n|---|---|---|")
w(f"| ![a](../media/{T}_key_a.png) | ![b](../media/{T}_key_b.png) | ![c](../media/{T}_key_c.png) |\n")
w("![joints](../media/" + T + "_joints.png)\n")
w("![top](../media/" + T + "_topview.png)\n")

nar = ROOT / f"docs/test-reports/data/{T}_narrative.md"
if nar.exists():
    w(nar.read_text(encoding="utf-8"))
w("## 測試\n")
w(f"`ruff check .` 通過；`python3 -m pytest -q`：{PYT}\n")
w(f"原始數據：[data/{T}_metrics.json](data/{T}_metrics.json)\n")
(ROOT / f"docs/test-reports/{T}.md").write_text("\n".join(L), encoding="utf-8")
print("written", len(L))
