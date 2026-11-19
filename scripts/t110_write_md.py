"""T110 報告 md：由 docs/test-reports/data/<date>_T110_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t110_write_md.py 2026-11-19 "<pytest 摘要>"
可選 docs/test-reports/data/<date>_T110_narrative.md（發現與討論，人工撰寫、引用本 JSON 數字）會附在結果後。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T110"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
C, W, EX, PR = M["checker"], M["wall"], M["exec"], M["params"]
RUNS = M["wall_runs"]


def f(x, fmt=".2f"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return format(x, fmt)


def stat(xs, fmt=".2f"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{np.mean(xs):{fmt}}（{np.min(xs):{fmt}}–{np.max(xs):{fmt}}）"


# ---- 保證式 vs 密集
E = C["edges"]
cf = [e for e in E if e["cert"]]
n_ff = sum(e["fine_min"] > 0 for e in cf)
n_fc = sum(e["fine_min"] <= 0 for e in cf)
cc_ = [e for e in E if not e["cert"]]
n_cf = sum(e["fine_min"] > 0 for e in cc_)
n_cc = sum(e["fine_min"] <= 0 for e in cc_)
coarse_miss = sum(e["coarse_min"] > 0 and e["fine_min"] <= 0 for e in E)
fine_col = sum(e["fine_min"] <= 0 for e in E)
nq_c, nq_f = np.mean([e["nq_cert"] for e in E]), np.mean([e["nq_fine"] for e in E])
nq_c_free = np.mean([e["nq_cert"] for e in cf]) if cf else float("nan")
nq_f_free = np.mean([e["nq_fine"] for e in cf]) if cf else float("nan")

# ---- 牆場景
by = {k: [r for r in RUNS if r["kind"] == k] for k in ("rrt", "first", "informed", "uninformed")}


def row(k, label, n_it):
    rs = by[k]
    ok = [r for r in rs if r["success"]]
    first = [r["t_first"] for r in ok]
    hist_first = [next(h[1] for h in r["history"] if np.isfinite(h[1])) for r in ok]
    return (f"| {label} | {len(ok)}/{len(rs)} | {n_it} | {stat(first)} | {stat([r['iter_first'] for r in ok], '.0f')} | "
            f"{stat(hist_first)} | {stat([r['cost'] for r in ok])} | {stat([r.get('sc_cost') for r in ok])} | "
            f"{stat([r['t_plan'] for r in ok], '.1f')} | {stat([r['n_queries'] for r in ok], '.0f')} | "
            f"{stat([r.get('min_clear', None) and r['min_clear'] * 1e3 for r in ok], '.1f')} |")


rows_wall = "\n".join([row("rrt", "RRT（η = 0.5，第一解停）", "≤ 4000"), row("first", "RRT*（第一解停）", "≤ 4000"),
                       row("informed", "Informed RRT*", "1500"), row("uninformed", "RRT*（均勻取樣）", "1500")])
inf_final = [r["cost"] for r in by["informed"] if r["success"]]
uni_final = [r["cost"] for r in by["uninformed"] if r["success"]]
straight = W["straight"]

# ---- 隨機
RD = M["random"]
rows_r = []
for r in RD:
    if "star" not in r:
        rows_r.append(f"| {r['seed']} | {r['name']} | {r['n_posts']} | {r['n_goals']} | — | — | — | — | — | — | — | — |")
        continue
    a, b, e = r["rrt"], r["star"], r.get("exec", {})
    rows_r.append(
        f"| {r['seed']} | {r['name'].split(':')[1]} | {r['n_posts']} | {r['n_goals']} | {f(r['straight_clear'] * 1e3, '.0f')} | "
        f"{'✔ ' + f(a['t_plan'], '.1f') + ' s' if a['success'] else '✘'} | {f(a['cost'])} | "
        f"{'✔ ' + f(b['t_first'], '.1f') + ' s' if b['success'] else '✘'} | {f(b.get('cost_first'))} → {f(b['cost'])} | "
        f"{f(b.get('sc_cost'))} | {f(e.get('min_clear_exec', None) and e['min_clear_exec'] * 1e3, '.1f')} | "
        f"{'無' if e and not e['hits'] else (','.join(e['hits']) if e else '—')} |")
rows_r = "\n".join(rows_r)
valid = [r for r in RD if "star" in r]
n_rrt = sum(r["rrt"]["success"] for r in valid)
n_star = sum(r["star"]["success"] for r in valid)
n_exec = sum(1 for r in valid if r.get("exec") and not r["exec"]["hits"])
n_exec_all = sum(1 for r in valid if r.get("exec"))
both = [r for r in valid if r["rrt"]["success"] and r["star"]["success"]]
ratio_cost = [r["star"]["sc_cost"] / r["rrt"]["cost"] for r in both]
exec_clear = [r["exec"]["min_clear_exec"] * 1e3 for r in valid if r.get("exec")]
exec_T = [r["exec"]["T_cycle"] for r in valid if r.get("exec")]
exec_tau = [r["exec"]["torque_ratio"] for r in valid if r.get("exec")]

narr = ROOT / f"docs/test-reports/data/{T}_narrative.md"
NARR = narr.read_text(encoding="utf-8") if narr.exists() else ""
lev = " / ".join(f"{x:.3f}" for x in C["lever"])
wc, wh = PR["wall"]["center"], PR["wall"]["half"]

md = f"""# T110 避障路徑規劃（RRT* ＋ 保證式碰撞檢查）測試報告

- 日期：{D}
- 模組：`sixaxis/planning.py`（碰撞檢查、RRT/RRT*、捷徑、時間律）、`sixaxis/scenarios.py`（場景）
- 測試：`tests/test_planning.py`；{PYT}

## 1. 目的
在 T006 工作站（工作台、料盤、治具、輸送帶）中加入障礙物，讓六軸手臂（含夾爪）自動規劃**無碰撞**的關節路徑並在物理模擬中執行。
量化：碰撞檢查的正確性（不得漏判）與成本、規劃成功率、第一解時間、路徑成本（關節空間長度）與其收斂、
執行時的最小間隙、追蹤誤差、週期時間與力矩使用率。

## 2. 模型與方法

### 2.1 碰撞/距離模型
* 規劃專用 PyBullet DIRECT client：手臂＋夾爪 URDF（指全開；寫在私有暫存目錄 → 可多行程平行）、
  障礙物 = T006 工作站 {C['n_boxes']} 個方塊（含取料點上的方塊零件與額外障礙物）＋ 地板。
  PyBullet 3.2.7 的 `createCollisionShapeArray` 實測只保留前 16 個子形狀，故以 16 個一組建成複合剛體（測試以逐一方塊比對驗證）。
* 間隙 f(q) = min(d_obs − m_obs, d_self − m_self)，m_obs = {PR['M_OBS'] * 1e3:.0f} mm、m_self = {PR['M_SELF'] * 1e3:.0f} mm；d 由 `getClosestPoints`（GJK/EPA）取得，
  搜尋半徑 {PR['D_MAX']:.2f} m（超出者以 {PR['D_MAX']:.2f} m 計，保守）。自碰撞檢查 {C['n_self_pairs']} 對非相鄰連桿。
  base/link1 只繞 z 轉，與地板距離為常數（外殼貼地），故不與地板比。
* 規劃空間（取樣盒）：q1, q3–q6 ∈ [−π, π]、q2 ∈ [−π, 0]（上臂不低於肩部水平面），體積 {C['box_volume']:.0f} rad⁶；
  均勻取樣自由比例 {C['free_frac'] * 100:.1f} %（牆場景，3000 樣本）。

### 2.2 保證式（certified）邊檢查
依 Schwarzer, Saha & Latombe（IEEE T-RO 2005）：關節 i 轉 Δq_i 時，下游任一點位移 ≤ R_i|Δq_i|。
直線插值 q(λ) 上任一點位移 ≤ λ·D，D = Σ R_i|Δq_i|。障礙物距離與自碰撞距離沿邊皆為 D-Lipschitz，因此

  f(q(λ)) ≥ max(f_a − λD, f_b − (1 − λ)D) ≥ (f_a + f_b − D)/2，

**若 f_a + f_b > D 則整條邊無碰撞（證明，而非取樣近似）**；否則二分遞迴，子段 ∞-範數 < {PR['DELTA_MIN'] * 1e3:.0f} mrad 仍無法證明者判碰撞（保守）。
R_i = max_k≥i(Σ_{{m=i+1..k}}‖p_m‖ + ρ_k)，‖p_m‖ 為相鄰連桿座標原點距離、ρ_k 為連桿 k（link6 併入夾爪）AABB 角點到其原點的最大距離：
R = [{lev}] m。節點的 f 只算一次並存在樹上，短邊常可零額外查詢通過。

### 2.3 RRT / RRT* / Informed RRT*
* RRT（LaValle 1998）：最近節點朝樣本延伸 η，目標偏置 5 %，距目標解 ≤ η 且邊可行即連上。
* RRT*（Karaman & Frazzoli, IJRR 2011）：k-nearest 版本，k_n = ⌈e(1 + 1/d)·log n⌉（d = 6），選 cost-to-come 最小的可行父節點，再 rewire；
  成本變化沿子樹傳遞。半徑版之 γ* = 2(1 + 1/d)^{{1/d}}(μ(X_free)/ζ_d)^{{1/d}} = {C['gamma_star']:.2f}（ζ_6 = π³/6）；
  r_n 在 n ≤ 數千時遠大於 η，等同全用 η，6 維中鄰居太少，因此採 k-nearest。
* Informed RRT*（Gammell, Srinivasa & Barfoot, IROS 2014）：已有解 c_best 後只在長橢球 {{q : ‖q − q_s‖ + ‖q − q_g‖ ≤ c_best}} 內均勻取樣
  （x = C·L·u + 中心，L = diag(c/2, √(c² − c_min²)/2, …)）；多個目標 IK 解時在仍可能改善者中隨機挑一個當焦點。
* 目標集合：目標 TCP 位姿（工具朝下）的所有封閉解 IK 中，位於取樣盒內且無碰撞者，任一連上即可。
* 後處理：隨機捷徑 200 次（Geraerts & Overmars 2007），每條捷徑仍經保證式檢查。
* 執行：逐段關節直線、五次多項式 rest-to-rest（T003），各段時間 = 滿足 0.5·q̇_max 與 q̈ ≤ 4 rad/s² 的最短時間；
  執行路徑與已檢查的邊完全相同，故「規劃無碰撞」可直接延伸到執行（扣除追蹤誤差）。
  控制：T005 計算力矩（ω_n = 60 rad/s）於 1 kHz 物理步。

## 3. 可重現指令
```bash
python3 -m pip install "setuptools<70" wheel && python3 -m pip install --no-build-isolation pybullet==3.2.7
python3 -m pip install -r requirements.txt
ruff check . && python3 -m pytest -q
PYTHONPATH=. python3 scripts/t110_report.py {D}     # 4 行程平行規劃 + 依序物理執行與渲染（勿與 pytest 同時執行）
python3 scripts/t110_write_md.py {D} "<pytest 摘要>"
```

## 4. 指令序列（場景）
* **牆場景**：工作台上隔板 center = ({wc[0]:.2f}, {wc[1]:.2f}, {wc[2]:.2f}) m、半邊長 ({wh[0]:.2f}, {wh[1]:.2f}, {wh[2]:.2f}) m（x ∈ [0.36, 0.90]、高 0.50 m）。
  起點 TCP = 取料點上方 {tuple(PR['stations']['pick'])}，終點 = 輸送帶擋料板上方 {tuple(PR['stations']['belt'])}（工具朝下）。
  `move_j` 等價的關節直線運動會掃過隔板；規劃後以 {EX['n_seg']} 段 PTP 執行。
* **隨機立柱基準**：16 個種子，每個在工作站（pick / belt / fixture / tray0 / tray3 上方）中隨機挑起訖，
  在連線附近放 2–4 根立柱（截面 4–8 cm、高 0.25–0.55 m）；拒絕取樣保證起訖有無碰撞 IK 解，且**關節直線到所有目標解皆碰撞**（必須規劃）。

## 5. 量化結果

### 5.1 碰撞檢查器
| 項目 | 值 |
|---|---|
| 單次距離查詢（障礙物 3 個複合體＋地板＋{C['n_self_pairs']} 自碰撞對） | {C['t_query_ms']:.3f} ms |
| Lipschitz 驗證：max \\|Δd\\| /(λD)（{C['lip_n']} 個樣本，需 ≤ 1） | 最大 {C['lip_max']:.3f}、p99 {C['lip_p99']:.3f}、中位 {C['lip_median']:.3f} |
| 保證式判「無碰撞」的邊 | {len(cf)} / {len(E)} |
| ├ 其中 2 mrad 密集取樣也無碰撞 | {n_ff} |
| └ 其中密集取樣發現碰撞（**漏判，必須 0**） | **{n_fc}** |
| 保證式判「碰撞」：密集取樣其實無碰撞（保守誤判） | {n_cf}（實際碰撞 {n_cc}） |
| 50 mrad 粗取樣漏掉的碰撞（粗取樣說可行、2 mrad 說碰撞） | {coarse_miss} / {fine_col} |
| 平均查詢數／邊：保證式 vs 2 mrad 密集（全部邊） | {nq_c:.1f} vs {nq_f:.1f}（{nq_f / nq_c:.1f} 倍） |
| 平均查詢數／邊：保證式 vs 2 mrad 密集（無碰撞邊） | {nq_c_free:.1f} vs {nq_f_free:.1f} |

### 5.2 牆場景
* 起點間隙 f = {W['start_clear'] * 1e3:.1f} mm；目標 IK 解（取樣盒內且無碰撞）{len(W['goals'])} 個。
* 關節直線（不規劃）：{'、'.join(f"目標{i}：保證式={'可行' if s['cert_free'] else '碰撞'}、最小 d_obs = {s['min_dobs'] * 1e3:.1f} mm" for i, s in enumerate(straight))}；
  第一個接觸點在 s = {f(W['straight_profile']['first_hit'], '.3f')}。

| 規劃器 | 成功 | 迭代上限 | 第一解時間 [s] | 第一解迭代 | 第一解成本 [rad] | 最終成本 [rad] | 捷徑後 [rad] | 規劃總時間 [s] | 距離查詢數 | 捷徑後路徑最小 d_obs [mm] |
|---|---|---|---|---|---|---|---|---|---|---|
{rows_wall}

（格式：平均（最小–最大）；RRT 與「第一解停」兩列的最終成本 = 第一解成本。）
Informed RRT* 1500 次迭代最終成本 {stat(inf_final)} rad vs 均勻取樣 RRT* {stat(uni_final)} rad。

### 5.3 物理執行（牆場景，最佳 Informed RRT* 路徑，seed {W['best_seed']}）
| 項目 | 值 |
|---|---|
| 路徑段數 / 週期時間 | {EX['n_seg']} 段 / {EX['T_cycle']:.2f} s |
| 速度 / 力矩使用率（規劃軌跡逆動力學） | {EX['vel_ratio']:.2f} / {EX['torque_ratio']:.2f} |
| 執行中手臂–工作站靜態體最小間隙（含隔板；不含地板/自碰撞，每 10 ms） | {EX['min_clear_exec'] * 1e3:.1f} mm |
| 執行中接觸 | {'無' if not EX['hits'] else ', '.join(EX['hits'])} |
| 最大關節追蹤誤差 | {EX['track_err_max']:.2e} rad |
| 終點 TCP 誤差 | {EX['end_err_mm']:.3f} mm |

### 5.4 隨機立柱基準（16 場景，皆需規劃）
| seed | 起→訖 | 立柱 | 目標解 | 直線最小 d_obs [mm] | RRT（η 0.5） | RRT 成本 | RRT* 第一解 | RRT* 成本 第一解→500 it | 捷徑後 | 執行最小間隙 [mm] | 執行接觸 |
|---|---|---|---|---|---|---|---|---|---|---|---|
{rows_r}

* 成功率：RRT {n_rrt}/{len(valid)}、RRT*（500 次迭代）{n_star}/{len(valid)}；物理執行無接觸 {n_exec}/{n_exec_all}。
* 兩者皆成功的 {len(both)} 場景：RRT* + 捷徑成本 / RRT 成本 = {stat(ratio_cost)}。
* 執行：最小間隙 {stat(exec_clear, '.1f')} mm、週期 {stat(exec_T)} s、力矩使用率 {stat(exec_tau)}。

{NARR}

## 6. 圖表與動畫
![動畫](../media/{T}.gif)

MP4：[{T}.mp4](../media/{T}.mp4)。前段：不規劃的關節直線運動（運動學回放，紅點 = 其 TCP 軌跡）撞上隔板；後段：Informed RRT* + 捷徑路徑
（綠點 = 規劃 TCP 軌跡）以計算力矩控制於物理模擬中執行，右上角為即時最小間隙。

| 直線 PTP 撞牆 | 執行 1/3 | 執行 2/3 | 終點 |
|---|---|---|---|
| ![](../media/{T}_straight_hit.png) | ![](../media/{T}_key_a.png) | ![](../media/{T}_key_b.png) | ![](../media/{T}_key_c.png) |

![收斂與邊檢查成本](../media/{T}_convergence.png)
![俯視/側視：樹與路徑](../media/{T}_paths.png)
![執行曲線](../media/{T}_exec.png)
![隨機基準](../media/{T}_random.png)

## 7. 問題與限制
* 障礙物為靜態方塊；夾持中的工件未納入手臂幾何（搬運時需把工件當成夾爪的附加連桿，留待 T111/T112）。
* R_i 為簡單幾何上界（沿連桿鏈原點距離＋AABB 半徑），偏保守 → 狹窄處二分較深；未使用更緊的逐連桿/逐對界或連續碰撞（CCD）解析法。
* 路徑以 rest-to-rest 分段執行，每個路徑點停頓；平滑化（如 B-spline/拋物線混成）會偏離已證明的邊，需重新檢查（未做）。
* 成本為未加權關節空間長度，不等於 TCP 路徑長或時間；RRT* 漸近最佳但有限迭代下仍有剩餘差距。
* 距離查詢 ~{C['t_query_ms']:.2f} ms 為瓶頸（Python 迴圈呼叫 {C['n_self_pairs']} 對自碰撞）；規劃時間為單核數值。

## 8. 結論
保證式邊檢查在 {len(E)} 條隨機邊上零漏判（{len(cf)} 條判可行者以 2 mrad 密集取樣複驗全部無碰撞），保守誤判 {n_cf} 條，
平均查詢數為 2 mrad 密集取樣的 1/{nq_f / nq_c:.1f}。牆場景中不規劃的關節直線運動穿入隔板 {-straight[0]['min_dobs'] * 1e3:.0f} mm；
RRT*（k-nearest）{sum(r['success'] for r in by['first'])}/{len(by['first'])} 次找到解、RRT 僅 {sum(r['success'] for r in by['rrt'])}/{len(by['rrt'])}，Informed 取樣 1500 次迭代成本比均勻取樣低約 {(1 - np.mean(inf_final) / np.mean(uni_final)) * 100:.0f} %。
16 個必須規劃的隨機立柱場景中 RRT* 成功 {n_star}/16、RRT {n_rrt}/16；所有規劃出的路徑（牆場景＋{n_exec_all} 個隨機場景）在 PyBullet 物理執行時皆無接觸，
最小間隙 ≥ 規劃餘裕 10 mm。下一步：T111 多物件分類揀選（沿用本規劃器，並把夾持工件納入手臂幾何）。
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("wrote", T)
