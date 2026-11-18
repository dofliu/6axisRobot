"""T109 報告 md：由 docs/test-reports/data/<date>_T109_metrics.json 產生（數字全來自實際執行）。
用法：python3 scripts/t109_write_md.py 2026-11-18 "<pytest 摘要>"
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sixaxis.insertion import TOOL_L  # noqa: E402

D = sys.argv[1]
PYT = sys.argv[2] if len(sys.argv) > 2 else ""
T = f"{D}_T109"
M = json.loads((ROOT / f"docs/test-reports/data/{T}_metrics.json").read_text(encoding="utf-8"))
PEG, HOLE, RCC, RIG, CS, IP, TH = M["peg"], M["hole"], M["rcc"], M["rigid"], M["contact"], M["insert"], M["theory"]
RUNS = M["runs"]
TMAX = math.degrees(2 * math.sqrt(IP["F_d"] * (HOLE["R"] - PEG["r"]) / (CS["mu"] * RCC["K_rot"])))


def f(x, fmt=".2f"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return format(x, fmt)


def res(r):
    return "OK" if r.get("ok") else f"NG：{r.get('error')}"


def pick(exp, name, **kv):
    for r in RUNS:
        if r["exp"] == exp and r["name"] == name and all(abs(r.get(k, 0) - v) < 1e-9 for k, v in kv.items()):
            return r
    return None


# ---- 倒角台
CH = M["chamfer"]
for c in CH:  # 倒角段（深度 < w_c）內的最大軸向力（排除倒角→孔壁轉角的瞬間尖峰）
    d_, fa_ = c["series"]["depth"], c["series"]["F_ax"]
    c["F_ax_ch"] = max([a for d0, a in zip(d_, fa_) if 0 < d0 < HOLE["w_c"]] or [float("nan")])
rowsC = "\n".join(
    f"| {c['e_mm']:.1f} | {c['F_lat_peak']:.2f} | {c['pred']['F_lat']:.2f} | "
    f"{(c['F_lat_peak'] / c['pred']['F_lat'] - 1) * 100 if c['pred']['F_lat'] > 0 else float('nan'):+.1f} | "
    f"{f(c['F_ax_ch'])} | {c['pred']['F_ax']:.2f} | {c['F_ax_peak']:.2f} | {f(c['ratio_med'], '.3f')} |" for c in CH)
CM = M["chamfer_mu"]
rowsMu = "\n".join(
    f"| {c['mu']:.1f} | {f(c['ratio_med'], '.3f')} | {c['ratio_pred']:.3f} | "
    f"{(c['ratio_med'] / c['ratio_pred'] - 1) * 100:+.1f} | {c['F_lat_peak']:.2f} |" for c in CM)
okC = [c for c in CH if c["pred"]["F_lat"] > 0.5]
devC = max(abs(c["F_lat_peak"] / c["pred"]["F_lat"] - 1) for c in okC) * 100
devMu = max(abs(c["ratio_med"] / c["ratio_pred"] - 1) for c in CM) * 100

# ---- 側向誤差掃描
E_LIST = sorted({round(r["e"] * 1e3, 3) for r in RUNS if r["exp"] == "lateral"})
SN = ["rigid_pos", "rcc_pos", "rcc_adm", "rcc_adm_search"]
SLAB = {"rigid_pos": "剛性腕＋位置", "rcc_pos": "RCC＋位置", "rcc_adm": "RCC＋導納", "rcc_adm_search": "RCC＋導納＋搜孔"}


def cell(r):
    if r is None:
        return "—"
    s = res(r)
    s += f"／{r['F_max']:.1f} N" if "F_max" in r else ""
    if r.get("ok"):
        s += f"／{r['cycle_s']:.2f} s"
    if r.get("searched"):
        s += f"（搜 {f(r.get('search_s'), '.2f')} s）"
    return s


rowsL = "\n".join(f"| {e:.2f} | " + " | ".join(cell(pick("lateral", n, e=e / 1e3)) for n in SN) + " |"
                  for e in E_LIST)
succ = {n: sum(1 for e in E_LIST if (pick("lateral", n, e=e / 1e3) or {}).get("ok")) for n in SN}
cap = TH["capture_mm"]
in_cap = [e for e in E_LIST if e <= cap]
out_cap = [e for e in E_LIST if e > cap]
adm_pred_ok = all((pick("lateral", "rcc_adm", e=e / 1e3) or {}).get("ok") == (e <= cap) for e in E_LIST)
pos_pred_ok = all((pick("lateral", "rcc_pos", e=e / 1e3) or {}).get("ok") == (e <= cap) for e in E_LIST)
srch = [pick("lateral", "rcc_adm_search", e=e / 1e3) for e in out_cap]

# ---- 傾角
TILTS = sorted({r["tilt"] for r in RUNS if r["exp"] == "tilt"})
WN = ["rigid", "rcc_Lc0", "rcc_Lc100"]
WLAB = {"rigid": "剛性腕", "rcc_Lc0": "RCC L_c = 0（柔順中心在銷端）", "rcc_Lc100": "柔順腕 L_c = 100 mm"}


def tcell(r):
    if r is None:
        return "—"
    return (f"{res(r)}／F {r['F_max']:.1f} N／M {r['M_max']:.3f} N·m／2 點 {f(r.get('tp_onset_mm'), '.2f')} mm"
            + (f"／{r['cycle_s']:.2f} s" if r.get("ok") else ""))


rowsT = "\n".join(
    f"| {t:.0f} | {f(TH['tilt_limit_mm'].get(str(t)), '.2f') if t > 0 else '∞'} | "
    + " | ".join(tcell(pick("tilt", w, tilt=t)) for w in WN) + " |" for t in TILTS)
rowsMuT = "\n".join(
    f"| {WLAB[r['name']]} | {r['mu']:.1f} | {res(r)} | {r['F_max']:.1f} | {r['M_max']:.3f} | {f(r.get('tp_onset_mm'))} | "
    f"{f(r.get('cycle_s'))} |"
    for r in sorted([r for r in RUNS if r["exp"] == "mu"], key=lambda r: (r["name"], r["mu"])))
JF = sorted([r for r in RUNS if "theta_max_deg" in r and r["tilt"] > 0],
            key=lambda r: (r["theta_max_deg"], r["tilt"]))
rowsJF = "\n".join(
    f"| {r['tilt']:.1f} | {r.get('F_d', IP['F_d']):.0f} | {r.get('mu', CS['mu']):.1f} | {r.get('K_rot', RCC['K_rot']):.0f} | "
    f"{r['theta_max_deg']:.2f} | {'不卡死' if r['tilt'] < r['theta_max_deg'] else '卡死'} | {res(r)} | "
    f"{f(r['jam_depth_pred_mm']) if math.isfinite(r['jam_depth_pred_mm']) else '—'} | "
    f"{f(r['depth_mm']) if not r['ok'] else '—'} | {f(r['F_max'], '.1f')} |" for r in JF)
agreeJF = sum(1 for r in JF if (r["tilt"] < r["theta_max_deg"]) == bool(r["ok"]))
jd = [(r["jam_depth_pred_mm"], r["depth_mm"]) for r in JF if not r["ok"] and math.isfinite(r["jam_depth_pred_mm"])]
jd_err = max(abs(b - a) for a, b in jd) if jd else float("nan")
rowsDemo = "\n".join(
    f"| {d['title'].replace('T109 ', '')} | {res(d)} | {f(d.get('depth_mm'))} | {f(d.get('F_max'), '.1f')} | "
    f"{f(d.get('M_max'), '.3f')} | {f(d.get('cycle_s'))} | {f(d.get('torque_util'), '.3f')} |" for d in M["demo"])

NARR = (ROOT / f"docs/test-reports/data/{T}_narrative.md")
narr = NARR.read_text(encoding="utf-8") if NARR.exists() else ""

narr = narr.replace("{len_jf}", str(len(JF)))
md = f"""# T109 軸孔插入 peg-in-hole（力控柔順）測試報告（{D}）

## 1. 目的
以 Phase 1 的手臂模型、計算力矩內環與 T108 的力感測/導納架構，完成工廠常見的軸孔裝配：
銷（Ø{PEG['r'] * 2e3:.0f} mm）插入帶 45° 倒角的孔（Ø{HOLE['R'] * 2e3:.2f} mm，徑向間隙 {TH['clearance_um']:.0f} µm，
間隙比 c* = {TH['clearance_ratio']:.4f}）至 {IP['depth'] * 1e3:.0f} mm 深。量化：
(a) 倒角滑入力與準靜態理論、(b) 孔位誤差（規劃端未知）下各策略的成功範圍與捕捉判據、(c) 銷在夾爪中的角度誤差
（兩點接觸、幾何極限、楔住判據）與柔順中心位置的影響、(d) 螺旋搜孔恢復、(e) 力/力矩峰值、週期時間、關節力矩使用率。

## 2. 模型與參數
* 程式：`sixaxis/insertion.py`（模組說明含全部公式與出處）。工具 `robots/ur5_peg.urdf`：法蘭 → 六軸力感測器 → RCC 上板（連桿 7），
  TCP（連桿 8）= RCC 未變形時的銷端中心，法蘭→TCP {TOOL_L * 1e3:.0f} mm。
* **RCC 柔順腕**（Whitney 1982）：銷體（銷＋夾爪＋RCC 下板，m = {PEG['mass']} kg、I_xx = {PEG['I_xx']:.1e}、I_zz = {PEG['I_zz']:.1e} kg·m²）
  與上板以 6 自由度彈簧–阻尼相連，柔順中心距銷端 L_c。
  K_lat = {RCC['K_lat']:.0f} N/m、K_ax = {RCC['K_ax']:.0e} N/m、K_rot = {RCC['K_rot']:.0f} N·m/rad、K_tor = {RCC['K_tor']:.0f} N·m/rad、ζ = {RCC['zeta']}。
  「剛性腕」基準：K_lat = {RIG['K_lat']:.0e}、K_ax = {RIG['K_ax']:.0e} N/m、K_rot = K_tor = {RIG['K_rot']:.0f} N·m/rad
  （更硬的值在 1 ms 手臂–工具顯式耦合下數值不穩，見 §6）。
* **接觸**：罰函數 + 錨點切向彈簧摩擦（Cundall & Strack 1979），可表示靜摩擦黏著；取樣：銷端外緣 {CS['n_rim']} 點 vs 工件實體有號距離
  （頂面/倒角/孔壁/孔底）、孔口兩條稜線各 {CS['n_edge']} 點 vs 銷圓柱（側面或端面）。μ = {CS['mu']}、
  k_rim = {CS['k_rim']:.0e} N/m、k_edge = {CS['k_edge']:.0e} N/m、切向剛性比 {CS['kt_ratio']}。銷體以 20 kHz 子步積分（手臂 1 kHz）。
* **控制**：內環 = 計算力矩（ω_n = 60 rad/s）＋力前饋 −Jᵀw_m；外環：
  - position：沿孔軸 v_ins = {IP['v_ins'] * 1e3:.0f} mm/s 直線下壓（力保護 {IP['F_max']:.0f} N）；
  - admittance：軸向 M z̈ + D ż = F_d − F_z（F_d = {IP['F_d']:.0f} N、M = {IP['M']}、D = {IP['D']:.0f} N·s/m、|ż| ≤ v_ins），
    側向 M_l ẍ + D_l ẋ = F_lat（D_l = {IP['D_lat']:.0f} N·s/m，側向力歸零 = 主動柔順）；
  - search：頂面著陸（力 ≥ 0.7 F_d、手部深度 < {IP['d_land'] * 1e3:.1f} mm、0.1 s）→ 以 F = {IP['F_search']:.0f} N 沿阿基米德螺旋
    （節距 {IP['pitch'] * 1e3:.1f} mm < 2·e_cap、{IP['v_search'] * 1e3:.0f} mm/s、最大半徑 {IP['r_search'] * 1e3:.0f} mm）搜孔，深度 > {IP['d_found'] * 1e3:.1f} mm 即落孔。
  - 判定：OVERFORCE（|F| > {IP['F_max']:.0f} N）、MISS（停在頂面）、JAM（孔內 {IP['t_jam']} s 無進展）、成功 = 銷端深度 ≥ {IP['depth'] * 1e3:.0f} − 0.2 mm。
* 感測：F/T 雜訊 σ = {IP['sigma_F']} N、一階低通 {IP['f_c']:.0f} Hz；感測器已扣除銷體靜重（歸零）。

### 準靜態理論（`sixaxis/insertion.py`）
| 項目 | 公式 | 本例數值 |
|---|---|---|
| 捕捉半徑 | e_cap = w_c + c | {cap:.2f} mm |
| 倒角滑入力比（α = 45°） | F_z/F_x = (1 + μ tanα)/(tanα − μ) | {TH['chamfer_ratio_mu03']:.3f}（μ = 0.3） |
| 倒角側向力峰值（L_c = 0） | F_x = K_lat·(e − c) | 見 §4.1 |
| 固定傾角 θ 的幾何極限深度 | d_b = (D_h − D cosθ)/tanθ − r sinθ（≈ Whitney (D_h − D cosθ)/sinθ） | 見 §4.3 |
| 楔住（Whitney） | 兩點接觸深度 l_2 < μD ⇔ θ ≳ 2c/(μD) | θ_w = {TH['wedge_deg']:.2f}° |
| 兩點接觸卡死 | N1 = −(M + (μr + l)F_x)/l、N2 = N1 + F_x；F_z ≤ μ(N1 + N2) | `two_point_statics` |
| 不卡死傾角（本專案推導） | θ_max = 2√(F_z c/(μ K_rot))（L_c = 0、F_x ≈ 0、θ(l) ≈ 2c/l） | {TMAX:.2f}°（F_z = 15 N、μ = 0.3、K_rot = 15） |

## 3. 可重現指令
```bash
python3 -m pip install "setuptools<70" wheel && python3 -m pip install --no-build-isolation pybullet==3.2.7
python3 -m pip install -r requirements.txt
ruff check . && python3 -m pytest -q
PYTHONPATH=. python3 scripts/t109_report.py {D}      # 掃描 4 行程平行，約 {M['total_s'] / 60:.0f} 分鐘
python3 scripts/t109_write_md.py {D} "<pytest 摘要>"
```

## 4. 指令序列與結果
每次插入：`move_j`（原點 → 孔口上方 {IP['approach'] * 1e3:.0f} mm）→ 直線下降至孔口上方 {IP['z_app'] * 1e3:.0f} mm（0.5 s，五次）→
插入（position / admittance，可含 search）→ 保持。實際孔位 = 規劃孔位 + e·(cos30°, sin30°)，規劃端未知。

### 4.1 倒角穿越台（無手臂，手部以 2 mm/s 等速下壓；RCC L_c = 0）
| e [mm] | 側向力峰值（模擬）[N] | K_lat(e − c) [N] | 差 [%] | 倒角段最大軸向力（模擬）[N] | 理論 K_lat(e − c)·(1+μ)/(1−μ) [N] | 全程軸向峰值（含轉角尖峰）[N] | 滑動段 F_ax/F_lat 中位數 |
|---|---|---|---|---|---|---|---|
{rowsC}

摩擦係數掃描（e = 0.8 mm）：

| μ | F_ax/F_lat（模擬中位數） | (1+μ)/(1−μ) | 差 [%] | 側向力峰值 [N] |
|---|---|---|---|---|
{rowsMu}

e − c ≥ 0.12 mm 的側向力峰值與 K_lat(e − c) 最大差 {devC:.1f} %；力比與準靜態式最大差 {devMu:.1f} %。

![chamfer](../media/{T}_chamfer.png)

### 4.2 孔位誤差掃描（手臂閉迴路，結果／最大力／週期）
| e [mm] | {' | '.join(SLAB[n] for n in SN)} |
|---|---|---|---|---|
{rowsL}

成功數：{'、'.join(f"{SLAB[n]} {succ[n]}/{len(E_LIST)}" for n in SN)}。捕捉半徑 e_cap = {cap:.2f} mm：
RCC＋導納的成敗與「e ≤ e_cap」{'完全一致' if adm_pred_ok else '不完全一致'}；RCC＋位置{'完全一致' if pos_pred_ok else '不完全一致'}。

![lateral](../media/{T}_lateral.png)

![series](../media/{T}_series.png)

### 4.3 銷在夾爪中的角度誤差（導納控制；表內：結果／最大力／最大力矩／兩點接觸起始深度／週期）
幾何極限欄 = w_c + d_b(θ)（銷端中心距頂面）。楔住臨界角 θ_w = {TH['wedge_deg']:.2f}°。

| θ [°] | 幾何極限 [mm] | {' | '.join(WLAB[w] for w in WN)} |
|---|---|---|---|---|
{rowsT}

θ = 3° 的摩擦係數變化：

| 腕 | μ | 結果 | 最大力 [N] | 最大力矩 [N·m] | 兩點接觸起始 [mm] | 週期 [s] |
|---|---|---|---|---|---|---|
{rowsMuT}

![tilt](../media/{T}_tilt.png)

### 4.4 不卡死傾角判據驗證（RCC L_c = 0、導納、側向力歸零）
θ_max = 2√(F_z c/(μ K_rot))（推導見 `jam_free_tilt`）；卡死深度預測 = w_c + l_j（`jam_depth`）。

| θ0 [°] | F_d [N] | μ | K_rot [N·m/rad] | θ_max [°] | 預測 | 模擬 | 預測卡死深度 [mm] | 模擬停止深度 [mm] | 最大力 [N] |
|---|---|---|---|---|---|---|---|---|---|
{rowsJF}

預測與模擬成敗一致 {agreeJF}/{len(JF)}；卡死深度預測與模擬停止深度最大差 {f(jd_err)} mm。

![jamfree](../media/{T}_jamfree.png)

### 4.5 展示（渲染）
| 情境 | 結果 | 深度 [mm] | 最大力 [N] | 最大力矩 [N·m] | 週期 [s] | 關節力矩使用率 |
|---|---|---|---|---|---|---|
{rowsDemo}

![anim](../media/{T}.gif)

[MP4]({'../media/' + T + '.mp4'})（{M['mp4_MB']:.2f} MB；GIF {M['gif_MB']:.2f} MB，{M['frames']} 幀）。左：全景；右：孔的剖視近照（工件 y > 0 半邊移除）。

| ![a](../media/{T}_key_a.png) | ![b](../media/{T}_key_b.png) |
|---|---|
| ![c](../media/{T}_key_c.png) | ![d](../media/{T}_key_d.png) |

![joints](../media/{T}_joints.png)

{narr}

## 7. 測試
`tests/test_insertion.py`：理論公式、兩點接觸靜力平衡、孔有號距離、銷體靜止懸吊、倒角台 vs 準靜態（側向力 ±10 %、力比 ±10 %）、
固定傾角幾何極限（逐點干涉搜尋）、工件網格、RCC＋導納插入成功、剛性腕位置控制 OVERFORCE、MISS 後螺旋搜孔成功。
全套結果：{PYT}
"""
(ROOT / f"docs/test-reports/{T}.md").write_text(md, encoding="utf-8")
print("written", T)
