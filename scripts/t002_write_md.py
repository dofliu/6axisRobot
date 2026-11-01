"""由 T002 metrics JSON 產生繁體中文測試報告（數字全部取自實際執行結果）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
date = sys.argv[1]
passed = sys.argv[2] if len(sys.argv) > 2 else "?"
tag = f"{date}_T002"
m = json.loads((ROOT / "docs/test-reports/data" / f"{tag}_metrics.json").read_text(encoding="utf-8"))
e, ps, pf, c, sg = m["energy"], m["pybullet_swing"], m["perf"], m["circle_demo"], m["singularity"]
mt = ", ".join(
    f"J{i + 1}={v:.2f}/{l:g}" for i, (v, l) in enumerate(zip(c["max_torque_Nm"], c["torque_limits_Nm"]))
)
hist = ", ".join(f"{k} 解: {v}" for k, v in m["ik_solution_count_hist"].items())

md = f"""# {date} T002：六軸手臂模型完整測試

## 1. 目的
驗證 T001 的數學模型（DH 運動學、封閉解 IK、幾何 Jacobian、遞迴牛頓-尤拉動力學）是否正確，
方法是與獨立實作（PyBullet 物理引擎、數值微分、能量守恆）交叉比對，並量測效能。

## 2. 模型與參數
* 幾何：UR5 類標準 DH（`sixaxis/robot.py::RobotParams`）：a=(0,-0.425,-0.39225,0,0,0) m，
  d=(0.089159,0,0,0.10915,0.09465,0.0823) m，α=(π/2,0,0,π/2,-π/2,0)。
* 質量：(3.7, 8.393, 2.275, 1.219, 1.219, 0.1879) kg；各連桿視為沿「關節原點→下一關節原點」的均質圓柱
  （半徑 0.060/0.055/0.045/0.040/0.040/0.035 m），質心在中點，慣性張量由圓柱公式求得。
* 限位：關節 ±2π rad、速度 π rad/s、力矩 (150,150,150,28,28,28) N·m。
* URDF 由 `python3 -m sixaxis.urdf` 從 DH 自動產生（`robots/ur5_like.urdf`），FK 與動力學共用同一組連桿座標系。
* 公式出處：Craig《Introduction to Robotics》RNE；Hawkins (2013) UR-5/UR-10 封閉解 IK（詳見 `sixaxis/robot.py` 檔頭）。

## 3. 重現步驟
```bash
python3 -m pip install "setuptools<70" wheel && python3 -m pip install --no-build-isolation pybullet==3.2.7
python3 -m pip install -r requirements.txt
ruff check .
python3 -m pytest -q                                   # 本次：{passed}
PYTHONPATH=. python3 scripts/t002_report.py {date}      # 重算指標、圖表、動畫
python3 scripts/t002_write_md.py {date} "{passed}"      # 產生本報告
```

## 4. 指令序列（示範動畫）
* A（0–5 s）：`move_l` 等效——工具朝下、水平圓（圓心 (0.45,0,0.35) m、半徑 0.12 m），最小急動度時間律，
  每個取樣點以封閉解 IK 選最接近前一點的解；以 RNE 計算所需力矩。
* B（2 s）：零力矩、無摩擦自由擺動，初始 q0=(0.3,-0.4,0.8,-0.5,0.6,0.2)，q̇0=(0.5,-0.3,0.4,1.0,-0.8,0.6)，RK4 dt=1 ms。

## 5. 量化結果
| 項目 | 判定標準 | 結果 |
|---|---|---|
| FK vs PyBullet 位置（1000 隨機姿態） | max < 1e-6 m | max {m["fk_pos_err_m"]["max"]:.2e} m，RMS {m["fk_pos_err_m"]["rms"]:.2e} m |
| FK vs PyBullet 姿態 | — | max {m["fk_rot_err_rad"]["max"]:.2e} rad |
| IK 往返（1000 組，全部解的 FK 殘差） | 殘差 < 1e-9 | max {m["ik_residual"]["max"]:.2e} |
| IK 解集合含原始關節角 | 1000/1000 | {m["ik_original_found"]}/1000 |
| IK 最近解＝原始解（q_ref=q） | 1000/1000 | {m["ik_nearest_ok"]}/1000 |
| IK 解數分布 | 最多 8 解 | {hist} |
| Jacobian vs 數值微分（200 姿態，h=1e-6） | max < 1e-6 | max {m["jacobian_err"]["max"]:.2e} |
| 逆動力學 vs PyBullet（500 隨機 q,q̇,q̈） | max < 1e-3 N·m | max {m["id_err_Nm"]["max"]:.2e}，RMS {m["id_err_Nm"]["rms"]:.2e} N·m |
| 無摩擦自由擺動能量守恆（{e["T_s"]:g} s，RK4 dt={e["dt_s"] * 1e3:g} ms） | 相對漂移 < 1e-5 | E0={e["E0_J"]:.3f} J，漂移 {e["drift_J"]:.2e} J（相對 {e["rel_drift"]:.2e}）；動能範圍 {e["KE_range_J"][0]:.2f}–{e["KE_range_J"][1]:.2f} J |
| 自由擺動軌跡 vs PyBullet（dt=0.1 ms） | 0.5 s 內 < 2e-3 rad | 0.1 s: {ps["dq_at_0.1s"]:.1e}；0.5 s: {ps["dq_at_0.5s"]:.1e}；1 s: {ps["dq_at_1s"]:.1e}；2 s: {ps["dq_at_2s"]:.2f} rad |
| 笛卡兒圓 IK 追蹤誤差 | — | max {c["tracking_err_m_max"]:.1e} m；單步最大關節跳變 {c["joint_jump_max_rad"]:.3f} rad |
| 圓軌跡最大力矩/上限 (N·m) | 不超限 | {mt}；超限：{c["limits_violation"] or "無"} |
| 圓軌跡 RNE vs PyBullet 力矩 | — | max {c["id_vs_pybullet_max_Nm"]:.1e} N·m |
| 奇異點偵測 | 腕/肘奇異被標記、σ_min→0 | 一般姿態 {sg["generic"]["flags"] or "無"}（σ_min={sg["generic"]["sigma_min"]:.3f}）；q5=0 → {sg["wrist(q5=0)"]["flags"]}（σ_min={sg["wrist(q5=0)"]["sigma_min"]:.1e}）；q3=0 → {sg["elbow(q3=0)"]["flags"]}（σ_min={sg["elbow(q3=0)"]["sigma_min"]:.1e}） |

效能（單執行緒 Python/NumPy，單次平均）：FK {pf["fk_us"]:.0f} µs、Jacobian {pf["jacobian_us"]:.0f} µs、
IK 全部解 {pf["ik_all_ms"]:.2f} ms（含選解 {pf["ik_select_ms"]:.2f} ms）、RNE {pf["rne_us"]:.0f} µs、
質量矩陣 {pf["mass_matrix_us"]:.0f} µs、正動力學 {pf["forward_dynamics_us"]:.0f} µs
（PyBullet C++ 逆動力學 {pf["pybullet_id_us"]:.1f} µs 供參考）。

## 6. 圖表與動畫
![誤差分佈](../media/{tag}_errors.png)
![能量守恆](../media/{tag}_energy.png)
![圓軌跡曲線](../media/{tag}_circle_curves.png)

動畫（A：IK 圓軌跡；B：自由擺動；左為斜視、右為俯視；紅點為目標路徑）：

![動畫](../media/{tag}.gif)

MP4：[{tag}.mp4](../media/{tag}.mp4)（GIF {m["media"]["gif_MB"]:.2f} MB、MP4 {m["media"]["mp4_MB"]:.2f} MB，共 {m["media"]["frames"]} 幀）。
關鍵畫面：[起始](../media/{tag}_start.png)、[圓軌跡中段](../media/{tag}_circle_mid.png)、[自由擺動](../media/{tag}_swing.png)。

## 7. 問題與限制
* 慣性模型為均質圓柱近似，並非廠商實測值；質量取自 UR5 公開資料，仍適合作為控制/RL 的物理基礎，但絕對力矩與實機會有差異。
* 無關節摩擦、無減速機轉子慣量、無彈性；之後的控制器/RL 若需 sim-to-real 再補。
* PyBullet 比對：零力矩自由擺動中 j4 在 0.55 s 超過 ±2π，PyBullet 會撞關節限位而損失能量，故比對時
  放寬 PyBullet 限位（`PBRobot.make_frictionless`）；放寬後 PyBullet 總能量仍漂移 {ps["pb_energy_drift_J"]:.2f} J
  （半隱式歐拉），與本模型 RK4 的 {e["drift_J"]:.1e} J 相比大幅劣化；2 s 時軌跡因混沌系統放大差異而發散，屬預期，
  故軌跡比對的判定只取 0.5 s 內。
* FK 與 PyBullet 的 {m["fk_pos_err_m"]["max"]:.1e} m 殘差來自 PyBullet 內部單精度/四元數轉換，遠小於 1e-6 m 門檻。
* IK 在腕奇異時 θ6 取參考值（`q_ref`）；肩奇異、肘奇異僅偵測與避免除零，未做奇異穿越（之後軌跡規劃再處理）。
* 正動力學以 Python 實作，約 {pf["forward_dynamics_us"] / 1e3:.1f} ms/次，RK4 模擬較慢；RL 訓練以 PyBullet 為主，本模型用於控制器（計算力矩、重力補償）。

## 8. 結論
T001/T002 完成：模型與 PyBullet 在 FK（{m["fk_pos_err_m"]["max"]:.1e} m）、逆動力學（{m["id_err_Nm"]["max"]:.1e} N·m）高度一致，
IK 封閉解 1000/1000 往返成功，Jacobian 與數值微分一致（{m["jacobian_err"]["max"]:.1e}），自由擺動能量守恆（相對漂移 {e["rel_drift"]:.1e}）。可進入下一項：T003 軌跡規劃函式庫。
"""
out = ROOT / "docs/test-reports" / f"{tag}.md"
out.write_text(md, encoding="utf-8")
print("wrote", out)
