# PROGRESS（每次執行追加，最新在最下）

## 2026-11-01｜Phase 0 建置 + T001 + T002
- 建立專案骨架：CLAUDE.md、docs/ROADMAP.md、requirements.txt、pyproject.toml（ruff/pytest）。
- T001：`sixaxis/robot.py`（DH、FK、Hawkins 封閉解 IK 最多 8 解、LM 數值 IK、幾何 Jacobian、奇異/限位偵測、RNE 逆動力學、質量矩陣、正動力學、RK4 模擬、能量）；`sixaxis/urdf.py` 產生 URDF；`sixaxis/pbsim.py` PyBullet 介面。
- T002：23 項 pytest 全數通過；`ruff check` 通過。關鍵指標：FK vs PyBullet 位置 max 4.0e-8 m；IK 1000/1000 往返、殘差 7e-12；Jacobian 誤差 2e-10；逆動力學 vs PyBullet max 1.3e-5 N·m；自由擺動能量相對漂移 3.3e-7。
- 報告：docs/test-reports/2026-11-01_T002.md；動畫：docs/media/2026-11-01_T002.gif/.mp4。
- 發現並修正：URDF 慣性值在 numpy 2 下被寫成 `np.float64(...)` 導致 PyBullet 慣性為 0；PyBullet 自由擺動需放寬關節限位且不可含地面。
- 下一步：T003 軌跡規劃函式庫。

## 2026-11-01｜T003 軌跡規劃函式庫
- 注意：本次 Routine 開始實作時，遠端 main 已有另一次 Routine 推送的 T001+T002（不同實作）；本次捨棄自己的重複版本
  （僅留在本地備份分支、未推送），改在遠端版本上接續做 T003。
- 新增 `sixaxis/trajectory.py`：五次/梯形時間律、`joint_ptp`（含依速度上限的最短時間）、`LinePath`+SLERP、
  `ArcPath`（三點圓弧）、`sample_cartesian`（IK+J⁺ 速度+RNE 力矩）、`concat`、`check_samples`、`fit_duration`（時間縮放）。
- 測試：`tests/test_trajectory.py` 12 項；全套 pytest 與 `ruff check` 通過（見報告）。
- 報告：docs/test-reports/2026-11-01_T003.md；動畫：docs/media/2026-11-01_T003.gif/.mp4。
  關鍵：直線度/圓弧半徑誤差 ≤1e-16 m 量級；笛卡兒終點誤差 1e-16；示範 823 點 IK 全數收斂；
  PyBullet FK 一致 2.9e-8 m；T=0.4 s 超速 2.98× → 自動縮放為 1.256 s 後速度使用率 0.95。
- 下一步：T004 控制器（PD+重力補償、計算力矩、阻抗/導納）。

## 2026-11-01｜T002a 手臂外觀擬真化（依 Dof 要求插入）
- 新增 `sixaxis/appearance.py`（UR 型外觀：依實機肩/肘偏置把上臂、前臂畫在平行關節軸方向的偏置平面，DH 不變；外露端判斷加藍色端蓋）與 `sixaxis/meshgen.py`（旋轉體 OBJ 網格，圓角、平滑法向、管件細分）；`urdf.py` 改為視覺用網格、碰撞用簡單圓柱。
- 渲染：光源、陰影、2× 超取樣反鋸齒、深度合成背景、工廠地坪貼圖（`robots/textures/floor.png`）。
- 測試 28 passed（新增 5 項外觀測試）；rebase 到 T003 之上後全套 40 passed；ruff 通過。標準姿態無自我碰撞（最小間隙 0.16 mm，底座↔肩外殼）；隨機關節角自我碰撞比例 47.7%。
- 媒體：docs/media/2026-11-01_T002a.gif（5.3 MB）/.mp4、before_after/hero/home/wrist_closeup PNG；報告 docs/test-reports/2026-11-01_T002a.md。
- 限制：慣性未隨外觀偏置移動（刻意保持 T002 結果），之後可另開項目更新。
- 註：另一次 Routine 已在本項之前推送 T003（其動畫為舊外觀，依規則保留不重做）；本項 rebase 於其上。
- 下一步：T004 控制器（之後的動畫將使用新外觀）。

## 2026-11-01｜T003 動畫以新外觀重新渲染（v2，依 Dof 要求）
- 背景：本日 Phase 0 為手動啟動，排程 Routine 隨即也啟動並完成 T003，其動畫為 T002a 之前的舊外觀。
- 新增 `scripts/t003_render_v2.py`：以相同程式碼重建示範軌跡（核對 823 點、8.216 s 與原 metrics 一致），用新外觀渲染；輸出 docs/media/2026-11-01_T003_v2.gif（1.65 MB）/.mp4 與關鍵畫面。原 T003 動畫與報告內容保留，報告末尾加附錄。
- 下一步：T004 控制器。
