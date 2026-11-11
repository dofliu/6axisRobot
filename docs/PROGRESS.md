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

## 2026-11-02｜T004 控制器
- 新增 `sixaxis/control.py`（PD+重力補償、計算力矩、笛卡兒阻抗、平移導納 + 內環追蹤器）與 `sixaxis/ctrlsim.py`（PyBullet TORQUE_CONTROL 閉迴路，dt=1 ms）。
- 測試：`tests/test_control.py` 6 項；全套 46 passed、`ruff check` 通過。
- 關鍵數值：CT 軌跡追蹤 RMS 1.15e-5 rad（PD+g 1.12e-3；質量 +20% 誤差模型 1.05e-3）；PD+g 階躍整定 0.326 s／CT 0.149 s；阻抗位移與 F/K 相對誤差 <1e-3；導納穩態 −27.5 mm vs 期望 −25 mm（內環柔度）。
- 發現：腕軸等效慣量 ~1e-4 kg·m²，固定增益在 1 ms 離散控制下發散 → 增益須依 M_ii 縮放。
- 報告：docs/test-reports/2026-11-02_T004.md；動畫 docs/media/2026-11-02_T004.gif/.mp4。
- 下一步：T005 夾爪與物件模型。

## 2026-11-03｜T005 夾爪與物件模型
- 新增 `sixaxis/gripper.py`（平行夾爪 URDF 片段、GEAR 連動、物件規格 cube/block/cyl、F_min 解析式）與 `sixaxis/graspsim.py`（抓取場景、計算力矩手臂、取放/靜態滑移實驗）；產生 `robots/ur5_gripper.urdf`（原手臂 URDF 不變）。
- 測試：`tests/test_gripper.py` 9 項；全套 55 passed、`ruff check` 通過。
- 關鍵數值：40 N 夾持下三種物件抬升 15 cm 滑移 ≈1 µm、釋放漂移 <0.11 mm；靜態夾持力轉折 1.2→1.3 N（解析 1.226 N）；快速抬升 3.6 m/s² 轉折 1.5→2.0 N（解析 1.68 N）。
- 發現：對手指關節呼叫 changeDynamics(maxJointVelocity) 會破壞手臂動力學；兩指須用 GEAR 連動、frictionAnchor 抑制蠕滑；PyBullet 位置馬達/每步重置驅動手臂皆造成假滑移，改用 PyBullet 逆動力學之計算力矩。
- 報告：docs/test-reports/2026-11-03_T005.md；動畫 docs/media/2026-11-03_T005.gif/.mp4。
- 下一步：T006 場景（工作台、輸送帶、料盤、固定治具；碰撞群組）。

## 2026-11-04｜T006 工作站場景
- 新增 `sixaxis/scene.py`（`Workcell`：工作台、輸送帶＋擋料板/護欄、2×2 料盤、固定治具、碰撞群組 arm/static/belt/part、`arm_collisions`/`min_clearance` 查詢）；`GraspScene(workcell=True)` 可使用（預設行為不變）。
- 輸送帶以庫侖限幅速度追隨力建模（帶面摩擦 0）。測試：`tests/test_scene.py` 19 項；全套 74 passed、`ruff check` 通過。
- 關鍵數值：帶速誤差 0.00 %、擋料板定位 0.011 mm；料盤 4 格×4 偏移全收容；治具滑入基準 RMS 2.4 mm／max 5.6 mm（非精密，已於報告說明）；
  7 工位首選 IK 解皆無碰撞；示範兩個取放週期（輸送帶→格0、桌面→格3）放置誤差 <0.5 mm、最小間隙 5.0 mm、無碰撞。
- 發現：角柱料盤只擋斜向（改端板+側銷）；夾爪抱著工件做 PTP 不可橫掃（會掃走桌面零件／擦到治具）→ 需先垂直抬起與中繼點，T110 避障規劃處理。
- 報告：docs/test-reports/2026-11-04_T006.md；動畫 docs/media/2026-11-04_T006.gif/.mp4。
- 下一步：T007 指令介面（JSON/DSL：move_j、move_l、move_c、grip、release、wait）。

## 2026-11-05｜T007 指令介面
- 新增 `sixaxis/commands.py`：JSON/DSL 解析與驗證（SYNTAX/RANGE）、`Executor`（運動學 dry-run 或 GraspScene 物理）、指令 move_j/move_l/move_c/grip/release/wait/set_speed；
  未給 T 時以 fit_duration 自動求時（使用率 ≤0.5·speed），給 T 超限 → LIMIT；另有 UNREACHABLE、COLLISION、GRIP_EMPTY；遇錯即停並回報序號與錯誤碼。
- 測試：`tests/test_commands.py` 26 項；全套 100 passed、`ruff check` 通過。
- 關鍵數值：兩個示範程式（DSL 桌面→料盤、JSON 輸送帶→料盤）皆成功，放置 XY 誤差 0.49／0.53 mm、Z 誤差 −0.01 mm；指令終點 TCP 誤差最大 0.74 mm；速度/力矩使用率最大 0.50／0.30。
- 報告：docs/test-reports/2026-11-05_T007.md；動畫 docs/media/2026-11-05_T007.gif/.mp4。
- 限制：move_l/move_c 無姿態目標；力矩檢查未含夾爪/工件質量；碰撞僅手臂對靜態場景。
- 下一步：T101 點到點搬運（Phase 2）。

## 2026-11-06｜T101 點到點搬運
- 新增 `sixaxis/tasks.py`（`transfer_program` 產生 8 道 T007 指令、`random_pairs`、`run_transfer` 量測成功率/誤差/週期/使用率）；腳本 `scripts/t101_report.py`、`t101_write_md.py`。
- 測試：`tests/test_tasks.py` 6 項；全套 106 passed、`ruff check` 通過。
- 關鍵數值：隨機起訖 N=30 成功 30/30；放置 XY 誤差平均 0.42／最大 0.80 mm、傾角最大 1.53°、週期平均 5.02 s、速度使用率 ≤0.50、力矩使用率最大 0.43；夾持力 5–80 N、speed 0.4–1.0、五次/梯形曲線皆成功。
- 發現：取放區必須縮到 x 0.36–0.44、y −0.01–0.02 m：較大範圍會與治具後擋板碰撞或角落抓取落空（一指 ~670 N、另一指 0），原因未深入；更大範圍留給 T110 避障。
- 報告：docs/test-reports/2026-11-06_T101.md；動畫 docs/media/2026-11-06_T101.gif/.mp4。
- 下一步：T102 取放 pick & place（可沿用 tasks.py，加入工件位置/姿態偏差與失敗處理）。

## 2026-11-10｜T102 取放 pick & place
- `sixaxis/tasks.py` 新增 `pick_place_program`（放入料盤格高度）、`run_pick_place`（可選 `recover=True`：抬升後 `inhand_ok` 檢查 → 放回 → 鬆開退到安全高度 → 讀回方塊位置重新感測 → 重取，最多 3 次）、`inhand_ok`；`transfer_program` 加 `z_place`；腳本 `scripts/t102_report.py`、`t102_write_md.py`。
- 測試：`tests/test_pickplace.py` 5 項；全套 112 passed、`ruff check` 通過。
- 關鍵數值：標稱 N=24 輪流放入 4 格 24/24 成功，放置 XY 最大 0.62 mm、週期平均 5.33 s、力矩使用率 ≤0.21；
  取點偏差：y 向 ±14 mm 內由夾爪對中吸收，|y|≥16 mm 指墊壓倒方塊（UNRECOVERABLE）；x 向無恢復 ±4 mm 即失敗，有恢復 ±20 mm 皆成功（2 次取點）；
  隨機感測誤差 ±12 mm、N=30：無恢復 4/30 → 有恢復 30/30。
- 發現：傾角/偏航須分開（總旋轉角會把偏航算進去；T101 的「傾角」為總旋轉角）；x 向偏差不會被兩指修正，須在抬升後檢查。
- 報告：docs/test-reports/2026-11-10_T102.md；動畫 docs/media/2026-11-10_T102.gif/.mp4。
- 限制：「感測」為模擬器直接讀值（理想）；工具姿態固定，不處理 yaw 偏差。
- 下一步：T103 疊棧/碼垛（沿用 `run_pick_place`，z 逐層遞增）。

## 2026-11-11｜T103 疊棧/碼垛
- 新增 `sixaxis/stacking.py`：樣式 `tower_slots`／`pyramid_slots`／`column_slots`、`stack_program`（搬運高度隨層數升高）、`run_stack`（供料點逐塊補料、
  每塊放完檢查全堆是否在位、倒塌即停、量測各層誤差/擾動/週期）、靜力穩定判據 `stack_margins`／`stable_height`（介面上方合質心須落在接觸區內；等距錯位 d < a/n）；
  腳本 `scripts/t103_report.py`、`t103_write_md.py`。
- 測試：`tests/test_stacking.py` 11 項；全套 123 passed、`ruff check` 通過。
- 關鍵數值：垂直疊棧 10/10 層成功，XY 最大 0.90 mm、|Z| 最大 0.120 mm、傾角最大 0.029°、平均週期 6.55 s、力矩使用率 ≤0.43；
  金字塔 3-2-1 與柱式 2×3 皆 6/6（XY 最大 0.54／0.72 mm）；錯位塔 d=6–27 mm 倒塌層數與解析 d<a/n 一致 10/10；
  隨機放置誤差 ±5/10/15/20 mm（各 N=6）堆滿 6 層 6/6、6/6、4/6、0/6，判據預測層數一致 24/24。
- 發現：標稱高度釋放會把下層擠動 0.50 mm，抬高 1 mm 釋放降到 0.01 mm；有 4 案最終樣式判據穩定但中途前綴不穩而倒塌 → 須逐個中間狀態檢查。
- 報告：docs/test-reports/2026-11-11_T103.md；動畫 docs/media/2026-11-11_T103.gif/.mp4。
- 限制：供料為理想生成（未接輸送帶）；工具姿態固定，同層只能沿 x 排列，無法逐層旋轉交錯；判據不含動態衝擊。
- 下一步：T104 輸送帶追蹤抓取。
