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

## 2026-11-12｜T104 輸送帶追蹤抓取
- 新增 `sixaxis/tracking.py`：`BeltEstimator`（open：光電＋標稱帶速；track：相機 50 Hz 的 α-β 濾波）、`Tracker`（CLIK 線上追蹤，接計算力矩控制）、
  `track_pick`（同步 0.3 s → 追蹤下降 0.3 s → 追蹤中夾持 → 抬升 0.5 s 並減速）、手內偏移補償放置 `place_held`、後備 `static_pick`、
  `run_track_cycle`、連續來料 `run_flow`、追蹤窗判據 `x_end_pred`／`v_max_window`；腳本 `scripts/t104_report.py`、`t104_write_md.py`。
- 測試：`tests/test_tracking.py` 10 項；全套 133 passed、`ruff check` 通過。
- 關鍵數值：帶速 0.05–0.25 m/s 追蹤夾取皆成功（追蹤 RMS ≤ 0.41 mm、手內 |dx| ≤ 0.54 mm、放置 XY ≤ 0.55 mm、週期 4.85–5.94 s、力矩使用率 ≤ 0.23）；
  0.28／0.30 m/s 依判據 v_max = 0.260 m/s 判為超窗（WINDOW），後備靜態抓取成功；帶速誤差 ±20 %：開迴路誤差符合 −ε·v·Δt（最大差 0.16 mm）、3/7 成功，
  閉迴路 7/7；隨機情境 N=30：閉迴路 30/30、開迴路 25/30；雜訊 σ 0–5 mm 共 25/25 成功（追蹤 RMS ≈ 0.46σ，與 α-β 理論一致）；
  連續來料 4 件：週期 8／7 s 全部追蹤，6 s 追蹤 3＋靜態 1，5 s 追蹤 1＋靜態 3，皆 4/4 放好。
- 發現：從抓取高度直接 PTP 到輸送帶會掃到治具側銷 → `goto_wait` 先垂直抬到 0.25 m；報告腳本與 pytest 不可同時跑（都會重寫 `robots/ur5_gripper.urdf`）；
  夾爪模型閉合只需 ~6 ms（無速度上限），加 0.1 s 力穩定判定。
- 報告：docs/test-reports/2026-11-12_T104.md；動畫 docs/media/2026-11-12_T104.gif/.mp4。
- 限制：相機為模擬器讀值＋高斯雜訊（無延遲/遮擋）；工具姿態固定不處理偏航，橫向僅 ±12 mm；追蹤段力矩使用率為事後記錄、未事前規劃；v_max 受帶長 0.40 m 限制。
- 下一步：T105 銲接路徑（直線＋圓弧焊道、固定姿態與速度）。

## 2026-11-13｜T105 銲接路徑
- 新增 `sixaxis/welding.py`：彎頸 MIG 銲槍（`robots/ur5_torch.urdf`，TCP = 銲絲尖端）、平面焊道 `Seam`／`turtle_seam`（直線＋圓弧、弧長參數化、切線連續檢查）、
  銲槍姿態 `tool_rotation`（AWS work／travel 角相對焊道座標固定）、等速時間律 `SpeedProfile`、接點航向平滑（移動平均）、`plan_seam`、
  `best_roll`／`weld_auto`（利用銲絲軸對稱自動選 roll）、`WeldCell`（計算力矩或 PD＋g、碰撞預檢、進刀→起弧→銲接→收弧→退刀、量測）、JSON 工作 `parse_job`／`run_job`；
  腳本 `scripts/t105_report.py`、`t105_write_md.py`。
- 測試：`tests/test_welding.py` 22 項；全套 155 passed、`ruff check` 通過。
- 關鍵數值：標稱 J 形焊道（直線 120 + R40 圓弧 + 直線 80 mm、10 mm/s、work 90°／travel +10°）路徑誤差最大 0.06 µm、追蹤延遲 ≈ 10 µm（1 個控制週期）、
  速度偏差最大 0.037 %、角度誤差 ≤ 0.003°、力矩使用率 0.17；速度掃描 5–110 mm/s 可行，預測上限 v_max = 115 mm/s（圓角腕部角速度），120／150 mm/s 回報 LIMIT；
  R5 圓角航向平滑 b = 2 mm：關節加速度需求 522 → 5.9 rad/s²、速度偏差 3.27 → 0.91 %，接點航向偏差 5.73°（理論 b/4R = 5.73°）；
  銲槍角度矩陣 9/9 成功；T 形填角、管–板整圓（腕 3 轉 360°）、30°／45° 斜板成功，60° 斜板四個 roll 皆被碰撞預檢擋下（COLLISION）；
  PD＋重力補償 10 mm/s 路徑誤差 95 µm、速度偏差 15.9 %（計算力矩 0.06 µm、0.037 %）。
- 發現：整圓焊道 roll 0° 會超出 J6 ±2π 限位，必須利用繞銲絲的 roll 自由度；無航向平滑時 100／110 mm/s 的接點暫態使實際關節速度超過額定（1.06／1.17）；
  最近點量測須沿切線做一階修正並用解析切線，否則圓弧上會出現 ~0.07° 的假角度誤差。
- 報告：docs/test-reports/2026-11-13_T105.md；動畫 docs/media/2026-11-13_T105.gif/.mp4。
- 限制：µm 級誤差為理想模型（無摩擦/撓性/雜訊）；不模擬電弧與熔池；焊道限單一平面、切線連續；圓角不自動降速；碰撞只預檢不繞開；roll 只有四個候選；填角焊噴嘴間隙僅 1.5 mm。
- 下一步：T106 噴塗表面覆蓋（沿用焊道座標、等速時間律與工具姿態定義）。

## 2026-11-14｜T106 噴塗表面覆蓋
- 新增 `sixaxis/spraying.py`：直式噴槍（`robots/ur5_spray.urdf`，TCP = 噴嘴口）、beta 分佈點源沉積模型 `SprayModel`（Balkan & Arikan 1999；Conner et al. 2005 一般曲面推廣：ḣ = F(θ)·cosγ/ρ²）、
  單道解析解 g(y)、多道 CV 與「連續合格區」最大道距 `max_spacing`、面板 `Panel`、光柵路徑 `raster_path`（180° 圓弧換道、切線連續）、固定姿態規劃 `plan_spray`、
  `SprayCell(WeldCell)`（扳機區、膜厚由實際噴嘴位姿積分、轉移效率、路徑/速度/噴軸量測）、JSON 工作 `parse_job`；`WeldCell` 工具 URDF/TCP 參數化（`tool_writer`、`t_tcp`，預設不變）；
  腳本 `scripts/t106_report.py`、`t106_write_md.py`。
- 測試：`tests/test_spraying.py` 10 項；全套 165 passed、`ruff check` 通過。
- 關鍵數值：單道數值 vs 解析最大相對誤差 1.4e-9；標稱（400×300 mm、v 0.3 m/s、Δ 50 mm、h 200 mm、7 道 15.0 s）平均 22.09 µm（理論 Q/vΔ = 22.22）、
  CV 1.58 %（面板−10 mm）、±10 % 面積 99.9 %、轉移效率 73.1 %、開槍段路徑偏差 ≤ 0.005 mm、速度偏差 ≤ 0.053 %、力矩使用率 0.34；
  道距掃描 30–90 mm 中央 CV 模擬 vs 解析最大差 0.29 pp，CV ≤ 2 % 最大道距 53.9 mm；速度 0.1–0.7 m/s 膜厚 ∝ 1/v（最大差 0.59 %），v_max 預測 0.84 m/s、0.9 m/s LIMIT；
  噴距 150–250 mm CV 與解析差 ≤ 0.01 pp；扳機 0→60 mm＋側邊外延 25 mm：全面板 CV 13.1 → 0.21 %、覆蓋率 84 → 100 %，轉移效率 80 → 57.5 %；
  斜板 30°／45°（噴槍垂直板面）膜厚與平板一致（CV 1.55 %）；PD＋g 於 0.6 m/s 速度偏差 27.6 %、CV 2.67 %（計算力矩 0.053 %、1.47 %）。
- 發現：CV(Δ/R) 非單調（beta 分佈銳利邊界），採「從小往上首次超標前」的保守道距；直式噴槍工作空間緊：面板中心 x = 0.50 m 時遠角 IK 無解（改 0.45 m），
  噴槍傾角 +10°/+15°/−30° 無解（−30° 腕部奇異）；roll 只轉 J6，無法如銲槍般迴避。
- 報告：docs/test-reports/2026-11-14_T106.md；動畫 docs/media/2026-11-14_T106.gif/.mp4。
- 限制：理想點源（無霧化/氣流/靜電/反彈/遮蔽），參數為代表值；僅平面面板；膜厚每 2 ms 積分。
- 下一步：T107 鎖螺絲（定位＋姿態＋旋轉）。

## 2026-11-15｜T107 鎖螺絲
- 新增 `sixaxis/screwing.py`：電動起子工具（`robots/ur5_driver.urdf`，TCP = 起子頭尖端）、`ScrewSpec`（M4×12 8.8）、`ThreadModel`
  （VDI 2230 T = F·(0.16P + 0.58μ_G·d2 + μ_K·D_Km/2)、柔度法接合剛性 c = 1.97e8 N/m、應力利用率 ν、自鎖、斜牙判據 φ_ct = atan(P/2d)）、
  `Nutrunner`（主軸 J·ω̇ = T_m − T_load、速度 PI、10 kHz 子步、二段式扭力/扭力–轉角策略、主動煞車、旋入圈數/扭力/轉角監控窗）、`bench_test`（接合模擬台）、
  `ScrewCell(WeldCell)`（取料、浮動起子頭彈簧、入孔捕捉 e_cap = 0.75 mm、阿基米德螺旋搜尋、依編碼器跟隨進給、反作用外力＋前饋、斜面）、JSON `parse_job`／`run_job`；
  腳本 `scripts/t107_report.py`、`t107_write_md.py`。
- 測試：`tests/test_screwing.py` 13 項；全套 178 passed、`ruff check` 通過。
- 關鍵數值：標稱平板 6 孔（含取料）6/6 OK，峰值扭力 2.5008 N·m（+0.034 %）、夾緊力 3511 N（ν = 0.779）、進給力 24.9–25.1 N、週期 6.3–6.9 s；
  單段 600 rpm 硬接合過衝 +38.4 %（TORQUE_NG），二段 60 rpm +0.03 %，能量法估計（實際切斷轉速）與模擬同量級；
  μ 0.08–0.16 夾緊力離散：扭力控制 ±29 %、扭力–轉角控制 ±6 %；孔位偏差 ≤ 0.7 mm 直接入孔，0.8–2.5 mm 無搜尋 NO_ENGAGE、螺旋搜尋 0.05–1.22 s 全部恢復；
  傾角 ≤ 4.9° 正常、≥ 5.1° 斜牙（旋入 0.73 圈 vs 12.86）全部偵測；手臂不跟隨（c0 = 5 mm）BIT_LIFT；20° 斜板與 JSON 工作皆 OK。
- 發現：計算力矩控制在 J6 等效剛性僅 M66·ωn² ≈ 2.6 N·m/rad，無反作用前饋時工具被 2.5 N·m 反轉 47°（靜態估計 56°）並吃掉轉角（夾緊力 3436 vs 3511 N）；
  座合斜坡負載使主軸速度迴路落後（60 rpm 指令、切斷時 18 rpm）；μ = 0.08 扭力控制 ν = 1.0007 已達降伏但轉角窗仍判 OK（監控盲點）。
- 報告：docs/test-reports/2026-11-15_T107.md；動畫 docs/media/2026-11-15_T107.gif/.mp4。
- 限制：螺紋為解析模型（非接觸模擬），不含自攻/嵌入鬆弛/降伏塑性；斜牙判據與 k_ct 為簡化假設；入孔只看徑向誤差；參數為代表值。
- 下一步：T108 去毛邊/拋光（力控、接觸力維持）。

## 2026-11-16｜T108 去毛邊/拋光
- 新增 `sixaxis/finishing.py`：研磨主軸工具（`robots/ur5_polisher.urdf`，F/T 感測器＋主軸＋R 12.5 mm 研磨墊，TCP = 墊面中心）、工件高度場 `Surface`
  （偏移/斜率/起伏形狀誤差、毛邊窄稜、殘留毛邊量測）、`PadModel`（Winkler 彈性基礎 p = k_w·d、逐格 Coulomb 摩擦、Preston ∂H/∂t = −K_p·p·|v|）、
  `FinishCell(WeldCell)`（法向力導納 M z̈ + D ż = F_d − F_m (+K_i∫) 外環＋計算力矩內環、感測雜訊/低通、力前饋、位置控制基準、觸地判定、
  NO_CONTACT／OVERFORCE）、理論 `loop_model`／`predict_touchdown`（伺服柔度 Λ_n）／`routh_kcrit`／`pass_depth`（單道解析剖面）、`raster_on`／`line_on`；
  `plan_spray` 增加 `t_tool_inv` 參數（預設不變）；腳本 `scripts/t108_report.py`、`t108_write_md.py`。
- 測試：`tests/test_finishing.py` 10 項；全套 188 passed、`ruff check` 通過。
- 關鍵數值：光柵拋光（6 道 × 100 mm、道距 10 mm、20 N、25 mm/s）導納力控在名目/±0.6 mm 偏移/0.5° 傾斜/0.3 mm 起伏下 F RMS 誤差 0.08–0.29 N，
  移除平均 12.56 µm（解析 12.56）、名目 CV 12.4 %（解析 12.7 %）；位置控制：+0.6 mm → 49.4 N、−0.6 mm → 完全脫離、傾斜/起伏 RMS 誤差 7.2／6.3 N；
  觸地超越量（k_e 1e4–1e6 N/m）模擬 vs 伺服柔度模型最大差 6.6 pp，k_e = 2e6 N/m 兩種阻尼皆 NO_CONTACT；
  斜面 0.5–3° 穩態誤差 0.123–0.779 N（預測 D·v·tanα 0.131–0.786），K_i = 20 /s 後 |殘留| ≤ 0.017 N；
  單道中心深度 vs Preston 解析差 ≤ 0.14 %（10/25/50 mm/s）；去毛邊（0.37 mm 毛邊、容許 50 µm）硬墊 5e5 N/m 於 10／20／40 N 皆 6 道達成，
  逐道衰減比 0.689–0.699（解析 ρ = 0.6986），母材過切 35／75／156 µm；軟墊 5e4 N/m 每道僅 −3.6 %（解析 ρ = 0.9648）。
- 發現：硬墊去毛邊速率與設定力無關（毛邊壓力 ≈ k_w·h_b），過切 ∝ 力 → 應用最小可維持接觸的力；起伏面即使力控移除 CV 仍 40 %（足印內壓力集中在波峰）；
  剛性內環的 Routh 判據不適用（計算力矩伺服剛性僅 Λ_n·ω_n² ≈ 17.7 kN/m，低於多數 k_e）；無力前饋時墊摩擦使橫向路徑偏差 0.010 → 1.08 mm；
  表面抬升時力為「超出」F_d（測試原先符號寫反，已修正）。
- 報告：docs/test-reports/2026-11-16_T108.md；動畫 docs/media/2026-11-16_T108.gif/.mp4。
- 限制：接觸為 2.5D Winkler 高度場解析模型（非剛體接觸）；K_p、μ、k_e 為代表值；毛邊為頂面窄稜近似；導納參數不隨 k_e 自動調整；穩定邊界附近集中參數模型不可靠。
- 下一步：T109 軸孔插入 peg-in-hole（力控柔順；沿用導納外環與 F/T 感測）。

## 2026-11-18｜T109 軸孔插入 peg-in-hole
- 新增 `sixaxis/insertion.py`：力感測器＋RCC 工具（`robots/ur5_peg.urdf`，TCP = 名目銷端）、`Hole`（45° 倒角孔有號距離：頂面/倒角/孔壁/孔底＋孔口兩條稜線）、
  `PegBody`（銷＋夾爪＋RCC 下板 6 自由度剛體、RCC 柔順中心 L_c 的平移/轉動彈簧–阻尼、罰函數法向力＋Cundall–Strack 錨點切向摩擦（可黏著）、20 kHz 子步）、
  `InsertCell(WeldCell)`（每步封閉解 IK、計算力矩＋力前饋、position／軸向導納＋側向力歸零、阿基米德螺旋搜孔、MISS／JAM／OVERFORCE 判定、剖視工件網格）、
  理論 `capture_radius`／`chamfer_ratio`／`chamfer_forces`／`tilt_depth_limit`／`wedge_angle`／`two_point_statics`／`jam_free_tilt`／`jam_depth`；
  腳本 `scripts/t109_report.py`（4 行程平行掃描、`--anim` 只重做動畫）、`t109_write_md.py`。
- 測試：`tests/test_insertion.py` 11 項；全套 199 passed、`ruff check` 通過。
- 關鍵數值（Ø12 銷／Ø12.16 孔、徑向間隙 80 µm、倒角 1 mm、目標深度 20 mm）：倒角台滑動段 F_ax/F_lat 與 (1+μ)/(1−μ) 差 ≤ 0.5 %（μ 0–0.5）、
  側向力峰值與 K_lat(e − c) 差 +2〜+5 %（e ≥ 0.4 mm）；孔位誤差掃描：RCC＋位置／RCC＋導納在 e ≤ 1.05 mm 成功、≥ 1.2 mm 失敗，與捕捉半徑 w_c + c = 1.08 mm 完全一致，
  剛性腕僅 e ≤ 0.3 mm（側向力 42 N），加螺旋搜孔 8/8 成功（e = 2.5 mm 搜 3.27 s、週期 7.3 s）；成功插入週期 3.8 s、關節力矩使用率 ≤ 0.18；
  傾斜銷兩點接觸起始深度與幾何極限一致（2°：5.57 vs 5.58 mm）；RCC（K_rot 15）1° 成功、2–4° 卡死。
- 發現：推導不卡死傾角 θ_max = 2√(F_z c/(μ K_rot))（兩點接觸靜力 + θ(l) ≈ 2c/l + RCC 回復力矩），以改變 F_d／K_rot／μ 的 13 組模擬驗證成敗 13/13 一致，
  卡死深度預測比模擬淺 0.2–1.1 mm；柔順中心離開銷端（L_c = 100 mm）在 μ = 0.1、3° 仍卡死而 L_c = 0 成功；e = 2.5 mm 搜孔時銷緣勾入倒角未達落孔門檻，螺旋拖曳 RCC 側向力 15.6 N。
- 報告：docs/test-reports/2026-11-18_T109.md；動畫 docs/media/2026-11-18_T109.gif/.mp4。
- 限制：罰函數接觸穿透數十 µm（停止深度偏深）；剛性腕剛性受 1 kHz 顯式耦合穩定性限制（2e5 N/m）；未模擬取料、孔軸誤差、黏滑；不卡死判據為平面小角度 L_c = 0 近似。
- 下一步：T110 避障路徑規劃（RRT*／碰撞檢查）。

## 2026-11-19｜T110 避障路徑規劃
- 新增 `sixaxis/planning.py`：`CollisionChecker`（規劃專用 DIRECT client、私有暫存 URDF 可多行程平行；障礙物 16 個一組複合剛體——PyBullet 3.2.7
  `createCollisionShapeArray` 實測只保留 16 個子形狀；地板；32 對自碰撞；間隙 f = d − 餘裕 10/5 mm）、保證式邊檢查 `edge_free`
  （Schwarzer–Saha–Latombe 2005：D = Σ R_i|Δq_i|，f_a + f_b > D 即證明無碰撞，否則二分至 1 mrad）、`RRTStar`（star=False 即 RRT；k-nearest
  k = ⌈e(1+1/d) log n⌉、Informed 長橢球取樣、多目標 IK 解、rewire 成本傳遞）、`gamma_star`、`shortcut`、`time_path`（逐段 rest-to-rest 五次多項式）；
  `sixaxis/scenarios.py`（T006 工作站方塊＋牆場景、隨機立柱場景拒絕取樣、`add_to_sim`）；腳本 `scripts/t110_report.py`（4 行程平行）、`t110_write_md.py`。
- 測試：`tests/test_planning.py` 7 項；全套 206 passed、`ruff check` 通過。
- 關鍵數值：300 條隨機邊保證式判可行 289 條，以 2 mrad 密集複驗漏判 0、保守誤判 6；平均查詢 104 vs 279 次／邊；Lipschitz 比值最大 0.31（需 ≤ 1）；
  單次距離查詢 0.49 ms。牆場景（隔板 x 0.36–0.90、高 0.5 m）關節直線穿入隔板 94 mm；RRT* 第一解 6/6（5–45 次迭代）、RRT（η 0.5）2/6；
  1500 次迭代 Informed RRT* 成本 1.87 rad vs 均勻 3.03 rad；執行（2 段、週期 1.99 s、力矩使用率 0.27）最小間隙 18.8 mm、無接觸、追蹤誤差 ≤ 1.4 mrad。
  16 個必須規劃的隨機立柱場景：RRT* 13/16、RRT 9/16；13 條路徑物理執行全部無接觸，最小間隙 10.4–20.4 mm。
- 發現：6 維中半徑版 RRT* 的 r_n 被 η 截斷、鄰居太少，改 k-nearest；基本 RRT 在窄目標區陷入 bug trap；均勻 RRT* 找到第一解後幾乎不再進步，Informed 取樣有效；
  R_i 上界約保守 3 倍（Lipschitz 實測 0.31）。
- 報告：docs/test-reports/2026-11-19_T110.md；動畫 docs/media/2026-11-19_T110.gif/.mp4。
- 限制：方塊靜態障礙物；夾持工件未納入手臂幾何；路徑點停頓（未做平滑後重檢）；3 個窄通道隨機場景兩種規劃器皆未在預算內找到解（未確認是否有解）。
- 下一步：T111 多物件分類揀選（沿用規劃器並把夾持工件納入碰撞幾何）。

## 2026-11-20｜T111 多物件分類揀選
- 新增 `sixaxis/sorting.py`：俯視相機 `OverheadCamera`（深度緩衝反投影 (PV)⁻¹、背景高度圖相減、8-連通、頂面點質心、旋轉卡尺最小外接矩形、
  圓形度判圓、色度分類＋形狀/高度一致性、手臂自我遮罩→退到觀測姿態重拍）、`grasp_candidates`（指墊/掌體俯視矩形 vs 鄰物與靜態體 2.5D 干涉、
  夾取高度＝指墊上緣高出頂面 3 mm）、`plan_sequence`（Held–Karp DP／貪婪／掃描，含可抓相依、死結時取最長部分序列）、
  `SortCell`（滾動時域：拍照→排序→接近→夾取→在手檢查/重抓→搬運→放料盤格或料箱底）；料箱（130×180 mm）、相機支架、光柵立柱場景。
- `planning.CollisionChecker` 擴充：`Box.yaw`（有向方塊）、`set_movable`（可替換的其餘物件障礙）、`attach/detach`（夾持工件附在 TCP，
  含工件–障礙物/地板/手臂連桿距離與 Lipschitz 上界 ρ_6 放大）；`GraspScene`/`make_arm_with_gripper` 可指定私有 URDF 路徑（可平行執行）。
- 腳本 `scripts/t111_report.py`（4 行程平行、84 個物理場景，`--anim` 只重做動畫/圖）、`t111_write_md.py`。
- 測試：`tests/test_sorting.py` 8 項；全套 214 passed、`ruff check` 通過。
- 關鍵數值：感知 240/240 偵測與分類正確、位置誤差平均 0.45 mm（最大 0.72）、yaw 最大 0.19°；序列（30 場景估計）DP 比掃描少 12.9 %、比貪婪少 4.0 %，
  物理實際週期 DP 39.1 s／貪婪 41.0 s／掃描 43.5 s；完整系統 54 場景 324 件全部正確（標準 36、立柱 8、4 mm 密集 10），料盤 108/108 入座（格心 0.94 mm），
  力矩使用率 ≤ 0.49；消融：固定 ψ = 0 僅 29/72、立柱不規劃 0/8 場景（15 次碰撞）、密集不檢查鄰物 47/60（推動鄰物最大 48 mm）。
- 發現/修正：夾在中心時掌體下緣恰與 70 mm 方柱頂齊 → 死結；指墊上緣與頂面共面時 PyBullet 接觸法向翻轉、鬆爪黏住；
  懸空鬆爪時圓柱偶發黏在指墊上被帶走 → 改放到箱底；move_l 升降不經規劃器 → 加升降行程取樣間隙檢查選格。
- 報告：docs/test-reports/2026-11-20_T111.md；動畫 docs/media/2026-11-20_T111.gif/.mp4。
- 限制：理想相機（無雜訊、外參已知、已知背景）；物件直立不堆疊；升降段為取樣檢查；序列成本未計 RRT* 繞行；DP 為 Python（n = 6 約 4–11 s）。
- 下一步：T112 機台上下料。

## 2026-11-21｜T112 機台上下料
- 新增 `sixaxis/tending.py`：`Machine`（CNC 加工機：運動學自動門梯形速度曲線、門洞光幕停止、稜柱關節虎鉗（力上限＝夾緊力 200 N、固定鉗口基準、
  鉗口位置窗 ±0.5 mm＋座面檢知 0.3 mm）、主軸循環＋側向切削力 F = k_c a_p h = 35 N、加工結束胚料換成品（口袋移除 21.6 g、質心重算）、
  PLC 信號紀錄與互鎖：手臂在機台區拒絕關門、未關門/未夾緊/監控 NG 拒絕啟動、循環中拒絕開門、預約啟動）；
  `TendCell`（料盤胚料 → 門外等待點 → 教導直線進門 → 虎鉗上/下料 → 輸送帶出料；S1 依序／S2 緩衝交換；切屑 NOT_SEATED → 吹氣重放、
  料盤空格跳格；大範圍移動用 T110 保證式檢查/RRT*）；`cycle_model`、`period_terms`、`vise_capture_test`、`clamp_hold_test`。
  `PBRobot.render` 新增 `eye` 參數。腳本 `scripts/t112_report.py`（4 行程平行）、`t112_write_md.py`。
- 測試：`tests/test_tending.py` 9 項；全套 223 passed、`ruff check` 通過。
- 關鍵數值：10 組策略實驗（T_m = 4/10/15/25/40 s × S1/S2、各 4 件）全部成功、手臂–機台接觸 0、力矩使用率 ≤ 0.25；
  S1 每件交換 T_ex = 15.97 s、S2 = 11.67 s（＋加工期間側工作 10.94 s），S2 週期縮短 9.7–14.7 %（T_m 10–25 s），T_m = 4 s 時手臂瓶頸反慢 4.3 %
  （交叉點 T_m* = 5.00 s）；週期模型 T_c = T_open + T_ex + max(T_close + T_m, T_side) 以 T_m = 15 s 分量預測其他 24 個週期，最大誤差 0.078 s。
  上料 40 次：x 誤差 ≤ 0.016 mm（固定鉗口推移約 5.6 mm）、y 0.53 mm（不定位）、座面間隙 ≤ 0；虎鉗捕捉 dx ∈ [−6, +6] mm（= 開口間隙）；
  Coulomb 夾持門檻 2μ_jF_c + μ_f mg vs 模擬滑移起點誤差 −7.2～+3.5 %（6 組 F_c）；開門實測 1.634 s vs 模型 1.633 s；互鎖/光幕 7 項全部按預期；
  切屑 2/2 偵測恢復、不恢復時 PLC 拒絕啟動；門洞教導路徑最小間隙 49.2 mm。
- 發現/修正：`ik_near(sym=True)` 選到 ψ+π 解時工具在搬運中自轉 180°，夾持力升到 80–120 N 且鬆爪卡住（改固定 ψ 迴避，根因待查 T005 夾爪模型）；
  成品複合碰撞形狀使指墊接觸異常 → 改外形方塊；切屑時鉗口位置仍在窗內（只靠鉗口位置會漏判，座面檢知必要）。
- 報告：docs/test-reports/2026-11-21_T112.md；動畫 docs/media/2026-11-21_T112.gif/.mp4。
- 限制：加工以定時循環＋定值切削力表示；門/光幕無反應時間；鉗口無倒角、y 不定位；單夾爪（未模擬雙夾爪換料）；確定性模擬。
- 下一步：Phase 2 全部完成，進入 Phase 3：T201 Gymnasium 環境 reach（觀測/動作/獎勵、與規劃控制基準比較）。

## 2026-11-22｜T201 Gymnasium 環境：reach
- 新增 `sixaxis/rlenv.py`：`ReachEnv`（註冊 `SixAxisReach-v0`）：PyBullet 手臂＋地面、2 ms 物理子步、20 Hz 策略、100 步截斷；
  動作 joint（關節速度 ±1 rad/s）或 ee（末端線速度 ±0.3 m/s，姿態保持＋阻尼最小平方）→ q_ref 積分（防飽和窗 0.3 rad）→
  計算力矩內環 τ = ID(q, q̇, K_p e + K_d ė)（ω_n = 25、臨界阻尼、τ_max 飽和）；觀測 27 維；dense（−d − 0.01‖a‖² + 5·成功 − 10·碰撞）/sparse 獎勵；
  成功 = d < 2 cm 且 ‖ṗ‖ < 0.1 m/s；碰撞（link2–6 撞地、非相鄰連桿自碰）終止；目標於前方方塊取樣並要求工具朝下 IK 可達。
  基準 `PlannerPolicy`（IK＋關節五次多項式）、`ResolvedRatePolicy`（笛卡兒 P＋DLS）、`RandomPolicy`；`run_episode` 共同指標；`throughput`。
- 腳本 `scripts/t201_report.py`（4 行程，約 53 s）、`t201_write_md.py`；requirements 加入 gymnasium==1.4.0。
- 測試：`tests/test_rlenv.py` 13 項（check_env、gym.make、觀測/FK、決定性、零動作保持、速度追蹤、成功/截斷/撞地終止、基準成功、隨機失敗）；
  全套 236 passed、`ruff check` 通過。
- 關鍵數值（N = 200 相同目標）：規劃 100 % 成功、平均 1.19 s（P95 1.75 s）；解析率 joint/ee 皆 100 %、1.31 s；隨機 0 %（joint 1 % 撞地）；
  力矩使用率 ≤ 0.37；規劃成功時間為解析率 91.0 %。靜態 5 s 末誤差 < 0.01 µm。速度步階：六軸誤差相同、與離散理論（半隱式 Euler）逐點差 7e-16 rad，
  峰值 6.42 mrad（連續理論 7.36）；吞吐量 joint 876／ee 773 步/s（單核）。
- 發現：link1 外殼與地面 1 mm 接觸 → 初版每回合第一步判碰撞（改排除 base/link1）；dense −d 獎勵的回報排序（解析率 1.27 > 規劃 0.42）與成功時間排序相反。
- 報告：docs/test-reports/2026-11-22_T201.md；動畫 docs/media/2026-11-22_T201.gif/.mp4。
- 限制：僅位置目標、無感測雜訊/延遲/動力學隨機化、內環為理想模型。
- 下一步：T202 PPO/SAC 訓練 reach 與學習曲線（加入 stable-baselines3；checkpoint 存 models/）。

## 2026-11-23｜T202 PPO/SAC 訓練 reach 與學習曲線
- 新增 `sixaxis/rltrain.py`：`train`（SB3 PPO/SAC、Monitor 記錄訓練成功率、固定 50 目標（種子 5000–5049）確定性評估的學習曲線、
  以（成功率, 回報）存 `models/<組態>_best.zip`、結束存 `_last.zip`＋`_log.json`、wall-time 預算回呼、`resume=True` 續訓）、`SB3Policy`（接 T201 `run_episode`）、`evaluate`、`load`。
  腳本 `scripts/t202_train.py`（sac/ppo × ee/joint 四組各 1 核平行、各 1800 s）、`t202_report.py`、`t202_write_md.py`；requirements 加入 stable-baselines3==2.9.2、torch==2.15.0（CPU）。
- 測試：`tests/test_rltrain.py` 9 項（PPO/SAC 短跑＋存檔＋續訓步數累加、時間預算中止、確定性、已存 SAC-ee 模型 20 目標 ≥ 90 % 成功、模型檔 < 50MB 可載入）；
  全套 245 passed、`ruff check` 通過。
- 訓練（30 分鐘/組、1 核）：SAC-ee 110,289 步（61 步/s）30,000 步達 90 %；PPO-ee 700,422 步、120,000 步；
  SAC-joint 114,820 步、80,000 步；PPO-joint 814,036 步、80,000 步（3.1 分鐘）。四組評估集皆 100 %。
- 關鍵數值（T201 同 200 目標）：四組 100 % 成功、0 碰撞；平均成功時間 SAC-ee 0.85 s、PPO-ee 0.89、SAC-joint 0.57、PPO-joint 0.59
  vs 規劃 1.19、解析率 1.31（同目標時間比 0.48–0.75）；相對速度上限參考時間比 0.89–1.01
  （規劃 1.85，理論 1.875/0.9 = 2.08）；末距離 10–14 mm（規劃 3.0）；最大力矩使用率 joint 策略 0.81–0.84（規劃 0.29）。
  起點擾動 ±0.3 rad：四組仍 100 %。決策 0.3–0.4 ms/步。
- 發現：成功即終止使「到達後」行為未受訓練——不終止跑滿 5 s 時 PPO-joint 只有 2 % 回合停在容差內（SAC-joint 96 %、ee 兩組 100 %）；
  dense −d 獎勵＋終止使策略剛跨過 2 cm 門檻即停，並以接近 bang-bang 的關節速度換時間。
- 報告：docs/test-reports/2026-11-23_T202.md；動畫 docs/media/2026-11-23_T202.gif/.mp4；模型 models/（8 個 zip，每檔 ≤ 3.3 MB）。
- 限制：單一種子/超參數；無雜訊/延遲/隨機化；SAC 在此 CPU 僅約 60 步/s。
- 下一步：T203 pick（夾爪抓取環境；改不終止＋停留/抓穩獎勵）。
