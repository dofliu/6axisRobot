# 6axisRobot 專案規則

## 目標
建立具物理與數學模型基礎的六軸機械手臂模擬系統，能依指令執行工廠常見動作，最終以強化學習達成主動動作控制與主動判斷完成工作。
由排程 Routine 自動開發：**每次執行只推進 `docs/ROADMAP.md` 的一項**。

## 技術棧
Python 3.11+、NumPy/SciPy（自寫運動學與動力學）、PyBullet（物理、碰撞、DIRECT 離屏渲染）、pytest、ruff、
imageio/ffmpeg（GIF/MP4）、matplotlib；Phase 3 加入 Gymnasium + Stable-Baselines3（PPO/SAC）。版本見 `requirements.txt`。

環境安裝注意：pybullet 無現成 wheel，需 `python3 -m pip install "setuptools<70" wheel` 後
`python3 -m pip install --no-build-isolation pybullet==3.2.7`；系統 `pip` 可能對應其他 Python，請一律用 `python3 -m pip`。
直接執行 scripts 需 `PYTHONPATH=.`。

## 程式結構
* `sixaxis/robot.py`：參數、FK/IK/Jacobian、RNE 動力學（`SixAxisRobot`）
* `sixaxis/urdf.py`：由 DH + 外觀產生 `robots/ur5_like.urdf` 與 `robots/meshes/*.obj`（改參數/外觀後需重新產生：`python3 -m sixaxis.urdf`）
* `sixaxis/appearance.py`、`sixaxis/meshgen.py`：UR 型外觀（只影響 visual/collision，不影響慣性與運動學）
* `sixaxis/pbsim.py`：PyBullet 介面（`PBRobot`）。數值比對請用不含地面的實例；渲染才加地面（`plane=True`，含工廠地坪貼圖）；`render()` 已含光影/陰影/反鋸齒，展示動畫請沿用
* `sixaxis/tasks.py`、`sixaxis/stacking.py`、`sixaxis/tracking.py`：Phase 2 任務產生器與量測（搬運/取放/疊棧碼垛、堆疊穩定判據、
  輸送帶追蹤抓取：α-β 估測 + CLIK 線上追蹤）。注意：報告腳本與 pytest 不可同時執行（都會重寫 `robots/ur5_gripper.urdf`）
* `sixaxis/welding.py`：Phase 2 連續路徑製程（T105 銲接）：銲槍工具 `robots/ur5_torch.urdf`、焊道 `Seam`/`turtle_seam`、work/travel 角姿態、
  等速時間律、`WeldCell`（自動選 roll、碰撞預檢、量測；工具 URDF/TCP 可由 `tool_writer`、`t_tcp` 替換）。其報告腳本與 pytest 同樣不可同時執行（重寫 `robots/ur5_torch.urdf`）
* `sixaxis/spraying.py`：T106 噴塗：直式噴槍 `robots/ur5_spray.urdf`、beta 分佈點源沉積模型 `SprayModel`（解析單道/多道、CV 與道距）、
  面板 `Panel`、光柵路徑 `raster_path`、固定姿態規劃 `plan_spray`、`SprayCell(WeldCell)`（扳機、膜厚積分、量測）。報告腳本與 pytest 不可同時執行
* `sixaxis/screwing.py`：T107 鎖螺絲：電動起子 `robots/ur5_driver.urdf`、`ThreadModel`（VDI 2230 扭力–夾緊力、柔度法接合剛性）、`Nutrunner`（主軸動力學 10 kHz 子步、
  二段式扭力/扭力–轉角策略、監控窗）、`bench_test`（無手臂接合模擬台）、`ScrewCell(WeldCell)`（浮動起子頭彈簧、入孔捕捉/螺旋搜尋、跟隨進給、
  反作用外力與前饋）。報告腳本與 pytest 不可同時執行（重寫 `robots/ur5_driver.urdf`）
* `sixaxis/finishing.py`：T108 去毛邊/拋光：研磨主軸 `robots/ur5_polisher.urdf`、工件高度場 `Surface`（形狀誤差/毛邊稜）、
  `PadModel`（Winkler 墊接觸、Coulomb 摩擦、Preston 移除）、`FinishCell(WeldCell)`（法向力導納外環 + 計算力矩內環、F/T 感測雜訊/低通、
  力前饋、位置控制基準）、理論 `loop_model`/`predict_touchdown`（含伺服柔度 Λ_n）/`routh_kcrit`/`pass_depth`。報告腳本與 pytest 不可同時執行
* `sixaxis/insertion.py`：T109 軸孔插入：力感測器＋RCC 工具 `robots/ur5_peg.urdf`、`Hole`（倒角孔有號距離、孔口稜線）、`PegBody`（RCC 6 自由度彈簧＋銷體剛體
  20 kHz 子步、罰函數接觸＋錨點切向摩擦可黏著）、`InsertCell(WeldCell)`（位置/軸向導納＋側向力歸零、螺旋搜孔、MISS/JAM/OVERFORCE 判定、`urdf_path` 可指定暫存 URDF 以平行執行）、
  準靜態理論 `capture_radius`/`chamfer_ratio`/`chamfer_forces`/`tilt_depth_limit`/`wedge_angle`/`two_point_statics`。報告腳本動畫段與 pytest 不可同時執行
* `sixaxis/planning.py`、`sixaxis/scenarios.py`：T110 避障規劃：`CollisionChecker`（私有暫存 URDF、障礙物 16 個一組複合體、自碰撞對、
  間隙 f = d − 餘裕）、保證式邊檢查 `edge_free`（Lipschitz 上界 D = Σ R_i|Δq_i|，f_a + f_b > D 即證明無碰撞，否則二分）、
  `RRTStar`（star=False 即 RRT；k-nearest、Informed 取樣、多目標 IK 解）、`shortcut`、`time_path`（逐段 rest-to-rest）；
  `scenarios.wall_scenario`/`random_scenario`/`add_to_sim`（T006 工作站＋額外方塊障礙物）。規劃部分可多行程平行（不改寫 robots/）
* `tests/`、`scripts/`（報告/動畫產生）、`docs/`（ROADMAP、PROGRESS、test-reports、media）

## 每次執行流程
0. **取得 repo 寫入權（務必最先做）**：用 ToolSearch 載入 `select:mcp__claude-code-remote__add_repo`，
   呼叫 `add_repo(owner=dofliu, repo=6axisRobot, access=push)`（即使 repo 已可讀/已 clone 也要呼叫，否則 push 會 403）。
   依回傳的路徑使用 clone（不要重複 clone）。只有載入並呼叫後仍失敗才停止，並回報完整錯誤、ToolSearch 查詢與 Dof 該做的事；
   不得改推其他 repo/分支。
   然後 `git fetch origin && git checkout main && git reset --hard origin/main`。已授權直接 commit/push 到 `main`（不開 PR）。
1. 閱讀本檔、`docs/ROADMAP.md`、`docs/PROGRESS.md` 最新一筆。
2. 選工作：優先接續 `[~]`，否則第一個相依皆完成的 `[ ]`；太大就拆子項，只做第一個。
3. 實作並寫測試：`ruff check .` 與 `python3 -m pytest` 全過才可推送。
4. 產生畫面動作：PyBullet 離屏多視角渲染 → `docs/media/<日期>_<TaskID>.gif`（≲8MB）與 `.mp4`，另存關鍵畫面 PNG、
   關節角/速度/力矩/誤差曲線（matplotlib）。
5. 測試報告 `docs/test-reports/<YYYY-MM-DD>_<TaskID>.md`（繁體中文：目的、模型與參數、可重現指令、指令序列、
   量化結果表、圖表與動畫、問題與限制、結論），更新 `docs/test-reports/INDEX.md`。數字須來自實際執行（可參考
   `scripts/t002_report.py` + `scripts/t002_write_md.py` 的「JSON → 報告」做法）。
6. 更新 ROADMAP 狀態（`[ ]`/`[~]`/`[x]`/`[!]`）與 `docs/PROGRESS.md`（日期、TaskID、內容、測試結果、下一步）。
7. commit（訊息 `T0xx: ...`）→ `git push origin main`（網路失敗 2s/4s/8s/16s 退避重試；403 回到第 0 步補 add_repo）。
8. 結束回報（繁體中文）：工作編號與標題、測試結果與關鍵數值、報告與動畫的 GitHub 連結
   （https://github.com/dofliu/6axisRobot/blob/main/...）、commit SHA、下次預計工作；之後呼叫 `/cloud-chat-log`，
   傳送「雲端記錄: TEAM DofLab」。Roadmap 全部完成則回報並停止。

## 硬性規則
* 一次只完成一項；絕不推送測試失敗的程式碼。無法完成時標 `[!]`，在 PROGRESS.md 記錄原因與嘗試，只推文件。
* 物理與數學模型須有依據（公式來源寫在 docs/ 或程式註解）；數值結果須來自實際執行，不得捏造。
* 不刪除既有測試報告與動畫。
* RL 訓練單次執行上限約 40 分鐘；未完成存 checkpoint 到 `models/`（單檔 < 50MB，必要時只存最佳）並下次接續。
