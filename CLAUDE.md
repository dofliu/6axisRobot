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
