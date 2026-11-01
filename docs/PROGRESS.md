# PROGRESS（每次執行追加，最新在最下）

## 2026-11-01｜Phase 0 建置 + T001 + T002
- 建立專案骨架：CLAUDE.md、docs/ROADMAP.md、requirements.txt、pyproject.toml（ruff/pytest）。
- T001：`sixaxis/robot.py`（DH、FK、Hawkins 封閉解 IK 最多 8 解、LM 數值 IK、幾何 Jacobian、奇異/限位偵測、RNE 逆動力學、質量矩陣、正動力學、RK4 模擬、能量）；`sixaxis/urdf.py` 產生 URDF；`sixaxis/pbsim.py` PyBullet 介面。
- T002：23 項 pytest 全數通過；`ruff check` 通過。關鍵指標：FK vs PyBullet 位置 max 4.0e-8 m；IK 1000/1000 往返、殘差 7e-12；Jacobian 誤差 2e-10；逆動力學 vs PyBullet max 1.3e-5 N·m；自由擺動能量相對漂移 3.3e-7。
- 報告：docs/test-reports/2026-11-01_T002.md；動畫：docs/media/2026-11-01_T002.gif/.mp4。
- 發現並修正：URDF 慣性值在 numpy 2 下被寫成 `np.float64(...)` 導致 PyBullet 慣性為 0；PyBullet 自由擺動需放寬關節限位且不可含地面。
- 下一步：T003 軌跡規劃函式庫。
