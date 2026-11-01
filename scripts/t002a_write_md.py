"""由 T002a metrics JSON 產生繁體中文報告。用法：python3 scripts/t002a_write_md.py YYYY-MM-DD "28 passed\""""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
date, passed = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "?")
tag = f"{date}_T002a"
m = json.loads((ROOT / "docs/test-reports/data" / f"{tag}_metrics.json").read_text(encoding="utf-8"))
g, sc = m["geometry"], m["self_clearance_m"]
clr = "；".join(f"{k}: {v['min'] * 1e3:.2f} mm（{v['pair']}）" for k, v in sc.items())

md = f"""# {date} T002a：手臂外觀擬真化（視覺/碰撞幾何與渲染）

## 1. 目的
T002 的動畫以單色基本圓柱呈現，外觀不像工業手臂。本項在**不改變任何運動學/動力學參數**的前提下，
把外觀改為 UR 系列造型，並提升渲染品質，作為之後所有動作測試與 RL 的展示基礎。

## 2. 模型與做法
* **外型依據（UR5 實機偏置）**：肩偏置 0.13585 m、肘偏置 −0.1197 m、腕 1 偏置 0.093 m，三者和即標準 DH 的 d4 = 0.10915 m。
  關節 2、3、4 軸互相平行，沿此軸平移連桿外型不改變運動學，因此上臂畫在 z2 = 0.13585 m、前臂畫在 z3 = 0.01615 m 的平面，
  和實機一樣側向錯開，DH 不變（`sixaxis/appearance.py` 檔頭有推導）。
* **關節外殼**：每個關節一個沿關節軸的圓角圓柱（倒角半徑 0.22r），兩條深灰密封環；藍色微凸端蓋**只畫在外露端**
  （`exposed_ends()`：端面中心在自身關節軸上，相對相鄰外殼的離軸距離與軸向座標不隨關節角改變，故判斷與姿態無關）。
  腕 2 外殼延伸包住腕 3 根部，腕 3 由側面伸出，與 UR 實機相同。
* **網格**：`sixaxis/meshgen.py` 以旋轉體（lathe）程式化產生 OBJ（48 等分、平滑法向；管件沿長度細分 24 段，
  消除 TinyRenderer 在長三角形上的光照帶狀瑕疵）。共 {g["mesh_files"]} 個網格、{g["mesh_triangles_total"]} 個三角形、
  {g["mesh_dir_MB"]:.1f} MB；視覺形狀 {g["visual_shapes"]} 個。
* **碰撞幾何**：保持簡單圓柱/方塊（{g["collision_shapes"]} 個），與外殼、臂管同尺寸同位置，計算快、之後避障可直接用。
* **質量/慣性不變**：PyBullet 以 `URDF_USE_INERTIA_FROM_FILE` 載入，慣性仍取自 `robot.py`；T002 的全部 FK/IK/動力學測試照常通過。
* **渲染**：TinyRenderer（無 GPU）＋光源方向隨鏡頭、陰影、環境/漫射/鏡面係數調整；2 倍超取樣後 Lanczos 縮小做反鋸齒；
  依深度緩衝把背景合成為廠房色漸層；地面改為程式產生的環氧地坪格線貼圖（`robots/textures/floor.png`）。

## 3. 重現步驟
```bash
python3 -m sixaxis.urdf                                   # 重新產生 URDF + meshes
ruff check . && python3 -m pytest -q                      # 本次：{passed}
PYTHONPATH=. python3 scripts/t002a_showcase.py {date}      # 動畫、關鍵畫面、前後對照、指標 JSON
python3 scripts/t002a_write_md.py {date} "{passed}"        # 產生本報告
```

## 4. 指令序列（展示動畫）
1. home (0, −π/2, π/2, −π/2, −π/2, 0) → demo (0, −1.2, 1.8, −2.1, −1.57, 0)，五次多項式時間律 3 s。
2. 平滑接到圓起點 1 s。
3. 工具朝下水平圓（圓心 (0.45, 0, 0.35) m、半徑 0.12 m）5 s，封閉解 IK 逐點選最近解。
左視角鏡頭全程環繞 360°，右視角為固定側視。

## 5. 量化結果
| 項目 | 結果 |
|---|---|
| 測試 | {passed}（新增 test_appearance：網格存在/一致、法向單位長、每連桿視覺細節、外露端判斷、標準姿態無自我碰撞） |
| 標準姿態非相鄰連桿最小間隙 | {clr} |
| 隨機 300 組關節角的自我碰撞比例 | {m["random_pose_self_collision_fraction"] * 100:.1f}%（UR 型手臂全關節 ±π 隨機取樣本來就常自撞；之後避障/RL 需納入） |
| 渲染時間 | 每視角 {m["render_ms_per_view"]:.0f} ms（2× 超取樣、含陰影，CPU） |
| 動畫 | {m["frames"]} 幀；GIF {m["gif_MB"]:.2f} MB（10 fps、640×240）、MP4 {m["mp4_MB"]:.2f} MB（20 fps、960×360） |

## 6. 圖表與動畫
前後對照（同一圓軌跡中點、同相機）：

![before-after](../media/{tag}_before_after.png)

![hero](../media/{tag}_hero.png)
![home](../media/{tag}_home.png)
![wrist](../media/{tag}_wrist_closeup.png)

動畫：

![動畫](../media/{tag}.gif)

MP4：[{tag}.mp4](../media/{tag}.mp4)

## 7. 問題與限制
* 外觀為程式化幾何近似，並非 UR 原廠 CAD；尺寸依公開偏置與目測比例。
* 質量/慣性仍為 T001 的「沿 DH 線段的均質圓柱」，**未**跟著外觀偏置移動質心（上臂實際質心應偏 0.136 m）。
  這是刻意保持 T002 驗證結果不變；若需更擬真的動力學，可在之後另開一項把慣性改為與外觀一致並重跑 T002。
* 底座與肩外殼間隙僅 0.16 mm（受 d1 與外殼半徑限制），屬外觀刻意貼合，不影響運動。
* TinyRenderer 為 CPU 軟體光柵，約 0.4 s/視角；長動畫需控制幀數。

## 8. 結論
手臂外觀已改為 UR 型（偏置臂、圓角外殼、端蓋、密封環、法蘭），並加入光影、陰影、反鋸齒與工廠地坪；
運動學/動力學不受影響（{passed}）。下一項：T003 軌跡規劃函式庫。
"""
(ROOT / "docs/test-reports" / f"{tag}.md").write_text(md, encoding="utf-8")
print("wrote", tag)
