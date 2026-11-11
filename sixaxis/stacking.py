"""T103 疊棧/碼垛：堆疊樣式產生器、逐層堆疊執行與量測、堆疊靜力穩定判據。

只組合 T007 指令（沿用 `tasks.transfer_program`）；每放一塊由「供料點」補一塊新方塊（模擬供料器）。
夾爪沿 y 閉合、工具姿態固定，故同層方塊只沿 x 排列（指墊不會伸進相鄰方塊之間）。

靜力穩定判據（`stack_margins`）：剛體方塊單柱堆疊，對每個介面 j（方塊 j−1 頂面與方塊 j 底面），
介面以上所有方塊的合質心之水平投影必須落在該介面的接觸區（兩方塊底面的交集，凸集）內，
否則重力對接觸區邊緣的力矩無法由（只能受壓的）法向接觸力平衡 → 傾倒。
等質量、每層等距錯位 d 的 n 層塔：介面上方有 m 塊時，合質心相對下方方塊中心偏 d·(m+1)/2，接觸區邊界在 a/2，
故需 d·(m+1)/2 < a/2；最嚴格為最底介面（m = n−1）→ d < a/n（a = 方塊邊長）。即經典的「積木懸伸」問題
（Paterson & Zwick, "Overhang", Amer. Math. Monthly 116(1), 2009；
靜力學零力矩條件見 Meriam & Kraige, Engineering Mechanics: Statics, ch. 3/5）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as pb

from .commands import Executor
from .gripper import CUBE
from .tasks import Z_CLEAR, Z_OBJ, _tilt_yaw, inhand_ok, transfer_program

A = CUBE.size[0]  # 方塊邊長 50 mm
H = CUBE.size[2]  # 層高
FEED_XY = (0.34, 0.0)  # 供料點（工作台上）
STACK_XY = (0.44, 0.0)  # 單柱疊棧位置
PYR_XY = (0.45, 0.0)  # 金字塔中心
PITCH = 0.052  # 同層方塊中心距（間隙 2 mm）
CLEAR_MARGIN = 0.07  # 搬運時 TCP 高於放置高度的最小餘裕
LAYER_RGBA = [(0.85, 0.25, 0.2, 1), (0.95, 0.6, 0.15, 1), (0.9, 0.82, 0.2, 1), (0.3, 0.65, 0.35, 1),
              (0.2, 0.5, 0.85, 1), (0.55, 0.35, 0.75, 1), (0.6, 0.6, 0.62, 1)]


@dataclass(frozen=True)
class Slot:
    x: float
    y: float
    layer: int  # 0 = 直接放在工作台上

    @property
    def z(self) -> float:
        return Z_OBJ + self.layer * H

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


def tower_slots(n: int, xy=STACK_XY, stagger: float = 0.0) -> list[Slot]:
    """單柱 n 層；stagger = 每層沿 +x 的錯位量（0 = 垂直疊棧）。"""
    return [Slot(xy[0] + k * stagger, xy[1], k) for k in range(n)]


def pyramid_slots(base: int = 3, center=PYR_XY, pitch: float = PITCH) -> list[Slot]:
    """沿 x 的金字塔碼垛：第 k 層 base−k 塊，跨在下層兩塊的接縫上（交錯壓縫）。"""
    return [Slot(center[0] + (i - (base - k - 1) / 2) * pitch, center[1], k)
            for k in range(base) for i in range(base - k)]


def column_slots(nx: int, layers: int, center=PYR_XY, pitch: float = PITCH) -> list[Slot]:
    """柱式碼垛：每層 nx 塊沿 x 排列、逐層對齊（不壓縫）。"""
    return [Slot(center[0] + (i - (nx - 1) / 2) * pitch, center[1], k)
            for k in range(layers) for i in range(nx)]


# ------------------------------------------------------------------ 靜力穩定判據（單柱）
def stack_margins(centers, a: float = A) -> np.ndarray:
    """單柱堆疊各介面的穩定餘裕 [m]（>0 穩定）。centers：(n,2) 各層中心 xy（由下而上，等質量、無偏航）。
    回傳長度 n−1：介面 j（方塊 j 與 j+1 之間）以上合質心到接觸區邊界的最小距離。"""
    c = np.atleast_2d(np.asarray(centers, float))
    out = []
    for j in range(len(c) - 1):
        com = c[j + 1:].mean(axis=0)
        lo = np.maximum(c[j], c[j + 1]) - a / 2
        hi = np.minimum(c[j], c[j + 1]) + a / 2
        out.append(float(min(np.min(com - lo), np.min(hi - com))))
    return np.array(out)


def stable_height(centers, a: float = A) -> int:
    """依序堆疊時，最多可堆到第幾層仍穩定（每加一層都重新檢查所有介面）。"""
    c = np.atleast_2d(np.asarray(centers, float))
    for k in range(2, len(c) + 1):
        if np.any(stack_margins(c[:k], a) <= 0.0):
            return k - 1
    return len(c)


# ------------------------------------------------------------------ 執行與量測
def stack_program(pick_xy, slot: Slot, dz_release: float = 0.0, **kw) -> list[dict]:
    """把供料點的方塊放到 slot：搬運高度隨層數升高（放置高度 + CLEAR_MARGIN，至少 Z_CLEAR）。"""
    z_clear = max(Z_CLEAR, slot.z + CLEAR_MARGIN)
    return transfer_program(pick_xy, slot.xy, z_clear=z_clear, z_place=slot.z + dz_release, **kw)


def _feed(scene, k: int, pick_xy, layer: int) -> int:
    """供料：第 0 塊用場景既有方塊，其餘在供料點新生成。"""
    cid = scene.cid
    pos = [pick_xy[0], pick_xy[1], Z_OBJ + 1e-3]
    if k == 0:
        body = scene.obj
        pb.resetBasePositionAndOrientation(body, pos, [0, 0, 0, 1], physicsClientId=cid)
        pb.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0], physicsClientId=cid)
    else:
        body = scene.g.spawn(CUBE, pos)
        if scene.wc:
            scene.wc.add_part(body)
    pb.changeVisualShape(body, -1, rgbaColor=LAYER_RGBA[layer % len(LAYER_RGBA)], physicsClientId=cid)
    scene.obj = scene.g.obj = body
    scene.wait(0.2)
    return body


def _pose(scene, body):
    pos, quat = pb.getBasePositionAndOrientation(body, physicsClientId=scene.cid)
    tilt, yaw = _tilt_yaw(quat)
    return np.array(pos), tilt, yaw


def cube_report(scene, body, slot: Slot, target_xy=None) -> dict:
    """方塊相對目標（預設 = slot 標稱）的位置誤差與傾角。"""
    pos, tilt, yaw = _pose(scene, body)
    tx, ty = target_xy if target_xy is not None else slot.xy
    return {"layer": slot.layer, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "dx_mm": float((pos[0] - tx) * 1e3), "dy_mm": float((pos[1] - ty) * 1e3),
            "xy_err_mm": float(np.hypot(pos[0] - tx, pos[1] - ty) * 1e3),
            "z_err_mm": float((pos[2] - slot.z) * 1e3), "tilt_deg": tilt, "yaw_deg": yaw}


def standing(rep: dict) -> bool:
    """方塊仍在其層位上（未滑落/傾倒）。"""
    return bool(abs(rep["z_err_mm"]) <= 5.0 and rep["tilt_deg"] <= 5.0 and rep["xy_err_mm"] <= 15.0)


def run_stack(scene, ex: Executor, slots: list[Slot], pick_xy=FEED_XY, offsets=None, dz_release: float = 0.0,
              check_inhand: bool = True, settle: float = 1.5, stop_on_collapse: bool = True, **kw) -> dict:
    """依序把方塊堆到 slots。offsets：(n,2) 各塊的額外放置偏移 [m]（放置誤差；量測以 slot+offset 為目標）。

    每塊放完並退回後檢查所有已放方塊是否仍在位（`standing`）；倒塌即停（`stop_on_collapse`）。
    回傳：每塊放置當下與最終（靜置 settle 秒後）的誤差、被後續放置擾動的位移、週期時間、力矩/速度使用率。
    """
    off = np.zeros((len(slots), 2)) if offsets is None else np.asarray(offsets, float)
    bodies, placed, cycles, results, err = [], [], [], [], None
    built = 0
    t_start = ex.t
    for k, slot in enumerate(slots):
        body = _feed(scene, k, pick_xy, slot.layer)
        ex.t = scene.t
        t0 = ex.t
        tgt = Slot(slot.x + off[k, 0], slot.y + off[k, 1], slot.layer)
        prog = stack_program(pick_xy, tgt, dz_release=dz_release, **kw)
        rep = ex.run(prog[:4])
        results += rep.results
        if rep.ok and check_inhand:
            good, _ = inhand_ok(scene, ex)
            if not good:
                err = f"INHAND_OFFSET@{k}"
        if rep.ok and err is None:
            rep = ex.run(prog[4:])
            results += rep.results
        if not rep.ok:
            err = f"{rep.error.code}@{k}:{rep.error.index}"
        bodies.append(body)
        if err:
            break
        cycles.append(ex.t - t0)
        placed.append(cube_report(scene, body, slot, tgt.xy))
        now = [cube_report(scene, b, s, (s.x + o[0], s.y + o[1]))
               for b, s, o in zip(bodies, slots, off, strict=False)]
        if not all(standing(r) for r in now):
            err = f"COLLAPSE@{k}"
            if stop_on_collapse:
                break
        else:
            built = k + 1
    scene.wait(settle)
    ex.t = scene.t
    final = [cube_report(scene, b, s, (s.x + o[0], s.y + o[1]))
             for b, s, o in zip(bodies, slots, off, strict=False)]
    if err is None and not all(standing(r) for r in final):
        err = "COLLAPSE@settle"
        built = int(np.argmin([standing(r) for r in final]))
    for p, f in zip(placed, final, strict=False):
        f["disturb_mm"] = float(np.linalg.norm([f["x"] - p["x"], f["y"] - p["y"], f["z"] - p["z"]]) * 1e3)
    mv = [r for r in results if r["cmd"].startswith("move") and r["status"] == "ok"]
    n = len(slots)
    ok = err is None and len(final) == n
    m = {
        "n": n, "built": built, "error": err, "placed": placed, "final": final, "cycles_s": cycles,
        "cycle_mean_s": float(np.mean(cycles)) if cycles else 0.0, "total_s": ex.t - t_start,
        "torque_util": max((r["torque_ratio"] for r in mv), default=0.0),
        "vel_util": max((r["vel_ratio"] for r in mv), default=0.0),
        "tcp_err_mm": max((r["target_err_mm"] for r in mv), default=0.0),
        "xy_max_mm": max((f["xy_err_mm"] for f in final), default=0.0),
        "z_abs_max_mm": max((abs(f["z_err_mm"]) for f in final), default=0.0),
        "tilt_max_deg": max((f["tilt_deg"] for f in final), default=0.0),
        "disturb_max_mm": max((f.get("disturb_mm", 0.0) for f in final), default=0.0),
        "results": results, "bodies": bodies,
    }
    m["success"] = bool(ok and m["xy_max_mm"] <= 2.0 and m["z_abs_max_mm"] <= 1.5
                        and m["tilt_max_deg"] <= 2.0)
    return m


__all__ = ["Slot", "tower_slots", "pyramid_slots", "column_slots", "stack_margins", "stable_height",
           "stack_program", "run_stack", "cube_report", "standing", "A", "H", "FEED_XY", "STACK_XY", "PYR_XY",
           "PITCH"]
