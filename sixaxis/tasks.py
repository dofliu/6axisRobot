"""Phase 2 工廠動作的共用工具：任務程式產生器與量測（T101 點到點搬運起）。

只組合 T007 指令（move_j / move_l / grip / release）；任務層不碰運動學或控制細節。
"""

from __future__ import annotations

import numpy as np
import pybullet as pb

from . import scene as S
from .commands import Executor
from .gripper import CUBE

Z_OBJ = S.BENCH_TOP + CUBE.size[2] / 2  # 方塊放在工作台上時的中心（= 抓取 TCP 高度）
Z_CLEAR = 0.25  # 搬運高度（TCP）
# 工作台上無干涉的取放區（避開料盤 y<-0.08、治具 y>0.09、手指開口方向）
ZONE_X, ZONE_Y = (0.36, 0.44), (-0.01, 0.02)
MIN_SEP = 0.03  # 起訖點最小距離（避免退化的短搬運）


def transfer_program(src, dst, force: float = 40.0, speed: float | None = None, profile: str = "quintic",
                     z_clear: float = Z_CLEAR) -> list[dict]:
    """點到點搬運：上方定位(PTP) → 直線下降 → 夾 → 直線抬升 → PTP 搬運至目的地上方 → 直線下降 → 放 → 抬升。"""
    kw = {} if speed is None else {"speed": speed}
    kw["profile"] = profile
    (sx, sy), (dx, dy) = src, dst
    p = lambda x, y, z: [float(x), float(y), float(z)]  # noqa: E731
    return [
        {"cmd": "move_j", "tcp": p(sx, sy, z_clear), "id": "above_src", **kw},
        {"cmd": "move_l", "tcp": p(sx, sy, Z_OBJ), "id": "descend_src", **kw},
        {"cmd": "grip", "force": force, "id": "grip"},
        {"cmd": "move_l", "tcp": p(sx, sy, z_clear), "id": "lift", **kw},
        {"cmd": "move_j", "tcp": p(dx, dy, z_clear), "id": "transport", **kw},
        {"cmd": "move_l", "tcp": p(dx, dy, Z_OBJ), "id": "descend_dst", **kw},
        {"cmd": "release", "id": "release"},
        {"cmd": "move_l", "tcp": p(dx, dy, z_clear), "id": "retreat", **kw},
    ]


def random_pairs(n: int, seed: int = 0) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """取放區內隨機起訖點（距離 ≥ MIN_SEP），固定種子可重現。"""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        a = (rng.uniform(*ZONE_X), rng.uniform(*ZONE_Y))
        b = (rng.uniform(*ZONE_X), rng.uniform(*ZONE_Y))
        if np.hypot(a[0] - b[0], a[1] - b[1]) >= MIN_SEP:
            out.append((a, b))
    return out


def run_transfer(scene, ex: Executor, src, dst, **kw) -> dict:
    """在既有場景把方塊從 src 搬到 dst，回傳量測。方塊先被重置到 src（水平、無速度）。"""
    obj, cid = scene.obj, scene.cid
    pb.resetBasePositionAndOrientation(obj, [src[0], src[1], Z_OBJ + 1e-3], [0, 0, 0, 1], physicsClientId=cid)
    pb.resetBaseVelocity(obj, [0, 0, 0], [0, 0, 0], physicsClientId=cid)
    scene.g.obj = obj
    scene.wait(0.2)  # 方塊落定
    t0 = ex.t
    rep = ex.run(transfer_program(src, dst, **kw))
    pos, quat = pb.getBasePositionAndOrientation(obj, physicsClientId=cid)
    pos = np.array(pos)
    tilt = float(np.degrees(2 * np.arccos(min(1.0, abs(quat[3])))))
    mv = [r for r in rep.results if r["cmd"].startswith("move") and r["status"] == "ok"]
    m = {
        "ok": rep.ok, "error": None if rep.ok else f"{rep.error.code}@{rep.error.index}",
        "cycle_s": rep.t_total - t0,
        "xy_err_mm": float(np.hypot(pos[0] - dst[0], pos[1] - dst[1]) * 1e3),
        "z_err_mm": float((pos[2] - Z_OBJ) * 1e3), "tilt_deg": tilt,
        "vel_util": max((r["vel_ratio"] for r in mv), default=0.0),
        "torque_util": max((r["torque_ratio"] for r in mv), default=0.0),
        "tcp_err_mm": max((r["target_err_mm"] for r in mv), default=0.0),
        "dist_mm": float(np.hypot(dst[0] - src[0], dst[1] - src[1]) * 1e3),
        "results": rep.results,
    }
    m["success"] = bool(m["ok"] and m["xy_err_mm"] <= 2.0 and abs(m["z_err_mm"]) <= 1.0 and tilt <= 2.0)
    return m


__all__ = ["transfer_program", "random_pairs", "run_transfer", "Z_OBJ", "Z_CLEAR"]
