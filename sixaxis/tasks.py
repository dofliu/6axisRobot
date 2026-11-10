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
                     z_clear: float = Z_CLEAR, z_place: float = Z_OBJ) -> list[dict]:
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
        {"cmd": "move_l", "tcp": p(dx, dy, z_place), "id": "descend_dst", **kw},
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


# ------------------------------------------------------------------ T102 取放（pick & place）
Z_POCKET = Z_OBJ + S.TRAY_PLATE_T  # 方塊放進料盤格後的中心高度
POCKET_TOL = 0.004  # 格內單邊間隙 4 mm（POCKET 58 mm − 方塊 50 mm）


def pick_place_program(pick_xy, place_xy, z_place: float = Z_POCKET, **kw) -> list[dict]:
    """取放：桌面 pick_xy → 料盤格（或任意 place_xy，高度 z_place）。結構同 `transfer_program`。"""
    return transfer_program(pick_xy, place_xy, z_place=z_place, **kw)


def _tilt_yaw(quat) -> tuple[float, float]:
    """方塊傾角（z 軸與世界 z 的夾角）與偏航角 [deg]；偏航由料盤格約束，不計入傾角。"""
    R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    yaw = (yaw + 45.0) % 90.0 - 45.0  # 方塊四重對稱
    return float(tilt), float(abs(yaw))


def _place_metrics(scene, place_xy, z_place) -> dict:
    pos, quat = pb.getBasePositionAndOrientation(scene.obj, physicsClientId=scene.cid)
    pos = np.array(pos)
    tilt, yaw = _tilt_yaw(quat)
    return {
        "xy_err_mm": float(np.hypot(pos[0] - place_xy[0], pos[1] - place_xy[1]) * 1e3),
        "z_err_mm": float((pos[2] - z_place) * 1e3), "tilt_deg": tilt, "yaw_deg": yaw,
    }


def _reset_cube(scene, xy, z=Z_OBJ):
    pb.resetBasePositionAndOrientation(scene.obj, [xy[0], xy[1], z + 1e-3], [0, 0, 0, 1],
                                       physicsClientId=scene.cid)
    pb.resetBaseVelocity(scene.obj, [0, 0, 0], [0, 0, 0], physicsClientId=scene.cid)
    scene.g.obj = scene.obj
    scene.wait(0.2)


def _obj_state(scene):
    pos, quat = pb.getBasePositionAndOrientation(scene.obj, physicsClientId=scene.cid)
    return np.array(pos), _tilt_yaw(quat)[0]


def inhand_ok(scene, ex: Executor, tol_xy: float = 0.0015, tol_z: float = 0.003) -> tuple[bool, dict]:
    """抬升後檢查「方塊是否在夾爪內且置中」：方塊中心與 TCP 的偏差（模擬視覺/觸覺回饋）。
    y 向由兩指對中（夾爪閉合方向），x 向偏差不會被修正，放入料盤時會撞到側銷，故須檢查。"""
    pos, tilt = _obj_state(scene)
    tcp = ex.tcp_of(ex._cur_q())
    d = pos - tcp
    ok = bool(np.hypot(d[0], d[1]) <= tol_xy and abs(d[2]) <= tol_z and tilt <= 3.0)
    return ok, {"dx_mm": float(d[0] * 1e3), "dy_mm": float(d[1] * 1e3), "dz_mm": float(d[2] * 1e3),
                "tilt_deg": tilt}


def _recover_to_clear(scene, ex: Executor) -> bool:
    """鬆開夾爪並垂直退到安全高度（先在原地鬆開；若 TCP 較低則直接上升）。"""
    scene.g.open()
    scene.wait(0.3)
    ex.holding = False
    ex.t = scene.t
    xy = ex.tcp_of(ex._cur_q())[:2].tolist()
    return ex.run([{"cmd": "move_l", "tcp": [*xy, Z_CLEAR], "id": "recover_up"}]).ok


def run_pick_place(scene, ex: Executor, true_pick, place_xy, cmd_pick=None, z_place: float = Z_POCKET,
                   recover: bool = False, max_attempts: int = 3, **kw) -> dict:
    """方塊實際在 true_pick，指令以 cmd_pick（預設同 true_pick；差值 = 定位/感測誤差）去取，放到 place_xy。

    recover=False：整段程式一次執行（量測原始容忍度）。
    recover=True ：分「取」「放」兩段；取完抬升後以 `inhand_ok` 檢查，失敗（GRIP_EMPTY/COLLISION/偏置/脫落）
    → 若仍持有則原位放回 → 鬆開退到安全高度 → 讀回方塊實際位置（模擬重新感測）→ 重取，最多 max_attempts 次。
    這只是最簡單的失敗恢復；完整的主動判斷留給 T208。
    """
    _reset_cube(scene, true_pick)
    cmd = tuple(cmd_pick if cmd_pick is not None else true_pick)
    t0, log, rep = ex.t, [], None
    results: list = []
    err = None
    if not recover:
        rep = ex.run(pick_place_program(cmd, place_xy, z_place=z_place, **kw))
        results = rep.results
        err = None if rep.ok else f"{rep.error.code}@{rep.error.index}"
        attempts = 1
        log.append({"attempt": 1, "cmd_pick": list(cmd), "stage": "all", "error": err})
    else:
        attempts, picked = 0, False
        while attempts < max_attempts and not picked:
            attempts += 1
            rep = ex.run(pick_place_program(cmd, place_xy, z_place=z_place, **kw)[:4])
            results += rep.results
            chk = {}
            if rep.ok:
                good, chk = inhand_ok(scene, ex)
                err = None if good else "INHAND_OFFSET"
            else:
                err = f"{rep.error.code}@{rep.error.index}"
            log.append({"attempt": attempts, "cmd_pick": list(cmd), "stage": "pick", "error": err, **chk})
            if err is None:
                picked = True
                break
            if rep.ok:  # 仍持有但偏置：放回桌面再重取
                xy = ex.tcp_of(ex._cur_q())[:2].tolist()
                ex.run([{"cmd": "move_l", "tcp": [*xy, Z_OBJ], "id": "putback"}])
            if not _recover_to_clear(scene, ex):
                break
            pos, tilt = _obj_state(scene)
            if tilt > 10.0 or pos[2] > Z_OBJ + 0.01:  # 方塊被撞倒/飛離：超出簡單恢復能力
                err = "UNRECOVERABLE"
                break
            cmd = (float(pos[0]), float(pos[1]))  # 重新感測
        if picked:
            rep = ex.run(pick_place_program(cmd, place_xy, z_place=z_place, **kw)[4:])
            results += rep.results
            err = None if rep.ok else f"{rep.error.code}@{rep.error.index}"
            log.append({"attempt": attempts, "stage": "place", "error": err})
    m = _place_metrics(scene, place_xy, z_place)
    mv = [r for r in results if r["cmd"].startswith("move") and r["status"] == "ok"]
    m.update({
        "ok": err is None, "attempts": attempts, "log": log, "cycle_s": ex.t - t0, "error": err,
        "torque_util": max((r["torque_ratio"] for r in mv), default=0.0),
        "vel_util": max((r["vel_ratio"] for r in mv), default=0.0),
        "offset_mm": float(np.hypot(*np.subtract(cmd_pick, true_pick)) * 1e3) if cmd_pick else 0.0,
        "results": results,
    })
    m["success"] = bool(m["ok"] and m["xy_err_mm"] <= 3.0 and abs(m["z_err_mm"]) <= 1.5
                        and m["tilt_deg"] <= 2.0 and m["yaw_deg"] <= 5.0)
    return m


__all__ = ["transfer_program", "random_pairs", "run_transfer", "Z_OBJ", "Z_CLEAR", "Z_POCKET", "POCKET_TOL",
           "pick_place_program", "run_pick_place", "inhand_ok"]
