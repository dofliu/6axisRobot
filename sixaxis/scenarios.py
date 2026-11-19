"""T110 避障規劃場景：T006 工作站 + 額外障礙物，起訖 TCP 與目標 IK 解集合。

* `wall_scenario`：工作台上一道隔板（y = 0.21、x ∈ [0.36, 0.90]、高 0.50 m），把「取料點上方」與
  「輸送帶擋料板上方」隔開。手臂本體在兩端姿態都不跨過隔板（UR 型肩/肘側向偏置使前臂在 x ≈ 0.25–0.35
  處穿越 y = 0.21，故隔板由 x = 0.36 起），但關節空間直線運動會掃過隔板 → 必須規劃。
* `random_scenario(seed)`：工作台/輸送帶區域內隨機 2–4 根立柱，起訖為隨機挑選的兩個工作站上方。
* `add_to_sim`：把額外障礙物以靜態方塊加入模擬中的 Workcell（碰撞群組 static，供執行時碰撞/間隙量測）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pybullet as pb

from . import scene as S
from .graspsim import Q_REF, tool_pose
from .gripper import CUBE
from .planning import Box, CollisionChecker, boxes_from_bodies, goal_configs

CUBE_BOX = Box((S.PICK_XY[0], S.PICK_XY[1], S.BENCH_TOP + CUBE.size[2] / 2),
               tuple(s / 2 for s in CUBE.size), "cube")
STATIONS = {  # 工作站上方 TCP（工具朝下）
    "pick": (S.PICK_XY[0], S.PICK_XY[1], 0.30),
    "belt": (S.BELT_PICK_XY[0], S.BELT_PICK_XY[1], 0.22),
    "fixture": (S.FIX_NOM[0], S.FIX_NOM[1], 0.24),
    "tray0": (*S.pocket_centers()[0], 0.24),
    "tray3": (*S.pocket_centers()[3], 0.24),
}


@lru_cache(maxsize=1)
def workcell_boxes() -> tuple:
    """T006 工作站全部靜態方塊 + 取料點上的方塊零件（於空白 client 建 Workcell，不載入/改寫手臂 URDF）。"""
    cid = pb.connect(pb.DIRECT)
    dummy = pb.createMultiBody(0, -1, -1, [0, 0, -5], physicsClientId=cid)
    b = boxes_from_bodies(cid, S.Workcell(cid, dummy).bodies)
    pb.disconnect(cid)
    return tuple(b) + (CUBE_BOX,)


@dataclass
class Scenario:
    name: str
    extra: list  # 額外障礙物（Box）
    tcp_start: tuple
    tcp_goal: tuple
    cc: CollisionChecker
    q_start: np.ndarray
    goals: list = field(default_factory=list)


def make_scenario(name: str, extra: list, tcp_start, tcp_goal, **cc_kw) -> Scenario:
    cc = CollisionChecker(list(workcell_boxes()) + list(extra), **cc_kw)
    sols = [q for q in cc.r.ik(tool_pose(tcp_start), Q_REF) if cc.free(q)]  # 最接近 Q_REF 的無碰撞解
    q0 = sols[0] if sols else cc.r.ik(tool_pose(tcp_start), Q_REF)[0]
    goals = goal_configs(cc, tool_pose(tcp_goal), q0)
    return Scenario(name, list(extra), tuple(tcp_start), tuple(tcp_goal), cc, q0, goals)


WALL = Box(((0.36 + 0.90) / 2, 0.21, 0.25), ((0.90 - 0.36) / 2, 0.01, 0.25), "wall")


def wall_scenario(**cc_kw) -> Scenario:
    return make_scenario("wall", [WALL], STATIONS["pick"], STATIONS["belt"], **cc_kw)


def random_scenario(seed: int, max_tries: int = 50, require_blocked: bool = False, **cc_kw) -> Scenario:
    """隨機 2–4 根立柱（截面 4–8 cm、高 0.25–0.55 m），中心在起訖兩工作站連線附近 ±0.12 m；
    立柱不得覆蓋起訖 TCP 正下方 8 cm 內。拒絕取樣：起點需有無碰撞 IK 解且目標至少一個無碰撞 IK 解
    （否則問題本身無解，不納入基準）；require_blocked=True 另要求「關節直線到所有目標解皆碰撞」（需規劃）。"""
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        sc = _random_once(rng, seed, **cc_kw)
        if sc.goals and sc.cc.free(sc.q_start) and not (
                require_blocked and any(sc.cc.edge_free(sc.q_start, g) for g in sc.goals)):
            return sc
        sc.cc.close()
    return sc


def _random_once(rng, seed, **cc_kw) -> Scenario:
    names = list(STATIONS)
    a, b = rng.choice(len(names), 2, replace=False)
    ps, pg = np.array(STATIONS[names[a]]), np.array(STATIONS[names[b]])
    extra = []
    for k in range(int(rng.integers(2, 5))):
        for _ in range(100):
            c = ps[:2] + rng.uniform(0.2, 0.8) * (pg[:2] - ps[:2]) + rng.uniform(-0.12, 0.12, 2)
            hx, hy = rng.uniform(0.02, 0.04, 2)
            if min(np.linalg.norm(c - ps[:2]), np.linalg.norm(c - pg[:2])) > 0.08 + max(hx, hy):
                break
        h = rng.uniform(0.25, 0.55)
        extra.append(Box((float(c[0]), float(c[1]), h / 2), (float(hx), float(hy), h / 2), f"post{k}"))
    return make_scenario(f"rand{seed}:{names[a]}->{names[b]}", extra, tuple(ps), tuple(pg), **cc_kw)


def add_to_sim(wc, boxes, rgba=(0.85, 0.55, 0.15, 1.0)) -> list[int]:
    """把額外障礙物加入 Workcell（靜態、static 群組；之後 arm_collisions/min_clearance 也會量到）。"""
    return [wc._box(f"obs_{b.name}", b.center, b.half, rgba) for b in boxes]


__all__ = ["CUBE_BOX", "STATIONS", "Scenario", "WALL", "add_to_sim", "make_scenario", "random_scenario",
           "wall_scenario", "workcell_boxes"]
