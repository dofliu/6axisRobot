"""避障路徑規劃（T110）：關節空間 RRT / RRT*、保證式（certified）連續碰撞檢查、捷徑後處理、時間參數化。

1. 碰撞模型
   * 規劃專用 PyBullet DIRECT client：載入手臂＋夾爪 URDF（指全開），障礙物合併為單一複合剛體
     （createCollisionShapeArray，含地板），每次查詢只需 1 次 getClosestPoints（基座連桿 −1 固定、忽略）。
   * 自碰撞：所有非相鄰、非剛性相連的連桿對（排除 base–link1–link2 肩部重疊、夾爪內部與 link6–夾爪）。
   * 間隙函數 f(q) = min(d_obs(q) − m_obs, d_self(q) − m_self)，d 為最近距離（getClosestPoints，GJK/EPA）；
     f > 0 即「保留安全餘裕的無碰撞」。超過搜尋距離 D_MAX 者以 D_MAX 計（下界，保守）。

2. 保證式邊檢查（Schwarzer, Saha & Latombe, "Adaptive dynamic collision checking for single and multiple
   articulated robots in complex environments", IEEE T-RO 21(3), 2005）
   * 關節 i 轉 Δq_i 時，下游任一點位移 ≤ R_i |Δq_i|，R_i = 下游幾何到關節 i 軸的距離上界。
     直線插值 q(λ) = q_a + λ(q_b − q_a) 上任一點位移 ≤ λ·D，D = Σ R_i |q_b,i − q_a,i|。
   * 因此每個距離函數（障礙物固定；自碰撞對的相對位移只受兩連桿間的關節影響，亦 ≤ D）沿邊為 D-Lipschitz：
     f(q(λ)) ≥ max(f_a − λD, f_b − (1−λ)D) ≥ (f_a + f_b − D)/2。
     ⇒ 若 f_a + f_b > D，整條邊無碰撞（不是取樣近似，而是證明）；否則二分並遞迴，
     子段長度 < δ_min 仍無法證明者判為碰撞（保守）。
   * R_i 由幾何求得：R_i = max_{k≥i} (Σ_{m=i+1..k} ‖p_m‖ + ρ_k)，‖p_m‖ 為相鄰連桿座標原點距離（DH 常數偏移），
     ρ_k 為連桿 k（link6 併入夾爪）之 AABB 角點到其座標原點最大距離（AABB 包含幾何 ⇒ 上界）。
   * 節點的 f 只算一次並快取，故短邊在 f_a + f_b > D 時零額外查詢。

3. RRT*（Karaman & Frazzoli, "Sampling-based algorithms for optimal motion planning", IJRR 30(7), 2011）
   * 成本 = 關節空間歐氏路徑長；steer 步長 η；目標偏置取樣；
   * 鄰域半徑 r_n = min(γ (log n / n)^{1/d}, η)，γ > γ* = 2(1 + 1/d)^{1/d} (μ(X_free)/ζ_d)^{1/d}，
     d = 6、ζ_6 = π³/6（單位球體積），μ(X_free) 以蒙地卡羅自由比例 × 取樣盒體積估計。
   * 選父（cost-to-come 最小且邊無碰撞）＋ rewire（經新節點較短者改接，並把成本差傳遞給子樹）。
   * 目標集合：目標 TCP 位姿的所有 IK 解（限取樣盒內且無碰撞）；任何一個連上即可，最終取成本最低者。
   * star=False 即為基本 RRT（LaValle 1998），供比較。

4. 後處理與時間律
   * 隨機捷徑（Geraerts & Overmars 2007）：隨機挑兩路徑點，若直線邊經保證式檢查無碰撞則刪去中間點。
   * 執行：逐段關節直線、五次多項式 rest-to-rest（T003 `joint_ptp`）——執行路徑與已檢查的邊完全一致，
     故「規劃無碰撞」可直接延伸到「執行無碰撞」（扣除控制追蹤誤差）。
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as pb

from .gripper import T_TCP, write_arm_with_gripper
from .robot import SixAxisRobot
from .trajectory import concat, joint_ptp, min_duration, sample_joint

D = 6
LOWER = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
UPPER = np.array([np.pi, 0.0, np.pi, np.pi, np.pi, np.pi])  # q2 ≤ 0：上臂在肩部水平面以上（地板側無用）
D_MAX = 0.50  # 距離查詢搜尋半徑 [m]
M_OBS, M_SELF = 0.010, 0.005  # 安全餘裕 [m]
CHUNK = 16  # 每個複合障礙物剛體的子形狀數
DELTA_MIN = 1e-3  # 二分最小子段（∞-範數，rad）
# 不檢查的自碰撞連桿對：相鄰連桿、base–link2（肩部外殼重疊）、夾爪內部、link6–夾爪（剛性相連）
_SKIP_SELF = {(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (-1, 1), (5, 7), (5, 8), (5, 9),
              (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)}


@dataclass(frozen=True)
class Box:
    """方塊障礙物（世界座標中心、半邊長；yaw = 繞世界 z 軸轉角，預設 0 即軸對齊）。"""

    center: tuple
    half: tuple
    name: str = "box"
    yaw: float = 0.0

    @property
    def quat(self) -> list:
        return [0.0, 0.0, float(np.sin(self.yaw / 2)), float(np.cos(self.yaw / 2))]


FLOOR = Box((0.0, 0.0, -0.05), (2.0, 2.0, 0.05), "floor")


def boxes_from_bodies(cid: int, bodies: dict) -> list[Box]:
    """由場景（例如 T006 Workcell.bodies）中單一方塊剛體讀出 Box（位置 + 碰撞形狀尺寸）。"""
    out = []
    for nm, b in bodies.items():
        pos = pb.getBasePositionAndOrientation(b, physicsClientId=cid)[0]
        for sd in pb.getCollisionShapeData(b, -1, physicsClientId=cid):
            if sd[2] != pb.GEOM_BOX:
                continue
            c = np.array(pos) + np.array(sd[5])
            out.append(Box(tuple(float(x) for x in c), tuple(float(x) / 2 for x in sd[3]), nm))
    return out


def _quat(R) -> list:
    """旋轉矩陣 → 四元數 [x, y, z, w]（Shepperd 法的穩定分支選擇）。"""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(R)).as_quat().tolist()


# ====================================================================== 碰撞檢查
class CollisionChecker:
    """規劃用碰撞/距離查詢（獨立 DIRECT client）。"""

    def __init__(self, boxes: list[Box], robot: SixAxisRobot | None = None, m_obs: float = M_OBS,
                 m_self: float = M_SELF, lower=LOWER, upper=UPPER, floor: bool = True):
        self.r = robot or SixAxisRobot()
        self.m_obs, self.m_self = m_obs, m_self
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.boxes = list(boxes)
        self.floor = None
        self.cid = pb.connect(pb.DIRECT)
        with tempfile.TemporaryDirectory() as td:  # 私有 URDF：可多行程平行、不改寫 robots/ 下共用檔
            path = write_arm_with_gripper(self.r, Path(td) / "arm_gripper.urdf")
            self.arm = pb.loadURDF(str(path), useFixedBase=True, physicsClientId=self.cid)
        self.n_links = pb.getNumJoints(self.arm, physicsClientId=self.cid)
        # 複合剛體（PyBullet 3.2.7 的 createCollisionShapeArray 實測只保留前 16 個子形狀 → 分批）
        self.static_obs = self._compound(self.boxes)
        self.movable: list[Box] = []
        self.movable_obs: list[int] = []
        self.obs = list(self.static_obs)
        self.held = None  # (body, T_tcp_obj 4×4)
        if floor:  # 地板獨立成一剛體：base 與 link1 只繞 z 轉，對地距離為常數（外殼貼地），不納入
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(FLOOR.half), physicsClientId=self.cid)
            self.floor = pb.createMultiBody(0, col, -1, list(FLOOR.center), physicsClientId=self.cid)
        links = range(-1, self.n_links)
        self.self_pairs = [(a, b) for a in links for b in links if a < b and (a, b) not in _SKIP_SELF
                           and pb.getCollisionShapeData(self.arm, a, physicsClientId=self.cid)
                           and pb.getCollisionShapeData(self.arm, b, physicsClientId=self.cid)]
        self.lever = self._lever_bounds()
        self.n_queries = 0

    def close(self):
        if pb.isConnected(self.cid):
            pb.disconnect(self.cid)

    def _compound(self, boxes) -> list[int]:
        out = []
        for i in range(0, len(boxes), CHUNK):
            bs = boxes[i:i + CHUNK]
            col = pb.createCollisionShapeArray(
                [pb.GEOM_BOX] * len(bs), halfExtents=[list(b.half) for b in bs],
                collisionFramePositions=[list(b.center) for b in bs],
                collisionFrameOrientations=[b.quat for b in bs], physicsClientId=self.cid)
            out.append(pb.createMultiBody(0, col, -1, [0, 0, 0], physicsClientId=self.cid))
        return out

    # --- 可變障礙物與夾持物（T111：工作台上其餘物件、夾持中的工件）
    def set_movable(self, boxes: list[Box]):
        """替換「可移動障礙物」集合（例如尚未揀選的物件）；靜態場景不變。"""
        for b in self.movable_obs:
            pb.removeBody(b, physicsClientId=self.cid)
        self.movable = list(boxes)
        self.movable_obs = self._compound(self.movable) if self.movable else []
        self.obs = list(self.static_obs) + self.movable_obs

    def attach(self, half, T_tcp_obj=None):
        """把夾持工件（方塊近似，半邊長 half，相對 TCP 的位姿 T_tcp_obj）併入手臂幾何。
        距離函數加入：工件–障礙物/地板、工件–手臂連桿 −1…4（link6 與夾爪剛性相連，不檢查）。
        Lipschitz 上界：工件屬於末端剛體，ρ_6 取 max(原值, ‖法蘭→工件 AABB 角點‖) → R_i 隨之放大（保守）。"""
        self.detach()
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[float(h) for h in half],
                                      physicsClientId=self.cid)
        body = pb.createMultiBody(0, col, -1, [0, 0, -5], physicsClientId=self.cid)
        T = np.eye(4) if T_tcp_obj is None else np.asarray(T_tcp_obj, float)
        T_fo = T_TCP @ T
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * np.asarray(half)
        rho_h = float(np.max(np.linalg.norm(corners @ T_fo[:3, :3].T + T_fo[:3, 3], axis=1)))
        self.held = (body, T_fo)
        self._lever_base = self.lever.copy()
        self.lever = self._lever_bounds(extra_rho6=rho_h)

    def detach(self):
        if self.held is not None:
            pb.removeBody(self.held[0], physicsClientId=self.cid)
            self.held = None
            self.lever = self._lever_base

    # --- 幾何上界 R_i
    def _lever_bounds(self, extra_rho6: float = 0.0) -> np.ndarray:
        self._set(np.zeros(6))
        rho = np.zeros(6)
        for k in range(self.n_links):
            o = np.array(pb.getLinkState(self.arm, min(k, 5), computeForwardKinematics=1,
                                         physicsClientId=self.cid)[4])
            lo, hi = (np.array(x) for x in pb.getAABB(self.arm, k, physicsClientId=self.cid))
            if not np.all(hi > lo):
                continue
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                for z in (lo[2], hi[2])])
            rho[min(k, 5)] = max(rho[min(k, 5)], float(np.max(np.linalg.norm(corners - o, axis=1))))
        rho[5] = max(rho[5], extra_rho6)
        off = np.array([np.linalg.norm(p) for p in self.r.pF[:6]])
        R = np.zeros(6)
        for i in range(6):
            R[i] = max(float(np.sum(off[i + 1:k + 1])) + rho[k] for k in range(i, 6))
        return R

    # --- 查詢
    def _set(self, q):
        for j in range(6):
            pb.resetJointState(self.arm, j, float(q[j]), physicsClientId=self.cid)
        if self.held is not None:
            T = self.r.fk(q) @ self.held[1]
            pb.resetBasePositionAndOrientation(self.held[0], T[:3, 3].tolist(), _quat(T[:3, :3]),
                                               physicsClientId=self.cid)

    def in_bounds(self, q) -> bool:
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))

    def distances(self, q) -> tuple[float, float]:
        """(手臂–障礙物最近距離, 自碰撞最近距離) [m]；超出 D_MAX 回傳 D_MAX。"""
        self._set(q)
        self.n_queries += 1
        d_obs = D_MAX
        for ob in self.obs:
            pts = pb.getClosestPoints(self.arm, ob, d_obs, physicsClientId=self.cid)
            d_obs = min([c[8] for c in pts if c[3] != -1] + [d_obs])
        if self.floor is not None:
            pts = pb.getClosestPoints(self.arm, self.floor, d_obs, physicsClientId=self.cid)
            d_obs = min([c[8] for c in pts if c[3] > 0] + [d_obs])
        if self.held is not None:
            hb = self.held[0]
            for ob in self.obs + ([self.floor] if self.floor is not None else []):
                pts = pb.getClosestPoints(hb, ob, d_obs, physicsClientId=self.cid)
                d_obs = min([c[8] for c in pts] + [d_obs])
        d_self = D_MAX
        if self.held is not None:
            for a in range(-1, 5):
                if not pb.getCollisionShapeData(self.arm, a, physicsClientId=self.cid):
                    continue
                p = pb.getClosestPoints(self.held[0], self.arm, d_self, linkIndexB=a,
                                        physicsClientId=self.cid)
                d_self = min([c[8] for c in p] + [d_self])
        for a, b in self.self_pairs:
            p = pb.getClosestPoints(self.arm, self.arm, d_self, linkIndexA=a, linkIndexB=b,
                                    physicsClientId=self.cid)
            if p:
                d_self = min(d_self, min(c[8] for c in p))
        return float(d_obs), float(d_self)

    def clearance(self, q) -> float:
        """f(q) = min(d_obs − m_obs, d_self − m_self)；取樣盒外回傳 −1。"""
        if not self.in_bounds(q):
            return -1.0
        d_obs, d_self = self.distances(q)
        return min(d_obs - self.m_obs, d_self - self.m_self)

    def free(self, q) -> bool:
        return self.clearance(q) > 0.0

    def motion_bound(self, qa, qb) -> float:
        """D = Σ R_i |Δq_i|：qa→qb 直線插值上任一點的最大位移上界 [m]。"""
        return float(self.lever @ np.abs(np.asarray(qb) - np.asarray(qa)))

    def edge_free(self, qa, qb, fa: float | None = None, fb: float | None = None) -> bool:
        """保證式直線邊檢查（見模組說明 2）。"""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        fa = self.clearance(qa) if fa is None else fa
        fb = self.clearance(qb) if fb is None else fb
        stack = [(qa, fa, qb, fb)]
        while stack:
            a, fa_, b, fb_ = stack.pop()
            if fa_ <= 0.0 or fb_ <= 0.0:
                return False
            if fa_ + fb_ > self.motion_bound(a, b):
                continue
            if np.max(np.abs(b - a)) < DELTA_MIN:
                return False
            m = 0.5 * (a + b)
            fm = self.clearance(m)
            stack.append((a, fa_, m, fm))
            stack.append((m, fm, b, fb_))
        return True

    def edge_dense(self, qa, qb, step: float = 0.005) -> float:
        """參考：等間隔（∞-範數 step）取樣的最小間隙（非保證，用於驗證保證式檢查）。"""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        n = max(int(np.ceil(np.max(np.abs(qb - qa)) / step)), 1)
        return min(self.clearance(qa + (qb - qa) * k / n) for k in range(n + 1))

    def free_fraction(self, n: int = 2000, rng=None) -> float:
        rng = rng or np.random.default_rng(0)
        return float(np.mean([self.free(rng.uniform(self.lower, self.upper)) for _ in range(n)]))


# ====================================================================== RRT / RRT*
def gamma_star(free_frac: float, lower=LOWER, upper=UPPER, d: int = D) -> float:
    """Karaman & Frazzoli (2011) Thm 38 的下限 γ*（PRM*/RRT* 漸近最佳所需）。"""
    vol = float(np.prod(np.asarray(upper) - np.asarray(lower))) * free_frac
    zeta = np.pi ** (d / 2) / _gamma_fn(d / 2 + 1)
    return 2.0 * (1 + 1 / d) ** (1 / d) * (vol / zeta) ** (1 / d)


def _gamma_fn(x):
    from math import gamma

    return gamma(x)


@dataclass
class PlanResult:
    success: bool
    path: list = field(default_factory=list)  # 關節路徑點（含起終點）
    cost: float = np.inf
    iters: int = 0
    nodes: int = 0
    t_plan: float = 0.0
    n_queries: int = 0
    t_first: float = np.nan  # 第一次找到解的時間 [s]
    iter_first: int = -1
    history: list = field(default_factory=list)  # (iter, best_cost)
    goal_index: int = -1
    tree: tuple | None = None  # (Q, parent) 供繪圖


class RRTStar:
    """neighbor="radius"：r_n = min(γ(log n/n)^{1/d}, η)；neighbor="knn"：k_n = ⌈k_c · e(1+1/d) · log n⌉
    （Karaman & Frazzoli 2011 之 k-nearest RRT*，k_RRT > e(1+1/d) 即漸近最佳；高維時半徑法在可行
    節點數下常無鄰居）。informed=True：已有解 c_best 後改在長橢球
    {q : ‖q − q_s‖ + ‖q − q_g‖ ≤ c_best} 內均勻取樣（Gammell, Srinivasa & Barfoot, IROS 2014），
    該集合恰為「能改善成本」的所有狀態，故不浪費樣本。"""

    def __init__(self, checker: CollisionChecker, eta: float = 1.0, goal_bias: float = 0.05,
                 gamma: float | None = None, star: bool = True, seed: int = 0, neighbor: str = "knn",
                 k_c: float = 1.0, informed: bool = True):
        self.cc = checker
        self.eta, self.goal_bias, self.star = eta, goal_bias, star
        self.gamma, self.neighbor, self.k_c, self.informed = gamma, neighbor, k_c, informed
        self.rng = np.random.default_rng(seed)

    def radius(self, n: int) -> float:
        if self.gamma is None:
            return self.eta
        return min(self.gamma * (np.log(n) / n) ** (1 / D), self.eta)

    def k_neighbors(self, n: int) -> int:
        return int(np.ceil(self.k_c * np.e * (1 + 1 / D) * np.log(max(n, 2))))

    def sample_informed(self, qs, qg, c_best: float, max_try: int = 200):
        """長橢球均勻取樣：x = C·L·u + center，u ~ 單位球均勻，L = diag(c/2, √(c²−c_min²)/2, …)。"""
        cc = self.cc
        c_min = float(np.linalg.norm(qg - qs))
        e1 = (qg - qs) / max(c_min, 1e-12)
        M = np.outer(e1, np.eye(D)[0])
        U, _, Vt = np.linalg.svd(M)
        C = U @ np.diag([1.0] * (D - 1) + [np.linalg.det(U) * np.linalg.det(Vt.T)]) @ Vt
        r2 = np.sqrt(max(c_best**2 - c_min**2, 0.0)) / 2
        L = np.diag([c_best / 2] + [r2] * (D - 1))
        ctr = (qs + qg) / 2
        for _ in range(max_try):
            u = self.rng.normal(size=D)
            u *= self.rng.random() ** (1 / D) / np.linalg.norm(u)
            x = C @ L @ u + ctr
            if cc.in_bounds(x):
                return x
        return self.rng.uniform(cc.lower, cc.upper)

    def plan(self, q_start, goals, max_iter: int = 2000, time_limit: float = 60.0,
             stop_at_first: bool = False) -> PlanResult:
        cc = self.cc
        t0, nq0 = time.perf_counter(), cc.n_queries
        q_start = np.asarray(q_start, float)
        goals = [np.asarray(g, float) for g in goals]
        f0 = cc.clearance(q_start)
        res = PlanResult(False)
        if f0 <= 0 or not goals:
            res.t_plan = time.perf_counter() - t0
            return res
        fg = [cc.clearance(g) for g in goals]
        cap = max_iter + len(goals) + 2
        Q = np.zeros((cap, D))
        F = np.zeros(cap)
        C = np.zeros(cap)
        P = -np.ones(cap, int)
        Q[0], F[0], n = q_start, f0, 1
        goal_node = [-1] * len(goals)
        children: dict[int, list] = {0: []}

        def add(q, f, parent, cost):
            nonlocal n
            Q[n], F[n], P[n], C[n] = q, f, parent, cost
            children[n] = []
            children[parent].append(n)
            n += 1
            return n - 1

        def best_goal():
            cs = [(C[g], i) for i, g in enumerate(goal_node) if g >= 0]
            return min(cs) if cs else (np.inf, -1)

        def propagate(i, delta):
            st = list(children[i])
            while st:
                k = st.pop()
                C[k] -= delta
                st.extend(children[k])

        it = 0
        for it in range(1, max_iter + 1):
            if time.perf_counter() - t0 > time_limit:
                break
            if stop_at_first and best_goal()[1] >= 0:
                break
            bc, bi = best_goal()
            if self.rng.random() < self.goal_bias:
                q_rand = goals[self.rng.integers(len(goals))]
            elif self.star and self.informed and bi >= 0:
                cand = [g for g in goals if np.linalg.norm(g - q_start) < bc]  # 仍可能改善的目標
                q_rand = self.sample_informed(q_start, cand[self.rng.integers(len(cand))], bc)
            else:
                q_rand = self.rng.uniform(cc.lower, cc.upper)
            dist = np.linalg.norm(Q[:n] - q_rand, axis=1)
            i_near = int(np.argmin(dist))
            d = dist[i_near]
            if d < 1e-9:
                continue
            q_new = Q[i_near] + (q_rand - Q[i_near]) * min(1.0, self.eta / d)
            f_new = cc.clearance(q_new)
            if f_new <= 0:
                continue
            # 選父
            if self.star:
                dn = np.linalg.norm(Q[:n] - q_new, axis=1)
                if self.neighbor == "knn":
                    near = np.argsort(dn)[:self.k_neighbors(n + 1)]
                else:
                    near = np.where(dn <= self.radius(n + 1))[0]
                if i_near not in near:
                    near = np.append(near, i_near)
                order = near[np.argsort(C[near] + dn[near])]
                parent = -1
                for k in order:
                    if cc.edge_free(Q[k], q_new, F[k], f_new):
                        parent = int(k)
                        break
                if parent < 0:
                    continue
                j = add(q_new, f_new, parent, C[parent] + dn[parent])
                # rewire
                for k in near:
                    if k == parent:
                        continue
                    c_alt = C[j] + dn[k]
                    if c_alt + 1e-12 < C[k] and cc.edge_free(q_new, Q[k], f_new, F[k]):
                        children[P[k]].remove(k)
                        children[j].append(k)
                        P[k] = j
                        delta = C[k] - c_alt
                        C[k] = c_alt
                        propagate(k, delta)
            else:
                if not cc.edge_free(Q[i_near], q_new, F[i_near], f_new):
                    continue
                j = add(q_new, f_new, i_near, C[i_near] + d * min(1.0, self.eta / d))
            # 嘗試連到目標
            for gi, g in enumerate(goals):
                dg = float(np.linalg.norm(g - Q[j]))
                if dg > self.eta or fg[gi] <= 0:
                    continue
                if goal_node[gi] < 0:
                    if cc.edge_free(Q[j], g, F[j], fg[gi]):
                        goal_node[gi] = add(g, fg[gi], j, C[j] + dg)
                elif (self.star and C[j] + dg + 1e-12 < C[goal_node[gi]]
                      and cc.edge_free(Q[j], g, F[j], fg[gi])):
                    k = goal_node[gi]
                    children[P[k]].remove(k)
                    children[j].append(k)
                    P[k] = j
                    delta = C[k] - (C[j] + dg)
                    C[k] = C[j] + dg
                    propagate(k, delta)
            bc, bi = best_goal()
            if bi >= 0 and np.isnan(res.t_first):
                res.t_first, res.iter_first = time.perf_counter() - t0, it
            if it % 25 == 0 or (bi >= 0 and (not res.history or res.history[-1][1] > bc + 1e-12)):
                res.history.append((it, float(bc)))
        bc, bi = best_goal()
        res.iters, res.nodes = it, n
        res.t_plan = time.perf_counter() - t0
        res.n_queries = cc.n_queries - nq0
        res.history.append((it, float(bc)))
        res.tree = (Q[:n].copy(), P[:n].copy())
        if bi >= 0:
            k = goal_node[bi]
            path = []
            while k >= 0:
                path.append(Q[k].copy())
                k = P[k]
            res.success, res.path, res.cost, res.goal_index = True, path[::-1], float(bc), bi
        return res


# ====================================================================== 後處理
def path_length(path) -> float:
    p = np.asarray(path)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def shortcut(cc: CollisionChecker, path, iters: int = 200, seed: int = 0) -> list:
    """隨機捷徑：保證式檢查可通過的直線邊取代中間路徑點。"""
    rng = np.random.default_rng(seed)
    p = [np.asarray(x, float) for x in path]
    for _ in range(iters):
        if len(p) <= 2:
            break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i < 2:
            continue
        if cc.edge_free(p[i], p[j]):
            p = p[:i + 1] + p[j:]
    return p


def path_min_clearance(cc: CollisionChecker, path, step: float = 0.01) -> float:
    """沿路徑（逐段直線）等間隔取樣的最小障礙物距離 d_obs [m]（量測用，不含餘裕）。"""
    out = D_MAX
    for a, b in zip(path[:-1], path[1:], strict=True):
        a, b = np.asarray(a), np.asarray(b)
        n = max(int(np.ceil(np.max(np.abs(b - a)) / step)), 1)
        for k in range(n + 1):
            out = min(out, cc.distances(a + (b - a) * k / n)[0])
    return float(out)


def time_path(robot: SixAxisRobot, path, vel_frac: float = 0.5, qdd_max: float = 4.0, dt: float = 0.01):
    """逐段五次多項式 rest-to-rest：各段 T = 滿足 vel_frac·qd_max 與 qdd_max 的最短時間。
    回傳 (各段 JointPTP, 串接後 Samples)。"""
    segs = []
    for a, b in zip(path[:-1], path[1:], strict=True):
        dq = np.asarray(b) - np.asarray(a)
        T = min_duration(dq, robot.qd_max * vel_frac, np.full(6, qdd_max), "quintic")
        segs.append(joint_ptp(a, b, T, "quintic"))
    smp = concat([sample_joint(robot, s, dt) for s in segs])
    return segs, smp


def goal_configs(cc: CollisionChecker, T_goal, q_ref=None) -> list[np.ndarray]:
    """目標位姿的所有 IK 解，限取樣盒內且無碰撞。"""
    out = []
    for q in cc.r.ik_all(T_goal, q_ref):
        q = np.asarray(q, float)
        if np.linalg.norm(cc.r.fk(q)[:3, 3] - np.asarray(T_goal)[:3, 3]) > 1e-6:
            continue
        if cc.free(q):
            out.append(q)
    return out


__all__ = ["Box", "CollisionChecker", "FLOOR", "PlanResult", "RRTStar", "boxes_from_bodies", "gamma_star",
           "goal_configs", "path_length", "path_min_clearance", "shortcut", "time_path"]
