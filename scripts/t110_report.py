"""T110 報告：避障路徑規劃（保證式碰撞檢查、RRT/RRT*、隨機場景基準、物理執行）→ JSON/圖/動畫。
用法：PYTHONPATH=. python3 scripts/t110_report.py 2026-11-19
（規劃實驗以 4 行程平行；每個 CollisionChecker 用私有暫存 URDF，不改寫 robots/ 下檔案；
 執行/動畫段用 GraspScene 會改寫 robots/ur5_gripper.urdf，勿與 pytest 同時執行。）
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from sixaxis import planning as P  # noqa: E402
from sixaxis import scenarios as SC  # noqa: E402
from sixaxis.graspsim import GraspScene  # noqa: E402
from sixaxis.gripper import CUBE, T_TCP  # noqa: E402
from sixaxis.robot import SixAxisRobot  # noqa: E402
from sixaxis.trajectory import check_samples  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T110"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
R = SixAxisRobot()
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def tcp(q):
    T = R.fk(q)
    return T[:3, 3] + T[:3, :3] @ T_TCP[:3, 3]


def summ(res: P.PlanResult, extra=None) -> dict:
    d = {"success": res.success, "cost": res.cost if res.success else None, "iters": res.iters, "nodes": res.nodes,
         "t_plan": res.t_plan, "n_queries": res.n_queries, "t_first": None if np.isnan(res.t_first) else res.t_first,
         "iter_first": res.iter_first, "history": res.history, "goal_index": res.goal_index,
         "path": [list(map(float, q)) for q in res.path]}
    return {**d, **(extra or {})}


# ====================================================================== 平行工作
def job_checker(_):
    sc = SC.wall_scenario()
    cc = sc.cc
    rng = np.random.default_rng(11)
    qs = [rng.uniform(cc.lower, cc.upper) for _ in range(500)]
    t = time.perf_counter()
    for q in qs:
        cc.distances(q)
    t_q = (time.perf_counter() - t) / len(qs)
    ff = cc.free_fraction(3000, np.random.default_rng(5))
    # Lipschitz：沿邊距離變化 / (λ D)
    ratios = []
    for _ in range(300):
        qa = rng.uniform(cc.lower, cc.upper)
        qb = qa + rng.normal(0, 0.2, 6)
        Dab = cc.motion_bound(qa, qb)
        da = np.array(cc.distances(qa))
        for lam in (0.25, 0.5, 0.75, 1.0):
            d = np.array(cc.distances(qa + lam * (qb - qa)))
            ok = (da < P.D_MAX - 1e-9) & (d < P.D_MAX - 1e-9)
            ratios += list(np.abs(d - da)[ok] / (lam * Dab))
    # 保證式 vs 密集取樣
    edges = []
    while len(edges) < 300:
        qa = sc.q_start + rng.normal(0, 0.8, 6)
        qb = qa + rng.normal(0, 0.35, 6)
        fa, fb = cc.clearance(qa), cc.clearance(qb)
        if fa <= 0 or fb <= 0:
            continue
        n0 = cc.n_queries
        cert = cc.edge_free(qa, qb, fa, fb)
        nq = cc.n_queries - n0
        n0 = cc.n_queries
        fine = cc.edge_dense(qa, qb, 0.002)
        nq_fine = cc.n_queries - n0
        coarse = cc.edge_dense(qa, qb, 0.05)
        edges.append({"cert": bool(cert), "fine_min": float(fine), "coarse_min": float(coarse), "nq_cert": nq,
                      "nq_fine": nq_fine, "len_inf": float(np.max(np.abs(qb - qa))),
                      "D": cc.motion_bound(qa, qb)})
    out = {"lever": cc.lever.tolist(), "n_self_pairs": len(cc.self_pairs), "n_boxes": len(cc.boxes),
           "t_query_ms": t_q * 1e3, "free_frac": ff, "gamma_star": P.gamma_star(ff),
           "lip_max": float(np.max(ratios)), "lip_p99": float(np.percentile(ratios, 99)),
           "lip_median": float(np.median(ratios)), "lip_n": len(ratios), "edges": edges,
           "box_volume": float(np.prod(cc.upper - cc.lower))}
    cc.close()
    return out


def job_wall(spec):
    kind, seed = spec
    sc = SC.wall_scenario()
    cc = sc.cc
    if kind == "rrt":
        res = P.RRTStar(cc, star=False, eta=0.5, seed=seed).plan(sc.q_start, sc.goals, max_iter=4000,
                                                                 time_limit=120, stop_at_first=True)
    elif kind == "first":
        res = P.RRTStar(cc, star=True, seed=seed).plan(sc.q_start, sc.goals, max_iter=4000, time_limit=120,
                                                       stop_at_first=True)
    else:
        res = P.RRTStar(cc, star=True, seed=seed, informed=(kind == "informed")).plan(
            sc.q_start, sc.goals, max_iter=1500, time_limit=400)
    extra = {"kind": kind, "seed": seed}
    if res.success:
        n0 = cc.n_queries
        t = time.perf_counter()
        s = P.shortcut(cc, res.path, 200, seed=seed)
        extra.update(sc_cost=P.path_length(s), sc_n=len(s), t_sc=time.perf_counter() - t, nq_sc=cc.n_queries - n0,
                     sc_path=[list(map(float, q)) for q in s], min_clear=P.path_min_clearance(cc, s, 0.01))
    if kind == "informed" and seed == 0:
        Q, Pp = res.tree
        extra["tree_tcp"] = [list(map(float, tcp(q))) for q in Q]
        extra["tree_parent"] = Pp.tolist()
    cc.close()
    return summ(res, extra)


def execute(sc_extra, path, render=None) -> dict:
    """GraspScene（T006 工作站＋額外障礙物）上以計算力矩控制執行逐段 PTP。"""
    segs, smp = P.time_path(R, path)
    chk = check_samples(R, smp)
    s = GraspScene(CUBE, workcell=True, render=render is not None)
    SC.add_to_sim(s.wc, sc_extra)
    s.sim.set_state(path[0])
    s._hold(path[0])
    s.wait(0.3)
    rec = {"t": [], "q_des": [], "q": [], "clear": [], "tcp": []}
    hits = set()
    t0 = s.t

    def cb(sc_):
        q = sc_.sim.get_state()[0]
        rec["t"].append(sc_.t - t0)
        rec["q_des"].append(sc_._q_cmd.copy())
        rec["q"].append(q)
        rec["clear"].append(sc_.wc.min_clearance())
        rec["tcp"].append(tcp(q))
        hits.update(sc_.wc.arm_collisions())
        if render is not None:
            render(sc_)

    for seg in segs:
        s.move_to(seg.q1, seg.duration, frame_cb=cb, cb_every=10)
    s.wait(0.3)
    cb(s)
    q_end = s.sim.get_state()[0]
    out = {"T_cycle": float(sum(g.duration for g in segs)), "n_seg": len(segs), "vel_ratio": chk["vel_ratio"],
           "torque_ratio": chk["torque_ratio"], "min_clear_exec": float(np.min(rec["clear"])),
           "hits": sorted(hits), "track_err_max": float(np.max(np.abs(np.array(rec["q"]) - np.array(rec["q_des"])))),
           "end_err_mm": float(np.linalg.norm(tcp(q_end) - tcp(path[-1])) * 1e3),
           "rec": {k: np.asarray(v).tolist() for k, v in rec.items()}}
    s.close()
    return out


def job_random(seed):
    sc = SC.random_scenario(seed, require_blocked=True)
    cc = sc.cc
    out = {"seed": seed, "name": sc.name, "n_goals": len(sc.goals), "n_posts": len(sc.extra),
           "start_free": cc.free(sc.q_start)}
    if not sc.goals or not out["start_free"]:
        cc.close()
        return out
    out["straight_free"] = any(cc.edge_free(sc.q_start, g) for g in sc.goals)
    out["straight_clear"] = max(P.path_min_clearance(cc, [sc.q_start, g], 0.01) for g in sc.goals)
    r1 = P.RRTStar(cc, star=False, eta=0.5, seed=seed).plan(sc.q_start, sc.goals, max_iter=4000, time_limit=90,
                                                           stop_at_first=True)
    r2 = P.RRTStar(cc, star=True, seed=seed).plan(sc.q_start, sc.goals, max_iter=500, time_limit=200)
    out["rrt"] = summ(r1)
    out["rrt"].pop("path")
    out["star"] = summ(r2)
    first = [h[1] for h in r2.history if np.isfinite(h[1])]
    out["star"]["cost_first"] = first[0] if first else None
    if r2.success:
        s = P.shortcut(cc, r2.path, 200, seed=seed)
        out["star"]["sc_cost"] = P.path_length(s)
        out["star"]["min_clear"] = P.path_min_clearance(cc, s, 0.01)
        out["sc_path"] = [list(map(float, q)) for q in s]
    cc.close()
    return out


# ====================================================================== 主程式
def main():
    M: dict = {"date": DATE, "params": {"D_MAX": P.D_MAX, "M_OBS": P.M_OBS, "M_SELF": P.M_SELF,
                                        "DELTA_MIN": P.DELTA_MIN, "lower": P.LOWER.tolist(), "upper": P.UPPER.tolist(),
                                        "wall": {"center": SC.WALL.center, "half": SC.WALL.half},
                                        "stations": SC.STATIONS}}
    wall_jobs = ([("informed", s) for s in range(4)] + [("uninformed", s) for s in range(4)]
                 + [("rrt", s) for s in range(6)] + [("first", s) for s in range(6)])
    with Pool(4) as pool:
        a_chk = pool.map_async(job_checker, [0])
        a_wall = pool.map_async(job_wall, wall_jobs, chunksize=1)
        a_rand = pool.map_async(job_random, list(range(16)), chunksize=1)
        M["checker"] = a_chk.get()[0]
        log("checker done")
        M["wall_runs"] = a_wall.get()
        log("wall done")
        M["random"] = a_rand.get()
        log("random done")
    for r in M["random"]:  # 物理執行（主行程依序；GraspScene 會改寫共用 URDF，不可平行）
        if "sc_path" in r:
            ex = execute(SC.random_scenario(r["seed"], require_blocked=True).extra, [np.array(q) for q in r.pop("sc_path")])
            ex.pop("rec")
            r["exec"] = ex
            log("exec rand", r["seed"], ex["min_clear_exec"], ex["hits"])
    sc = SC.wall_scenario()
    cc = sc.cc
    M["wall"] = {"q_start": sc.q_start.tolist(), "goals": [g.tolist() for g in sc.goals],
                 "start_clear": cc.clearance(sc.q_start),
                 "straight": [{"cert_free": cc.edge_free(sc.q_start, g),
                               "min_dobs": P.path_min_clearance(cc, [sc.q_start, g], 0.005)} for g in sc.goals]}
    # 最佳路徑（informed RRT* 捷徑後成本最低者）
    inf = [r for r in M["wall_runs"] if r["kind"] == "informed" and r["success"]]
    best = min(inf, key=lambda r: r["sc_cost"])
    M["wall"]["best_seed"] = best["seed"]
    path = [np.array(q) for q in best["sc_path"]]
    # 直線 PTP 第一個碰撞點（運動學）
    g0 = sc.goals[0]
    lam = np.linspace(0, 1, 401)
    dl = np.array([cc.distances(sc.q_start + x * (g0 - sc.q_start))[0] for x in lam])
    M["wall"]["straight_profile"] = {"lam": lam[::4].tolist(), "d_obs": dl[::4].tolist(),
                                     "first_hit": float(lam[np.argmax(dl <= 0)]) if np.any(dl <= 0) else None}
    cc.close()
    log("animating")
    M["exec"] = animate(sc, path, g0)
    log("exec", M["exec"]["min_clear_exec"], M["exec"]["hits"])
    plots(M, sc)
    rec = M["exec"].pop("rec")
    M["exec"]["rec_len"] = len(rec["t"])
    for r in M["wall_runs"]:
        r.pop("tree_tcp", None)
        r.pop("tree_parent", None)
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=1, ensure_ascii=False, default=float),
                                              encoding="utf-8")
    log("done")


def animate(sc, path, g_straight) -> dict:
    SIZE = (400, 300)
    frames: list = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    def lab(img, left, right, col=(150, 190, 255)):
        im_ = Image.fromarray(img)
        d_ = ImageDraw.Draw(im_)
        d_.rectangle([0, 0, im_.width, 24], fill=(20, 24, 32))
        d_.text((8, 4), left, fill=(235, 238, 242), font=font)
        d_.text((im_.width - 8 - d_.textlength(right, font=font), 4), right, fill=col, font=font)
        return np.asarray(im_)

    def views(sim):
        v0 = sim.render(target=(0.30, 0.20, 0.25), distance=1.35, yaw=-35, pitch=-28, size=SIZE, fov=45)
        v1 = sim.render(target=(0.45, 0.22, 0.25), distance=1.25, yaw=95, pitch=-12, size=SIZE, fov=45)
        return v0, v1

    # 1) 直線 PTP（運動學回放）→ 碰撞
    s = GraspScene(CUBE, workcell=True, render=True)
    SC.add_to_sim(s.wc, sc.extra)
    for x in np.linspace(0, 1, 15):
        s.sim.add_marker(tcp(sc.q_start + x * (g_straight - sc.q_start)), 0.006, (0.95, 0.2, 0.15, 1))
    n_hit = None
    for k, x in enumerate(np.linspace(0, 1, 41)):
        s.sim.set_state(sc.q_start + x * (g_straight - sc.q_start))
        hit = s.wc.arm_collisions()
        v0, v1 = views(s.sim)
        txt, col = ("COLLISION with " + ",".join(h.replace("obs_", "") for h in hit), (255, 110, 90)) if hit else (
            f"clearance {s.wc.min_clearance() * 1e3:5.0f} mm", (150, 190, 255))
        frames.append(np.hstack([lab(v0, "Straight joint PTP (no planning)", f"s = {x:.2f}"),
                                 lab(v1, "kinematic replay", txt, col)]))
        if hit:
            n_hit = k
            for _ in range(8):
                frames.append(frames[-1])
            break
    Image.fromarray(frames[-1]).save(MEDIA / f"{TAG}_straight_hit.png")
    s.close()
    n_straight = len(frames)

    # 2) RRT* 路徑執行（物理、計算力矩控制）
    segs, smp = P.time_path(R, path)
    state = {"last": -1.0}

    def rend(sc_):
        if sc_.t - state["last"] < 0.08 - 1e-9:
            return
        state["last"] = sc_.t
        v0, v1 = views(sc_.sim)
        c = sc_.wc.min_clearance() * 1e3
        frames.append(np.hstack([lab(v0, "Informed RRT* + shortcut (torque ctrl)", f"t = {sc_.t:.2f} s"),
                                 lab(v1, f"{len(path) - 1} segments, rest-to-rest", f"clearance {c:5.0f} mm")]))

    import sixaxis.graspsim as gs

    orig = gs.GraspScene.__init__

    def init_with_markers(self, *a, **kw):
        orig(self, *a, **kw)
        if kw.get("render"):
            for q in smp.q[::10]:
                self.sim.add_marker(tcp(q), 0.006, (0.2, 0.85, 0.35, 1))

    gs.GraspScene.__init__ = init_with_markers
    try:
        ex = execute(sc.extra, path, render=rend)
    finally:
        gs.GraspScene.__init__ = orig
    for _ in range(6):
        frames.append(frames[-1])
    imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=12, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((640, 240), Image.LANCZOS) for x in frames[::2]]
    pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=160, loop=0, optimize=True)
    nf = len(frames)
    for nm, i in (("key_a", n_straight + (nf - n_straight) // 3), ("key_b", n_straight + 2 * (nf - n_straight) // 3),
                  ("key_c", nf - 1)):
        Image.fromarray(frames[i]).save(MEDIA / f"{TAG}_{nm}.png")
    ex.update(n_frames=nf, n_straight_frames=n_straight, straight_hit_frame=n_hit,
              path=[list(map(float, q)) for q in path])
    return ex


def plots(M, sc):
    runs = M["wall_runs"]
    # 1. 收斂
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for kind, col in (("informed", "C0"), ("uninformed", "C1")):
        for r in [r for r in runs if r["kind"] == kind]:
            h = np.array([x for x in r["history"] if np.isfinite(x[1])])
            if len(h):
                ax[0].step(h[:, 0], h[:, 1], where="post", color=col, alpha=0.8,
                           label=f"RRT* {kind}" if r["seed"] == 0 else None)
    for kind, mk in (("rrt", "x"), ("first", "o")):
        rr = [r for r in runs if r["kind"] == kind and r["success"]]
        ax[0].scatter([r["iter_first"] for r in rr], [r["cost"] for r in rr], marker=mk, color="k" if kind == "rrt"
                      else "C2", zorder=5, label="RRT (eta 0.5) first sol." if kind == "rrt" else "RRT* first sol.")
    ax[0].set(xlabel="iteration", ylabel="path cost  [rad, joint-space length]", title="Wall scenario: cost convergence")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    e = M["checker"]["edges"]
    ax[1].hist([x["nq_cert"] for x in e], bins=30, alpha=0.7, label="certified (adaptive bisection)")
    ax[1].hist([x["nq_fine"] for x in e], bins=30, alpha=0.7, label="dense 2 mrad")
    ax[1].set(xscale="log", xlabel="distance queries per edge", ylabel="edges", title="Edge check cost")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_convergence.png", dpi=110)
    plt.close(fig)

    # 2. 俯視 / 側視：場景、樹、路徑
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    boxes = list(SC.workcell_boxes()) + sc.extra
    for b in boxes:
        c, h = b.center, b.half
        col = "#d98c26" if b.name == "wall" else "#9aa0aa"
        ax[0].add_patch(mp.Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1], fc=col, ec="k", lw=0.4,
                                     alpha=0.5))
        if b.name == "wall" or c[2] - h[2] < 0.2:
            ax[1].add_patch(mp.Rectangle((c[0] - h[0], c[2] - h[2]), 2 * h[0], 2 * h[2], fc=col, ec="k", lw=0.4,
                                         alpha=0.35))
    tr = next(r for r in runs if r["kind"] == "informed" and r["seed"] == 0)
    tt, pp = np.array(tr["tree_tcp"]), tr["tree_parent"]
    for i, p_ in enumerate(pp):
        if p_ >= 0:
            ax[0].plot(tt[[p_, i], 0], tt[[p_, i], 1], color="0.6", lw=0.3)
    rec = M["exec"]["rec"]
    et = np.array(rec["tcp"])
    q0, g0 = np.array(M["wall"]["q_start"]), np.array(M["wall"]["goals"][0])
    st = np.array([tcp(q0 + x * (g0 - q0)) for x in np.linspace(0, 1, 60)])
    for k, a in enumerate(ax):
        j = 1 if k == 0 else 2
        a.plot(st[:, 0], st[:, j], "r--", lw=1.5, label="straight joint PTP (collides)")
        a.plot(et[:, 0], et[:, j], "g-", lw=2, label="executed RRT* path (TCP)")
        a.plot(*np.array(SC.STATIONS["pick"])[[0, j]], "ks", ms=7)
        a.plot(*np.array(SC.STATIONS["belt"])[[0, j]], "k*", ms=11)
        a.set_aspect("equal")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="lower left")
    ax[0].set(xlim=(-0.3, 0.9), ylim=(-0.4, 0.7), xlabel="x [m]", ylabel="y [m]",
              title="Top view (grey: RRT* tree, TCP projection)")
    ax[1].set(xlim=(-0.3, 0.9), ylim=(0, 0.9), xlabel="x [m]", ylabel="z [m]", title="Side view")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_paths.png", dpi=110)
    plt.close(fig)

    # 3. 執行
    t = np.array(rec["t"])
    q, qd_ = np.array(rec["q"]), np.array(rec["q_des"])
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for j in range(6):
        ax[0, 0].plot(t, q[:, j], label=f"q{j + 1}")
    ax[0, 0].set(xlabel="t [s]", ylabel="joint angle [rad]", title="Joint angles (executed)")
    ax[0, 0].legend(fontsize=7, ncol=3)
    v = np.gradient(q, t, axis=0)
    for j in range(6):
        ax[0, 1].plot(t, v[:, j])
    ax[0, 1].set(xlabel="t [s]", ylabel="joint velocity [rad/s]", title="Joint velocities (rest at each waypoint)")
    ax[1, 0].semilogy(t, np.max(np.abs(q - qd_), axis=1) + 1e-12)
    ax[1, 0].set(xlabel="t [s]", ylabel="max |q - q_des| [rad]", title="Tracking error (computed torque)")
    ax[1, 1].plot(t, np.array(rec["clear"]) * 1e3)
    ax[1, 1].axhline(P.M_OBS * 1e3, color="r", ls="--", label="planning margin 10 mm")
    ax[1, 1].set(xlabel="t [s]", ylabel="min clearance arm-scene [mm]", title="Clearance during execution")
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_exec.png", dpi=110)
    plt.close(fig)

    # 4. 隨機場景
    rr = [r for r in M["random"] if "star" in r]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(len(rr))
    ax[0].bar(x - 0.2, [r["rrt"]["t_plan"] if r["rrt"]["success"] else 0 for r in rr], 0.4, label="RRT first sol.")
    ax[0].bar(x + 0.2, [r["star"]["t_first"] or 0 for r in rr], 0.4, label="RRT* first sol.")
    ax[0].set(xlabel="random scene seed", ylabel="time to first solution [s]", title="Random posts benchmark")
    ax[0].legend(fontsize=8)
    ax[1].bar(x - 0.27, [r["rrt"]["cost"] or 0 for r in rr], 0.27, label="RRT")
    ax[1].bar(x, [r["star"]["cost"] or 0 for r in rr], 0.27, label="RRT* 500 it")
    ax[1].bar(x + 0.27, [r["star"].get("sc_cost") or 0 for r in rr], 0.27, label="RRT* + shortcut")
    ax[1].set(xlabel="random scene seed", ylabel="path cost [rad]", title="Path cost")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x, [str(r["seed"]) for r in rr])
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_random.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
