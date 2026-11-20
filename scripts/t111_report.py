"""T111 報告：多物件分類揀選（感知、序列、物理執行、避障、消融）→ JSON/圖/動畫。
用法：PYTHONPATH=. python3 scripts/t111_report.py 2026-11-20 [--anim]
（實驗以 4 行程平行；每個 SortCell 用私有暫存 URDF，不改寫 robots/ 下檔案，可與其他工作並行。
 --anim：只重做動畫與曲線圖（讀既有 JSON）。）
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

from sixaxis import scene as S  # noqa: E402
from sixaxis import sorting as SO  # noqa: E402
from sixaxis.planning import Box  # noqa: E402
from sixaxis.robot import wrap_pi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = next((a for a in sys.argv[1:] if not a.startswith("--")), time.strftime("%Y-%m-%d"))
TAG = f"{DATE}_T111"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
T0 = time.time()
POST = Box((0.60, -0.20, 0.15), (0.015, 0.04, 0.15), "post")  # 料盤右側光柵立柱（高 0.30 m），擋在取料區 → B 箱之間
N_PERC, N_SEQ, N_PHYS, N_POST, N_DENSE = 40, 30, 12, 8, 10
DENSE_GAP = 0.004
COLORS = {"red": "#d43c2c", "blue": "#3366cc", "green": "#2f9e44"}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def slim(r: dict) -> dict:
    """去掉 numpy 型別與大型欄位，方便 JSON。"""
    def cv(x):
        if isinstance(x, dict):
            return {k: cv(v) for k, v in x.items() if k not in ("results",)}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    return cv(r)


# ====================================================================== 平行工作
def job_perception(seed):
    """靜態場景：放好物件、手臂在觀測姿態，拍一次，比對真值。"""
    parts = SO.scatter(1000 + seed)
    c = SO.SortCell(parts)
    t = time.perf_counter()
    info: dict = {}
    dets = c.cam.detect(c.s.cid, c.s.sim.id, out=info)
    t_det = time.perf_counter() - t
    idx = SO.match(dets, parts)
    items = []
    for d, j in zip(dets, idx, strict=True):
        if j < 0:
            items.append({"cls": d.cls, "matched": False})
            continue
        p = parts[j]
        sp = p.cls.spec
        items.append({"cls": d.cls, "true": p.cls.name, "matched": True, "consistent": d.consistent,
                      "pos_err_mm": float(np.hypot(*(d.xy - np.array(p.xy))) * 1e3),
                      "yaw_err_deg": SO.yaw_error(d, p), "h_err_mm": (d.height - sp.height) * 1e3,
                      "w_mm": d.w * 1e3, "l_mm": d.l * 1e3, "round": d.roundness, "n_px": d.n_px,
                      "chroma": d.chroma.tolist()})
    out = {"seed": seed, "n_true": len(parts), "n_det": len(dets), "n_matched": int(sum(j >= 0 for j in idx)),
           "t_detect": t_det, "px_mm": c.cam.px_size * 1e3, "items": items}
    c.close()
    return out


def job_sequence(seed):
    """序列比較（運動學估計）：同一場景以三種策略排序。"""
    parts = SO.scatter(2000 + seed)
    dets = [_gt_det(p) for p in parts]
    q0 = np.asarray(wrap_pi(SO.home_q()), float)
    out = {"seed": seed, "blocked": [not SO.blockers(dets, i) for i in range(len(dets))]}
    for m in ("scan", "greedy", "optimal"):
        t = time.perf_counter()
        order, est = SO.plan_sequence(dets, q0, m)
        out[m] = {"order": order, "est": est, "t": time.perf_counter() - t}
    return out


def job_sequence_dense(seed):
    parts = SO.scatter(3000 + seed, gap=DENSE_GAP)
    dets = [_gt_det(p) for p in parts]
    q0 = np.asarray(wrap_pi(SO.home_q()), float)
    out = {"seed": seed, "blocked": [not SO.blockers(dets, i) for i in range(len(dets))],
           "blocked_nocheck": [not SO.blockers(dets, i, check=False) for i in range(len(dets))]}
    for m in ("scan", "greedy", "optimal"):
        order, est = SO.plan_sequence(dets, q0, m)
        out[m] = {"order": order, "est": est}
    return out


def _gt_det(p):
    sp = p.cls.spec
    w, l = (sp.size[0], sp.size[1]) if sp.shape == "box" else (2 * sp.size[0],) * 2
    return SO.Detection(p.cls.name, np.array(p.xy), p.yaw, w, l, sp.height, 1.0 if sp.shape == "box" else 0.785,
                        0, p.cls.chroma)


def job_phys(spec):
    """物理執行一個場景。spec = (組別, seed, SortConfig kwargs, extra, gap)。"""
    group, seed, kw, extra, gap = spec
    parts = SO.scatter(seed, gap=gap)
    c = SO.SortCell(parts, SO.SortConfig(**kw), extra=[POST] if extra else [])
    t = time.perf_counter()
    r = c.run()
    r["wall_s"] = time.perf_counter() - t
    r["motions"] = [{k: v for k, v in m.items() if k != "results"} for m in c.motion_log]
    r["group"], r["seed"] = group, seed
    # 每次運動的力矩/速度使用率與時間
    r["moves"] = [{"tag": m["tag"], "dur": sum(x.get("duration", 0.0) for x in m.get("results", [])),
                   "torque": max([x.get("torque_ratio", 0.0) for x in m.get("results", [])] + [0.0])}
                  for m in c.motion_log]
    c.close()
    return slim(r)


# ====================================================================== 主程式
def main():
    M: dict = {"params": {
        "zone": SO.ZONE, "gap": SO.GAP, "dense_gap": DENSE_GAP, "z_clear": SO.Z_CLEAR, "bin_lift": SO.BIN_LIFT,
        "bin_c": SO.BIN_C, "bin_in": [SO.BIN_IN_X, SO.BIN_IN_Y], "bin_slot": SO.BIN_SLOT, "bin_wall_top": SO.BIN_WALL_TOP, "grip_margin": SO.GRIP_MARGIN,
        "open_w": SO.OPEN_W, "cam_eye": SO.CAM_EYE, "cam_fov": 21.0, "cam_px": 480, "round_t": SO.ROUND_T,
        "occl_px": SO.OCCL_PX, "post": [POST.center, POST.half], "t_fixed": SO.T_FIXED,
        "classes": {k: {"spec": v.spec.name, "size": v.spec.size, "mass": v.spec.mass, "dest": v.dest,
                        "rgba": v.spec.rgba} for k, v in SO.CLASSES.items()},
        "n": {"perc": N_PERC, "seq": N_SEQ, "phys": N_PHYS, "post": N_POST, "dense": N_DENSE}}}
    with Pool(4) as pool:
        log("perception")
        M["perception"] = pool.map(job_perception, range(N_PERC))
        log("sequence")
        M["sequence"] = pool.map(job_sequence, range(N_SEQ))
        M["sequence_dense"] = pool.map(job_sequence_dense, range(N_DENSE))
        specs = []
        for sd in range(N_PHYS):
            for m in ("optimal", "greedy", "scan"):
                specs.append((f"seq_{m}", sd, {"sequence": m}, False, SO.GAP))
            specs.append(("fixed_yaw", sd, {"fixed_yaw": True}, False, SO.GAP))
        for sd in range(N_POST):
            specs.append(("post_plan", sd, {}, True, SO.GAP))
            specs.append(("post_noplan", sd, {"plan": False}, True, SO.GAP))
        for sd in range(N_DENSE):
            specs.append(("dense_full", 3000 + sd, {}, False, DENSE_GAP))
            specs.append(("dense_nocheck", 3000 + sd, {"grasp_check": False}, False, DENSE_GAP))
        log(f"physical runs: {len(specs)}")
        res = pool.map(job_phys, specs, chunksize=1)
    M["phys"] = res
    log("physical done")
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(slim(M), ensure_ascii=False, indent=1), encoding="utf-8")
    plots(M)
    M["anim"] = animate()
    (DATA / f"{TAG}_metrics.json").write_text(json.dumps(slim(M), ensure_ascii=False, indent=1), encoding="utf-8")
    log("done")


# ====================================================================== 圖
def plots(M):
    P = M["perception"]
    it = [i for p in P for i in p["items"] if i["matched"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for c in SO.CLASSES:
        e = [i["pos_err_mm"] for i in it if i["true"] == c]
        ax[0].hist(e, bins=20, alpha=0.6, color=COLORS[c], label=f"{c} (n={len(e)})")
        y = [i["yaw_err_deg"] for i in it if i["true"] == c and i["yaw_err_deg"] == i["yaw_err_deg"]]
        if y:
            ax[1].hist(y, bins=20, alpha=0.6, color=COLORS[c], label=c)
        ax[2].scatter([i["w_mm"] for i in it if i["true"] == c], [i["round"] for i in it if i["true"] == c], s=10,
                      color=COLORS[c], label=c)
    ax[2].axhline(SO.ROUND_T, color="k", ls="--", lw=0.8)
    ax[0].set(xlabel="position error [mm]", ylabel="count", title="Centroid error (top face)")
    ax[1].set(xlabel="yaw error [deg] (mod symmetry)", title="Orientation error")
    ax[2].set(xlabel="min-area-rect side w [mm]", ylabel="hull / rect area", title="Shape feature (round < 0.90)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_perception.png", dpi=120)
    plt.close(fig)

    # 序列
    Sq = M["sequence"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    est = {m: np.array([s[m]["est"] for s in Sq]) for m in ("scan", "greedy", "optimal")}
    ax[0].boxplot([est[m] for m in est], tick_labels=list(est))
    ax[0].set(ylabel="estimated cycle [s]", title=f"Sequencing (kinematic estimate, {len(Sq)} scenes)")
    ph = [r for r in M["phys"] if r["group"].startswith("seq_")]
    for k, m in enumerate(("optimal", "greedy", "scan")):
        rs = sorted([r for r in ph if r["group"] == f"seq_{m}"], key=lambda r: r["seed"])
        ax[1].plot([r["seed"] for r in rs], [r["cycle_s"] for r in rs], "o-", label=m, alpha=0.8)
    ax[1].set(xlabel="scene seed", ylabel="simulated cycle [s]", title="Physical execution (6 parts)")
    for a in ax:
        a.grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_sequence.png", dpi=120)
    plt.close(fig)

    # 消融
    groups = ["seq_optimal", "fixed_yaw", "post_plan", "post_noplan", "dense_full", "dense_nocheck"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for k, g in enumerate(groups):
        rs = [r for r in M["phys"] if r["group"] == g]
        n = sum(r["n_parts"] for r in rs)
        c = sum(r["n_correct"] for r in rs)
        ax.bar(k, c / max(n, 1) * 100, color="#4c78a8" if "no" not in g and g != "fixed_yaw" else "#e45756")
        ax.text(k, c / max(n, 1) * 100 + 1, f"{c}/{n}", ha="center", fontsize=9)
    ax.set_xticks(range(len(groups)), groups, rotation=15)
    ax.set(ylabel="parts sorted correctly [%]", ylim=(0, 110), title="Ablations (physical runs)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_ablation.png", dpi=120)
    plt.close(fig)


# ====================================================================== 動畫（障礙物場景 seed 0）
def animate() -> dict:
    SIZE = (400, 300)
    frames: list = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    rec = {"t": [], "q": [], "qd": [], "tau": [], "err": []}
    state = {"last": -1.0, "cell": None, "phase": ""}

    def lab(img, left, right, col=(150, 190, 255)):
        im_ = Image.fromarray(img)
        d_ = ImageDraw.Draw(im_)
        d_.rectangle([0, 0, im_.width, 24], fill=(20, 24, 32))
        d_.text((8, 4), left, fill=(235, 238, 242), font=font)
        d_.text((im_.width - 8 - d_.textlength(right, font=font), 4), right, fill=col, font=font)
        return np.asarray(im_)

    def cam_panel(cell):
        lv = cell.last_view if cell is not None else {}
        if not lv:
            return np.full((300, 300, 3), 30, np.uint8)
        im = Image.fromarray(lv["rgb"].astype(np.uint8)).convert("RGB")
        d_ = ImageDraw.Draw(im)
        cam = cell.cam
        PV = np.array(cam.proj).reshape(4, 4).T @ np.array(cam.view).reshape(4, 4).T
        n = cam.size

        def px(p):
            x = PV @ np.array([p[0], p[1], p[2], 1.0])
            x = x[:3] / x[3]
            return ((x[0] + 1) / 2 * n, (1 - x[1]) / 2 * n)

        for d in lv["dets"]:
            z = S.BENCH_TOP + d.height
            pts = [px((*c, z)) for c in d.poly]
            col = {"red": (255, 80, 60), "blue": (80, 140, 255), "green": (60, 220, 90)}[d.cls]
            d_.polygon(pts, outline=col, width=3)
            c = px((*d.xy, z))
            d_.ellipse([c[0] - 4, c[1] - 4, c[0] + 4, c[1] + 4], fill=(255, 255, 255))
            d_.text((c[0] + 6, c[1] - 14), f"{d.cls} {np.degrees(d.yaw) % 180:.0f}°", fill=(255, 255, 255),
                    font=font)
        im = im.resize((300, 300), Image.LANCZOS)
        return lab(np.asarray(im), "overhead camera", f"t={lv['t']:.1f}s", (255, 220, 120))

    def views(sim):
        v0 = sim.render(target=(0.42, -0.15, 0.15), distance=1.25, yaw=-60, pitch=-32, size=SIZE, fov=45)
        v1 = sim.render(target=(0.44, -0.18, 0.12), distance=1.05, yaw=180, pitch=-45, size=SIZE, fov=45)
        return v0, v1

    def cb(sc):
        q, qd = sc.sim.get_state()
        rec["t"].append(sc.t)
        rec["q"].append(q.tolist())
        rec["qd"].append(qd.tolist())
        rec["tau"].append(state.get("tau", [0.0] * 6))
        rec["err"].append(float(np.max(np.abs(sc._q_cmd - q))))
        if sc.t - state["last"] < 0.15 - 1e-9:
            return
        state["last"] = sc.t
        cell = state["cell"]
        v0, v1 = views(sc.sim)
        ndone = sum(1 for e in (cell.log if cell else []) if e.get("result") == "PLACED")
        frames.append(np.hstack([lab(v0, "T111 multi-object sorting", f"t = {sc.t:5.2f} s"),
                                 lab(v1, f"sorted {ndone}/6", state["phase"]), cam_panel(cell)]))

    parts = SO.scatter(0)
    cell = SO.SortCell(parts, SO.SortConfig(), render=True, extra=[POST], frame_cb=cb, cb_every=10)
    state["cell"] = cell
    s = cell.s
    orig_torque = s._arm_torque

    def tau_rec():
        tau = orig_torque()
        state["tau"] = tau.tolist()
        return tau

    s._arm_torque = tau_rec
    orig_goto, orig_lin = cell.goto, cell.lin

    def goto(q, tag):
        state["phase"] = tag + (" (RRT*)" if cell.cfg.plan else "")
        ok = orig_goto(q, tag)
        if cell.motion_log and cell.motion_log[-1].get("planned"):
            state["phase"] = f"{tag}: RRT* {cell.motion_log[-1]['t_plan']:.1f}s"
        return ok

    def lin(z, tag):
        state["phase"] = tag
        return orig_lin(z, tag)

    cell.goto, cell.lin = goto, lin
    r = cell.run()
    for _ in range(8):
        frames.append(frames[-1])
    imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=10, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((550, 165), Image.LANCZOS) for x in frames[::2]]
    pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=200, loop=0, optimize=True)
    nf = len(frames)
    for nm, i in (("key_a", nf // 5), ("key_b", nf // 2), ("key_c", nf - 1)):
        Image.fromarray(frames[i]).save(MEDIA / f"{TAG}_{nm}.png")
    # 相機偵測畫面（第一次拍照）
    first = SO.SortCell(SO.scatter(0))
    first.perceive()
    Image.fromarray(cam_panel(first)).resize((480, 480), Image.LANCZOS).save(MEDIA / f"{TAG}_camera.png")
    first.close()
    curves(rec, cell, r)
    out = slim({k: v for k, v in r.items() if k not in ("vision",)})
    out.update(n_frames=nf, motions=[{k: v for k, v in m.items() if k != "results"} for m in cell.motion_log],
               max_track_err_mrad=float(np.max(rec["err"]) * 1e3) if rec["err"] else None)
    cell.close()
    return out


def curves(rec, cell, r):
    t = np.array(rec["t"])
    q = np.unwrap(np.array(rec["q"]), axis=0)
    qd = np.array(rec["qd"])
    tau = np.array(rec["tau"])
    tt = t
    lim = cell.s.r.tau_max if hasattr(cell.s.r, "tau_max") else None
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for j in range(6):
        ax[0].plot(t, q[:, j], label=f"q{j + 1}")
        ax[1].plot(t, qd[:, j])
        ax[2].plot(tt, tau[:, j] / (lim[j] if lim is not None else 1.0))
    ax[3].semilogy(t, np.maximum(np.array(rec["err"]) * 1e3, 1e-4), color="k")
    for e in cell.log:
        col = "#2f9e44" if e.get("result") == "PLACED" else "#e45756"
        for a in ax:
            a.axvspan(e["t_start"], e["t_end"], color=col, alpha=0.06)
        ax[0].text(e["t_start"] + 0.2, ax[0].get_ylim()[1] if False else 3.3, e["cls"], fontsize=8, color=COLORS[e["cls"]])
    for m in cell.motion_log:
        if m.get("planned"):
            rs = m.get("results", [])
            if rs:
                for a in ax:
                    a.axvspan(rs[0]["t_start"], rs[-1]["t_end"], color="#f59f00", alpha=0.18)
    ax[0].legend(ncol=6, fontsize=8)
    ax[0].set(ylabel="q [rad]", title="Joint trajectories (orange = RRT* planned segments)")
    ax[1].set(ylabel="qd [rad/s]")
    ax[2].set(ylabel="τ / τ_max" if lim is not None else "τ [Nm]")
    ax[3].set(ylabel="max |q_cmd − q| [mrad]", xlabel="t [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=110)
    plt.close(fig)

    # 俯視 TCP 路徑
    tcp = np.array([cell.ex.tcp_of(x) for x in np.array(rec["q"])[::20]])
    fig, ax = plt.subplots(figsize=(7, 7))
    for b in SO.sort_boxes() + [POST]:
        if b.name == "boom":
            continue
        ax.add_patch(mp.Rectangle((b.center[0] - b.half[0], b.center[1] - b.half[1]), 2 * b.half[0], 2 * b.half[1],
                                  fc="#adb5bd" if "post" not in b.name else "#f59f00", ec="k", lw=0.4, alpha=0.6))
    for b, poly in ((None, p) for p in SO.static_footprints()):
        ax.add_patch(mp.Polygon(poly[0], fc="#ced4da", ec="k", lw=0.3, alpha=0.6))
    for p in SO.scatter(0):
        ax.add_patch(mp.Polygon(p.poly, fc=COLORS[p.cls.name], alpha=0.5))
    ax.plot(tcp[:, 0], tcp[:, 1], "k-", lw=0.8, alpha=0.7, label="TCP path")
    ax.add_patch(mp.Rectangle((SO.ZONE[0], SO.ZONE[2]), SO.ZONE[1] - SO.ZONE[0], SO.ZONE[3] - SO.ZONE[2], fill=False,
                              ls="--", ec="#495057"))
    ax.set(aspect="equal", xlim=(0.1, 0.7), ylim=(-0.52, 0.2), xlabel="x [m]", ylabel="y [m]",
           title="Top view: pick zone, tray, bins A/B, post, TCP path")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_topview.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if "--anim" in sys.argv:
        M = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
        plots(M)
        M["anim"] = animate()
        (DATA / f"{TAG}_metrics.json").write_text(json.dumps(slim(M), ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    else:
        main()
