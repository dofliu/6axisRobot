"""T109 報告：軸孔插入 peg-in-hole（RCC 柔順腕、力控柔順、搜孔）→ JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t109_report.py 2026-11-18
注意：與 pytest 不可同時執行（動畫段會重寫 robots/ur5_peg.urdf；掃描段每個行程用暫存 URDF）。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from sixaxis.insertion import (  # noqa: E402
    RIGID,
    ContactSpec,
    Hole,
    HoleSpec,
    InsertCell,
    InsertParams,
    PegBody,
    PegSpec,
    RCCSpec,
    capture_radius,
    chamfer_forces,
    chamfer_ratio,
    jam_depth,
    jam_free_tilt,
    tilt_depth_limit,
    two_point_statics,
    wedge_angle,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T109"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
t_start = time.time()
PEG, HOLE, RCC, CS, IP = PegSpec(), HoleSpec(), RCCSpec(), ContactSpec(), InsertParams()
C_ERR = capture_radius(PEG, HOLE)
ANG = np.radians(30.0)  # 孔位誤差方向
DROP = {"fi0", "fi1", "fi_ins"}


def log(*a):
    print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)


def clean(r):
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items() if k not in DROP}


_URDF = None


def _urdf():
    global _URDF
    if _URDF is None:
        _URDF = Path(tempfile.mkdtemp(prefix=f"t109_{os.getpid()}_")) / "ur5_peg.urdf"
    return _URDF


def run_case(cfg: dict) -> dict:
    """一次插入（子行程）。cfg：name、e（m）、wrist（rcc/rigid/L_c 值）、strategy、search、tilt、mu、keep。"""
    w = cfg.get("wrist", "rcc")
    rcc = RIGID if w == "rigid" else RCCSpec(L_c=float(cfg.get("L_c", 0.0)), K_rot=float(cfg.get("K_rot", RCC.K_rot)))
    cs = ContactSpec(mu=cfg.get("mu", CS.mu))
    e = cfg.get("e", 0.0)
    t0 = time.time()
    c = InsertCell(hole_err=(e * np.cos(ANG), e * np.sin(ANG)), rcc=rcc, contact=cs,
                   grip_tilt_deg=cfg.get("tilt", 0.0), urdf_path=_urdf(), seed=cfg.get("seed", 0))
    ip = InsertParams(strategy=cfg.get("strategy", "admittance"), search=cfg.get("search", False),
                      v_ins=cfg.get("v_ins", IP.v_ins), F_d=cfg.get("F_d", IP.F_d))
    m = c.insert(ip)
    out = {**cfg, **clean(m), "wall_s": time.time() - t0}
    if cfg.get("keep"):
        s = c.series(m["fi_ins"], m["fi1"])
        k = slice(None, None, 4)
        out["series"] = {key: np.asarray(v[k]).tolist() for key, v in s.items()}
    c.close()
    return out


# ------------------------------------------------------------------ 實驗設定
E_LIST = [0.0, 0.3, 0.6, 0.9, 1.05, 1.2, 1.6, 2.5]
STRATS = {"rigid_pos": dict(wrist="rigid", strategy="position"),
          "rcc_pos": dict(wrist="rcc", strategy="position"),
          "rcc_adm": dict(wrist="rcc", strategy="admittance"),
          "rcc_adm_search": dict(wrist="rcc", strategy="admittance", search=True)}
TILTS = [0.0, 1.0, 2.0, 3.0, 4.0]
WRISTS = {"rigid": dict(wrist="rigid"), "rcc_Lc0": dict(wrist="rcc", L_c=0.0), "rcc_Lc100": dict(wrist="rcc", L_c=0.10)}


def chamfer_bench(e: float, mu: float, v: float = 0.002) -> dict:
    """無手臂的倒角穿越台：手部（RCC 上板）以 v 等速下壓，側向固定在 e。回傳側向力峰值與滑動段 F_ax/F_lat。"""
    h = Hole((0, 0, 0), HOLE)
    b = PegBody(PEG, RCC, ContactSpec(mu=mu), h)
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [e, 0, 0.0008]
    b.reset(T)
    vv = np.array([0, 0, -v])
    W = PEG.mass * 9.81
    fl, fa, dd = [], [], []
    for _ in range(int(0.006 / v / 1e-3)):
        T[:3, 3] += vv * 1e-3
        F, _ = b.advance(T, vv, np.zeros(3), 1e-3, 20)
        fl.append(-F[0])
        fa.append(F[2] + W)  # 去除銷體靜重 → 接觸軸向力
        dd.append(h.depth_of(b.p))
        if dd[-1] > 0.003:
            break
    fl, fa, dd = map(np.array, (fl, fa, dd))
    sl = (dd > 0.2 * HOLE.w_c) & (dd < 0.8 * HOLE.w_c) & (fl > 0.3)
    return {"e_mm": e * 1e3, "mu": mu, "F_lat_peak": float(fl.max()), "F_ax_peak": float(fa.max()),
            "ratio_med": float(np.median(fa[sl] / fl[sl])) if sl.any() else float("nan"),
            "pred": chamfer_forces(e, PEG, HOLE, RCC, mu), "ratio_pred": chamfer_ratio(mu),
            "series": {"depth": dd[::5].tolist(), "F_lat": fl[::5].tolist(), "F_ax": fa[::5].tolist()}}


def animate(M: dict):
    """動畫 + 關節曲線（單行程，渲染）。"""
    SIZE = (400, 300)
    frames: list = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    def lab(img, left, right):
        im_ = Image.fromarray(img)
        d_ = ImageDraw.Draw(im_)
        d_.rectangle([0, 0, im_.width, 24], fill=(20, 24, 32))
        d_.text((8, 4), left, fill=(235, 238, 242), font=font)
        d_.text((im_.width - 8 - d_.textlength(right, font=font), 4), right, fill=(150, 190, 255), font=font)
        return np.asarray(im_)

    class Rec:
        def __init__(self, title, period=0.08):
            self.title, self.period, self.last = title, period, -1.0

        def __call__(self, cc):
            if cc.t - self.last < self.period - 1e-9:
                return
            if cc.phase == "approach" and cc.t - self.last < 0.2:
                return
            self.last = cc.t
            hc = cc.hole.c
            v0 = cc.sim.render(target=(hc[0], hc[1], hc[2] + 0.12), distance=0.75, yaw=50, pitch=-25, size=SIZE,
                               fov=42)
            v1 = cc.sim.render(target=(hc[0], hc[1], hc[2] - 0.004), distance=0.12, yaw=180, pitch=-18, size=SIZE,
                               fov=40)
            Fz = float(cc.wr_true[2]) if cc.frec["t"] else 0.0
            d = cc.hole.depth_of(cc.body.p) * 1e3
            frames.append(np.hstack([lab(v0, self.title, f"t = {cc.t:.2f} s"),
                                     lab(v1, f"{cc.phase}  depth {d:+5.2f} mm", f"F_ax {Fz:5.1f} N")]))

    demo = []
    M["demo_frames"] = []
    joint = None
    for title, kw, ipk in (
        ("RCC + admittance + spiral search, e=1.6 mm", dict(hole_err=(1.6e-3 * np.cos(ANG), 1.6e-3 * np.sin(ANG))),
         dict(search=True)),
        ("RCC (K_rot 5), peg tilted 3 deg", dict(grip_tilt_deg=3.0, rcc=RCCSpec(K_rot=5.0)), dict()),
        ("rigid wrist, position ctrl, e=0.6 mm",
         dict(rcc=RIGID, hole_err=(0.6e-3 * np.cos(ANG), 0.6e-3 * np.sin(ANG))), dict(strategy="position")),
    ):
        c = InsertCell(render=True, **kw)
        c.hook = Rec(title)
        n0 = len(c.rec["t"])
        m = c.insert(InsertParams(**ipk))
        c.hold(0.4)
        demo.append({"title": title, **clean(m)})
        if joint is None:
            joint = {k: np.array(c.rec[k][n0:]) for k in ("t", "q", "qd", "tau", "q_ref")}
        c.close()
        log("demo", title, m["ok"], m["error"], len(frames))
        M.setdefault("demo_frames", []).append(len(frames))
    M["demo"] = demo
    imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=15, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((600, 225), Image.LANCZOS) for x in frames[::2]]
    pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=130, loop=0, optimize=True)
    nf = len(frames)
    b1, b2 = M["demo_frames"][0], M["demo_frames"][1]
    for nm, idx_ in (("key_a", int(b1 * 0.55)), ("key_b", b1 - 6), ("key_c", int(b1 + (b2 - b1) * 0.75)),
                     ("key_d", nf - 4)):
        Image.fromarray(frames[idx_]).save(MEDIA / f"{TAG}_{nm}.png")
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    t = joint["t"] - joint["t"][0]
    for j in range(6):
        ax[0, 0].plot(t, np.degrees(joint["q"][:, j]), label=f"J{j + 1}")
        ax[0, 1].plot(t, joint["qd"][:, j], label=f"J{j + 1}")
        ax[1, 0].plot(t, joint["tau"][:, j], label=f"J{j + 1}")
        ax[1, 1].plot(t, np.degrees(joint["q_ref"][:, j] - joint["q"][:, j]) * 3600, label=f"J{j + 1}")
    for a, yl in zip(ax.flat, ("q [deg]", "qd [rad/s]", "tau [N m]", "q_ref - q [arcsec]"), strict=True):
        a.set_ylabel(yl)
        a.set_xlabel("t [s]")
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7, ncol=3)
    ax[1, 1].set_ylim(-60, 60)
    fig.suptitle("demo 1 (RCC + admittance + search, e = 1.6 mm): joint trajectories")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=110)
    plt.close(fig)
    M["frames"] = nf
    M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
    M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
    M["anim_s"] = time.time() - t_start
    M.setdefault("total_s", M["anim_s"])
    for r in M["runs"]:
        r.pop("series", None)
    json.dump(M, open(DATA / f"{TAG}_metrics.json", "w"), indent=1, ensure_ascii=False, default=float)
    log("done frames", nf, "gif MB", M["gif_MB"], "mp4 MB", M["mp4_MB"])


if __name__ == "__main__":
    if "--anim" in sys.argv:  # 只重做動畫/關節曲線（沿用既有 JSON 的掃描結果）
        M0 = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
        animate(M0)
        sys.exit(0)
    M: dict = {"date": DATE, "peg": PEG.__dict__, "hole": HOLE.__dict__, "rcc": RCC.__dict__, "rigid": RIGID.__dict__,
               "contact": CS.__dict__, "insert": IP.__dict__,
               "theory": {"capture_mm": C_ERR * 1e3, "clearance_um": (HOLE.R - PEG.r) * 1e6,
                          "clearance_ratio": 2 * (HOLE.R - PEG.r) / (2 * PEG.r),
                          "chamfer_ratio_mu03": chamfer_ratio(0.3),
                          "wedge_deg": float(np.degrees(wedge_angle(PEG, HOLE, CS.mu))),
                          "tilt_limit_mm": {str(t): (HOLE.w_c + tilt_depth_limit(np.radians(t), PEG, HOLE)) * 1e3
                                            for t in TILTS if t > 0}}}
    # --- C 倒角穿越台（主行程，快速）
    log("chamfer bench")
    with Pool(4) as pool:
        M["chamfer"] = pool.starmap(chamfer_bench, [(e * 1e-3, 0.3) for e in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                                                          0.9, 1.0)])
        M["chamfer_mu"] = pool.starmap(chamfer_bench, [(0.8e-3, mu) for mu in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)])
    # 兩點接觸靜力（理論）：示意卡死圖用格點
    lam = np.linspace(-1.5, 1.5, 121)
    M["jam_grid"] = {"l_mm": [2.0, 5.0, 10.0], "fx_over_fz": lam.tolist(),
                     "jam": [[two_point_statics(fx, 1.0, 0.0, l * 1e-3, PEG.r, CS.mu)["jam"] for fx in lam]
                             for l in (2.0, 5.0, 10.0)]}
    # --- B、D 掃描（平行）
    cases = []
    for sn, sc in STRATS.items():
        for e in E_LIST:
            cases.append({"exp": "lateral", "name": sn, "e": e * 1e-3, **sc,
                          "keep": (sn == "rcc_adm_search" and e in (0.9, 2.5)) or (sn == "rcc_pos" and e == 0.9)})
    for wn, wc in WRISTS.items():
        for th in TILTS:
            cases.append({"exp": "tilt", "name": wn, "tilt": th, "strategy": "admittance", **wc,
                          "keep": th == 3.0})
    for mu in (0.1, 0.5):
        cases.append({"exp": "mu", "name": "rcc_Lc100", "tilt": 3.0, "wrist": "rcc", "L_c": 0.10, "mu": mu,
                      "strategy": "admittance"})
        cases.append({"exp": "mu", "name": "rcc_Lc0", "tilt": 3.0, "wrist": "rcc", "L_c": 0.0, "mu": mu,
                      "strategy": "admittance"})
    for th, Fd, Kr in ((2.5, 40.0, 15.0), (3.5, 40.0, 15.0), (2.0, 15.0, 5.0), (3.0, 15.0, 5.0), (4.0, 15.0, 5.0),
                       (1.5, 15.0, 15.0), (1.5, 15.0, 30.0)):
        cases.append({"exp": "jamfree", "name": f"Fd{Fd:.0f}_Kr{Kr:.0f}", "tilt": th, "wrist": "rcc", "L_c": 0.0,
                      "F_d": Fd, "K_rot": Kr, "strategy": "admittance"})
    for r_ in cases:
        if r_["exp"] in ("tilt", "mu", "jamfree") and r_.get("wrist") == "rcc" and r_.get("L_c", 0.0) == 0.0:
            mu_, Fd_, Kr_ = r_.get("mu", CS.mu), r_.get("F_d", IP.F_d), r_.get("K_rot", RCC.K_rot)
            r_["theta_max_deg"] = float(np.degrees(jam_free_tilt(Fd_, PEG, HOLE, mu_, Kr_)))
            r_["jam_depth_pred_mm"] = (HOLE.w_c + jam_depth(np.radians(r_["tilt"]), Fd_, PEG, HOLE, mu_, Kr_)) * 1e3
    log("cases", len(cases))
    with Pool(4) as pool:
        res = []
        for r in pool.imap_unordered(run_case, sorted(cases, key=lambda c: c.get("wrist") == "rigid")):
            res.append(r)
            log(r["exp"], r["name"], f"e={r.get('e', 0) * 1e3:.2f}", f"tilt={r.get('tilt', 0)}",
                r["ok"], r["error"], f"{r['wall_s']:.0f}s")
    M["runs"] = res
    json.dump(M, open(DATA / f"{TAG}_metrics.json", "w"), indent=1, ensure_ascii=False, default=float)
    log("sweeps done")

    # ------------------------------------------------------------------ 圖
    def get(exp, name, key="e"):
        rr = [r for r in res if r["exp"] == exp and r["name"] == name]
        return sorted(rr, key=lambda r: r.get(key, 0))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(STRATS)
    lbl = {"rigid_pos": "rigid wrist, position", "rcc_pos": "RCC, position", "rcc_adm": "RCC + admittance",
           "rcc_adm_search": "RCC + admittance + spiral search"}
    for i, n in enumerate(names):
        rr = get("lateral", n)
        for r in rr:
            ok = r["ok"]
            ax[0].scatter(r["e"] * 1e3, i, s=120, marker="o" if ok else "x", c="#2a9d55" if ok else "#d1495b")
            ax[0].annotate(r["error"] or f"{r['cycle_s']:.1f}s", (r["e"] * 1e3, i + 0.18), ha="center", fontsize=7)
        ax[1].plot([r["e"] * 1e3 for r in rr], [r["F_max"] for r in rr], "o-", label=lbl[n])
    ax[0].axvline(C_ERR * 1e3, ls="--", c="k", lw=1)
    ax[0].text(C_ERR * 1e3 + 0.03, -0.5, "capture w_c + c", fontsize=8)
    ax[0].set_yticks(range(len(names)), [lbl[n] for n in names], fontsize=8)
    ax[0].set_ylim(-0.7, len(names) - 0.4)
    ax[0].set_xlabel("hole position error e [mm]")
    ax[0].set_title("success (o) / failure (x)")
    ax[1].axvline(C_ERR * 1e3, ls="--", c="k", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("hole position error e [mm]")
    ax[1].set_ylabel("max |F| on wrist [N]")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_lateral.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ch = M["chamfer"]
    ee = np.array([c["e_mm"] for c in ch])
    ax[0].plot(ee, [c["F_lat_peak"] for c in ch], "o", label="sim: peak lateral")
    ax[0].plot(ee, [c["pred"]["F_lat"] for c in ch], "-", label="K_lat (e - c)")
    ax[0].plot(ee, [c["F_ax_peak"] for c in ch], "s", label="sim: peak axial")
    ax[0].plot(ee, [c["pred"]["F_ax"] for c in ch], "--", label="K_lat (e - c) (1+mu tan a)/(tan a - mu)")
    ax[0].set_xlabel("lateral error e [mm]")
    ax[0].set_ylabel("force [N]")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    cm = M["chamfer_mu"]
    mus = np.array([c["mu"] for c in cm])
    ax[1].plot(mus, [c["ratio_med"] for c in cm], "o", label="sim (median while sliding)")
    mm = np.linspace(0, 0.5, 50)
    ax[1].plot(mm, [chamfer_ratio(x) for x in mm], "-", label="(1+mu)/(1-mu)")
    ax[1].set_xlabel("friction mu")
    ax[1].set_ylabel("F_axial / F_lateral on chamfer")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    s = ch[-3]["series"]
    ax[2].plot(np.array(s["depth"]) * 1e3, s["F_lat"], label="F_lat")
    ax[2].plot(np.array(s["depth"]) * 1e3, s["F_ax"], label="F_ax (contact)")
    ax[2].axvspan(0, HOLE.w_c * 1e3, color="0.9")
    ax[2].set_xlabel("peg depth [mm] (shaded = chamfer)")
    ax[2].set_title(f"bench e = {ch[-3]['e_mm']:.1f} mm, mu = 0.3", fontsize=9)
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_chamfer.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for wn in WRISTS:
        rr = get("tilt", wn, "tilt")
        th = [r["tilt"] for r in rr]
        ax[0].plot(th, [r["F_max"] for r in rr], "o-", label=wn)
        ax[1].plot(th, [r["M_max"] for r in rr], "o-", label=wn)
        ax[2].plot(th, [r["tp_onset_mm"] for r in rr], "o", label=f"{wn}: 2-pt onset depth")
        for r in rr:
            if not r["ok"]:
                ax[0].annotate(r["error"], (r["tilt"], r["F_max"]), fontsize=7)
    tt = np.linspace(0.6, 4.2, 60)
    ax[2].plot(tt, [(HOLE.w_c + tilt_depth_limit(np.radians(x), PEG, HOLE)) * 1e3 for x in tt], "k-",
               label="geometric limit w_c + l(t)")
    ax[2].axhline((HOLE.w_c + CS.mu * 2 * PEG.r) * 1e3, ls=":", c="r", label="w_c + mu D (wedging)")
    ax[0].set_ylabel("max |F| [N]")
    ax[1].set_ylabel("max |M| [N m]")
    ax[2].set_ylabel("depth [mm]")
    for a in ax:
        a.set_xlabel("peg tilt in gripper [deg]")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    ax[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_tilt.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    jf = [r for r in res if "theta_max_deg" in r]
    for r in jf:
        ok = r["ok"]
        ax[0].scatter(r["theta_max_deg"], r["tilt"], s=90, marker="o" if ok else "x", c="#2a9d55" if ok else "#d1495b")
        ax[0].annotate(f"F{r.get('F_d', IP.F_d):.0f}/mu{r.get('mu', CS.mu)}/K{r.get('K_rot', RCC.K_rot):.0f}",
                       (r["theta_max_deg"], r["tilt"]), fontsize=6, xytext=(4, -9), textcoords="offset points")
        if not ok and np.isfinite(r.get("jam_depth_pred_mm", np.inf)):
            ax[1].scatter(r["jam_depth_pred_mm"], r["depth_mm"], c="#d1495b")
    xx = np.linspace(1, 4.5, 10)
    ax[0].plot(xx, xx, "k--", lw=1, label="tilt = theta_max (prediction boundary)")
    ax[0].set_xlabel("predicted jam-free bound theta_max = 2 sqrt(F c/(mu K_rot)) [deg]")
    ax[0].set_ylabel("initial peg tilt [deg]")
    ax[0].set_title("o = inserted, x = JAM (RCC, L_c = 0)", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot([3, 12], [3, 12], "k--", lw=1)
    ax[1].set_xlabel("predicted jam depth w_c + l_j [mm]")
    ax[1].set_ylabel("simulated stall depth [mm]")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_jamfree.png", dpi=110)
    plt.close(fig)

    keep = [r for r in res if "series" in r]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for r in keep:
        s = r["series"]
        t = np.array(s["t"]) - s["t"][0]
        lab_ = f"{r['name']} e={r.get('e', 0) * 1e3:.1f}mm tilt={r.get('tilt', 0)}"
        F = np.array(s["F"])
        ax[0, 0].plot(t, np.array(s["depth"]) * 1e3, label=lab_)
        ax[0, 1].plot(t, F[:, 2], label=lab_)
        ax[1, 0].plot(t, np.linalg.norm(F[:, :2], axis=1), label=lab_)
        hl = np.array(s["hand_lat"]) * 1e3
        ax[1, 1].plot(hl[:, 0], hl[:, 1], label=lab_)
    ax[0, 0].set_ylabel("peg depth [mm]")
    ax[0, 1].set_ylabel("axial force on wrist [N]")
    ax[1, 0].set_ylabel("lateral force [N]")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("t from insert start [s]")
    ax[1, 1].set_xlabel("hand x [mm]")
    ax[1, 1].set_ylabel("hand y [mm]")
    ax[1, 1].set_aspect("equal")
    for e_ in (0.9, 2.5):
        ax[1, 1].plot(e_ * np.cos(ANG), e_ * np.sin(ANG), "k*")
    for a in ax.flat:
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_series.png", dpi=110)
    plt.close(fig)
    log("plots done")

    animate(M)
