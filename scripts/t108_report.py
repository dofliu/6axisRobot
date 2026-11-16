"""T108 報告：去毛邊/拋光（力控、接觸力維持）→ JSON / 圖 / 動畫。
用法：PYTHONPATH=. python3 scripts/t108_report.py 2026-11-16
注意：與 pytest 不可同時執行（都會重寫 robots/ur5_polisher.urdf）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as pb  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from sixaxis.finishing import (  # noqa: E402
    FinishCell,
    FinishParams,
    ForceParams,
    PadModel,
    Surface,
    line_on,
    loop_model,
    normal_mass,
    pass_depth,
    predict_touchdown,
    raster_on,
    removal_stats,
    routh_kcrit,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
TAG = f"{DATE}_T108"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
t_start = time.time()
C = (0.45, 0.0, 0.15)
SPACING, NSTROKE, LSTROKE = 0.010, 6, 0.100
U_IN, W_IN = 0.035, 0.020
DROP = {"i0", "i1", "fi0", "fi1", "fi_begin"}
PAD, FP, PRM = PadModel(), ForceParams(), FinishParams()


def clean(r):
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items() if k not in DROP}


def log(*a):
    print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)


M: dict = {"date": DATE, "pad": {**PAD.__dict__, "k_w": PAD.k_w, "omega": PAD.omega},
           "force": FP.__dict__, "finish": {k: v for k, v in PRM.__dict__.items() if k != "force"},
           "raster": {"spacing": SPACING, "n": NSTROKE, "length": LSTROKE, "u_in": U_IN, "w_in": W_IN},
           "loop_nominal": loop_model(PAD.k_e, FP.M, FP.D), "routh_kcrit_nominal": routh_kcrit(FP.M, FP.D, FP.f_c)}


def raster_pred(pad: PadModel, F: float, v: float):
    """解析：各道（忽略換道圓弧）單道剖面疊加 → 內部區平均/CV。"""
    w = np.linspace(-W_IN, W_IN, 801)
    ws = (np.arange(NSTROKE) - (NSTROKE - 1) / 2) * SPACING
    h = sum(pass_depth(w - wk, F, v, pad) for wk in ws)
    return w, h


# ------------------------------------------------------------------ A/B 拋光：表面誤差情境 × 控制模式
SCEN = [("nominal", "名目平面", {}), ("off_p", "整體高 +0.6 mm", {"offset": 6e-4}),
        ("off_n", "整體低 −0.6 mm", {"offset": -6e-4}), ("tilt", "橫向傾斜 0.5°", {"slope_w_deg": 0.5}),
        ("wave", "起伏 0.3 mm／λ 40 mm", {"wave_amp": 3e-4, "wave_len": 0.04})]
polish, series_B, maps = [], {}, {}
for key, label, shape in SCEN:
    for mode in ("admittance", "position"):
        c = FinishCell()
        s = Surface(C, size=(0.16, 0.10)).set_shape(**shape)
        c.add_surface(s)
        seam, _ = raster_on(s, SPACING, NSTROKE, LSTROKE)
        m = c.finish(seam, FinishParams(force=ForceParams(mode=mode)))
        st = removal_stats(s, U_IN, W_IN)
        row = {"scenario": key, "label": label, **clean(m), **{f"rem_{k}": v for k, v in st.items()}}
        if key == "nominal" and mode == "admittance":
            lam = normal_mass(c, c.rec["q"][m["i0"]], s.n)
            M["lambda_n"] = lam
            series_B["nom_q"] = np.array(c.rec["q"][m["i0"]:m["i1"]])
            series_B["nom_tau"] = np.array(c.rec["tau"][m["i0"]:m["i1"]])
            series_B["nom_qd"] = np.array(c.rec["qd"][m["i0"]:m["i1"]])
        f = c.fseries(m["fi0"], m["fi1"]) if m.get("fi0") is not None else None
        if f is not None:
            series_B[(key, mode)] = (f["t"] - f["t"][0], f["Fn"], f["z"])
        maps[(key, mode)] = s.removal().copy()
        polish.append(row)
        log("polish", key, mode, m["error"], f"F_rms={m.get('F_rms_err', -1):.2f}",
            f"rem={st['mean_um']:.2f}um cv={st['cv']:.3f}")
        c.close()
wp, hp = raster_pred(PAD, FP.F_d, PRM.speed)
M["polish"] = polish
M["raster_pred"] = {"mean_um": float(hp.mean() * 1e6), "cv": float(hp.std() / hp.mean())}
gu, gw = s.gu, s.gw

# ------------------------------------------------------------------ C 觸地：接觸剛性 × 阻尼
LAM = M["lambda_n"]
touch, tseries = [], {}
for ke in (1e4, 5e4, 2e5, 5e5, 1e6, 2e6):
    for dmode in ("fixed", "crit"):
        D = FP.D if dmode == "fixed" else float(2 * np.sqrt(ke * FP.M))
        c = FinishCell(pad=PadModel(k_e=ke))
        s = Surface(C, size=(0.16, 0.10))
        c.add_surface(s)
        fp = ForceParams(D=D)
        m = c.finish(line_on(s, -0.02, 0.02, 0.0), FinishParams(force=fp))
        F = np.array(c.frec["Fn"])
        ph = np.array(c.frec["phase"])
        land = np.where(ph == "land")[0]
        k0 = land[np.argmax(F[land] > 0)] if (F[land] > 0).any() else land[0]
        Fl = F[k0:land[-1] + 1]
        pc = predict_touchdown(ke, fp.M, D, fp.F_d, fp.v_app, fp.f_c, lam=LAM, series=True)
        pd = predict_touchdown(ke, fp.M, D, fp.F_d, fp.v_app, fp.f_c, lam=LAM, dt=1e-3)
        pr = predict_touchdown(ke, fp.M, D, fp.F_d, fp.v_app, fp.f_c, lam=None)
        lm = loop_model(ke, fp.M, D)
        row = {"k_e": ke, "D": D, "D_mode": dmode, "zeta_ideal": lm["zeta"], "routh_kcrit": routh_kcrit(fp.M, D, fp.f_c),
               "pred_rigid_os": pr["overshoot"], "pred_rigid_osc": pr["osc_std"],
               "pred_servo_os": pc["overshoot"], "pred_servo_osc": pc["osc_std"],
               "pred_servo1ms_os": pd["overshoot"], "pred_servo1ms_osc": pd["osc_std"],
               "sim_os": float(Fl.max() / fp.F_d - 1), "sim_land_s": m.get("land_s"), "error": m["error"],
               "F_rms_err": m.get("F_rms_err"), "sim_tail_std": float(Fl[-100:].std())}
        touch.append(row)
        tseries[(ke, dmode)] = (np.arange(len(Fl)) * c.dt, Fl, pc["t"], pc["F"])
        log("touch", f"{ke:.0e}", dmode, f"sim {row['sim_os']:.3f} pred {row['pred_servo_os']:.3f}", m["error"])
        c.close()
M["touchdown"] = touch

# ------------------------------------------------------------------ D 斜坡：穩態力誤差 D·v·tanα 與積分項
slope = []
for a in (0.5, 1.0, 2.0, 3.0):
    for ki in (0.0, 20.0):
        c = FinishCell()
        s = Surface(C, size=(0.16, 0.10)).set_shape(slope_u_deg=a)
        c.add_surface(s)
        m = c.finish(line_on(s, -0.03, 0.03, 0.0), FinishParams(force=ForceParams(K_i=ki)))
        F = np.array(c.frec["Fn"][m["fi0"]:m["fi1"]])
        t = np.arange(len(F)) * c.dt
        mid = (t > 0.8) & (t < t[-1] - 0.8)
        slope.append({"slope_deg": a, "K_i": ki, "e_pred": FP.D * PRM.speed * np.tan(np.radians(a)),
                      "e_sim": float(F[mid].mean() - FP.F_d), "F_rms_err": m["F_rms_err"], "error": m["error"]})
        log("slope", a, ki, f"{slope[-1]['e_sim']:.3f} vs {slope[-1]['e_pred']:.3f}")
        c.close()
M["slope"] = slope

# ------------------------------------------------------------------ E 單道移除剖面與速度
speed_rows, prof = [], {}
for v in (0.010, 0.025, 0.050):
    c = FinishCell()
    s = Surface(C, size=(0.12, 0.05), res=2.5e-4)
    c.add_surface(s)
    m = c.finish(line_on(s, -0.04, 0.04, 0.0), FinishParams(speed=v))
    R = s.removal()[np.abs(s.gu) < 0.01].mean(0)
    h = pass_depth(s.gw, m["F_mean"], v, PAD)
    core = np.abs(s.gw) < 0.9 * PAD.radius
    speed_rows.append({"v": v, "F_mean": m["F_mean"], "h0_sim_um": float(R[np.argmin(np.abs(s.gw))] * 1e6),
                       "h0_pred_um": float(pass_depth(0.0, m["F_mean"], v, PAD) * 1e6),
                       "prof_max_err_pct": float(np.max(np.abs(R[core] - h[core])) / h.max() * 100),
                       "hmax_sim_um": float(R.max() * 1e6), "hmax_pred_um": float(h.max() * 1e6),
                       "F_rms_err": m["F_rms_err"]})
    prof[v] = (s.gw.copy(), R, h)
    log("speed", v, speed_rows[-1]["h0_sim_um"], speed_rows[-1]["h0_pred_um"])
    c.close()
M["speed"] = speed_rows

# ------------------------------------------------------------------ F 力前饋與感測雜訊（起伏面）
ffn = []
for ff, sig in ((True, 0.3), (False, 0.3), (True, 0.0), (True, 1.0)):
    c = FinishCell()
    s = Surface(C, size=(0.16, 0.10)).set_shape(wave_amp=3e-4, wave_len=0.04)
    c.add_surface(s)
    m = c.finish(line_on(s, -0.04, 0.04, 0.0), FinishParams(force=ForceParams(force_ff=ff, sigma_F=sig)))
    ffn.append({"force_ff": ff, "sigma_F": sig, **{k: m.get(k) for k in ("F_rms_err", "F_max_err", "F_std",
                                                                        "land_overshoot", "lat_max_mm", "error")}})
    log("ff/noise", ff, sig, m["F_rms_err"])
    c.close()
M["ff_noise"] = ffn


# ------------------------------------------------------------------ G 去毛邊：硬墊/軟墊 × 力
def burr_fn(u):
    return 0.25e-3 + 0.08e-3 * np.sin(2 * np.pi * u / 0.05 + 1.0) + 0.04e-3 * np.sin(2 * np.pi * u / 0.017)


def deburr_surface():
    s = Surface(C, size=(0.14, 0.04), res=2.5e-4)
    s.add_ridge(0.0, 1e-3, burr_fn, u_range=(-0.05, 0.05))
    return s


TOL = 0.05e-3
DEB = [("hard_10", "硬墊 5e5 N/m、10 N", 5e5, 10.0, 2000.0, 8),
       ("hard_20", "硬墊 5e5 N/m、20 N", 5e5, 20.0, 2000.0, 8),
       ("hard_40", "硬墊 5e5 N/m、40 N", 5e5, 40.0, 2000.0, 8),
       ("soft_20", "軟墊 5e4 N/m、20 N", 5e4, 20.0, 632.0, 3)]
deburr, burr_prof = [], {}
for key, label, ke, Fd, D, nmax in DEB:
    c = FinishCell(pad=PadModel(k_e=ke))
    s = deburr_surface()
    c.add_surface(s)
    seam = line_on(s, -0.06, 0.06, -0.0075)
    in_r = np.abs(s.gu) <= 0.05
    base = ~s.ridge & (np.abs(s.gu)[:, None] <= 0.05)
    hist = [{"pass": 0, "burr_max_um": float(s.burr()[in_r].max() * 1e6),
             "burr_mean_um": float(s.burr()[in_r].mean() * 1e6), "gouge_um": 0.0}]
    burr_prof[key] = [s.burr().copy()]
    t_tot = 0.0
    for k in range(1, nmax + 1):
        m = c.finish(seam, FinishParams(speed=0.010, force=ForceParams(F_d=Fd, D=D)))
        t_tot += m["cycle_s"]
        b = s.burr()
        burr_prof[key].append(b.copy())
        hist.append({"pass": k, "burr_max_um": float(b[in_r].max() * 1e6), "burr_mean_um": float(b[in_r].mean() * 1e6),
                     "gouge_um": float(s.removal()[base].max() * 1e6), "F_rms_err": m["F_rms_err"],
                     "F_max": m["F_max"], "torque_util": m["torque_util"], "cycle_s": m["cycle_s"], "error": m["error"]})
        log("deburr", key, k, hist[-1]["burr_max_um"], hist[-1]["gouge_um"])
        if b[in_r].max() <= TOL or not m["ok"]:
            break
    deburr.append({"key": key, "label": label, "k_e": ke, "F_d": Fd, "D": D, "passes": hist,
                   "done": bool(hist[-1]["burr_max_um"] <= TOL * 1e6), "time_s": t_tot})
    c.close()
M["deburr"] = deburr
M["deburr_tol_um"] = TOL * 1e6
gu_d = deburr_surface().gu

# ------------------------------------------------------------------ 圖
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
for fam in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
    if any(fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [fam, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
COL = ["#2a6fdb", "#d1495b", "#00a6a6", "#edae49", "#66a182", "#8d6cab"]

# 1 觸地
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, ke in enumerate((5e4, 2e5, 5e5, 1e6)):
    for j, dm in enumerate(("fixed", "crit")):
        ts, Fs, tp, Fp = tseries[(ke, dm)]
        ax[j].plot(ts * 1e3, Fs, color=COL[i], lw=1.0, label=f"k_e={ke:.0e} 模擬")
        ax[j].plot(tp * 1e3, Fp, color=COL[i], lw=1.0, ls="--", alpha=0.8)
for j, ttl in enumerate((f"固定 D = {FP.D:.0f} N·s/m", "D = 2√(k_e·M)（理想 ζ = 1）")):
    ax[j].axhline(FP.F_d, color="k", lw=0.6)
    ax[j].set(xlim=(0, 400), xlabel="自接觸起 t [ms]", ylabel="法向力 [N]", title=f"觸地暫態：{ttl}（虛線＝伺服柔度模型）")
    ax[j].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_touchdown.png", dpi=130)
plt.close(fig)

# 2 移除深度圖與剖面
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
for j, (key, mode) in enumerate((("nominal", "admittance"), ("wave", "position"))):
    im = ax[j].imshow(maps[(key, mode)].T * 1e6, origin="lower", aspect="equal", cmap="viridis",
                      extent=[gu[0] * 1e3, gu[-1] * 1e3, gw[0] * 1e3, gw[-1] * 1e3], vmin=0,
                      vmax=np.percentile(maps[("nominal", "admittance")], 99.5) * 1.6e6)
    ax[j].add_patch(plt.Rectangle((-U_IN * 1e3, -W_IN * 1e3), 2 * U_IN * 1e3, 2 * W_IN * 1e3, fill=False, ec="w", lw=0.8))
    ax[j].set(xlabel="u [mm]", ylabel="w [mm]", title=f"移除深度 [µm]：{'名目面＋導納' if j == 0 else '起伏面＋位置控制'}")
    plt.colorbar(im, ax=ax[j], shrink=0.8)
mid = np.abs(gu) <= U_IN
ax[2].plot(gw * 1e3, maps[("nominal", "admittance")][mid].mean(0) * 1e6, color=COL[0], label="模擬（|u| ≤ 35 mm 平均）")
ax[2].plot(wp * 1e3, hp * 1e6, "k--", lw=1, label="解析：單道 h(w) 疊加")
ax[2].set(xlabel="w [mm]", ylabel="移除深度 [µm]", title="橫向剖面（6 道、道距 10 mm）")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_removal.png", dpi=130)
plt.close(fig)

# 3 情境：力
fig, ax = plt.subplots(len(SCEN), 1, figsize=(11, 9), sharex=True)
for i, (key, label, _) in enumerate(SCEN):
    for mode, col in (("position", COL[1]), ("admittance", COL[0])):
        if (key, mode) in series_B:
            t, F, _ = series_B[(key, mode)]
            ax[i].plot(t, F, color=col, lw=0.7, label="位置控制" if mode == "position" else "導納力控")
    ax[i].axhline(FP.F_d, color="k", lw=0.6)
    ax[i].set(ylabel="F_n [N]", title=label)
    ax[i].legend(fontsize=7, loc="upper right")
ax[-1].set_xlabel("拋光段 t [s]")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_scenarios.png", dpi=120)
plt.close(fig)

# 4 斜坡 + 速度
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for ki, col in ((0.0, COL[1]), (20.0, COL[0])):
    rr = [r for r in slope if r["K_i"] == ki]
    ax[0].plot([r["slope_deg"] for r in rr], [r["e_sim"] for r in rr], "o-", color=col, label=f"模擬 K_i = {ki:.0f}")
aa = np.linspace(0, 3.2, 50)
ax[0].plot(aa, FP.D * PRM.speed * np.tan(np.radians(aa)), "k--", lw=1, label="預測 D·v·tanα")
ax[0].set(xlabel="上坡角 α [°]", ylabel="穩態 F − F_d [N]", title="斜面穩態力誤差")
ax[0].legend(fontsize=7)
for i, v in enumerate(prof):
    w_, R_, h_ = prof[v]
    ax[1].plot(w_ * 1e3, R_ * 1e6, color=COL[i], label=f"模擬 v = {v * 1e3:.0f} mm/s")
    ax[1].plot(w_ * 1e3, h_ * 1e6, "k--", lw=0.8)
ax[1].set(xlabel="w [mm]", ylabel="移除深度 [µm]", title="單道剖面（虛線＝Preston 解析）")
ax[1].legend(fontsize=7)
vv = np.array([r["v"] for r in speed_rows])
ax[2].loglog(vv * 1e3, [r["h0_sim_um"] for r in speed_rows], "o", color=COL[0], label="模擬中心深度")
ax[2].loglog(vv * 1e3, [r["h0_pred_um"] for r in speed_rows], "k--", label="K_p·F·ω/(π v)")
ax[2].set(xlabel="v [mm/s]", ylabel="h(0) [µm]", title="深度 ∝ 1/v")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_slope_speed.png", dpi=130)
plt.close(fig)

# 5 去毛邊
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for i, d in enumerate(deburr):
    P = d["passes"]
    ax[0].plot([p["pass"] for p in P], [p["burr_max_um"] for p in P], "o-", color=COL[i], label=d["label"])
    ax[1].plot([p["pass"] for p in P], [p["gouge_um"] for p in P], "s-", color=COL[i], label=d["label"])
ax[0].axhline(TOL * 1e6, color="k", ls=":", lw=1)
ax[0].set(xlabel="道次", ylabel="殘留毛邊最大高 [µm]", title=f"毛邊高（容許 {TOL * 1e6:.0f} µm）")
ax[1].set(xlabel="道次", ylabel="母材最大過切 [µm]", title="母材過切（gouge）")
for a_ in ax[:2]:
    a_.legend(fontsize=7)
bp = burr_prof["hard_20"]
for k, b in enumerate(bp):
    ax[2].plot(gu_d * 1e3, b * 1e6, color=plt.cm.viridis(k / max(len(bp) - 1, 1)), lw=0.9, label=f"道次 {k}")
ax[2].set(xlabel="u [mm]", ylabel="毛邊高 [µm]", title="硬墊 20 N：各道次毛邊剖面")
ax[2].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_deburr.png", dpi=130)
plt.close(fig)

# 6 關節（名目拋光）
q, qd, tau = series_B["nom_q"], series_B["nom_qd"], series_B["nom_tau"]
tt = np.arange(len(q)) * 1e-3
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for j in range(6):
    ax[0].plot(tt, np.degrees(q[:, j] - q[0, j]), lw=0.8, label=f"J{j + 1}")
    ax[1].plot(tt, np.degrees(qd[:, j]), lw=0.8)
    ax[2].plot(tt, tau[:, j], lw=0.8)
ax[0].set(ylabel="Δq [°]", title="名目拋光（導納力控）關節軌跡")
ax[0].legend(ncol=6, fontsize=7)
ax[1].set(ylabel="q̇ [°/s]")
ax[2].set(ylabel="τ [N·m]", xlabel="t [s]")
fig.tight_layout()
fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=120)
plt.close(fig)
log("figures done")

# ------------------------------------------------------------------ 動畫
SIZE = (400, 320)
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
    def __init__(self, title, overview, period=0.12):
        self.title, self.overview, self.last, self.period = title, overview, -1.0, period
        self.trail_last = None

    def __call__(self, cc):
        if cc.phase == "run" and cc.frec["Fn"] and cc.frec["Fn"][-1] > 1.0 and cc.trail:
            p = cc.tcp_pose()[:3, 3]
            if self.trail_last is None or np.linalg.norm(p - self.trail_last) > 0.004:
                s = cc.surf
                u, w, _ = s.local(p)
                vs = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.0115, length=1e-4, rgbaColor=(0.86, 0.90, 0.95, 1),
                                          physicsClientId=cc.cid)
                pb.createMultiBody(0, -1, vs, list(s.world(u, w, 1.5e-4)), physicsClientId=cc.cid)
                self.trail_last = p
        if cc.t - self.last < self.period - 1e-9:
            return
        self.last = cc.t
        T = cc.tcp_pose()
        v0 = cc.sim.render(target=self.overview[0], distance=self.overview[3], yaw=self.overview[1],
                           pitch=self.overview[2], size=SIZE, fov=42)
        v1 = cc.sim.render(target=tuple(T[:3, 3] + np.array([0, 0, 0.02])), distance=0.20, yaw=20, pitch=-24,
                           size=SIZE, fov=40)
        F = cc.frec["Fn"][-1] if cc.frec["Fn"] else 0.0
        frames.append(np.hstack([lab(v0, self.title, f"t = {cc.t:.1f} s"),
                                 lab(v1, f"{cc.phase}  F_n = {F:5.1f} N (set {cc.fp.F_d:.0f})",
                                     f"z = {getattr(cc, '_z', 0) * 1e3:+.2f} mm")]))


c = FinishCell(render=True)
c.trail = True
s = Surface(C, size=(0.16, 0.10)).set_shape(wave_amp=3e-4, wave_len=0.04)
c.add_surface(s)
c.hook = Rec("T108 polishing: admittance 20 N, wavy part", ((0.45, 0.0, 0.16), 50, -30, 0.75))
seam, _ = raster_on(s, SPACING, NSTROKE, LSTROKE)
demo = [clean(c.finish(seam, FinishParams()))]
c.move_j(c.q_home)
n_pol = len(frames)
c.close()
c = FinishCell(render=True, pad=PadModel(k_e=5e5))
c.trail = False
s = deburr_surface()
c.add_surface(s, burr_vis_scale=6.0)
c.hook = Rec("T108 deburring: hard pad 20 N (burr shown x6)", ((0.45, 0.0, 0.16), 35, -28, 0.55), period=0.15)
seam = line_on(s, -0.06, 0.06, -0.0075)
for _ in range(3):
    demo.append(clean(c.finish(seam, FinishParams(speed=0.010, force=ForceParams(F_d=20.0, D=2000.0)))))
c.move_j(c.q_home)
c.close()
M["demo"] = demo
imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=20, macro_block_size=None, quality=6)
gif = [Image.fromarray(x).resize((600, 240), Image.LANCZOS) for x in frames[::2]]
pal = [g.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
pal[0].save(MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=100, loop=0, optimize=True)
for nm, idx_ in (("key_a", n_pol // 2), ("key_b", n_pol - 8), ("key_c", n_pol + (len(frames) - n_pol) // 2)):
    Image.fromarray(frames[idx_]).save(MEDIA / f"{TAG}_{nm}.png")
M["frames"] = len(frames)
M["gif_MB"] = (MEDIA / f"{TAG}.gif").stat().st_size / 1e6
M["mp4_MB"] = (MEDIA / f"{TAG}.mp4").stat().st_size / 1e6
M["total_s"] = time.time() - t_start
(DATA / f"{TAG}_metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
log("frames", len(frames), "gif MB", M["gif_MB"], "mp4 MB", M["mp4_MB"])
