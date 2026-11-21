"""T112 報告：機台上下料（CNC 自動門＋虎鉗＋PLC 互鎖、S1 依序 / S2 緩衝交換、週期模型、虎鉗捕捉、夾持力、故障恢復）
→ JSON/圖/動畫。
用法：PYTHONPATH=. python3 scripts/t112_report.py 2026-11-21 [--anim]
（實驗以 4 行程平行；TendCell 用私有暫存 URDF，不改寫 robots/ 下檔案。--anim：只重做圖與動畫（讀既有 JSON）。）
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
from sixaxis import tending as T  # noqa: E402
from sixaxis.gripper import CUBE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATE = next((a for a in sys.argv[1:] if not a.startswith("--")), time.strftime("%Y-%m-%d"))
TAG = f"{DATE}_T112"
MEDIA, DATA = ROOT / "docs/media", ROOT / "docs/test-reports/data"
DATA.mkdir(exist_ok=True)
T0 = time.time()
TMS = (4.0, 10.0, 15.0, 25.0, 40.0)
T_REF = 15.0  # 由此 T_m 的量測分量預測其他 T_m（樣本外驗證）
DXS = [x * 1e-3 for x in range(-8, 9)]
FCS = (25.0, 50.0, 100.0, 150.0, 200.0, 300.0)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def slim(r):
    if isinstance(r, dict):
        return {k: slim(v) for k, v in r.items() if k != "results"}
    if isinstance(r, (list, tuple)):
        return [slim(v) for v in r]
    if isinstance(r, np.ndarray):
        return r.tolist()
    if isinstance(r, (np.floating, np.integer)):
        return r.item()
    if isinstance(r, np.bool_):
        return bool(r)
    return r


# ====================================================================== 平行工作
def job_run(spec):
    name, kw = spec
    t = time.perf_counter()
    c = T.TendCell(T.TendConfig(**kw))
    r = c.run()
    r["terms"] = T.period_terms(r["events"], r["segments"])
    r.update(
        name=name,
        cfg=kw,
        wall_s=time.perf_counter() - t,
        motions=[{k: v for k, v in m.items() if k != "results"} for m in c.motion_log],
    )
    c.close()
    return slim(r)


def job_capture(a):
    dx, dy, chip = a
    return slim(T.vise_capture_test(dx, dy, chip=chip))


def job_hold(F):
    return slim(T.clamp_hold_test(F))


def job_interlock(_=None):
    """互鎖/光幕功能測試（同一工作站內依序執行）。"""
    c = T.TendCell(T.TendConfig(n_parts=1))
    m = c.m
    out = {}
    # 1) 門關著時啟動循環（未夾緊、無零件）→ 拒絕
    out["start_no_part"] = m.cycle_start()
    # 2) 開門、手臂進入機台、命令關門 → 拒絕
    m.open_door()
    c.wait_until(lambda: m.sig("door_open"), 5.0)
    out["door_time_meas"] = next(t for t, n, v in m.signals if n == "door_open" and v) - next(
        e["t"] for e in m.events if e["event"] == "DOOR_OPEN_CMD"
    )
    part = c.pick_tray()
    c.to_door()
    c.enter()
    out["arm_in_zone_inside"] = m.arm_in_zone()
    out["close_inside"] = m.close_door()
    # 3) 放入、夾緊後，門開著啟動 → 拒絕
    c.lin((T.X_PLACE, T.VY, T.Z_PART + 0.0095), "down")
    c.release()
    m.clamp(part.body)
    c.vise_part = part
    c.lin((T.X_PLACE, T.VY, T.Z_ENTRY), "up")
    c.wait_until(lambda: m.sig("clamped"), 3.0)
    out["start_door_open"] = m.cycle_start()
    # 4) 繞過關門互鎖（模擬 PLC 失效）且手臂停在門洞平面 → 光幕停止門
    c.lin((T.VX, T.FY + 0.02, T.Z_ENTRY), "in_curtain")
    out["arm_in_curtain"] = m.arm_in_zone("curtain")
    m._door_move(0.0)
    c.s.wait(1.0)
    out["trip"] = m.trip
    out["door_s_at_trip_mm"] = m.door_s * 1e3
    out["min_gap_door_arm_mm"] = (
        min([x[8] for x in T.pb.getClosestPoints(c.s.sim.id, m.door, 0.5, physicsClientId=c.s.cid)] + [0.5])
        * 1e3
    )
    # 5) 退出 → 正常關門、啟動；循環中開門 → 拒絕
    c.exit()
    out["arm_in_zone_out"] = m.arm_in_zone()
    out["close_outside"] = m.close_door()
    c.wait_until(lambda: m.sig("door_closed"), 5.0)
    out["start_ok"] = m.cycle_start()
    c.s.wait(1.0)
    out["open_during_cycle"] = m.open_door()
    out["events"] = [e for e in m.events]
    c.close()
    return slim(out)


def job_clearance(_=None):
    """進出門洞（P_OUT → P_IN）與虎鉗升降的 TCP 直線上，手臂＋夾持胚料對機台的最小間隙（規劃用碰撞模型取樣）。"""
    from sixaxis.graspsim import Q_REF
    from sixaxis.planning import CollisionChecker
    from sixaxis.scenarios import workcell_boxes
    from sixaxis.sorting import grasp_offset, ik_near

    # floor=False：肩部連桿對地板距離為定值（約 28 mm，與路徑無關），會遮蔽門洞間隙，故不納入
    cc = CollisionChecker([b for b in workcell_boxes() if b.name != "cube"] + T.machine_boxes("open"), floor=False)
    off = grasp_offset(T.PART_A)
    T_to = np.eye(4)
    T_to[2, 3] = off  # 物件中心在 TCP 下方 off：R_tcp = diag(1,−1,−1) → TCP 系 +z 為世界 −z
    out = {}
    for held in (False, True):
        if held:
            cc.attach([T.PART_A / 2] * 3, T_to)
        pts = [np.array(T.P_OUT) + (np.array(T.P_IN) - np.array(T.P_OUT)) * u for u in np.linspace(0, 1, 31)]
        pts += [np.array([T.X_PLACE, T.VY, z]) for z in np.linspace(T.Z_ENTRY, T.Z_PART + off + 0.0015, 16)]
        q = ik_near(T.P_OUT, 0.0, Q_REF, sym=False)
        ds, ys = [], []
        for p in pts:
            q = ik_near(tuple(p), 0.0, q, sym=False)
            ds.append(cc.distances(q)[0])
            ys.append(float(p[1]) if abs(p[0] - T.VX) < 1e-9 else np.nan)
        out["held" if held else "empty"] = {
            "min_mm": float(np.min(ds) * 1e3),
            "d_mm": (np.array(ds) * 1e3).tolist(),
            "pts": [p.tolist() for p in pts],
        }
    cc.close()
    return out


# ====================================================================== 主程式
def main():
    M: dict = {
        "params": {
            "VX": T.VX,
            "VY": T.VY,
            "FY": T.FY,
            "door": [T.DX0, T.DX1, T.SILL, T.DOOR_TOP],
            "door_stroke": T.DOOR_STROKE,
            "door_v": T.DOOR_V,
            "door_a": T.DOOR_A,
            "door_time": T.door_time(),
            "jaw_open": T.JAW_OPEN,
            "F_clamp": T.F_CLAMP,
            "mu_jaw": T.MU_JAW,
            "mu_floor": T.MU_FLOOR,
            "mu_part": CUBE.friction,
            "seat_tol": T.SEAT_TOL,
            "jaw_win": T.JAW_WIN,
            "part": [T.PART_A, CUBE.mass],
            "finished": T.finished_props(),
            "F_cut": T.F_CUT,
            "k_c": T.K_C,
            "a_p": T.A_P,
            "h": T.H_CHIP,
            "pass_t": T.PASS_T,
            "z_entry": T.Z_ENTRY,
            "p_out": T.P_OUT,
            "p_in": T.P_IN,
            "buf": T.BUF_SLOTS,
            "buf_top": T.BUF_TOP,
            "belt_drop": T.BELT_DROP,
            "v_lin": T.V_LIN,
            "T_spindle": T.T_SPINDLE,
            "tms": TMS,
            "t_ref": T_REF,
        }
    }
    specs = [(f"{s}_{tm:g}", {"strategy": s, "T_m": tm}) for tm in TMS for s in ("serial", "buffer")]
    specs += [
        ("chip_recover", {"strategy": "buffer", "T_m": 10.0, "chip_at": (1, 2)}),
        ("chip_norecover", {"strategy": "buffer", "T_m": 10.0, "chip_at": (1,), "recover": False}),
        ("empty_pocket", {"strategy": "serial", "T_m": 6.0, "empty_pockets": (1,)}),
        ("place_err_px", {"strategy": "serial", "T_m": 6.0, "n_parts": 2, "place_err": (0.004, 0.0)}),
        ("place_err_nx", {"strategy": "serial", "T_m": 6.0, "n_parts": 2, "place_err": (-0.004, 0.002)}),
    ]
    with Pool(4) as pool:
        log("capture / hold / interlock / clearance")
        cap = pool.map_async(
            job_capture,
            [(dx, 0.0, False) for dx in DXS]
            + [(0.0, dy * 1e-3, False) for dy in (-3, 3)]
            + [(0.0, 0.0, True)],
        )
        hold = pool.map_async(job_hold, FCS)
        il = pool.apply_async(job_interlock)
        cl = pool.apply_async(job_clearance)
        log(f"runs: {len(specs)}")
        runs = pool.map(job_run, specs, chunksize=1)
        M["capture"], M["hold"], M["interlock"], M["clearance"] = cap.get(), hold.get(), il.get(), cl.get()
    M["runs"] = runs
    M["model"] = model_check(runs)
    log("runs done")
    (DATA / f"{TAG}_metrics.json").write_text(
        json.dumps(slim(M), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    plots(M)
    M["anim"] = animate()
    (DATA / f"{TAG}_metrics.json").write_text(
        json.dumps(slim(M), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log("done")


def model_check(runs) -> dict:
    """以 T_m = T_REF 的量測分量（各週期序號 k 分別取）預測其他 T_m 的週期，與實測比較。"""
    by = {
        (r["cfg"]["strategy"], r["cfg"]["T_m"]): r
        for r in runs
        if r["name"].split("_")[0] in ("serial", "buffer")
    }
    out = {"rows": [], "terms_ref": {}}
    for s in ("serial", "buffer"):
        ref = by[(s, T_REF)]["terms"]
        out["terms_ref"][s] = ref
        for tm in TMS:
            r = by[(s, tm)]
            for pt in r["terms"]:
                rt = next(x for x in ref if x["k"] == pt["k"])
                pred = T.cycle_model(tm, rt["T_open"], rt["T_ex"], rt["T_close"], rt["T_side"])
                out["rows"].append(
                    {
                        "strategy": s,
                        "T_m": tm,
                        "k": pt["k"],
                        "meas": pt["period"],
                        "pred": float(pred["T_c"]),
                        "robot_bound": bool(pred["robot_bound"]),
                        "idle": pt["machine_idle"],
                        "T_side": pt["T_side"],
                        "T_ex": pt["T_ex"],
                    }
                )
    err = [abs(r["meas"] - r["pred"]) for r in out["rows"] if r["T_m"] != T_REF]
    out["max_abs_err"] = float(max(err))
    out["max_rel_err"] = float(
        max(abs(r["meas"] - r["pred"]) / r["meas"] for r in out["rows"] if r["T_m"] != T_REF)
    )
    return out


# ====================================================================== 圖
CAT = {  # 手臂活動分類（甘特圖）
    "in_machine": (
        "enter",
        "exit",
        "over_",
        "vise_",
        "unload_",
        "regrip",
        "wait_clamp",
        "wait_unclamp",
        "air_blow",
    ),
    "wait": ("wait_door", "wait_machine", "wait_close"),
    "buffer": ("to_bufA", "to_bufB", "bufA_", "bufB_"),
    "tray": ("to_tray", "tray_", "grip_tray"),
    "belt": ("to_belt", "belt_"),
    "door_approach": ("to_door",),
}
CAT_COL = {
    "in_machine": "#e8590c",
    "wait": "#ced4da",
    "buffer": "#7048e8",
    "tray": "#d6336c",
    "belt": "#1c7ed6",
    "door_approach": "#495057",
}


def cat_of(tag):
    for k, pre in CAT.items():
        if tag.startswith(pre):
            return k
    return None


def gantt(ax, r, t_max=None):
    ev = r["events"]
    st = [e["t"] for e in ev if e["event"] == "CYCLE_START"]
    dn = [e["t"] for e in ev if e["event"] == "CYCLE_DONE"]
    for a, b in zip(st, dn, strict=False):
        ax.broken_barh([(a, b - a)], (2.6, 0.8), color="#2f9e44")
    sig = r["signals"]
    # 門：關/開/移動
    door_open = [t for t, n, v in sig if n == "door_open" and v]
    door_closed = [t for t, n, v in sig if n == "door_closed" and v]
    ocmd = [e["t"] for e in ev if e["event"] == "DOOR_OPEN_CMD"]
    ccmd = [e["t"] for e in ev if e["event"] == "DOOR_CLOSE_CMD"]
    for t in ocmd:
        t1 = min([x for x in door_open if x >= t] or [t])
        ax.broken_barh([(t, t1 - t)], (1.6, 0.8), color="#fab005")
        t2 = min([x for x in ccmd if x > t1] or [t1])
        ax.broken_barh([(t1, t2 - t1)], (1.6, 0.8), color="#ffe066")
    for t in ccmd:
        t1 = min([x for x in door_closed if x >= t] or [t])
        ax.broken_barh([(t, t1 - t)], (1.6, 0.8), color="#fab005")
    for tag, a, b in r["segments"]:
        c = cat_of(tag)
        if c and b > a:
            ax.broken_barh([(a, b - a)], (0.6, 0.8), color=CAT_COL[c])
    ax.set_yticks([1.0, 2.0, 3.0], ["robot", "door", "spindle"])
    ax.set_xlim(0, t_max or r["T_total"])
    ax.grid(alpha=0.3, axis="x")


def plots(M):
    runs = {r["name"]: r for r in M["runs"]}
    # 1) 甘特圖 S1 vs S2（T_m = 15）
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True)
    tmax = max(runs["serial_15"]["T_total"], runs["buffer_15"]["T_total"])
    for a, nm, ttl in ((ax[0], "serial_15", "S1 serial"), (ax[1], "buffer_15", "S2 buffer exchange")):
        gantt(a, runs[nm], tmax)
        a.set_title(
            f"{ttl}  (T_m = 15 s, 4 parts): mean period "
            f"{np.mean(runs[nm]['periods']):.2f} s, utilization {runs[nm]['utilization'] * 100:.1f} %",
            fontsize=10,
        )
    hs = [mp.Patch(color=c, label=k) for k, c in CAT_COL.items()] + [
        mp.Patch(color="#2f9e44", label="machining"),
        mp.Patch(color="#fab005", label="door moving"),
        mp.Patch(color="#ffe066", label="door open"),
    ]
    ax[1].legend(handles=hs, ncol=9, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax[1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_gantt.png", dpi=110)
    plt.close(fig)

    # 2) 週期模型 vs 實測
    md = M["model"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    tm = np.linspace(0, 45, 181)
    cols = {"serial": "#1c7ed6", "buffer": "#e8590c"}
    for s in ("serial", "buffer"):
        ref = md["terms_ref"][s]
        rt = next(x for x in ref if x["k"] == 2)
        pr = T.cycle_model(tm, rt["T_open"], rt["T_ex"], rt["T_close"], rt["T_side"])
        lab = "S1 serial" if s == "serial" else "S2 buffer"
        ax[0].plot(tm, pr["T_c"], color=cols[s], label=f"{lab} model (k=2 terms @ T_m={T_REF:g}s)")
        ax[1].plot(tm, pr["U"] * 100, color=cols[s], label=lab)
        rows = [r for r in md["rows"] if r["strategy"] == s]
        for k, mk in ((1, "s"), (2, "o"), (3, "^")):
            rr = [r for r in rows if r["k"] == k]
            ax[0].scatter(
                [r["T_m"] for r in rr],
                [r["meas"] for r in rr],
                marker=mk,
                color=cols[s],
                s=28,
                facecolors="none" if k != 2 else cols[s],
                label=f"{lab} measured k={k}",
            )
        for nm, r in runs.items():
            if r["cfg"].get("strategy") == s and nm.startswith(s) and len(r["periods"]) > 1:
                ax[1].scatter(r["T_m"], r["T_m"] / r["periods"][1] * 100, color=cols[s], s=28)
    ax[0].plot(tm, tm, "k:", lw=0.8, label="T_c = T_m (ideal)")
    ax[0].set(
        xlabel="machining time T_m [s]",
        ylabel="cycle period T_c [s]",
        title=f"Cycle model T_c = T_open + T_ex + max(T_close+T_m, T_side)\nmax |error| = "
        f"{md['max_abs_err']:.2f} s (out-of-sample)",
    )
    ax[1].set(
        xlabel="machining time T_m [s]",
        ylabel="machine utilization U = T_m/T_c [%]",
        title="Utilization (points: measured k=2)",
    )
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_cycle_model.png", dpi=110)
    plt.close(fig)

    # 3) 虎鉗捕捉
    cap = [c for c in M["capture"] if c["dy_mm"] == 0 and c["seat_gap_mm"] < 100 and c["dx_mm"] == c["dx_mm"]]
    cap = sorted([c for c in M["capture"] if c["dy_mm"] == 0], key=lambda c: c["dx_mm"])
    cap = [c for c in cap if not (c["dx_mm"] == 0 and c["seat_gap_mm"] > 1)]  # 去掉切屑那筆
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    dx = [c["dx_mm"] for c in cap]
    ax[0].plot(dx, [c["x_err_mm"] for c in cap], "o-", label="clamped x error")
    ax[0].plot(dx, [c["z_err_mm"] for c in cap], "s--", label="clamped z error")
    ax[0].axvspan(
        -T.JAW_OPEN / 2 * 1e3, T.JAW_OPEN / 2 * 1e3, color="#2f9e44", alpha=0.1, label="jaw clearance ±6 mm"
    )
    ax[0].set(
        xlabel="placement offset dx [mm]",
        ylabel="error [mm]",
        title="Vise datum capture (x located by fixed jaw)",
    )
    ax[1].semilogy(dx, [max(c["seat_gap_mm"], 1e-3) for c in cap], "o-", label="seat gap")
    ax[1].axhline(T.SEAT_TOL * 1e3, color="r", ls="--", label="seat sensor threshold 0.3 mm")
    ax[1].plot(dx, [max(abs(c["jaw_mm"]), 1e-3) for c in cap], "s-", label="|jaw position| (window 0.5 mm)")
    ax[1].set(xlabel="placement offset dx [mm]", ylabel="[mm]", title="Clamp monitoring")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_capture.png", dpi=110)
    plt.close(fig)

    # 4) 夾持力
    H = M["hold"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fc = np.linspace(0, 320, 50)
    ax[0].plot(fc, [T.slip_threshold(f) for f in fc], "k-", label="Coulomb: 2 μ_j F_c + μ_f m g")
    ax[0].plot(
        [h["F_c"] for h in H], [h["F_slip"] for h in H], "o", color="#e8590c", label="simulated slip onset"
    )
    ax[0].axhline(T.F_CUT, color="#1c7ed6", ls="--", label=f"cutting force {T.F_CUT:.0f} N")
    ax[0].set(
        xlabel="clamp force F_c [N]", ylabel="lateral force at slip [N]", title="Workholding friction limit"
    )
    for h in H:
        tr = np.array(h["trace"])
        if len(tr):
            ax[1].plot(tr[:, 0], tr[:, 1], label=f"F_c = {h['F_c']:.0f} N")
    ax[1].set(
        xlabel="lateral force F_y [N] (ramp 20 N/s)",
        ylabel="part displacement y [mm]",
        title="Displacement under ramped lateral force",
    )
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_clamp.png", dpi=110)
    plt.close(fig)

    # 5) 門洞間隙
    cl = M["clearance"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, lab in (("empty", "arm only"), ("held", "arm + raw part")):
        ax.plot(cl[k]["d_mm"], "o-", ms=3, label=f"{lab}: min {cl[k]['min_mm']:.1f} mm")
    ax.axvline(30.5, color="k", ls=":", lw=0.8)
    ax.text(5, 5, "P_out → P_in (through door)", fontsize=8)
    ax.text(32, 5, "descend into vise", fontsize=8)
    ax.set(
        xlabel="sample along taught path",
        ylabel="distance to machine/scene [mm]",
        title="Clearance along the door-entry and vise strokes (planning collision model)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_clearance.png", dpi=110)
    plt.close(fig)

    # 6) 佈置俯視
    fig, ax = plt.subplots(figsize=(7, 7))
    for b in T.machine_boxes("closed"):
        if b.name in ("m_roof", "m_table", "spindle_col"):
            continue
        fc_ = {
            "door_closed": "#fab005",
            "buffer": "#7048e8",
            "jaw_fixed": "#c92a2a",
            "jaw_move": "#e8590c",
            "spindle": "#343a40",
        }.get(b.name, "#adb5bd")
        ax.add_patch(
            mp.Rectangle(
                (b.center[0] - b.half[0], b.center[1] - b.half[1]),
                2 * b.half[0],
                2 * b.half[1],
                fc=fc_,
                ec="k",
                lw=0.4,
                alpha=0.6,
            )
        )
    ax.add_patch(
        mp.Rectangle(
            (S.BENCH_X - S.BENCH_HX, -S.BENCH_HY),
            2 * S.BENCH_HX,
            2 * S.BENCH_HY,
            fc="#e9ecef",
            ec="k",
            lw=0.4,
        )
    )
    ax.add_patch(
        mp.Rectangle(
            (S.BELT_X0, S.BELT_Y0), S.BELT_X1 - S.BELT_X0, S.BELT_Y1 - S.BELT_Y0, fc="#343a40", alpha=0.6
        )
    )
    for x, y in S.pocket_centers():
        ax.add_patch(mp.Rectangle((x - 0.025, y - 0.025), 0.05, 0.05, fc="#d43c2c", alpha=0.7))
    for k, (x, y) in T.BUF_SLOTS.items():
        ax.text(x, y, k, ha="center", va="center", color="w", fontsize=9)
    ax.plot(*T.P_OUT[:2], "k^", label="P_out (door wait point)")
    ax.plot(*T.P_IN[:2], "kv", label="P_in (over vise)")
    ax.plot(*T.BELT_DROP, "b*", ms=10, label="belt drop")
    ax.plot(0, 0, "ko", ms=12, label="robot base")
    ax.set(
        aspect="equal",
        xlim=(-0.15, 0.9),
        ylim=(-0.9, 0.6),
        xlabel="x [m]",
        ylabel="y [m]",
        title="T112 cell layout (top view)",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_layout.png", dpi=110)
    plt.close(fig)


# ====================================================================== 動畫
def animate() -> dict:
    SIZE = (400, 300)
    frames: list = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        fsmall = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = fsmall = ImageFont.load_default()
    rec = {"t": [], "q": [], "qd": [], "tau": [], "err": []}
    state = {"last": -1.0, "cell": None, "phase": ""}

    def lab(img, left, right, col=(150, 190, 255)):
        im_ = Image.fromarray(img)
        d_ = ImageDraw.Draw(im_)
        d_.rectangle([0, 0, im_.width, 24], fill=(20, 24, 32))
        d_.text((8, 4), left, fill=(235, 238, 242), font=font)
        d_.text((im_.width - 8 - d_.textlength(right, font=font), 4), right, fill=col, font=font)
        return np.asarray(im_)

    def status(cell, t):
        im = Image.new("RGB", (300, 300), (24, 28, 36))
        d = ImageDraw.Draw(im)
        m = cell.m
        d.text((10, 6), "PLC / cell status", fill=(255, 220, 120), font=font)
        sigs = [
            ("door_open", m.sig("door_open")),
            ("door_closed", m.sig("door_closed")),
            ("clamped", m.sig("clamped")),
            ("cycle_busy", m.sig("cycle_busy")),
            ("cycle_done", m.sig("cycle_done") or m.done),
            ("arm_in_zone", m.arm_in_zone()),
        ]
        for i, (n, v) in enumerate(sigs):
            y = 32 + 20 * i
            d.ellipse([12, y + 2, 24, y + 14], fill=(60, 220, 90) if v else (90, 95, 105))
            d.text((32, y), n, fill=(230, 232, 236), font=fsmall)
        d.text((160, 32), f"door {m.door_s * 1e3:5.0f} mm", fill=(230, 232, 236), font=fsmall)
        d.text((160, 52), f"jaw  {m.jaw_pos() * 1e3:6.2f} mm", fill=(230, 232, 236), font=fsmall)
        if cell.vise_part is not None:
            d.text((160, 72), f"seat gap {m.seat_gap() * 1e3:6.2f} mm", fill=(230, 232, 236), font=fsmall)
        tc = (t - m.cycle) if m.cycle is not None else None
        d.text(
            (160, 92),
            f"cycle {tc:5.1f}/{m.T_m:.0f} s" if tc is not None else "cycle  —",
            fill=(230, 232, 236),
            font=fsmall,
        )
        d.text(
            (160, 112),
            f"done {len([e for e in m.events if e['event'] == 'CYCLE_DONE'])}  out {len(cell.out_parts)}",
            fill=(230, 232, 236),
            font=fsmall,
        )
        # 最近 40 s 時間線
        y0 = 178
        d.text(
            (10, y0 - 18), "last 40 s:  spindle / door / robot-in-machine", fill=(180, 185, 195), font=fsmall
        )
        t0 = t - 40

        def X(tt):
            return 10 + (tt - t0) / 40 * 280

        ev = m.events
        st = [e["t"] for e in ev if e["event"] == "CYCLE_START"]
        dn = [e["t"] for e in ev if e["event"] == "CYCLE_DONE"] + ([t] if m.cycle is not None else [])
        for a, b in zip(st, dn, strict=False):
            if b > t0:
                d.rectangle([X(max(a, t0)), y0, X(b), y0 + 14], fill=(47, 158, 68))
        for tag, a, b in cell.log:
            if b > t0 and cat_of(tag) == "in_machine":
                d.rectangle([X(max(a, t0)), y0 + 40, X(b), y0 + 54], fill=(232, 89, 12))
        sg = [(tt, n, v) for tt, n, v in m.signals if n in ("door_open", "door_closed")]
        for tt, n, v in sg:
            if tt > t0:
                d.line([X(tt), y0 + 20, X(tt), y0 + 34], fill=(250, 176, 5), width=2)
        flt = [f for f in cell.faults if f["fault"] != "CHIP_INJECTED"]
        y = y0 + 64
        for f in flt[-3:]:
            d.text(
                (10, y), f"t={f['t']:.1f}s {f['fault']} → {f['action']}", fill=(255, 120, 100), font=fsmall
            )
            y += 16
        d.text((10, 280), state["phase"][:42], fill=(150, 190, 255), font=fsmall)
        return np.asarray(im)

    def views(sim):
        v0 = sim.render(target=(0.30, -0.28, 0.22), eye=(-0.50, 0.75, 1.0), size=SIZE, fov=45, yaw=30)
        v1 = sim.render(target=(T.VX, T.VY, 0.17), eye=(T.VX + 0.12, -0.79, 0.42), size=SIZE, fov=60, yaw=180)
        return v0, v1

    def cb(sc):
        q, qd = sc.sim.get_state()
        rec["t"].append(sc.t)
        rec["q"].append(q.tolist())
        rec["qd"].append(qd.tolist())
        rec["tau"].append(state.get("tau", [0.0] * 6))
        rec["err"].append(float(np.max(np.abs(sc._q_cmd - q))))
        if sc.t - state["last"] < 0.2 - 1e-9:
            return
        state["last"] = sc.t
        cell = state["cell"]
        v0, v1 = views(sc.sim)
        frames.append(
            np.hstack(
                [
                    lab(v0, "T112 machine tending (S2 buffer)", f"t = {sc.t:5.1f} s"),
                    lab(v1, "inside machine: vise", state["phase"][:22]),
                    status(cell, sc.t),
                ]
            )
        )

    cfg = T.TendConfig(strategy="buffer", T_m=8.0, n_parts=3, chip_at=(1,))
    cell = T.TendCell(cfg, render=True, frame_cb=cb, cb_every=10)
    state["cell"] = cell
    s = cell.s
    orig_torque = s._arm_torque

    def tau_rec():
        tau = orig_torque()
        state["tau"] = tau.tolist()
        return tau

    s._arm_torque = tau_rec
    orig_lin, orig_goto = cell.lin, cell.goto

    def lin(xyz, tag):
        state["phase"] = tag
        return orig_lin(xyz, tag)

    def goto(q, tag):
        state["phase"] = tag
        return orig_goto(q, tag)

    def wait_until(cond, timeout=60.0, tag=None):  # 等待期間也要出畫面
        t0 = cell.t
        if tag:
            state["phase"] = tag
        while not cond():
            if cell.t - t0 > timeout:
                return False
            s.step(10)
            cb(s)
        cell.ex.t = cell.t
        if tag and cell.t - t0 > 1e-9:
            cell.seg(tag, t0)
        return True

    cell.lin, cell.goto, cell.wait_until = lin, goto, wait_until
    r = cell.run()
    for _ in range(10):
        frames.append(frames[-1])
    imageio.mimsave(MEDIA / f"{TAG}.mp4", frames, fps=10, macro_block_size=None, quality=6)
    gif = [Image.fromarray(x).resize((660, 180), Image.LANCZOS) for x in frames[::2]]
    pal = [g.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for g in gif]
    pal[0].save(
        MEDIA / f"{TAG}.gif", save_all=True, append_images=pal[1:], duration=200, loop=0, optimize=True
    )
    nf = len(frames)
    # 關鍵畫面：進入機台上料、切屑恢復、加工中
    t_frames = np.arange(nf) * 0.2

    def at(t):
        return int(min(nf - 1, max(0, np.searchsorted(t_frames, t))))

    flt = [f for f in r["faults"] if f["fault"] == "NOT_SEATED"]
    keys = {
        "key_load": r["loads"][0]["t0"] + 3.0,
        "key_cycle": r["starts"][1] + 4.0 if len(r["starts"]) > 1 else 30,
        "key_recover": flt[0]["t"] + 1.0 if flt else 50,
        "key_end": 1e9,
    }
    for nm, tt in keys.items():
        Image.fromarray(frames[at(tt)]).save(MEDIA / f"{TAG}_{nm}.png")
    curves(rec, cell, r)
    out = slim({k: v for k, v in r.items() if k not in ("signals",)})
    out.update(
        n_frames=nf,
        max_track_err_mrad=float(np.max(rec["err"]) * 1e3) if rec["err"] else None,
        terms=T.period_terms(r["events"], r["segments"]),
    )
    cell.close()
    return out


def curves(rec, cell, r):
    t = np.array(rec["t"])
    q = np.unwrap(np.array(rec["q"]), axis=0)
    qd = np.array(rec["qd"])
    tau = np.array(rec["tau"])
    lim = cell.s.r.tau_max
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for j in range(6):
        ax[0].plot(t, q[:, j], label=f"q{j + 1}")
        ax[1].plot(t, qd[:, j])
        ax[2].plot(t, tau[:, j] / lim[j])
    ax[3].semilogy(t, np.maximum(np.array(rec["err"]) * 1e3, 1e-4), color="k")
    st, dn = r["starts"], r["dones"]
    for a, b in zip(st, dn, strict=False):
        for x in ax:
            x.axvspan(a, b, color="#2f9e44", alpha=0.08)
    for f in r["faults"]:
        if f["fault"] == "NOT_SEATED":
            for x in ax:
                x.axvline(f["t"], color="r", ls="--", lw=0.8)
    ax[0].legend(ncol=6, fontsize=8)
    ax[0].set(ylabel="q [rad]", title="Joint trajectories (green = machining, red = NOT_SEATED detected)")
    ax[1].set(ylabel="qd [rad/s]")
    ax[2].set(ylabel="τ / τ_max")
    ax[3].set(ylabel="max |q_cmd − q| [mrad]", xlabel="t [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA / f"{TAG}_joints.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if "--anim" in sys.argv:
        M = json.loads((DATA / f"{TAG}_metrics.json").read_text(encoding="utf-8"))
        plots(M)
        M["anim"] = animate()
        (DATA / f"{TAG}_metrics.json").write_text(
            json.dumps(slim(M), ensure_ascii=False, indent=1), encoding="utf-8"
        )
    else:
        main()
