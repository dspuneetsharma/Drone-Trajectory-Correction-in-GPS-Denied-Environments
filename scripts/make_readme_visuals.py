#!/usr/bin/env python3
"""README visuals for the ML-corrected VINS trajectory (MH_05 test flight). Reads saved results only.
Output: assets/readme/*.png + path_progress_mh05.gif
Usage: python scripts/make_readme_visuals.py"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import animation

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "assets", "readme"); os.makedirs(OUT, exist_ok=True)
SEQ = "MH_05_difficult"
GT_C, VINS_C, ML_C = "#000000", "#FF1744", "#2979FF"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "axes.titleweight": "bold", "axes.titlesize": 14, "legend.frameon": True, "legend.framealpha": 0.95})

T = pd.read_csv(os.path.join(ROOT, "outputs", "ml_correction_phase2", f"{SEQ}_trajectories_TEST.csv"))
gp = os.path.join(ROOT, "Data", "machine_hall", SEQ, SEQ, "mav0", "state_groundtruth_estimate0", "data.csv")
gt_t = np.loadtxt(gp, delimiter=",", comments="#", usecols=0, dtype=np.int64); G = np.loadtxt(gp, delimiter=",", comments="#", usecols=(1, 2, 3))
t = T.timestamp_ns.values
i = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1); i = np.where(np.abs(gt_t[i - 1] - t) < np.abs(gt_t[i] - t), i - 1, i); ok = np.abs(gt_t[i] - t) < 10_000_000
B = G[i[ok]]; ts = T.t_s.values[ok]


def align(A, B):   # SE3 (rotation + translation, no scale) alignment, same as the evaluation
    ma, mb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    return (R @ A.T).T + (mb - R @ ma)


V = align(T[["VINS_x", "VINS_y", "VINS_z"]].values[ok], B); M = align(T[["selected_x", "selected_y", "selected_z"]].values[ok], B)
eV, eM = np.linalg.norm(V - B, axis=1), np.linalg.norm(M - B, axis=1)
ateV, ateM = np.sqrt(np.mean(eV ** 2)), np.sqrt(np.mean(eM ** 2))
print(f"ATE VINS {ateV:.3f} m, ML {ateM:.3f} m")
LG, LV, LM = "Ground truth", f"Basic VINS (ATE {ateV:.2f} m)", f"ML-corrected VINS (ATE {ateM:.2f} m)"


def draw_paths(ax, show=("gt", "vins", "ml"), lab=True):
    if "gt" in show: ax.plot(B[:, 0], B[:, 1], color=GT_C, lw=4.5, solid_capstyle="round", label=LG if lab else None, zorder=1)
    if "vins" in show: ax.plot(V[:, 0], V[:, 1], color=VINS_C, lw=2.2, ls=(0, (6, 3)), label=LV if lab else None, zorder=2)
    if "ml" in show: ax.plot(M[:, 0], M[:, 1], color=ML_C, lw=2.2, label=LM if lab else None, zorder=3)
    ax.set_aspect("equal", "datalim"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


# 1. main path comparison with zoom inset where basic VINS is furthest from the truth
fig, ax = plt.subplots(figsize=(10, 8.5)); draw_paths(ax)
k = int(np.argmax(eV)); c = B[k]; r = 1.3
axin = ax.inset_axes([0.64, 0.62, 0.34, 0.34]); draw_paths(axin, lab=False)
axin.set_aspect("equal", adjustable="box"); axin.set_xlim(c[0] - r, c[0] + r); axin.set_ylim(c[1] - r, c[1] + r)
axin.set_xticks([]); axin.set_yticks([]); axin.set_xlabel(""); axin.set_ylabel(""); axin.grid(False)
for sp in axin.spines.values(): sp.set_visible(True); sp.set_color("#555"); sp.set_linewidth(1.2)
axin.set_title("zoom: largest VINS drift", fontsize=10); ax.indicate_inset_zoom(axin, edgecolor="#555")
ax.legend(loc="upper left", fontsize=11); ax.set_title("Unseen test flight (EuRoC MH_05): estimated drone path, top view")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "path_comparison_mh05.png"), dpi=150); plt.close(fig)

# 2. ATE bar
fig, ax = plt.subplots(figsize=(6.5, 4.6))
b = ax.bar(["Basic VINS", "ML-corrected VINS"], [ateV, ateM], color=[VINS_C, ML_C], width=0.55)
ax.bar_label(b, labels=[f"{ateV:.2f} m", f"{ateM:.2f} m"], fontsize=13, fontweight="bold", padding=3)
ax.set_ylabel("ATE [m]  (lower is better)"); ax.set_ylim(0, ateV * 1.25); ax.grid(axis="x", visible=False)
ax.set_title(f"Average path error: {100 * (1 - ateM / ateV):.0f}% lower"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "ate_bar.png"), dpi=150); plt.close(fig)

# 3. error over time
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(ts, eV, color=VINS_C, lw=2, ls=(0, (6, 3)), label="Basic VINS"); ax.plot(ts, eM, color=ML_C, lw=2, label="ML-corrected VINS")
ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]"); ax.legend(loc="upper left"); ax.set_title("Position error during the test flight")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "error_over_time_mh05.png"), dpi=150); plt.close(fig)

# 4. side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
draw_paths(axs[0], ("gt", "vins")); axs[0].legend(loc="upper left", fontsize=10); axs[0].set_title("Basic VINS vs ground truth")
draw_paths(axs[1], ("gt", "ml")); axs[1].legend(loc="upper left", fontsize=10); axs[1].set_title("ML-corrected VINS vs ground truth")
lim = (min(axs[0].get_xlim()[0], axs[1].get_xlim()[0]), max(axs[0].get_xlim()[1], axs[1].get_xlim()[1]))
for a in axs: a.set_xlim(*lim)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "path_vins_vs_ml.png"), dpi=130); plt.close(fig)

# 5. GIF: path progression
N = 150; idx = np.linspace(1, len(B) - 1, N).astype(int)
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=80)
ax.plot(B[:, 0], B[:, 1], color=GT_C, lw=4, alpha=0.18, zorder=0)
lg, = ax.plot([], [], color=GT_C, lw=4, label="Ground truth", zorder=1)
lv, = ax.plot([], [], color=VINS_C, lw=2, ls=(0, (6, 3)), label="Basic VINS", zorder=2)
lm, = ax.plot([], [], color=ML_C, lw=2, label="ML-corrected VINS", zorder=3)
dg, = ax.plot([], [], "o", color=GT_C, ms=9, mec="white", zorder=4); dv, = ax.plot([], [], "o", color=VINS_C, ms=8, mec="white", zorder=5); dm, = ax.plot([], [], "o", color=ML_C, ms=8, mec="white", zorder=6)
txt = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
ax.set_aspect("equal", "datalim"); ax.set_xlim(B[:, 0].min() - 1, B[:, 0].max() + 1); ax.set_ylim(B[:, 1].min() - 1, B[:, 1].max() + 1)
ax.legend(loc="upper left", fontsize=10); ax.set_title("EuRoC MH_05: path drawn over time", fontsize=13); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()


def upd(f):
    j = idx[f]
    lg.set_data(B[:j, 0], B[:j, 1]); lv.set_data(V[:j, 0], V[:j, 1]); lm.set_data(M[:j, 0], M[:j, 1])
    dg.set_data([B[j, 0]], [B[j, 1]]); dv.set_data([V[j, 0]], [V[j, 1]]); dm.set_data([M[j, 0]], [M[j, 1]])
    txt.set_text(f"t = {ts[j]:5.1f} s\nerror now:  VINS {eV[j]:.2f} m  |  ML {eM[j]:.2f} m"); return lg, lv, lm, dg, dv, dm, txt


anim = animation.FuncAnimation(fig, upd, frames=N, blit=False)
anim.save(os.path.join(OUT, "path_progress_mh05.gif"), writer=animation.PillowWriter(fps=12)); plt.close(fig)
for f in sorted(os.listdir(OUT)): print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")
