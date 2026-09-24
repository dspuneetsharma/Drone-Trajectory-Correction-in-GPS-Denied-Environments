"""Phase-2 plots, trajectory CSVs, report and experiment-log update (development flights only)."""
import os, json, time, re, numpy as np, pandas as pd
from p2_common import *

C = dict(GT="#3a3a3a", VINS="#e34948", fixed="#eda100", current="#4a3aa7", selected="#2a78d6", oracle="#1baf7a")
LBL = dict(VINS="VINS (unchanged)", fixed="VINS + fixed scale", current="Phase-1 ML (HGB scale, IMU+VINS)", selected="Phase-2 selected ML")
BACK_C = {"hgb": "#2a78d6", "lgbm": "#eb6834", "xgb": "#1baf7a", "cat": "#eda100", "gru": "#e87ba4", "tcn": "#4a3aa7"}
FOLD_ORDER = ["MH_03_medium"] + TRAIN
SHORT = {f: f[:5] for f in FOLD_ORDER}
REPORT = P("docs", "ML_CORRECTION_PHASE2_REPORT.md"); LOG = P("docs", "experiments", "EXPERIMENT_LOG.md"); CSV = P("docs", "experiments", "experiment_results.csv")
ASSETS = P("assets", "ml_correction_phase2"); OUTT = P("outputs", "ml_correction_phase2"); RES = P2("results")


def _plt():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e8e8", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
    return plt


def cand_name(r): return f"{r['target']} · {r['fs']} · {r['backend']}-{r['cfg']} · λ={r['lam']:g}"


def fold_preds(r, fold):
    return np.load(P2("cache", "preds", f"{r['target']}__{r['fs']}__{r['backend']}__{r['cfg']}__{fold}.npz"))


def methods_for(D, S, sel):
    """dp corrections per fold for VINS / fixed / current / selected"""
    import p2_run as R
    B = R.baselines(D); cur = S[(S.target == "T1") & (S.fs == "IV_p1") & (S.backend == "hgb") & (S.cfg == "A") & (S.lam == 1.0)]
    out = {}
    for f in FOLD_ORDER:
        df = D[f]; sfix = B[f]["sfix"]; m = {"VINS": apply_correction(df, "baseline_vins", None, sfix, 1), "fixed": apply_correction(df, "baseline_fixed", None, sfix, 1)}
        if len(cur): m["current"] = apply_correction(df, "T1", fold_preds(cur.iloc[0], f)["pred"], sfix, 1.0)
        if sel is not None:
            r = S.loc[sel]; m["selected"] = apply_correction(df, r.target, fold_preds(r, f)["pred"], sfix, float(r.lam))
        out[f] = m
    return out


def metrics_table(D, MTH):
    rows = []
    for f in FOLD_ORDER:
        df = D[f]; fx = None
        for k, dpc in MTH[f].items():
            r = evaluate(df, integrate(df, dpc)); rows.append(dict(flight=f, method=k, **r))
    T = pd.DataFrame(rows)
    ref = T[T.method == "fixed"].set_index("flight").ate; vi = T[T.method == "VINS"].set_index("flight").ate
    T["impr_vs_vins_pct"] = [100 * (1 - r.ate / vi[r.flight]) for r in T.itertuples()]
    T["impr_vs_fixed_pct"] = [100 * (1 - r.ate / ref[r.flight]) for r in T.itertuples()]
    return T


def write_trajectories(D, MTH):
    for f in FOLD_ORDER:
        df = D[f]; T = df[["sequence", "timestamp_ns", "t_s"]].copy()
        for k, dpc in MTH[f].items():
            Pc = integrate(df, dpc); T[f"{k}_x"], T[f"{k}_y"], T[f"{k}_z"] = Pc[:, 0], Pc[:, 1], Pc[:, 2]
        T.to_csv(os.path.join(OUTT, f"{f}_trajectories_dev.csv"), index=False, float_format="%.6f")


# ------------------------------------------------------------------ plots
def plot_oracle(O):
    plt = _plt(); A = O[O.component == "all"]; fig, ax = plt.subplots(figsize=(12, 4.8)); x = np.arange(len(FOLD_ORDER))
    items = [("fixed", None)] + [(t, t) for t in ["T1", "T2", "T3", "T4"]]; w = 0.16
    cols = {"fixed": C["fixed"], "T1": "#4a3aa7", "T2": "#e87ba4", "T3": "#2a78d6", "T4": "#1baf7a"}
    for i, (name, t) in enumerate(items):
        vals = [A[(A.fold == f)].fixed_ate.iloc[0] if t is None else A[(A.fold == f) & (A.target == t)].ate.iloc[0] for f in FOLD_ORDER]
        b = ax.bar(x + (i - 2) * w, vals, w * 0.9, color=cols[name], label="fixed scale (deployable)" if t is None else f"{t} oracle")
        ax.bar_label(b, fmt="%.3f", fontsize=7.5, padding=1)
    ax.set_xticks(x, FOLD_ORDER); ax.set_ylabel("ATE RMSE [m]"); ax.legend(ncol=5, loc="upper left", fontsize=9); ax.set_ylim(0, A.fixed_ate.max() * 1.3)
    ax.set_title("Step 0 - ORACLE ceilings: perfect labels (centred future GT) - NOT deployable, upper bounds only")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "step0_oracle_ceilings.png"), dpi=140); plt.close(fig)


def plot_stageA(S):
    plt = _plt(); A = S[S.backend == "hgb"]; A = A[A.fs != "IV_p1"]
    targets = [t for t in ["T1", "T2", "T3", "T4"] if t in set(A.target)]; fss = ["IV", "IVT", "ALL"]
    M = np.full((len(targets), 3), np.nan); ann = [["" for _ in fss] for _ in targets]
    for i, t in enumerate(targets):
        for j, fs in enumerate(fss):
            g = A[(A.target == t) & (A.fs == fs)]
            if len(g): r = g.sort_values("mean_gain").iloc[-1]; M[i, j] = r.mean_gain; ann[i][j] = f"{r.mean_gain:+.1f}%\nwins {r.wins}/4, worst {r.min_gain:+.1f}%"
    fig, ax = plt.subplots(figsize=(8.5, 1.3 + 1.1 * len(targets))); lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("div", ["#e34948", "#f2f2f0", "#2a78d6"])
    ax.imshow(M, cmap=cm, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(targets)):
        for j in range(3): ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3), ["IMU+VINS", "IMU+VINS+tracking", "all causal"]); ax.set_yticks(range(len(targets)), targets); ax.grid(False)
    ax.set_title("Step A (HistGradientBoosting): mean ATE gain vs fixed scale over 4 dev evaluations"); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "stepA_target_x_features.png"), dpi=140); plt.close(fig)


def plot_backends(S, combos):
    plt = _plt(); fig, ax = plt.subplots(figsize=(12, 4.6)); x = np.arange(len(combos)); bes = ["hgb", "lgbm", "xgb", "cat"]; w = 0.2
    for i, be in enumerate(bes):
        vals, mins = [], []
        for (t, fs) in combos:
            g = S[(S.target == t) & (S.fs == fs) & (S.backend == be)]
            r = g.sort_values("mean_gain").iloc[-1] if len(g) else None; vals.append(np.nan if r is None else r.mean_gain); mins.append(np.nan if r is None else r.min_gain)
        b = ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=BACK_C[be], label={"hgb": "HistGradientBoosting", "lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost"}[be])
        ax.bar_label(b, fmt="%+.1f", fontsize=8, padding=1); ax.scatter(x + (i - 1.5) * w, mins, color="#3a3a3a", marker="_", s=120, zorder=3)
    ax.axhline(0, color="#3a3a3a", lw=0.8); ax.set_xticks(x, [f"{t} · {fs}" for t, fs in combos]); ax.set_ylabel("mean ATE gain vs fixed [%]  (dash = worst flight)")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 1.25); ax.legend(ncol=4, loc="upper right"); ax.set_title("Step B: tree libraries on the best target/feature combinations (best of 2 configs × 2 λ each)")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "stepB_tree_libraries.png"), dpi=140); plt.close(fig)


def plot_temporal(S, tnote):
    if tnote.get("status") != "ran": return
    plt = _plt(); fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    sc = tnote.get("screen", {}); names = list(sc)
    axs[0].bar(range(len(names)), [sc[n] for n in names], color=[BACK_C[n[:3]] for n in names]); axs[0].axhline(tnote["best_tree_mh03_gain"], color=C["selected"], ls="--", label="best tree (same target)")
    axs[0].axhline(tnote["best_tree_mh03_gain"] - 2, color="#999", ls=":", label="stop threshold (-2 pp)")
    axs[0].set_xticks(range(len(names)), [n.replace("gru", "GRU ").replace("tcn", "TCN ") for n in names]); axs[0].set_ylabel("MH_03 ATE gain vs fixed [%]"); axs[0].legend(loc="upper left")
    axs[0].set_ylim(0, tnote["best_tree_mh03_gain"] * 1.3); axs[0].set_title("Screening (1 seed, MH_03, causal history 1/3/5 s)")
    T = S[S.backend.str.startswith(("gru", "tcn"))]; E = S[~S.backend.str.startswith(("gru", "tcn"))]
    if T.empty:
        axs[1].axis("off"); axs[1].text(0.02, 0.6, "No GRU/TCN branch passed the screening rule\n(best window within 2 pp of the best tree on MH_03),\nso no full 3-seed / 4-fold evaluation was run.\n\n"
                                        f"Best tree on the same target: MH_03 gain {tnote['best_tree_mh03_gain']:+.1f}%", fontsize=11, va="center")
        fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "stepC_temporal.png"), dpi=140); plt.close(fig); return
    rows = [("best tree", E.sort_values("mean_gain").iloc[-1])] + [(r.backend, r) for _, r in T.sort_values("mean_gain", ascending=False).drop_duplicates("backend").iterrows()]
    axs[1].bar(range(len(rows)), [r.mean_gain for _, r in rows], color=[C["selected"]] + [BACK_C[n[:3]] for n, _ in rows[1:]])
    axs[1].scatter(range(len(rows)), [r.min_gain for _, r in rows], color="#3a3a3a", marker="_", s=150, zorder=3)
    axs[1].set_xticks(range(len(rows)), [n for n, _ in rows]); axs[1].set_ylabel("mean gain vs fixed [%] (dash = worst)"); axs[1].set_title("Full evaluation (3 seeds, 4 dev evaluations)")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "stepC_temporal.png"), dpi=140); plt.close(fig)
    import glob
    cur = sorted(glob.glob(P2("cache", "curves", "*__full__MH_03_medium.json")))
    if cur:
        fig, axs = plt.subplots(1, len(cur), figsize=(6 * len(cur), 3.8), squeeze=False)
        for ax, p in zip(axs[0], cur):
            for c in json.load(open(p)):
                e = [x["epoch"] for x in c["curve"]]; ax.plot(e, [x["train"] for x in c["curve"]], color="#999", lw=1); ax.plot(e, [x["holdout"] for x in c["curve"]], lw=1.5, label=f"seed {c['seed']} holdout")
            ax.set_title(os.path.basename(p).split("__")[2] + " training curves (MH_03 fold)"); ax.set_xlabel("epoch"); ax.set_ylabel("weighted MSE (std. units)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "stepC_training_curves.png"), dpi=130); plt.close(fig)


def plot_per_flight(T):
    plt = _plt(); ms = [m for m in ["VINS", "fixed", "current", "selected"] if m in set(T.method)]
    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(FOLD_ORDER)); w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        v = [T[(T.flight == f) & (T.method == m)].ate.iloc[0] for f in FOLD_ORDER]; b = ax.bar(x + (i - (len(ms) - 1) / 2) * w, v, w * 0.92, color=C[m], label=LBL[m])
        ax.bar_label(b, fmt="%.3f", fontsize=8.5, padding=1)
    ax.set_xticks(x, [f"{f}\n({'validation' if f == 'MH_03_medium' else 'held-out fold'})" for f in FOLD_ORDER]); ax.set_ylabel("ATE RMSE [m]")
    ax.set_ylim(0, T.ate.max() * 1.25); ax.legend(ncol=4, loc="upper left"); ax.set_title("Development ATE - every number out-of-sample (MH_03 trained on MH_01/02/04; others leave-one-flight-out)")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "dev_ate_per_flight.png"), dpi=140); plt.close(fig)


def plot_drift(T):
    plt = _plt(); ms = [m for m in ["VINS", "fixed", "current", "selected"] if m in set(T.method)]
    keys = [("rpe1s", "1 s drift [cm]", 100), ("rpe3s", "3 s drift [cm]", 100), ("rpe5s", "5 s drift [cm]", 100),
            ("along_pct", "along-track [% dist]", 1), ("lateral_pct", "lateral [% dist]", 1), ("vertical_pct", "vertical [% dist]", 1)]
    fig, axs = plt.subplots(2, 3, figsize=(16, 7.5)); x = np.arange(len(FOLD_ORDER)); w = 0.8 / len(ms)
    for ax, (k, lab, sc) in zip(axs.flat, keys):
        for i, m in enumerate(ms):
            ax.bar(x + (i - (len(ms) - 1) / 2) * w, [sc * T[(T.flight == f) & (T.method == m)][k].iloc[0] for f in FOLD_ORDER], w * 0.92, color=C[m], label=LBL[m])
        ax.axhline(0, color="#3a3a3a", lw=0.6); ax.set_xticks(x, [SHORT[f] for f in FOLD_ORDER]); ax.set_title(lab)
    axs[0, 0].legend(fontsize=8.5); fig.suptitle("Relative drift and 3 s drift decomposition (lower |value| is better)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "dev_drift_metrics.png"), dpi=140); plt.close(fig)


def plot_trajectories(D, MTH, S, sel):
    plt = _plt()
    for f in FOLD_ORDER:
        df = D[f]; gt_t, gp, _, _ = load_gt(f); t = df.timestamp_ns.values
        i = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1); i = np.where(np.abs(gt_t[i - 1] - t) < np.abs(gt_t[i] - t), i - 1, i); ok = np.abs(gt_t[i] - t) < 10_000_000
        B = gp[i[ok]]; tt = df.t_s.values[ok]
        fig = plt.figure(figsize=(16, 8.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4]); a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
        a0.plot(B[:, 0], B[:, 1], color=C["GT"], lw=3, label="ground truth")
        for m, dpc in MTH[f].items():
            A = integrate(df, dpc)[ok]; _, R, tr = umeyama(A, B); Aa = (R @ A.T).T + tr; E = np.linalg.norm(Aa - B, axis=1)
            a0.plot(Aa[:, 0], Aa[:, 1], color=C[m], lw=1.2, label=f"{LBL[m]}  ATE {np.sqrt(np.mean(E ** 2)):.3f} m"); a1.plot(tt, E, color=C[m], lw=1.2, label=LBL[m])
        a0.set_aspect("equal", "datalim"); a0.legend(fontsize=8.5, loc="best"); a0.set_xlabel("x [m]"); a0.set_ylabel("y [m]"); a0.set_title(f"{f}: top view (SE3-aligned)")
        a1.set_ylabel("position error [m]"); a1.legend(fontsize=8.5); a1.set_title("Error over time")
        if sel is not None:
            r = S.loc[sel]; pr = fold_preds(r, f)["pred"] * float(r.lam)
            if r.target == "T1": a2.plot(df.t_s, pr[:, 0], color=C["selected"], lw=1); a2.set_ylabel("scale applied")
            else:
                for j, (c, col) in enumerate(zip(TARGETS[r.target]["comps"], ["#2a78d6", "#eb6834", "#1baf7a"])):
                    a2.plot(df.t_s, 100 * pr[:, j], color=col, lw=0.8, label={"f": "forward", "l": "lateral", "u": "vertical"}[c])
                a2.set_ylabel("predicted residual [cm/s]"); a2.legend(fontsize=8.5, ncol=3)
            a2.set_title(f"Correction applied by the selected model ({cand_name(r)}) - causal, no GT at runtime")
        a2.set_xlabel("time [s]")
        how = "validation (trained on MH_01/02/04)" if f == "MH_03_medium" else "held-out fold (trained on the other 3 development flights)"
        fig.suptitle(f"Ground truth vs VINS vs fixed scale vs ML - {how}", x=0.01, ha="left", fontweight="bold"); fig.tight_layout()
        fig.savefig(os.path.join(ASSETS, f"trajectory_{f}.png"), dpi=125); plt.close(fig)


# ------------------------------------------------------------------ report
def fmt(x, n=3): return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def best_row(g):
    if g is None or len(g) == 0: return None
    e = g[g.eligible]; return (e if len(e) else g).sort_values("mean_gain").iloc[-1]


def make_all(D, FS, S, sel, L, O, tnote, frozen, info, combos, valid_targets):
    MTH = methods_for(D, S, sel); T = metrics_table(D, MTH); T.to_csv(os.path.join(RES, "dev_metrics_methods.csv"), index=False)
    write_trajectories(D, MTH)
    plot_oracle(O); plot_stageA(S); plot_backends(S, combos); plot_temporal(S, tnote); plot_per_flight(T); plot_drift(T); plot_trajectories(D, MTH, S, sel)
    txt = report_text(S, sel, T, O, tnote, frozen, info, combos, valid_targets, FS)
    open(REPORT, "w", encoding="utf-8").write(txt)
    update_log(S, sel, T, tnote, frozen, combos); update_csv(T, S, sel)


def report_text(S, sel, T, O, tnote, frozen, info, combos, valid_targets, FS):
    trees = S[~S.backend.str.startswith(("gru", "tcn"))]; temp = S[S.backend.str.startswith(("gru", "tcn"))]
    r_sel = None if sel is None else S.loc[sel]
    cur = S[(S.target == "T1") & (S.fs == "IV_p1") & (S.backend == "hgb") & (S.cfg == "A") & (S.lam == 1.0)]; cur = cur.iloc[0] if len(cur) else None
    per_t = {t: best_row(trees[(trees.target == t) & (trees.fs != "IV_p1")]) for t in ["T1", "T2", "T3", "T4"]}
    per_fs = {fs: best_row(trees[trees.fs == fs]) for fs in ["IV", "IVT", "ALL"]}
    per_be = {}
    for be in ["hgb", "lgbm", "xgb", "cat"]:
        g = trees[(trees.backend == be) & trees.apply(lambda r: (r.target, r.fs) in [tuple(c) for c in combos], axis=1)]
        per_be[be] = best_row(g) if len(g) else None
    ran_be = [b for b in per_be if per_be[b] is not None]
    best_t = max([t for t in per_t if per_t[t] is not None], key=lambda t: per_t[t].mean_gain)
    best_tree = best_row(trees[trees.fs != "IV_p1"])
    best_temp = best_row(temp) if len(temp) else None
    Oa = O[O.component == "all"]
    L = []; A = L.append
    A("# ML correction – Phase 2 report (development; pre-final-test)")
    A(f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `ml/correction_phase2/src/p2_run.py` on {info.get('platform','?')} "
      f"(python {info.get('python')}, sklearn {info.get('sklearn')}, lightgbm {info.get('lightgbm')}, xgboost {info.get('xgboost')}, catboost {info.get('catboost')}, "
      f"torch {info.get('torch')}, GPU: {info.get('gpu', 'none')}).*  **MH_05 was not built, loaded or evaluated for anything in this report.**\n")
    A("## Answers")
    A(f"1. **Best correction target:** **{best_t}** – {TARGETS[best_t]['desc']}. Best deployable mean ATE gain over fixed scale: " +
      ", ".join(f"{t} {per_t[t].mean_gain:+.1f}%" for t in per_t if per_t[t] is not None) + " (best configuration per target, 4 dev evaluations).")
    if best_tree is not None: A(f"2. **Best model:** {cand_name(best_tree)} → mean gain {best_tree.mean_gain:+.1f}% vs fixed, worst flight {best_tree.min_gain:+.1f}%, MH_03 ATE {best_tree.mh03_ate:.3f} m.")
    if len(ran_be) > 1:
        A("3. **XGBoost / CatBoost / LightGBM vs HistGradientBoosting** (same target/feature combos): " +
          ", ".join(f"{b} {per_be[b].mean_gain:+.1f}%" for b in ran_be) + ". " +
          ("None beats HGB by more than the 1 pp tie margin → tie; HGB kept (simplest)." if all(per_be[b].mean_gain <= per_be["hgb"].mean_gain + 1.0 for b in ran_be) else
           "At least one library beats HGB by more than 1 pp: " + ", ".join(b for b in ran_be if per_be[b].mean_gain > per_be["hgb"].mean_gain + 1.0) + "."))
    else: A("3. **XGBoost / CatBoost / LightGBM:** not available in this environment (see environment); only HistGradientBoosting ran.")
    s3 = best_row(trees[trees.target.isin(["T3", "T4"])]); s1 = best_row(trees[(trees.target == "T1")])
    if s3 is not None and s1 is not None:
        A(f"4. **3D local residual vs scale-only:** best 3D ({s3.target}) {s3.mean_gain:+.1f}% (worst flight {s3.min_gain:+.1f}%) vs best scale-only (T1) {s1.mean_gain:+.1f}% "
          f"(worst {s1.min_gain:+.1f}%) → {'**yes, 3D correction is better**' if s3.mean_gain > s1.mean_gain + 1 else 'no clear gain'}.")
    if tnote.get("status") != "ran": A(f"5. **TCN / GRU:** {tnote.get('status')}.")
    elif best_temp is None: A(f"5. **TCN / GRU:** screened on MH_03 ({', '.join(f'{k} {v:+.1f}%' for k, v in tnote['screen'].items())}); every branch was >2 pp below the best tree → stopped. No real gain.")
    else: A(f"5. **TCN / GRU:** best {best_temp.backend} {best_temp.mean_gain:+.1f}% (worst {best_temp.min_gain:+.1f}%) vs best tree {best_tree.mean_gain:+.1f}% → "
            + ("**justified** (beats trees by >1 pp)." if best_temp.mean_gain > best_tree.mean_gain + 1 else "**not justified** (no gain beyond the 1 pp tie margin; trees are simpler)."))
    A("6. **Feature group that generalised best:** " + ", ".join(f"{fs} {per_fs[fs].mean_gain:+.1f}% (eligible: {bool(per_fs[fs].eligible)})" for fs in per_fs if per_fs[fs] is not None) +
      ". Tracking/image features must beat IMU+VINS on ≥3/4 flights (rule 5) to be kept.")
    if r_sel is not None:
        lofo = [T[(T.flight == f) & (T.method == "selected")].ate.iloc[0] for f in TRAIN]
        A(f"7. **Best development result:** selected model MH_03 ATE **{r_sel.mh03_ate:.3f} m** (fixed {T[(T.flight == 'MH_03_medium') & (T.method == 'fixed')].ate.iloc[0]:.3f}, VINS "
          f"{T[(T.flight == 'MH_03_medium') & (T.method == 'VINS')].ate.iloc[0]:.3f}); LOFO ATE MH_01/02/04 = {', '.join(f'{x:.3f}' for x in lofo)} m (mean {np.mean(lofo):.3f}).")
        A(f"8. **Beats fixed scale on all development flights?** {'**Yes – 4/4**' if r_sel.wins == 4 else f'{int(r_sel.wins)}/4'} (per-flight gains: " +
          ", ".join(f"{SHORT[f]} {r_sel[f'gain_{f[:5]}']:+.1f}%" for f in FOLD_ORDER) + ").")
        A(f"9. **Risks:** only 4 development flights from one environment (Machine Hall); the selection was made over {len(S)} candidates on the same 4 evaluations "
          "(optimistic by construction); the lateral/vertical corrections learn turn- and climb-dependent systematic errors that may differ on other platforms; "
          "MH_05 (difficult, fast) is outside the speed range of most training data.")
        A(f"10. **Model to freeze for MH_05:** **{cand_name(r_sel)}** (file `{frozen['model_file']}`, sha256 `{frozen['model_sha256'][:16]}…`).")
    else:
        A("7–10. **No candidate satisfied the frozen selection rule → recommendation: keep the fixed-scale baseline; no ML model is frozen.**")
    A("\n## Step 0 – correction targets and ORACLE ceilings (not deployable)")
    A("All targets are residuals **on top of the fixed-scale motion** (λ = 0 reproduces the fixed-scale baseline exactly). Labels use ground truth; "
      "the centred windows and the local GT alignment are **label-only** and never runtime inputs.\n")
    A("| target | definition |\n|---|---|")
    for t, v in TARGETS.items(): A(f"| {t} | {v['desc']} |")
    A("| T4v (rejected) | residual of VINS *state velocity* – inconsistent with the position integration (perfect labels give 0.34–0.67 m ATE, worse than VINS) → scientifically invalid, excluded |")
    A("\nFrame: forward = horizontal direction of VINS velocity at t_k, up = gravity, lateral = up × forward (all causal). True motion is expressed in VINS's local "
      "trajectory frame (yaw-only alignment of GT onto VINS over a centred ±5 s window, labels only), so slow heading drift – unobservable from IMU+VINS – is not part of any label.\n")
    A("| ORACLE ATE [m] | " + " | ".join(FOLD_ORDER) + " |\n|---|" + "---|" * 4)
    A("| fixed scale (deployable) | " + " | ".join(fmt(Oa[Oa.fold == f].fixed_ate.iloc[0]) for f in FOLD_ORDER) + " |")
    for t in ["T1", "T2", "T3", "T4", "T4v"]: A(f"| {t} oracle | " + " | ".join(fmt(Oa[(Oa.fold == f) & (Oa.target == t)].ate.iloc[0]) for f in FOLD_ORDER) + " |")
    A("\n![oracle](../assets/ml_correction_phase2/step0_oracle_ceilings.png)")
    A("\n## Protocol")
    A("- Train MH_01/02/04, validate MH_03; robustness = leave-one-flight-out over the 4 development flights (fixed scale refit per fold). 4 development evaluations per candidate.")
    A("- Feature groups: **IV** = IMU + VINS state + derived motion (speed, Δspeed, horizontal/vertical speed, yaw rate, speed×yaw rate); **IVT** = IV + feature-tracking statistics; "
      "**ALL** = all causal features incl. image quality. Excluded always: x/y/height, time since start, track age, anything GT-derived or future.")
    A(f"- Models: HistGradientBoosting (step A); LightGBM / XGBoost / CatBoost (step B, on {', '.join(f'{t}·{fs}' for t, fs in combos)}); small causal GRU/TCN with 1/3/5 s "
      "history (step C). Two fixed configurations per tree library, correction strength λ ∈ {0.5, 1}. Early stopping of neural nets on the last 20 % of each *training* flight only.")
    A("- **Selection rule (fixed before results):** (1) beat fixed scale on MH_03 and on ≥3/4 development evaluations; (2) no flight >2 % worse than fixed; (3) rank by mean gain over fixed; "
      "(4) within 1 pp prefer the simpler candidate (trees < GRU/TCN; T1/T2 < T3/T4; IV < IVT < ALL); (5) tracking/image features only if they beat IMU+VINS on ≥3/4 flights.")
    A("\n## Step A – target × feature set (HistGradientBoosting)\n![A](../assets/ml_correction_phase2/stepA_target_x_features.png)")
    A("\n## Step B – tree libraries\n![B](../assets/ml_correction_phase2/stepB_tree_libraries.png)")
    A(f"\n## Step C – temporal models\nStatus: **{tnote.get('status')}**" + (f"; screening MH_03 gains: {json.dumps({k: round(v, 2) for k, v in tnote.get('screen', {}).items()})}; full evaluation: {tnote.get('full')}" if tnote.get("status") == "ran" else ""))
    if tnote.get("status") == "ran": A("![C](../assets/ml_correction_phase2/stepC_temporal.png)\n![curves](../assets/ml_correction_phase2/stepC_training_curves.png)")
    A("\n## Top candidates (all 4 development evaluations, gains vs fixed scale in %)")
    cols = ["target", "fs", "backend", "cfg", "lam", "mh03_ate", "gain_MH_03", "gain_MH_01", "gain_MH_02", "gain_MH_04", "mean_gain", "min_gain", "wins", "complexity", "eligible"]
    top = S.head(15)[cols]
    A("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for _, r in top.iterrows(): A("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    A(f"\nFull table: `ml/correction_phase2/results/candidate_summary.csv` ({len(S)} candidates); per-fold metrics: `all_fold_metrics.csv`.")
    A("\n## Development comparison – VINS vs fixed vs phase-1 ML vs selected (all out-of-sample)")
    cols2 = ["ate", "impr_vs_vins_pct", "impr_vs_fixed_pct", "final_err", "ate_vert", "along_pct", "lateral_pct", "vertical_pct", "rpe1s", "rpe3s", "rpe5s"]
    A("| flight | method | " + " | ".join(cols2) + " |\n|---|---|" + "---|" * len(cols2))
    for f in FOLD_ORDER:
        for m in ["VINS", "fixed", "current", "selected"]:
            g = T[(T.flight == f) & (T.method == m)]
            if len(g): r = g.iloc[0]; A(f"| {SHORT[f]} | {LBL[m]} | " + " | ".join(f"{r[c]:.3f}" for c in cols2) + " |")
    if r_sel is not None:
        wd = T[T.method == "selected"].impr_vs_fixed_pct.min(); A(f"\nWorst-flight change vs fixed scale for the selected model: **{wd:+.1f}%**.")
    A("\n![per flight](../assets/ml_correction_phase2/dev_ate_per_flight.png)\n![drift](../assets/ml_correction_phase2/dev_drift_metrics.png)")
    for f in FOLD_ORDER: A(f"![{f}](../assets/ml_correction_phase2/trajectory_{f}.png)")
    A("\n## Known failure cases / caveats")
    A("- Heading (yaw) drift is not corrected – it is unobservable without an absolute reference; cross-track error that comes from heading drift remains.")
    A("- The VINS *state velocity* residual (T4v) is not a valid target: velocity and position increments of VINS disagree by ~1 mm/step on average (estimator corrections), "
      "so integrating a velocity correction drifts. The deployed T4 uses finite-difference velocity (position increments) instead.")
    A("- Per-step residual labels are noisy (per-step R² ≈ 0); the models learn the systematic, motion-dependent part (turn- and climb-dependent errors), not the per-step noise.")
    A("- Selection over many candidates on 4 evaluations is optimistic; MH_05 is the only unbiased estimate.")
    if frozen:
        A("\n## PRE-FINAL-TEST DECLARATION (frozen before MH_05)")
        A(f"- Target: **{frozen['target']}** – {frozen['target_definition']}\n- Features: **{frozen['feature_set']}** ({len(frozen['features'])} features, list in `ml/correction_phase2/models/SELECTED.json`)")
        A(f"- Model: **{frozen['backend']} config {frozen['config']}** {json.dumps(frozen['hyperparameters'])}; trained on {', '.join(frozen['trained_on'])}")
        A(f"- Correction strength λ = **{frozen['lam']}**; fixed-scale baseline **{frozen['fixed_scale']}**; clip {json.dumps(frozen['clip'])}")
        A(f"- Integration: {frozen['integration']}")
        A(f"- Model file `{frozen['model_file']}` sha256 `{frozen['model_sha256']}`; code and data hashes in `SELECTED.json`.")
        A("- MH_05 procedure: create `ml/correction_phase2/FINAL_TEST_APPROVED` (user-authorised), build MH_05 steps once (phase-1 builder, unchanged), evaluate the frozen model "
          "once with `p2_final.py`; non-selected candidates only POST-HOC, labelled as such, never used for selection.")
    return "\n".join(L) + "\n"


def update_log(S, sel, T, tnote, frozen, combos):
    begin, end = "<!-- PHASE2-BEGIN -->", "<!-- PHASE2-END -->"
    r = None if sel is None else S.loc[sel]
    body = [begin, "", "---", f"## ML correction – Phase 2 (development, MH_05 untouched) – updated {time.strftime('%Y-%m-%d %H:%M')}",
            "Targets T1–T4 (+ T4v rejected), feature sets IV/IVT/ALL, HGB + LightGBM/XGBoost/CatBoost + GRU/TCN; selection rule fixed before results "
            "(see `docs/ML_CORRECTION_PHASE2_REPORT.md`). Full results: `ml/correction_phase2/results/`.",
            f"- Stage B combos: {combos}; temporal stage: {tnote.get('status')}"]
    if r is not None:
        body.append(f"- **SELECTION DECISION: {cand_name(r)}** – MH_03 ATE {r.mh03_ate:.4f} m, mean gain vs fixed {r.mean_gain:+.2f}%, worst {r.min_gain:+.2f}%, wins {int(r.wins)}/4. "
                    f"Frozen as `{frozen['model_file']}` sha256 {frozen['model_sha256'][:16]}…")
    else: body.append("- **SELECTION DECISION: no candidate satisfies the rule – keep fixed scale.**")
    body.append("\n| flight | VINS | fixed | phase-1 ML | selected |\n|---|---|---|---|---|")
    for f in FOLD_ORDER:
        g = T[T.flight == f].set_index("method").ate; body.append(f"| {f} | " + " | ".join(fmt(g.get(m, np.nan), 4) for m in ["VINS", "fixed", "current", "selected"]) + " |")
    body += ["", end]
    txt = open(LOG, encoding="utf-8").read() if os.path.exists(LOG) else ""
    txt = re.sub(re.escape(begin) + ".*?" + re.escape(end), "\n".join(body), txt, flags=re.S) if begin in txt else txt.rstrip() + "\n\n" + "\n".join(body) + "\n"
    open(LOG, "w", encoding="utf-8").write(txt)


def update_csv(T, S, sel):
    if not os.path.exists(CSV): return
    cols = pd.read_csv(CSV, nrows=0).columns.tolist(); old = pd.read_csv(CSV); old = old[~old.experiment_id.astype(str).str.startswith("P2_")]
    rows = []
    for _, r in T[T.method.isin(["current", "selected"])].iterrows():
        d = {c: "" for c in cols}
        d.update(experiment_id=f"P2_{r.method}", sequence=r.flight, repeat=1, config_name="E0_official (unchanged VINS) + phase-2 correction",
                 overrides=("phase-1 HGB scale" if r.method == "current" else cand_name(S.loc[sel])), ate_rmse=round(r.ate, 6), max_error=round(r.max_err, 6),
                 final_error=round(r.final_err, 6), scale_factor=round(r.scale, 5), along_track_drift_pct=round(r.along_pct, 5), cross_track_drift_pct=round(r.cross_pct, 5),
                 rpe_trans_3s_mean=round(r.rpe3s, 7), status="ok", notes=f"phase-2 dev ({'val' if r.flight == 'MH_03_medium' else 'LOFO'})", evaluated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        rows.append(d)
    pd.concat([old, pd.DataFrame(rows, columns=cols)]).to_csv(CSV, index=False)
