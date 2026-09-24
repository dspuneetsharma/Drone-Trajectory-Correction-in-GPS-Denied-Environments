#!/usr/bin/env python3
"""Writes docs/ML_CORRECTION_FINAL_REPORT.md (+ experiment-log section + results rows) from the saved development and
one-time MH_05 results. Reads files only - trains/evaluates nothing."""
import os, sys, json, re, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_common import P, P2, TARGETS, TRAIN
from p2_report import best_row, cand_name, FOLD_ORDER, SHORT

S = pd.read_csv(P2("results", "candidate_summary.csv")); TN = json.load(open(P2("results", "temporal_stage.json")))
SEL = json.load(open(P2("models", "SELECTED.json"))); DM = pd.read_csv(P2("results", "dev_metrics_methods.csv"))
O = pd.read_csv(P2("results", "oracle_ceilings.csv")); FM = pd.read_csv(P2("results", "final", "test_metrics.csv"))
V = json.load(open(P2("results", "final", "test_verdict.json"))); ENV = json.load(open(P2("results", "environment.json")))
CT = pd.read_csv(P2("results", "causality_test.csv")) if os.path.exists(P2("results", "causality_test.csv")) else None
selection = json.load(open(P2("results", "selection.json")))
trees = S[~S.backend.str.startswith(("gru", "tcn"))]; temp = S[S.backend.str.startswith(("gru", "tcn"))]
sel = S[(S.target == SEL["target"]) & (S.fs == SEL["feature_set"]) & (S.backend == SEL["backend"]) & (S.cfg == SEL["config"]) & (S.lam == SEL["lam"])].iloc[0]
per_t = {t: best_row(trees[(trees.target == t) & (trees.fs != "IV_p1")]) for t in ["T1", "T2", "T3", "T4"]}
per_fs = {fs: best_row(trees[trees.fs == fs]) for fs in ["IV", "IVT", "ALL"]}
combos = [tuple(c) for c in selection["combos_B"]]
per_be = {be: best_row(trees[(trees.backend == be) & trees.apply(lambda r: (r.target, r.fs) in combos, axis=1)]) for be in ["hgb", "lgbm", "xgb", "cat"]}
best_temp = best_row(temp) if len(temp) else None; best_tree = best_row(trees[trees.fs != "IV_p1"])
cur = S[(S.fs == "IV_p1") & (S.lam == 1.0)].iloc[0]
f = FM.set_index("method"); dev_sel = DM[DM.method == "selected"].set_index("flight")
L = []; A = L.append
A("# ML correction layer – FINAL REPORT (including the one-time MH_05 test)")
A(f"*Generated {time.strftime('%Y-%m-%d %H:%M')}. Development and selection on MH_01–MH_04 only; MH_05 opened once after the freeze ({V['evaluated']}).*\n")
A("**Pipeline:** camera + IMU → unchanged open-source VINS-Fusion → causal ML predicts a local motion residual → corrected local motion → integrated trajectory → ATE.\n")
A("## Final test – MH_05_difficult (unseen, evaluated exactly once)")
A("| method | ATE RMSE [m] | vs VINS | vs fixed | final-point error [m] | vertical ATE [m] | along [%] | lateral [%] | vertical [%] | 1 s / 3 s / 5 s drift [cm] |\n|---|---|---|---|---|---|---|---|---|---|")
for m, lab in [("VINS", "Original VINS"), ("fixed", "Fixed-scale VINS (1.04003)"), ("selected", f"**ML-corrected (frozen: {V['model']})**"), ("phase1_POSTHOC", "phase-1 scale model – POST-HOC reference only")]:
    if m in f.index:
        r = f.loc[m]; A(f"| {lab} | {r.ate:.4f} | {r.impr_vs_vins_pct:+.1f}% | {r.impr_vs_fixed_pct:+.1f}% | {r.final_err:.3f} | {r.ate_vert:.3f} | {r.along_pct:+.2f} | {r.lateral_pct:.2f} | {r.vertical_pct:.2f} | "
          f"{100 * r.rpe1s:.1f} / {100 * r.rpe3s:.1f} / {100 * r.rpe5s:.1f} |")
A("\n![final](../assets/ml_correction_phase2/FINAL_MH05_comparison.png)\n")
A("## Answers")
A("1. **Correction targets tested:** T1 along-track scale (3 s, multiplicative – phase-1 target); T2 along-track displacement residual (3 s); T3 local 3-D displacement residual "
  "[forward, lateral, vertical] (3 s); T4 local 3-D velocity residual (finite-difference, per 0.1 s step). T4v (VINS *state*-velocity residual) was tested at the oracle stage and "
  "rejected as scientifically invalid (perfect labels still give 0.34–0.67 m ATE because VINS velocity and position increments disagree). Every target is a residual on top of the fixed scale.")
bt = max([t for t in per_t if per_t[t] is not None], key=lambda t: per_t[t].mean_gain)
A(f"2. **Best target:** **{bt}** (best deployable mean dev gain vs fixed: " + ", ".join(f"{t} {per_t[t].mean_gain:+.1f}%" for t in per_t if per_t[t] is not None) + ").")
A(f"3. **Best tabular model:** {cand_name(best_tree)} (mean dev gain {best_tree.mean_gain:+.1f}%, worst flight {best_tree.min_gain:+.1f}%).")
ran = [b for b in per_be if per_be[b] is not None]
A("4. **XGBoost / CatBoost / LightGBM vs HistGradientBoosting** (same target/feature combos): " + ", ".join(f"{b} {per_be[b].mean_gain:+.1f}% (worst {per_be[b].min_gain:+.1f}%)" for b in ran) + ". "
  + ("No library beat HGB by more than the 1 pp tie margin, so the simpler HGB is kept." if all(per_be[b].mean_gain <= per_be['hgb'].mean_gain + 1 for b in ran) else
     "Beat HGB by >1 pp: " + ", ".join(b for b in ran if per_be[b].mean_gain > per_be['hgb'].mean_gain + 1) + "."))
s1 = best_row(trees[trees.target == "T1"]); s3 = best_row(trees[trees.target.isin(["T3", "T4"])])
A(f"5. **3-D local correction vs scale-only:** development: best 3-D {s3.mean_gain:+.1f}% vs best scale-only {s1.mean_gain:+.1f}% mean gain over fixed. "
  f"MH_05: selected {SEL['target']} model {f.loc['selected'].ate:.4f} m" + (f" vs phase-1 scale model {f.loc['phase1_POSTHOC'].ate:.4f} m (post-hoc)." if "phase1_POSTHOC" in f.index else "."))
if TN.get("status") != "ran": A(f"6. **TCN / GRU:** {TN.get('status')}.")
elif best_temp is None: A("6. **TCN / GRU:** screened on MH_03 with 1/3/5 s history (" + ", ".join(f"{k} {v:+.1f}%" for k, v in TN["screen"].items()) + "); all branches >2 pp below the best tree → stopped. No gain; complexity not justified.")
else: A(f"6. **TCN / GRU:** best {best_temp.backend} {best_temp.mean_gain:+.1f}% (worst {best_temp.min_gain:+.1f}%) vs best tree {best_tree.mean_gain:+.1f}% → "
       + ("justified." if best_temp.mean_gain > best_tree.mean_gain + 1 else "not enough to justify the complexity (within/below the 1 pp tie margin)."))
A("7. **Feature group that generalised best:** " + ", ".join(f"{fs} {per_fs[fs].mean_gain:+.1f}%" for fs in per_fs if per_fs[fs] is not None) + " (IV = IMU + VINS state + derived motion; IVT adds tracking; ALL adds image quality).")
lofo = [dev_sel.loc[x].ate for x in TRAIN]
A(f"8. **Best LOFO performance (selected model):** MH_01 {lofo[0]:.3f}, MH_02 {lofo[1]:.3f}, MH_04 {lofo[2]:.3f} m (mean {np.mean(lofo):.3f}; gains vs fixed "
  + ", ".join(f"{sel[f'gain_{x[:5]}']:+.1f}%" for x in TRAIN) + ").")
A(f"9. **MH_03 validation ATE:** {sel.mh03_ate:.4f} m (fixed {DM[(DM.flight == 'MH_03_medium') & (DM.method == 'fixed')].ate.iloc[0]:.4f}, VINS {DM[(DM.flight == 'MH_03_medium') & (DM.method == 'VINS')].ate.iloc[0]:.4f}).")
A(f"10. **Frozen before MH_05:** {V['model']} – `{SEL['model_file']}` sha256 `{SEL['model_sha256']}`, frozen {SEL['created']}.")
A(f"11. **Original VINS MH_05 ATE:** {V['vins_ate']:.4f} m")
A(f"12. **Fixed-scale MH_05 ATE:** {V['fixed_ate']:.4f} m")
A(f"13. **ML-corrected MH_05 ATE:** **{V['ml_ate']:.4f} m**")
A(f"14. **Improvement:** {V['impr_vs_vins_pct']:+.1f}% vs original VINS; {V['impr_vs_fixed_pct']:+.1f}% vs fixed scale.")
fe = V["final_err"]; A(f"15. **Final-point error:** VINS {fe['VINS']:.3f} m, fixed {fe['fixed']:.3f} m, ML {fe['selected']:.3f} m → "
                       + ("**improved** vs both." if fe["selected"] < min(fe["VINS"], fe["fixed"]) else "improved vs fixed only." if fe["selected"] < fe["fixed"] else "**worse** than " + ("VINS and fixed." if fe["selected"] > max(fe["VINS"], fe["fixed"]) else "at least one baseline.")))
A("16. **Remaining failure modes:** heading/yaw drift is not corrected (unobservable without an absolute reference); corrections are learned from 4 Machine-Hall flights of one "
  "platform and may not transfer to other vehicles/environments; per-step residuals are mostly noise – only the systematic motion-dependent part is corrected; see per-flight worst cases above and in the phase-2 report.")
useful = V["impr_vs_fixed_pct"] > 1.0
A(f"17. **Is the ML contribution useful beyond fixed scale?** " + (f"**Yes on the unseen test flight** ({V['impr_vs_fixed_pct']:+.1f}% ATE vs fixed scale), consistent with development (4/4 flights, mean {sel.mean_gain:+.1f}%)." if useful else
      f"**Not demonstrated on the unseen test flight** ({V['impr_vs_fixed_pct']:+.1f}% vs fixed scale) although development showed {sel.mean_gain:+.1f}% – the fixed scale remains the robust baseline."))
if useful and V["beats_vins"]:
    A(f"18. **Claim:** An unchanged open-source VINS-Fusion stack plus a causal, lightweight ML residual-correction layer ({V['model']}; target: {TARGETS[SEL['target']]['desc']}; "
      f"features: {SEL['feature_set']}; no GPS or ground truth at runtime) reduced ATE on an unseen EuRoC flight by {V['impr_vs_vins_pct']:.0f}% vs VINS "
      f"and {V['impr_vs_fixed_pct']:.0f}% vs a fixed global scale correction. Evidence is from one dataset (5 flights, one environment); generalisation to other platforms is untested.")
else:
    A(f"18. **Claim:** VINS-Fusion's dominant error on EuRoC is a systematic ~4 % distance under-estimation that a fixed scale removes ({V['vins_ate']:.3f} → {V['fixed_ate']:.3f} m on MH_05); "
      "a causal ML residual layer improved development flights but did not add a demonstrated gain on the unseen test flight. Report both honestly.")
A("\n## Development summary (all out-of-sample)")
A("| flight | VINS | fixed | phase-1 ML | selected |\n|---|---|---|---|---|")
for fl in FOLD_ORDER:
    g = DM[DM.flight == fl].set_index("method").ate; A(f"| {fl} | " + " | ".join(f"{g.get(m, np.nan):.4f}" for m in ["VINS", "fixed", "current", "selected"]) + " |")
A(f"\nCandidates evaluated: {len(S)}; selection rule fixed before results (see phase-2 report). Causality test of runtime columns: "
  + ("PASS (max |diff| = 0)" if CT is not None and (CT.max_abs_diff < 1e-9).all() else "see results/causality_test.csv") + ".")
A(f"Environment: {ENV.get('platform')}, python {ENV.get('python')}, sklearn {ENV.get('sklearn')}, lightgbm {ENV.get('lightgbm')}, xgboost {ENV.get('xgboost')}, catboost {ENV.get('catboost')}, torch {ENV.get('torch')}, GPU {ENV.get('gpu', 'none')}.")
A("\nFiles: `ml/correction_phase2/` (code, data, models, results), `outputs/ml_correction_phase2/` (trajectory CSVs incl. `MH_05_difficult_trajectories_TEST.csv`), "
  "`assets/ml_correction_phase2/` (plots), `docs/ML_CORRECTION_PHASE2_REPORT.md` (development + pre-final-test declaration).")
A("\n**MH_05 is now a consumed test set.** Any further development must use MH_01–MH_04 or new datasets; MH_05 must not be used for tuning.")
open(P("docs", "ML_CORRECTION_FINAL_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# experiment log + results csv
begin, end = "<!-- FINAL-BEGIN -->", "<!-- FINAL-END -->"
body = [begin, "", "---", f"## ML correction – FINAL TEST on MH_05 (one-time) – {V['evaluated']}",
        f"Frozen model {V['model']} (sha256 {SEL['model_sha256'][:16]}…). ATE: VINS {V['vins_ate']:.4f} m, fixed {V['fixed_ate']:.4f} m, **ML {V['ml_ate']:.4f} m** "
        f"({V['impr_vs_vins_pct']:+.1f}% vs VINS, {V['impr_vs_fixed_pct']:+.1f}% vs fixed). Final-point error VINS {fe['VINS']:.3f} / fixed {fe['fixed']:.3f} / ML {fe['selected']:.3f} m. "
        "MH_05 is now consumed. Report: `docs/ML_CORRECTION_FINAL_REPORT.md`.", "", end]
LOGP = P("docs", "experiments", "EXPERIMENT_LOG.md"); txt = open(LOGP, encoding="utf-8").read()
txt = re.sub(re.escape(begin) + ".*?" + re.escape(end), "\n".join(body), txt, flags=re.S) if begin in txt else txt.rstrip() + "\n\n" + "\n".join(body) + "\n"
open(LOGP, "w", encoding="utf-8").write(txt)
CSVP = P("docs", "experiments", "experiment_results.csv")
if os.path.exists(CSVP):
    cols = pd.read_csv(CSVP, nrows=0).columns.tolist(); old = pd.read_csv(CSVP); old = old[~old.experiment_id.astype(str).str.startswith("P2F_")]; rows = []
    for m in ["VINS", "fixed", "selected"]:
        r = f.loc[m]; d = {c: "" for c in cols}
        d.update(experiment_id=f"P2F_{m}", sequence="MH_05_difficult", repeat=1, config_name="E0_official (unchanged VINS)" + ("" if m == "VINS" else " + correction"),
                 overrides={"VINS": "none", "fixed": "fixed scale 1.04003", "selected": V["model"]}[m], ate_rmse=round(r.ate, 6), max_error=round(r.max_err, 6), final_error=round(r.final_err, 6),
                 scale_factor=round(r.scale, 5), along_track_drift_pct=round(r.along_pct, 5), cross_track_drift_pct=round(r.cross_pct, 5), rpe_trans_3s_mean=round(r.rpe3s, 7),
                 status="ok", notes="FINAL one-time MH_05 test", evaluated=V["evaluated"])
        rows.append(d)
    pd.concat([old, pd.DataFrame(rows, columns=cols)]).to_csv(CSVP, index=False)
print("FINAL REPORT WRITTEN")
