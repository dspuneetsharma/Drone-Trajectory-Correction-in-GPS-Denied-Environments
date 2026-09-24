#!/usr/bin/env python3
"""ONE-TIME final evaluation of the FROZEN phase-2 model on the untouched test flight MH_05.

Preconditions (all enforced):
  * ml/correction_phase2/models/SELECTED.json exists (written by p2_run.py) and the model file hash matches it
  * docs/ML_CORRECTION_PHASE2_REPORT.md contains the PRE-FINAL-TEST DECLARATION
  * ml/correction_phase2/FINAL_TEST_APPROVED exists (user authorisation)
  * ml/correction/data/MH_05_difficult_steps.csv exists (built once by the unchanged phase-1 builder c01_build_steps.py)
  * ml/correction_phase2/results/final/test_metrics.csv does NOT exist (no second attempt)
Nothing is trained, tuned or re-selected. Primary comparison: VINS vs fixed scale vs the SELECTED model.
The phase-1 model is evaluated as a clearly labelled POST-HOC reference only.
Usage: python ml/correction_phase2/src/p2_final.py --confirm
"""
import os, sys, json, time, argparse, warnings, hashlib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_common as pc
from p2_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--confirm", action="store_true"); a = ap.parse_args()
if not a.confirm: sys.exit("add --confirm (one-time MH_05 evaluation)")
OUT = P2("results", "final"); os.makedirs(OUT, exist_ok=True)
if os.path.exists(os.path.join(OUT, "test_metrics.csv")): sys.exit("MH_05 has already been evaluated - refusing a second attempt.")
for need in (P2("FINAL_TEST_APPROVED"), P2("models", "SELECTED.json"), steps_path(TEST_FLIGHT), P("docs", "ML_CORRECTION_PHASE2_REPORT.md")):
    if not os.path.exists(need): sys.exit(f"precondition missing: {need}")
assert "PRE-FINAL-TEST DECLARATION" in open(P("docs", "ML_CORRECTION_PHASE2_REPORT.md"), encoding="utf-8").read(), "pre-final-test report missing"
FZ = json.load(open(P2("models", "SELECTED.json")))
mfile = P(FZ["model_file"]); assert sha(mfile) == FZ["model_sha256"], "selected model changed after freeze"
LOG = os.path.join(OUT, "test_access_log.txt")
open(LOG, "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} MH_05 unlocked for the one-time test; SELECTED.json sha {sha(P2('models', 'SELECTED.json'))}\n")

# ---- unlock MH_05 for this process only
_orig = pc.refuse_test
def _refuse(seq):
    if seq == TEST_FLIGHT: return
    _orig(seq)
pc.refuse_test = _refuse
pc.DEV = pc.DEV + [TEST_FLIGHT]

df = pc.build_local(TEST_FLIGHT); df.to_csv(P2("data", f"{TEST_FLIGHT}_local.csv"), index=False, float_format="%.8g")
sfix, lam, target = FZ["fixed_scale"], FZ["lam"], FZ["target"]


def predict(model_path, features):
    if model_path.endswith(".pt"):
        import torch, p2_models as M
        b = torch.load(model_path, map_location="cpu", weights_only=False); dev = M.torch_device(allow_cpu=True)
        return M.predict_temporal(b, df[b["features"]].values.astype(float), dev)
    import joblib
    b = joblib.load(model_path); return np.stack([m.predict(df[b["features"]].values) for m in b["models"]], 1)


methods = {"VINS": apply_correction(df, "baseline_vins", None, sfix, 1), "fixed": apply_correction(df, "baseline_fixed", None, sfix, 1)}
pred_sel = predict(mfile, FZ["features"]); methods["selected"] = apply_correction(df, target, pred_sel, sfix, lam)
post = P2("models", "candidates", "T1__IV_p1__hgb__A__MH_03_medium.joblib")         # phase-1 model re-trained in phase 2 (control) - POST-HOC reference only
if os.path.exists(post): methods["phase1_POSTHOC"] = apply_correction(df, "T1", predict(post, None), sfix, 1.0)
rows = []
for k, dpc in methods.items():
    r = evaluate(df, integrate(df, dpc))
    rows.append(dict(flight=TEST_FLIGHT, method=k, role="primary" if k in ("VINS", "fixed", "selected") else "POST-HOC (not used for any decision)", **r))
M_ = pd.DataFrame(rows); v = M_.set_index("method").ate
M_["impr_vs_vins_pct"] = 100 * (1 - M_.ate / v["VINS"]); M_["impr_vs_fixed_pct"] = 100 * (1 - M_.ate / v["fixed"])
M_.to_csv(os.path.join(OUT, "test_metrics.csv"), index=False)
fe = M_.set_index("method").final_err
verdict = dict(model=f"{target} · {FZ['feature_set']} · {FZ['backend']}-{FZ['config']} · λ={lam:g}", vins_ate=float(v["VINS"]), fixed_ate=float(v["fixed"]), ml_ate=float(v["selected"]),
               impr_vs_vins_pct=float(100 * (1 - v["selected"] / v["VINS"])), impr_vs_fixed_pct=float(100 * (1 - v["selected"] / v["fixed"])),
               final_err=dict(VINS=float(fe["VINS"]), fixed=float(fe["fixed"]), selected=float(fe["selected"])),
               beats_vins=bool(v["selected"] < v["VINS"]), beats_fixed=bool(v["selected"] < v["fixed"]), evaluated=time.strftime("%Y-%m-%d %H:%M:%S"))
json.dump(verdict, open(os.path.join(OUT, "test_verdict.json"), "w"), indent=1)
T = df[["sequence", "timestamp_ns", "t_s"]].copy()
for k, dpc in methods.items():
    Pc = integrate(df, dpc); T[f"{k}_x"], T[f"{k}_y"], T[f"{k}_z"] = Pc[:, 0], Pc[:, 1], Pc[:, 2]
os.makedirs(P("outputs", "ml_correction_phase2"), exist_ok=True)
T.to_csv(P("outputs", "ml_correction_phase2", f"{TEST_FLIGHT}_trajectories_TEST.csv"), index=False, float_format="%.6f")
open(LOG, "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} MH_05 evaluated once: {json.dumps(verdict)}\n")
print(M_[["method", "role", "ate", "final_err", "impr_vs_vins_pct", "impr_vs_fixed_pct"]].round(4).to_string(index=False)); print(json.dumps(verdict, indent=1))

# ---- final comparison figure (GT vs VINS vs fixed vs ML)
import p2_report as R
plt = R._plt(); gt_t, gp, _, _ = load_gt(TEST_FLIGHT); t = df.timestamp_ns.values
i = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1); i = np.where(np.abs(gt_t[i - 1] - t) < np.abs(gt_t[i] - t), i - 1, i); ok = np.abs(gt_t[i] - t) < 10_000_000; B = gp[i[ok]]
fig = plt.figure(figsize=(17, 8.6)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3]); a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
a0.plot(B[:, 0], B[:, 1], color=R.C["GT"], lw=3, label="ground truth")
for k, col, lab in [("VINS", R.C["VINS"], "Original VINS"), ("fixed", R.C["fixed"], "Fixed-scale VINS"), ("selected", R.C["selected"], "ML-corrected VINS (frozen model)")]:
    A_ = integrate(df, methods[k])[ok]; _, Rm, tr = umeyama(A_, B); Aa = (Rm @ A_.T).T + tr; E = np.linalg.norm(Aa - B, axis=1)
    a0.plot(Aa[:, 0], Aa[:, 1], color=col, lw=1.3, label=f"{lab}  ATE {np.sqrt(np.mean(E ** 2)):.3f} m"); a1.plot(df.t_s.values[ok], E, color=col, lw=1.3, label=lab)
    a2.plot(df.t_s.values[ok], Aa[:, 2] - B[:, 2], color=col, lw=1.1, label=lab)
a0.set_aspect("equal", "datalim"); a0.legend(fontsize=9); a0.set_xlabel("x [m]"); a0.set_ylabel("y [m]"); a0.set_title("MH_05_difficult (unseen test flight): top view, SE3-aligned")
a1.set_ylabel("position error [m]"); a1.legend(fontsize=9); a1.set_title("Error over time")
a2.axhline(0, color="#3a3a3a", lw=0.6); a2.set_ylabel("height error [m]"); a2.set_xlabel("time [s]"); a2.legend(fontsize=9); a2.set_title("Vertical error over time")
fig.suptitle(f"FINAL TEST (evaluated once): ground truth vs original VINS vs fixed scale vs ML correction - {verdict['model']}", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(P("assets", "ml_correction_phase2", "FINAL_MH05_comparison.png"), dpi=130); plt.close(fig)
print("FINAL_TEST_DONE")
