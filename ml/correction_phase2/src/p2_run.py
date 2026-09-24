#!/usr/bin/env python3
"""Phase-2 master runner (development only, MH_05 is never listed, built or loaded).

  python ml/correction_phase2/src/p2_run.py            # everything, resumable (finished jobs are cached and skipped)
  options: --allow-cpu-temporal   run GRU/TCN on CPU if CUDA is missing (otherwise the temporal stage is recorded as BLOCKED)
           --skip-temporal        skip stage C
Stages: env -> lock check -> data -> oracle (step 0) -> A (HGB: targets x feature sets) -> B (XGBoost/CatBoost/LightGBM)
        -> C (GRU/TCN, 1/3/5 s) -> selection (rule fixed before results) -> freeze -> trajectories/plots/report/log -> STOP.
"""
import os, sys, json, time, argparse, traceback, warnings, platform, glob
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_common import *
import p2_models as M

RES, CACHE, MOD, DATA = P2("results"), P2("cache"), P2("models"), P2("data")
OUTT, ASSETS = P("outputs", "ml_correction_phase2"), P("assets", "ml_correction_phase2")
for d in (RES, CACHE, os.path.join(CACHE, "preds"), os.path.join(CACHE, "curves"), MOD, os.path.join(MOD, "candidates"), DATA, OUTT, ASSETS):
    os.makedirs(d, exist_ok=True)

TARGET_LIST = ["T1", "T2", "T3", "T4"]
FS_LIST = ["IV", "IVT", "ALL"]
CONTROL = ("T1", "IV_p1", "hgb", "A")                       # the current phase-1 model (ML_imu_vins), re-run as a control
FOLDS = [dict(test="MH_03_medium", train=TRAIN, kind="val")] + \
        [dict(test=f, train=[g for g in DEV if g != f], kind="LOFO") for f in TRAIN]
FOLD_ORDER = ["MH_03_medium"] + TRAIN

# ---- selection rule: registered BEFORE any phase-2 result (see docs/ML_CORRECTION_PHASE2_REPORT.md, "Selection rule")
RULE = dict(min_wins=3, max_loss_pct=-2.0, tie_pp=1.0)
COMPLEXITY = dict(backend={"hgb": 0.0, "lgbm": 0.5, "xgb": 0.5, "cat": 0.5, "gru": 2.0, "tcn": 2.0},
                  target={"T1": 0.0, "T2": 0.5, "T3": 1.0, "T4": 1.0}, fs={"IV_p1": 0.0, "IV": 0.0, "IVT": 1.0, "ALL": 2.0}, cfg={"A": 0.0, "B": 0.25})

LOGF = os.path.join(RES, "run_log.txt")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); open(LOGF, "a", encoding="utf-8").write(s + "\n")


def status(stage, msg, **kw):
    save_json(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), stage=stage, msg=msg, **kw), os.path.join(RES, "STATUS.json"))


def failure(key, err):
    log("FAILED", key, err.splitlines()[-1] if err else "")
    open(os.path.join(RES, "failures.txt"), "a", encoding="utf-8").write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {key}\n{err}\n")


# ------------------------------------------------------------------ env + lock
def stage_env(a):
    info = dict(python=sys.version.split()[0], executable=sys.executable, platform=platform.platform(), cpu_count=os.cpu_count(), **M.versions())
    try:
        import torch; info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available(): info["gpu"] = torch.cuda.get_device_name(0); info["torch_cuda"] = torch.version.cuda
    except Exception as e: info["cuda_available"] = False; info["torch_error"] = type(e).__name__
    info["backends"] = {b: M.backend_available(b) for b in M.TREE_BACKENDS + ["gru", "tcn"]}
    save_json(info, os.path.join(RES, "environment.json")); log("ENV", json.dumps(info))
    # MH_05 lock
    bad = [p for p in glob.glob(P2("**", "*MH_05*"), recursive=True)]
    assert not bad, f"MH_05 artefacts found in phase-2 folders before the final test: {bad}"
    assert TEST_FLIGHT not in DEV
    try: refuse_test(TEST_FLIGHT); raise AssertionError("lock failed")
    except PermissionError: pass
    log("LOCK OK: MH_05 not in DEV, loaders refuse it, no MH_05 files in ml/correction_phase2")
    return info


def stage_data():
    man_p = os.path.join(DATA, "manifest.json"); man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    for s in DEV:
        ins = {os.path.relpath(p, ROOT): sha(p) for p in (steps_path(s), P("outputs", s, "vio.csv"), gt_path(s))}
        out = os.path.join(DATA, f"{s}_local.csv")
        if os.path.exists(out) and man.get(s, {}).get("inputs") == ins and man.get(s, {}).get("code") == sha(os.path.join(HERE, "p2_common.py")):
            continue
        df = build_local(s); df.to_csv(out, index=False, float_format="%.8g")
        man[s] = dict(inputs=ins, code=sha(os.path.join(HERE, "p2_common.py")), output_sha=sha(out), rows=len(df)); log("DATA built", s, len(df), "rows")
    save_json(man, man_p)
    return {s: load_local(s) for s in DEV}


def fold_sfix(D, fold):
    if fold["kind"] == "val": return S_FIX
    d = pd.concat([D[f] for f in fold["train"]]); m = d.w_c3 > 0
    return float(np.sum(d.s_c3[m] * d.w_c3[m]) / np.sum(d.w_c3[m]))


# ------------------------------------------------------------------ jobs
def key_of(target, fs, backend, cfg, fold_test): return f"{target}__{fs}__{backend}__{cfg}__{fold_test}"


def pred_path(key): return os.path.join(CACHE, "preds", key + ".npz")


def train_matrix(D, flights, target, feats, sfix):
    Xs, Ys, Ws = [], [], []
    for f in flights:
        Y, w, v = labels(D[f], target, sfix); Xs.append(D[f][feats].values[v]); Ys.append(Y[v]); Ws.append(w[v])
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)


def run_tree_job(D, FS, target, fs, backend, cfg, fold):
    key = key_of(target, fs, backend, cfg, fold["test"]); pp = pred_path(key)
    if os.path.exists(pp): return "cached"
    feats = FS[fs]; sfix = fold_sfix(D, fold); t0 = time.time()
    X, Y, w = train_matrix(D, fold["train"], target, feats, sfix)
    models = M.fit_trees(backend, cfg, X, Y, w); pred = M.predict_trees(models, D[fold["test"]][feats].values)
    if fold["kind"] == "val":                                   # the TRAIN-only model (MH_01/02/04) is the deployable candidate
        import joblib
        mp = os.path.join(MOD, "candidates", key + ".joblib")
        joblib.dump(dict(models=models, features=feats, target=target, backend=backend, cfg=cfg, params=M.CONFIGS[backend][cfg],
                         s_fix=sfix, train=fold["train"], versions=M.versions()), mp)
    np.savez_compressed(pp, pred=pred, secs=time.time() - t0, sfix=sfix)
    return f"{time.time() - t0:.1f}s"


def run_temporal_job(D, FS, target, arch, win, fold, seeds, dev, tag):
    key = key_of(target, "IV", f"{arch}{win}", tag, fold["test"]); pp = pred_path(key)
    if os.path.exists(pp): return "cached"
    feats = FS["IV"]; sfix = fold_sfix(D, fold); t0 = time.time()
    data = {}
    for f in fold["train"] + [fold["test"]]:
        Y, w, v = labels(D[f], target, sfix); data[f] = (D[f][feats].values.astype(float), np.nan_to_num(Y), w, v)
    pred, bundle = M.fit_predict_temporal(arch, M.WINDOWS[win], data, fold["train"], fold["test"], seeds, dev,
                                          curve_path=os.path.join(CACHE, "curves", key + ".json"))
    if fold["kind"] == "val":
        import torch
        bundle.update(features=feats, target=target, s_fix=sfix, train=fold["train"], config=M.TEMPORAL[arch], train_cfg=M.TRAIN_CFG, versions=M.versions())
        torch.save(bundle, os.path.join(MOD, "candidates", key + ".pt"))
    np.savez_compressed(pp, pred=pred, secs=time.time() - t0, sfix=sfix)
    return f"{time.time() - t0:.1f}s"


# ------------------------------------------------------------------ evaluation
_base = {}


def baselines(D):
    if _base: return _base
    for fold in FOLDS:
        df = D[fold["test"]]; sfix = fold_sfix(D, fold)
        _base[fold["test"]] = dict(sfix=sfix, VINS=evaluate(df, integrate(df, apply_correction(df, "baseline_vins", None, sfix, 1))),
                                   fixed=evaluate(df, integrate(df, apply_correction(df, "baseline_fixed", None, sfix, 1))))
    return _base


def evaluate_candidates(D, keys_prefix=None):
    """-> long table: one row per (candidate, lambda, fold)"""
    B = baselines(D); rows = []
    for pp in sorted(glob.glob(os.path.join(CACHE, "preds", "*.npz"))):
        key = os.path.basename(pp)[:-4]; target, fs, backend, cfg, test = key.split("__")
        if keys_prefix and not key.startswith(keys_prefix): continue
        z = np.load(pp); df = D[test]; sfix = float(z["sfix"])
        for lam in LAMBDAS:
            r = evaluate(df, integrate(df, apply_correction(df, target, z["pred"], sfix, lam)))
            rows.append(dict(target=target, fs=fs, backend=backend, cfg=cfg, lam=lam, fold=test, secs=float(z["secs"]), **r,
                             gain_vs_fixed=100 * (1 - r["ate"] / B[test]["fixed"]["ate"]), gain_vs_vins=100 * (1 - r["ate"] / B[test]["VINS"]["ate"])))
    return pd.DataFrame(rows)


def summarise(L):
    """one row per candidate (target, fs, backend, cfg, lam) over the 4 development evaluations"""
    out = []
    for k, g in L.groupby(["target", "fs", "backend", "cfg", "lam"]):
        if set(g.fold) != set(FOLD_ORDER): continue
        g = g.set_index("fold").loc[FOLD_ORDER]; gains = g.gain_vs_fixed.values
        be = k[2].rstrip("0123456789s") if k[2][:3] in ("gru", "tcn") else k[2]
        out.append(dict(target=k[0], fs=k[1], backend=k[2], cfg=k[3], lam=k[4],
                        mh03_ate=g.loc["MH_03_medium", "ate"], mh03_gain=gains[0], **{f"gain_{f[:5]}": gains[i] for i, f in enumerate(FOLD_ORDER)},
                        **{f"ate_{f[:5]}": g.ate.values[i] for i, f in enumerate(FOLD_ORDER)},
                        mean_gain=gains.mean(), min_gain=gains.min(), wins=int((gains > 0).sum()), lofo_mean_ate=g.ate.values[1:].mean(),
                        mean_gain_vs_vins=g.gain_vs_vins.mean(), train_secs=g.secs.sum(),
                        complexity=COMPLEXITY["backend"][be] + COMPLEXITY["target"][k[0]] + COMPLEXITY["fs"][k[1]] + COMPLEXITY["cfg"].get(k[3], 0.25)))
    return pd.DataFrame(out)


def apply_rule(S, L):
    """Frozen selection rule. Returns S with eligibility columns and the selected row index."""
    S = S.copy()
    S["r1_mh03"] = S.mh03_gain > 0
    S["r1_wins"] = S.wins >= RULE["min_wins"]
    S["r2_worst"] = S.min_gain >= RULE["max_loss_pct"]
    # rule 5: tracking / image features must show cross-flight benefit over the IMU+VINS version of the same model
    r5, r5_note = [], []
    for _, r in S.iterrows():
        if r.fs not in ("IVT", "ALL"): r5.append(True); r5_note.append(""); continue
        base = S[(S.target == r.target) & (S.fs == "IV") & (S.backend == r.backend) & (S.cfg == r.cfg) & (S.lam == r.lam)]
        if base.empty: r5.append(False); r5_note.append("no IV counterpart"); continue
        b = base.iloc[0]; better = sum(r[f"ate_{f[:5]}"] < b[f"ate_{f[:5]}"] for f in FOLD_ORDER)
        ok = better >= 3 and r.mean_gain > b.mean_gain; r5.append(ok); r5_note.append(f"beats IV on {better}/4, mean {r.mean_gain - b.mean_gain:+.2f} pp")
    S["r5_features"] = r5; S["r5_note"] = r5_note
    S["eligible"] = S.r1_mh03 & S.r1_wins & S.r2_worst & S.r5_features
    S = S.sort_values("mean_gain", ascending=False).reset_index(drop=True)
    E = S[S.eligible]
    if E.empty: return S, None
    best = E.mean_gain.max(); tied = E[E.mean_gain >= best - RULE["tie_pp"]]
    sel = tied.sort_values(["complexity", "mean_gain"], ascending=[True, False]).index[0]
    return S, sel


# ------------------------------------------------------------------ stages
def stage_oracle(D):
    rows = []; B = baselines(D)
    for fold in FOLDS:
        df = D[fold["test"]]; sfix = fold_sfix(D, fold)
        for T in TARGET_LIST + ["T4v"]:
            Y, w, v = labels(df, T, sfix); Yo = np.where(v[:, None], Y, np.nan)
            comps = [("all", None)] + ([(c, j) for j, c in enumerate("flu")] if T == "T3" else [])
            for cname, j in comps:
                Z = Yo if j is None else np.where(np.arange(3)[None, :] == j, Yo, np.nan)
                r = evaluate(df, integrate(df, apply_correction(df, T, Z, sfix, 1.0)))
                rows.append(dict(fold=fold["test"], target=T, component=cname, ate=r["ate"], fixed_ate=B[fold["test"]]["fixed"]["ate"],
                                 vins_ate=B[fold["test"]]["VINS"]["ate"], gain_vs_fixed=100 * (1 - r["ate"] / B[fold["test"]]["fixed"]["ate"]),
                                 final_err=r["final_err"], note="ORACLE: uses centred future GT labels - NOT deployable"))
    O = pd.DataFrame(rows); O.to_csv(os.path.join(RES, "oracle_ceilings.csv"), index=False)
    piv = O[O.component == "all"].pivot(index="target", columns="fold", values="ate")
    log("ORACLE (not deployable) ATE:\n" + piv[FOLD_ORDER].round(4).to_string())
    valid = [T for T in TARGET_LIST if (O[(O.target == T) & (O.component == "all")].gain_vs_fixed > 0).mean() >= 0.75]
    log("targets with oracle headroom over fixed on >=3/4 folds:", valid)
    return O, valid


def stage_A(D, FS, targets):
    jobs = [(T, fs, "hgb", c) for T in targets for fs in FS_LIST for c in ("A", "B")] + [CONTROL]
    run_jobs(D, FS, jobs, "A")


def run_jobs(D, FS, jobs, stage):
    n = len(jobs) * len(FOLDS); i = 0
    for (T, fs, be, c) in jobs:
        if not M.backend_available(be): log(f"[{stage}] backend {be} unavailable - skipped"); continue
        for fold in FOLDS:
            i += 1; key = key_of(T, fs, be, c, fold["test"]); status(stage, f"{i}/{n} {key}")
            try: r = run_tree_job(D, FS, T, fs, be, c, fold)
            except Exception: failure(key, traceback.format_exc()); continue
            if r != "cached": log(f"[{stage}] {i}/{n} {key} {r}")


def stage_B(D, FS, SA):
    """backends on the 2 best (target, feature set) combos from stage A (+ the phase-1 combo T1/IV)."""
    best = SA.sort_values(["eligible", "mean_gain"], ascending=[False, False]).drop_duplicates(["target", "fs"])
    combos = [tuple(x) for x in best[["target", "fs"]].values[:2]]
    if ("T1", "IV") not in combos: combos.append(("T1", "IV"))
    log("[B] combos:", combos)
    jobs = [(T, fs, be, c) for (T, fs) in combos for be in ("lgbm", "xgb", "cat") for c in ("A", "B")]
    run_jobs(D, FS, jobs, "B"); return combos


def stage_C(D, FS, S_trees, a):
    note = {}
    if not (M.backend_available("gru")):
        note["status"] = "BLOCKED: PyTorch not installed in this environment"; log("[C]", note["status"]); return note
    try: dev = M.torch_device(a.allow_cpu_temporal)
    except RuntimeError as e: note["status"] = f"BLOCKED: {e}"; log("[C]", note["status"]); return note
    E = S_trees[S_trees.eligible]; top = (E if len(E) else S_trees).iloc[0]; target = top.target
    best_tree_mh03 = S_trees[S_trees.target == target].mh03_gain.max()
    note.update(status="ran", device=str(dev), target=target, best_tree_mh03_gain=float(best_tree_mh03), screen={}, full=[])
    log(f"[C] temporal on target {target}, IV features, device {dev}; best tree MH_03 gain for this target {best_tree_mh03:+.2f}%")
    val = FOLDS[0]; B = baselines(D)
    for arch in M.TEMPORAL:
        best = None
        for win in M.WINDOWS:
            key = key_of(target, "IV", f"{arch}{win}", "screen", val["test"]); status("C", f"screen {key}")
            try: r = run_temporal_job(D, FS, target, arch, win, val, M.SEEDS_SCREEN, dev, "screen")
            except Exception: failure(key, traceback.format_exc()); continue
            z = np.load(pred_path(key)); df = D[val["test"]]
            g = max(100 * (1 - evaluate(df, integrate(df, apply_correction(df, target, z["pred"], S_FIX, lam)))["ate"] / B[val["test"]]["fixed"]["ate"]) for lam in LAMBDAS)
            note["screen"][f"{arch}{win}"] = float(g); log(f"[C] screen {arch} {win}: MH_03 gain vs fixed {g:+.2f}% ({r})")
            if best is None or g > best[1]: best = (win, g)
        if best is None: continue
        if best[1] < best_tree_mh03 - 2.0:
            log(f"[C] {arch}: best window {best[0]} {best[1]:+.2f}% is >2 pp below the best tree ({best_tree_mh03:+.2f}%) -> branch stopped"); continue
        note["full"].append(f"{arch}{best[0]}")
        for fold in FOLDS:
            key = key_of(target, "IV", f"{arch}{best[0]}", "full", fold["test"]); status("C", f"full {key}")
            try: r = run_temporal_job(D, FS, target, arch, best[0], fold, M.SEEDS_FULL, dev, "full"); log(f"[C] full {key} {r}")
            except Exception: failure(key, traceback.format_exc())
    return note


def freeze(D, FS, S, sel, info):
    r = S.loc[sel]; key = key_of(r.target, r.fs, r.backend, r.cfg, "MH_03_medium")
    ext = ".pt" if r.backend[:3] in ("gru", "tcn") else ".joblib"; src = os.path.join(MOD, "candidates", key + ext)
    import shutil
    dst = os.path.join(MOD, "SELECTED" + ext); shutil.copyfile(src, dst)
    feats = FS["IV"] if ext == ".pt" else FS[r.fs]
    frozen = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), status="FROZEN before MH_05",
                  target=r.target, target_definition=TARGETS[r.target]["desc"], feature_set=r.fs, features=feats,
                  backend=r.backend, config=r.cfg, hyperparameters=(M.TEMPORAL.get(r.backend[:3]) if ext == ".pt" else M.CONFIGS[r.backend][r.cfg]),
                  lam=float(r.lam), fixed_scale=S_FIX, clip=dict(T1=[0.9, 1.2], rate_mps=RATE_CLIP),
                  integration="p_c(k) = p_c(k-1) + dp_c(k); T1: dp_c = s_k*dp, s_k = s_fix + lam*(pred - s_fix); T2-T4: dp_c = s_fix*dp + lam*dt*H_k*pred "
                              "(H_k = [forward, lateral, up] from VINS velocity at t_k)",
                  trained_on=TRAIN, model_file=os.path.relpath(dst, ROOT), model_sha256=sha(dst), candidate_file=os.path.relpath(src, ROOT),
                  dev_results=dict(mh03_ate=float(r.mh03_ate), mh03_gain_vs_fixed=float(r.mh03_gain), mean_gain_vs_fixed=float(r.mean_gain),
                                   min_gain=float(r.min_gain), wins=int(r.wins)),
                  selection_rule=RULE, code_sha256={f: sha(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")},
                  data_sha256={s: sha(os.path.join(DATA, f"{s}_local.csv")) for s in DEV}, environment=info)
    save_json(frozen, os.path.join(MOD, "SELECTED.json")); log("FROZEN", json.dumps({k: frozen[k] for k in ("target", "feature_set", "backend", "config", "lam", "model_sha256")}))
    return frozen


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--allow-cpu-temporal", action="store_true"); ap.add_argument("--skip-temporal", action="store_true")
    ap.add_argument("--no-report", action="store_true"); a = ap.parse_args()
    log("=" * 20, "phase-2 run start", ROOT)
    status("env", "checking"); info = stage_env(a)
    status("data", "building"); D = stage_data(); FS = feature_sets(D["MH_01_easy"])
    save_json({k: v for k, v in FS.items()}, os.path.join(RES, "feature_sets.json"))
    save_json({k: v["desc"] for k, v in TARGETS.items()}, os.path.join(RES, "target_definitions.json"))
    status("oracle", "step 0"); O, valid_targets = stage_oracle(D)
    status("A", "start"); stage_A(D, FS, valid_targets)
    L = evaluate_candidates(D); SA, _ = apply_rule(summarise(L[L.backend == "hgb"]), L)
    log("[A] top:\n" + SA.head(12)[["target", "fs", "cfg", "lam", "mh03_ate", "mean_gain", "min_gain", "wins", "eligible"]].round(3).to_string())
    status("B", "start"); combos = stage_B(D, FS, SA)
    L = evaluate_candidates(D); S_trees, _ = apply_rule(summarise(L[~L.backend.str.startswith(("gru", "tcn"))]), L)
    temporal_note = {"status": "skipped (--skip-temporal)"} if a.skip_temporal else stage_C(D, FS, S_trees, a)
    save_json(temporal_note, os.path.join(RES, "temporal_stage.json"))
    status("select", "applying frozen rule")
    L = evaluate_candidates(D); L = L[~(L.cfg == "screen")]
    S, sel = apply_rule(summarise(L), L)
    L.to_csv(os.path.join(RES, "all_fold_metrics.csv"), index=False); S.to_csv(os.path.join(RES, "candidate_summary.csv"), index=False)
    frozen = None
    if sel is None: log("NO candidate satisfies the selection rule -> recommendation: keep the fixed-scale baseline")
    else:
        log("SELECTED:", S.loc[sel][["target", "fs", "backend", "cfg", "lam", "mh03_ate", "mean_gain", "min_gain", "wins"]].to_dict())
        frozen = freeze(D, FS, S, sel, info)
    save_json(dict(selected=None if sel is None else int(sel), combos_B=combos, valid_targets=valid_targets), os.path.join(RES, "selection.json"))
    if not a.no_report:
        status("report", "plots + report + log")
        import p2_report
        p2_report.make_all(D, FS, S, sel, L, O, temporal_note, frozen, info, combos, valid_targets)
    status("done", "PHASE2_DONE - stopped before MH_05"); log("PHASE2_DONE - stopped before MH_05 (MH_05 untouched)")


if __name__ == "__main__":
    main()
