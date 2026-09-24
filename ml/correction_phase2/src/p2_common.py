"""Phase 2 of the ML residual-correction layer: shared data, targets, integration and evaluation.

MH_05 is never listed, built or loaded here: DEV contains MH_01..MH_04 only, and every loader refuses MH_05.
Ground truth is used for labels and evaluation only; nothing derived from it is ever a model input.
"""
import os, json, hashlib, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DRONE_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
NS = 1_000_000_000
TRAIN, VAL = ["MH_01_easy", "MH_02_easy", "MH_04_difficult"], ["MH_03_medium"]
DEV = TRAIN + VAL
TEST_FLIGHT = "MH_05_difficult"                     # only named so it can be refused
S_FIX = 1.04003                                     # frozen fixed-scale baseline (phase 1)
LAMBDAS = [0.5, 1.0]                                # shrinkage toward the fixed-scale baseline (lambda = 0 IS the fixed baseline)
RATE_CLIP = 0.5                                     # m/s clip for residual-rate labels
WIN3 = 31                                           # centred 3 s window at 10 Hz
LOCAL_HALF = 50                                     # +-5 s local yaw alignment for labels


def P(*a): return os.path.join(ROOT, *a)
P2 = lambda *a: P("ml", "correction_phase2", *a)


def refuse_test(seq):
    if seq == TEST_FLIGHT or seq not in DEV:
        raise PermissionError(f"{seq}: phase 2 is development-only (MH_01..MH_04). MH_05 stays untouched.")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ geometry
def qnorm(q): return q / np.linalg.norm(q, axis=-1, keepdims=True)


def q2R(q):
    q = qnorm(np.atleast_2d(q)); w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                     np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


def slerp(q0, q1, u):
    q0, q1 = qnorm(q0), qnorm(q1); d = np.sum(q0*q1, -1); q1 = np.where(d[:, None] < 0, -q1, q1); d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1)); s = np.sin(th); small = s < 1e-6
    a = np.where(small, 1-u, np.sin((1-u)*th)/np.where(small, 1, s)); b = np.where(small, u, np.sin(u*th)/np.where(small, 1, s))
    return qnorm(a[:, None]*q0 + b[:, None]*q1)


def rotvec(R):
    """log map of rotation matrices (n,3,3) -> (n,3)"""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1); th = np.arccos(c)
    v = np.stack([R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]], 1)
    f = np.where(th < 1e-6, 0.5, th / np.maximum(2*np.sin(th), 1e-9)); return v * f[:, None]


def umeyama(A, B, scale=False):
    ma, mb = A.mean(0), B.mean(0); X, Y = A - ma, B - mb; U, S, Vt = np.linalg.svd(Y.T @ X / len(A)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / X.var(0).sum() if scale else 1.0; return s, R, mb - s * R @ ma


# ------------------------------------------------------------------ raw inputs
def load_vio(seq):
    refuse_test(seq)
    rows = [l.strip().rstrip(",").split(",") for l in open(P("outputs", seq, "vio.csv"))]
    t = np.array([int(r[0]) for r in rows], dtype=np.int64); v = np.array([[float(x) for x in r[1:11]] for r in rows])
    return t, v[:, 0:3], v[:, 3:7], v[:, 7:10]


def gt_path(seq): return P("Data", "machine_hall", seq, seq, "mav0", "state_groundtruth_estimate0", "data.csv")


def load_gt(seq):
    refuse_test(seq)
    p = gt_path(seq); t = np.loadtxt(p, delimiter=",", comments="#", usecols=0, dtype=np.int64); a = np.loadtxt(p, delimiter=",", comments="#")
    return t, a[:, 1:4], a[:, 4:8], a[:, 8:11]


def steps_path(seq): return P("ml", "correction", "data", f"{seq}_steps.csv")


# ------------------------------------------------------------------ phase-2 local-frame dataset
EXCLUDED = {"f_vins_x", "f_vins_y", "f_vins_height", "f_time_since_start_s", "f_trk_age_s"}
DERIVED_P1 = ["d_speed", "d_dspeed_1s", "d_dspeed_3s", "d_speed_1s"]
DERIVED_P2 = ["d_vh", "d_vz", "d_yawrate_1s", "d_centrip_1s"]


def build_local(seq):
    """Adds to the phase-1 step table (runtime columns unchanged):
    runtime (causal, VINS only): heading frame H_k = [f, l, u] (f = horizontal direction of VINS velocity at t_k, u = world up, l = u x f),
        VINS step increment and VINS velocity in H_k, derived motion features.
    labels (GT, training/evaluation only): true step increment and true velocity rotated into VINS's world frame through
        VINS's own attitude (g_w = R_e R_g^T x_gt_world), expressed in H_k. Rotating through VINS's attitude means attitude /
        yaw errors are NOT part of any label: only translational errors are learned (heading is not observable here).
    """
    refuse_test(seq)
    df = pd.read_csv(steps_path(seq)).copy()
    t, p, q, v = load_vio(seq)
    assert len(t) == len(df) and np.array_equal(t, df.timestamp_ns.values), f"{seq}: steps/vio misaligned"
    gt_t, gp, gq, gv = load_gt(seq)
    i = np.searchsorted(gt_t, t); ok = (i > 0) & (i < len(gt_t)); i = np.clip(i, 1, len(gt_t) - 1)
    t0, t1 = gt_t[i-1], gt_t[i]; ok &= (t1 - t0) <= 20_000_000; u = ((t - t0) / np.maximum(t1 - t0, 1)).astype(float)
    qg = slerp(gq[i-1], gq[i], u); vg = gv[i-1] + u[:, None] * (gv[i] - gv[i-1])
    Re, Rg = q2R(q), q2R(qg); n = len(t)
    dt = np.r_[0.0, np.diff(t) / NS]
    # heading frame (causal: VINS velocity at t_k)
    f = np.zeros((n, 3)); last = np.array([1.0, 0, 0])
    for k in range(n):
        h = np.array([v[k, 0], v[k, 1], 0.0]); nh = np.linalg.norm(h)
        if nh > 0.05: last = h / nh
        f[k] = last
    up = np.tile([0.0, 0, 1], (n, 1)); l = np.cross(up, f)
    H = np.stack([f, l, up], 2)                                   # columns f, l, u ; world = H @ local
    loc = lambda x: np.einsum("nji,nj->ni", H, x)                 # world -> local
    dp = df[["dpx", "dpy", "dpz"]].values; Rprev = np.concatenate([Re[:1], Re[:-1]])
    # ---- labels: true step increment expressed in VINS's LOCAL TRAJECTORY frame.
    # For every step a yaw-only alignment (+ translation) of the GT path onto the VINS path over a centred 10 s window
    # (LABEL ONLY - uses future GT). Slow heading drift (unobservable) is absorbed by the local yaw; short-term
    # along/lateral/vertical path errors remain in the label. Does not depend on VINS/GT attitude or body-frame calibration.
    pg = gp[i-1] + u[:, None] * (gp[i] - gp[i-1]); pg[~ok] = np.nan
    good = ok & ~np.isnan(pg).any(1); pgf = np.where(good[:, None], pg, 0.0); wgt = good.astype(float)
    cs = lambda x: np.r_[0.0, np.cumsum(x)]
    a = np.clip(np.arange(n) - LOCAL_HALF, 0, n); b = np.clip(np.arange(n) + LOCAL_HALF + 1, 0, n)
    S = lambda x: (cs(x)[b] - cs(x)[a])
    N = S(wgt); mex, mey = S(p[:, 0] * wgt) / np.maximum(N, 1), S(p[:, 1] * wgt) / np.maximum(N, 1)
    mgx, mgy = S(pgf[:, 0]) / np.maximum(N, 1), S(pgf[:, 1]) / np.maximum(N, 1)
    sxx = S(pgf[:, 0] * p[:, 0] * wgt) - N * mgx * mex; syy = S(pgf[:, 1] * p[:, 1] * wgt) - N * mgy * mey
    sxy = S(pgf[:, 0] * p[:, 1] * wgt) - N * mgx * mey; syx = S(pgf[:, 1] * p[:, 0] * wgt) - N * mgy * mex
    psi = np.arctan2(sxy - syx, sxx + syy)                      # GT-world -> VINS-world local yaw
    dpg = np.r_[[[np.nan] * 3], np.diff(pg, axis=0)]
    c_, s_ = np.cos(psi), np.sin(psi)
    gw = np.stack([c_ * dpg[:, 0] - s_ * dpg[:, 1], s_ * dpg[:, 0] + c_ * dpg[:, 1], dpg[:, 2]], 1)
    gw[N < LOCAL_HALF] = np.nan                                 # need at least half a window of GT
    # diagnostic only (NOT a model target): VINS state velocity vs GT velocity, rotated by the same local yaw
    vgw = np.stack([c_ * vg[:, 0] - s_ * vg[:, 1], s_ * vg[:, 0] + c_ * vg[:, 1], vg[:, 2]], 1)
    E, G, VE, VG = loc(dp), loc(gw), loc(v), loc(vgw)
    for j, c in enumerate("flu"):
        df[f"e_{c}"] = E[:, j]; df[f"g_{c}"] = G[:, j]; df[f"ve_{c}"] = VE[:, j]; df[f"vg_{c}"] = VG[:, j]
        df[f"H_{c}x"], df[f"H_{c}y"], df[f"H_{c}z"] = H[:, 0, j], H[:, 1, j], H[:, 2, j]
    df["dt"] = dt; df["local_yaw"] = psi
    df.loc[~ok, [f"vg_{c}" for c in "flu"]] = np.nan
    # derived causal features (VINS only, samples <= t_k)
    sp = df.step_len.values / np.maximum(np.r_[0.1, np.diff(t) / NS], 1e-3); sp[0] = 0
    df = df.copy(); df["d_speed"] = sp; d_ds = np.r_[0, np.diff(sp)] / 0.1
    df["d_dspeed_1s"] = pd.Series(d_ds).rolling(10, min_periods=1).mean().values
    df["d_dspeed_3s"] = pd.Series(d_ds).rolling(30, min_periods=1).mean().values
    df["d_speed_1s"] = df.d_speed.rolling(10, min_periods=1).mean()
    df["d_vh"] = np.linalg.norm(v[:, :2], axis=1); df["d_vz"] = v[:, 2]
    dR = np.einsum("nji,njk->nik", Rprev, Re); w_world = np.einsum("nij,nj->ni", Re, rotvec(dR))
    yr = np.where(dt > 0, w_world[:, 2] / np.maximum(dt, 1e-3), 0.0)
    df["d_yawrate_1s"] = pd.Series(yr).rolling(10, min_periods=1).mean().values
    df["d_centrip_1s"] = df.d_vh * df.d_yawrate_1s
    return df


def load_local(seq):
    refuse_test(seq)
    return pd.read_csv(P2("data", f"{seq}_local.csv"))


# ------------------------------------------------------------------ feature sets
def feature_sets(df):
    base = [c for c in df.columns if c.startswith("f_") and c not in EXCLUDED]
    iv0 = [c for c in base if c.startswith(("f_imu_", "f_vins_", "f_kf_", "f_cons_"))]
    return {"IV_p1": iv0 + DERIVED_P1,                                       # exact phase-1 'ML_imu_vins' features
            "IV": iv0 + DERIVED_P1 + DERIVED_P2,                             # IMU + VINS state + derived motion
            "IVT": iv0 + [c for c in base if c.startswith("f_trk_")] + DERIVED_P1 + DERIVED_P2,
            "ALL": base + DERIVED_P1 + DERIVED_P2}


# ------------------------------------------------------------------ targets
TARGETS = {
    "T1": dict(desc="along-track scale over a centred 3 s window (multiplicative; phase-1 target)", comps=["s"]),
    "T2": dict(desc="along-track displacement residual rate [m/s], centred 3 s mean, additive along f", comps=["f"]),
    "T3": dict(desc="local-frame displacement residual rate [forward, lateral, vertical] m/s, centred 3 s mean", comps=["f", "l", "u"]),
    "T4": dict(desc="local-frame velocity residual [dv_f, dv_l, dv_u] m/s, instantaneous (finite-difference velocity over one 0.1 s step)", comps=["f", "l", "u"]),
}


def labels(df, target, s_fix):
    """Label matrix Y (n x d), weights w (n), valid mask. Residuals are relative to the fixed-scale motion s_fix * VINS,
    so a prediction of 0 (T2-T4) or s_fix (T1) reproduces the fixed-scale baseline exactly."""
    n = len(df)
    if target == "T1":
        y = df.s_c3.clip(0.9, 1.2).values[:, None]; w = df.w_c3.values.copy(); valid = (w > 0) & ~np.isnan(y[:, 0])
        return y, w, valid
    dt = df.dt.values; okdt = dt > 0.05
    if target in ("T2", "T3", "T4"):
        comps = TARGETS[target]["comps"]; Y = np.full((n, len(comps)), np.nan)
        for j, c in enumerate(comps):
            r = np.where(okdt, (df[f"g_{c}"].values - s_fix * df[f"e_{c}"].values) / np.where(okdt, dt, 1), np.nan)
            Y[:, j] = r if target == "T4" else pd.Series(r).rolling(WIN3, center=True, min_periods=WIN3).mean().values
    elif target == "T4v":                     # diagnostic only: residual of VINS STATE velocity (inconsistent with position integration)
        Y = np.stack([df[f"vg_{c}"].values - s_fix * df[f"ve_{c}"].values for c in "flu"], 1)
    else:
        raise KeyError(target)
    valid = ~np.isnan(Y).any(1) & (df.gt_ok.values == 1)
    Y = np.clip(Y, -RATE_CLIP, RATE_CLIP)
    return Y, np.ones(n), valid


def apply_correction(df, target, pred, s_fix, lam):
    """Corrected world-frame step increments. pred: n x d model output (NaN -> no correction beyond fixed)."""
    dp = df[["dpx", "dpy", "dpz"]].values
    if target == "baseline_vins": return dp
    if target == "baseline_fixed": return s_fix * dp
    pred = np.asarray(pred, float).reshape(len(df), -1)
    if target == "T1":
        s = s_fix + lam * (np.nan_to_num(pred[:, 0], nan=s_fix) - s_fix); return s[:, None] * dp
    comps = ["f", "l", "u"] if target == "T4v" else TARGETS[target]["comps"]; corr = np.zeros_like(dp)
    for j, c in enumerate(comps):
        Hc = df[[f"H_{c}x", f"H_{c}y", f"H_{c}z"]].values
        corr += (lam * df.dt.values * np.nan_to_num(pred[:, j]))[:, None] * Hc
    return s_fix * dp + corr


def integrate(df, dpc):
    return df[["px", "py", "pz"]].values[0] + np.cumsum(dpc, axis=0)


# ------------------------------------------------------------------ evaluation (same alignment as the baseline evaluator)
_gt = {}


def evaluate(df, Pc):
    """ATE after SE3 Umeyama (no scale), GT matched by nearest timestamp < 10 ms. Relative drift over 1/3/5 s after alignment,
    3 s drift split into along-track / lateral (horizontal, perpendicular) / vertical as % of GT distance."""
    seq = df.sequence.iloc[0]
    if seq not in _gt: _gt[seq] = load_gt(seq)
    gt, gp, _, _ = _gt[seq]; t = df.timestamp_ns.values
    i = np.clip(np.searchsorted(gt, t), 1, len(gt) - 1); i = np.where(np.abs(gt[i - 1] - t) < np.abs(gt[i] - t), i - 1, i); ok = np.abs(gt[i] - t) < 10_000_000
    A, B = Pc[ok], gp[i[ok]]; _, R, tr = umeyama(A, B); Aa = (R @ A.T).T + tr; Ev = Aa - B; E = np.linalg.norm(Ev, axis=1); sc, _, _ = umeyama(A, B, True)
    out = dict(ate=float(np.sqrt(np.mean(E ** 2))), max_err=float(E.max()), final_err=float(E[-1]), scale=float(sc),
               ate_horiz=float(np.sqrt(np.mean(np.sum(Ev[:, :2] ** 2, 1)))), ate_vert=float(np.sqrt(np.mean(Ev[:, 2] ** 2))))
    tt = t[ok]
    for W in (1, 3, 5):
        j = np.searchsorted(tt, tt - W * NS); m = (tt - tt[j]) >= W * NS - 60_000_000
        e = (Aa[m] - Aa[j[m]]) - (B[m] - B[j[m]]); out[f"rpe{W}s"] = float(np.linalg.norm(e, axis=1).mean())
        if W == 3:
            g = B[m] - B[j[m]]; L = np.linalg.norm(g, axis=1); mv = L > 0.3; ug = g[mv] / L[mv, None]; ee = e[mv]
            al = np.einsum("ni,ni->n", ee, ug)
            perp = ee - al[:, None] * ug; zhat = np.array([0, 0, 1.0])
            lat_dir = np.cross(zhat, ug); ln = np.linalg.norm(lat_dir, axis=1); lat_dir = lat_dir / np.maximum(ln, 1e-9)[:, None]
            lat = np.abs(np.einsum("ni,ni->n", perp, lat_dir)); vert = np.abs(ee[:, 2] - al * ug[:, 2])
            Ls = L[mv].sum()
            out["along_pct"] = float(100 * al.sum() / Ls); out["cross_pct"] = float(100 * np.linalg.norm(perp, axis=1).sum() / Ls)
            out["lateral_pct"] = float(100 * lat.sum() / Ls); out["vertical_pct"] = float(100 * vert.sum() / Ls)
    return out


def save_json(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p, "w"), indent=1, default=float)
