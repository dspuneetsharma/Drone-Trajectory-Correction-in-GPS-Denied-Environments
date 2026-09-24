#!/usr/bin/env python3
"""Step-level dataset for the ML residual-correction layer (one row per VINS output, 10 Hz).
Usage: python3 ml/correction/src/c01_build_steps.py [SEQ ...]      (default MH_01..MH_04)
MH_05 is refused unless ml/correction/FINAL_TEST_APPROVED exists (created only after the user approves the final test).

Row k contains
  * VINS increment of step k: world dp_e = p_e(k)-p_e(k-1) and its body-frame version d_e = R_e(k-1)^T dp_e   (runtime-available)
  * causal features at t_k (same definitions as ml/data, computed from data stamped <= t_k)          (runtime-available)
  * labels (ground truth, training/evaluation only):
      s_step        = (d_e . d_g) / |d_e|^2 for the single step (noisy)
      s_c1, s_c3    = along-track scale over a centred 1 s / 3 s window: (D_e . D_g)/|D_e|^2, D = body-frame displacement over the window
      w_c1, w_c3    = |D_e|^2 (weight: least-squares on displacement, so near-stationary windows do not dominate)
      d_g (step)    = GT body-frame increment for 3-D residual targets
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ml", "src"))
from common import load_vio, load_gt, interp_gt, q2R
import build_dataset as B
NS = 1_000_000_000
SEQS = sys.argv[1:] or ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult"]
if "MH_05_difficult" in SEQS and not os.path.exists(os.path.join(ROOT, "ml", "correction", "FINAL_TEST_APPROVED")):
    sys.exit("MH_05 is locked: final test not yet approved.")

def window_scale(t, p, R, pg, Rg, W):
    """centred window [t-W/2, t+W/2]: along-track scale label and weight"""
    n = len(t); h = int(W / 2 * NS)
    a = np.searchsorted(t, t - h, side="left"); b = np.minimum(np.searchsorted(t, t + h, side="right") - 1, n - 1)
    De = np.einsum("nji,nj->ni", R[a], p[b] - p[a]); Dg = np.einsum("nji,nj->ni", Rg[a], pg[b] - pg[a])
    w = np.sum(De * De, 1); s = np.sum(De * Dg, 1) / np.maximum(w, 1e-12)
    full = (t[b] - t[a]) >= 2 * h - 60_000_000
    s[~full | np.isnan(s)] = np.nan; w[~full] = 0
    return s, np.nan_to_num(w)

for seq in SEQS:
    inp = B.load_inputs(seq); t, p, q, v = inp["t"], inp["p"], inp["q"], inp["v"]
    R = q2R(q); gt, gp, gq = load_gt(seq); pg, qg, ok = interp_gt(gt, gp, gq, t); Rg = q2R(np.where(np.isnan(qg), 1.0, qg))
    F = B.features(inp, t)                                           # causal features for every step (incl. warm-up; NaN where no history)
    dp = np.r_[[[0, 0, 0]], np.diff(p, axis=0)]; de = np.r_[[[0, 0, 0]], np.einsum("nji,nj->ni", R[:-1], np.diff(p, axis=0))]
    dg = np.r_[[[np.nan] * 3], np.einsum("nji,nj->ni", Rg[:-1], np.diff(pg, axis=0))]
    ee = np.sum(de * de, 1); s_step = np.where(ee > 1e-8, np.sum(de * dg, 1) / np.maximum(ee, 1e-12), np.nan)
    s1, w1 = window_scale(t, p, R, pg, Rg, 1.0); s3, w3 = window_scale(t, p, R, pg, Rg, 3.0)
    df = pd.DataFrame(dict(sequence=seq, k=np.arange(len(t)), timestamp_ns=t, t_s=(t - t[0]) / NS,
                           px=p[:, 0], py=p[:, 1], pz=p[:, 2], dpx=dp[:, 0], dpy=dp[:, 1], dpz=dp[:, 2],
                           dex=de[:, 0], dey=de[:, 1], dez=de[:, 2], step_len=np.sqrt(ee),
                           gt_ok=ok.astype(int), dgx=dg[:, 0], dgy=dg[:, 1], dgz=dg[:, 2],
                           s_step=s_step, s_c1=s1, w_c1=w1, s_c3=s3, w_c3=w3))
    df = pd.concat([df, F.reset_index(drop=True)], axis=1)
    df["in_flight"] = (df["f_vins_dist_3s"] > 0.5).astype(int)
    out = os.path.join(ROOT, "ml", "correction", "data", f"{seq}_steps.csv"); df.to_csv(out, index=False, float_format="%.8g")
    m = df.w_c1 > 0
    print(f"{seq}: {len(df)} steps, labelled {int((~df.s_c1.isna()).sum())}, weighted-mean s_c1 {np.average(df.s_c1[m].fillna(1), weights=df.w_c1[m]):.4f}")
