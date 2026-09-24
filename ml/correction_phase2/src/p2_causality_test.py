#!/usr/bin/env python3
"""Causality test: every runtime column built by p2_common.build_local at step K must be identical when the inputs are
truncated at K (i.e. it cannot depend on future samples). f_* features were verified by the same truncation test in ml/src."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_common as pc
RUNTIME = ["d_speed", "d_dspeed_1s", "d_dspeed_3s", "d_speed_1s", "d_vh", "d_vz", "d_yawrate_1s", "d_centrip_1s",
           "H_fx", "H_fy", "H_lx", "H_ly", "e_f", "e_l", "e_u", "dt"]
out = []
for seq in pc.DEV:
    full = pc.build_local(seq); steps = pd.read_csv(pc.steps_path(seq)); t, p, q, v = pc.load_vio(seq); os_, ov = pc.steps_path, pc.load_vio
    tmp = pc.P2("cache", "_trunc.csv")
    for K in np.linspace(60, len(full) - 2, 5).astype(int):
        steps.iloc[:K + 1].to_csv(tmp, index=False)
        pc.steps_path = lambda s: tmp; pc.load_vio = lambda s, K=K: (t[:K + 1], p[:K + 1], q[:K + 1], v[:K + 1])
        tr = pc.build_local(seq); pc.steps_path, pc.load_vio = os_, ov
        out.append(dict(sequence=seq, K=int(K), max_abs_diff=float(np.nanmax(np.abs(tr[RUNTIME].values[-1] - full[RUNTIME].values[K])))))
R = pd.DataFrame(out); R.to_csv(pc.P2("results", "causality_test.csv"), index=False)
print(R.to_string(index=False)); print("CAUSALITY TEST", "PASS" if (R.max_abs_diff < 1e-9).all() else "FAIL")
