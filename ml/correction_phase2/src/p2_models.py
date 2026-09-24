"""Model backends for phase 2: gradient-boosted trees (HGB / LightGBM / XGBoost / CatBoost) and small causal GRU / TCN.
Every backend maps runtime features (<= t_k) to the correction target; one tree model per target component."""
import time, json, numpy as np

CONFIGS = {   # compact, fixed candidates (no large search). 'A' = shallow/strongly regularised, 'B' = slightly deeper
    "hgb": {"A": dict(max_depth=2, learning_rate=0.1, min_samples_leaf=200, max_iter=200, l2_regularization=1.0),
            "B": dict(max_depth=3, learning_rate=0.05, min_samples_leaf=100, max_iter=300, l2_regularization=1.0)},
    "lgbm": {"A": dict(num_leaves=4, max_depth=2, learning_rate=0.1, n_estimators=200, min_child_samples=200, reg_lambda=1.0),
             "B": dict(num_leaves=8, max_depth=3, learning_rate=0.05, n_estimators=300, min_child_samples=100, reg_lambda=1.0,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8)},
    "xgb": {"A": dict(max_depth=2, learning_rate=0.1, n_estimators=200, min_child_weight=200, reg_lambda=1.0),
            "B": dict(max_depth=3, learning_rate=0.05, n_estimators=300, min_child_weight=100, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8)},
    "cat": {"A": dict(depth=2, learning_rate=0.1, iterations=200, l2_leaf_reg=3.0),
            "B": dict(depth=4, learning_rate=0.05, iterations=400, l2_leaf_reg=3.0, rsm=0.8)},
}
TREE_BACKENDS = ["hgb", "lgbm", "xgb", "cat"]
TEMPORAL = {"gru": dict(hidden=32, dropout=0.1), "tcn": dict(channels=32, kernel=3, dilations=(1, 2, 4, 8), dropout=0.1)}
WINDOWS = {"1s": 10, "3s": 30, "5s": 50}
TRAIN_CFG = dict(lr=1e-3, weight_decay=1e-4, batch=256, max_epochs=80, patience=10, holdout_frac=0.2)
SEEDS_SCREEN, SEEDS_FULL = [0], [0, 1, 2]


def backend_available(name):
    mod = {"hgb": "sklearn", "lgbm": "lightgbm", "xgb": "xgboost", "cat": "catboost", "gru": "torch", "tcn": "torch"}[name]
    try: __import__(mod); return True
    except Exception: return False


def versions():
    out = {}
    for m in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "torch", "matplotlib", "joblib"]:
        try: out[m] = __import__(m).__version__
        except Exception as e: out[m] = f"MISSING ({type(e).__name__})"
    return out


def _tree(name, params):
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=0, **params)
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=0, n_jobs=4, verbose=-1, **params)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=0, n_jobs=4, tree_method="hist", **params)
    if name == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=0, thread_count=4, verbose=0, allow_writing_files=False, **params)
    raise KeyError(name)


def fit_trees(name, cfg, X, Y, w):
    """one model per target component. Raw sample weights (reproduces phase 1 exactly for T1); for XGBoost the weights are
    normalised to mean 1 so that min_child_weight means 'number of samples' as in the other libraries."""
    if name == "xgb": w = w / w.mean()
    models = []
    for j in range(Y.shape[1]):
        m = _tree(name, CONFIGS[name][cfg]); m.fit(X, Y[:, j], sample_weight=w); models.append(m)
    return models


def predict_trees(models, X):
    return np.stack([m.predict(X) for m in models], 1)


# ------------------------------------------------------------------ temporal (PyTorch)
def torch_device(allow_cpu=False):
    import torch
    if torch.cuda.is_available(): return torch.device("cuda")
    if allow_cpu: return torch.device("cpu")
    raise RuntimeError("CUDA not available to PyTorch - temporal models refuse to run on CPU (pass --allow-cpu-temporal to override explicitly)")


def _build(arch, n_in, n_out):
    import torch, torch.nn as nn
    c = TEMPORAL[arch]
    if arch == "gru":
        class GRUNet(nn.Module):
            def __init__(s):
                super().__init__(); s.drop = nn.Dropout(c["dropout"]); s.rnn = nn.GRU(n_in, c["hidden"], batch_first=True); s.head = nn.Linear(c["hidden"], n_out)
            def forward(s, x): o, _ = s.rnn(s.drop(x)); return s.head(o[:, -1])
        return GRUNet()

    class Block(nn.Module):
        def __init__(s, ci, co, d):
            super().__init__(); s.pad = (c["kernel"] - 1) * d
            s.conv = nn.Conv1d(ci, co, c["kernel"], dilation=d); s.act = nn.ReLU(); s.drop = nn.Dropout(c["dropout"])
            s.res = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()
        def forward(s, x):                                   # x: (B, C, L); left padding only -> causal
            y = s.drop(s.act(s.conv(nn.functional.pad(x, (s.pad, 0))))); return s.act(y + s.res(x))

    class TCN(nn.Module):
        def __init__(s):
            super().__init__(); chans = c["channels"]; layers = []; ci = n_in
            for d in c["dilations"]: layers.append(Block(ci, chans, d)); ci = chans
            s.net = nn.Sequential(*layers); s.head = nn.Linear(chans, n_out)
        def forward(s, x): return s.head(s.net(x.transpose(1, 2))[:, :, -1])
    return TCN()


class SeqFolds:
    """Per-flight standardised feature arrays; windows never cross flights; first rows padded by repetition (causal)."""
    def __init__(self, flights_X, feats_train_rows):
        allX = np.concatenate(feats_train_rows)
        self.med = np.nanmedian(allX, 0); self.med = np.where(np.isnan(self.med), 0, self.med)
        Z = np.where(np.isnan(allX), self.med, allX); self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-6
        self.X = {k: ((np.where(np.isnan(v), self.med, v) - self.mu) / self.sd).astype(np.float32) for k, v in flights_X.items()}


def _windows(torch, Xf, idx, L, dev):
    Xp = np.concatenate([np.repeat(Xf[:1], L - 1, 0), Xf], 0)
    T = torch.from_numpy(Xp).to(dev).unfold(0, L, 1).transpose(1, 2)   # (n, L, F)
    return T[torch.as_tensor(idx, device=dev)]


def fit_predict_temporal(arch, L, data, train_flights, test_flight, seeds, dev, curve_path=None):
    """data[flight] = (X (n,F), Y (n,d), w (n), valid (n)). Early stopping on the LAST 20% of each TRAINING flight
    (never on the validation / held-out flight). Returns mean prediction over seeds for every row of test_flight."""
    import torch
    torch.backends.cudnn.benchmark = False
    S = SeqFolds({f: data[f][0] for f in train_flights + [test_flight]}, [data[f][0][data[f][3]] for f in train_flights])
    Ytr = np.concatenate([data[f][1][data[f][3]] for f in train_flights]); ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-9
    fit_rows, hold_rows = [], []
    for f in train_flights:
        idx = np.where(data[f][3])[0]; cut = int(len(idx) * (1 - TRAIN_CFG["holdout_frac"]))
        fit_rows.append((f, idx[:cut])); hold_rows.append((f, idx[cut:]))

    def batches(rows):
        Xs = torch.cat([_windows(torch, S.X[f], i, L, dev) for f, i in rows])
        Ys = torch.from_numpy(np.concatenate([(data[f][1][i] - ym) / ys for f, i in rows]).astype(np.float32)).to(dev)
        W = np.concatenate([data[f][2][i] for f, i in rows]); W = torch.from_numpy((W / W.mean()).astype(np.float32)).to(dev)
        return Xs, Ys, W
    Xa, Ya, Wa = batches(fit_rows); Xh, Yh, Wh = batches(hold_rows)
    n_out = Ya.shape[1]; preds, curves, states = [], [], []
    Xt = _windows(torch, S.X[test_flight], np.arange(len(S.X[test_flight])), L, dev)
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        m = _build(arch, Xa.shape[2], n_out).to(dev); opt = torch.optim.Adam(m.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
        best, best_state, bad, curve = np.inf, None, 0, []
        g = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(TRAIN_CFG["max_epochs"]):
            m.train(); perm = torch.randperm(len(Xa), generator=g).to(dev); tl = 0.0
            for b in range(0, len(Xa), TRAIN_CFG["batch"]):
                j = perm[b:b + TRAIN_CFG["batch"]]; opt.zero_grad()
                loss = ((m(Xa[j]) - Ya[j]) ** 2 * Wa[j, None]).mean(); loss.backward(); opt.step(); tl += loss.item() * len(j)
            m.eval()
            with torch.no_grad(): hl = float(((m(Xh) - Yh) ** 2 * Wh[:, None]).mean())
            curve.append(dict(epoch=ep, train=tl / len(Xa), holdout=hl))
            if hl < best - 1e-6: best, bad, best_state = hl, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= TRAIN_CFG["patience"]: break
        m.load_state_dict(best_state); m.eval(); states.append({k: v.cpu() for k, v in best_state.items()})
        with torch.no_grad(): p = torch.cat([m(Xt[b:b + 4096]) for b in range(0, len(Xt), 4096)]).cpu().numpy()
        preds.append(p * ys + ym); curves.append(dict(seed=seed, epochs=len(curve), best_holdout=best, curve=curve))
    if curve_path:
        json.dump(curves, open(curve_path, "w"))
    bundle = dict(arch=arch, L=L, n_in=int(Xa.shape[2]), n_out=int(n_out), med=S.med, mu=S.mu, sd=S.sd, ym=ym, ys=ys, seeds=list(seeds), states=states)
    return np.mean(preds, 0), bundle


def predict_temporal(bundle, X, dev):
    import torch
    Xs = ((np.where(np.isnan(X), bundle['med'], X) - bundle['mu']) / bundle['sd']).astype(np.float32)
    Xt = _windows(torch, Xs, np.arange(len(Xs)), bundle['L'], dev); P = []
    for st in bundle['states']:
        m = _build(bundle['arch'], bundle['n_in'], bundle['n_out']).to(dev); m.load_state_dict(st); m.eval()
        with torch.no_grad(): P.append(torch.cat([m(Xt[b:b + 4096]) for b in range(0, len(Xt), 4096)]).cpu().numpy())
    return np.mean(P, 0) * bundle['ys'] + bundle['ym']
