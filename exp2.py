import os
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
import datetime
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class _NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
data_folder = Path('/home/Datasets')
folders = [f for f in data_folder.iterdir()
           if f.is_dir() and (f / "singleLetters.mat").exists()]

def parse_date(name):
    p = name.split(".")
    return datetime.date(int(p[1]), int(p[2]), int(p[3]))

dataset_paths = {f"day{i+1}_data": f / "singleLetters.mat"
                 for i, f in enumerate(sorted(folders, key=lambda f: parse_date(f.name)))}

GO_BIN, T_BINS, SIGMA = 56, 100, 16

def load_day(path):
    mat  = sio.loadmat(path, squeeze_me=True)
    data = {k.split("_")[-1]: np.asarray(v).astype(np.float32)
            for k, v in mat.items() if k.startswith("neuralActivityCube_")}
    processed = {}
    for c, cube in data.items():
        cube_smooth = gaussian_filter1d(cube.astype(float), sigma=SIGMA,
                                        axis=1, truncate=6)
        processed[c] = cube_smooth[:, GO_BIN:GO_BIN + T_BINS, :]
    X, y, lmap = [], [], {}
    for token in sorted(processed):
        if token == 'doNothing': continue
        if token not in lmap: lmap[token] = len(lmap)
        X.append(processed[token])
        y.append(np.full(processed[token].shape[0], lmap[token], dtype=np.int32))
    return np.concatenate(X), np.concatenate(y)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CHANNEL SELECTION  — raw temporal variance, top-45
# ══════════════════════════════════════════════════════════════════════════════
def select_channels(X: np.ndarray, n: int = 45) -> np.ndarray:
    """
    X shape: (Trials, Time, Channels)
    Score = trial-averaged temporal variance per channel.
    """
    var_per_ch = np.mean(np.var(X, axis=1), axis=0)   # (n_channels,)
    return np.argsort(-var_per_ch)[:n]

# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
def lasso_logreg_multi_test(X_tr_scaled, y_tr, test_sets_scaled, chan_map, C_lasso=0.5):
    scores_dict = {d: np.zeros((len(y_te), 31))
                   for d, (_, y_te) in test_sets_scaled.items()}
    feature_counts, selected_features_dict = [], {}
    L = len(chan_map)

    for cls in np.unique(y_tr):
        lr1 = LogisticRegression(penalty='l1', solver='liblinear', C=C_lasso,
                                 class_weight='balanced', random_state=42, max_iter=500)
        # lr1 = LogisticRegression(penalty='l1', solver='liblinear', C=C_lasso,
        #                     class_weight='balanced', max_iter=500)
        lr1.fit(X_tr_scaled, (y_tr == cls).astype(int))
        f = np.where(lr1.coef_[0] != 0)[0]
        if len(f) == 0:
            f = np.arange(min(10, X_tr_scaled.shape[1]))
        feature_counts.append(len(f))
        selected_features_dict[int(cls)] = [
            (int(idx // L), int(chan_map[idx % L])) for idx in f
        ]
        lr2 = LogisticRegression(solver='lbfgs', C=1/20, max_iter=500,
                                 class_weight='balanced', random_state=42)
        # lr2 = LogisticRegression(solver='lbfgs', C=1/20, max_iter=500,
        #                     class_weight='balanced')
        lr2.fit(X_tr_scaled[:, f], (y_tr == cls).astype(int))
        for d, (X_te_scaled, _) in test_sets_scaled.items():
            scores_dict[d][:, cls] = lr2.predict_proba(X_te_scaled[:, f])[:, 1]

    acc_results = {}
    for d, (_, y_te) in test_sets_scaled.items():
        preds = np.argmax(scores_dict[d], axis=1)
        acc_results[d] = float(np.mean(preds == y_te) * 100)

    return acc_results, float(np.mean(feature_counts)), selected_features_dict

# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("Loading all 10 days…")
    all_X, all_y = {}, {}
    for d in range(1, 11):
        key = f"day{d}_data"
        if key in dataset_paths:
            all_X[d], all_y[d] = load_day(dataset_paths[key])

    available_days  = sorted(all_X.keys())
    methods = ['Separate', 'ChunkPool', 'TotalPool']

    acc_res  = {m: {d: [] for d in available_days} for m in methods}
    feat_res = {m: {d: [] for d in available_days} for m in methods}
    exported_features = {m: {} for m in methods}

    chunk_groups = [[1], [2], [3, 4, 5, 6], [7, 8, 9, 10]]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    day_folds = {d: list(skf.split(all_X[d], all_y[d])) for d in available_days}

    print(f"\n{'='*60}\n  5-FOLD CROSS-DAY EVALUATION\n{'='*60}")

    for f_idx in range(5):
        print(f"\n--- Fold {f_idx+1}/5 ---")
        tr_X, te_X, tr_y, te_y = {}, {}, {}, {}
        for d in available_days:
            ti, ei    = day_folds[d][f_idx]
            tr_X[d]   = all_X[d][ti]; te_X[d]  = all_X[d][ei]
            tr_y[d]   = all_y[d][ti]; te_y[d]  = all_y[d][ei]

        # ── SEPARATE ─────────────────────────────────────────────────────
        print("  → Separate…")
        for d in available_days:
            chans     = select_channels(tr_X[d], n=45)
            X_tr_flat = tr_X[d][:, :, chans].reshape(len(tr_y[d]), -1)
            X_te_flat = te_X[d][:, :, chans].reshape(len(te_y[d]), -1)
            sc = StandardScaler()
            acc_d, avg_f, feats_d = lasso_logreg_multi_test(
                sc.fit_transform(X_tr_flat), tr_y[d],
                {d: (sc.transform(X_te_flat), te_y[d])}, chan_map=chans)
            acc_res['Separate'][d].append(acc_d[d])
            feat_res['Separate'][d].append(avg_f)
            exported_features['Separate'].setdefault(
                f"day{d}", {})[f"fold{f_idx+1}"] = feats_d

        # ── CHUNK POOLING ─────────────────────────────────────────────────
        print("  → Chunk Pooling…")
        for group in chunk_groups:
            group = [d for d in group if d in available_days]
            if not group: continue
            X_pool_raw = np.concatenate([tr_X[d] for d in group])
            y_pool     = np.concatenate([tr_y[d] for d in group])
            chans      = select_channels(X_pool_raw, n=45)
            tr_sc_list, test_sets_sc = [], {}
            for d in group:
                X_tr_flat = tr_X[d][:, :, chans].reshape(len(tr_y[d]), -1)
                X_te_flat = te_X[d][:, :, chans].reshape(len(te_y[d]), -1)
                sc = StandardScaler()
                tr_sc_list.append(sc.fit_transform(X_tr_flat))
                test_sets_sc[d] = (sc.transform(X_te_flat), te_y[d])
            acc_d, avg_f, feats_d = lasso_logreg_multi_test(
                np.concatenate(tr_sc_list), y_pool, test_sets_sc, chan_map=chans)
            cname = f"chunk_{'_'.join(map(str,group))}"
            exported_features['ChunkPool'].setdefault(
                cname, {})[f"fold{f_idx+1}"] = feats_d
            for d in group:
                acc_res['ChunkPool'][d].append(acc_d[d])
                feat_res['ChunkPool'][d].append(avg_f)

        # ── TOTAL POOLING ─────────────────────────────────────────────────
        print("  → Total Pooling…")
        X_total_raw = np.concatenate([tr_X[d] for d in available_days])
        y_total     = np.concatenate([tr_y[d] for d in available_days])
        chans       = select_channels(X_total_raw, n=45)
        tr_sc_list, test_sets_sc = [], {}
        for d in available_days:
            X_tr_flat = tr_X[d][:, :, chans].reshape(len(tr_y[d]), -1)
            X_te_flat = te_X[d][:, :, chans].reshape(len(te_y[d]), -1)
            sc = StandardScaler()
            tr_sc_list.append(sc.fit_transform(X_tr_flat))
            test_sets_sc[d] = (sc.transform(X_te_flat), te_y[d])
        acc_d, avg_f, feats_d = lasso_logreg_multi_test(
            np.concatenate(tr_sc_list), y_total, test_sets_sc, chan_map=chans)
        exported_features['TotalPool'].setdefault(
            "all_days", {})[f"fold{f_idx+1}"] = feats_d
        for d in available_days:
            acc_res['TotalPool'][d].append(acc_d[d])
            feat_res['TotalPool'][d].append(avg_f)

