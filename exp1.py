import json
import numpy as np
import scipy.io as sio
from pathlib import Path
import datetime
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class NpEncoder(json.JSONEncoder):
    """Handles numpy ints/floats/arrays in JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ==========================================
# 1. DATA LOADING & PREFILTERING
# ==========================================
data_folder = Path('/home/Datasets')
folders = [f for f in data_folder.iterdir() if f.is_dir() and (f / "singleLetters.mat").exists()]

def parse_date(name):
    p = name.split(".")
    return datetime.date(int(p[1]), int(p[2]), int(p[3]))

dataset_paths = {f"day{i+1}_data": f / "singleLetters.mat"
                 for i, f in enumerate(sorted(folders, key=lambda f: parse_date(f.name)))}

GO_BIN, T_BINS, SIGMA = 56, 100, 16

def load_day(path):
    mat = sio.loadmat(path, squeeze_me=True)
    data = {k.split("_")[-1]: np.asarray(v) for k, v in mat.items() if k.startswith("neuralActivityCube_")}
    smoothed = {c: gaussian_filter1d(a, sigma=SIGMA, axis=1, output=float, truncate=6) for c, a in data.items()}
    X, y, lmap = [], [], {}
    for token in sorted(smoothed):
        if token == 'doNothing': continue
        if token not in lmap: lmap[token] = len(lmap)
        X.append(smoothed[token])
        y.append(np.full(smoothed[token].shape[0], lmap[token], dtype=np.int32))
    return np.concatenate(X)[:, GO_BIN:GO_BIN+T_BINS, :], np.concatenate(y)

# ==========================================
# 2. PURE FEATURE SELECTORS
# ==========================================
def dnc_select_miBMI(X, y, per_class_K=1000, eps=1e-12, chan_map=None, prefix=""):
    n_trials, T_used, L = X.shape
    classes = np.unique(y)
    n_classes = classes.size
    if chan_map is None: chan_map = np.arange(L)

    Xf = X.reshape(n_trials, T_used * L).astype(np.float64)
    F = Xf.shape[1]
    class_list = list(classes)
    mu, var, Pj = {}, {}, {}

    for c in class_list:
        Xc = Xf[y == c]
        count = Xc.shape[0]
        Pj[c] = count / float(n_trials)
        mu[c]  = Xc.mean(axis=0) if count > 0 else np.zeros(F, dtype=np.float64)
        var[c] = np.maximum(Xc.var(axis=0), 0.0) if count > 0 else np.zeros(F, dtype=np.float64)

    per_class_scores = np.zeros((n_classes, F), dtype=np.float64)
    for idx_c, c in enumerate(class_list):
        if prefix: print(f"\r{prefix} [Saliency] | Class {idx_c+1}/{n_classes}", end="", flush=True)
        others = [j for j in class_list if j != c]
        m = len(others)
        D = np.empty((m, F), dtype=np.float64)
        pj_list = np.empty(m, dtype=np.float64)
        for k, j in enumerate(others):
            num   = np.abs(mu[c] - mu[j])
            denom = np.maximum(np.sqrt(Pj[c]*var[c] + Pj[j]*var[j]), eps)
            D[k, :] = np.exp(np.maximum(num / denom, 1e-12))
            pj_list[k] = Pj[j]
        logD = np.log(D)
        weighted_log_prod = (pj_list[:, None] * logD).sum(axis=0)
        denom_lin = (pj_list[:, None] * D).sum(axis=0) + eps
        logzeta = 2.0 * weighted_log_prod - np.log(denom_lin)
        zeta = np.exp(np.clip(logzeta, -700, 700))
        per_class_scores[idx_c, :] = np.nan_to_num(zeta, nan=0.0, posinf=0.0, neginf=0.0)

    per_class_topk = {}
    for idx_c, c in enumerate(class_list):
        if prefix: print(f"\r{prefix} [mRMR]    | Class {idx_c+1}/{n_classes}", end="", flush=True)
        zeta = per_class_scores[idx_c, :]
        K_target = min(per_class_K, F)
        Xc = Xf[y == c]
        if Xc.shape[0] < 2:
            order = np.argsort(-zeta)[:K_target]
        else:
            Xc_std = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + eps)
            N_c = Xc.shape[0]
            selected_local = []
            penalty   = np.ones(F, dtype=np.float64)
            objective = zeta.copy()
            for _ in range(K_target):
                best_idx = np.argmax(objective)
                selected_local.append(best_idx)
                objective[best_idx] = -1.0
                rho = np.abs((Xc_std[:, best_idx] @ Xc_std) / N_c)
                penalty *= (1.0 - rho)
                np.multiply(zeta, penalty, out=objective)
                for idx in selected_local:
                    objective[idx] = -1.0
            order = selected_local
        per_class_topk[int(c)] = [(int(lf // L), int(chan_map[lf % L])) for lf in order]

    if prefix: print(f"\r{prefix} | Done.                                        ", flush=True)
    return per_class_topk

def xgb_select_miBMI_analysis(X, y, n_bootstraps=3, per_class_K=1000, random_state=42, chan_map=None, prefix=""):
    classes = np.unique(y)
    n_trials, T_used, L = X.shape
    if chan_map is None: chan_map = np.arange(L)
    Xf = X.reshape(n_trials, -1).astype(np.float32)
    F  = Xf.shape[1]

    per_class_topk = {}
    params = {
        "objective": "binary:logistic", "max_depth": 4, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 1,
        "tree_method": "hist", "device": "cuda", "verbosity": 0
    }

    for i, c in enumerate(classes):
        if prefix: print(f"\r{prefix} | Class {i+1}/{len(classes)}", end="", flush=True)
        y_bin  = (y == c).astype(np.float32)
        dtrain = xgb.DMatrix(Xf, label=y_bin)
        avg_imp = np.zeros(F, dtype=np.float32)
        for b in range(n_bootstraps):
            params["seed"] = int(random_state) + (b * 100)
            booster   = xgb.train(params, dtrain, num_boost_round=300)
            gain_dict = booster.get_score(importance_type="gain")
            for k, v in gain_dict.items():
                idx = int(k[1:])
                if idx < F: avg_imp[idx] += float(v)
        avg_imp /= n_bootstraps
        nz = np.nonzero(avg_imp > 0.0)[0]
        selected_idx = nz[np.argsort(-avg_imp[nz])][:per_class_K] if len(nz) > 0 else []
        per_class_topk[int(c)] = [(int(lf // L), int(chan_map[lf % L])) for lf in selected_idx]

    if prefix: print(f"\r{prefix} | Done.                                        ", flush=True)
    return per_class_topk

def lasso_select_features(X_tr_flat_sc, y_tr, L, C_lasso=0.5, chan_map=None, prefix=""):
    if chan_map is None: chan_map = np.arange(L)
    feature_counts = []
    per_class_feats = {}
    classes = np.unique(y_tr)
    for i, cls in enumerate(classes):
        if prefix: print(f"\r{prefix} | Class {i+1}/{len(classes)}", end="", flush=True)
        lr1 = LogisticRegression(penalty='l1', solver='liblinear', C=C_lasso,
                                 class_weight='balanced', max_iter=500)
        lr1.fit(X_tr_flat_sc, (y_tr == cls).astype(int))
        f = np.where(lr1.coef_[0] != 0)[0]
        if len(f) == 0: f = np.arange(min(10, X_tr_flat_sc.shape[1]))
        feature_counts.append(len(f))
        per_class_feats[int(cls)] = [(int(idx // L), int(chan_map[idx % L])) for idx in f]
    if prefix: print(f"\r{prefix} | Done. (Avg K: {np.mean(feature_counts):.1f})                 ", flush=True)
    return per_class_feats, np.mean(feature_counts)

# ==========================================
# 3. HIGH-SPEED CLASSIFIERS
# ==========================================
def lda_train_predict(X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, feats, L):
    classes = np.unique(y_tr)
    n_test  = X_te_flat_sc.shape[0]
    scores  = np.full((n_test, len(classes)), -np.inf)
    for idx_c, c in enumerate(classes):
        f = feats[c]
        if len(f) == 0: continue
        flat_idx = [t * L + chan for t, chan in f]
        V_tr, V_te = X_tr_flat_sc[:, flat_idx], X_te_flat_sc[:, flat_idx]
        pos_mask = (y_tr == c)
        n_pos, n_neg = pos_mask.sum(), len(y_tr) - pos_mask.sum()
        Vs_pos, Vs_neg = V_tr[pos_mask], V_tr[~pos_mask]
        mu_pos, mu_neg = Vs_pos.mean(axis=0), Vs_neg.mean(axis=0)
        S_pos = np.cov(Vs_pos, rowvar=False, ddof=1)
        S_neg = np.cov(Vs_neg, rowvar=False, ddof=1)
        if len(f) == 1:
            Sigma    = np.array([[(n_pos-1)*S_pos + (n_neg-1)*S_neg]]) / (n_pos+n_neg-2) + 20*np.eye(1)
            invSigma = 1.0 / Sigma
        else:
            Sigma    = ((n_pos-1)*S_pos + (n_neg-1)*S_neg) / (n_pos+n_neg-2) + 20*np.eye(len(f))
            invSigma = np.linalg.inv(Sigma)
        w = invSigma.dot(mu_pos - mu_neg)
        b = -0.5 * (mu_pos + mu_neg).dot(w)
        scores[:, idx_c] = V_te.dot(w) + b
    return float(np.mean(np.array([classes[a] for a in np.argmax(scores, axis=1)]) == y_te) * 100)

def logreg_train_predict(X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, feats, L, C_reg=1/20):
    classes = np.unique(y_tr)
    n_test  = X_te_flat_sc.shape[0]
    scores  = np.full((n_test, len(classes)), -np.inf)
    for idx_c, c in enumerate(classes):
        f = feats[c]
        if len(f) == 0: continue
        flat_idx = [t * L + chan for t, chan in f]
        V_tr, V_te = X_tr_flat_sc[:, flat_idx], X_te_flat_sc[:, flat_idx]
        lr = LogisticRegression(penalty='l2', C=C_reg, solver='lbfgs',
                                max_iter=500, class_weight='balanced')
        lr.fit(V_tr, (y_tr == c).astype(int))
        scores[:, idx_c] = V_te.dot(lr.coef_.ravel()) + lr.intercept_[0]
    return float(np.mean(np.array([classes[a] for a in np.argmax(scores, axis=1)]) == y_te) * 100)

# ==========================================
# 4. MAIN SWEEP PIPELINE
# ==========================================
if __name__ == "__main__":
    k_grid   = [10, 50, 128, 250, 500]  
    C_values = [0.1, 0.5, 1.0]
    C_keys   = [f'c{c}' for c in C_values]    
    n_days   = 10
    N_FOLDS  = 5

    # ---------- sweep result arrays ----------
    res = {
        cond: {key: np.zeros((n_days, len(k_grid)))
               for key in ('xgb_lr', 'xgb_lda', 'dnc_lr', 'dnc_lda')}
        for cond in ('all')
    }
    res_folds = {
        cond: {key: np.zeros((n_days, N_FOLDS, len(k_grid)))
               for key in ('xgb_lr', 'xgb_lda', 'dnc_lr', 'dnc_lda')}
        for cond in ('all')
    }

    # ---------- lasso result arrays (per C value) ----------
    def _zeros1(n): return np.zeros(n)
    def _zeros2(n, m): return np.zeros((n, m))

    lasso_res = {
        cond: {
            ck: {
                'lr_acc':        _zeros1(n_days), 'lda_acc':        _zeros1(n_days),
                'k':             _zeros1(n_days),
                'xgb_lr_at_lk':  _zeros1(n_days), 'xgb_lda_at_lk': _zeros1(n_days),
                'dnc_lr_at_lk':  _zeros1(n_days), 'dnc_lda_at_lk': _zeros1(n_days),
            }
            for ck in C_keys
        }
        for cond in ('all')
    }
    lasso_folds = {
        cond: {
            ck: {
                'lr_acc':        _zeros2(n_days, N_FOLDS), 'lda_acc':        _zeros2(n_days, N_FOLDS),
                'xgb_lr_at_lk':  _zeros2(n_days, N_FOLDS), 'xgb_lda_at_lk': _zeros2(n_days, N_FOLDS),
                'dnc_lr_at_lk':  _zeros2(n_days, N_FOLDS), 'dnc_lda_at_lk': _zeros2(n_days, N_FOLDS),
            }
            for ck in C_keys
        }
        for cond in ('all')
    }

    # ---------- feature export ----------
    exported_features = {'no_prefilter': {}}

    # ==========================================
    for day in range(1, n_days + 1):
        day_key = f"day{day}_data"
        if day_key not in dataset_paths: continue

        print(f"\n{'='*80}\n PROCESSING DAY {day}\n{'='*80}")
        X_day, y_day = load_day(dataset_paths[day_key])
        L_all = X_day.shape[2]

        exported_features['no_prefilter'][f"day{day}"]    = {}

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_day, y_day)):
            print(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
            X_tr, X_te = X_day[tr_idx], X_day[te_idx]
            y_tr, y_te = y_day[tr_idx], y_day[te_idx]

            sc_all = StandardScaler()
            X_tr_flat_sc = sc_all.fit_transform(X_tr.reshape(len(y_tr), -1))
            X_te_flat_sc = sc_all.transform    (X_te.reshape(len(y_te), -1))

            # =====================================
            # PHASE 1: ALL CHANNELS
            # =====================================
            print("   [ALL CHANNELS]")

            # --- XGBoost sweep ---
            xgb_all = xgb_select_miBMI_analysis(X_tr, y_tr, per_class_K=1000, prefix="    -> XGB Selection")
            print("    -> XGB Grid: [", end=" ", flush=True)
            for i, K in enumerate(k_grid):
                print(f"{K}", end=" ", flush=True)
                tr_xgb  = {c: f[:K] for c, f in xgb_all.items()}
                acc_lr  = logreg_train_predict(X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, tr_xgb, L=L_all)
                acc_lda = lda_train_predict   (X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, tr_xgb, L=L_all)
                res['all']['xgb_lr'] [day-1, i] += acc_lr  / N_FOLDS
                res['all']['xgb_lda'][day-1, i] += acc_lda / N_FOLDS
                res_folds['all']['xgb_lr'] [day-1, fold, i] = acc_lr
                res_folds['all']['xgb_lda'][day-1, fold, i] = acc_lda
            print("]")

            # --- DNC sweep ---
            dnc_all = dnc_select_miBMI(X_tr, y_tr, per_class_K=1000, prefix="    -> DNC Selection")
            print("    -> DNC Grid: [", end=" ", flush=True)
            for i, K in enumerate(k_grid):
                print(f"{K}", end=" ", flush=True)
                tr_dnc  = {c: f[:K] for c, f in dnc_all.items()}
                acc_lr  = logreg_train_predict(X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, tr_dnc, L=L_all)
                acc_lda = lda_train_predict   (X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, tr_dnc, L=L_all)
                res['all']['dnc_lr'] [day-1, i] += acc_lr  / N_FOLDS
                res['all']['dnc_lda'][day-1, i] += acc_lda / N_FOLDS
                res_folds['all']['dnc_lr'] [day-1, fold, i] = acc_lr
                res_folds['all']['dnc_lda'][day-1, fold, i] = acc_lda
            print("]")

            # --- Lasso C sweep + XGB/DNC at each lasso avg-K ---
            fold_feats_all = {'xgb_1000': xgb_all, 'dnc_1000': dnc_all}

            for C, ck in zip(C_values, C_keys):
                l_feats, l_k = lasso_select_features(
                    X_tr_flat_sc, y_tr, L=L_all, C_lasso=C, prefix=f"    -> Lasso C={C}")

                fold_feats_all[f'lasso_{ck}'] = l_feats   # save lasso features for this C

                acc_lr  = logreg_train_predict(X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, l_feats, L=L_all)
                acc_lda = lda_train_predict   (X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, l_feats, L=L_all)

                lasso_res['all'][ck]['lr_acc'] [day-1] += acc_lr  / N_FOLDS
                lasso_res['all'][ck]['lda_acc'][day-1] += acc_lda / N_FOLDS
                lasso_res['all'][ck]['k']      [day-1] += l_k     / N_FOLDS
                lasso_folds['all'][ck]['lr_acc'] [day-1, fold] = acc_lr
                lasso_folds['all'][ck]['lda_acc'][day-1, fold] = acc_lda

                # XGB and DNC at this C's avg-K
                K_match = max(1, int(round(l_k)))
                print(f"    -> XGB/DNC @LassoK={K_match} (C={C}): ", end="", flush=True)

                xm = {c: f[:K_match] for c, f in xgb_all.items()}
                xm_lr  = logreg_train_predict(X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, xm, L=L_all)
                xm_lda = lda_train_predict   (X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, xm, L=L_all)

                dm = {c: f[:K_match] for c, f in dnc_all.items()}
                dm_lr  = logreg_train_predict(X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, dm, L=L_all)
                dm_lda = lda_train_predict   (X_tr_flat_sc, y_tr, X_te_flat_sc, y_te, dm, L=L_all)

                print(f"XGB LR={xm_lr:.1f} LDA={xm_lda:.1f} | DNC LR={dm_lr:.1f} LDA={dm_lda:.1f}")

                lasso_res['all'][ck]['xgb_lr_at_lk'] [day-1] += xm_lr  / N_FOLDS
                lasso_res['all'][ck]['xgb_lda_at_lk'][day-1] += xm_lda / N_FOLDS
                lasso_res['all'][ck]['dnc_lr_at_lk'] [day-1] += dm_lr  / N_FOLDS
                lasso_res['all'][ck]['dnc_lda_at_lk'][day-1] += dm_lda / N_FOLDS
                lasso_folds['all'][ck]['xgb_lr_at_lk'] [day-1, fold] = xm_lr
                lasso_folds['all'][ck]['xgb_lda_at_lk'][day-1, fold] = xm_lda
                lasso_folds['all'][ck]['dnc_lr_at_lk'] [day-1, fold] = dm_lr
                lasso_folds['all'][ck]['dnc_lda_at_lk'][day-1, fold] = dm_lda

            exported_features['no_prefilter'][f"day{day}"][f"fold{fold+1}"] = fold_feats_all

