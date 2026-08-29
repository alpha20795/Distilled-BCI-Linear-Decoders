import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import datetime
import warnings
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if HAS_MAMBA:
    print("Mamba successfully imported.")
else:
    print("Mamba import failed — falling back to GRU.")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
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
    mat = sio.loadmat(path, squeeze_me=True)
    data = {k.split("_")[-1]: np.asarray(v)
            for k, v in mat.items() if k.startswith("neuralActivityCube_")}
    smoothed = {c: gaussian_filter1d(a, sigma=SIGMA, axis=1, output=float, truncate=6)
                for c, a in data.items()}
    X, y, lmap = [], [], {}
    for token in sorted(smoothed):
        if token == 'doNothing': continue
        if token not in lmap: lmap[token] = len(lmap)
        X.append(smoothed[token])
        y.append(np.full(smoothed[token].shape[0], lmap[token], dtype=np.int32))
    return np.concatenate(X), np.concatenate(y)   

all_X, all_y = {}, {}
for k, v in dataset_paths.items():
    d = int(k.replace('day','').replace('_data',''))
    if 2 <= d <= 9:
        all_X[d], all_y[d] = load_day(v)

# ══════════════════════════════════════════════════════════════════════════════
# 2. AUGMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════
class GaussianNoise(nn.Module):
    def __init__(self, std=0.08): super().__init__(); self.std = std
    def forward(self, x):
        return x + torch.randn_like(x) * self.std if self.training else x

class SpatialDropout1D(nn.Module):
    def __init__(self, p=0.4): super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        mask = torch.empty(x.size(0), 1, x.size(2), device=x.device).bernoulli_(1 - self.p)
        return (x * mask) / (1 - self.p)

def temporal_crop_jitter(x, base_start=GO_BIN, crop_len=T_BINS, max_shift=8):
    if not x.requires_grad:
        return x[:, base_start:base_start + crop_len, :]
    shift = np.random.randint(-max_shift, max_shift + 1)
    return x[:, base_start + shift: base_start + shift + crop_len, :]

def temporal_cutout(x, cutout_len=15):
    if not x.requires_grad:
        return x
    B, T, C = x.shape
    mask = torch.ones((B, T, C), device=x.device)
    for i in range(B):
        start = np.random.randint(0, T - cutout_len)
        mask[i, start:start + cutout_len, :] = 0.0
    return x * mask

def perform_mixup(x, y, d_tensor, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        combined_key = y * 1000 + d_tensor
        sorted_idx = torch.argsort(combined_key)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        d_sorted = d_tensor[sorted_idx]
        x_mixed = lam * x_sorted + (1 - lam) * torch.roll(x_sorted, shifts=1, dims=0)
        same_class = (y_sorted == torch.roll(y_sorted, shifts=1, dims=0))
        same_day   = (d_sorted == torch.roll(d_sorted, shifts=1, dims=0))
        valid_mask = (same_class & same_day).view(-1, 1, 1).float()
        x_final = x_mixed * valid_mask + x_sorted * (1 - valid_mask)
        shuffle_idx = torch.randperm(x.size(0)).to(x.device)
        return x_final[shuffle_idx], y_sorted[shuffle_idx], d_sorted[shuffle_idx]
    return x, y, d_tensor

# ══════════════════════════════════════════════════════════════════════════════
# 3. ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class MambaBlock(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.seq = (Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                    if HAS_MAMBA else nn.GRU(d_model, d_model, batch_first=True))
        self.use_mamba = HAS_MAMBA
    def forward(self, x):
        if self.use_mamba: return self.seq(x)
        out, _ = self.seq(x); return out

class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(),
            nn.Linear(d_model // 2, 1))
    def forward(self, x):
        attn_weights = F.softmax(self.attention_mlp(x), dim=1)
        return torch.sum(x * attn_weights, dim=1)

class SimpleReadIn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, x):
        out = self.conv(x).permute(0, 2, 1)
        return self.norm(out).permute(0, 2, 1)

class Encoder(nn.Module):
    def __init__(self, n_ch=192, embed_dim=128, latent_dim=96,
                 dropout=0.4, noise_std=0.08):
        super().__init__()
        self.noise        = GaussianNoise(std=noise_std)
        self.spatial_drop = SpatialDropout1D(p=dropout)
        half = embed_dim // 2
        self.proj_in_1   = SimpleReadIn(n_ch // 2, half)
        self.proj_in_2   = SimpleReadIn(n_ch // 2, half)
        self.array_mixer = nn.Linear(embed_dim, embed_dim)
        self.mamba1      = MambaBlock(d_model=embed_dim)
        self.mamba2      = MambaBlock(d_model=embed_dim)
        self.norm1       = nn.LayerNorm(embed_dim)
        self.norm2       = nn.LayerNorm(embed_dim)
        self.enc_proj    = nn.Conv1d(embed_dim, latent_dim, kernel_size=1)
        self.bn          = nn.BatchNorm1d(latent_dim)
        self.attn_pool   = TemporalAttentionPooling(d_model=latent_dim)

    def forward(self, x):
        x  = self.spatial_drop(self.noise(x))
        xt = x.permute(0, 2, 1)
        e  = torch.cat([self.proj_in_1(xt[:, :96, :]),
                        self.proj_in_2(xt[:, 96:, :])], dim=1).permute(0, 2, 1)
        et = e + self.array_mixer(e)
        et = et + self.mamba1(self.norm1(et))
        et = et + self.mamba2(self.norm2(et))
        z  = self.bn(self.enc_proj(et.permute(0, 2, 1)))
        return self.attn_pool(z.permute(0, 2, 1))

class LinearClassifier(nn.Module):
    def __init__(self, latent_dim=96, num_classes=31):
        super().__init__()
        self.net = nn.Linear(latent_dim, num_classes)
    def forward(self, z): return self.net(z)

# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTRASTIVE LOSS
# ══════════════════════════════════════════════════════════════════════════════
def cross_day_supcon_loss(z, y_letter, y_day, temperature=0.05):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    same_letter = (y_letter.unsqueeze(0) == y_letter.unsqueeze(1))
    same_day    = (y_day.unsqueeze(0)    == y_day.unsqueeze(1))
    pos_mask    = same_letter & ~same_day
    ignore_mask = same_letter & same_day
    sim         = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim     = torch.exp(sim).masked_fill(ignore_mask, 0.0)
    log_prob    = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    n_pos       = pos_mask.float().sum(dim=1)
    has_pos     = n_pos > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / (n_pos + 1e-8)
    return loss[has_pos].mean()

# ══════════════════════════════════════════════════════════════════════════════
# 5. BATCH BUILDER + TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def build_batch(X_tensors, y_tensors, day_tensors, days, n_per_class=8, num_classes=31):
    X_list, yl_list, yd_list = [], [], []
    for d in days:
        X_d, y_d, yd = X_tensors[d], y_tensors[d], day_tensors[d]
        for c in range(num_classes):
            idx = (y_d == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0: continue
            chosen = idx[torch.randperm(len(idx))[:min(len(idx), n_per_class)]]
            X_list.append(X_d[chosen])
            yl_list.append(y_d[chosen])
            yd_list.append(yd[chosen])
    return torch.cat(X_list), torch.cat(yl_list), torch.cat(yd_list)

def prep(X): return torch.tensor(X, dtype=torch.float32).to(DEVICE)

def train_contrastive(train_X, train_y, days, epochs=800,
                      latent_dim=96, lambda_clf=1.0, lambda_con=2.0,
                      temperature=0.05, n_per_class=8, num_classes=31):
    encoder    = Encoder(latent_dim=latent_dim).to(DEVICE)
    letter_clf = LinearClassifier(latent_dim=latent_dim,
                                  num_classes=num_classes).to(DEVICE)
    optimizer  = optim.AdamW(
        list(encoder.parameters()) + list(letter_clf.parameters()),
        lr=1e-3, weight_decay=1e-3)
    scheduler  = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, steps_per_epoch=1, epochs=epochs)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_tr_tensors = {d: prep(train_X[d]) for d in days}
    y_tr_tensors = {d: torch.tensor(train_y[d], dtype=torch.long).to(DEVICE)
                    for d in days}
    day_tensors  = {d: torch.full((len(train_X[d]),), d,
                                  dtype=torch.long).to(DEVICE)
                    for d in days}

    for epoch in range(epochs):
        encoder.train(); letter_clf.train(); optimizer.zero_grad()

        X_b, yl_b, yd_b = build_batch(
            X_tr_tensors, y_tr_tensors, day_tensors, days,
            n_per_class=n_per_class, num_classes=num_classes)
        X_b.requires_grad_(True)
        X_b = temporal_crop_jitter(X_b, max_shift=8)

        if epoch > 100:
            X_b = temporal_cutout(X_b, cutout_len=15)
            X_b, yl_b, yd_b = perform_mixup(X_b, yl_b, yd_b, alpha=0.4)

        z        = encoder(X_b)
        loss_clf = criterion(letter_clf(z), yl_b)
        loss_con = cross_day_supcon_loss(z, yl_b, yd_b, temperature)
        loss     = lambda_clf * loss_clf + lambda_con * loss_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            encoder.eval(); letter_clf.eval()
            with torch.no_grad():
                accs = []
                for d in days:
                    crop = X_tr_tensors[d][:, GO_BIN:GO_BIN + T_BINS, :]
                    acc  = (letter_clf(encoder(crop)).argmax(1) ==
                            y_tr_tensors[d]).float().mean().item()
                    accs.append(acc)
                print(f"    Epoch {epoch+1:>4}  train acc: {np.mean(accs)*100:.1f}%")
            encoder.train(); letter_clf.train()

    encoder.eval()
    return encoder   # final epoch — no leakage

# ══════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
def direct_logreg(Z_tr, y_tr, Z_te, y_te):
    sc = StandardScaler()
    lr = LogisticRegression(C=1.0, max_iter=1000,
                            class_weight='balanced', random_state=42)
    lr.fit(sc.fit_transform(Z_tr), y_tr)
    train_acc = lr.score(sc.transform(Z_tr), y_tr) * 100
    preds     = lr.predict(sc.transform(Z_te))
    test_acc  = accuracy_score(y_te, preds) * 100
    return train_acc, test_acc, preds

def lasso_logreg(X_tr_s, y_tr, X_te_s, y_te, C_lasso=0.5):
    sc = StandardScaler()
    X_tr_scaled = sc.fit_transform(X_tr_s)
    X_te_scaled = sc.transform(X_te_s)
    scores        = np.zeros((len(y_te), 31))
    feature_counts = []
    for cls in np.unique(y_tr):
        lr1 = LogisticRegression(penalty='l1', solver='liblinear', C=C_lasso,
                                 class_weight='balanced', random_state=42,
                                 max_iter=1000)
        lr1.fit(X_tr_scaled, (y_tr == cls).astype(int))
        f = np.where(lr1.coef_[0] != 0)[0]
        if len(f) == 0:
            f = np.arange(min(10, X_tr_scaled.shape[1]))
        feature_counts.append(len(f))
        lr2 = LogisticRegression(solver='lbfgs', C=1/80, max_iter=1000,
                                 class_weight='balanced', random_state=42)
        lr2.fit(X_tr_scaled[:, f], (y_tr == cls).astype(int))
        scores[:, cls] = lr2.predict_proba(X_te_scaled[:, f])[:, 1]
    preds    = np.argmax(scores, axis=1)
    acc      = np.mean(preds == y_te) * 100
    avg_feat = np.mean(feature_counts)
    return acc, preds, avg_feat

def quantize_weights(W, bits):
    if bits >= 32:
        return W.copy()
    q_max = (2 ** (bits - 1)) - 1
    scale = np.max(np.abs(W), axis=1, keepdims=True) / q_max
    scale[scale == 0] = 1e-9
    W_q = np.clip(np.round(W / scale), -q_max, q_max) * scale
    return W_q.astype(np.float32)


def lasso_logreg_with_quantization(X_tr_s, y_tr, X_te_s, y_te,
                                    C_lasso=0.5, bits_list=[32, 16, 8, 4]):
    sc = StandardScaler()
    X_tr_scaled = sc.fit_transform(X_tr_s)
    X_te_scaled = sc.transform(X_te_s)

    n_cls = 31
    W = np.zeros((n_cls, X_tr_scaled.shape[1]), dtype=np.float32)
    b = np.zeros(n_cls, dtype=np.float32)
    feat_counts = []

    for cls in range(n_cls):
        lr1 = LogisticRegression(
            penalty='l1', solver='liblinear', C=C_lasso,
            class_weight='balanced', random_state=42, max_iter=1000)
        lr1.fit(X_tr_scaled, (y_tr == cls).astype(int))
        f = np.where(lr1.coef_[0] != 0)[0]
        if len(f) == 0:
            f = np.arange(min(10, X_tr_scaled.shape[1]))
        feat_counts.append(len(f))

        lr2 = LogisticRegression(
            solver='lbfgs', C=1/80, max_iter=1000,
            class_weight='balanced', random_state=42)
        lr2.fit(X_tr_scaled[:, f], (y_tr == cls).astype(int))
        W[cls, f] = lr2.coef_[0]
        b[cls]    = lr2.intercept_[0]

    avg_feat = np.mean(feat_counts)

    results = {}
    for bits in bits_list:
        W_q = quantize_weights(W, bits)
        b_q = quantize_weights(b.reshape(1, -1), bits).flatten()

        logits = X_te_scaled @ W_q.T + b_q
        preds  = np.argmax(logits, axis=1)
        acc    = accuracy_score(y_te, preds) * 100

        total_active = int(np.sum(W != 0))
        n_classes    = W.shape[0]
        weight_bits  = total_active * bits
        index_bits   = total_active * 16
        bias_bits    = n_classes * bits
        mem_kb       = (weight_bits + index_bits + bias_bits) / 8 / 1024

        results[bits] = {'acc': acc, 'mem_kb': mem_kb}
        print(f"   [{bits:2d}b] Acc: {acc:.2f}% | Mem: {mem_kb:.2f} KB | "
              f"Avg feat/class: {avg_feat:.1f}")

    return results, avg_feat

# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    days_all = sorted(list(all_X.keys()))

    print(f"\n{'='*80}\n GENERATING 5-FOLD STRATIFIED SPLITS\n{'='*80}")
    cv_splits = {}
    for d in days_all:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits[d] = list(skf.split(all_X[d], all_y[d]))

    per_session_acc     = {d: [] for d in days_all}
    quant_results_folds = {}
    omp_pool_per_fold   = []
    teacher_acc_folds   = []
    omp_acc_folds       = []
    lasso_acc_folds     = []
    lasso_feat_folds    = []

    for f_idx in range(5):
        print(f"\n{'='*80}\n FOLD {f_idx + 1}/5\n{'='*80}")

        train_X, train_y, test_X, test_y = {}, {}, {}, {}
        for d in days_all:
            tr_idx, te_idx    = cv_splits[d][f_idx]
            train_X[d], train_y[d] = all_X[d][tr_idx], all_y[d][tr_idx]
            test_X[d],  test_y[d]  = all_X[d][te_idx], all_y[d][te_idx]

        # ── Train teacher ─────────────────────────────────────────────────────
        print(f"\n [Fold {f_idx+1}] Training teacher (800 epochs)...")
        teacher = train_contrastive(train_X, train_y, days_all, epochs=800)

        teacher_path = f"teacher_fold_{f_idx+1}.pth"
        torch.save(teacher.state_dict(), teacher_path)
        print(f" [SAVE] Teacher → {teacher_path}")

        # ── Extract latents ───────────────────────────────────────────────────
        teacher.eval()
        Z_train_list, X_train_list, y_train_list = [], [], []
        Z_test_list,  X_test_list,  y_test_list  = [], [], []

        with torch.no_grad():
            for d in days_all:
                Xt = torch.tensor(train_X[d], dtype=torch.float32).to(DEVICE)
                Xc = Xt[:, GO_BIN:GO_BIN + T_BINS, :]
                Z_train_list.append(F.normalize(teacher(Xc), dim=1).cpu().numpy())
                crop = train_X[d][:, GO_BIN:GO_BIN + T_BINS, :]
                X_train_list.append(crop.reshape(crop.shape[0], -1))
                y_train_list.append(train_y[d])

            for d in days_all:
                Xt = torch.tensor(test_X[d], dtype=torch.float32).to(DEVICE)
                Xc = Xt[:, GO_BIN:GO_BIN + T_BINS, :]
                Z_test_list.append(F.normalize(teacher(Xc), dim=1).cpu().numpy())
                crop = test_X[d][:, GO_BIN:GO_BIN + T_BINS, :]
                X_test_list.append(crop.reshape(crop.shape[0], -1))
                y_test_list.append(test_y[d])

        Z_teacher_train = np.concatenate(Z_train_list)
        X_raw_train     = np.concatenate(X_train_list)
        y_train_flat    = np.concatenate(y_train_list)

        Z_teacher_test  = np.concatenate(Z_test_list)
        X_raw_test      = np.concatenate(X_test_list)
        y_test_flat     = np.concatenate(y_test_list)

        # ── Teacher latent accuracy (day-wise) ────────────────────────────────
        direct_train_acc, direct_acc, teacher_preds = direct_logreg(
            Z_teacher_train, y_train_flat, Z_teacher_test, y_test_flat)

        teacher_acc_folds.append(direct_acc) 

        day_res_teacher = []
        idx_start = 0
        for d in days_all:
            n_d   = len(test_y[d])
            acc_d = accuracy_score(test_y[d],
                                   teacher_preds[idx_start:idx_start + n_d]) * 100
            day_res_teacher.append(f"D{d}: {acc_d:.1f}%")
            idx_start += n_d

        print(f"\n [*] TEACHER LATENT | Train: {direct_train_acc:.2f}% | "
              f"Overall: {direct_acc:.2f}% | " + " | ".join(day_res_teacher))

        # ── OMP prefilter ─────────────────────────────────────────────────────
        print(f"\n [STAGE 1] Running OMP Prefilter...")
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=96)
        omp.fit(X_raw_train, Z_teacher_train)
        omp_flat_indices = np.nonzero(np.any(omp.coef_ != 0, axis=0))[0]
        print(f" -> Pool: {len(omp_flat_indices)} features from 19,200")

        omp_save_path = f"omp_indices_fold_{f_idx+1}.npy"
        np.save(omp_save_path, omp_flat_indices)
        print(f" [SAVE] OMP indices → {omp_save_path}")

        X_tr_prefiltered = X_raw_train[:, omp_flat_indices]
        X_te_prefiltered = X_raw_test[:,  omp_flat_indices]

        omp_pool_per_fold.append(len(omp_flat_indices))    # after OMP prefilter block

        # ── OMP-full L2 OvR diagnostic ────────────────────────────────────────
        print(f"\n [DIAGNOSTIC] L2 OvR on all {len(omp_flat_indices)} OMP features...")
        sc_diag    = StandardScaler()
        X_tr_diag  = sc_diag.fit_transform(X_tr_prefiltered)
        X_te_diag  = sc_diag.transform(X_te_prefiltered)

        scores_ovr = np.zeros((len(y_test_flat), 31))
        for cls in np.unique(y_train_flat):
            lr_ovr = LogisticRegression(solver='lbfgs', C=1/80, max_iter=1000,
                                        class_weight='balanced', random_state=42)
            lr_ovr.fit(X_tr_diag, (y_train_flat == cls).astype(int))
            scores_ovr[:, cls] = lr_ovr.predict_proba(X_te_diag)[:, 1]
        preds_ovr  = np.argmax(scores_ovr, axis=1)
        acc_ovr    = accuracy_score(y_test_flat, preds_ovr) * 100

        omp_acc_folds.append(acc_ovr)

        day_res_ovr = []
        idx_start = 0
        for d in days_all:
            n_d   = len(test_y[d])
            acc_d = accuracy_score(test_y[d],
                                   preds_ovr[idx_start:idx_start + n_d]) * 100
            day_res_ovr.append(f"D{d}: {acc_d:.1f}%")
            idx_start += n_d
        print(f" [*] OMP-full OvR L2 | Overall: {acc_ovr:.2f}% | " +
              " | ".join(day_res_ovr))

        # ── Lasso sweep ───────────────────────────────────────────────────────
        print(f"\n {'*'*60}\n [GRID] Lasso OvR sweep\n {'*'*60}")
        for c_lasso in [0.5]:
            overall_acc, preds, avg_feat = lasso_logreg(
                X_tr_prefiltered, y_train_flat,
                X_te_prefiltered, y_test_flat,
                C_lasso=c_lasso)

            if c_lasso == 0.5:
                lasso_acc_folds.append(overall_acc)
                lasso_feat_folds.append(avg_feat)
                idx_start = 0
                for d in days_all:
                    n_d = len(test_y[d])
                    acc_d = accuracy_score(
                        test_y[d], preds[idx_start:idx_start + n_d]) * 100
                    per_session_acc[d].append(acc_d)
                    idx_start += n_d

            day_results = []
            idx_start = 0
            for d in days_all:
                n_d   = len(test_y[d])
                acc_d = accuracy_score(test_y[d],
                                       preds[idx_start:idx_start + n_d]) * 100
                day_results.append(f"D{d}: {acc_d:.1f}%")
                idx_start += n_d

            print(f" [*] Lasso C={c_lasso} | Feat/class: {avg_feat:.1f} | "
                  f"Overall: {overall_acc:.2f}% | " + " | ".join(day_results))

        # ── Quantization at C=0.5 ────────────────────────────────────────
        print(f"\n [QUANTIZATION] Lasso C=0.5 across bit widths...")
        quant_results, _ = lasso_logreg_with_quantization(
            X_tr_s=X_tr_prefiltered,
            y_tr=y_train_flat,
            X_te_s=X_te_prefiltered,
            y_te=y_test_flat,
            C_lasso=0.5,
            bits_list=[32, 16, 8, 4]
        )
        quant_results_folds[f_idx + 1] = quant_results

    import json

    n_channels  = list(all_X.values())[0].shape[2]   
    n_time_bins = T_BINS                              

    results_to_save = {
        "raw_features": n_channels * n_time_bins,

        "omp_pool_per_fold": omp_pool_per_fold,
        "teacher_acc_folds": teacher_acc_folds,
        "omp_acc_folds": omp_acc_folds,
        "lasso_acc_folds": lasso_acc_folds,
        "lasso_avg_feat_per_class_folds": lasso_feat_folds,

        "quant_results": {
            str(fold): {
                str(bits): {"acc": vals["acc"], "mem_kb": vals["mem_kb"]}
                for bits, vals in fold_data.items()
            }
            for fold, fold_data in quant_results_folds.items()
        },

        "per_session_acc": {
            str(f_idx + 1): {
                str(d): per_session_acc[d][f_idx]
                for d in per_session_acc
                if f_idx < len(per_session_acc[d])
            }
            for f_idx in range(5)
        },
    }

    with open("bci_results.json", "w") as fp:
        json.dump(results_to_save, fp, indent=2)


    expected_len = 5
    for name, lst in [("omp_pool_per_fold", omp_pool_per_fold),
                       ("teacher_acc_folds", teacher_acc_folds),
                       ("omp_acc_folds", omp_acc_folds),
                       ("lasso_acc_folds", lasso_acc_folds),
                       ("lasso_feat_folds", lasso_feat_folds)]:
        if len(lst) != expected_len:
            print(f"WARNING: '{name}' has {len(lst)} entries, expected {expected_len}")

    print("Saved bci_results.json — run plot_bci_results.py to generate figures.")