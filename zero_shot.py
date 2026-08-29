
import os
import numpy as np
import scipy.io as sio
from pathlib import Path
import datetime
import warnings
import copy
from scipy.ndimage import gaussian_filter1d
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
print(f"Device: {DEVICE}  |  Mamba: {HAS_MAMBA}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GO_BIN, T_BINS, SIGMA = 56, 100, 16
TEST_DAYS    = [1,10]
TRAIN_DAYS  = list(range(2, 10))   # Days 2-9 full
SWEEP       = [0, 1, 2, 4, 8]
N_RUNS      = 5                    # runs for n > 0
EPOCHS      = 800
N_PER_CLASS = 8                    # trials per class per day in contrastive batch
NUM_CLASSES = 31
OMP_COEFS   = 96
C_LASSO     = 0.5

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
data_folder = Path('/home/Datasets')
folders     = [f for f in data_folder.iterdir()
               if f.is_dir() and (f / "singleLetters.mat").exists()]

def parse_date(name):
    p = name.split(".")
    return datetime.date(int(p[1]), int(p[2]), int(p[3]))

dataset_paths = {f"day{i+1}_data": f / "singleLetters.mat"
                 for i, f in enumerate(sorted(folders, key=lambda f: parse_date(f.name)))}

def load_day(path):
    mat  = sio.loadmat(path, squeeze_me=True)
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
    # return FULL smoothed sequence (no cropping yet)
    return np.concatenate(X), np.concatenate(y)

print("Loading data (Days 2–10)...")
all_X, all_y = {}, {}
for k, v in dataset_paths.items():
    d = int(k.replace('day', '').replace('_data', ''))
    if 1 <= d <= 10:
        all_X[d], all_y[d] = load_day(v)
        print(f"  Day {d}: {all_X[d].shape}")

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
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
    start = base_start + shift
    end   = start + crop_len
    return x[:, start:end, :]

def temporal_cutout(x, cutout_len=15):
    if x.requires_grad:
        B, T, C = x.shape
        start = np.random.randint(0, T - cutout_len)
        mask  = torch.ones((B, T, C), device=x.device)
        mask[:, start:start + cutout_len, :] = 0.0
        return x * mask
    return x

def perform_mixup(x, y, d_tensor, alpha=0.4):
    lam         = np.random.beta(alpha, alpha)
    combined    = y * 1000 + d_tensor
    sort_idx    = torch.argsort(combined)
    xs, ys, ds  = x[sort_idx], y[sort_idx], d_tensor[sort_idx]
    xm          = lam * xs + (1 - lam) * torch.roll(xs, 1, 0)
    same_cls    = ys == torch.roll(ys, 1, 0)
    same_day    = ds == torch.roll(ds, 1, 0)
    valid       = (same_cls & same_day).view(-1, 1, 1).float()
    x_out       = xm * valid + xs * (1 - valid)
    shuf        = torch.randperm(x.size(0), device=x.device)
    return x_out[shuf], ys[shuf], ds[shuf]

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
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
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                 nn.Tanh(),
                                 nn.Linear(d_model // 2, 1))
    def forward(self, x):
        return torch.sum(x * F.softmax(self.mlp(x), dim=1), dim=1)

class SimpleReadIn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.LayerNorm(out_ch)
    def forward(self, x):
        return self.norm(self.conv(x).permute(0, 2, 1)).permute(0, 2, 1)

class Encoder(nn.Module):
    def __init__(self, n_ch=192, embed_dim=128, latent_dim=96,
                 dropout=0.4, noise_std=0.08):
        super().__init__()
        self.noise        = GaussianNoise(std=noise_std)
        self.spatial_drop = SpatialDropout1D(p=dropout)
        half              = embed_dim // 2
        self.proj_in_1    = SimpleReadIn(n_ch // 2, half)
        self.proj_in_2    = SimpleReadIn(n_ch // 2, half)
        self.array_mixer  = nn.Linear(embed_dim, embed_dim)
        self.mamba1       = MambaBlock(embed_dim)
        self.mamba2       = MambaBlock(embed_dim)
        self.norm1        = nn.LayerNorm(embed_dim)
        self.norm2        = nn.LayerNorm(embed_dim)
        self.enc_proj     = nn.Conv1d(embed_dim, latent_dim, kernel_size=1)
        self.bn           = nn.BatchNorm1d(latent_dim)
        self.attn_pool    = TemporalAttentionPooling(latent_dim)

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
# CONTRASTIVE LOSS
# ══════════════════════════════════════════════════════════════════════════════
def cross_day_supcon_loss(z, y_letter, y_day, temperature=0.05):
    z           = F.normalize(z, dim=1)
    sim         = torch.matmul(z, z.T) / temperature
    same_letter = y_letter.unsqueeze(0) == y_letter.unsqueeze(1)
    same_day    = y_day.unsqueeze(0)    == y_day.unsqueeze(1)
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
# BATCH BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_batch(X_t, y_t, d_t, days, n_per=N_PER_CLASS):
    Xl, yl, dl = [], [], []
    for d in days:
        for c in range(NUM_CLASSES):
            idx = (y_t[d] == c).nonzero(as_tuple=True)[0]
            if len(idx) == 0: continue
            chosen = idx[torch.randperm(len(idx))[:min(len(idx), n_per)]]
            Xl.append(X_t[d][chosen]); yl.append(y_t[d][chosen]); dl.append(d_t[d][chosen])
    return torch.cat(Xl), torch.cat(yl), torch.cat(dl)

def prep(X): return torch.tensor(X, dtype=torch.float32).to(DEVICE)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_teacher(train_X, train_y, active_days, epochs=EPOCHS):

    encoder    = Encoder().to(DEVICE)
    clf        = LinearClassifier().to(DEVICE)
    optimizer  = optim.AdamW(list(encoder.parameters()) + list(clf.parameters()),
                             lr=1e-3, weight_decay=1e-3)
    scheduler  = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3,
                                               steps_per_epoch=1, epochs=epochs)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_t = {d: prep(train_X[d]) for d in active_days}
    y_t = {d: torch.tensor(train_y[d], dtype=torch.long).to(DEVICE) for d in active_days}
    d_t = {d: torch.full((len(train_X[d]),), d, dtype=torch.long).to(DEVICE)
           for d in active_days}

    for epoch in range(epochs):
        encoder.train(); clf.train(); optimizer.zero_grad()
        Xb, ylb, ydb = build_batch(X_t, y_t, d_t, active_days)
        Xb.requires_grad_(True)
        Xb = temporal_crop_jitter(Xb, max_shift=8)
        if epoch > 100:
            Xb = temporal_cutout(Xb)
            Xb, ylb, ydb = perform_mixup(Xb, ylb, ydb)
        z    = encoder(Xb)
        loss = criterion(clf(z), ylb) + 2.0 * cross_day_supcon_loss(z, ylb, ydb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        if (epoch + 1) % 200 == 0:
            encoder.eval(); clf.eval()
            with torch.no_grad():
                accs = []
                for d in active_days:
                    X_eval = X_t[d][:, GO_BIN:GO_BIN + T_BINS, :]
                    z_d    = encoder(X_eval)
                    accs.append((clf(z_d).argmax(1) == y_t[d]).float().mean().item())
            print(f"    Epoch {epoch+1}/{epochs}  train acc: {np.mean(accs)*100:.1f}%")
            encoder.train(); clf.train()

    encoder.eval()
    return encoder

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
def teacher_latent_acc(Z_tr, y_tr, Z_te, y_te):
    sc  = StandardScaler()
    lr  = LogisticRegression(C=1.0, max_iter=1000,
                             class_weight='balanced', random_state=42)
    lr.fit(sc.fit_transform(Z_tr), y_tr)
    return lr.score(sc.transform(Z_te), y_te) * 100

def omp_baseline_acc(X_tr, y_tr, X_te, y_te):
    """L2 OvR on ALL OMP-prefiltered features (no Lasso)."""
    sc       = StandardScaler()
    Xtr_s    = sc.fit_transform(X_tr)
    Xte_s    = sc.transform(X_te)
    scores   = np.zeros((len(y_te), NUM_CLASSES))
    for cls in np.unique(y_tr):
        lr = LogisticRegression(solver='lbfgs', C=1/80, max_iter=1000,
                                class_weight='balanced', random_state=42)
        lr.fit(Xtr_s, (y_tr == cls).astype(int))
        scores[:, cls] = lr.predict_proba(Xte_s)[:, 1]
    return accuracy_score(y_te, np.argmax(scores, axis=1)) * 100

def lasso_l2_ovr(X_tr, y_tr, X_te, y_te, C_lasso=C_LASSO):
    sc       = StandardScaler()
    Xtr_s    = sc.fit_transform(X_tr)
    Xte_s    = sc.transform(X_te)
    scores   = np.zeros((len(y_te), NUM_CLASSES))
    feat_cnt = []
    for cls in np.unique(y_tr):
        lr1 = LogisticRegression(penalty='l1', solver='liblinear', C=C_lasso,
                                 class_weight='balanced', random_state=42, max_iter=1000)
        lr1.fit(Xtr_s, (y_tr == cls).astype(int))
        f   = np.where(lr1.coef_[0] != 0)[0]
        if len(f) == 0: f = np.arange(min(10, Xtr_s.shape[1]))
        feat_cnt.append(len(f))
        lr2 = LogisticRegression(solver='lbfgs', C=1/80, max_iter=1000,
                                 class_weight='balanced', random_state=42)
        lr2.fit(Xtr_s[:, f], (y_tr == cls).astype(int))
        scores[:, cls] = lr2.predict_proba(Xte_s[:, f])[:, 1]
    acc = accuracy_score(y_te, np.argmax(scores, axis=1)) * 100
    return acc, float(np.mean(feat_cnt))

# ══════════════════════════════════════════════════════════════════════════════
# ONE RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_one(test_day, n_trials, run_idx, rng):

    tag = f"day{test_day}_n{n_trials}_run{run_idx}"
    enc_path  = f"encoder_{tag}.pth"
    feat_path = f"omp_features_{tag}.npy"

    train_X, train_y = {d: all_X[d] for d in TRAIN_DAYS}, \
                       {d: all_y[d] for d in TRAIN_DAYS}
    active_days = TRAIN_DAYS.copy()

    day_X, day_y = all_X[test_day], all_y[test_day]     
    tr_idx, te_idx = [], []
    for cls in np.unique(day_y):
        cls_idx = np.where(day_y == cls)[0]
        rng.shuffle(cls_idx)
        actual_n = min(n_trials, len(cls_idx))
        tr_idx.extend(cls_idx[:actual_n])
        te_idx.extend(cls_idx[actual_n:])

    if n_trials > 0:
        train_X[test_day] = day_X[tr_idx]                
        train_y[test_day] = day_y[tr_idx]                 
        active_days.append(test_day)                      

    X_te  = day_X[te_idx]
    y_te  = day_y[te_idx]

    n_tr_total = sum(len(train_X[d]) for d in active_days)
    print(f"\n  [{tag}]  active_days={active_days}  "
          f"train_trials={n_tr_total}  test_trials={len(y_te)}")


    encoder = Encoder().to(DEVICE)
    if os.path.exists(enc_path):
        print(f"    Loading encoder from {enc_path}")
        encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    else:
        print(f"    Training encoder ({EPOCHS} epochs)...")
        encoder = train_teacher(train_X, train_y, active_days)
        torch.save(encoder.state_dict(), enc_path)
        print(f"    Saved → {enc_path}")


    encoder.eval()
    Z_tr_list, X_tr_list, y_tr_list = [], [], []
    with torch.no_grad():
        for d in active_days:
            xt = prep(train_X[d])
            xt_crop = xt[:, GO_BIN:GO_BIN + T_BINS, :]
            Z_tr_list.append(F.normalize(encoder(xt_crop), dim=1).cpu().numpy())
            X_tr_crop = train_X[d][:, GO_BIN:GO_BIN + T_BINS, :]
            X_tr_list.append(X_tr_crop.reshape(len(train_X[d]), -1))
            y_tr_list.append(train_y[d])
        xt_te = prep(X_te)
        xt_te_crop = xt_te[:, GO_BIN:GO_BIN + T_BINS, :]
        Z_te  = F.normalize(encoder(xt_te_crop), dim=1).cpu().numpy()

    Z_tr = np.concatenate(Z_tr_list)
    X_tr = np.concatenate(X_tr_list)
    y_tr = np.concatenate(y_tr_list)
    
    X_te_crop = X_te[:, GO_BIN:GO_BIN + T_BINS, :]
    X_te_flat = X_te_crop.reshape(len(y_te), -1)


    enc_acc = teacher_latent_acc(Z_tr, y_tr, Z_te, y_te)
    print(f"    Teacher latent acc       : {enc_acc:.2f}%")


    if os.path.exists(feat_path):
        omp_idx = np.load(feat_path)
        print(f"    Loaded OMP features from {feat_path}  ({len(omp_idx)} features)")
    else:
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=OMP_COEFS)
        omp.fit(X_tr, Z_tr)
        omp_idx = np.nonzero(np.any(np.abs(omp.coef_) > 1e-10, axis=0))[0]
        np.save(feat_path, omp_idx)
        print(f"    OMP prefiltered 19200 → {len(omp_idx)} features  Saved → {feat_path}")

    X_tr_omp = X_tr[:, omp_idx]
    X_te_omp = X_te_flat[:, omp_idx]


    omp_acc = omp_baseline_acc(X_tr_omp, y_tr, X_te_omp, y_te)
    print(f"    OMP-full L2 OvR acc      : {omp_acc:.2f}%  ({len(omp_idx)} features)")


    lasso_acc, avg_feat = lasso_l2_ovr(X_tr_omp, y_tr, X_te_omp, y_te, C_LASSO)
    print(f"    Lasso (C={C_LASSO}) OvR acc    : {lasso_acc:.2f}%  "
          f"(avg {avg_feat:.1f} features/class)")

    return enc_acc, omp_acc, lasso_acc, len(omp_idx), avg_feat

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_summary = {}   

    for test_day in TEST_DAYS:
        print(f"\n{'═'*70}")
        print(f"  GENERALISATION TO DAY {test_day}")
        print(f"  Base training: Days {TRAIN_DAYS}")
        print(f"  Sweep: {SWEEP} trials/class from Day {test_day}")
        print(f"{'═'*70}")

        summary = {}

        for n_trials in SWEEP:
            n_runs = 1 if n_trials == 0 else N_RUNS
            print(f"\n{'─'*70}")
            print(f"  Day {test_day} | n_trials = {n_trials}  "
                  f"({n_runs} run{'s' if n_runs > 1 else ''})")
            print(f"{'─'*70}")

            enc_accs, omp_accs, lasso_accs = [], [], []
            n_omp_feats, avg_lasso_feats   = [], []

            for run in range(n_runs):
                rng = np.random.RandomState(run)
                enc_a, omp_a, las_a, n_omp, avg_f = run_one(test_day, n_trials, run, rng)
                enc_accs.append(enc_a);  omp_accs.append(omp_a)
                lasso_accs.append(las_a); n_omp_feats.append(n_omp)
                avg_lasso_feats.append(avg_f)

            summary[n_trials] = dict(
                enc_mean=np.mean(enc_accs),   enc_std=np.std(enc_accs),
                omp_mean=np.mean(omp_accs),   omp_std=np.std(omp_accs),
                las_mean=np.mean(lasso_accs), las_std=np.std(lasso_accs),
                omp_feats=np.mean(n_omp_feats),
                lasso_feats=np.mean(avg_lasso_feats),
            )

        all_summary[test_day] = summary

        # ── Per-day summary table ────────────────────────────────────────────
        print(f"\n\n{'═'*70}")
        print(f"  SUMMARY — Day {test_day} Generalisation")
        print(f"{'═'*70}")
        print(f"  {'N':>4}  {'Teacher':>14}  {'OMP-full':>14}  "
              f"{'Lasso C=0.5':>14}  {'OMP feats':>10}  {'Lasso feats':>11}")
        print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*11}")
        for n, s in summary.items():
            print(f"  {n:>4}  "
                  f"{s['enc_mean']:>6.2f}±{s['enc_std']:>5.2f}  "
                  f"{s['omp_mean']:>6.2f}±{s['omp_std']:>5.2f}  "
                  f"{s['las_mean']:>6.2f}±{s['las_std']:>5.2f}  "
                  f"{s['omp_feats']:>10.0f}  "
                  f"{s['lasso_feats']:>11.1f}")

