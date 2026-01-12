# train_cnn_lstm_tuned.py
# Tuned CNN-LSTM for CAPTURE-24 (Walmsley-4)
#
# Reused from official repo:
#   - deep_models.Resnet  (CNN backbone)
#   - augmentation.Augment (data augmentation)
#
# Own work / changes:
#   - build sequences within each participant (pid)
#   - CNN as feature extractor + LSTM head
#   - plain PyTorch training loop + plots + confusion matrix + metrics.json
#
# Tuning:
#   - longer sequences (SEQ_LEN)
#   - more epochs
#   - slightly smaller LR + weight_decay
#   - dropout in LSTM head
#   - gradient clipping (stability)
#   - best epoch selection based on validation macro-F1
#   - validation split is created from derivation participants (P001–P100)


import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix

import deep_models as models          # from repo
from augmentation import Augment      # from repo



# Settings (paths + seed)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../capture24
DATADIR = os.path.join(SCRIPT_DIR, "prepared_data")
OUTDIR  = os.path.join(SCRIPT_DIR, "outputs", "cnn_lstm_tuned")
OUTDIR  = os.path.abspath(OUTDIR)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# allow cpu fallback (still uses cuda if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["sleep", "sedentary", "light", "moderate-vigorous"]
LAB = {"sleep": 0, "sedentary": 1, "light": 2, "moderate-vigorous": 3}



# Tuned hyperparameters

SEQ_LEN = 20            # bigger context helps LSTM smooth transitions
BATCH   = 64            # if GPU fits; if OOM -> 32
EPOCHS  = 25            # more training
LR      = 5e-4          # usually more stable than 1e-3 here
WEIGHT_DECAY = 1e-4     # mild regularization
GRAD_CLIP = 1.0         # stability for RNNs

LSTM_HIDDEN = 256       # stronger temporal head
LSTM_DROPOUT = 0.2      # regularize head

# fix: simple val split inside derivation set (avoid tuning on test)
# we keep deriv/test definition identical to repo, but carve out VAL_PIDS from deriv.
VAL_PIDS = 15           # number of participants from P001..P100 used for validation

# fix: optional class weights (helps minority class like moderate-vigorous)
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = [1.0, 1.0, 1.2, 1.6]   # sleep, sedentary, light, moderate-vigorous

# fix: simple LR scheduler 
USE_SCHEDULER = True
SCHED_STEP = 8
SCHED_GAMMA = 0.5

# augmentation: a bit milder than aggressive 
AUG = dict(
    jitter_sigma=0.05, jitter_prob=0.5,
    shift_window=20,   shift_prob=0.3,
    twarp_sigma=0.2,   twarp_knots=4, twarp_prob=0.2,
    mwarp_sigma=0.2,   mwarp_knots=4, mwarp_prob=0.2
)



# Dataset 
class SeqDS(Dataset):
    """
    Input windows:
      X: (N, L, C)
      Y: (N,) string labels
      pid: (N,) participant ids

    Output per item:
      x_seq: (T, C, L)
      y_seq: (T,)
    """
    def __init__(self, X, Y, pid, T, aug=None):
        self.X, self.Y, self.pid = X, Y, pid
        self.T = int(T)
        self.aug = aug

        self.starts = []
        for p in np.unique(pid):
            idx = np.where(pid == p)[0]
            idx = np.sort(idx)
            n = (len(idx) // self.T) * self.T
            for i in range(0, n, self.T):
                self.starts.append(idx[i])

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        p = self.pid[s]

        idx = np.where(self.pid == p)[0]
        idx = np.sort(idx)
        pos = np.searchsorted(idx, s)
        chunk = idx[pos:pos + self.T]

        xs, ys = [], []
        for k in chunk:
            x = self.X[k]  # (L,C)
            if self.aug is not None:
                x = self.aug(x)
            xs.append(torch.tensor(x.T, dtype=torch.float32))  # (C,L)
            ys.append(LAB[str(self.Y[k])])

        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)



# Model (repo CNN + tuned LSTM head)
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # IMPORTANT: use same repo backbone style for comparability
        cnn = models.Resnet(
            3,          # n_channels
            4,          # outsize
            [64],       # n_filters_list
            [7],        # kernel_size_list
            [3],        # n_resblocks_list
            [3],        # resblock_kernel_size_list
            [2],        # downfactor_list
            [2],        # downorder_list (must satisfy their downsample assert)
            0.2,        # drop1
            0.2,        # drop2
            128,        # fc_size (feature dim = 128)
            False,      # is_cnnlstm (we add our own)
        )

        # CNN feature extractor: drop final classifier layer
        self.cnn = nn.Sequential(*list(cnn.resnet.children())[:-1])

        # LSTM head
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=LSTM_HIDDEN,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if LSTM_DROPOUT > 0 else 0.0,
            num_layers=2,   # small boost vs 1 layer
        )

        self.fc = nn.Linear(2 * LSTM_HIDDEN, 4)

    def forward(self, x):
        # x: (B,T,C,L)
        B, T, C, L = x.shape
        x = x.view(B * T, C, L)           # (B*T,C,L)
        f = self.cnn(x).mean(-1)          # (B*T,128)
        f = f.view(B, T, 128)             # (B,T,128)
        h, _ = self.lstm(f)               # (B,T,2*hidden)
        return self.fc(h)                 # (B,T,4)



# Helpers
def eval_model(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE))
            pred = out.argmax(-1).cpu().numpy().reshape(-1)
            yt.append(yb.numpy().reshape(-1))
            yp.append(pred)
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)

    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    mcc = matthews_corrcoef(yt, yp)
    kappa = cohen_kappa_score(yt, yp)
    acc = (yt == yp).mean()
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    return yt, yp, acc, f1, mcc, kappa, cm



# Main
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    X   = np.load(os.path.join(DATADIR, "X.npy"), mmap_mode="r")
    Y   = np.load(os.path.join(DATADIR, "Y_Walmsley2020.npy"), allow_pickle=True)
    pid = np.load(os.path.join(DATADIR, "pid.npy"), allow_pickle=True)

    # deriv/test split: P001..P100 derivation, rest test
    deriv = np.isin(pid, [f"P{i:03d}" for i in range(1, 101)])

    # fix: create train/val split from deriv participants (no test leakage)
    pid_deriv = pid[deriv]
    unique_pids = np.unique(pid_deriv)

    rng = np.random.default_rng(SEED)
    val_pids = rng.choice(unique_pids, size=min(VAL_PIDS, len(unique_pids)), replace=False)

    deriv_val = deriv & np.isin(pid, val_pids)
    deriv_trn = deriv & (~np.isin(pid, val_pids))

    aug = Augment(**AUG)

    train_ds = SeqDS(X[deriv_trn], Y[deriv_trn], pid[deriv_trn], SEQ_LEN, aug=aug)
    val_ds   = SeqDS(X[deriv_val], Y[deriv_val], pid[deriv_val], SEQ_LEN, aug=None)

    test_ds  = SeqDS(X[~deriv], Y[~deriv], pid[~deriv], SEQ_LEN, aug=None)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = CNNLSTM().to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # optional class weights for CE loss
    if USE_CLASS_WEIGHTS:
        w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # optional scheduler
    if USE_SCHEDULER:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)
    else:
        sched = None

    # debug shapes once
    xb0, yb0 = next(iter(train_ld))
    print("DEBUG batch:", xb0.shape, yb0.shape, "| device:", DEVICE)
    print("VAL pids:", len(val_pids), "| train sequences:", len(train_ds), "| val sequences:", len(val_ds), "| test sequences:", len(test_ds))

    losses = []
    val_f1s = []
    test_accs = []
    test_f1s = []

    best_f1 = -1.0
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        s = 0.0

        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            out = model(xb)                               # (B,T,4)
            loss = loss_fn(out.view(-1, 4), yb.view(-1))   # flatten

            opt.zero_grad()
            loss.backward()

            # RNN stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()
            s += loss.item()

        train_loss = s / max(len(train_ld), 1)
        losses.append(train_loss)

        # fix: evaluate on val for best epoch selection (no test leakage)
        _, _, _, val_f1, _, _, _ = eval_model(model, val_ld)
        val_f1s.append(val_f1)

        # (optional) still track test each epoch for plotting, but NOT used for selecting best epoch
        yt_t, yp_t, acc_t, f1_t, mcc_t, kappa_t, cm_t = eval_model(model, test_ld)
        test_accs.append(acc_t)
        test_f1s.append(f1_t)

        print(f"epoch {ep+1}/{EPOCHS} loss={train_loss:.4f} val_f1={val_f1:.4f} test_acc={acc_t:.4f} test_f1={f1_t:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if sched is not None:
            sched.step()

    # restore best epoch weights (selected by val_f1)
    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval (best epoch by val)
    yt, yp, acc, f1, mcc, kappa, cm = eval_model(model, test_ld)

    print("\nFINAL (best epoch by val_f1, reported on test):")
    print("acc      =", acc)
    print("f1_macro =", f1)
    print("mcc      =", mcc)
    print("kappa    =", kappa)

    # save model + matrices
    torch.save(model.state_dict(), os.path.join(OUTDIR, "cnn_lstm_tuned.pt"))
    np.save(os.path.join(OUTDIR, "cm_raw.npy"), cm)

    # plots
    plt.figure()
    plt.plot(losses)
    plt.title("train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(val_f1s)
    plt.title("val macro-F1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "val_f1.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(test_accs)
    plt.title("test accuracy")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "acc.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(test_f1s)
    plt.title("test macro-F1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "f1.png"), dpi=200)
    plt.close()

    # normalized confusion matrix plot
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(6, 5))
    plt.imshow(cmn)
    plt.title("confusion matrix (normalized)")
    plt.xticks(range(4), CLASSES, rotation=25, ha="right")
    plt.yticks(range(4), CLASSES)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cm_norm.png"), dpi=200)
    plt.close()

    # metrics.json for paper
    with open(os.path.join(OUTDIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "cnn_lstm_tuned",
                "seed": SEED,
                "seq_len": SEQ_LEN,
                "batch": BATCH,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "lstm_hidden": LSTM_HIDDEN,
                "lstm_dropout": LSTM_DROPOUT,
                "val_pids": int(len(val_pids)),
                "use_class_weights": bool(USE_CLASS_WEIGHTS),
                "class_weights": CLASS_WEIGHTS if USE_CLASS_WEIGHTS else None,
                "use_scheduler": bool(USE_SCHEDULER),
                "scheduler": {"type": "StepLR", "step_size": SCHED_STEP, "gamma": SCHED_GAMMA} if USE_SCHEDULER else None,
                "n_samples": int(len(yt)),
                "acc": float(acc),
                "f1_macro": float(f1),
                "mcc": float(mcc),
                "kappa": float(kappa),
            },
            f,
            indent=2
        )

    print("\nSaved to:", OUTDIR)


if __name__ == "__main__":
    main()
