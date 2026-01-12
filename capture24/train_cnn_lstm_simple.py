# cnn_lstm_simple_min.py
# Simple CNN-LSTM for CAPTURE-24 (Walmsley-4)
#
# Reused from official repo:
#   - deep_models.Resnet  (CNN backbone)
#   - augmentation.Augment (data augmentation methods)
#
# Own work / changes:
#   - build sequences within each participant (pid) so we do NOT mix time across people
#   - use CNN as feature extractor (remove final classifier) + add an LSTM head
#   - plain PyTorch training loop + simple plots + confusion matrix
#
# Goal:
#   Compare a temporal model (CNN-LSTM) to CNN-only baseline and  CNN+HMM from original repo
#   Labels: 4 classes (Walmsley 2020)

import os
import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix

from torchviz import make_dot # for plotting model architecture


import deep_models as models          # from repo
from augmentation import Augment      # from repo

# Settings (relative)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../capture24
DATADIR = os.path.join(SCRIPT_DIR, "prepared_data")
OUTDIR  = os.path.join(SCRIPT_DIR, "outputs", "cnn_lstm_simple")
OUTDIR  = os.path.abspath(OUTDIR)


# reproducibility (same seed as CNN-HMM baseline)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


SEQ_LEN = 10          # sequence length = how many consecutive windows are fed into the LSTM
BATCH   = 32
EPOCHS  = 10
LR      = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["sleep", "sedentary", "light", "moderate-vigorous"]

# Map string labels -> class IDs (0..3)
LAB = {
    "sleep": 0,
    "sedentary": 1,
    "light": 2,
    "moderate-vigorous": 3
}


# Dataset where we sequence windows per participant
class SeqDS(Dataset):
    """
    We have window data in flat form:
      X: (N, L, C)  where L=window length (e.g. 1000), C=channels (3)
      Y: (N,) string labels
      pid: (N,) participant IDs like "P001", "P002", ...

    We want to feed an LSTM, so we build sequences:
      return x_seq: (T, C, L) and y_seq: (T,)
    where T = SEQ_LEN.
    IMPORTANT: sequences are built within the same participant only!!!
    """
    def __init__(self, X, Y, pid, T, aug=None):
        self.X, self.Y, self.pid = X, Y, pid
        self.T = int(T)
        self.aug = aug

        # starts = list of start indices that can form a full sequence of length T
        self.starts = []
        for p in np.unique(pid):
            idx = np.where(pid == p)[0]     # all windows for this participant
            idx = np.sort(idx)              # keep temporal order within participant 
            n = (len(idx) // self.T) * self.T  # drop leftover so sequences are complete
            for i in range(0, n, self.T):
                self.starts.append(idx[i])  # start index of each chunk

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        """
        For the i-th sequence:
          - find which participant it belongs to
          - take the next T windows of that same participant
          - convert each window from (L,C) -> (C,L) for Conv1d
        """
        s = self.starts[i]          # global start index in this subset
        p = self.pid[s]             # participant ID at start

        idx = np.where(self.pid == p)[0]   # all indices of this participant in the subset
        idx = np.sort(idx)
        pos = np.searchsorted(idx, s)      # position of start index inside idx list
        chunk = idx[pos:pos + self.T]      # take the next T windows

        xs, ys = [], []
        for k in chunk:
            x = self.X[k]   # window: (L,C) e.g. (1000,3)

            # augmentation is applied only during training
            if self.aug is not None:
                x = self.aug(x)

            # Conv1d expects (C,L), so transpose
            xs.append(torch.tensor(x.T, dtype=torch.float32))   # (C,L)
            ys.append(LAB[str(self.Y[k])])                      # label -> int

        # stack windows into a sequence:
        # xs: list of T tensors (C,L) -> (T,C,L)
        # ys: list of T ints -> (T,)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


# Model (repo CNN + own LSTM head)
class CNNLSTM(nn.Module):
    """
    Forward input: x_seq (B,T,C,L)
    Output: logits (B,T,4)

    Steps:
      1) run CNN feature extractor on each window
      2) global average pool over time dimension -> feature vector per window
      3) feed sequence of feature vectors into LSTM
      4) linear layer -> per-timestep class logits
    """
    def __init__(self):
        super().__init__()

        # CNN backbone from repo (same hyperparams as baseline for comparability)
        cnn = models.Resnet(
            3,          # n_channels
            4,          # outsize
            [64],       # n_filters_list   
            [7],        # kernel_size_list 
            [3],        # n_resblocks_list 
            [3],        # resblock_kernel_size_list 
            [2],        # downfactor_list  
            [2],        # downorder_list   
            0.2,        # drop1
            0.2,        # drop2
            128,        # fc_size
            False,      # is_cnnlstm
        )



        # Use CNN as feature extractor: drop final classifier layer
        # Repo stores layers in cnn.resnet (Sequential). Last layer is the classifier.
        self.cnn = nn.Sequential(*list(cnn.resnet.children())[:-1])

        # LSTM takes feature_dim=128 per timestep
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # bidirectional => hidden dims doubled (128*2=256)
        self.fc = nn.Linear(256, 4)

    def forward(self, x):
        # x shape: (B,T,C,L)
        B, T, C, L = x.shape

        # merge batch and time to run CNN on all windows at once:
        x = x.view(B * T, C, L)       # (B*T,C,L)

        # CNN feature maps output shape depends on repo implementation,
        # but after global average pooling over last dimension we want (B*T,128)
        f = self.cnn(x).mean(-1)      # (B*T,128)

        # reshape back to sequences for LSTM:
        f = f.view(B, T, 128)         # (B,T,128)

        h, _ = self.lstm(f)           # (B,T,256)
        logits = self.fc(h)           # (B,T,4)
        return logits

# Main: here we have training + evaluation + plots
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load data (prepared_data)
    X   = np.load(os.path.join(DATADIR, "X.npy"), mmap_mode="r")                 # (N,L,C)
    Y   = np.load(os.path.join(DATADIR, "Y_Walmsley2020.npy"), allow_pickle=True) # (N,)
    pid = np.load(os.path.join(DATADIR, "pid.npy"), allow_pickle=True)            # (N,)

    # Derivation/test split (same as repo): P001..P100 derivation, rest test
    deriv = np.isin(pid, [f"P{i:03d}" for i in range(1, 101)])

    # Augmentation from repo (only train)
    aug = Augment(
        jitter_sigma=0.05, jitter_prob=0.5,
        shift_window=20,   shift_prob=0.3,
        twarp_sigma=0.2,   twarp_knots=4, twarp_prob=0.2,
        mwarp_sigma=0.2,   mwarp_knots=4, mwarp_prob=0.2
    )

    train_ds = SeqDS(X[deriv],  Y[deriv],  pid[deriv],  SEQ_LEN, aug=aug)
    test_ds  = SeqDS(X[~deriv], Y[~deriv], pid[~deriv], SEQ_LEN, aug=None)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = CNNLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    accs = []

    # quick debug of one batch shapes
    xb0, yb0 = next(iter(train_ld))
    print("DEBUG batch shapes:")
    print("xb:", xb0.shape, " (expected B,T,C,L)")
    print("yb:", yb0.shape, " (expected B,T)")
    print("device:", DEVICE)

    # save model architecture as png 
    model.eval()
    with torch.no_grad():
        xb_arch = xb0.to(DEVICE)
        out_arch = model(xb_arch)

    dot = make_dot(
        out_arch,
        params=dict(model.named_parameters())
    )
    dot.format = "png"
    dot.render(os.path.join(OUTDIR, "model_arch"))


    for ep in range(EPOCHS):
        # train 
        model.train()
        s = 0.0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            out = model(xb)                           # (B,T,4)
            loss = loss_fn(out.view(-1, 4), yb.view(-1))  # flatten over timesteps

            opt.zero_grad()
            loss.backward()
            opt.step()

            s += loss.item()

        losses.append(s / max(len(train_ld), 1))

        # quick test accuracy after each epoch 
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in test_ld:
                out = model(xb.to(DEVICE))                 # (B,T,4)
                pred = out.argmax(-1).cpu().numpy().reshape(-1)
                yt.append(yb.numpy().reshape(-1))
                yp.append(pred)

        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        accs.append((yt == yp).mean())

        print(f"epoch {ep+1}/{EPOCHS} loss={losses[-1]:.4f} test_acc={accs[-1]:.4f}")

    # save model weights
    torch.save(model.state_dict(), os.path.join(OUTDIR, "cnn_lstm.pt"))

    # final metrics + confusion matrix
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    mcc = matthews_corrcoef(yt, yp)
    kappa = cohen_kappa_score(yt, yp)

    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    np.save(os.path.join(OUTDIR, "cm_raw.npy"), cm)

    print("\nFINAL metrics (test):")
    print("f1_macro =", f1)
    print("mcc      =", mcc)
    print("kappa    =", kappa)

    # plot: train loss 
    plt.figure()
    plt.plot(losses)
    plt.title("train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "loss.png"), dpi=200)
    plt.close()

    # plot: test accuracy 
    plt.figure()
    plt.plot(accs)
    plt.title("test accuracy (quick)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "acc.png"), dpi=200)
    plt.close()

    # plot: confusion matrix normalized 
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

    # save metrics summary 
    with open(os.path.join(OUTDIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"seq_len={SEQ_LEN}\n")
        f.write(f"n_samples={len(yt)}\n")
        f.write(f"f1_macro={f1}\n")
        f.write(f"mcc={mcc}\n")
        f.write(f"kappa={kappa}\n")

    print("\nSaved to:", os.path.abspath(OUTDIR))

    # save metrics as json (useful for paper / comparison)
    with open(os.path.join(OUTDIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "cnn_lstm_simple",
                "seed": SEED,
                "seq_len": SEQ_LEN,
                "n_samples": int(len(yt)),
                "f1_macro": float(f1),
                "mcc": float(mcc),
                "kappa": float(kappa),
            },
            f,
            indent=2
        )



if __name__ == "__main__":
    main()
