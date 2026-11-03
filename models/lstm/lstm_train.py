import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import LSTMVolatilityModel

HIDDEN   = 64
LAYERS   = 2
DROPOUT  = 0.2
EPOCHS   = 50
BATCH    = 32
LR       = 1e-3
PATIENCE = 10


def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train(X_seq, y_seq, device):
    n_features = X_seq.shape[2]
    model      = LSTMVolatilityModel(n_features, HIDDEN, LAYERS, DROPOUT).to(device)
    opt        = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn    = nn.MSELoss()

    val_size = min(252, len(X_seq) // 5)
    X_tr, X_val = X_seq[:-val_size], X_seq[-val_size:]
    y_tr, y_val = y_seq[:-val_size], y_seq[-val_size:]

    tr_dl  = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                        batch_size=BATCH, shuffle=False)

    best_loss = float("inf")
    best_wts  = None
    wait      = 0

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        v = []
        with torch.no_grad():
            for xb, yb in val_dl:
                v.append(loss_fn(model(xb.to(device)), yb.to(device)).item())
        val_loss = np.mean(v)

        if val_loss < best_loss:
            best_loss = val_loss
            best_wts  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_wts)
    model.eval()
    return model