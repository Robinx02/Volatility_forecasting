import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tft_model import TFTModel

EPOCHS   = 50
BATCH    = 64
LR       = 0.01
PATIENCE = 10
CLIP     = 1.0

OBS_FEATURES = ["rv_lag1", "rv_5d", "rv_22d", "sq_return", "vix_level", "vix_chg", "brent_ret", "dxy_level", "dxy_ret", "india10y", "india10y_chg", "us10y", "us10y_chg", "rate_spread", "callmoney", "nifty_ret", "em_ret", "is_mpc_day", "is_fomc_day",]

KNOWN_FEATURES = ["day_of_week", "month", "week_of_year"]


def quantile_loss(pred, target, q):
    e = target - pred
    return torch.mean(torch.max(q * e, (q - 1) * e))


def combined_loss(p50, p10, p90, y):
    return (quantile_loss(p50, y, 0.5)
            + quantile_loss(p10, y, 0.1)
            + quantile_loss(p90, y, 0.9))


def make_sequences(X_obs, X_known, y, enc_len):
    Xo, Xk, ys = [], [], []
    for i in range(len(y) - enc_len):
        Xo.append(X_obs[i : i + enc_len])
        Xk.append(X_known[i : i + enc_len + 1])
        ys.append(y[i + enc_len])
    return (np.array(Xo,  dtype=np.float32),
            np.array(Xk,  dtype=np.float32),
            np.array(ys,  dtype=np.float32))


def train(Xo, Xk, ys, device, enc_len=63):
    n_obs   = Xo.shape[2]
    n_known = Xk.shape[2]

    model = TFTModel(
        n_obs=n_obs, n_known=n_known, n_static=0,
        d=64, n_heads=1, n_lstm_layers=1,
        dropout=0.3, encoder_len=enc_len,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    val_size = min(252, len(ys) // 5)
    Xo_tr, Xo_val = Xo[:-val_size], Xo[-val_size:]
    Xk_tr, Xk_val = Xk[:-val_size], Xk[-val_size:]
    y_tr,  y_val  = ys[:-val_size], ys[-val_size:]

    tr_ds  = TensorDataset(torch.from_numpy(Xo_tr), torch.from_numpy(Xk_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(Xo_val), torch.from_numpy(Xk_val), torch.from_numpy(y_val))
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    best_loss = float("inf")
    best_wts = None
    wait = 0

    #similarly adding the progressbar here also for training time
    pbar = tqdm(range(1, EPOCHS +1), desc = "training" , leave = False)
    for ep in pbar:
        model.train()
        train_losses = []
        for xo, xk, yb in tr_dl:
            xo, xk, yb = xo.to(device), xk.to(device), yb.to(device)
            opt.zero_grad()
            p50, p10, p90, _, _ = model(xo, xk)
            loss = combined_loss(p50, p10, p90, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        v = []
        with torch.no_grad():
            for xo, xk, yb in val_dl:
                xo, xk, yb = xo.to(device), xk.to(device), yb.to(device)
                p50, p10, p90, _, _ = model(xo, xk)
                v.append(combined_loss(p50, p10, p90, yb).item())
        val_loss = np.mean(v)
        train_loss = np.mean(train_losses)

        mem_used = torch.cuda.memory_allocated(device) / 1e6 if device.type =="cuda" else 0
        pbar.set_postfix({
            "ep" : ep,
            "tr_loss" : f"{train_loss:.6f}",
            "val_loss" : f"{val_loss:.6f}",
            "wait" : wait,
            "VRAM_MB" : f"{mem_used:.0f}",
        })


        if val_loss < best_loss:
            best_loss = val_loss
            best_wts  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                tqdm.write(f" early stop at epoch {ep} val_loss = {val_loss:.6f}")
                break

    model.load_state_dict(best_wts)
    model.eval()
    return model