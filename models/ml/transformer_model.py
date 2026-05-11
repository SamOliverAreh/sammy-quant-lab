"""Transformer forecaster — multivariate + in-sample output (v4.2)."""
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerForecaster(nn.Module):
    def __init__(self, n_features=1, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj    = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc          = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                           dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.proj(x); x = self.pos_enc(x); x = self.encoder(x)
        return self.head(x[:, -1, :])


def _prepare(series, window, exog=None):
    y_vals   = series.values.reshape(-1, 1)
    sc_y     = StandardScaler()
    y_scaled = sc_y.fit_transform(y_vals)
    if exog is not None and len(exog.columns) > 0:
        sc_X     = StandardScaler()
        X_scaled = sc_X.fit_transform(exog.values)
        data     = np.hstack([y_scaled, X_scaled])
    else:
        data, sc_X = y_scaled, None
    n_feat = data.shape[1]
    X, y   = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i]); y.append(y_scaled[i])
    return np.array(X), np.array(y), sc_y, sc_X, n_feat


def train_transformer(series, window=20, epochs=50, lr=5e-4,
                      d_model=32, nhead=4, num_layers=2, exog=None):
    X, y, sc_y, sc_X, n_feat = _prepare(series, window, exog)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model   = TransformerForecaster(n_features=n_feat, d_model=d_model,
                                    nhead=nhead, num_layers=num_layers)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses  = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        p = model(X_t); l = loss_fn(p, y_t)
        l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); losses.append(l.item())
    model.eval()
    with torch.no_grad():
        insample = sc_y.inverse_transform(model(X_t).numpy()).flatten()
    return model, sc_y, sc_X, losses, insample, series.values[window:]


def forecast_transformer(model, series, sc_y, steps=30, window=20,
                         scaler_X=None, exog=None):
    model.eval()
    y_sc = sc_y.transform(series.values.reshape(-1,1))
    data = np.hstack([y_sc, scaler_X.transform(exog.values)]) \
           if (scaler_X is not None and exog is not None and len(exog.columns)>0) else y_sc
    seq   = data[-window:].copy(); preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq.reshape(1,window,-1), dtype=torch.float32)
            p = model(x).item(); preds.append(p)
            nr = seq[-1:].copy(); nr[0,0] = p
            seq = np.vstack([seq[1:], nr])
    return sc_y.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
