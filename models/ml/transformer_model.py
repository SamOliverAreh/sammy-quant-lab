"""
Transformer-based time-series forecasting model (encoder-only).
Uses positional encoding + multi-head self-attention.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.1, window=20):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq, 1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])   # predict from last token


def _prepare(series, window):
    scaler = MinMaxScaler()
    data   = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y   = [], []
    for i in range(window, len(data)):
        X.append(data[i - window: i])
        y.append(data[i])
    return np.array(X), np.array(y), scaler


def train_transformer(series, window=20, epochs=50, lr=5e-4, d_model=32, nhead=4, num_layers=2):
    X, y, scaler = _prepare(series, window)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model   = TransformerForecaster(d_model=d_model, nhead=nhead, num_layers=num_layers, window=window)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    return model, scaler, losses


def forecast_transformer(model, series, scaler, steps=30, window=20):
    model.eval()
    data  = scaler.transform(series.values.reshape(-1, 1))
    seq   = data[-window:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq.reshape(1, window, 1), dtype=torch.float32)
            p = model(x).item()
            preds.append(p)
            seq = np.vstack([seq[1:], [[p]]])
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
