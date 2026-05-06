"""
GRU (Gated Recurrent Unit) forecasting model.
Lighter than LSTM, often converges faster on financial time-series.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class GRUModel(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(1, hidden, layers, batch_first=True, dropout=dropout)
        self.fc  = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def _prepare(series, window):
    scaler = MinMaxScaler()
    data   = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y   = [], []
    for i in range(window, len(data)):
        X.append(data[i - window: i])
        y.append(data[i])
    return np.array(X), np.array(y), scaler


def train_gru(series, window=20, epochs=50, lr=0.001, hidden=64, layers=2):
    X, y, scaler = _prepare(series, window)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model     = GRUModel(hidden=hidden, layers=layers)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    loss_fn   = nn.MSELoss()

    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        losses.append(loss.item())

    return model, scaler, losses


def forecast_gru(model, series, scaler, steps=30, window=20):
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
