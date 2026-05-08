"""GRU model — univariate and multivariate."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class GRUModel(nn.Module):
    def __init__(self, n_features=1, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.fc  = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def _prepare(series, window, exog=None):
    y_vals   = series.values.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_vals)
    if exog is not None and len(exog.columns) > 0:
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(exog.values)
        data     = np.hstack([y_scaled, X_scaled])
    else:
        data, scaler_X = y_scaled, None
    n_feat = data.shape[1]
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window: i])
        y.append(y_scaled[i])
    return np.array(X), np.array(y), scaler_y, scaler_X, n_feat


def train_gru(series, window=20, epochs=50, lr=0.001, hidden=64, layers=2, exog=None):
    X, y, scaler_y, scaler_X, n_feat = _prepare(series, window, exog)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model   = GRUModel(n_features=n_feat, hidden=hidden, layers=layers)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()
    losses  = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        p = model(X_t)
        l = loss_fn(p, y_t)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(l.item())
    model.eval()
    with torch.no_grad():
        insample = scaler_y.inverse_transform(model(X_t).numpy()).flatten()
    return model, scaler_y, scaler_X, losses, insample, series.values[window:]


def forecast_gru(model, series, scaler_y, steps=30, window=20, scaler_X=None, exog=None):
    model.eval()
    y_scaled = scaler_y.transform(series.values.reshape(-1, 1))
    if scaler_X is not None and exog is not None and len(exog.columns) > 0:
        data = np.hstack([y_scaled, scaler_X.transform(exog.values)])
    else:
        data = y_scaled
    seq   = data[-window:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq.reshape(1, window, -1), dtype=torch.float32)
            p = model(x).item()
            preds.append(p)
            new_row       = seq[-1:].copy()
            new_row[0, 0] = p
            seq = np.vstack([seq[1:], new_row])
    return scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
