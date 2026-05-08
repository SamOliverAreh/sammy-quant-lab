"""
LSTM model — univariate and multivariate (exogenous features supported).
Target: log_returns (stationary, mean-reverting).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class LSTMModel(nn.Module):
    def __init__(self, n_features=1, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc   = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def prepare_sequences(series, window=20, exog=None):
    """
    Build supervised sequences from univariate series + optional exog features.
    Returns X (n_samples, window, n_features), y (n_samples, 1), scaler_y, scaler_X
    """
    y_vals = series.values.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_vals)

    if exog is not None and len(exog.columns) > 0:
        X_raw    = exog.values
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        data     = np.hstack([y_scaled, X_scaled])
    else:
        data     = y_scaled
        scaler_X = None

    n_feat = data.shape[1]
    X, y   = [], []
    for i in range(window, len(data)):
        X.append(data[i - window: i])
        y.append(y_scaled[i])
    return np.array(X), np.array(y), scaler_y, scaler_X, n_feat


def train_lstm(series, window=20, epochs=50, lr=0.001, hidden=64, layers=2, exog=None):
    X, y, scaler_y, scaler_X, n_feat = prepare_sequences(series, window, exog)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model     = LSTMModel(n_features=n_feat, hidden=hidden, layers=layers)
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

    # In-sample predictions (for actual vs predicted plot)
    model.eval()
    with torch.no_grad():
        insample_scaled = model(X_t).numpy()
    insample = scaler_y.inverse_transform(insample_scaled).flatten()

    return model, scaler_y, scaler_X, losses, insample, series.values[window:]


def forecast_lstm(model, series, scaler_y, steps=30, window=20, scaler_X=None, exog=None):
    model.eval()
    y_vals   = series.values.reshape(-1, 1)
    y_scaled = scaler_y.transform(y_vals)

    if scaler_X is not None and exog is not None and len(exog.columns) > 0:
        X_scaled = scaler_X.transform(exog.values)
        data     = np.hstack([y_scaled, X_scaled])
    else:
        data = y_scaled

    seq   = data[-window:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq.reshape(1, window, -1), dtype=torch.float32)
            p = model(x).item()
            preds.append(p)
            # Roll sequence: update only the target column, hold exog fixed
            new_row    = seq[-1:].copy()
            new_row[0, 0] = p
            seq = np.vstack([seq[1:], new_row])

    return scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()