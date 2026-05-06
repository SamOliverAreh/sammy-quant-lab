import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def prepare_sequences(series, window=20):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window: i])
        y.append(data[i])

    return np.array(X), np.array(y), scaler


def train_lstm(series, window=20, epochs=50, lr=0.001, hidden=64, layers=2):
    X, y, scaler = prepare_sequences(series, window)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel(hidden=hidden, layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()

    train_losses = []
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        train_losses.append(loss.item())

    return model, scaler, train_losses


def forecast_lstm(model, series, scaler, steps=30, window=20):
    model.eval()

    data = scaler.transform(series.values.reshape(-1, 1))
    seq = data[-window:].copy()

    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq.reshape(1, window, 1), dtype=torch.float32)
            p = model(x).item()
            preds.append(p)
            seq = np.vstack([seq[1:], [[p]]])

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
