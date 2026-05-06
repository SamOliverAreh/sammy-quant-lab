import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def direction_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2:
        return 50.0
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)


def sharpe_ratio(returns, risk_free=0.0, periods=252):
    excess = np.array(returns) - risk_free / periods
    std = excess.std()
    return float((excess.mean() / std) * np.sqrt(periods)) if std != 0 else 0.0


def max_drawdown(prices):
    prices = np.array(prices)
    peak   = np.maximum.accumulate(prices)
    return float(((prices - peak) / (peak + 1e-12)).min())


def evaluate_all(y_true, y_pred):
    return {
        "RMSE":                   rmse(y_true, y_pred),
        "MAE":                    mae(y_true, y_pred),
        "MAPE":                   mape(y_true, y_pred),
        "Direction Accuracy (%)": direction_accuracy(y_true, y_pred),
    }
