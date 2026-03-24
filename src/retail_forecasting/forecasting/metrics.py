from __future__ import annotations

import math

import numpy as np


def mae(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(y_true, y_pred) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual).sum()
    if denominator <= 1e-12:
        return 0.0
    return float(np.abs(actual - predicted).sum() / denominator)


def rmsse(y_true, y_pred, train_history, season_length: int = 7) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    train = np.asarray(train_history, dtype=float)

    if len(train) <= season_length:
        scale_errors = np.diff(train)
    else:
        scale_errors = train[season_length:] - train[:-season_length]

    if len(scale_errors) == 0:
        return 0.0

    scale = np.mean(scale_errors**2)
    if scale <= 1e-12:
        scale = 1.0

    forecast_mse = np.mean((actual - predicted) ** 2)
    return float(math.sqrt(forecast_mse / scale))


def compute_metrics(y_true, y_pred, train_history, season_length: int = 7) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "rmsse": rmsse(y_true, y_pred, train_history, season_length=season_length),
        "wape": wape(y_true, y_pred),
    }
