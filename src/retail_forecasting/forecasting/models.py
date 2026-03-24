from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from retail_forecasting.pipeline.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_recursive_feature_row,
)


def seasonal_naive_forecast(
    train_values: pd.Series | np.ndarray, horizon: int, season_length: int
) -> np.ndarray:
    series = np.asarray(train_values, dtype=float)
    if len(series) == 0:
        return np.zeros(horizon, dtype=float)
    season = series[-season_length:] if len(series) >= season_length else series[-1:]
    repeats = math.ceil(horizon / len(season))
    predictions = np.tile(season, repeats)[:horizon]
    return np.maximum(predictions, 0.0)


def holt_winters_forecast(
    train_values: pd.Series | np.ndarray, horizon: int, season_length: int
) -> np.ndarray:
    series = np.asarray(train_values, dtype=float)
    if len(series) < max(season_length * 2, 12) or np.allclose(series, series[0]):
        return seasonal_naive_forecast(series, horizon, season_length)

    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=season_length,
            initialization_method="estimated",
        ).fit(optimized=True)
        predictions = model.forecast(horizon)
    except Exception:
        predictions = seasonal_naive_forecast(series, horizon, season_length)
    return np.maximum(np.asarray(predictions, dtype=float), 0.0)


def build_ml_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def fit_ml_model(training_frame: pd.DataFrame) -> Pipeline:
    model = build_ml_pipeline()
    model.fit(training_frame[FEATURE_COLUMNS], training_frame["demand"])
    return model


def recursive_ml_forecast(
    model: Pipeline,
    history_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    season_length: int,
) -> np.ndarray:
    working_history = history_frame.sort_values("date").copy()
    predictions: list[float] = []

    for _, future_row in future_frame.sort_values("date").iterrows():
        feature_row = build_recursive_feature_row(working_history, future_row, season_length)
        prediction = max(float(model.predict(feature_row)[0]), 0.0)
        predictions.append(prediction)

        next_history_row = future_row.copy()
        next_history_row["demand"] = prediction
        working_history = pd.concat(
            [working_history, pd.DataFrame([next_history_row])],
            ignore_index=True,
        )

    return np.asarray(predictions, dtype=float)
