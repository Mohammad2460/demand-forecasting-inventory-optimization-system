from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from retail_forecasting.config import get_settings
from retail_forecasting.db import connect, execute_sql_file
from retail_forecasting.forecasting.metrics import compute_metrics
from retail_forecasting.forecasting.models import (
    fit_ml_model,
    holt_winters_forecast,
    recursive_ml_forecast,
    seasonal_naive_forecast,
)
from retail_forecasting.pipeline.features import FEATURE_COLUMNS, prepare_training_frame


TRAINING_SQL_FILES = ("model_evaluation_summary.sql",)


@dataclass(frozen=True)
class BacktestWindow:
    split_id: str
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def build_backtest_windows(
    unique_dates: list[pd.Timestamp], horizon: int, n_windows: int
) -> list[BacktestWindow]:
    ordered_dates = sorted(pd.to_datetime(pd.Index(unique_dates)).unique())
    if len(ordered_dates) <= horizon * n_windows:
        raise ValueError("Not enough history to build the requested rolling backtest windows.")

    windows: list[BacktestWindow] = []
    total_dates = len(ordered_dates)
    for offset in range(n_windows, 0, -1):
        valid_start_idx = total_dates - (horizon * offset)
        valid_end_idx = valid_start_idx + horizon - 1
        train_end_idx = valid_start_idx - 1
        windows.append(
            BacktestWindow(
                split_id=f"split_{n_windows - offset + 1}",
                train_end=ordered_dates[train_end_idx],
                valid_start=ordered_dates[valid_start_idx],
                valid_end=ordered_dates[valid_end_idx],
            )
        )
    return windows


def _persist_table(connection, table_name: str, frame: pd.DataFrame) -> None:
    connection.register(f"{table_name}_df", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}_df")
    connection.unregister(f"{table_name}_df")


def _load_required_table(connection, table_name: str) -> pd.DataFrame:
    tables = connection.execute("SHOW TABLES").fetchdf()["name"].tolist()
    if table_name not in tables:
        raise ValueError(
            f"Required table '{table_name}' not found. Run the ingestion pipeline first."
        )
    return connection.execute(f"SELECT * FROM {table_name}").df()


def _build_model_summary(evaluation_detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        evaluation_detail.groupby("model_name", as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_rmsse=("rmsse", "mean"),
            mean_wape=("wape", "mean"),
            series_count=("series_id", "nunique"),
            split_count=("split_id", "nunique"),
        )
        .sort_values(["mean_rmsse", "mean_rmse", "mean_mae"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    summary["is_best"] = False
    if not summary.empty:
        summary.loc[0, "is_best"] = True
    return summary


def _generate_production_forecasts(
    best_model_name: str,
    fact_sales: pd.DataFrame,
    future_exog: pd.DataFrame,
    feature_frame: pd.DataFrame,
    season_length: int,
) -> tuple[pd.DataFrame, object | None]:
    production_rows: list[dict[str, object]] = []
    selected_model = None
    if best_model_name == "hist_gradient_boosting":
        selected_model = fit_ml_model(feature_frame)

    grouped_history = {
        series_id: frame.sort_values("date").copy()
        for series_id, frame in fact_sales.groupby("series_id", sort=False)
    }
    grouped_future = {
        series_id: frame.sort_values("date").copy()
        for series_id, frame in future_exog.groupby("series_id", sort=False)
    }

    for series_id, history_frame in grouped_history.items():
        future_frame = grouped_future.get(series_id)
        if future_frame is None or future_frame.empty:
            continue

        history_values = history_frame["demand"].to_numpy(dtype=float)
        if best_model_name == "seasonal_naive":
            predictions = seasonal_naive_forecast(
                history_values, len(future_frame), season_length
            )
        elif best_model_name == "holt_winters":
            predictions = holt_winters_forecast(
                history_values, len(future_frame), season_length
            )
        else:
            predictions = recursive_ml_forecast(
                selected_model, history_frame, future_frame, season_length
            )

        for horizon_day, (_, future_row) in enumerate(future_frame.iterrows(), start=1):
            production_rows.append(
                {
                    "series_id": series_id,
                    "store_id": future_row["store_id"],
                    "item_id": future_row["item_id"],
                    "forecast_date": pd.Timestamp(future_row["date"]),
                    "horizon_day": horizon_day,
                    "prediction": float(predictions[horizon_day - 1]),
                    "model_name": best_model_name,
                }
            )

    return pd.DataFrame(production_rows), selected_model


def run_training_pipeline(
    database_path: Path | None = None,
    artifact_dir: Path | None = None,
    horizon: int | None = None,
    season_length: int | None = None,
    backtest_windows: int = 2,
) -> dict[str, object]:
    settings = get_settings()
    database_path = database_path or settings.warehouse_path
    artifact_dir = artifact_dir or settings.artifact_dir
    horizon = horizon or settings.forecast_horizon
    season_length = season_length or settings.season_length

    with connect(database_path) as connection:
        fact_sales = _load_required_table(connection, "fact_sales")
        future_exog = _load_required_table(connection, "future_exog")
        series_catalog = _load_required_table(connection, "series_catalog")

    fact_sales["date"] = pd.to_datetime(fact_sales["date"])
    future_exog["date"] = pd.to_datetime(future_exog["date"])
    feature_frame = prepare_training_frame(fact_sales)

    windows = build_backtest_windows(
        fact_sales["date"].sort_values().unique().tolist(),
        horizon=horizon,
        n_windows=backtest_windows,
    )

    evaluation_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for window in windows:
        split_training_frame = feature_frame.loc[feature_frame["date"] <= window.train_end].copy()
        if split_training_frame.empty:
            raise ValueError(
                f"No training rows available for {window.split_id}. Check the history length."
            )
        ml_model = fit_ml_model(split_training_frame)

        split_history = fact_sales.loc[fact_sales["date"] <= window.train_end].copy()
        split_validation = fact_sales.loc[
            (fact_sales["date"] >= window.valid_start) & (fact_sales["date"] <= window.valid_end)
        ].copy()

        grouped_history = {
            series_id: frame.sort_values("date").copy()
            for series_id, frame in split_history.groupby("series_id", sort=False)
        }

        for series_id, validation_frame in split_validation.groupby("series_id", sort=False):
            history_frame = grouped_history.get(series_id)
            if history_frame is None or history_frame.empty:
                continue

            validation_frame = validation_frame.sort_values("date").copy()
            train_values = history_frame["demand"].to_numpy(dtype=float)
            actual_values = validation_frame["demand"].to_numpy(dtype=float)

            model_predictions = {
                "seasonal_naive": seasonal_naive_forecast(
                    train_values, len(validation_frame), season_length
                ),
                "holt_winters": holt_winters_forecast(
                    train_values, len(validation_frame), season_length
                ),
                "hist_gradient_boosting": recursive_ml_forecast(
                    ml_model, history_frame, validation_frame, season_length
                ),
            }

            for model_name, predictions in model_predictions.items():
                metrics = compute_metrics(
                    actual_values, predictions, train_values, season_length=season_length
                )
                evaluation_rows.append(
                    {
                        "split_id": window.split_id,
                        "series_id": series_id,
                        "store_id": validation_frame["store_id"].iloc[0],
                        "item_id": validation_frame["item_id"].iloc[0],
                        "forecast_start": window.valid_start,
                        "forecast_end": window.valid_end,
                        "model_name": model_name,
                        **metrics,
                    }
                )

                for point_index, (_, validation_row) in enumerate(validation_frame.iterrows()):
                    prediction_rows.append(
                        {
                            "split_id": window.split_id,
                            "series_id": series_id,
                            "store_id": validation_row["store_id"],
                            "item_id": validation_row["item_id"],
                            "date": pd.Timestamp(validation_row["date"]),
                            "model_name": model_name,
                            "actual": float(validation_row["demand"]),
                            "prediction": float(predictions[point_index]),
                        }
                    )

    evaluation_detail = pd.DataFrame(evaluation_rows)
    validation_predictions = pd.DataFrame(prediction_rows)
    model_summary = _build_model_summary(evaluation_detail)
    best_model_name = str(model_summary.loc[model_summary["is_best"], "model_name"].iloc[0])

    series_error_profile = (
        evaluation_detail.loc[evaluation_detail["model_name"] == best_model_name]
        .groupby(["series_id", "store_id", "item_id"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_rmsse=("rmsse", "mean"),
            mean_wape=("wape", "mean"),
        )
    )

    production_forecast, selected_model = _generate_production_forecasts(
        best_model_name=best_model_name,
        fact_sales=fact_sales,
        future_exog=future_exog,
        feature_frame=feature_frame,
        season_length=season_length,
    )

    with connect(database_path) as connection:
        _persist_table(connection, "model_evaluation_detail", evaluation_detail)
        _persist_table(connection, "validation_predictions", validation_predictions)
        _persist_table(connection, "model_summary", model_summary)
        _persist_table(connection, "series_error_profile", series_error_profile)
        _persist_table(connection, "production_forecast", production_forecast)
        _persist_table(connection, "series_catalog", series_catalog)

        for sql_name in TRAINING_SQL_FILES:
            execute_sql_file(connection, settings.sql_dir / sql_name)

    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "generated_at": generated_at,
        "selected_model": best_model_name,
        "horizon": horizon,
        "season_length": season_length,
        "feature_columns": FEATURE_COLUMNS,
        "models": model_summary.to_dict(orient="records"),
    }
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if selected_model is not None:
        with (artifact_dir / "hist_gradient_boosting.pkl").open("wb") as model_file:
            pickle.dump(selected_model, model_file)

    return {
        "generated_at": generated_at,
        "selected_model": best_model_name,
        "backtest_windows": [window.split_id for window in windows],
        "series_count": int(fact_sales["series_id"].nunique()),
        "forecast_rows": int(len(production_forecast)),
        "summary": model_summary.to_dict(orient="records"),
    }
