from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ["store_id", "item_id", "dept_id", "cat_id", "state_id"]
NUMERIC_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "sell_price_filled",
    "price_change_7",
    "day_of_week",
    "week_of_year",
    "month",
    "is_month_start",
    "is_month_end",
    "snap_active",
    "has_event",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _safe_price_fill(series: pd.Series) -> pd.Series:
    return series.ffill().bfill().fillna(0.0)


def _select_snap_flag(frame: pd.DataFrame) -> pd.Series:
    conditions = [
        frame["state_id"].eq("CA"),
        frame["state_id"].eq("TX"),
        frame["state_id"].eq("WI"),
    ]
    choices = [frame["snap_CA"], frame["snap_TX"], frame["snap_WI"]]
    return pd.Series(np.select(conditions, choices, default=0), index=frame.index)


def add_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    enriched["sell_price_filled"] = (
        enriched.groupby("series_id")["sell_price"].transform(_safe_price_fill)
    )
    enriched["day_of_week"] = enriched["date"].dt.dayofweek.astype(int)
    enriched["week_of_year"] = enriched["date"].dt.isocalendar().week.astype(int)
    enriched["month"] = enriched["date"].dt.month.astype(int)
    enriched["is_month_start"] = enriched["date"].dt.is_month_start.astype(int)
    enriched["is_month_end"] = enriched["date"].dt.is_month_end.astype(int)
    enriched["has_event"] = (
        enriched[["event_name_1", "event_name_2"]].notna().any(axis=1).astype(int)
    )
    enriched["snap_active"] = _select_snap_flag(enriched).astype(int)
    return enriched


def prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_base_features(frame).sort_values(["series_id", "date"]).reset_index(drop=True)
    demand_groups = enriched.groupby("series_id")["demand"]
    price_groups = enriched.groupby("series_id")["sell_price_filled"]

    for lag in (1, 7, 14, 28):
        enriched[f"lag_{lag}"] = demand_groups.shift(lag)

    enriched["rolling_mean_7"] = demand_groups.transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).mean()
    )
    enriched["rolling_mean_28"] = demand_groups.transform(
        lambda series: series.shift(1).rolling(28, min_periods=28).mean()
    )
    enriched["rolling_std_28"] = demand_groups.transform(
        lambda series: series.shift(1).rolling(28, min_periods=28).std()
    )
    enriched["price_change_7"] = (
        enriched["sell_price_filled"] / price_groups.shift(7).replace(0, np.nan)
    ) - 1.0
    enriched["rolling_std_28"] = enriched["rolling_std_28"].fillna(0.0)
    enriched["price_change_7"] = enriched["price_change_7"].replace([np.inf, -np.inf], np.nan)
    enriched["price_change_7"] = enriched["price_change_7"].fillna(0.0)

    return enriched.dropna(subset=["lag_28", "rolling_mean_28"]).reset_index(drop=True)


def _lag(values: Iterable[float], lag: int) -> float:
    values = list(values)
    if not values:
        return 0.0
    if len(values) >= lag:
        return float(values[-lag])
    return float(values[-1])


def _rolling(values: Iterable[float], window: int) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return np.asarray([0.0], dtype=float)
    if len(values) < window:
        return values
    return values[-window:]


def build_recursive_feature_row(
    history_frame: pd.DataFrame, future_row: pd.Series | dict, season_length: int
) -> pd.DataFrame:
    row = pd.Series(future_row)
    history = history_frame.sort_values("date").copy()
    history["sell_price"] = history["sell_price"].ffill().bfill().fillna(0.0)

    demand_values = history["demand"].astype(float).tolist()
    price_values = history["sell_price"].astype(float).tolist()

    current_price = row.get("sell_price", np.nan)
    if pd.isna(current_price):
        current_price = price_values[-1] if price_values else 0.0
    price_lag_7 = _lag(price_values or [current_price], 7)
    iso_calendar = pd.Timestamp(row["date"]).isocalendar()

    feature_row = {
        "store_id": row["store_id"],
        "item_id": row["item_id"],
        "dept_id": row["dept_id"],
        "cat_id": row["cat_id"],
        "state_id": row["state_id"],
        "lag_1": _lag(demand_values, 1),
        "lag_7": _lag(demand_values, 7),
        "lag_14": _lag(demand_values, 14),
        "lag_28": _lag(demand_values, 28),
        "rolling_mean_7": float(_rolling(demand_values, 7).mean()),
        "rolling_mean_28": float(_rolling(demand_values, 28).mean()),
        "rolling_std_28": float(_rolling(demand_values, 28).std(ddof=0)),
        "sell_price_filled": float(current_price),
        "price_change_7": 0.0
        if price_lag_7 == 0
        else float(current_price / price_lag_7 - 1.0),
        "day_of_week": int(pd.Timestamp(row["date"]).dayofweek),
        "week_of_year": int(iso_calendar.week),
        "month": int(pd.Timestamp(row["date"]).month),
        "is_month_start": int(pd.Timestamp(row["date"]).is_month_start),
        "is_month_end": int(pd.Timestamp(row["date"]).is_month_end),
        "snap_active": int(
            row.get(f"snap_{row['state_id']}", 0)
            if f"snap_{row['state_id']}" in row
            else 0
        ),
        "has_event": int(
            pd.notna(row.get("event_name_1")) or pd.notna(row.get("event_name_2"))
        ),
    }

    features = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    return features
