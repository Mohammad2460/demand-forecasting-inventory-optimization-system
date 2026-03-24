from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retail_forecasting.config import get_settings
from retail_forecasting.db import connect, execute_sql_file


REQUIRED_FILES = {
    "calendar": "calendar.csv",
    "sell_prices": "sell_prices.csv",
    "sales_train_validation": "sales_train_validation.csv",
}

INGEST_SQL_FILES = ("sales_trends.sql", "demand_seasonality.sql")


def load_raw_m5_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_FILES.values() if not (raw_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required M5 files in {raw_dir}: {joined}. "
            "Expected calendar.csv, sell_prices.csv, and sales_train_validation.csv."
        )

    calendar = pd.read_csv(raw_dir / REQUIRED_FILES["calendar"], parse_dates=["date"])
    prices = pd.read_csv(raw_dir / REQUIRED_FILES["sell_prices"])
    sales = pd.read_csv(raw_dir / REQUIRED_FILES["sales_train_validation"])
    return {"calendar": calendar, "prices": prices, "sales": sales}


def select_top_items(sales: pd.DataFrame, top_n_items: int) -> list[str]:
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    ranked = sales.groupby("item_id", as_index=True)[day_columns].sum().sum(axis=1)
    return ranked.sort_values(ascending=False).head(top_n_items).index.tolist()


def build_fact_sales(
    sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame, top_items: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    sales_subset = sales.loc[sales["item_id"].isin(top_items)].copy()

    melted = sales_subset.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_columns,
        var_name="d",
        value_name="demand",
    )

    calendar_columns = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    fact_sales = (
        melted.merge(calendar[calendar_columns], on="d", how="left")
        .merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .assign(
            series_id=lambda frame: frame["store_id"] + "__" + frame["item_id"],
            demand=lambda frame: frame["demand"].astype(float),
            sell_price=lambda frame: frame["sell_price"].astype(float),
        )
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )

    series_catalog = (
        fact_sales.groupby("series_id", as_index=False)
        .agg(
            store_id=("store_id", "first"),
            item_id=("item_id", "first"),
            dept_id=("dept_id", "first"),
            cat_id=("cat_id", "first"),
            state_id=("state_id", "first"),
            total_demand=("demand", "sum"),
            history_days=("date", "nunique"),
            history_start=("date", "min"),
            history_end=("date", "max"),
        )
        .sort_values(["store_id", "item_id"])
        .reset_index(drop=True)
    )

    filtered_prices = prices.loc[prices["item_id"].isin(top_items)].copy()
    return fact_sales, series_catalog, filtered_prices


def build_future_exog(
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    series_catalog: pd.DataFrame,
    actual_end_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    future_calendar = (
        calendar.loc[calendar["date"] > actual_end_date]
        .sort_values("date")
        .head(horizon)
        .copy()
    )
    if len(future_calendar) < horizon:
        raise ValueError(
            "Calendar does not contain enough future dates to build the forecast horizon."
        )

    future_calendar = future_calendar[
        [
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
    ].copy()

    future_exog = (
        series_catalog.assign(_join_key=1)
        .merge(future_calendar.assign(_join_key=1), on="_join_key")
        .drop(columns="_join_key")
        .merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )

    future_exog["sell_price"] = (
        future_exog.groupby("series_id")["sell_price"].transform(lambda series: series.ffill().bfill())
    )
    return future_exog


def _persist_table(connection, table_name: str, frame: pd.DataFrame) -> None:
    connection.register(f"{table_name}_df", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}_df")
    connection.unregister(f"{table_name}_df")


def ingest_to_warehouse(
    raw_dir: Path | None = None,
    database_path: Path | None = None,
    top_n_items: int | None = None,
    horizon: int | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, object]:
    settings = get_settings()
    raw_dir = raw_dir or settings.raw_m5_dir
    database_path = database_path or settings.warehouse_path
    top_n_items = top_n_items or settings.top_n_items
    horizon = horizon or settings.forecast_horizon
    artifact_dir = artifact_dir or settings.artifact_dir

    source_tables = load_raw_m5_tables(raw_dir)
    top_items = select_top_items(source_tables["sales"], top_n_items)
    fact_sales, series_catalog, filtered_prices = build_fact_sales(
        source_tables["sales"], source_tables["calendar"], source_tables["prices"], top_items
    )
    future_exog = build_future_exog(
        source_tables["calendar"],
        filtered_prices,
        series_catalog,
        fact_sales["date"].max(),
        horizon,
    )

    calendar_dim = source_tables["calendar"].copy()
    calendar_dim["date"] = pd.to_datetime(calendar_dim["date"])

    with connect(database_path) as connection:
        _persist_table(connection, "dim_calendar", calendar_dim)
        _persist_table(connection, "dim_prices", filtered_prices)
        _persist_table(connection, "series_catalog", series_catalog)
        _persist_table(connection, "fact_sales", fact_sales)
        _persist_table(connection, "future_exog", future_exog)

        for sql_name in INGEST_SQL_FILES:
            execute_sql_file(connection, settings.sql_dir / sql_name)

    metadata = {
        "top_items": top_items,
        "raw_dir": str(raw_dir),
        "database_path": str(database_path),
        "history_start": fact_sales["date"].min().date().isoformat(),
        "history_end": fact_sales["date"].max().date().isoformat(),
        "future_start": future_exog["date"].min().date().isoformat(),
        "future_end": future_exog["date"].max().date().isoformat(),
        "row_counts": {
            "fact_sales": int(len(fact_sales)),
            "series_catalog": int(len(series_catalog)),
            "future_exog": int(len(future_exog)),
        },
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "ingestion_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
