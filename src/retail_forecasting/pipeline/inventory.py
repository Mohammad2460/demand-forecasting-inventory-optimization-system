from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from retail_forecasting.config import get_settings
from retail_forecasting.db import connect, execute_sql_file


INVENTORY_SQL_FILES = ("inventory_risk.sql",)


def round_to_case_pack(quantity: float, case_pack: int) -> int:
    if quantity <= 0:
        return 0
    if case_pack <= 1:
        return int(math.ceil(quantity))
    return int(math.ceil(quantity / case_pack) * case_pack)


def generate_inventory_policy_inputs(
    series_catalog: pd.DataFrame, recent_demand: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    catalog = (
        series_catalog.merge(recent_demand, on=["series_id", "store_id", "item_id"], how="left")
        .sort_values(["store_id", "item_id"])
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)

    catalog["lead_time_days"] = rng.choice([3, 7, 10, 14, 21], size=len(catalog))
    catalog["service_level_target"] = rng.choice(
        [0.90, 0.95, 0.98], size=len(catalog), p=[0.3, 0.5, 0.2]
    )
    catalog["case_pack"] = rng.choice([6, 12, 24], size=len(catalog), p=[0.25, 0.5, 0.25])

    demand_anchor = catalog["recent_avg_demand"].fillna(
        catalog["total_demand"] / catalog["history_days"].clip(lower=1)
    )
    on_hand_multiplier = rng.uniform(0.75, 1.75, size=len(catalog))
    current_on_hand = np.maximum(
        np.round(demand_anchor * catalog["lead_time_days"] * on_hand_multiplier).astype(int),
        0,
    )
    catalog["current_on_hand"] = current_on_hand

    return catalog[
        [
            "series_id",
            "store_id",
            "item_id",
            "lead_time_days",
            "service_level_target",
            "case_pack",
            "current_on_hand",
        ]
    ]


def compute_inventory_recommendations(
    forecast_frame: pd.DataFrame,
    policy_inputs: pd.DataFrame,
    error_profile: pd.DataFrame,
) -> pd.DataFrame:
    forecast_frame = forecast_frame.sort_values(["series_id", "forecast_date"]).copy()
    error_lookup = error_profile.set_index("series_id")
    recommendation_rows: list[dict[str, object]] = []

    for _, policy_row in policy_inputs.iterrows():
        series_id = policy_row["series_id"]
        series_forecast = forecast_frame.loc[forecast_frame["series_id"] == series_id].copy()
        if series_forecast.empty:
            continue

        lead_time_days = int(policy_row["lead_time_days"])
        service_level = float(policy_row["service_level_target"])
        case_pack = int(policy_row["case_pack"])
        current_on_hand = int(policy_row["current_on_hand"])

        forecast_values = series_forecast["prediction"].astype(float).tolist()
        mean_daily_forecast = float(np.mean(forecast_values)) if forecast_values else 0.0
        if lead_time_days <= len(forecast_values):
            lead_time_demand = float(np.sum(forecast_values[:lead_time_days]))
        else:
            lead_time_demand = mean_daily_forecast * lead_time_days

        rmse_value = (
            float(error_lookup.loc[series_id, "mean_rmse"])
            if series_id in error_lookup.index
            else max(mean_daily_forecast * 0.25, 0.1)
        )
        safety_stock = float(NormalDist().inv_cdf(service_level) * rmse_value * math.sqrt(lead_time_days))
        reorder_point = lead_time_demand + safety_stock
        days_of_cover = (
            float(current_on_hand / mean_daily_forecast)
            if mean_daily_forecast > 1e-12
            else 9999.0
        )
        reorder_gap = max(reorder_point - current_on_hand, 0.0)
        recommended_qty = round_to_case_pack(reorder_gap, case_pack)

        recommendation_rows.append(
            {
                "series_id": series_id,
                "store_id": policy_row["store_id"],
                "item_id": policy_row["item_id"],
                "lead_time_days": lead_time_days,
                "service_level_target": service_level,
                "case_pack": case_pack,
                "current_on_hand": current_on_hand,
                "lead_time_demand": lead_time_demand,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "days_of_cover": days_of_cover,
                "recommended_order_qty": recommended_qty,
                "status": "Reorder now" if recommended_qty > 0 else "Healthy",
            }
        )

    return pd.DataFrame(recommendation_rows).sort_values(
        ["store_id", "recommended_order_qty", "item_id"], ascending=[True, False, True]
    )


def _persist_table(connection, table_name: str, frame: pd.DataFrame) -> None:
    connection.register(f"{table_name}_df", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}_df")
    connection.unregister(f"{table_name}_df")


def build_inventory_recommendations(
    database_path: Path | None = None, seed: int = 42
) -> dict[str, object]:
    settings = get_settings()
    database_path = database_path or settings.warehouse_path

    with connect(database_path) as connection:
        fact_sales = connection.execute("SELECT * FROM fact_sales").df()
        production_forecast = connection.execute("SELECT * FROM production_forecast").df()
        series_catalog = connection.execute("SELECT * FROM series_catalog").df()
        error_profile = connection.execute("SELECT * FROM series_error_profile").df()

    fact_sales["date"] = pd.to_datetime(fact_sales["date"])
    production_forecast["forecast_date"] = pd.to_datetime(production_forecast["forecast_date"])

    recent_cutoff = fact_sales["date"].max() - pd.Timedelta(days=27)
    recent_demand = (
        fact_sales.loc[fact_sales["date"] >= recent_cutoff]
        .groupby(["series_id", "store_id", "item_id"], as_index=False)
        .agg(recent_avg_demand=("demand", "mean"))
    )

    policy_inputs = generate_inventory_policy_inputs(series_catalog, recent_demand, seed=seed)
    recommendations = compute_inventory_recommendations(
        forecast_frame=production_forecast,
        policy_inputs=policy_inputs,
        error_profile=error_profile,
    )

    with connect(database_path) as connection:
        _persist_table(connection, "inventory_policy_inputs", policy_inputs)
        _persist_table(connection, "inventory_recommendations", recommendations)

        for sql_name in INVENTORY_SQL_FILES:
            execute_sql_file(connection, settings.sql_dir / sql_name)

    reorder_count = int((recommendations["recommended_order_qty"] > 0).sum())
    return {
        "series_count": int(len(recommendations)),
        "reorder_count": reorder_count,
        "top_recommendations": recommendations.head(5).to_dict(orient="records"),
    }
