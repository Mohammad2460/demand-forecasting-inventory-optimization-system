from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from retail_forecasting.config import get_settings
from retail_forecasting.db import connect


def _open_connection(database_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    return connect(database_path or get_settings().warehouse_path)


def _show_tables(connection: duckdb.DuckDBPyConnection) -> list[str]:
    return connection.execute("SHOW TABLES").fetchdf()["name"].tolist()


def _require_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    if table_name not in _show_tables(connection):
        raise ValueError(
            f"Required table '{table_name}' is missing. Run the pipeline before starting the API."
        )


def get_health_payload() -> dict[str, object]:
    settings = get_settings()
    with _open_connection() as connection:
        tables = _show_tables(connection)
        best_model = None
        if "model_summary" in tables:
            summary = connection.execute(
                "SELECT model_name FROM model_summary WHERE is_best = TRUE LIMIT 1"
            ).fetchone()
            best_model = summary[0] if summary else None

    return {
        "status": "ok" if tables else "empty",
        "warehouse_path": str(settings.warehouse_path),
        "available_tables": tables,
        "best_model": best_model,
    }


def get_catalog() -> dict[str, object]:
    with _open_connection() as connection:
        _require_table(connection, "series_catalog")
        catalog = connection.execute(
            "SELECT store_id, item_id FROM series_catalog ORDER BY store_id, item_id"
        ).df()

    return {
        "stores": sorted(catalog["store_id"].unique().tolist()),
        "series": catalog.to_dict(orient="records"),
    }


def get_metrics() -> dict[str, object]:
    settings = get_settings()
    with _open_connection() as connection:
        _require_table(connection, "model_summary")
        summary = connection.execute(
            """
            SELECT
                model_name,
                mean_mae,
                mean_rmse,
                mean_rmsse,
                mean_wape,
                series_count,
                split_count,
                is_best
            FROM model_summary
            ORDER BY is_best DESC, mean_rmsse ASC
            """
        ).df()

        reorder_now_count = 0
        if "inventory_recommendations" in _show_tables(connection):
            reorder_now_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM inventory_recommendations
                    WHERE recommended_order_qty > 0
                    """
                ).fetchone()[0]
            )

    best_model = summary.loc[summary["is_best"], "model_name"].iloc[0]
    series_count = int(summary["series_count"].max())
    return {
        "best_model": best_model,
        "forecast_horizon": settings.forecast_horizon,
        "series_count": series_count,
        "reorder_now_count": reorder_now_count,
        "models": summary.to_dict(orient="records"),
    }


def _series_exists(connection: duckdb.DuckDBPyConnection, store_id: str, item_id: str) -> bool:
    _require_table(connection, "series_catalog")
    exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM series_catalog
        WHERE store_id = ? AND item_id = ?
        """,
        [store_id, item_id],
    ).fetchone()[0]
    return bool(exists)


def get_forecast(store_id: str, item_id: str, horizon: int) -> dict[str, object]:
    with _open_connection() as connection:
        _require_table(connection, "production_forecast")
        _require_table(connection, "fact_sales")
        if not _series_exists(connection, store_id, item_id):
            raise ValueError(f"Series {store_id}/{item_id} was not found.")

        history = connection.execute(
            """
            SELECT date, demand
            FROM fact_sales
            WHERE store_id = ? AND item_id = ?
            ORDER BY date DESC
            LIMIT 90
            """,
            [store_id, item_id],
        ).df()
        forecast = connection.execute(
            """
            SELECT forecast_date, prediction, horizon_day, model_name
            FROM production_forecast
            WHERE store_id = ? AND item_id = ?
            ORDER BY forecast_date
            LIMIT ?
            """,
            [store_id, item_id, horizon],
        ).df()

    if forecast.empty:
        raise ValueError(f"No production forecast exists yet for {store_id}/{item_id}.")

    history = history.sort_values("date")
    return {
        "store_id": store_id,
        "item_id": item_id,
        "model_name": forecast["model_name"].iloc[0],
        "horizon": int(len(forecast)),
        "history": [
            {"date": row["date"].date().isoformat(), "demand": float(row["demand"])}
            for _, row in history.iterrows()
        ],
        "forecast": [
            {
                "date": row["forecast_date"].date().isoformat(),
                "prediction": float(row["prediction"]),
                "horizon_day": int(row["horizon_day"]),
            }
            for _, row in forecast.iterrows()
        ],
    }


def get_series_history(store_id: str, item_id: str) -> dict[str, object]:
    with _open_connection() as connection:
        _require_table(connection, "fact_sales")
        if not _series_exists(connection, store_id, item_id):
            raise ValueError(f"Series {store_id}/{item_id} was not found.")

        history = connection.execute(
            """
            SELECT date, demand
            FROM fact_sales
            WHERE store_id = ? AND item_id = ?
            ORDER BY date DESC
            LIMIT 180
            """,
            [store_id, item_id],
        ).df()

    history = history.sort_values("date")
    return {
        "store_id": store_id,
        "item_id": item_id,
        "history": [
            {"date": row["date"].date().isoformat(), "demand": float(row["demand"])}
            for _, row in history.iterrows()
        ],
    }


def get_recommendations(store_id: str | None = None) -> dict[str, object]:
    with _open_connection() as connection:
        _require_table(connection, "inventory_recommendations")
        query = """
            SELECT
                store_id,
                item_id,
                lead_time_days,
                service_level_target,
                case_pack,
                current_on_hand,
                lead_time_demand,
                safety_stock,
                reorder_point,
                days_of_cover,
                recommended_order_qty,
                status
            FROM inventory_recommendations
        """
        params: list[object] = []
        if store_id:
            query += " WHERE store_id = ?"
            params.append(store_id)
        query += " ORDER BY recommended_order_qty DESC, item_id"
        recommendations = connection.execute(query, params).df()

    return {
        "store_id": store_id,
        "recommendations": recommendations.to_dict(orient="records"),
    }
