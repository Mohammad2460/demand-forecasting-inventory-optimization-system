from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail_forecasting.forecasting.train import run_training_pipeline
from retail_forecasting.pipeline.ingest import ingest_to_warehouse
from retail_forecasting.pipeline.inventory import build_inventory_recommendations


def create_synthetic_m5_dataset(
    raw_dir: Path, history_days: int = 42, future_days: int = 7
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    total_days = history_days + future_days
    calendar_dates = pd.date_range("2024-01-01", periods=total_days, freq="D")

    calendar_rows = []
    for day_index, current_date in enumerate(calendar_dates, start=1):
        calendar_rows.append(
            {
                "date": current_date.date().isoformat(),
                "wm_yr_wk": 1000 + ((day_index - 1) // 7),
                "weekday": current_date.day_name(),
                "wday": current_date.dayofweek + 1,
                "month": current_date.month,
                "year": current_date.year,
                "d": f"d_{day_index}",
                "event_name_1": "PromoWeekend" if current_date.dayofweek == 5 else None,
                "event_type_1": "Cultural" if current_date.dayofweek == 5 else None,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": int(current_date.dayofweek in (4, 5)),
                "snap_TX": int(current_date.dayofweek in (5, 6)),
                "snap_WI": int(current_date.dayofweek in (3, 4)),
            }
        )
    pd.DataFrame(calendar_rows).to_csv(raw_dir / "calendar.csv", index=False)

    store_defs = [("CA_1", "CA"), ("TX_1", "TX")]
    item_defs = [("ITEM_1", "FOODS"), ("ITEM_2", "HOBBIES")]
    sales_rows = []
    for store_index, (store_id, state_id) in enumerate(store_defs):
        for item_index, (item_id, cat_id) in enumerate(item_defs):
            row = {
                "id": f"{item_id}_{store_id}_validation",
                "item_id": item_id,
                "dept_id": f"{cat_id}_1",
                "cat_id": cat_id,
                "store_id": store_id,
                "state_id": state_id,
            }
            base_level = 16 + (item_index * 4) + (store_index * 2)
            seasonal_pattern = [0, 1, 3, 2, 4, 6, 2]
            for day_offset in range(history_days):
                seasonality = seasonal_pattern[day_offset % 7]
                trend = day_offset * 0.20
                demand = int(round(base_level + seasonality + trend))
                row[f"d_{day_offset + 1}"] = demand
            sales_rows.append(row)
    pd.DataFrame(sales_rows).to_csv(raw_dir / "sales_train_validation.csv", index=False)

    calendar_df = pd.DataFrame(calendar_rows)
    weekly_keys = sorted(calendar_df["wm_yr_wk"].unique().tolist())
    price_rows = []
    for store_index, (store_id, _) in enumerate(store_defs):
        for item_index, (item_id, _) in enumerate(item_defs):
            for week_index, wm_yr_wk in enumerate(weekly_keys):
                price_rows.append(
                    {
                        "store_id": store_id,
                        "item_id": item_id,
                        "wm_yr_wk": wm_yr_wk,
                        "sell_price": round(
                            4.5 + (item_index * 0.40) + (store_index * 0.15) + (week_index * 0.02),
                            2,
                        ),
                    }
                )
    pd.DataFrame(price_rows).to_csv(raw_dir / "sell_prices.csv", index=False)


@pytest.fixture()
def built_system(tmp_path: Path) -> dict[str, Path]:
    raw_dir = tmp_path / "data" / "raw" / "m5"
    artifact_dir = tmp_path / "outputs" / "artifacts"
    warehouse_dir = tmp_path / "outputs" / "warehouse"
    db_path = warehouse_dir / "test.duckdb"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    create_synthetic_m5_dataset(raw_dir)

    ingest_to_warehouse(
        raw_dir=raw_dir,
        database_path=db_path,
        top_n_items=2,
        horizon=7,
        artifact_dir=artifact_dir,
    )
    run_training_pipeline(
        database_path=db_path,
        artifact_dir=artifact_dir,
        horizon=7,
        season_length=7,
        backtest_windows=1,
    )
    build_inventory_recommendations(database_path=db_path, seed=7)

    return {
        "raw_dir": raw_dir,
        "artifact_dir": artifact_dir,
        "db_path": db_path,
    }
