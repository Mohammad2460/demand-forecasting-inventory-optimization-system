from __future__ import annotations

import pandas as pd
import pytest

from retail_forecasting.pipeline.features import prepare_training_frame


def test_prepare_training_frame_creates_leakage_safe_lags() -> None:
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    frame = pd.DataFrame(
        {
            "series_id": ["CA_1__ITEM_1"] * 35,
            "store_id": ["CA_1"] * 35,
            "item_id": ["ITEM_1"] * 35,
            "dept_id": ["FOODS_1"] * 35,
            "cat_id": ["FOODS"] * 35,
            "state_id": ["CA"] * 35,
            "date": dates,
            "demand": list(range(1, 36)),
            "sell_price": [4.5] * 35,
            "event_name_1": [None] * 35,
            "event_name_2": [None] * 35,
            "snap_CA": [0] * 35,
            "snap_TX": [0] * 35,
            "snap_WI": [0] * 35,
        }
    )

    transformed = prepare_training_frame(frame)
    target_row = transformed.loc[transformed["date"] == pd.Timestamp("2024-01-29")].iloc[0]

    assert target_row["lag_1"] == 28
    assert target_row["lag_7"] == 22
    assert target_row["rolling_mean_7"] == pytest.approx(25.0)
    assert target_row["rolling_mean_28"] == pytest.approx(14.5)
