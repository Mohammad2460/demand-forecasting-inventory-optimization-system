from __future__ import annotations

from statistics import NormalDist

import pandas as pd
import pytest

from retail_forecasting.pipeline.inventory import (
    compute_inventory_recommendations,
    round_to_case_pack,
)


def test_round_to_case_pack_rounds_up() -> None:
    assert round_to_case_pack(0, 6) == 0
    assert round_to_case_pack(13, 6) == 18
    assert round_to_case_pack(12, 6) == 12


def test_inventory_recommendation_uses_reorder_gap_and_case_pack() -> None:
    forecast_frame = pd.DataFrame(
        {
            "series_id": ["CA_1__ITEM_1"] * 7,
            "forecast_date": pd.date_range("2024-02-01", periods=7, freq="D"),
            "prediction": [10.0] * 7,
        }
    )
    policy_inputs = pd.DataFrame(
        {
            "series_id": ["CA_1__ITEM_1"],
            "store_id": ["CA_1"],
            "item_id": ["ITEM_1"],
            "lead_time_days": [3],
            "service_level_target": [0.95],
            "case_pack": [6],
            "current_on_hand": [20],
        }
    )
    error_profile = pd.DataFrame(
        {
            "series_id": ["CA_1__ITEM_1"],
            "mean_rmse": [2.0],
        }
    )

    result = compute_inventory_recommendations(forecast_frame, policy_inputs, error_profile)
    recommendation = result.iloc[0]

    expected_safety_stock = NormalDist().inv_cdf(0.95) * 2.0 * (3 ** 0.5)
    expected_reorder_point = 30.0 + expected_safety_stock

    assert recommendation["lead_time_demand"] == pytest.approx(30.0)
    assert recommendation["safety_stock"] == pytest.approx(expected_safety_stock)
    assert recommendation["reorder_point"] == pytest.approx(expected_reorder_point)
    assert recommendation["recommended_order_qty"] == 18
    assert recommendation["days_of_cover"] == pytest.approx(2.0)
