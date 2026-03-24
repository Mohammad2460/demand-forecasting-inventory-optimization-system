from __future__ import annotations

import pytest

from retail_forecasting.forecasting.metrics import compute_metrics, mae, rmse, rmsse, wape


def test_metric_functions_match_expected_values() -> None:
    actual = [10, 14, 12]
    predicted = [11, 13, 12]
    train_history = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    assert mae(actual, predicted) == pytest.approx(2 / 3)
    assert rmse(actual, predicted) == pytest.approx((2 / 3) ** 0.5)
    assert wape(actual, predicted) == pytest.approx(2 / 36)
    assert rmsse(actual, predicted, train_history, season_length=7) == pytest.approx(
        ((2 / 3) / 49) ** 0.5
    )


def test_compute_metrics_returns_expected_keys() -> None:
    metrics = compute_metrics([1, 2], [1, 3], [1, 2, 3, 4, 5, 6, 7, 8], season_length=7)
    assert set(metrics) == {"mae", "rmse", "rmsse", "wape"}
