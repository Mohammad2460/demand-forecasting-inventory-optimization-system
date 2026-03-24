from __future__ import annotations

import importlib
import sys

from fastapi.testclient import TestClient

from retail_forecasting.config import get_settings


def test_api_endpoints_return_expected_shapes(built_system, monkeypatch) -> None:
    monkeypatch.setenv("WAREHOUSE_PATH", str(built_system["db_path"]))
    get_settings.cache_clear()
    sys.modules.pop("retail_forecasting.api.main", None)

    api_module = importlib.import_module("retail_forecasting.api.main")
    client = TestClient(api_module.app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert "production_forecast" in health_response.json()["available_tables"]

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics_payload = metrics_response.json()
    assert metrics_payload["best_model"] in {
        "seasonal_naive",
        "holt_winters",
        "hist_gradient_boosting",
    }
    assert len(metrics_payload["models"]) == 3

    catalog_response = client.get("/catalog")
    assert catalog_response.status_code == 200
    catalog_payload = catalog_response.json()
    first_series = catalog_payload["series"][0]

    forecast_response = client.get(
        "/forecast",
        params={"store_id": first_series["store_id"], "item_id": first_series["item_id"], "horizon": 7},
    )
    assert forecast_response.status_code == 200
    forecast_payload = forecast_response.json()
    assert len(forecast_payload["forecast"]) == 7
    assert len(forecast_payload["history"]) > 0

    recommendation_response = client.get(
        "/recommendations", params={"store_id": first_series["store_id"]}
    )
    assert recommendation_response.status_code == 200
    assert len(recommendation_response.json()["recommendations"]) > 0
