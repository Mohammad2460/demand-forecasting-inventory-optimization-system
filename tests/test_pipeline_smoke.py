from __future__ import annotations

import duckdb


def test_end_to_end_pipeline_builds_expected_outputs(built_system) -> None:
    with duckdb.connect(str(built_system["db_path"])) as connection:
        tables = set(connection.execute("SHOW TABLES").fetchdf()["name"].tolist())
        expected_tables = {
            "fact_sales",
            "future_exog",
            "inventory_policy_inputs",
            "inventory_recommendations",
            "model_evaluation_detail",
            "model_summary",
            "production_forecast",
            "series_catalog",
            "series_error_profile",
            "validation_predictions",
        }
        assert expected_tables.issubset(tables)

        best_model = connection.execute(
            "SELECT model_name FROM model_summary WHERE is_best = TRUE"
        ).fetchone()[0]
        forecast_rows = connection.execute(
            "SELECT COUNT(*) FROM production_forecast"
        ).fetchone()[0]
        recommendation_rows = connection.execute(
            "SELECT COUNT(*) FROM inventory_recommendations"
        ).fetchone()[0]

    assert best_model in {"seasonal_naive", "holt_winters", "hist_gradient_boosting"}
    assert forecast_rows > 0
    assert recommendation_rows > 0
