from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from retail_forecasting.config import get_settings


settings = get_settings()
API_BASE_URL = os.getenv("DASHBOARD_API_URL", settings.dashboard_api_url).rstrip("/")


def fetch_json(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
    query = f"?{urlencode(params)}" if params else ""
    with urlopen(f"{API_BASE_URL}{path}{query}") as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(show_spinner=False)
def load_catalog() -> dict[str, object]:
    return fetch_json("/catalog")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, object]:
    return fetch_json("/metrics")


def render_overview() -> None:
    metrics = load_metrics()
    recommendations = fetch_json("/recommendations")
    recommendations_df = pd.DataFrame(recommendations["recommendations"])
    best_model_row = next(model for model in metrics["models"] if model["is_best"])

    st.title("Demand Forecasting + Inventory Optimization")
    st.caption("An API-backed internal analytics app for retail planning decisions.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", best_model_row["model_name"].replace("_", " ").title())
    col2.metric("Series Tracked", f"{metrics['series_count']:,}")
    col3.metric("Reorder Now", f"{metrics['reorder_now_count']:,}")
    col4.metric("Best RMSSE", f"{best_model_row['mean_rmsse']:.3f}")

    st.subheader("Model Benchmark")
    benchmark_df = pd.DataFrame(metrics["models"])
    benchmark_chart = px.bar(
        benchmark_df,
        x="model_name",
        y=["mean_mae", "mean_rmse", "mean_rmsse", "mean_wape"],
        barmode="group",
        title="Validation Metrics Across Forecast Approaches",
    )
    st.plotly_chart(benchmark_chart, use_container_width=True)

    st.subheader("Top Reorder Recommendations")
    if recommendations_df.empty:
        st.info("No inventory recommendations are available yet.")
    else:
        st.dataframe(
            recommendations_df.head(20),
            width="stretch",
            hide_index=True,
        )


def render_forecast_explorer() -> None:
    catalog = load_catalog()
    series_df = pd.DataFrame(catalog["series"])
    store_id = st.selectbox("Store", options=sorted(series_df["store_id"].unique().tolist()))
    item_options = (
        series_df.loc[series_df["store_id"] == store_id, "item_id"].sort_values().unique().tolist()
    )
    item_id = st.selectbox("Item", options=item_options)
    forecast_payload = fetch_json(
        "/forecast",
        {"store_id": store_id, "item_id": item_id, "horizon": settings.forecast_horizon},
    )

    history_df = pd.DataFrame(forecast_payload["history"])
    forecast_df = pd.DataFrame(forecast_payload["forecast"])
    history_df["series"] = "History"
    history_df["value"] = history_df["demand"]
    forecast_df["series"] = "Forecast"
    forecast_df["value"] = forecast_df["prediction"]
    forecast_df = forecast_df.rename(columns={"date": "date"})

    line_df = pd.concat(
        [
            history_df[["date", "series", "value"]],
            forecast_df[["date", "series", "value"]],
        ],
        ignore_index=True,
    )
    chart = px.line(
        line_df,
        x="date",
        y="value",
        color="series",
        title=f"Store {store_id} | Item {item_id} | Model: {forecast_payload['model_name']}",
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(forecast_df, width="stretch", hide_index=True)


def render_model_comparison() -> None:
    metrics = load_metrics()
    benchmark_df = pd.DataFrame(metrics["models"])
    st.subheader("Model Comparison")
    st.dataframe(benchmark_df, width="stretch", hide_index=True)

    metric_to_plot = st.selectbox("Metric", options=["mean_mae", "mean_rmse", "mean_rmsse", "mean_wape"])
    chart = px.bar(
        benchmark_df.sort_values(metric_to_plot),
        x="model_name",
        y=metric_to_plot,
        color="is_best",
        title=f"{metric_to_plot} by model",
    )
    st.plotly_chart(chart, use_container_width=True)


def render_inventory_planner() -> None:
    catalog = load_catalog()
    store_options = [None] + catalog["stores"]
    store_label = st.selectbox("Store Filter", options=store_options, format_func=lambda value: value or "All stores")
    recommendations = fetch_json(
        "/recommendations",
        {"store_id": store_label} if store_label else None,
    )
    recommendations_df = pd.DataFrame(recommendations["recommendations"])

    if recommendations_df.empty:
        st.info("No recommendations available.")
        return

    st.subheader("Inventory Planner")
    st.dataframe(recommendations_df, width="stretch", hide_index=True)

    chart = px.bar(
        recommendations_df.head(20),
        x="item_id",
        y="recommended_order_qty",
        color="status",
        title="Top recommended order quantities",
    )
    st.plotly_chart(chart, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Retail Forecasting Dashboard",
        layout="wide",
    )

    try:
        health = fetch_json("/health")
    except Exception as exc:
        st.error(
            "The dashboard could not reach the API. Start the FastAPI service first.\n\n"
            f"Configured API URL: {API_BASE_URL}\n\n"
            f"Error: {exc}"
        )
        return

    st.sidebar.title("Navigation")
    st.sidebar.caption(f"API status: {health['status']}")
    page = st.sidebar.radio(
        "Page",
        options=[
            "Executive Overview",
            "Forecast Explorer",
            "Model Comparison",
            "Inventory Planner",
        ],
    )

    if page == "Executive Overview":
        render_overview()
    elif page == "Forecast Explorer":
        render_forecast_explorer()
    elif page == "Model Comparison":
        render_model_comparison()
    else:
        render_inventory_planner()


if __name__ == "__main__":
    main()
