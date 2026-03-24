from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from retail_forecasting.api.schemas import (
    CatalogResponse,
    ForecastResponse,
    HealthResponse,
    MetricsResponse,
    RecommendationResponse,
    SeriesHistoryResponse,
)
from retail_forecasting.api.service import (
    get_catalog,
    get_forecast,
    get_health_payload,
    get_metrics,
    get_recommendations,
    get_series_history,
)
from retail_forecasting.config import get_settings


settings = get_settings()
app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Forecast demand, compare models, and serve inventory recommendations.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**get_health_payload())


@app.get("/catalog", response_model=CatalogResponse)
def catalog() -> CatalogResponse:
    try:
        return CatalogResponse(**get_catalog())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    try:
        return MetricsResponse(**get_metrics())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    store_id: str = Query(..., description="Store identifier."),
    item_id: str = Query(..., description="Item identifier."),
    horizon: int = Query(
        default=settings.forecast_horizon,
        ge=1,
        le=settings.forecast_horizon,
        description="Forecast horizon in days.",
    ),
) -> ForecastResponse:
    try:
        return ForecastResponse(**get_forecast(store_id, item_id, horizon))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/recommendations", response_model=RecommendationResponse)
def recommendations(store_id: str | None = None) -> RecommendationResponse:
    try:
        return RecommendationResponse(**get_recommendations(store_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/series/{store_id}/{item_id}", response_model=SeriesHistoryResponse)
def series_history(store_id: str, item_id: str) -> SeriesHistoryResponse:
    try:
        return SeriesHistoryResponse(**get_series_history(store_id, item_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
