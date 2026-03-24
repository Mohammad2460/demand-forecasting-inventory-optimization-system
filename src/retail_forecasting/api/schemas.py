from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class HistoryPoint(BaseModel):
    date: date
    demand: float


class ForecastPoint(BaseModel):
    date: date
    prediction: float
    horizon_day: int


class ModelMetric(BaseModel):
    model_name: str
    mean_mae: float
    mean_rmse: float
    mean_rmsse: float
    mean_wape: float
    series_count: int
    split_count: int
    is_best: bool


class MetricsResponse(BaseModel):
    best_model: str
    forecast_horizon: int
    series_count: int
    reorder_now_count: int
    models: list[ModelMetric]


class ForecastResponse(BaseModel):
    store_id: str
    item_id: str
    model_name: str
    horizon: int
    history: list[HistoryPoint]
    forecast: list[ForecastPoint]


class RecommendationItem(BaseModel):
    store_id: str
    item_id: str
    lead_time_days: int
    service_level_target: float
    case_pack: int
    current_on_hand: int
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    days_of_cover: float
    recommended_order_qty: int
    status: str


class RecommendationResponse(BaseModel):
    store_id: str | None
    recommendations: list[RecommendationItem]


class SeriesHistoryResponse(BaseModel):
    store_id: str
    item_id: str
    history: list[HistoryPoint]


class CatalogSeries(BaseModel):
    store_id: str
    item_id: str


class CatalogResponse(BaseModel):
    stores: list[str]
    series: list[CatalogSeries]


class HealthResponse(BaseModel):
    status: str
    warehouse_path: str
    available_tables: list[str]
    best_model: str | None
