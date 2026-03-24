# Interview Guide

## Elevator Pitch

I built a retail decision-support system that forecasts item-store demand and converts those forecasts into reorder recommendations. The project is not just a notebook model. It includes a reproducible data pipeline, SQL analysis layer, model benchmarking workflow, API, and dashboard.

## Business Problem

Retail teams need to answer two linked questions:

1. How much demand should we expect next?
2. Given current stock and supplier lead time, what should we reorder now?

This project connects both steps. That makes it more useful than a forecasting-only project.

## Why These 3 Models

### Seasonal naive

This is the baseline. It says, "repeat the recent seasonal pattern." It is important because a forecasting model should beat a simple benchmark, not just produce a low error number in isolation.

### Holt-Winters

This is the classical time-series model in v1. It captures level, trend, and seasonality in a way that is easy to explain and strong enough to be meaningful on retail data.

### HistGradientBoostingRegressor

This is the ML model. It uses engineered lag, rolling, calendar, SNAP, event, and price features. It represents the "feature-based forecasting" approach and gives a modern tabular ML comparison without unnecessary infrastructure.

## Why RMSSE

I used RMSSE as the primary scale-normalized metric because retail demand series have very different scales. Some items move every day, while others move much less often. RMSSE normalizes forecast error using each series' historical seasonal variation, so model comparison is fairer across item-store combinations.

I also reported WAPE because it is easier to communicate to business stakeholders.

## Inventory Recommendation Logic

The inventory layer in v1 is intentionally simple:

- lead time demand comes from the forecast
- safety stock comes from forecast error and service level
- reorder point is lead time demand plus safety stock
- recommended order quantity is the shortfall from current stock to reorder point
- recommendations are rounded to case-pack size

That means the project ends with an operational recommendation, not only a chart.

## Questions You Should Be Ready To Answer

### How did you prevent leakage?

Lag and rolling features are shifted so each row only uses information available before the prediction date. For the ML model, future forecasts are generated recursively, which avoids using actual future demand values during prediction.

### Why not use MAPE?

MAPE becomes unstable when actual demand is zero or near zero. Retail demand often has sparse or intermittent series, so RMSSE and WAPE are more reliable.

### Why not use ARIMA or a deep learning model?

For v1, the goal was a polished, explainable system. Holt-Winters gave a strong classical benchmark, and HistGradientBoosting covered the ML angle with less tuning overhead and easier reproducibility.

### Is the inventory data real?

The inventory policy inputs are deterministic and synthetic because the M5 dataset does not include real stock and procurement tables. I documented that clearly and kept the logic realistic enough for business demonstration without claiming the source had those fields.

## Strong Resume Bullets

- Built a retail demand forecasting and inventory optimization system using Python, SQL, DuckDB, FastAPI, and Streamlit.
- Engineered a reproducible pipeline to transform raw M5 sales data into store-item daily forecasting features and evaluation tables.
- Benchmarked seasonal naive, Holt-Winters, and HistGradientBoosting models using rolling backtests with MAE, RMSE, RMSSE, and WAPE.
- Designed a reorder recommendation engine using lead time demand, safety stock, reorder point, and case-pack rounding to translate forecasts into inventory actions.

## What Makes This Stronger Than A Notebook

- It has a data pipeline
- It persists outputs in a warehouse
- It includes SQL assets
- It exposes an API
- It has a usable dashboard
- It tells a business story from prediction to decision
