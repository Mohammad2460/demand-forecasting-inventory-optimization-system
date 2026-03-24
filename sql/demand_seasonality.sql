CREATE OR REPLACE VIEW demand_seasonality AS
SELECT
    store_id,
    item_id,
    weekday,
    AVG(demand) AS avg_daily_demand,
    STDDEV_SAMP(demand) AS demand_std,
    AVG(sell_price) AS avg_sell_price,
    COUNT(*) AS observations
FROM fact_sales
GROUP BY 1, 2, 3
ORDER BY store_id, item_id, weekday;
