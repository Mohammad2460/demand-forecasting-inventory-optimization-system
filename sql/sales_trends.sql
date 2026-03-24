CREATE OR REPLACE VIEW sales_trends AS
SELECT
    series_id,
    store_id,
    item_id,
    date,
    demand,
    sell_price,
    AVG(demand) OVER (
        PARTITION BY series_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_demand,
    AVG(demand) OVER (
        PARTITION BY series_id
        ORDER BY date
        ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
    ) AS rolling_28d_demand
FROM fact_sales;
