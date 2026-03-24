CREATE OR REPLACE VIEW inventory_risk AS
SELECT
    recommendations.store_id,
    recommendations.item_id,
    recommendations.current_on_hand,
    recommendations.lead_time_days,
    recommendations.lead_time_demand,
    recommendations.safety_stock,
    recommendations.reorder_point,
    recommendations.days_of_cover,
    recommendations.recommended_order_qty,
    recommendations.status,
    errors.mean_rmsse,
    forecasts.model_name
FROM inventory_recommendations AS recommendations
LEFT JOIN series_error_profile AS errors
    ON recommendations.series_id = errors.series_id
LEFT JOIN (
    SELECT DISTINCT series_id, model_name
    FROM production_forecast
) AS forecasts
    ON recommendations.series_id = forecasts.series_id
ORDER BY recommendations.recommended_order_qty DESC, recommendations.store_id, recommendations.item_id;
