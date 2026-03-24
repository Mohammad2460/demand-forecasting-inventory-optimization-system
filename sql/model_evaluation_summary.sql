CREATE OR REPLACE VIEW model_evaluation_summary AS
SELECT
    model_name,
    AVG(mae) AS avg_mae,
    AVG(rmse) AS avg_rmse,
    AVG(rmsse) AS avg_rmsse,
    AVG(wape) AS avg_wape,
    COUNT(DISTINCT series_id) AS series_count,
    COUNT(DISTINCT split_id) AS split_count
FROM model_evaluation_detail
GROUP BY 1
ORDER BY avg_rmsse, avg_rmse, avg_mae;
