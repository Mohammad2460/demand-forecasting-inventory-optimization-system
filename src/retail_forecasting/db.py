from __future__ import annotations

from pathlib import Path

import duckdb

from retail_forecasting.config import get_settings


def connect(database_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    settings = get_settings()
    target = database_path or settings.warehouse_path
    target.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(target))


def execute_sql_file(
    connection: duckdb.DuckDBPyConnection, sql_path: Path
) -> duckdb.DuckDBPyConnection:
    query = sql_path.read_text(encoding="utf-8")
    connection.execute(query)
    return connection
