from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _default_root_dir() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()


@dataclass(frozen=True)
class Settings:
    root_dir: Path = field(default_factory=_default_root_dir)
    raw_m5_dir: Path = field(
        default_factory=lambda: _default_root_dir()
        / os.getenv("RAW_M5_DIR", "data/raw/m5")
    )
    warehouse_path: Path = field(
        default_factory=lambda: _default_root_dir()
        / os.getenv("WAREHOUSE_PATH", "outputs/warehouse/m5.duckdb")
    )
    artifact_dir: Path = field(
        default_factory=lambda: _default_root_dir()
        / os.getenv("ARTIFACT_DIR", "outputs/artifacts")
    )
    sql_dir: Path = field(default_factory=lambda: _default_root_dir() / "sql")
    docs_dir: Path = field(default_factory=lambda: _default_root_dir() / "docs")
    forecast_horizon: int = field(
        default_factory=lambda: int(os.getenv("FORECAST_HORIZON", "28"))
    )
    top_n_items: int = field(default_factory=lambda: int(os.getenv("TOP_N_ITEMS", "50")))
    season_length: int = field(default_factory=lambda: int(os.getenv("SEASON_LENGTH", "7")))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    dashboard_api_url: str = field(
        default_factory=lambda: os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000")
    )

    @property
    def output_dir(self) -> Path:
        return self.warehouse_path.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.raw_m5_dir.mkdir(parents=True, exist_ok=True)
    settings.warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings
