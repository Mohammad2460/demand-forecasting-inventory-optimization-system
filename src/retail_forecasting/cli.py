from __future__ import annotations

import argparse
from pprint import pprint

from retail_forecasting.config import get_settings
from retail_forecasting.forecasting.train import run_training_pipeline
from retail_forecasting.pipeline.ingest import ingest_to_warehouse
from retail_forecasting.pipeline.inventory import build_inventory_recommendations


def _print_summary(result: dict) -> None:
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{key}:")
            pprint(value)
        else:
            print(f"{key}: {value}")


def run_all() -> None:
    ingest_summary = ingest_to_warehouse()
    train_summary = run_training_pipeline()
    inventory_summary = build_inventory_recommendations()
    print("Ingestion summary")
    _print_summary(ingest_summary)
    print("\nTraining summary")
    _print_summary(train_summary)
    print("\nInventory summary")
    _print_summary(inventory_summary)


def serve_api() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "retail_forecasting.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retail demand forecasting and inventory optimization system."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Ingest raw M5 data into DuckDB.")
    subparsers.add_parser("train", help="Train and evaluate forecasting models.")
    subparsers.add_parser(
        "recommend", help="Build inventory policy inputs and recommendations."
    )
    subparsers.add_parser("run-all", help="Run ingestion, training, and inventory.")
    subparsers.add_parser("serve-api", help="Serve the FastAPI application.")

    args = parser.parse_args()

    if args.command == "ingest":
        _print_summary(ingest_to_warehouse())
    elif args.command == "train":
        _print_summary(run_training_pipeline())
    elif args.command == "recommend":
        _print_summary(build_inventory_recommendations())
    elif args.command == "run-all":
        run_all()
    elif args.command == "serve-api":
        serve_api()


if __name__ == "__main__":
    main()
