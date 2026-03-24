# Contributing

Thanks for contributing to this project.

The goal of this repository is to keep the codebase production-style, easy to understand, and easy to maintain, so small, clean improvements are preferred over large mixed changes.

## Local Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project with development dependencies:

   ```bash
   pip install '.[dev]'
   ```

3. Place the M5 source files in `data/raw/m5` if you want to run the full real-data workflow.

## Common Commands

Run the full pipeline:

```bash
retail-forecast run-all
```

Start the API:

```bash
retail-forecast serve-api
```

Start the dashboard:

```bash
streamlit run src/retail_forecasting/dashboard/app.py
```

Run tests:

```bash
pytest
```

## Contribution Guidelines

- Keep changes focused and easy to review.
- Prefer separate commits for docs, tooling, and feature work.
- Update the README when the project behavior or setup changes.
- Do not commit raw M5 CSV files, local `.env` files, or generated warehouse artifacts.
- Before pushing, run `pytest` and make sure the app still starts locally.

## Suggested Commit Style

Examples:

- `docs: clarify local setup and configuration`
- `ci: add github actions test workflow`
- `feat: improve forecast explorer filtering`
- `fix: correct reorder point calculation edge case`
