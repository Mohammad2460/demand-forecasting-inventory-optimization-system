PYTHON ?= python3

.PHONY: install ingest train recommend run api dashboard test

install:
	$(PYTHON) -m pip install '.[dev]'

ingest:
	$(PYTHON) -m retail_forecasting.cli ingest

train:
	$(PYTHON) -m retail_forecasting.cli train

recommend:
	$(PYTHON) -m retail_forecasting.cli recommend

run:
	$(PYTHON) -m retail_forecasting.cli run-all

api:
	$(PYTHON) -m retail_forecasting.cli serve-api

dashboard:
	streamlit run src/retail_forecasting/dashboard/app.py

test:
	pytest
