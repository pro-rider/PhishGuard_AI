# ==============================================================================
# PhishGuard_AI - Makefile
# ==============================================================================
SHELL := /bin/bash

.PHONY: help venv install dev-install run run-worker test lint train export docker-build docker-up docker-down clean

help:
	@echo "Targets: venv install dev-install run run-worker test lint train export docker-build docker-up docker-down clean"

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

dev-install: install
	pip install -r requirements-ml.txt

run:
	uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000 --reload

run-worker:
	celery -A messaging.celery_app.celery_app worker --loglevel=INFO

test:
	pytest -q

lint:
	ruff check .
	bandit -r api_gateway core engine security storage policy

train:
	cd mlops_pipeline && python -m stages.01_prepare_data && python -m stages.02_train && python -m stages.03_evaluate

export:
	cd mlops_pipeline && python -m stages.04_export_onnx --config configs/models/url_model.yaml

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache