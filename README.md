# PhishGuard_AI

Enterprise-grade AI phishing detection platform. Multi-layer analysis of email,
URL, and attachment content with quarantine, sandboxing, case management, and
a production MLOps pipeline.

## Highlights
- **API gateway** (FastAPI) with auth, rate limiting, tenant isolation, and correlation IDs
- **Detection engine** with modular analyzers (headers, NLP, visual similarity, IOC, attachments)
- **Reputation & IOC** extraction, SSRF guard, PII redaction, audit-log hash chaining
- **Policy engine** with configurable risk thresholds (block / quarantine / review)
- **Quarantine, sandbox, and case-management** sub-systems
- **MLOps pipeline** (prepare → train → evaluate → ONNX export) with LightGBM / HistGB
- **Observability** (Prometheus metrics, tracing, alerting) and Celery workers
- **Docker + GitHub Actions CI/CD**

## Quick Start

```bash
# 1. Create a virtual environment (Python 3.11 or 3.12 recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env       # then edit real values

# 4. Run the API gateway
uvicorn api_gateway.main:app --reload
# Swagger UI: http://localhost:8000/docs

# 5. Run the Celery worker (separate terminal)
celery -A messaging.celery_app.celery_app worker --loglevel=INFO
```

## MLOps Pipeline

```bash
cd mlops_pipeline
pip install -r ../requirements-ml.txt
python -m stages.01_prepare_data --config configs/pipeline.yaml
python -m stages.02_train          --config configs/pipeline.yaml
python -m stages.03_evaluate       --config configs/pipeline.yaml
python -m stages.04_export_onnx    --config configs/pipeline.yaml
```

## Tests

```bash
pytest -q
```

## Docs
See [docs](docs/README.md) for architecture, API, setup, and deployment guides.