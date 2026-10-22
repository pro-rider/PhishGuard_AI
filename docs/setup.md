# Setup

## Requirements
- Python 3.11 or 3.12 (3.13+ may break some ML dependencies)
- PostgreSQL 16, Redis 7 (or use docker-compose)
- Optional: MinIO for object storage

## Steps
1. Create venv and install `requirements.txt`.
2. Copy `.env.example` to `.env` and set secrets.
3. `uvicorn api_gateway.main:app --reload`
4. Run `celery -A messaging.celery_app.celery_app worker` for async tasks.

## Docker
```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up -d
```