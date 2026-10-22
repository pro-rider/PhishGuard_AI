# Deployment

## Containers
- `Dockerfile.api` - API gateway (exposed on :8000)
- `Dockerfile.worker` - Celery worker
- `Dockerfile.sandbox` - isolated sandbox (not host-exposed)

## Production checks
- Set `ENVIRONMENT=production`
- Enable PII redaction, signed model artifacts, and integrity checksums
- Run ONNX parity and quality gates before promoting a model
- Monitor drift, error rate, and latency via Prometheus

## CI/CD
GitHub Actions workflows under `.github/workflows/`:
- `ci-cd.yml` - lint, test, typecheck
- `model-validation-gate.yml` - PR model evaluation gate
- `docker-build.yml` - build images
- `model-registry-sync.yml` - register candidate models
- `adversarial-eval.yml` - scheduled adversarial evaluation