# API Reference

The API gateway is a FastAPI application rooted at `api_gateway.main:app`.

## Entry
- Base path: `/api/v1`
- Interactive docs: `/docs` (Swagger UI)
- Health: `GET /health`

## Modules
- `auth_guard` - JWT / API-key auth middleware
- `rate_limiter` - per-tenant rate limiting (slowapi)
- `tenant_isolation` - tenant scoping middleware
- `correlation_id` - request correlation propagation

## Example

```bash
curl -X POST http://localhost:8000/api/v1/scans/email \
  -H "Content-Type: application/json" \
  -d '{"email_id": "msg-1", "metadata": {"subject": "Invoice"}}'
```

All scan submissions return a `ScanResult` with verdict, score, confidence, and
analyzer findings.