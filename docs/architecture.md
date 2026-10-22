# Architecture

PhishGuard_AI is a layered, fail-closed phishing detection platform.

## Layers
1. **API gateway** - auth, rate limiting, tenant isolation, correlation
2. **Engine** - `DetectionOrchestrator` schedules analyzers by resource class
   (async-io, CPU-bound, subprocess, external-network). Every network access
   passes through the SSRF guard.
3. **Analyzers** - header, NLP, visual similarity, URL, IOC, attachment feature
4. **Policy engine** - maps a score to verdict/action via risk thresholds
5. **Quarantine / Sandbox / Case management** - post-detection workflows
6. **MLOps pipeline** - prepare → train → evaluate → ONNX export → deploy
7. **Observability** - Prometheus metrics, tracing, alerting
8. **Messaging** - Celery async tasks

## Trust boundaries
- Analyzer inputs are untrusted; all values are bounded/sanitized.
- SSRF guard is fail-closed for any external network analyzer.
- Audit logs are hash-chained for tamper evidence.
- PII/credentials are redacted before logging.