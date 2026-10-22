# PhishGuard_AI — Improvement Roadmap & Gap Analysis

Deep analysis of the proposed hardening roadmap against the actual codebase,
with status, feasibility, and a practical execution order.

> Status legend: ✅ implemented & verified · 🟡 scaffolded / partial · ❌ not done

---

## 1. Critical Production Hardening

| Item | Status | Codebase reality | Plan |
|---|---|---|---|
| Persistence (cases/quarantine) | ✅ **done (SQLite)** | Was in-memory; `database/` was empty. Added `storage/persistence.py` (write-through SQLite) wired into `CaseManager` + `EmailQuarantineManager`; records **and** the quarantine encryption key survive restart. | Next: PostgreSQL + SQLAlchemy/Alembic for real deployments |
| Key management | 🟡 | Auto-provisioned AES-GCM key, now persisted to SQLite meta (dev-grade). | Vault / KMS / Key Vault + rotation |
| Authentication | 🟡 | Simple API key; **open mode** when `API_KEYS` unset. Middleware exists (`AuthGuardMiddleware`). | JWT + roles (Admin/Analyst/Viewer) + optional OIDC |
| Rate limiting | 🟡 | `RateLimiterMiddleware` exists (in-memory). | Redis-backed + per-tenant quotas |
| Observability | 🟡 | `observability/metrics.py`, `tracing.py`, `alerting.py` exist. | Wire OpenTelemetry → Prometheus/Grafana |
| SSRF & input safety | 🟡 | `SSRFGuard` used by network-bound analyzers. | Expand allow/deny lists, normalization, sandbox timeouts |

## 2. Feature Completeness

| Item | Status | Notes |
|---|---|---|
| Remaining analyzers importable | ✅ | Fixed missing enum members (`AnalyzerResourceClass.COMPUTE/NETWORK_IO`, `ScanType.DOMAIN`) — all 10 previously-broken modules now import. Not yet registered (need engines). |
| Visual similarity / screenshot comparison | 🟡 | `web/visual_similarity.py` exists; needs a real browser in sandbox. |
| Threat-intel feeds | 🟡 | `threat_intel/{virustotal,abuseipdb,urlhaus,aggregator}.py` exist; not wired into analyzers. |
| YARA ruleset management | 🟡 | `attachment/yara_scanner.py` exists; needs ruleset store + upload API. |
| Full MIME parsing + attachment recursion | ❌ | Currently limited to first 1024 chars of `email_text`. |
| Async long-running scans (Celery) | 🟡 | `messaging/celery_app.py` + `docker/Dockerfile.worker` exist. |
| Case lifecycle (SLA, escalation, audit) | 🟡 | `CaseWorkflowManager` + status machine exist; no SLA/escalation. |
| Evidence attachment | 🟡 | `case_management/evidence.py` exists. |
| SIEM/SOAR + Jira/ServiceNow auto-ticket | ❌ | None wired. |
| Quarantine release/approval/retention/gateway integration | 🟡 | Lifecycle states exist; no gateway connectors. |

## 3. Operator Experience & Dashboard

| Item | Status | Notes |
|---|---|---|
| SPA / HTMX+Alpine frontend | ❌ | Single-file HTML dashboard (functional). |
| Real-time updates (WebSocket/SSE) | ❌ | Polling via Refresh only. |
| Advanced filtering / bulk actions | 🟡 | Basic case table filtering exists. |
| Multi-tenant switcher, dark mode, audit view | ❌ | — |

## 4. ML / MLOps Maturity

- 🟡 `ml_inference/` + `mlops_pipeline/` exist with feature extractors.
- ❌ No model versioning / A-B / canary, DVC/MLflow, feature store, ONNX/TensorRT, SHAP/LIME, adversarial testing.

## 5. Deployment & Operations

- 🟡 `docker/docker-compose.yml` + `Dockerfile.{api,sandbox,worker}` exist.
- ❌ No CI/CD, no secrets manager integration, no horizontal scaling, no backup/DR.

## 6. Nice-to-Have / Differentiating

- ❌ Browser/Outlook/Gmail add-in, public API + dev portal, threat-hunting queries, automated reports, i18n, Slack/Teams/Discord rich cards, sandbox detonation network sim.

---

## Recommended Execution Order (practical)

1. ✅ **Persistence (SQLite)** — done this session (biggest risk reduction).
2. **PostgreSQL + Alembic** swap (replace SQLite adapter, keep same interface).
3. **Auth hardening** — JWT + RBAC (Admin/Analyst/Viewer), keep open mode for dev.
4. **Wire more analyzers** — register offline-safe ones (URLParser, DomainAnalyzer typosquat/entropy, BodyAnalyzer with MIME support) now that they import.
5. **Observability** — `/metrics` endpoint + OTel traces, dashboard latency/error tiles.
6. **Async scans** — Celery + Redis for long scans with SSE progress.
7. **Dashboard upgrade** — SSE live updates + bulk actions; consider HTMX before a full SPA.
8. **Threat-intel wiring + YARA store**.
9. **SIEM/SOAR + email-gateway connectors.**
10. **MLOps** — MLflow + ONNX + explainability.

---

## Delivered this session

- `storage/persistence.py` — SQLite write-through store (cases/quarantine/meta).
- `case_management/cases.py` — optional `db_path`, load-on-start + write-through.
- `quarantine/email_quarantine.py` — optional `db_path`, durable encryption key via meta.
- `api_gateway/main.py` — `DB_PATH` (env-overridable), managers now persistent.
- `engine/orchestrator.py` — added `COMPUTE`/`NETWORK_IO`/`DOMAIN` enum members + dispatcher routing; all bundled analyzers import again.
- `.gitignore` — `storage/data/` added.

Verified: cases + quarantine + encrypted bodies survive a full process restart.
