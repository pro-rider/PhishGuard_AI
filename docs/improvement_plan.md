# PhishGuard_AI — Improvement Roadmap & Gap Analysis

Deep analysis of the proposed hardening roadmap against the actual codebase,
with status, feasibility, and a practical execution order.

> Status legend: ✅ implemented & verified · 🟡 scaffolded / partial · ❌ not done

---

## 1. Critical Production Hardening

| Item | Status | Codebase reality | Plan |
|---|---|---|---|
| Persistence (cases/quarantine) | ✅ **done (SQLAlchemy + Alembic)** | Was stdlib-SQLite-only. Added `storage/database.py` (SQLite default, **PostgreSQL via `SQLALCHEMY_DATABASE_URI`**), `storage/models.py` (cases/quarantine/scan_jobs/audit_log/meta), `storage/repository.py` (drop-in `SqlAlchemyStore`), `alembic.ini` + `migrations/0001_initial`. Managers migrated; existing DBs stay compatible. | Alembic on production PG; Vault/KMS for keys |
| Key management | 🟡 | Auto-provisioned AES-GCM key, persisted to SQLite meta (dev-grade). | Vault / KMS / Key Vault + rotation |
| Authentication | ✅ **done (JWT + RBAC)** | Simple API key; **open mode** when `API_KEYS` unset. Middleware exists (`AuthGuardMiddleware`). | Now: JWT (HS256) via `POST /auth/token` + roles Admin/Analyst/Viewer + tenant scope. Next: OIDC / SSO. |
| Rate limiting | 🟡 | `RateLimiterMiddleware` exists (in-memory). | Redis-backed + per-tenant quotas |
| Observability | 🟡 | `observability/metrics.py`, `tracing.py`, `alerting.py` exist. `/ready` now probes the DB. | Wire OpenTelemetry → Prometheus/Grafana |
| SSRF & input safety | 🟡 | `SSRFGuard` used by network-bound analyzers. | Expand allow/deny lists, normalization, sandbox timeouts |
| Async long-running scans | ✅ **done** | `/scan` is synchronous. Added `/scan/async` (202 + durable job) + `/scan/jobs/{id}` + `/scan/jobs/{id}/events` (SSE). | Celery + Redis for multi-worker scale-out, webhooks |
| Audit logging | ✅ **done** | No audit trail. Added `storage/audit.py` immutable trail + `GET /audit`; hooked into scan/case/quarantine/auth. | Export, retention, tamper-evident hashing |

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
3. ✅ **Auth hardening** — JWT + RBAC (Admin/Analyst/Viewer), keep open mode for dev.
4. ✅ **Async scans** — `/scan/async` jobs + SSE (BackgroundTasks; swap to Celery+Redis for scale-out).
5. ✅ **Audit logging** — immutable DB trail + `GET /audit`.
6. ✅ **Observability / readiness** — `/ready` probes the DB; JSON logging + correlation ids already present.
7. **Wire more analyzers** — register offline-safe ones (URLParser, DomainAnalyzer typosquat/entropy) now that they import.
8. **Prometheus + OTel** metrics/traces; dashboard SSE live tiles.
9. **PostgreSQL + Alembic in production** (scaffold exists; switch via `SQLALCHEMY_DATABASE_URI`).
10. **Vault/KMS** key management; **MIME parsing**; **YARA store**; **SIEM/SOAR + gateway connectors**; **dashboard SPA**; **CI/CD**.

---

## Delivered this session

- **Persistence → SQLAlchemy + Alembic** — `storage/database.py` (SQLite default, Postgres via `SQLALCHEMY_DATABASE_URI`), `storage/models.py`, `storage/repository.py` (`SqlAlchemyStore`, drop-in), `alembic.ini` + `migrations/0001_initial`. Managers migrated; existing DBs stay compatible; Alembic `upgrade head` verified.
- **Async long-running scans** — `storage/scan_jobs.py` (`ScanJobManager`, durable pending→running→completed/failed), `POST /scan/async` (202 + job), `GET /scan/jobs/{id}`, `GET /scan/jobs/{id}/events` (SSE). BackgroundTasks executes; sync `/scan` unchanged.
- **Audit logging** — `storage/audit.py` (immutable trail: user/action/target/timestamp/IP) + `GET /audit`; hooked into scan/case/quarantine/auth.
- **DB-aware readiness** — `/ready` writes/reads the store and reports `db: ok|error`.
- **Integration tests** — `tests/integration/test_api.py` (standalone, throwaway DB): readiness, JWT+RBAC, sync scan, async job + SSE, cases, audit.
- (Earlier this session) **JWT + RBAC auth**, SQLite persistence, analyzer enum fixes.

Verified: persistence across restart; Alembic migrations; async job completes + SSE; RBAC 403 for viewer writes; 401 no-creds; open mode still passes; integration suite green.

## Delivered this session (round 3 — observability & detection transparency)

- **Prometheus `/metrics`** — `observability/metrics.py` (existing) now wired into the API. Module-level, idempotent instruments (`_setup_metrics`): `scan_total{verdict}` counter, `scan_latency_seconds` histogram, `scan_jobs_active` gauge, `cases_total{severity}`, `quarantine_total`. `/metrics` serves Prometheus text format (exempt from auth). Scrape with Prometheus/Grafana.
- **Explainability — top signals** — `/scan` (sync + async) now returns `signals`: the top contributing analyzers with score/confidence and their top findings (`_build_signals`), so analysts see *why* a verdict was reached.
- **Feature-flagged analyzer registration** — `ENABLE_EXTRA_ANALYZERS` (default **OFF**; graceful try/except) wires `url_parser` in. Because extra analyzers change weighted aggregation, it stays off until verdicts are re-baselined on your corpus.
- **Fixed latent CPU-pool bug** — `ConstrainedMetadata` stored an unpicklable `MappingProxyType`, so *any* CPU-bound analyzer (routed to the ProcessPoolExecutor) failed with `cannot pickle 'mappingproxy'`. Added `__reduce__` to make immutable metadata picklable — verified `url_parser` (CPU-bound, SSRF/homograph detection) now succeeds through the pool.

Status vs roadmap: items #6 (observability) and #7 (wire analyzers, safely behind a flag) are now partly done; item #8 (Prometheus) done. Remaining: OTel traces, dashboard SSE live tiles, PostgreSQL-in-prod, Vault/KMS, MIME parsing, YARA store, SIEM/SOAR + gateway connectors, dashboard SPA, CI/CD.

## Delivered this session (round 4 — secure-by-default / auth hardening)

- **Secure by default (fail-closed)** — `DEV_INSECURE` (default **OFF**). Open mode, the dev `SECRET_KEY`, and `/docs` + `/openapi.json` are refused unless the operator explicitly opts in with `DEV_INSECURE=true` (localhost only). `run_server.ps1` now sets it for local dev; production is locked down by default.
- **Least-privilege API keys** — a bare API key now defaults to **viewer** (was analyst); escalate explicitly via `key|role|tenant`.
- **Role-matrix audit + tenant isolation** — every route confirmed to carry `require_roles`; **fixed a cross-tenant leak**: `GET /scan/jobs/{id}` and `.../events` are now tenant-scoped (404 for another tenant's job). Full matrix documented in `project_guide.md` §2.
- **Brute-force protection on `/auth/token`** — in-memory login throttle (5 failures / 15 min window → 429 lockout, reset on success) + failed-login audit rows (`auth.token.failed`). Move to Redis for multi-worker.
- **Unit tests** — `tests/unit/test_auth_security.py` (fail-closed open mode, least-privilege keys, parse, throttle). Both unit + integration suites green.

**Honest remaining (infra/external-bound, documented, not faked):** KMS/Vault envelope encryption + key rotation (currently dev-grade persisted key), Postgres RLS + real columns (currently JSON-blob tables), Celery/Redis async runner + heartbeats/stuck-job recovery, MIME/header analyzers + full-message parsing, threat-intel feeds with caching, labeled-dataset eval harness (precision/recall), SSRF DNS-pinning/redirect revalidation, CI/CD (ruff/mypy/pytest+coverage/bandit/Trivy), gunicorn+multiprocess+Redis rate limiting, SRI/CSP + cookie-based dashboard auth, API versioning (/v1).

## Delivered this session (round 5 — analyst-grade case / quarantine workflow)

- **Case workflow API** — `GET/PATCH /cases/{id}`, `POST /cases/{id}/transition` (workflow state machine, illegal moves → 409, optional transition comment), `POST/GET /cases/{id}/comments`, `POST/GET /cases/{id}/evidence` (ioc/finding/attachment/report/log), `GET /cases/{id}/timeline` (audit + comments + evidence merged, chronological). All tenant-scoped (cross-tenant → 404) and RBAC (writes admin+analyst, reads +viewer).
- **Quarantine workflow API** — `GET /quarantine/{id}`, `POST /quarantine/{id}/release`, `POST /quarantine/{id}/delete` (409 on illegal transition), `GET /quarantine/{id}/preview` (decrypt-on-demand, sensitive read is audited).
- **Durability** — comments and evidence are now **persistent** (new `case_comments` / `case_evidence` tables + Alembic migration `0002`, SqlAlchemyStore write-through, reload on startup), matching cases/quarantine.
- **Fixed a latent bug** — `CaseManager.update()` reconstructed cases with `status` as a string, breaking `.value`; now coerced to the `CaseStatus` enum.
- **Tests** — integration suite extended to ~33 checks covering the full workflow (create→update→transition→comments→evidence→timeline, quarantine create→preview→release→delete, RBAC denials, cross-tenant 404). Alembic `0001→0002` verified clean on a throwaway DB. Unit suite still green.
