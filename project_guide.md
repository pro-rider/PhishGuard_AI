# PhishGuard_AI — Project Guide

PhishGuard_AI is a phishing-detection platform that inspects URLs, emails, attachments and web content, scores them, and routes findings into an incident workflow (cases, quarantine, alerts). This guide covers setup, running, the API, and the operator dashboard.

---

## 1. Quick Start

> The project runs in **open mode** by default — no API key required for local development.

```powershell
cd C:\Users\Administrator\Desktop\PhishGuard_AI
.\run_server.ps1
```

- **Dashboard:** http://127.0.0.1:8000/
- **Interactive API docs (Swagger):** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

`run_server.ps1` clears any leftover `API_KEYS` environment variable, activates the venv, and starts `uvicorn` with hot-reload.

### Manual start (alternative)

```powershell
cd C:\Users\Administrator\Desktop\PhishGuard_AI
.\venv\Scripts\Activate.ps1
uvicorn api_gateway.main:app --reload
```

### Virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: training / MLOps toolchain
pip install -r requirements-ml.txt
```

---

## 2. Authentication Modes

The auth guard (`api_gateway/middleware/auth_guard.py`) resolves a caller to a **principal** (identity + roles + tenant) from either a Bearer JWT or a static API key, then role-based access control (`security/rbac.py`'s `require_roles`) enforces Admin / Analyst / Viewer per route.

> **Secure by default (fail-closed).** Open (no-auth) mode is **refused** unless you
> explicitly set `DEV_INSECURE=true` (local development, localhost-only). A real
> `SECRET_KEY` (≥32 chars) is required in every non-insecure deployment, and
> `/docs` + `/openapi.json` are disabled unless `DEV_INSECURE=true`. Use
> `.\run_server.ps1` for local dev (it sets `DEV_INSECURE=true`).

| Mode | When | Behavior |
|------|------|----------|
| **Open** (dev only) | No `API_KEYS`, no `AUTH_USERS`, **and** `DEV_INSECURE=true` | All endpoints accessible; no principal attached |
| **API-key** | `API_KEYS` set | Non-exempt endpoints require `X-API-Key` (or `Authorization: Bearer <key>`). Plain keys grant **`viewer`** (least privilege); use `key\|role1,role2\|tenant` for finer control |
| **JWT + RBAC** | `AUTH_USERS` set | `POST /auth/token` exchanges credentials for an HS256 JWT carrying roles + tenant; protected routes enforce roles |

Public/exempt endpoints (always open): `/`, `/dashboard`, `/health`, `/ready`, `/metrics`, `/auth/token`. (`/docs` and `/openapi.json` are public **only** in `DEV_INSECURE` mode.)

### Role matrix

| Route | admin | analyst | viewer |
|-------|:-----:|:-------:|:------:|
| `POST /scan` | ✓ | ✓ | — |
| `POST /scan/async` | ✓ | ✓ | — |
| `GET /scan/jobs/{id}` | ✓ | ✓ | ✓ |
| `GET /scan/jobs` | ✓ | ✓ | ✓ |
| `GET /scan/jobs/{id}/events` | ✓ | ✓ | ✓ |
| `POST /cases` | ✓ | ✓ | — |
| `GET /cases` | ✓ | ✓ | ✓ |
| `GET /cases/{id}` · `PATCH /cases/{id}` | ✓ | ✓ | read-only ✓ |
| `POST /cases/{id}/transition` | ✓ | ✓ | — |
| `POST /cases/{id}/comments` · `GET /cases/{id}/comments` | ✓ | ✓ | read ✓ |
| `POST /cases/{id}/evidence` · `GET /cases/{id}/evidence` | ✓ | ✓ | read ✓ |
| `GET /cases/{id}/timeline` | ✓ | ✓ | ✓ |
| `POST /quarantine` | ✓ | ✓ | — |
| `GET /quarantine` | ✓ | ✓ | ✓ |
| `GET /quarantine/{id}` · `GET /quarantine/{id}/preview` | ✓ | ✓ | ✓ |
| `POST /quarantine/{id}/release` · `POST /quarantine/{id}/delete` | ✓ | ✓ | — |
| `GET /alerts/severities` | ✓ | ✓ | ✓ |
| `GET /audit` | ✓ | ✓ | ✓ |

Job reads (`GET /scan/jobs/{id}`, `.../events`) are **tenant-scoped** — a caller
can never read another tenant's scan.

### Enabling JWT + RBAC

`AUTH_USERS` is a `;`-separated list of `username:password:role1,role2:tenant`:

```powershell
$env:AUTH_USERS = "admin:ChangeMe#1:admin:system;analyst:ChangeMe#2:analyst:system;viewer:ChangeMe#3:viewer:system"
$env:SECRET_KEY   = "a-long-random-secret-at-least-32-chars"
uvicorn api_gateway.main:app
```

Get a token:

```powershell
curl -X POST http://127.0.0.1:8000/auth/token -H "Content-Type: application/json" `
  -d '{"username":"analyst","password":"ChangeMe#2"}'
```

Use it: `Authorization: Bearer <access_token>`. Tokens expire per `ACCESS_TOKEN_EXPIRE_MINUTES` (default 30).

### Enabling API-key mode

```powershell
$env:API_KEYS = "key-a,key-b"
uvicorn api_gateway.main:app
```

The dashboard has an **API Key** field in the header — fill it in when the server runs in enforced mode.

---

## 3. Project Structure

```
PhishGuard_AI/
├── api_gateway/          # FastAPI app, middleware (auth, CORS, correlation, tenant, rate-limit)
├── engine/               # Detection pipeline & analyzers
│   └── analyzers/        # url, ip, hash, email_header, email_nlp, attachment, web_content, visual_similarity
├── ioc/                  # Indicator extraction (URLs, IPs, hashes)
├── policy/               # Policy & risk thresholds, tenant policy
├── quarantine/           # Email quarantine (encrypted at rest), release & retention
├── case_management/      # Incident cases, comments, evidence, workflow
├── security/             # SSRF guard; identity.py (JWT/Principal/PBKDF2), rbac.py (require_roles)
├── storage/              # persistence (SQLite fallback), database.py (engine), models.py (ORM),
│                         # repository.py (SqlAlchemyStore), scan_jobs.py, audit.py, encryption (AES-GCM)
├── observability/        # Alerting, metrics, tracing
├── messaging/            # Celery app
├── reputation/           # Threat-intel reputation (ip, hash)
├── sandbox/              # Attachment sandboxing & malware analysis
├── ml_inference/         # ML feature extractors / inference
├── mlops_pipeline/       # Training, evaluation, ONNX export, DVC
├── core/                 # Bootstrap & root config
├── frontend/             # Operator dashboard (single-file HTML)
├── config/internal/      # Hardened settings (pydantic-settings)
├── tests/                # Unit tests + tests/integration/test_api.py (API integration suite)
├── migrations/           # Alembic migration scripts (+ alembic.ini at root)
├── docker/               # API / sandbox / worker Dockerfiles + compose
├── run_server.ps1        # Dev launcher (open mode)
└── requirements*.txt
```

### 3.1 Module responsibilities (whole-project map)

| Package | What it owns | Key files |
|---------|--------------|-----------|
| `api_gateway/` | The FastAPI application: route wiring, middleware chain, request models, auth guard, all HTTP endpoints (scan, async jobs, cases, quarantine, alerts, audit, auth). | `main.py`, `middleware/` |
| `engine/` | The detection orchestrator + analyzer framework (score aggregation, verdict, deadlines, fail-closed). | `orchestrator.py`, `analyzers/` |
| `engine/analyzers/email/` | Offline email analyzers: **`header_analyzer`** (SPF/DKIM/DMARC + Reply-To/display-name spoofing — **registered**), plus `body_analyzer`, `nlp_detector`, `attachment_detector` (available, not yet registered). | `header_analyzer.py`, `body_analyzer.py` |
| `engine/analyzers/url/` | URL analysis; `url_parser` is feature-flagged via `ENABLE_EXTRA_ANALYZERS`. | `url_parser.py` |
| `ioc/` | Indicator extraction from text/URLs (domains, IPs, hashes). | — |
| `policy/` | Risk thresholds, verdict mapping, per-tenant policy. | — |
| `quarantine/` | Quarantine manager: encrypted-at-rest storage, release/delete/preview, retention. | `email_quarantine.py` |
| `case_management/` | Incident workflow: cases, comments, evidence, status state machine, timeline. | `cases.py`, `comments.py`, `evidence.py`, `workflow.py` |
| `security/` | SSRF guard, identity (JWT + PBKDF2), RBAC (`require_roles`). | `ssrf_guard.py`, `identity.py`, `rbac.py` |
| `storage/` | Durable persistence: SQLAlchemy engine, ORM models, repository adapter, scan-job manager, audit, AES-GCM encryption. | `database.py`, `models.py`, `repository.py`, `scan_jobs.py`, `audit.py`, `encryption.py` |
| `observability/` | Prometheus metrics, alerting, tracing, structured logging. | `metrics.py`, `alerting.py` |
| `messaging/` | Celery app (scaffold for scale-out job execution). | — |
| `reputation/` | Threat-intel reputation lookups (IP, hash). | — |
| `sandbox/` | Attachment sandboxing & malware analysis. | — |
| `ml_inference/` / `mlops_pipeline/` | ML feature extraction / training & evaluation (independent research subtree). | — |
| `config/internal/` | Hardened pydantic-settings (fail-closed, secret enforcement). | `settings.py` |
| `frontend/` | Single-file operator dashboard served at `/`. | `operator-dashboard.html` |

---

## 4. Detection Pipeline & Scoring

How an input becomes a verdict. The orchestrator (`engine/orchestrator.py`) runs
registered analyzers against a `ScanContext` and aggregates their results.

### 4.1 Flow

```
POST /scan (email_text / email_headers / target_url / filename)
  → ScanContext(scan_type=COMPREHENSIVE, email_payload, metadata, feature toggles)
  → orchestrator.run_scan(context)
      1. Each registered analyzer runs concurrently under a deadline (monotonic).
      2. _aggregate_results sums only SUCCESS analyzers:
           overall_score   = Σ (score_contribution × confidence) / Σ confidence
           confidence_score = Σ confidence / N
      3. Verdict derived from overall_score via policy thresholds
         (INFORMATIONAL → LOW → MEDIUM → HIGH → CRITICAL).
      4. Result carries `signals` (top contributing analyzers + their findings)
         for explainability.
  → if result.is_malicious (HIGH/CRITICAL): auto-create a case (idempotent per scan_id)
  → respond { scan_id, execution_state, overall_score, confidence_score, verdict,
             case_created, signals }
```

### 4.2 Analyzer execution states (graceful degradation)

| State | Meaning | Effect on score |
|-------|---------|-----------------|
| `SUCCESS` | Ran and returned findings | Contributes `score × confidence` to the weighted sum |
| `INVALID_INPUT` | Analyzer skipped (e.g. no headers supplied to `header_analyzer`) | Contributes nothing; never fails the scan |
| `SKIPPED` | Disabled / missing external engine | Contributes nothing |
| `TIMEOUT` / `FAILED` | Deadline hit / exception (sanitized) | Contributes nothing; captured as an error |

`fail_closed_on_total_failure` is on: if **no** analyzer succeeds, the scan is `FAILED`
rather than a clean `INFORMATIONAL` — a skipped analyzer can never produce a false "clean".

### 4.3 Registered analyzers

Registered by default in `api_gateway/main.py::create_app`:

| Analyzer | Reads | Emits |
|----------|-------|-------|
| `nlp_detector` | plain-text `email_text` | keyword/urgency/credential-bait signals |
| `heuristic_analyzer` | plain-text `email_text` | bounded heuristic rules (urgency, credential bait, …) |
| `header_analyzer` | full `email_headers` (RFC 5322) | SPF/DKIM/DMARC alignment, `Reply-To` vs `From` mismatch, display-name / Return-Path spoofing (e.g. HIGH `EMAIL.ALIGNMENT.REPLY_TO_MISMATCH`) |

`ENABLE_EXTRA_ANALYZERS=true` additionally registers offline extras (e.g. `url_parser`);
this changes weighted aggregation, so re-baseline verdicts on your corpus first. Other
bundled analyzers (body, attachment, web-content, visual-similarity) are **not** registered
because they need external engines (browser, live DNS/SSL, YARA rulesets) that could
otherwise risk fail-closed verdicts.

### 4.4 Auto-case escalation

A scan whose verdict is `HIGH` or `CRITICAL` automatically creates an incident case
(idempotent per `scan_id`), so confirmed threats appear on the dashboard's Cases view
without manual triage.

---

## 5. Operator Dashboard

`frontend/operator-dashboard.html` — a single-file, self-contained dashboard served by the API at **`/`**.

Views: **Overview** (KPIs + case-severity chart), **Scan Console** (run a scan → verdict gauge), **Cases**, **Quarantine Queue**, **Alert Severities**.

It talks to the live API: when served by the server it auto-uses the same origin; when opened as a local file it defaults to `http://127.0.0.1:8000`. Charts use ECharts (loaded via jsDelivr).

---

## 6. API Reference

### `POST /scan`
Run a detection scan.

```json
{
  "target_url": "https://example.com/login",
  "email_text": "URGENT: verify your account",
  "email_headers": "Return-Path: <bounce@evil.com>\r\nFrom: \"PayPal Security\" <security@paypal.com>\r\nReply-To: <verify@evil.com>",
  "filename": "invoice.pdf",
  "enable_sandbox": false,
  "enable_yara": true
}
```

**`email_headers` (optional)** — a full RFC 5322 header block. When supplied, the
`header_analyzer` runs real header-security checks (SPF/DKIM/DMARC alignment from
`Authentication-Results`, `Reply-To` vs `From` mismatch, display-name / Return-Path
spoofing). It contributes a `header_analyzer` signal (e.g. a **HIGH**
`EMAIL.ALIGNMENT.REPLY_TO_MISMATCH` finding). When omitted it returns
`INVALID_INPUT` and is skipped, so plain-text scans are unaffected.

Response:

```json
{
  "scan_id": "scan_...",
  "scan_type": "comprehensive",
  "execution_state": "completed",
  "total_execution_time_ms": 0.01,
  "overall_score": 0.0,
  "confidence_score": 1.0,
  "verdict": "INFORMATIONAL",
  "signals": [
    {
      "analyzer": "heuristic_analyzer",
      "score": 45.0,
      "confidence": 0.617,
      "findings": [{"rule_id": "email.urgency", "severity": "MEDIUM", "message": "..."}]
    }
  ]
}
```

`verdict` ∈ `INFORMATIONAL | LOW | MEDIUM | HIGH | CRITICAL`. `signals` is the
**explainability** output — the top contributing analyzers and their findings, so an
analyst can see *why* a verdict was reached (also present in async job results).

### `POST /scan/async`
Submit a long-running scan. Returns a **durable job** immediately (`202`-style payload):

```json
{
  "job_id": "job_...",
  "status": "pending",
  "poll_url": "/scan/jobs/job_...",
  "events_url": "/scan/jobs/job_.../events"
}
```

### `GET /scan/jobs/{job_id}`
Poll a scan job. `status` ∈ `pending | running | completed | failed`; `result` holds the same payload as `POST /scan` once completed.

### `GET /scan/jobs/{job_id}/events`
Server-Sent Events (SSE) stream: progress events, then a `result` event and a `done` event. Subscribe with `EventSource` or `curl -N`.

### `GET /scan/jobs`
List scan jobs (latest first).

### `GET /audit`
Read the immutable audit trail (who did what, when, from which IP).

### `GET /ready`
Readiness probe that actually checks the durable store: `{ "status": "ready", "db": "ok" }`.

### `GET /metrics`
Prometheus text-format metrics for scraping (exempt from auth): `scan_total{verdict}`,
`scan_latency_seconds` histogram, `scan_jobs_active`, `cases_total{severity}`,
`quarantine_total`. Point Prometheus/Grafana at this endpoint.

### `POST /cases`
Create an incident case. Body: `{ "title", "description", "severity", "assignee" }`.

### `GET /cases?status=<status>`
List cases. `status` ∈ `open | in_progress | on_hold | resolved | closed | archived`.

### `GET /cases/{case_id}` · `PATCH /cases/{case_id}`
Fetch one case / update its editable fields (`title`, `description`, `assignee`, `severity`).

### `POST /cases/{case_id}/transition`
Status transition. Body: `{ "new_status", "comment?" }`. Transitioned per the workflow
state machine; an illegal move returns `409`. A transition comment is stored if provided.

### `POST /cases/{case_id}/comments` · `GET /cases/{case_id}/comments`
Add / list case comments. Add body: `{ "body", "visibility": "internal|customer" }`.

### `POST /cases/{case_id}/evidence` · `GET /cases/{case_id}/evidence`
Attach / list evidence references. Attach body: `{ "kind", "source", "storage_key", "metadata?" }`;
`kind` ∈ `ioc | finding | attachment | report | log`.

### `GET /cases/{case_id}/timeline`
Chronological activity timeline for a case (audit events + comments + evidence).

### `POST /quarantine`
Quarantine a message. Body: `{ "message_id", "body", "risk_score" }`.

### `GET /quarantine`
List the quarantine queue.

### `GET /quarantine/{qid}` · `POST /quarantine/{qid}/release` · `POST /quarantine/{qid}/delete` · `GET /quarantine/{qid}/preview`
Fetch an item; release / delete it (with an optional `reason`, audited, `409` on illegal
transition); preview the decrypted body (sensitive read — access is logged).

### `GET /alerts/severities`
Return the recognized severity scale.

---

## 7. Persistence

- **Default:** SQLite via SQLAlchemy at `storage/data/phishguard.db` (gitignored). Set `PHISHGUARD_DB` to relocate.
- **PostgreSQL:** set `SQLALCHEMY_DATABASE_URI=postgresql+psycopg2://user:pass@host/db` (add the driver) — the app and migrations automatically target it.
- **Migrations:** `python -m alembic upgrade head` (see `alembic.ini` / `migrations/`). Existing databases stay compatible (same `id`/`data` table layout). Tables: `cases`, `quarantine`, `scan_jobs`, `audit_log`, `case_comments`, `case_evidence`, `meta` (case comments + evidence are now durable too).

## 8. Tests

```powershell
python tests/integration/test_api.py
```

Covers readiness, JWT+RBAC, sync scan, async job + SSE, cases, quarantine, the audit trail, and the full case/quarantine **workflow** (transition/comments/evidence/timeline/release/delete/preview) against a throwaway SQLite DB.

---

## 9. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:

| Gate | Command | Failure = |
|------|---------|-----------|
| Lint | `ruff check api_gateway config engine security storage quarantine case_management observability tests` | Any F/E/W/B issue (correctness set; see `pyproject.toml`) |
| Syntax | `python -m compileall -q <core dirs> tests` | Any module that fails to compile |
| Security | `bandit -r <core dirs> -lll` | Any **HIGH** severity finding |
| Deps | `pip-audit -r requirements-ci.txt` | Any known vulnerability in the CI dependency set |
| Unit | `python tests/unit/test_auth_security.py` | Auth / security-posture regression |
| Header | `python tests/unit/test_header_analyzer.py` | Email header analyzer (spoofing / alignment) regression |
| Integration | `python tests/integration/test_api.py` | Any end-to-end API regression |

CI installs `requirements-ci.txt` — the *tested core runtime* — not the aspirational
`requirements.txt`, so the pipeline is green and reproducible. **Dependency note:**
the core runtime was moved to `fastapi>=0.159` / `starlette>=1.9`, resolving the
starlette 0.46.x CVE set that `pip-audit` was flagging.

## 10. Unit Tests & Syntax Check

```powershell
.\venv\Scripts\python.exe -m pytest tests/unit -q
```

The two standalone unit suites can also be run directly:

```powershell
.\venv\Scripts\python.exe tests\unit\test_auth_security.py
.\venv\Scripts\python.exe tests\unit\test_header_analyzer.py
```

A `compileall` over the whole tree is a quick syntax gate:

```powershell
.\venv\Scripts\python.exe -m compileall -q .
```

---

## 11. Configuration

Settings live in `config/internal/settings.py` (pydantic-settings). Override via environment variables or a `.env` file (see `.env.example`).

Highlights: `API_KEYS`, `VIRUSTOTAL_API_KEY`, `ABUSEIPDB_API_KEY`, `URLHAUS_API_KEY`, SSRF protections (`ALLOW_*_ACCESS`), rate-limit bounds.

---

## 12. Security Notes

- **SSRF guard** (`security/ssrf_guard.py`): blocks private/link-local/loopback targets by default; DNS-based resolution may reject benchmark IPs (normal).
- **Encryption at rest** (`storage/encryption.py`): AES-GCM keyring. The quarantine manager auto-provisions an in-process active key so it works out of the box; bring your own key manager for persistence.
- **Error hygiene**: internal errors are never surfaced raw (sanitized 500s).

---

## 13. Known Limitations & Roadmap

- **Analyzers are now registered (default set).** `api_gateway/main.py` registers `NLPDetectorAnalyzer` + `HeuristicContentAnalyzer` (offline-safe, operate on the plain-text email / URL the gateway receives). `/scan` now returns real verdicts (HIGH/CRITICAL reachable), and a HIGH/CRITICAL scan **auto-creates a case** (idempotent per `scan_id`). Verified end-to-end: phishing email → `CRITICAL` + case; benign email → `INFORMATIONAL`, no case.
- **Bundled analyzers now import.** Fixed missing enum members (`AnalyzerResourceClass.COMPUTE` / `NETWORK_IO`, `ScanType.DOMAIN`) — all previously-crashing modules load. They are **not yet registered** because they need external engines (browser, live DNS/SSL, yara rulesets) that would risk fail-closed verdicts. See `docs/improvement_plan.md`.
- The email **body** analyzers are still fed only the first 1024 chars of `email_text` (the `ConstrainedMetadata` value cap). The **header** analyzer, by contrast, consumes the full `email_headers` block (up to 256 KB) supplied via `/scan`, so header-level spoofing/alignment signals are not truncated.
- **Persistence:** cases + quarantine + comments + evidence + the quarantine encryption key survive restarts via SQLAlchemy (SQLite default; PostgreSQL via `SQLALCHEMY_DATABASE_URI` + Alembic migrations). Key management still dev-grade (persisted key) — see roadmap for Vault/KMS.
- **Full roadmap:** the prioritized production roadmap (P0 detection hardening → P1 analyst workflow → P2 enterprise/platform), the "honest remaining" list (Vault/KMS encryption, Postgres RLS, Celery/Redis jobs, MIME body analyzers, threat-intel feeds, eval harness, SSRF pinning, gunicorn + Redis rate limiting, SPA/CSP dashboard, API versioning, mypy, Trivy), and every delivery round is tracked in **`docs/improvement_plan.md`**.

---

## 14. Git Workflow

```powershell
git add -A
git commit -m "message"
git push origin main
```

Repo: `https://github.com/pro-rider/PhishGuard_AI`

---

## 15. Maintenance & Operations Guide

How to keep PhishGuard_AI running, upgrading, and safe. Read this before changing
schema, auth, analyzers, or endpoints.

### 15.1 Daily operations (quick reference)

```powershell
# Run the dev server (open mode, hot-reload, :8000)
cd C:\Users\Administrator\Desktop\PhishGuard_AI
.\run_server.ps1

# Run the API integration suite (throwaway SQLite DB)
python tests\integration\test_api.py

# Syntax gate over all source
python -m compileall -q api_gateway config case_management quarantine storage engine security tests

# Apply database migrations (see 15.3)
python -m alembic upgrade head
```

Key URLs: dashboard `/` · API docs `/docs` · health `/health` · readiness `/ready`
(check actually queries the DB).

### 15.2 Runtime configuration

Settings live in `config/internal/settings.py` (pydantic-settings). Override with
environment variables or `.env`. The most important knobs:

| Env / field | Effect |
|---|---|
| `AUTH_USERS` | Enable JWT + RBAC (`username:password:roles:tenant;…`). Open mode when unset. |
| `API_KEYS` | Enable API-key mode (`key` or `key\|role1,role2\|tenant`). |
| `SECRET_KEY` | JWT signing secret (≥32 chars; required in staging/production). |
| `PHISHGUARD_DB` | SQLite DB file path. |
| `SQLALCHEMY_DATABASE_URI` | Switch persistence to PostgreSQL (app + migrations auto-target it). |
| `VIRUSTOTAL_API_KEY` etc. | Threat-intel providers. |
| `ALLOW_*_ACCESS`, `ENFORCE_SSRF_PROTECTION` | Outbound SSRF guard. |
| `ENABLE_EXTRA_ANALYZERS` | Default `False`. Registers extra offline analyzers (e.g. `url_parser`) that change weighted aggregation — enable only after re-baselining verdicts on your corpus. |
| `DEV_INSECURE` | Default `False`. **Fail-closed**: open mode, dev `SECRET_KEY`, and `/docs` are refused unless `true` (local dev, localhost only). `run_server.ps1` sets it. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT lifetime (default 30). `/auth/token` also enforces a failed-login lockout (5 failures / 15 min). |

### 15.3 Database maintenance

**Default (SQLite)** — no setup. DB file: `storage/data/phishguard.db` (gitignored).
Stop the server before deleting it to start clean.

**Switch to PostgreSQL (production):**
1. Install the driver for your Python: `pip install "psycopg[binary]"`.
2. Set `SQLALCHEMY_DATABASE_URI=postgresql+psycopg2://user:pass@host:5432/db`.
3. `python -m alembic upgrade head` once.

**Schema migrations (Alembic):**
```powershell
# 1) Create a migration after editing storage/models.py:
python -m alembic revision --autogenerate -m "describe the change"
# 2) Review migrations/versions/<new>.py, then apply:
python -m alembic upgrade head
# Rollback one step: python -m alembic downgrade -1
```
- All durable tables share the `(id TEXT PK, data TEXT JSON)` layout in `storage/models.py`.
- `storage/repository.py::SqlAlchemyStore` is the single write-through/load adapter the
  managers use; keep its interface stable when adding tables.

**Backups:** the SQLite file is the whole data set — back it up (or the Postgres DB +
object storage) on a schedule; test restoration.

### 15.4 Adding a new analyzer (safe registration)

Analyzers live in `engine/analyzers/`. Only offline-safe analyzers are registered by
default (in `api_gateway/main.py::create_app`). To add one:
1. Implement the analyzer in its module (mirror an existing one's interface).
2. Import + instantiate it and add it to the `for analyzer in ( ... )` registration tuple.
3. **Graceful degradation:** if it needs an external engine (browser, live DNS/SSL, YARA
   rulesets, threat-intel) that may be absent, guard its import/init in a try/except and
   `logger.warning(...)` + skip, so it can never fail the whole scan.
4. Run `python tests\integration\test_api.py`; confirm a phishing input reaches the
   desired verdict and a benign input stays low.

### 15.5 Adding a new API endpoint

In `api_gateway/main.py::create_app`:
1. Add the route with an RBAC dependency:
   `@app.post("/x", dependencies=[Depends(require_roles("admin", "analyst"))])`
   (reads allow `admin, analyst, viewer`).
2. Attach the audit hook: `_record_audit(request, "x.create", "x", id, {...})`.
3. If it's a long action, consider the async job pattern (`/scan/async` + `/scan/jobs/{id}`).
4. Add a coverage assertion to `tests/integration/test_api.py`.

### 15.6 Authentication & tenants

- Open mode → API-key mode → JWT+RBAC mode, resolved by `AuthGuardMiddleware`.
- Roles: `admin` (all), `analyst` (write ops), `viewer` (read-only). Enforced per route
  via `security/rbac.py::require_roles`.
- Tenant scope comes from the principal (JWT/API key) and is enforced by
  `TenantIsolationMiddleware`; data is always listed/scoped by `request.state.tenant_id`.
- Never hardcode secrets. In staging/production, staging a real `SECRET_KEY` is enforced
  at startup.

### 15.7 Async scans

`POST /scan/async` returns a durable job (`pending → running → completed | failed`).
Poll `GET /scan/jobs/{id}` or subscribe to SSE at `GET /scan/jobs/{id}/events`. Jobs
survive restarts (SQLAlchemy `scan_jobs` table). For scale-out across machines, move the
executor to Celery + Redis (scaffold exists in `messaging/`) and keep the same job table.

### 15.8 Audit trail

Every privileged action writes an immutable record (`user, action, target, timestamp,
IP`) via `storage/audit.py`; read it with `GET /audit`. There is deliberately no
update/delete path. Export it periodically for compliance.

### 15.9 Common troubleshooting

| Symptom | Cause / fix |
|---|---|
| `401 unauthorized` on `/scan` | Server in enforced mode; add a valid `X-API-Key` or `Authorization: Bearer <token>`. |
| `403 insufficient permissions` | Caller's role can't write; use an `analyst`/`admin` token. |
| DB file locked on delete | The server holds it; stop `run_server.ps1` first. |
| `alembic upgrade` no-op | Tables already exist via `init_db`; that's expected. |
| Port 8000 in use | Stop the other process or change `PORT`. |
| Threat-intel providers unreachable | Missing/expired provider keys; the pipeline skips them gracefully. |
| CPU-bound analyzer returns FAILED with a pickle error | Resolved: `ConstrainedMetadata` is now picklable so the ProcessPoolExecutor dispatch works. If a new CPU-bound analyzer fails, check it has no unpicklable members. |

### 15.10 Release checklist

1. `python -m compileall -q <source dirs>` — no errors.
2. `python tests\integration\test_api.py` — all checks pass.
3. If schema changed: `python -m alembic upgrade head` against a copy, then production.
4. Confirm open mode + a JWT path both work.
5. `git add -A; git commit -m "…"; git push origin main`.

---


