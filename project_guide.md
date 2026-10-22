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

| Mode | When | Behavior |
|------|------|----------|
| **Open** (default for dev) | No `API_KEYS`, no `AUTH_USERS` | All endpoints accessible; no principal attached |
| **API-key** | `API_KEYS` set | Non-exempt endpoints require `X-API-Key` (or `Authorization: Bearer <key>`). Plain keys grant `analyst`; use `key\|role1,role2\|tenant` for finer control |
| **JWT + RBAC** | `AUTH_USERS` set | `POST /auth/token` exchanges credentials for an HS256 JWT carrying roles + tenant; protected routes enforce roles |

Public/exempt endpoints (always open): `/`, `/dashboard`, `/health`, `/ready`, `/metrics`, `/docs`, `/openapi.json`, `/auth/token`.

### Role matrix

| Route | admin | analyst | viewer |
|-------|:-----:|:-------:|:------:|
| `POST /scan` | ✓ | ✓ | — |
| `POST /cases` | ✓ | ✓ | — |
| `GET /cases` | ✓ | ✓ | ✓ |
| `POST /quarantine` | ✓ | ✓ | — |
| `GET /quarantine` | ✓ | ✓ | ✓ |
| `GET /alerts/severities` | ✓ | ✓ | ✓ |

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
├── security/             # SSRF guard, etc.
├── storage/              # Encryption (AES-GCM keyring), object storage, redis client
├── observability/        # Alerting, metrics, tracing
├── messaging/            # Celery app
├── reputation/           # Threat-intel reputation (ip, hash)
├── sandbox/              # Attachment sandboxing & malware analysis
├── ml_inference/         # ML feature extractors / inference
├── mlops_pipeline/       # Training, evaluation, ONNX export, DVC
├── core/                 # Bootstrap & root config
├── frontend/             # Operator dashboard (single-file HTML)
├── config/internal/      # Hardened settings (pydantic-settings)
├── tests/                # Unit tests
├── docker/               # API / sandbox / worker Dockerfiles + compose
├── run_server.ps1        # Dev launcher (open mode)
└── requirements*.txt
```

---

## 4. Operator Dashboard

`frontend/operator-dashboard.html` — a single-file, self-contained dashboard served by the API at **`/`**.

Views: **Overview** (KPIs + case-severity chart), **Scan Console** (run a scan → verdict gauge), **Cases**, **Quarantine Queue**, **Alert Severities**.

It talks to the live API: when served by the server it auto-uses the same origin; when opened as a local file it defaults to `http://127.0.0.1:8000`. Charts use ECharts (loaded via jsDelivr).

---

## 5. API Reference

### `POST /scan`
Run a detection scan.

```json
{
  "target_url": "https://example.com/login",
  "email_text": "From: ...\nSubject: ...",
  "filename": "invoice.pdf",
  "enable_sandbox": false,
  "enable_yara": true
}
```

Response:

```json
{
  "scan_id": "scan_...",
  "scan_type": "comprehensive",
  "execution_state": "completed",
  "total_execution_time_ms": 0.01,
  "overall_score": 0.0,
  "confidence_score": 1.0,
  "verdict": "INFORMATIONAL"
}
```

`verdict` ∈ `INFORMATIONAL | LOW | MEDIUM | HIGH | CRITICAL`.

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

### `POST /cases`
Create an incident case. Body: `{ "title", "description", "severity", "assignee" }`.

### `GET /cases?status=<status>`
List cases. `status` ∈ `open | in_progress | on_hold | resolved | closed | archived`.

### `POST /quarantine`
Quarantine a message. Body: `{ "message_id", "body", "risk_score" }`.

### `GET /quarantine`
List the quarantine queue.

### `GET /alerts/severities`
Return the recognized severity scale.

---

## 6. Persistence

- **Default:** SQLite via SQLAlchemy at `storage/data/phishguard.db` (gitignored). Set `PHISHGUARD_DB` to relocate.
- **PostgreSQL:** set `SQLALCHEMY_DATABASE_URI=postgresql+psycopg2://user:pass@host/db` (add the driver) — the app and migrations automatically target it.
- **Migrations:** `python -m alembic upgrade head` (see `alembic.ini` / `migrations/`). Existing databases stay compatible (same `id`/`data` table layout).

## 7. Tests

```powershell
python tests/integration/test_api.py
```

Covers readiness, JWT+RBAC, sync scan, async job + SSE, cases, and the audit trail against a throwaway SQLite DB.

---

## 8. Unit Tests & Syntax Check

```powershell
.\venv\Scripts\python.exe -m pytest tests/unit -q
```

A `compileall` over the whole tree is a quick syntax gate:

```powershell
.\venv\Scripts\python.exe -m compileall -q .
```

---

## 9. Configuration

Settings live in `config/internal/settings.py` (pydantic-settings). Override via environment variables or a `.env` file (see `.env.example`).

Highlights: `API_KEYS`, `VIRUSTOTAL_API_KEY`, `ABUSEIPDB_API_KEY`, `URLHAUS_API_KEY`, SSRF protections (`ALLOW_*_ACCESS`), rate-limit bounds.

---

## 10. Security Notes

- **SSRF guard** (`security/ssrf_guard.py`): blocks private/link-local/loopback targets by default; DNS-based resolution may reject benchmark IPs (normal).
- **Encryption at rest** (`storage/encryption.py`): AES-GCM keyring. The quarantine manager auto-provisions an in-process active key so it works out of the box; bring your own key manager for persistence.
- **Error hygiene**: internal errors are never surfaced raw (sanitized 500s).

---

## 11. Known Limitations & Roadmap

- **Analyzers are now registered (default set).** `api_gateway/main.py` registers `NLPDetectorAnalyzer` + `HeuristicContentAnalyzer` (offline-safe, operate on the plain-text email / URL the gateway receives). `/scan` now returns real verdicts (HIGH/CRITICAL reachable), and a HIGH/CRITICAL scan **auto-creates a case** (idempotent per `scan_id`). Verified end-to-end: phishing email → `CRITICAL` + case; benign email → `INFORMATIONAL`, no case.
- **Bundled analyzers now import.** Fixed missing enum members (`AnalyzerResourceClass.COMPUTE` / `NETWORK_IO`, `ScanType.DOMAIN`) — all previously-crashing modules load. They are **not yet registered** because they need external engines (browser, live DNS/SSL, yara rulesets) that would risk fail-closed verdicts. See `docs/improvement_plan.md`.
- Email analyzers are fed only the first 1024 chars of `email_text` (the `ConstrainedMetadata` value cap); the composite heuristic uses bounded heuristics on that snippet.
- **Persistence:** cases + quarantine + the quarantine encryption key survive restarts via SQLAlchemy (SQLite default; PostgreSQL via `SQLALCHEMY_DATABASE_URI` + Alembic migrations). Key management still dev-grade (persisted key) — see roadmap for Vault/KMS.

---

## 12. Git Workflow

```powershell
git add -A
git commit -m "message"
git push origin main
```

Repo: `https://github.com/pro-rider/PhishGuard_AI`
