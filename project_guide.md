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

The auth guard (`api_gateway/middleware/auth_guard.py`) has two modes:

| Mode | When | Behavior |
|------|------|----------|
| **Open** (default for dev) | No `API_KEYS` configured | All endpoints accessible without a key |
| **Enforced** | `API_KEYS` set (comma-separated) | Non-exempt endpoints require `X-API-Key` or `Authorization: Bearer <key>` |

Public/exempt endpoints (always open): `/`, `/dashboard`, `/health`, `/ready`, `/metrics`, `/docs`, `/openapi.json`.

To enable auth for a deployment:

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

## 6. Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/unit -q
```

A `compileall` over the whole tree is a quick syntax gate:

```powershell
.\venv\Scripts\python.exe -m compileall -q .
```

---

## 7. Configuration

Settings live in `config/internal/settings.py` (pydantic-settings). Override via environment variables or a `.env` file (see `.env.example`).

Highlights: `API_KEYS`, `VIRUSTOTAL_API_KEY`, `ABUSEIPDB_API_KEY`, `URLHAUS_API_KEY`, SSRF protections (`ALLOW_*_ACCESS`), rate-limit bounds.

---

## 8. Security Notes

- **SSRF guard** (`security/ssrf_guard.py`): blocks private/link-local/loopback targets by default; DNS-based resolution may reject benchmark IPs (normal).
- **Encryption at rest** (`storage/encryption.py`): AES-GCM keyring. The quarantine manager auto-provisions an in-process active key so it works out of the box; bring your own key manager for persistence.
- **Error hygiene**: internal errors are never surfaced raw (sanitized 500s).

---

## 9. Known Limitations & Roadmap

- **Analyzers are not registered yet.** The engine lists analyzer names but no instances are wired up, so `/scan` currently returns `INFORMATIONAL` / score `0` (the pipeline runs end to end, just with no analyzers). Register the default analyzers in `DetectionOrchestrator` to produce real scores.
- In-memory stores for cases / quarantine: production deployments should swap in a database + persistent keyring.

---

## 10. Git Workflow

```powershell
git add -A
git commit -m "message"
git push origin main
```

Repo: `https://github.com/pro-rider/PhishGuard_AI`
