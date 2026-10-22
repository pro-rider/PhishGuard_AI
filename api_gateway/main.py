"""
PhishGuard_AI - API Gateway

FastAPI entry point exposing the detection pipeline, case management, and
quarantine workflows over HTTP.

Security / reliability principles:
- Middleware enforces authentication, tenant isolation, correlation ids, and
  rate limits.
- Request payloads are validated and size-bounded.
- Internal errors are never surfaced raw (sanitized 500s).
- Health/ready/metrics are exempt from authentication.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTasks
from starlette.responses import Response, StreamingResponse

from api_gateway.middleware.auth_guard import (
    AuthGuardMiddleware,
    build_identity_store,
)
from api_gateway.middleware.correlation_id import CorrelationIdMiddleware
from api_gateway.middleware.rate_limiter import RateLimiterMiddleware
from api_gateway.middleware.tenant_isolation import TenantIsolationMiddleware
from case_management.cases import (
    CaseError,
    CaseManager,
    CaseNotFoundError,
    CaseStatus,
    CaseValidationError,
    SecurityCase,
)
from case_management.comments import (
    CommentError,
    CommentManager,
    CommentVisibility,
)
from case_management.evidence import (
    EvidenceError,
    EvidenceKind,
    EvidenceManager,
)
from config.internal.settings import settings
from engine.analyzers.email.nlp_detector import NLPDetectorAnalyzer
from engine.analyzers.heuristic_analyzer import HeuristicContentAnalyzer
from engine.orchestrator import (
    ConstrainedMetadata,
    DetectionOrchestrator,
    PayloadReference,
    PipelineResult,
    ScanContext,
    ScanType,
)
from observability.alerting import AlertSeverity
from observability.metrics import metrics
from quarantine.email_quarantine import (
    EmailQuarantineManager,
    InvalidTransitionError,
    MessageNotFoundError,
)
from security.rbac import require_roles
from storage.audit import AuditManager
from storage.repository import SqlAlchemyStore
from storage.scan_jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    ScanJobManager,
    ScanJobNotFoundError,
)

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_TEXT_LENGTH = 1_000_000
MAX_FILENAME_LENGTH = 255
MAX_TITLE_LENGTH = 256
MAX_DESCRIPTION_LENGTH = 16_384

# Operator dashboard: served as the landing page when the API runs.
FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
DASHBOARD_PATH = os.path.join(FRONTEND_DIR, "operator-dashboard.html")

# Durable persistence: cases & quarantine records are written through to a
# local SQLite database so they survive process restarts. Override the path
# with the PHISHGUARD_DB env var for custom deployments.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.environ.get(
    "PHISHGUARD_DB",
    os.path.join(PROJECT_ROOT, "storage", "data", "phishguard.db"),
)

# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================


class ScanRequest(BaseModel):
    """Body of a detection scan request."""

    model_config = ConfigDict(extra="forbid")

    target_url: Optional[str] = Field(default=None, max_length=8192)
    email_text: Optional[str] = Field(default=None, max_length=MAX_TEXT_LENGTH)
    filename: Optional[str] = Field(default=None, max_length=MAX_FILENAME_LENGTH)
    enable_sandbox: bool = False
    enable_yara: bool = True


class CaseCreateRequest(BaseModel):
    """Body for creating a new security case."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_LENGTH)
    severity: str = Field(default="medium", max_length=16)
    assignee: str = Field(default="", max_length=128)


class QuarantineRequest(BaseModel):
    """Body for quarantining a message."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1, max_length=128)
    body: str = Field(default="", max_length=MAX_TEXT_LENGTH)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CaseUpdateRequest(BaseModel):
    """Body for updating editable scalar fields on a case."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, min_length=1, max_length=256)
    description: Optional[str] = Field(default=None, max_length=16384)
    assignee: Optional[str] = Field(default=None, max_length=128)
    severity: Optional[str] = Field(default=None, max_length=16)


class CaseTransitionRequest(BaseModel):
    """Body for a validated case status transition."""

    model_config = ConfigDict(extra="forbid")

    new_status: str = Field(min_length=1, max_length=16)
    comment: Optional[str] = Field(default=None, max_length=8192)


class CommentCreateRequest(BaseModel):
    """Body for adding a comment to a case."""

    model_config = ConfigDict(extra="forbid")

    body: str = Field(min_length=1, max_length=8192)
    visibility: str = Field(default="internal", max_length=16)


class EvidenceAttachRequest(BaseModel):
    """Body for attaching an evidence reference to a case."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=16)
    source: str = Field(min_length=1, max_length=128)
    storage_key: str = Field(min_length=1, max_length=1024)
    metadata: Optional[Dict[str, str]] = Field(default=None)


class QuarantineActionRequest(BaseModel):
    """Body for a quarantine release/delete action."""

    model_config = ConfigDict(extra="forbid")

    reason: Optional[str] = Field(default=None, max_length=1024)


class AuthTokenRequest(BaseModel):
    """Body for exchanging credentials for a JWT (enabled via AUTH_USERS)."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


# ============================================================================
# APPLICATION FACTORY
# ============================================================================


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    # Fail-closed by default: interactive docs / OpenAPI schema are only
    # exposed when the operator explicitly opts into insecure dev. In a secure
    # deployment they are disabled (they leak internals / attack surface).
    dev_insecure = bool(getattr(settings, "DEV_INSECURE", False))
    app = FastAPI(
        title="PhishGuard_AI API",
        version="1.0.0",
        docs_url="/docs" if dev_insecure else None,
        openapi_url="/openapi.json" if dev_insecure else None,
    )

    # Middleware order matters: correlation -> tenant -> rate limit -> auth.
    app.add_middleware(AuthGuardMiddleware)
    app.add_middleware(RateLimiterMiddleware)
    app.add_middleware(TenantIsolationMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # CORS: allow the operator dashboard and local/dev origins to call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "*",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    orchestrator = DetectionOrchestrator()
    # Register the default analyzer set. Only offline-safe analyzers that produce
    # genuine signals from the plain-text email / URL input are wired in — analyzers
    # that depend on RFC822 headers, external storage, a browser, or live network
    # are intentionally excluded so the pipeline never fails closed spuriously.
    for analyzer in (
        NLPDetectorAnalyzer(),
        HeuristicContentAnalyzer(),
    ):
        orchestrator.register_analyzer(analyzer)

    # Feature-flagged extra analyzers (default OFF): structural analyzers like
    # url_parser add strong signals (homograph / SSRF / userinfo / high-entropy)
    # but change weighted aggregation, so enabling requires re-baselining verdicts.
    # Registration is guarded so a broken analyzer can never fail the scan.
    if getattr(settings, "ENABLE_EXTRA_ANALYZERS", False):
        try:
            from engine.analyzers.url.url_parser import URLParser

            orchestrator.register_analyzer(URLParser())
            logger.info("Extra analyzer registered: url_parser")
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning("Extra analyzer url_parser not registered: %s", exc)

    # Prometheus metric instruments (registered once, module-level).
    _setup_metrics()


    case_manager = CaseManager(db_path=DB_PATH)
    comment_manager = CommentManager(db_path=DB_PATH)
    evidence_manager = EvidenceManager(db_path=DB_PATH)
    quarantine_manager = EmailQuarantineManager(db_path=DB_PATH)

    # Durable backend for async scan jobs + the immutable audit trail.
    store = SqlAlchemyStore(DB_PATH)
    job_manager = ScanJobManager(store)
    audit_manager = AuditManager(store)

    # ------------------------------------------------------------------
    # Health & metrics
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "service": "phishguard-ai"}

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        """Serve the operator dashboard as the landing page.

        The interactive API explorer remains available at /docs.
        """
        return FileResponse(
            DASHBOARD_PATH, media_type="text/html"
        )

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> FileResponse:
        """Explicit alias for the operator dashboard."""
        return FileResponse(
            DASHBOARD_PATH, media_type="text/html"
        )

    @app.get("/ready")
    async def ready() -> Dict[str, object]:
        """Readiness probe that actually verifies the durable store responds."""
        db_ok = True
        try:
            store.upsert("scan_jobs", "__probe__", {"status": "probe"})
            db_ok = store.load("scan_jobs", "__probe__") is not None
        except Exception as exc:  # noqa: BLE001
            db_ok = False
            logger.warning("readiness probe: db check failed: %s", exc)
        return {"status": "ready" if db_ok else "degraded", "db": "ok" if db_ok else "error"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Prometheus text-format metrics for scraping (exempt from auth)."""
        return Response(
            metrics.render(), media_type="text/plain; version=0.0.4"
        )

    # ------------------------------------------------------------------
    # Audit helper (immutable trail; must never break the request)
    # ------------------------------------------------------------------

    def _record_audit(
        request: Request,
        action: str,
        target_type: str,
        target_id: str = "",
        detail: Any = None,
    ) -> None:
        try:
            principal = getattr(request.state, "principal", None)
            actor = principal.username if principal is not None else "anonymous"
            tenant = getattr(request.state, "tenant_id", "system")
            ip = request.client.host if request.client is not None else None
            audit_manager.record(
                tenant_id=tenant,
                actor=actor,
                action=action,
                target_type=target_type,
                target_id=target_id,
                ip=ip,
                detail=detail,
            )
        except Exception as exc:  # noqa: BLE001 - audit must not break the request
            logger.warning("audit record failed for %s: %s", action, exc)

    # ------------------------------------------------------------------
    # Authentication (JWT token issuance)
    # ------------------------------------------------------------------

    @app.post("/auth/token", response_model=Dict[str, object])
    async def auth_token(payload: AuthTokenRequest, request: Request) -> Dict[str, object]:
        """Exchange username/password for a JWT when AUTH_USERS is configured.

        Returns an access token carrying the caller's roles + tenant scope. In
        open mode (no AUTH_USERS) this returns 401 so callers fall back to the
        unauthenticated endpoints.
        """
        identity = build_identity_store()
        if identity is None:
            raise HTTPException(status_code=401, detail="auth not configured")
        ip = request.client.host if request.client is not None else "?"
        if _login_throttle.is_locked(payload.username, ip):
            raise HTTPException(status_code=429, detail="too many failed login attempts")
        principal = identity.authenticate(payload.username, payload.password)
        if principal is None:
            _login_throttle.record_failure(payload.username, ip)
            _record_audit(
                request,
                "auth.token.failed",
                "principal",
                payload.username,
                {"ip": ip},
            )
            raise HTTPException(status_code=401, detail="invalid credentials")
        _login_throttle.record_success(payload.username, ip)
        token = identity.issue_token(principal)
        ttl = getattr(identity, "_ttl", 1800)
        _record_audit(
            request,
            "auth.token",
            "principal",
            payload.username,
            {"roles": sorted(principal.roles), "tenant": principal.tenant},
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": ttl,
            "roles": sorted(principal.roles),
            "tenant": principal.tenant,
        }

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @app.post(
        "/scan",
        response_model=Dict[str, Any],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def scan(payload: ScanRequest, request: Request) -> Dict[str, Any]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        correlation_id = getattr(request.state, "correlation_id", None)
        result = await _run_detection(
            orchestrator, case_manager, payload, tenant_id, correlation_id
        )
        _record_audit(
            request,
            "scan.run",
            "scan",
            result["scan_id"],
            {"verdict": result["verdict"], "score": result["overall_score"]},
        )
        _scan_total(verdict=result.get("verdict", "unknown"))
        _scan_latency(value=max(result.get("total_execution_time_ms", 0) / 1000.0, 0.0))
        return result
    # ------------------------------------------------------------------
    # Async detection (long-running scans): job + status + SSE events
    # ------------------------------------------------------------------

    @app.post(
        "/scan/async",
        response_model=Dict[str, Any],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def scan_async(
        payload: ScanRequest,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> Dict[str, Any]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        correlation_id = getattr(request.state, "correlation_id", None)
        principal = getattr(request.state, "principal", None)
        actor = principal.username if principal is not None else "anonymous"
        job = job_manager.create(
            tenant_id=tenant_id,
            actor=actor,
            correlation_id=correlation_id,
            request_summary={
                "target_url": (payload.target_url or "")[:512],
                "email_len": len(payload.email_text or ""),
                "filename": payload.filename,
                "enable_sandbox": payload.enable_sandbox,
                "enable_yara": payload.enable_yara,
            },
        )
        background_tasks.add_task(
            _run_scan_job,
            job_manager=job_manager,
            orchestrator=orchestrator,
            case_manager=case_manager,
            audit_manager=audit_manager,
            job_id=job["job_id"],
            payload=payload,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            actor=actor,
            ip=(request.client.host if request.client is not None else None),
        )
        return {
            "job_id": job["job_id"],
            "status": STATUS_PENDING,
            "poll_url": f"/scan/jobs/{job['job_id']}",
            "events_url": f"/scan/jobs/{job['job_id']}/events",
        }

    @app.get(
        "/scan/jobs/{job_id}",
        response_model=Dict[str, Any],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def get_scan_job(job_id: str, request: Request) -> Dict[str, Any]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        try:
            job = job_manager.get(job_id)
        except ScanJobNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")
        # Tenant isolation: a caller must never read another tenant's scan.
        if (job or {}).get("tenant_id", "system") != tenant_id:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get(
        "/scan/jobs",
        response_model=List[Dict[str, Any]],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def list_scan_jobs(request: Request) -> List[Dict[str, Any]]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        return job_manager.list(tenant_id=tenant_id)

    @app.get(
        "/scan/jobs/{job_id}/events",
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def scan_job_events(job_id: str, request: Request) -> StreamingResponse:
        """Server-sent events stream for a scan job's progress + final result."""
        tenant_id = getattr(request.state, "tenant_id", "system")
        try:
            job = job_manager.get(job_id)
        except ScanJobNotFoundError:
            raise HTTPException(status_code=404, detail="job not found")
        # Tenant isolation for the stream, same as the job read.
        if (job or {}).get("tenant_id", "system") != tenant_id:
            raise HTTPException(status_code=404, detail="job not found")

        async def event_stream():
            def sse(event_type: str, data: Any) -> str:
                import json

                return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

            seen = 0
            last_status = None
            for _ in range(600):  # ~60s of polling before giving up
                job = job_manager.get(job_id)
                events = list(job.get("events") or [])
                for ev in events[seen:]:
                    seen += 1
                    yield sse(ev.get("type", "progress"), ev)
                status = job.get("status")
                if status != last_status:
                    last_status = status
                    yield sse("status", {"status": status})
                if status in {STATUS_COMPLETED, STATUS_FAILED}:
                    yield sse("result", job)
                    yield sse("done", {"job_id": job_id})
                    return
                await asyncio.sleep(0.1)
            yield sse("done", {"job_id": job_id, "timeout": True})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    @app.post(
        "/cases",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def create_case(payload: CaseCreateRequest, request: Request) -> Dict[str, object]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        case: SecurityCase = case_manager.create(
            tenant_id=tenant_id,
            title=payload.title,
            description=payload.description,
            severity=payload.severity,
            assignee=payload.assignee,
        )
        _cases_total(severity=str(case.severity.value if hasattr(case.severity, "value") else case.severity))
        _record_audit(
            request,
            "case.create",
            "case",
            case.case_id,
            {"title": payload.title, "severity": payload.severity},
        )
        return case.to_dict()

    @app.get(
        "/cases",
        response_model=List[Dict[str, object]],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def list_cases(
        request: Request, status: Optional[str] = None
    ) -> List[Dict[str, object]]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        status_enum = None
        if status:
            try:
                status_enum = CaseStatus(status)
            except ValueError:
                return []
        return [c.to_dict() for c in case_manager.list_for_tenant(tenant_id, status=status_enum)]

    # ------------------------------------------------------------------
    # Case workflow (update / transition / comments / evidence / timeline)
    # ------------------------------------------------------------------

    def _require_case_tenant(case_id: str, request: Request) -> SecurityCase:
        try:
            case = case_manager.get(case_id)
        except CaseNotFoundError:
            raise HTTPException(status_code=404, detail="case not found")
        if case.tenant_id != getattr(request.state, "tenant_id", "system"):
            raise HTTPException(status_code=404, detail="case not found")
        return case

    def _actor(request: Request) -> str:
        principal = getattr(request.state, "principal", None)
        return principal.username if principal is not None else "anonymous"

    @app.get(
        "/cases/{case_id}",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def get_case(case_id: str, request: Request) -> Dict[str, object]:
        return _require_case_tenant(case_id, request).to_dict()

    @app.patch(
        "/cases/{case_id}",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def update_case(
        case_id: str, payload: CaseUpdateRequest, request: Request
    ) -> Dict[str, object]:
        _require_case_tenant(case_id, request)
        try:
            case = case_manager.update(
                case_id,
                title=payload.title,
                description=payload.description,
                assignee=payload.assignee,
                severity=payload.severity,
            )
        except CaseValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        _record_audit(
            request, "case.update", "case", case_id, payload.model_dump()
        )
        return case.to_dict()

    @app.post(
        "/cases/{case_id}/transition",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def transition_case(
        case_id: str, payload: CaseTransitionRequest, request: Request
    ) -> Dict[str, object]:
        case = _require_case_tenant(case_id, request)
        try:
            new_status = CaseStatus(payload.new_status)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid status")
        previous = case.status.value
        try:
            updated = case_manager.transition(case_id, new_status)
        except CaseError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        if payload.comment:
            comment_manager.add(
                case_id=case_id, author=_actor(request), body=payload.comment
            )
        _record_audit(
            request,
            "case.transition",
            "case",
            case_id,
            {"from": previous, "to": new_status.value},
        )
        return updated.to_dict()

    @app.post(
        "/cases/{case_id}/comments",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def add_comment(
        case_id: str, payload: CommentCreateRequest, request: Request
    ) -> Dict[str, object]:
        _require_case_tenant(case_id, request)
        visibility = (
            CommentVisibility(payload.visibility)
            if payload.visibility in {"internal", "customer"}
            else CommentVisibility.INTERNAL
        )
        try:
            comment = comment_manager.add(
                case_id=case_id,
                author=_actor(request),
                body=payload.body,
                visibility=visibility,
            )
        except CommentError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        _record_audit(
            request,
            "case.comment.add",
            "case",
            case_id,
            {"comment_id": comment.comment_id, "visibility": visibility.value},
        )
        return comment.to_dict()

    @app.get(
        "/cases/{case_id}/comments",
        response_model=List[Dict[str, object]],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def list_comments(
        case_id: str, request: Request
    ) -> List[Dict[str, object]]:
        _require_case_tenant(case_id, request)
        return [c.to_dict() for c in comment_manager.list_for_case(case_id)]

    @app.post(
        "/cases/{case_id}/evidence",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def attach_evidence(
        case_id: str, payload: EvidenceAttachRequest, request: Request
    ) -> Dict[str, object]:
        _require_case_tenant(case_id, request)
        try:
            kind = EvidenceKind(payload.kind)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid evidence kind")
        try:
            record = evidence_manager.attach(
                case_id=case_id,
                kind=kind,
                source=payload.source,
                storage_key=payload.storage_key,
                metadata=payload.metadata,
            )
        except EvidenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        _record_audit(
            request,
            "case.evidence.attach",
            "case",
            case_id,
            {"evidence_id": record.evidence_id, "kind": kind.value},
        )
        return record.to_dict()

    @app.get(
        "/cases/{case_id}/evidence",
        response_model=List[Dict[str, object]],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def list_evidence(
        case_id: str, request: Request
    ) -> List[Dict[str, object]]:
        _require_case_tenant(case_id, request)
        return [e.to_dict() for e in evidence_manager.list_for_case(case_id)]

    @app.get(
        "/cases/{case_id}/timeline",
        response_model=List[Dict[str, object]],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def case_timeline(
        case_id: str, request: Request
    ) -> List[Dict[str, object]]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        _require_case_tenant(case_id, request)
        events: List[Dict[str, object]] = []
        for row in audit_manager.list(tenant_id=tenant_id, limit=1000):
            if row.get("target_id") == case_id:
                events.append(
                    {
                        "ts": row.get("timestamp_utc", ""),
                        "type": "audit",
                        "actor": row.get("actor", ""),
                        "action": row.get("action", ""),
                        "detail": row.get("detail", ""),
                    }
                )
        for c in comment_manager.list_for_case(case_id):
            events.append(
                {
                    "ts": c.created_utc,
                    "type": "comment",
                    "actor": c.author,
                    "body": c.body,
                    "visibility": c.visibility.value,
                }
            )
        for e in evidence_manager.list_for_case(case_id):
            events.append(
                {
                    "ts": e.created_utc,
                    "type": "evidence",
                    "kind": e.kind.value,
                    "source": e.source,
                }
            )
        events.sort(key=lambda ev: str(ev.get("ts", "")))
        return events

    # ------------------------------------------------------------------
    # Quarantine
    # ------------------------------------------------------------------

    @app.post(
        "/quarantine",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def quarantine_message(
        payload: QuarantineRequest, request: Request
    ) -> Dict[str, object]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        record = quarantine_manager.quarantine(
            message_id=payload.message_id,
            tenant_id=tenant_id,
            raw_body=payload.body.encode("utf-8"),
            risk_score=payload.risk_score,
        )
        _quarantine_total()
        _record_audit(
            request,
            "quarantine.create",
            "quarantine",
            record.quarantine_id,
            {"message_id": payload.message_id, "risk_score": payload.risk_score},
        )
        return record.to_dict()

    @app.get(
        "/quarantine",
        response_model=List[Dict[str, object]],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def list_quarantine(request: Request) -> List[Dict[str, object]]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        return [
            r.to_dict()
            for r in quarantine_manager.list_for_tenant(tenant_id)
        ]

    # ------------------------------------------------------------------
    # Quarantine workflow (get / release / delete / preview)
    # ------------------------------------------------------------------

    def _require_quarantine_tenant(qid: str, request: Request):
        try:
            record = quarantine_manager.get(qid)
        except MessageNotFoundError:
            raise HTTPException(status_code=404, detail="quarantine item not found")
        if record.tenant_id != getattr(request.state, "tenant_id", "system"):
            raise HTTPException(status_code=404, detail="quarantine item not found")
        return record

    @app.get(
        "/quarantine/{qid}",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def get_quarantine(qid: str, request: Request) -> Dict[str, object]:
        return _require_quarantine_tenant(qid, request).to_dict()

    @app.post(
        "/quarantine/{qid}/release",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def release_quarantine(
        qid: str, payload: QuarantineActionRequest, request: Request
    ) -> Dict[str, object]:
        _require_quarantine_tenant(qid, request)
        try:
            record = quarantine_manager.release(qid)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        _record_audit(
            request,
            "quarantine.release",
            "quarantine",
            qid,
            {"reason": payload.reason},
        )
        return record.to_dict()

    @app.post(
        "/quarantine/{qid}/delete",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst"))],
    )
    async def delete_quarantine(
        qid: str, payload: QuarantineActionRequest, request: Request
    ) -> Dict[str, object]:
        _require_quarantine_tenant(qid, request)
        try:
            record = quarantine_manager.delete(qid)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        _record_audit(
            request,
            "quarantine.delete",
            "quarantine",
            qid,
            {"reason": payload.reason},
        )
        return record.to_dict()

    @app.get(
        "/quarantine/{qid}/preview",
        response_model=Dict[str, object],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def preview_quarantine(
        qid: str, request: Request
    ) -> Dict[str, object]:
        _require_quarantine_tenant(qid, request)
        body = quarantine_manager.read_body(qid)
        text = body.decode("utf-8", errors="replace")
        _record_audit(
            request,
            "quarantine.preview",
            "quarantine",
            qid,
            {"bytes": len(body)},
        )
        return {
            "quarantine_id": qid,
            "body": text,
            "decoded": True,
            "note": "Sensitive content read; access is logged.",
        }

    # ------------------------------------------------------------------
    # Alerts (read-only summary)
    # ------------------------------------------------------------------

    @app.get(
        "/alerts/severities",
        response_model=Dict[str, str],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def alert_severities() -> Dict[str, str]:
        return {s.value: s.value for s in AlertSeverity}

    # ------------------------------------------------------------------
    # Audit trail (immutable, read-only)
    # ------------------------------------------------------------------

    @app.get(
        "/audit",
        response_model=List[Dict[str, Any]],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def audit_log(request: Request) -> List[Dict[str, Any]]:
        tenant_id = getattr(request.state, "tenant_id", "system")
        return audit_manager.list(tenant_id=tenant_id)

    return app


def _build_email_payload(
    email_text: Optional[str], filename: Optional[str]
) -> Optional[PayloadReference]:
    """Build a validated email payload reference from request text."""
    if not email_text:
        return None
    body = email_text.encode("utf-8")
    return PayloadReference(
        reference_id=f"ref_{uuid.uuid4().hex[:16]}",
        sha256_hash=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
        mime_type="text/plain",
        filename=filename,
    )


def _build_signals(result: "PipelineResult") -> List[Dict[str, object]]:
    """Derive the top contributing signals from per-analyzer results (explainability).

    Returns a short, human-readable list of what most drove the verdict, so an
    analyst can understand *why* a scan was flagged without reading every rule.
    """
    signals: List[Dict[str, object]] = []
    for name, ar in (result.analyzer_results or {}).items():
        score = getattr(ar, "score_contribution", 0.0) or 0.0
        if score <= 0:
            continue
        confidence = getattr(ar, "confidence", 0.0) or 0.0
        findings = getattr(ar, "findings", None) or []
        top_findings: List[Dict[str, str]] = []
        for f in list(findings)[:3]:
            sev = getattr(f, "severity", None)
            sev = sev.value if hasattr(sev, "value") else (sev or "unknown")
            top_findings.append(
                {
                    "rule_id": getattr(f, "rule_id", "") or "",
                    "severity": str(sev),
                    "message": (getattr(f, "message", "") or "")[:200],
                }
            )
        signals.append(
            {
                "analyzer": name,
                "score": round(float(score), 2),
                "confidence": round(float(confidence), 3),
                "findings": top_findings,
            }
        )
    signals.sort(key=lambda s: float(s["score"]), reverse=True)
    return signals[:5]


async def _run_detection(    orchestrator: DetectionOrchestrator,
    case_manager: CaseManager,
    payload: ScanRequest,
    tenant_id: str,
    correlation_id: Optional[str],
) -> Dict[str, Any]:
    """Execute one scan and return the result payload (shared sync + async)."""
    context = ScanContext(
        scan_id=f"scan_{uuid.uuid4().hex[:12]}",
        scan_type=ScanType.COMPREHENSIVE,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        target_url=payload.target_url,
        email_payload=_build_email_payload(payload.email_text, payload.filename),
        enable_sandbox=payload.enable_sandbox,
        enable_yara=payload.enable_yara,
        metadata=ConstrainedMetadata(
            {
                "api_gateway": "true",
                "email_text": (payload.email_text or "")[:1024],
            }
        ),
    )
    result: PipelineResult = await orchestrator.run_scan(context)

    # End-to-end closure: escalate confirmed threats straight into the incident
    # workflow so operators see them on the dashboard without manual triage.
    # Idempotent per scan_id.
    if result.is_malicious:
        _auto_create_case(case_manager, tenant_id, result)

    return {
        "scan_id": result.scan_id,
        "scan_type": result.scan_type.value,
        "execution_state": result.execution_state.value,
        "total_execution_time_ms": result.total_execution_time_ms,
        "overall_score": result.overall_score,
        "confidence_score": result.confidence_score,
        "verdict": result.verdict.value,
        "case_created": result.is_malicious,
        "signals": _build_signals(result),
    }


# ============================================================================
# LOGIN BRUTE-FORCE THROTTLE
# ============================================================================

_LOGIN_MAX_FAILURES = 5
_LOGIN_WINDOW_SECONDS = 900      # failures counted within this rolling window
_LOGIN_LOCKOUT_SECONDS = 900     # how long an account/IP stays locked


class _LoginThrottle:
    """In-memory failed-login throttle (per username + per client IP).

    Fail-closed and fail-open safe: unknown callers simply accumulate counts;
    a caller over the threshold is rejected with 429 until the lockout elapses.
    State is per-process — move to Redis for a multi-worker deployment.
    """

    def __init__(self) -> None:
        self._failures: dict = {}
        self._locks: dict = {}

    def _key(self, username: str, ip: str) -> str:
        return f"{username}|{ip}"

    def record_failure(self, username: str, ip: str) -> None:
        now = time.time()
        key = self._key(username, ip)
        bucket = self._failures.setdefault(key, [])
        bucket.append(now)
        # prune stale entries
        self._failures[key] = [t for t in bucket if now - t < _LOGIN_WINDOW_SECONDS]
        if len(self._failures[key]) >= _LOGIN_MAX_FAILURES:
            self._locks[key] = now + _LOGIN_LOCKOUT_SECONDS

    def record_success(self, username: str, ip: str) -> None:
        key = self._key(username, ip)
        self._failures.pop(key, None)
        self._locks.pop(key, None)

    def is_locked(self, username: str, ip: str) -> bool:
        key = self._key(username, ip)
        until = self._locks.get(key, 0)
        if until > time.time():
            return True
        if until:
            self._locks.pop(key, None)
        return False


_login_throttle = _LoginThrottle()


# ============================================================================
# PROMETHEUS INSTRUMENTATION (module-level so create_app and job executors share)
# ============================================================================

_METRICS_READY = False
_scan_total = None
_scan_latency = None
_scan_jobs_active = None
_cases_total = None
_quarantine_total = None


def _setup_metrics() -> None:
    """Register Prometheus instruments once (idempotent across app factory calls)."""
    global _METRICS_READY, _scan_total, _scan_latency
    global _scan_jobs_active, _cases_total, _quarantine_total
    if _METRICS_READY:
        return
    _scan_total = metrics.counter("scan_total", "Scans completed by verdict", ("verdict",))
    _scan_latency = metrics.histogram(
        "scan_latency_seconds",
        "Scan execution time (seconds)",
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0),
    )
    _scan_jobs_active = metrics.gauge("scan_jobs_active", "Async scan jobs currently running")
    _cases_total = metrics.counter("cases_total", "Cases created", ("severity",))
    _quarantine_total = metrics.counter("quarantine_total", "Messages quarantined")
    _METRICS_READY = True


def _refresh_active_jobs(job_manager: ScanJobManager) -> None:
    """Recompute the running-jobs gauge from durable job state."""
    try:
        running = sum(
            1
            for j in (job_manager.list() or [])
            if (j or {}).get("status") == "running"
        )
        if _scan_jobs_active is not None:
            _scan_jobs_active(value=float(running))
    except Exception:  # noqa: BLE001 - instrumentation is fail-open
        pass


async def _run_scan_job(    *,
    job_manager: ScanJobManager,
    orchestrator: DetectionOrchestrator,
    case_manager: CaseManager,
    audit_manager: AuditManager,
    job_id: str,
    payload: ScanRequest,
    tenant_id: str,
    correlation_id: Optional[str],
    actor: str,
    ip: Optional[str],
) -> None:
    """Background executor for an async scan job (queued via BackgroundTasks)."""
    try:
        job_manager.start(job_id)
        _refresh_active_jobs(job_manager)
        job_manager.append_event(job_id, "started", "scan started")
        result = await _run_detection(
            orchestrator, case_manager, payload, tenant_id, correlation_id
        )
        job_manager.complete(job_id, result)
        _refresh_active_jobs(job_manager)
        job_manager.append_event(
            job_id, "completed", f"verdict={result['verdict']}"
        )
        _scan_total(verdict=result.get("verdict", "unknown"))
        _scan_latency(value=max(result.get("total_execution_time_ms", 0) / 1000.0, 0.0))
        audit_manager.record(
            tenant_id=tenant_id,
            actor=actor,
            action="scan.async",
            target_type="scan_job",
            target_id=job_id,
            ip=ip,
            detail={"verdict": result["verdict"], "score": result["overall_score"]},
        )
    except Exception as exc:  # noqa: BLE001 - mark the job failed, keep the API alive
        logger.exception("async scan job %s failed", job_id)
        job_manager.fail(job_id, str(exc))
        _refresh_active_jobs(job_manager)
        job_manager.append_event(job_id, "failed", str(exc)[:4000])
        try:
            audit_manager.record(
                tenant_id=tenant_id,
                actor=actor,
                action="scan.async.failed",
                target_type="scan_job",
                target_id=job_id,
                ip=ip,
                detail={"error": str(exc)[:1000]},
            )
        except Exception:  # noqa: BLE001
            pass


def _auto_create_case(
    case_manager: CaseManager, tenant_id: str, result: PipelineResult
) -> None:
    """Escalate a confirmed HIGH/CRITICAL scan into an incident case (idempotent)."""
    scan_id = result.scan_id

    # Idempotency guard: never create two cases for the same scan.
    for existing in case_manager.list_for_tenant(tenant_id):
        if existing.custom_fields.get("scan_id") == scan_id:
            return

    title = f"Auto-detected {result.verdict.value} threat"
    description = (
        f"Created automatically from scan {scan_id}. "
        f"Overall score {result.overall_score}, confidence {result.confidence_score}. "
        f"Analyzers: {', '.join(sorted(result.analyzer_results.keys()))}."
    )
    try:
        case_manager.create(
            tenant_id=tenant_id,
            title=title,
            description=description,
            severity=result.verdict.value.lower(),
            assignee="system",
            tags=["auto", "scan"],
            custom_fields={
                "scan_id": scan_id,
                "verdict": result.verdict.value,
                "score": str(result.overall_score),
            },
        )
        logger.info(
            "Auto-created case for malicious scan %s (verdict=%s)",
            scan_id,
            result.verdict.value,
        )
    except Exception as exc:  # noqa: BLE001 - case creation must not break the scan
        logger.warning("Failed to auto-create case for scan %s: %s", scan_id, exc)


# ============================================================================
# MODULE SINGLETON
# ============================================================================

app = create_app()


__all__ = [
    "app",
    "create_app",
    "ScanRequest",
    "CaseCreateRequest",
    "QuarantineRequest",
]
