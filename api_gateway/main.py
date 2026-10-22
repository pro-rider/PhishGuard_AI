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

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from api_gateway.middleware.auth_guard import (
    AuthGuardMiddleware,
    build_identity_store,
)
from api_gateway.middleware.correlation_id import CorrelationIdMiddleware
from api_gateway.middleware.rate_limiter import RateLimiterMiddleware
from api_gateway.middleware.tenant_isolation import TenantIsolationMiddleware
from case_management.cases import CaseManager, CaseStatus, SecurityCase
from case_management.comments import CommentManager
from case_management.evidence import EvidenceManager, EvidenceKind
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
from observability.alerting import AlertManager, AlertSeverity, SecurityAlert
from quarantine.email_quarantine import (
    EmailQuarantineManager,
    QuarantineStatus,
)
from security.rbac import require_roles

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
    app = FastAPI(
        title="PhishGuard_AI API",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
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

    case_manager = CaseManager(db_path=DB_PATH)
    comment_manager = CommentManager()
    evidence_manager = EvidenceManager()
    quarantine_manager = EmailQuarantineManager(db_path=DB_PATH)
    alert_manager = AlertManager()

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
    async def ready() -> Dict[str, str]:
        return {"status": "ready"}

    # ------------------------------------------------------------------
    # Authentication (JWT token issuance)
    # ------------------------------------------------------------------

    @app.post("/auth/token", response_model=Dict[str, object])
    async def auth_token(payload: AuthTokenRequest) -> Dict[str, object]:
        """Exchange username/password for a JWT when AUTH_USERS is configured.

        Returns an access token carrying the caller's roles + tenant scope. In
        open mode (no AUTH_USERS) this returns 401 so callers fall back to the
        unauthenticated endpoints.
        """
        identity = build_identity_store()
        if identity is None:
            raise HTTPException(status_code=401, detail="auth not configured")
        principal = identity.authenticate(payload.username, payload.password)
        if principal is None:
            raise HTTPException(status_code=401, detail="invalid credentials")
        token = identity.issue_token(principal)
        ttl = getattr(identity, "_ttl", 1800)
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

        # End-to-end closure: escalate confirmed threats straight into the
        # incident workflow so operators see them on the dashboard without
        # manual triage. Idempotent per scan_id.
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
        }

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
    # Alerts (read-only summary)
    # ------------------------------------------------------------------

    @app.get(
        "/alerts/severities",
        response_model=Dict[str, str],
        dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
    )
    async def alert_severities() -> Dict[str, str]:
        return {s.value: s.value for s in AlertSeverity}

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
