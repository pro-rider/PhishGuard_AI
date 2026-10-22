"""
PhishGuard_AI - Tenant Policy Store

Per-tenant policy resolution with allowlists, blocklists, and threshold
overrides layered on top of the global policy engine.

Security / reliability principles:
- Tenant identifiers are validated and bounded.
- Allow/block lists are size-bounded; entries are normalized for matching.
- Unknown tenants fall back to the global defaults (never more permissive).
- Lookups are exact and fail-safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from policy.policy_engine import PolicyConfig, PolicyDecision, PolicyEngine

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_TENANT_ID_LENGTH = 128
MAX_LIST_ENTRY_LENGTH = 1024
MAX_LIST_ENTRIES = 100_000
_TENANT_ID_REGEX = __import__("re").compile(r"^[A-Za-z0-9_\-.]{1,128}$")


class TenantPolicyError(RuntimeError):
    """Base error for tenant policy operations."""


# ============================================================================
# MODELS
# ============================================================================


class TenantPolicy(BaseModel):
    """Policy overrides and lists for a single tenant."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=MAX_TENANT_ID_LENGTH)
    config: Optional[PolicyConfig] = None
    allowlist_domains: List[str] = Field(default_factory=list)
    blocklist_domains: List[str] = Field(default_factory=list)
    blocklist_hashes: List[str] = Field(default_factory=list)
    allowlist_emails: List[str] = Field(default_factory=list)


# ============================================================================
# STORE
# ============================================================================


class TenantPolicyStore:
    """
    Thread-safe store of per-tenant policies.

    A shared :class:`PolicyEngine` applies global thresholds; tenant config
    overrides replace them when present. List membership is O(1) after
    normalization.
    """

    def __init__(self, *, global_engine: Optional[PolicyEngine] = None) -> None:
        self._global_engine = global_engine or PolicyEngine()
        self._lock = threading.RLock()
        self._policies: Dict[str, TenantPolicy] = {}
        self._allow_domains: Dict[str, FrozenSet[str]] = {}
        self._block_domains: Dict[str, FrozenSet[str]] = {}
        self._block_hashes: Dict[str, FrozenSet[str]] = {}
        self._allow_emails: Dict[str, FrozenSet[str]] = {}

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tenant(tenant_id: str) -> str:
        if not isinstance(tenant_id, str) or not _TENANT_ID_REGEX.match(tenant_id):
            raise TenantPolicyError(f"Invalid tenant id: {tenant_id!r}")
        return tenant_id

    @staticmethod
    def _normalize_list(values: List[str], *, limit: int = MAX_LIST_ENTRIES) -> FrozenSet[str]:
        normalized: Set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value:
                continue
            if len(value) > MAX_LIST_ENTRY_LENGTH:
                value = value[:MAX_LIST_ENTRY_LENGTH]
            normalized.add(value.lower())
            if len(normalized) >= limit:
                break
        return frozenset(normalized)

    def upsert(self, policy: TenantPolicy) -> None:
        tenant = self._validate_tenant(policy.tenant_id)
        with self._lock:
            self._policies[tenant] = policy
            self._allow_domains[tenant] = self._normalize_list(policy.allowlist_domains)
            self._block_domains[tenant] = self._normalize_list(policy.blocklist_domains)
            self._block_hashes[tenant] = self._normalize_list(policy.blocklist_hashes)
            self._allow_emails[tenant] = self._normalize_list(policy.allowlist_emails)

    def remove(self, tenant_id: str) -> bool:
        tenant = self._validate_tenant(tenant_id)
        with self._lock:
            for table in (
                self._policies,
                self._allow_domains,
                self._block_domains,
                self._block_hashes,
                self._allow_emails,
            ):
                table.pop(tenant, None)
        return True

    def clear(self) -> None:
        with self._lock:
            self._policies.clear()
            self._allow_domains.clear()
            self._block_domains.clear()
            self._block_hashes.clear()
            self._allow_emails.clear()

    def tenant_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._policies)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tenant_id: str,
        *,
        risk_score: float,
        sender_domain: str = "",
        sender_email: str = "",
        artifact_hashes: Optional[List[str]] = None,
    ) -> PolicyDecision:
        """Evaluate a risk assessment for a tenant, honoring its lists/overrides."""
        tenant = self._validate_tenant(tenant_id)
        engine = self._engine_for(tenant)
        decision = engine.evaluate(risk_score=risk_score)

        # Tenant blocklists override risk-based decisions.
        domain = (sender_domain or "").lower()
        email = (sender_email or "").lower()

        with self._lock:
            if domain and domain in self._block_domains.get(tenant, ()):
                decision = self._forced_block(risk_score, "tenant domain blocklist")
            if email and email in self._allow_emails.get(tenant, ()):
                decision = self._forced_allow(risk_score, "tenant email allowlist")
            if domain and domain in self._allow_domains.get(tenant, ()):
                decision = self._forced_allow(risk_score, "tenant domain allowlist")
            if artifact_hashes:
                blocked = self._block_hashes.get(tenant, ())
                if any(h.lower() in blocked for h in artifact_hashes if h):
                    decision = self._forced_block(risk_score, "tenant hash blocklist")

        return decision

    def is_domain_blocked(self, tenant_id: str, domain: str) -> bool:
        tenant = self._validate_tenant(tenant_id)
        with self._lock:
            return (domain or "").lower() in self._block_domains.get(tenant, ())

    def is_domain_allowed(self, tenant_id: str, domain: str) -> bool:
        tenant = self._validate_tenant(tenant_id)
        with self._lock:
            return (domain or "").lower() in self._allow_domains.get(tenant, ())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _engine_for(self, tenant_id: str) -> PolicyEngine:
        with self._lock:
            policy = self._policies.get(tenant_id)
        if policy is not None and policy.config is not None:
            return PolicyEngine(config=policy.config)
        return self._global_engine

    @staticmethod
    def _forced_block(risk_score: float, reason: str) -> PolicyDecision:
        return PolicyDecision(
            disposition="block",
            action="reject",
            reason=reason,
            risk_score=risk_score,
            matched_policy="tenant-blocklist",
        )

    @staticmethod
    def _forced_allow(risk_score: float, reason: str) -> PolicyDecision:
        return PolicyDecision(
            disposition="allow",
            action="deliver",
            reason=reason,
            risk_score=risk_score,
            matched_policy="tenant-allowlist",
        )


# ============================================================================
# SINGLETON
# ============================================================================

tenant_policy_store = TenantPolicyStore()


__all__ = [
    "TenantPolicy",
    "TenantPolicyStore",
    "TenantPolicyError",
    "tenant_policy_store",
    "MAX_LIST_ENTRIES",
]
