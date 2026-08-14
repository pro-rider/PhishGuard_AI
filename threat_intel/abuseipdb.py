"""
PhishGuard AI - AbuseIPDB API v2 Threat Intelligence Provider
==============================================================

Version: 14.0.0
File Path: threat_intel/abuseipdb.py

Production-grade asynchronous AbuseIPDB API v2 integration.

Responsibilities
----------------
* IP reputation enrichment
* Abuse confidence score normalization
* Abuse category normalization
* Rate-limit awareness
* IPv4/IPv6 validation and canonicalization
* Defensive provider-response parsing
* Timeout and transport error handling
* Provider health checking
* Safe HTTP client lifecycle management
* Deterministic normalized threat reports

Security principles
-------------------
* Never place API keys in URLs.
* Never log API keys.
* Never treat unsupported IOC types as CLEAN.
* Never trust provider JSON blindly.
* Never allow unbounded provider-controlled strings into normalized output.
* Never leak transient HTTP clients.
* Never query private/reserved addresses accidentally unless explicitly allowed.

The implementation follows AbuseIPDB API v2 semantics documented at:
https://docs.abuseipdb.com/
"""

from __future__ import annotations

import ipaddress
import time
import uuid
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set

import httpx

from threat_intel.base import (
    BaseThreatProvider,
    IOCType,
    NormalizedThreatReport,
    ProviderTimeoutError,
    RateLimitExceededError,
    ThreatIntelAPIError,
    ThreatSeverity,
)


# ==============================================================================
# PROVIDER CONSTANTS
# ==============================================================================

class AbuseIPDBProvider(BaseThreatProvider):
    """
    Production AbuseIPDB API v2 provider.

    The provider currently supports only IP IOC enrichment.

    Notes
    -----
    AbuseIPDB's `abuseConfidenceScore` represents reputation based on abuse
    reports. It is NOT an antivirus-engine detection count.

    Therefore:
        positive_detections -> number of abuse reports
        total_engines       -> 0

    Additional AbuseIPDB-specific counters are preserved in raw_response and
    normalized tags.
    """

    BASE_URL = "https://api.abuseipdb.com/api/v2"

    CHECK_ENDPOINT = f"{BASE_URL}/check"

    DEFAULT_TIMEOUT_SECONDS = 5.0

    MIN_MAX_AGE_DAYS = 1
    MAX_MAX_AGE_DAYS = 365

    DEFAULT_MAX_AGE_DAYS = 90

    DEFAULT_MALICIOUS_THRESHOLD = 0.40

    # Hard safety limits for provider-controlled payloads.
    MAX_RESPONSE_BYTES = 5 * 1024 * 1024
    MAX_CATEGORIES = 64
    MAX_TAGS = 64
    MAX_TAG_LENGTH = 256

    # AbuseIPDB documented category mapping.
    CATEGORY_MAP: Dict[int, str] = {
        3: "fraud_orders",
        4: "ddos_attack",
        5: "ftp_brute_force",
        6: "ping_of_death",
        7: "phishing",
        8: "fraud_voip",
        9: "open_proxy",
        10: "web_spam",
        11: "email_spam",
        12: "blog_spam",
        13: "vpn_ip",
        14: "port_scan",
        15: "hacking",
        16: "sql_injection",
        17: "spoofing",
        18: "brute_force",
        19: "bad_web_bot",
        20: "exploited_host",
        21: "web_app_attack",
        22: "ssh_abuse",
        23: "iot_targeted",
        24: "phishing",
        25: "cloud_service_abuse",
        26: "malware_distribution",
        27: "spam",
        28: "proxy_abuse",
        29: "credential_stuffing",
        30: "email_harvesting",
    }

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_age_in_days: int = DEFAULT_MAX_AGE_DAYS,
        custom_client: Optional[httpx.AsyncClient] = None,
        malicious_threshold: float = DEFAULT_MALICIOUS_THRESHOLD,
        allow_private_ips: bool = False,
        allow_reserved_ips: bool = False,
        healthcheck_ip: str = "8.8.8.8",
    ) -> None:
        """
        Initialize the AbuseIPDB provider.

        Parameters
        ----------
        api_key:
            AbuseIPDB API v2 key.

        timeout_seconds:
            Per-request timeout.

        max_age_in_days:
            Historical reporting window. AbuseIPDB accepts 1-365 days.

        custom_client:
            Optional externally managed httpx.AsyncClient.

        malicious_threshold:
            Normalized score at or above which an IOC is considered malicious.

        allow_private_ips:
            Whether RFC1918/private IP addresses may be queried.

        allow_reserved_ips:
            Whether reserved/loopback/link-local/etc. addresses may be queried.

        healthcheck_ip:
            IP used by check_health().
        """

        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "AbuseIPDB API key must be a non-empty string."
            )

        if not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be numeric.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        if not isinstance(max_age_in_days, int):
            raise TypeError("max_age_in_days must be an integer.")

        if not (
            self.MIN_MAX_AGE_DAYS
            <= max_age_in_days
            <= self.MAX_MAX_AGE_DAYS
        ):
            raise ValueError(
                "max_age_in_days must be between "
                f"{self.MIN_MAX_AGE_DAYS} and "
                f"{self.MAX_MAX_AGE_DAYS}."
            )

        if not isinstance(malicious_threshold, (int, float)):
            raise TypeError("malicious_threshold must be numeric.")

        if not 0.0 <= float(malicious_threshold) <= 1.0:
            raise ValueError(
                "malicious_threshold must be between 0.0 and 1.0."
            )

        super().__init__(
            api_key=api_key.strip(),
            timeout_seconds=float(timeout_seconds),
        )

        self.max_age_in_days = max_age_in_days
        self.malicious_threshold = float(malicious_threshold)

        self.allow_private_ips = bool(allow_private_ips)
        self.allow_reserved_ips = bool(allow_reserved_ips)

        self.healthcheck_ip = self._normalize_ip(healthcheck_ip)

        self._external_client = custom_client
        self._owned_client: Optional[httpx.AsyncClient] = None
        self._closed = False

    # ==========================================================================
    # PROVIDER CONTRACT
    # ==========================================================================

    @property
    def provider_name(self) -> str:
        """Return stable provider identifier."""
        return "abuseipdb"

    @property
    def supported_ioc_types(self) -> Set[IOCType]:
        """Return IOC types supported by AbuseIPDB."""
        return {IOCType.IP}

    # ==========================================================================
    # PUBLIC QUERY METHODS
    # ==========================================================================

    async def query_ip(
        self,
        ip_address: str,
    ) -> NormalizedThreatReport:
        """
        Query AbuseIPDB for an IPv4 or IPv6 address.

        Raises
        ------
        ValueError
            Invalid or disallowed IP address.

        RateLimitExceededError
            AbuseIPDB rate limit reached.

        ProviderTimeoutError
            Request exceeded configured timeout.

        ThreatIntelAPIError
            Provider/API/transport/response parsing failure.
        """

        clean_ip = self._validate_and_normalize_ip(ip_address)

        params: Dict[str, Any] = {
            "ipAddress": clean_ip,
            "maxAgeInDays": self.max_age_in_days,
            "verbose": "true",
        }

        return await self._execute_query(
            endpoint=self.CHECK_ENDPOINT,
            params=params,
            ioc_value=clean_ip,
        )

    async def query_url(
        self,
        url: str,
    ) -> NormalizedThreatReport:
        """
        Return an explicit UNKNOWN report for unsupported URL IOC queries.

        IMPORTANT:
        Unsupported does not mean CLEAN.
        """
        return self._create_unsupported_report(
            ioc_value=str(url).strip(),
            ioc_type=IOCType.URL,
        )

    async def query_domain(
        self,
        domain: str,
    ) -> NormalizedThreatReport:
        """
        Return an explicit UNKNOWN report for unsupported domain queries.
        """
        return self._create_unsupported_report(
            ioc_value=str(domain).strip(),
            ioc_type=IOCType.DOMAIN,
        )

    async def check_health(self) -> bool:
        """
        Check API connectivity and authentication.

        This uses the configured healthcheck IP and therefore consumes an API
        request quota. It should not be called excessively.

        Returns
        -------
        bool
            True only for a successful API response.
        """

        if self._closed:
            return False

        params = {
            "ipAddress": self.healthcheck_ip,
            "maxAgeInDays": 1,
        }

        try:
            response = await self._request(
                endpoint=self.CHECK_ENDPOINT,
                params=params,
            )

            return response.status_code == 200

        except (
            ProviderTimeoutError,
            RateLimitExceededError,
            ThreatIntelAPIError,
        ):
            return False

    # ==========================================================================
    # HTTP EXECUTION
    # ==========================================================================

    async def _execute_query(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        ioc_value: str,
    ) -> NormalizedThreatReport:
        """
        Execute an AbuseIPDB request and normalize the result.
        """

        start_time = time.perf_counter()

        try:
            response = await self._request(
                endpoint=endpoint,
                params=params,
            )

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            self._validate_response_size(response)

            status = response.status_code

            # ------------------------------------------------------------------
            # SUCCESS
            # ------------------------------------------------------------------

            if status == 200:
                payload = self._decode_json_response(response)

                return self._parse_abuseipdb_payload(
                    payload=payload,
                    ioc_value=ioc_value,
                    elapsed_ms=elapsed_ms,
                )

            # ------------------------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------------------------

            if status == 429:
                retry_after = self._parse_retry_after(
                    response.headers.get("Retry-After")
                )

                raise RateLimitExceededError(
                    provider=self.provider_name,
                    retry_after=retry_after,
                )

            # ------------------------------------------------------------------
            # AUTHENTICATION / AUTHORIZATION
            # ------------------------------------------------------------------

            if status in {401, 403}:
                raise ThreatIntelAPIError(
                    provider=self.provider_name,
                    status_code=status,
                    message=self._extract_error_message(response),
                )

            # ------------------------------------------------------------------
            # NOT FOUND
            # ------------------------------------------------------------------

            if status == 404:
                return self._create_not_found_report(
                    ioc_value=ioc_value,
                    elapsed_ms=elapsed_ms,
                )

            # ------------------------------------------------------------------
            # VALIDATION ERROR
            # ------------------------------------------------------------------

            if status == 422:
                raise ThreatIntelAPIError(
                    provider=self.provider_name,
                    status_code=status,
                    message=self._extract_error_message(response),
                )

            # ------------------------------------------------------------------
            # PAYMENT / PLAN LIMIT
            # ------------------------------------------------------------------

            if status == 402:
                raise ThreatIntelAPIError(
                    provider=self.provider_name,
                    status_code=status,
                    message=(
                        "AbuseIPDB rejected the request because the "
                        "configured API plan does not permit it."
                    ),
                )

            # ------------------------------------------------------------------
            # SERVER-SIDE ERROR
            # ------------------------------------------------------------------

            if 500 <= status <= 599:
                raise ThreatIntelAPIError(
                    provider=self.provider_name,
                    status_code=status,
                    message="AbuseIPDB server-side error.",
                )

            # ------------------------------------------------------------------
            # OTHER HTTP ERROR
            # ------------------------------------------------------------------

            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=status,
                message=self._extract_error_message(response),
            )

        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider=self.provider_name,
                timeout_seconds=self.timeout_seconds,
            ) from None

        except httpx.RequestError as exc:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=503,
                message=f"HTTP transport failure: {type(exc).__name__}",
            ) from exc

        except (
            RateLimitExceededError,
            ThreatIntelAPIError,
            ProviderTimeoutError,
        ):
            raise

        except ValueError as exc:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message=f"Invalid provider response: {exc}",
            ) from exc

        except Exception as exc:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=500,
                message=(
                    "Unexpected AbuseIPDB provider failure: "
                    f"{type(exc).__name__}"
                ),
            ) from exc

    async def _request(
        self,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> httpx.Response:
        """
        Perform an HTTP GET request.

        API keys are sent exclusively through HTTP headers.
        """

        if self._closed:
            raise RuntimeError(
                "AbuseIPDBProvider has already been closed."
            )

        client = await self._get_client()

        headers = {
            "Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "PhishGuard-AI-ThreatIntel/14.0",
        }

        request_id = uuid.uuid4().hex

        headers["X-PhishGuard-Request-ID"] = request_id

        return await client.get(
            endpoint,
            headers=headers,
            params=dict(params),
            timeout=self.timeout_seconds,
        )

    # ==========================================================================
    # RESPONSE PARSING
    # ==========================================================================

    def _parse_abuseipdb_payload(
        self,
        payload: Mapping[str, Any],
        ioc_value: str,
        elapsed_ms: float,
    ) -> NormalizedThreatReport:
        """
        Convert AbuseIPDB response into NormalizedThreatReport.

        Parsing is deliberately defensive because provider payloads are
        external/untrusted data.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("Provider payload must be a JSON object.")

        data = payload.get("data", {})

        if not isinstance(data, Mapping):
            raise ValueError("Provider 'data' field must be an object.")

        confidence_score = self._safe_float(
            data.get("abuseConfidenceScore"),
            default=0.0,
            minimum=0.0,
            maximum=100.0,
        )

        risk_score = confidence_score / 100.0

        total_reports = self._safe_int(
            data.get("totalReports"),
            default=0,
            minimum=0,
        )

        distinct_users = self._safe_int(
            data.get("numDistinctUsers"),
            default=0,
            minimum=0,
        )

        is_whitelisted = data.get("isWhitelisted")

        severity = NormalizedThreatReport.calculate_severity(
            risk_score
        )

        is_malicious = (
            risk_score >= self.malicious_threshold
        )

        threat_categories = self._extract_categories(data)

        raw_tags = self._extract_tags(
            data=data,
            total_reports=total_reports,
            distinct_users=distinct_users,
            is_whitelisted=is_whitelisted,
        )

        # AbuseIPDB reports are not antivirus engines.
        #
        # We therefore do NOT fabricate an engine count.
        #
        # positive_detections is retained as a compatibility field and
        # represents positive abuse observations/reports.
        #
        # total_engines remains zero because no antivirus-engine scan count
        # exists in this provider response.

        normalized_response = self._build_safe_raw_response(
            payload
        )

        return NormalizedThreatReport(
            provider_name=self.provider_name,
            ioc_value=ioc_value,
            ioc_type=IOCType.IP,
            risk_score=risk_score,
            severity=severity,
            is_malicious=is_malicious,
            positive_detections=total_reports,
            total_engines=0,
            threat_categories=threat_categories,
            raw_tags=raw_tags,
            response_time_ms=max(0.0, float(elapsed_ms)),
            raw_response=normalized_response,
        )

    # ==========================================================================
    # CATEGORY / TAG NORMALIZATION
    # ==========================================================================

    def _extract_categories(
        self,
        data: Mapping[str, Any],
    ) -> List[str]:
        """
        Extract deterministic normalized AbuseIPDB categories.
        """

        reports = data.get("reports", [])

        if not isinstance(reports, list):
            return []

        category_names: Set[str] = set()

        for report in reports:
            if not isinstance(report, Mapping):
                continue

            categories = report.get("categories", [])

            if not isinstance(categories, list):
                continue

            for category_id in categories:
                try:
                    numeric_id = int(category_id)
                except (TypeError, ValueError):
                    continue

                category_name = self.CATEGORY_MAP.get(
                    numeric_id,
                    f"category_{numeric_id}",
                )

                category_name = self._sanitize_tag(
                    category_name
                )

                if category_name:
                    category_names.add(category_name)

                if len(category_names) >= self.MAX_CATEGORIES:
                    break

            if len(category_names) >= self.MAX_CATEGORIES:
                break

        return sorted(category_names)[: self.MAX_CATEGORIES]

    def _extract_tags(
        self,
        data: Mapping[str, Any],
        total_reports: int,
        distinct_users: int,
        is_whitelisted: Any,
    ) -> List[str]:
        """
        Produce bounded deterministic tags from provider metadata.
        """

        tags: Set[str] = set()

        if is_whitelisted is True:
            tags.add("whitelisted")

        if is_whitelisted is False:
            tags.add("not_whitelisted")

        usage_type = data.get("usageType")

        if isinstance(usage_type, str):
            normalized_usage = self._sanitize_tag(
                usage_type
            )

            if normalized_usage:
                tags.add(f"usage:{normalized_usage}")

        isp = data.get("isp")

        if isinstance(isp, str):
            normalized_isp = self._sanitize_tag(
                isp,
                max_length=96,
            )

            if normalized_isp:
                tags.add(f"isp:{normalized_isp}")

        domain = data.get("domain")

        if isinstance(domain, str):
            normalized_domain = self._sanitize_tag(
                domain,
                max_length=96,
            )

            if normalized_domain:
                tags.add(f"domain:{normalized_domain}")

        country_code = data.get("countryCode")

        if isinstance(country_code, str):
            normalized_country = self._sanitize_tag(
                country_code,
                max_length=16,
            )

            if normalized_country:
                tags.add(f"country:{normalized_country}")

        if total_reports > 0:
            tags.add("has_abuse_reports")

        if total_reports >= 10:
            tags.add("multiple_abuse_reports")

        if distinct_users >= 2:
            tags.add("multiple_reporters")

        return sorted(tags)[: self.MAX_TAGS]

    # ==========================================================================
    # INPUT VALIDATION
    # ==========================================================================

    def _validate_and_normalize_ip(
        self,
        ip_address: str,
    ) -> str:
        """
        Validate and canonicalize IPv4/IPv6 input.
        """

        if not isinstance(ip_address, str):
            raise TypeError("IP address must be a string.")

        candidate = ip_address.strip()

        if not candidate:
            raise ValueError("IP address cannot be empty.")

        return self._normalize_ip(candidate)

    def _normalize_ip(
        self,
        ip_address: str,
    ) -> str:
        """
        Canonicalize IP using Python's ipaddress implementation.
        """

        try:
            parsed = ipaddress.ip_address(
                ip_address.strip()
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid IP address: {ip_address!r}"
            ) from exc

        if not self.allow_private_ips and parsed.is_private:
            raise ValueError(
                f"Private IP addresses are disabled: {ip_address!r}"
            )

        if not self.allow_reserved_ips:
            disallowed = (
                parsed.is_loopback
                or parsed.is_link_local
                or parsed.is_unspecified
                or parsed.is_reserved
                or parsed.is_multicast
            )

            if disallowed:
                raise ValueError(
                    f"Reserved/special IP address is disabled: "
                    f"{ip_address!r}"
                )

        return str(parsed)

    # ==========================================================================
    # REPORT BUILDERS
    # ==========================================================================

    def _create_unsupported_report(
        self,
        ioc_value: str,
        ioc_type: IOCType,
    ) -> NormalizedThreatReport:
        """
        Build UNKNOWN report for an IOC type not supported by AbuseIPDB.

        Never return CLEAN here.

        CLEAN means the provider evaluated the IOC and found no meaningful
        evidence of abuse.

        UNKNOWN means the provider cannot evaluate that IOC type.
        """

        return NormalizedThreatReport(
            provider_name=self.provider_name,
            ioc_value=ioc_value,
            ioc_type=ioc_type,
            risk_score=0.0,
            severity=ThreatSeverity.UNKNOWN,
            is_malicious=False,
            positive_detections=0,
            total_engines=0,
            threat_categories=[],
            raw_tags=["unsupported_ioc_type"],
            response_time_ms=0.0,
            raw_response={
                "status": "unsupported_ioc_type",
                "ioc_type": ioc_type.value,
            },
        )

    def _create_not_found_report(
        self,
        ioc_value: str,
        elapsed_ms: float,
    ) -> NormalizedThreatReport:
        """
        Build a report for a successful provider query with no matching
        reputation record.

        This is CLEAN only because the provider successfully evaluated the IP
        and returned no abuse record.
        """

        return NormalizedThreatReport(
            provider_name=self.provider_name,
            ioc_value=ioc_value,
            ioc_type=IOCType.IP,
            risk_score=0.0,
            severity=ThreatSeverity.CLEAN,
            is_malicious=False,
            positive_detections=0,
            total_engines=0,
            threat_categories=[],
            raw_tags=["not_found"],
            response_time_ms=max(0.0, elapsed_ms),
            raw_response={
                "status": "not_found",
            },
        )

    # ==========================================================================
    # SAFE RESPONSE HANDLING
    # ==========================================================================

    def _decode_json_response(
        self,
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """
        Decode provider JSON safely.
        """

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                "AbuseIPDB returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "AbuseIPDB JSON root must be an object."
            )

        return payload

    def _extract_error_message(
        self,
        response: httpx.Response,
    ) -> str:
        """
        Extract bounded provider error information without exposing secrets.
        """

        try:
            payload = response.json()

            errors = (
                payload.get("errors")
                if isinstance(payload, dict)
                else None
            )

            if isinstance(errors, list):
                messages: List[str] = []

                for error in errors[:8]:
                    if isinstance(error, Mapping):
                        detail = error.get("detail")

                        if detail:
                            messages.append(
                                str(detail)[:512]
                            )

                if messages:
                    return "; ".join(messages)

        except Exception:
            pass

        text = response.text.strip()

        if not text:
            return f"HTTP {response.status_code}"

        return text[:512]

    def _validate_response_size(
        self,
        response: httpx.Response,
    ) -> None:
        """
        Reject unexpectedly large provider responses.
        """

        content_length = response.headers.get(
            "Content-Length"
        )

        if content_length:
            try:
                size = int(content_length)

                if size > self.MAX_RESPONSE_BYTES:
                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=502,
                        message=(
                            "AbuseIPDB response exceeded "
                            f"{self.MAX_RESPONSE_BYTES} bytes."
                        ),
                    )

            except ValueError:
                # Invalid Content-Length is not trusted.
                pass

    @staticmethod
    def _build_safe_raw_response(
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Preserve provider payload while guaranteeing a dictionary result.

        The payload itself contains provider data, not the API key, because the
        key is supplied only through request headers.
        """

        return dict(payload)

    # ==========================================================================
    # NUMERIC DEFENSIVE PARSING
    # ==========================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Convert provider value to bounded finite float.
        """

        try:
            result = float(value)

            if not result == result:  # NaN
                return default

            if result in {
                float("inf"),
                float("-inf"),
            }:
                return default

            return min(
                maximum,
                max(minimum, result),
            )

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(
        value: Any,
        default: int,
        minimum: int,
    ) -> int:
        """
        Convert provider value to bounded non-negative integer.
        """

        try:
            result = int(value)

            return max(
                minimum,
                result,
            )

        except (TypeError, ValueError):
            return default

    # ==========================================================================
    # RATE LIMIT UTILITIES
    # ==========================================================================

    @staticmethod
    def _parse_retry_after(
        value: Optional[str],
    ) -> Optional[int]:
        """
        Parse Retry-After header.

        AbuseIPDB documents Retry-After as seconds to wait.
        """

        if not value:
            return None

        try:
            seconds = int(value)

            if seconds < 0:
                return None

            return seconds

        except (TypeError, ValueError):
            return None

    # ==========================================================================
    # TAG SANITIZATION
    # ==========================================================================

    @classmethod
    def _sanitize_tag(
        cls,
        value: str,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Normalize provider-controlled tag strings.

        This prevents uncontrolled whitespace/control characters and keeps
        downstream logs/metrics manageable.
        """

        if not isinstance(value, str):
            return ""

        limit = max_length or cls.MAX_TAG_LENGTH

        cleaned = " ".join(
            value.strip().split()
        )

        cleaned = cleaned.replace(
            "\x00",
            "",
        )

        return cleaned[:limit].lower()

    # ==========================================================================
    # HTTP CLIENT LIFECYCLE
    # ==========================================================================

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Return external client or lazily create an internally owned client.
        """

        if self._external_client is not None:
            return self._external_client

        if self._owned_client is None:
            self._owned_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.timeout_seconds
                ),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
                follow_redirects=False,
                http2=True,
            )

        return self._owned_client

    async def aclose(self) -> None:
        """
        Close internally owned HTTP resources.

        Externally supplied clients are never closed by this provider.
        """

        if self._closed:
            return

        self._closed = True

        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None

    async def __aenter__(self) -> "AbuseIPDBProvider":
        """
        Support:

            async with AbuseIPDBProvider(...) as provider:
                ...
        """

        if self._closed:
            raise RuntimeError(
                "Cannot enter a closed AbuseIPDBProvider."
            )

        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Close internally owned resources."""
        await self.aclose()

    # ==========================================================================
    # DIAGNOSTIC / OPERATIONAL HELPERS
    # ==========================================================================

    def get_rate_limit_snapshot(
        self,
        response: httpx.Response,
    ) -> Dict[str, Optional[str]]:
        """
        Extract documented AbuseIPDB rate-limit headers.

        Useful for observability without exposing the API key.
        """

        return {
            "limit": response.headers.get(
                "X-RateLimit-Limit"
            ),
            "remaining": response.headers.get(
                "X-RateLimit-Remaining"
            ),
            "reset": response.headers.get(
                "X-RateLimit-Reset"
            ),
            "retry_after": response.headers.get(
                "Retry-After"
            ),
        }