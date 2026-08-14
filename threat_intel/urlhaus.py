"""
PhishGuard AI - URLhaus Threat Intelligence Provider
====================================================

Version: 14.0.0
File Path: threat_intel/urlhaus.py

Production-grade asynchronous integration with the abuse.ch URLhaus
Community API.

Security / reliability goals
-----------------------------
- Strict IOC validation and canonicalization.
- Explicit URLhaus authentication handling.
- Safe async HTTP client lifecycle management.
- Connection reuse when a client is injected.
- Bounded request timeout.
- Retry handling for transient failures.
- Rate-limit handling with Retry-After support.
- Provider-response validation.
- Deterministic normalization of tags/categories.
- Conservative risk scoring.
- Correct distinction between "not observed" and "clean".
- No false AV-engine semantics for URLhaus intelligence.
- Bounded raw response retention.
- No secret leakage through exception messages.
- IPv4/IPv6 support.
- Defensive handling of malformed provider responses.
- Backwards-compatible BaseThreatProvider interface.
"""

from __future__ import annotations

import asyncio
import ipaddress
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Set
from urllib.parse import urlsplit

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


class URLhausProvider(BaseThreatProvider):
    """
    Production URLhaus provider.

    URLhaus is an IOC intelligence source focused on malicious URLs and
    malware-distribution infrastructure. It should therefore not be modeled
    as an antivirus-engine consensus provider.

    The provider supports:
        - URL lookup
        - Domain/host lookup
        - IP lookup
    """

    DEFAULT_BASE_URL = "https://urlhaus-api.abuse.ch/v1"

    # Conservative limits prevent pathological provider responses from
    # consuming excessive memory.
    MAX_INPUT_LENGTH = 8192
    MAX_TAG_LENGTH = 256
    MAX_CATEGORY_LENGTH = 256
    MAX_RAW_RESPONSE_BYTES = 1_000_000
    MAX_ASSOCIATED_URLS = 10_000

    DEFAULT_CONNECT_TIMEOUT = 2.0
    DEFAULT_READ_TIMEOUT = 5.0
    DEFAULT_WRITE_TIMEOUT = 5.0
    DEFAULT_POOL_TIMEOUT = 2.0

    MAX_RETRIES = 2
    RETRY_BASE_DELAY = 0.25

    # URLhaus is malware-distribution intelligence. The score deliberately
    # does NOT pretend to represent a probability.
    ONLINE_URL_SCORE = 0.95
    OFFLINE_URL_SCORE = 0.70
    UNKNOWN_URL_STATUS_SCORE = 0.50

    @property
    def provider_name(self) -> str:
        return "urlhaus"

    @property
    def supported_ioc_types(self) -> Set[IOCType]:
        return {
            IOCType.URL,
            IOCType.DOMAIN,
            IOCType.IP,
        }

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: float = 5.0,
        custom_client: Optional[httpx.AsyncClient] = None,
        base_url: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
        max_raw_response_bytes: int = MAX_RAW_RESPONSE_BYTES,
    ) -> None:
        """
        Initialize URLhaus provider.

        Args:
            api_key:
                URLhaus Auth-Key.

            timeout_seconds:
                Maximum logical request timeout.

            custom_client:
                Optional externally managed AsyncClient.

            base_url:
                Configurable API endpoint, useful for testing and controlled
                deployments.

            max_retries:
                Number of retries for transient failures.

            max_raw_response_bytes:
                Maximum retained serialized response size.
        """
        timeout_seconds = float(timeout_seconds)

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be a finite positive number."
            )

        if max_retries < 0 or max_retries > 5:
            raise ValueError(
                "max_retries must be between 0 and 5."
            )

        if max_raw_response_bytes <= 0:
            raise ValueError(
                "max_raw_response_bytes must be greater than zero."
            )

        # URLhaus currently requires authentication for Community API access.
        # Keep the constructor technically optional only for compatibility,
        # but fail when an actual API request is attempted.
        super().__init__(
            api_key=api_key.strip() if api_key else None,
            timeout_seconds=timeout_seconds,
        )

        self._external_client = custom_client
        self._owns_client = custom_client is None

        self.base_url = (
            base_url.strip().rstrip("/")
            if base_url
            else self.DEFAULT_BASE_URL
        )

        self.max_retries = int(max_retries)
        self.max_raw_response_bytes = int(max_raw_response_bytes)

        self._client: Optional[httpx.AsyncClient] = custom_client
        self._client_lock = asyncio.Lock()

    # ======================================================================
    # CLIENT LIFECYCLE
    # ======================================================================

    async def __aenter__(self) -> "URLhausProvider":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """
        Lazily create a reusable AsyncClient.

        This prevents the original implementation's per-request client
        creation/resource leak.
        """
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is None:
                timeout = httpx.Timeout(
                    timeout=self.timeout_seconds,
                    connect=min(
                        self.timeout_seconds,
                        self.DEFAULT_CONNECT_TIMEOUT,
                    ),
                    read=min(
                        self.timeout_seconds,
                        self.DEFAULT_READ_TIMEOUT,
                    ),
                    write=min(
                        self.timeout_seconds,
                        self.DEFAULT_WRITE_TIMEOUT,
                    ),
                    pool=min(
                        self.timeout_seconds,
                        self.DEFAULT_POOL_TIMEOUT,
                    ),
                )

                limits = httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0,
                )

                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    follow_redirects=False,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "PhishGuard-AI/14.0",
                    },
                )

        return self._client

    async def aclose(self) -> None:
        """
        Close internally owned HTTP resources.

        Externally supplied clients remain owned by the caller.
        """
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ======================================================================
    # PUBLIC IOC QUERIES
    # ======================================================================

    async def query_url(
        self,
        url: str,
    ) -> NormalizedThreatReport:
        """
        Query URLhaus for an exact URL.
        """
        clean_url = self._normalize_url(url)

        return await self._execute_query(
            endpoint=f"{self.base_url}/url/",
            data={"url": clean_url},
            ioc_value=clean_url,
            ioc_type=IOCType.URL,
        )

    async def query_domain(
        self,
        domain: str,
    ) -> NormalizedThreatReport:
        """
        Query URLhaus for a domain/host.
        """
        clean_domain = self._normalize_domain(domain)

        return await self._execute_query(
            endpoint=f"{self.base_url}/host/",
            data={"host": clean_domain},
            ioc_value=clean_domain,
            ioc_type=IOCType.DOMAIN,
        )

    async def query_ip(
        self,
        ip_address: str,
    ) -> NormalizedThreatReport:
        """
        Query URLhaus for an IPv4 or IPv6 host.
        """
        clean_ip = self._normalize_ip(ip_address)

        return await self._execute_query(
            endpoint=f"{self.base_url}/host/",
            data={"host": clean_ip},
            ioc_value=clean_ip,
            ioc_type=IOCType.IP,
        )

    async def check_health(self) -> bool:
        """
        Perform a lightweight authenticated API health check.

        Health status means the provider endpoint is reachable and accepts
        the supplied authentication. It does not depend on whether the
        selected IOC exists in URLhaus.
        """
        try:
            if not self.api_key:
                return False

            client = await self._ensure_client()

            response = await client.post(
                f"{self.base_url}/host/",
                data={"host": "127.0.0.1"},
                headers=self._get_headers(),
                timeout=self.timeout_seconds,
            )

            return response.status_code == 200

        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            RateLimitExceededError,
            ThreatIntelAPIError,
        ):
            return False

    # ======================================================================
    # HTTP EXECUTION
    # ======================================================================

    async def _execute_query(
        self,
        endpoint: str,
        data: Dict[str, str],
        ioc_value: str,
        ioc_type: IOCType,
    ) -> NormalizedThreatReport:
        """
        Execute a URLhaus API request with bounded retries.
        """
        self._require_api_key()

        client = await self._ensure_client()
        headers = self._get_headers()

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()

            try:
                response = await client.post(
                    endpoint,
                    data=data,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )

                elapsed_ms = (
                    time.perf_counter() - start_time
                ) * 1000.0

                # ----------------------------------------------------------
                # Rate limiting
                # ----------------------------------------------------------

                if response.status_code == 429:
                    retry_after = self._parse_retry_after(
                        response.headers.get("Retry-After")
                    )

                    raise RateLimitExceededError(
                        provider=self.provider_name,
                        retry_after=retry_after,
                    )

                # ----------------------------------------------------------
                # Transient upstream failures
                # ----------------------------------------------------------

                if response.status_code in {500, 502, 503, 504}:
                    if attempt < self.max_retries:
                        await self._backoff(attempt)
                        continue

                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=response.status_code,
                        message="URLhaus upstream service temporarily unavailable.",
                    )

                # ----------------------------------------------------------
                # Authentication failures
                # ----------------------------------------------------------

                if response.status_code in {401, 403}:
                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=response.status_code,
                        message="URLhaus authentication or authorization failed.",
                    )

                # ----------------------------------------------------------
                # Other HTTP failures
                # ----------------------------------------------------------

                if response.status_code != 200:
                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=response.status_code,
                        message=(
                            "URLhaus returned an unexpected HTTP status."
                        ),
                    )

                payload = self._decode_json_response(response)

                return self._parse_urlhaus_payload(
                    payload=payload,
                    ioc_value=ioc_value,
                    ioc_type=ioc_type,
                    elapsed_ms=elapsed_ms,
                )

            except httpx.TimeoutException as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    raise ProviderTimeoutError(
                        provider=self.provider_name,
                        timeout_seconds=self.timeout_seconds,
                    ) from exc

                await self._backoff(attempt)

            except httpx.NetworkError as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=503,
                        message="URLhaus network communication failed.",
                    ) from exc

                await self._backoff(attempt)

            except (
                RateLimitExceededError,
                ThreatIntelAPIError,
                ProviderTimeoutError,
            ):
                raise

            except Exception as exc:
                raise ThreatIntelAPIError(
                    provider=self.provider_name,
                    status_code=500,
                    message="Unexpected URLhaus provider failure.",
                ) from exc

        raise ThreatIntelAPIError(
            provider=self.provider_name,
            status_code=503,
            message="URLhaus request failed after retry policy.",
        ) from last_error

    async def _backoff(self, attempt: int) -> None:
        """
        Small bounded exponential backoff.
        """
        delay = min(
            self.RETRY_BASE_DELAY * (2**attempt),
            2.0,
        )
        await asyncio.sleep(delay)

    # ======================================================================
    # RESPONSE PARSING
    # ======================================================================

    def _decode_json_response(
        self,
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """
        Safely decode and validate JSON response structure.
        """
        content_length = response.headers.get("Content-Length")

        if content_length:
            try:
                if int(content_length) > self.max_raw_response_bytes:
                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=502,
                        message="URLhaus response exceeded configured size limit.",
                    )
            except ValueError:
                # Ignore malformed Content-Length and rely on parsed object
                # size constraints below.
                pass

        try:
            payload = response.json()
        except ValueError as exc:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message="URLhaus returned malformed JSON.",
            ) from exc

        if not isinstance(payload, dict):
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message="URLhaus returned an invalid response structure.",
            )

        return payload

    def _parse_urlhaus_payload(
        self,
        payload: Dict[str, Any],
        ioc_value: str,
        ioc_type: IOCType,
        elapsed_ms: float,
    ) -> NormalizedThreatReport:
        """
        Convert URLhaus response into the common PhishGuard schema.
        """
        query_status = str(
            payload.get("query_status", "unknown")
        ).strip().lower()

        # --------------------------------------------------------------
        # IMPORTANT:
        # "not found" != "clean".
        #
        # URLhaus only tells us that it has no matching intelligence.
        # It cannot establish benignity.
        # --------------------------------------------------------------

        if query_status in {
            "no_results",
            "invalid_url",
            "invalid_host",
        }:
            return self._build_unknown_report(
                ioc_value=ioc_value,
                ioc_type=ioc_type,
                elapsed_ms=elapsed_ms,
                payload=payload,
                reason=query_status,
            )

        if query_status != "ok":
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message=(
                    f"URLhaus returned unexpected query status: "
                    f"{query_status[:64]}"
                ),
            )

        if ioc_type == IOCType.URL:
            return self._parse_url_result(
                payload=payload,
                ioc_value=ioc_value,
                elapsed_ms=elapsed_ms,
            )

        return self._parse_host_result(
            payload=payload,
            ioc_value=ioc_value,
            ioc_type=ioc_type,
            elapsed_ms=elapsed_ms,
        )

    # ======================================================================
    # URL RESULT NORMALIZATION
    # ======================================================================

    def _parse_url_result(
        self,
        payload: Dict[str, Any],
        ioc_value: str,
        elapsed_ms: float,
    ) -> NormalizedThreatReport:
        """
        Normalize an exact URL lookup.
        """
        url_status = str(
            payload.get("url_status", "unknown")
        ).strip().lower()

        threat = self._safe_string(
            payload.get("threat"),
            self.MAX_CATEGORY_LENGTH,
        )

        tags = self._normalize_strings(
            payload.get("tags"),
            self.MAX_TAG_LENGTH,
        )

        categories: List[str] = []

        if threat:
            categories.append(threat.lower())

        if payload.get("larted") in {True, "true", "1", 1}:
            tags.append("larted")

        blacklist_data = payload.get("blacklists")
        if isinstance(blacklist_data, Mapping):
            for provider, status in blacklist_data.items():
                provider_name = self._safe_string(
                    provider,
                    64,
                )
                status_name = self._safe_string(
                    status,
                    64,
                )

                if provider_name and status_name:
                    tags.append(
                        f"blacklist:{provider_name.lower()}:{status_name.lower()}"
                    )

        tags = self._unique_sorted(tags)

        if url_status == "online":
            risk_score = self.ONLINE_URL_SCORE

        elif url_status == "offline":
            risk_score = self.OFFLINE_URL_SCORE

        else:
            risk_score = self.UNKNOWN_URL_STATUS_SCORE

        severity = NormalizedThreatReport.calculate_severity(
            risk_score
        )

        return NormalizedThreatReport(
            provider_name=self.provider_name,
            ioc_value=ioc_value,
            ioc_type=IOCType.URL,
            risk_score=risk_score,
            severity=severity,
            is_malicious=True,
            positive_detections=1,
            # URLhaus is not an AV engine.
            total_engines=1,
            threat_categories=self._unique_sorted(categories),
            raw_tags=tags,
            response_time_ms=max(0.0, elapsed_ms),
            raw_response=self._bounded_raw_response(payload),
        )

    # ======================================================================
    # HOST RESULT NORMALIZATION
    # ======================================================================

    def _parse_host_result(
        self,
        payload: Dict[str, Any],
        ioc_value: str,
        ioc_type: IOCType,
        elapsed_ms: float,
    ) -> NormalizedThreatReport:
        """
        Normalize URLhaus host/domain/IP intelligence.

        The host endpoint may contain multiple URLs. We use the number of
        observed malicious URLs as evidence, while keeping the score bounded.
        """
        urls = payload.get("urls")

        if not isinstance(urls, list):
            urls = []

        urls = urls[: self.MAX_ASSOCIATED_URLS]

        positive_detections = 0
        online_count = 0

        tags: List[str] = []
        categories: List[str] = []

        for item in urls:
            if not isinstance(item, Mapping):
                continue

            positive_detections += 1

            status = str(
                item.get("url_status", "")
            ).strip().lower()

            if status == "online":
                online_count += 1

            item_tags = self._normalize_strings(
                item.get("tags"),
                self.MAX_TAG_LENGTH,
            )
            tags.extend(item_tags)

            threat = self._safe_string(
                item.get("threat"),
                self.MAX_CATEGORY_LENGTH,
            )

            if threat:
                categories.append(threat.lower())

        # --------------------------------------------------------------
        # No associated malware URLs.
        # This is "not observed", not "clean".
        # --------------------------------------------------------------

        if positive_detections == 0:
            return self._build_unknown_report(
                ioc_value=ioc_value,
                ioc_type=ioc_type,
                elapsed_ms=elapsed_ms,
                payload=payload,
                reason="no_associated_urls",
            )

        # --------------------------------------------------------------
        # Conservative host scoring.
        #
        # A single observed malicious URL is already meaningful, but
        # we don't let the count alone create an artificial probability.
        # --------------------------------------------------------------

        if online_count > 0:
            risk_score = min(
                0.95,
                0.75 + min(online_count, 4) * 0.05,
            )
        else:
            risk_score = min(
                0.80,
                0.60 + min(positive_detections, 10) * 0.02,
            )

        severity = NormalizedThreatReport.calculate_severity(
            risk_score
        )

        return NormalizedThreatReport(
            provider_name=self.provider_name,
            ioc_value=ioc_value,
            ioc_type=ioc_type,
            risk_score=risk_score,
            severity=severity,
            is_malicious=True,
            positive_detections=positive_detections,
            # This field is retained for schema compatibility, but URLhaus
            # itself does not represent these observations as AV engines.
            total_engines=positive_detections,
            threat_categories=self._unique_sorted(categories),
            raw_tags=self._unique_sorted(tags),
            response_time_ms=max(0.0, elapsed_ms),
            raw_response=self._bounded_raw_response(payload),
        )

    # ======================================================================
    # UNKNOWN / NO-EVIDENCE REPORT
    # ======================================================================

    def _build_unknown_report(
        self,
        ioc_value: str,
        ioc_type: IOCType,
        elapsed_ms: float,
        payload: Dict[str, Any],
        reason: str,
    ) -> NormalizedThreatReport:
        """
        Build a conservative report for an IOC without URLhaus evidence.

        This deliberately uses UNKNOWN rather than CLEAN.
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
            raw_tags=[
                "urlhaus:no_match",
                f"urlhaus:status:{reason}",
            ],
            response_time_ms=max(0.0, elapsed_ms),
            raw_response=self._bounded_raw_response(payload),
        )

    # ======================================================================
    # IOC VALIDATION / NORMALIZATION
    # ======================================================================

    @classmethod
    def _normalize_url(cls, value: str) -> str:
        """
        Validate and normalize a URL without performing network resolution.

        This provider must never fetch the supplied URL itself.
        """
        if not isinstance(value, str):
            raise ValueError("URL must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("URL cannot be empty.")

        if len(value) > cls.MAX_INPUT_LENGTH:
            raise ValueError(
                f"URL exceeds maximum length of {cls.MAX_INPUT_LENGTH}."
            )

        parsed = urlsplit(value)

        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError(
                "URLhaus URL queries require an http or https URL."
            )

        if not parsed.netloc:
            raise ValueError("URL must contain a valid host.")

        if any(char.isspace() for char in value):
            raise ValueError("URL cannot contain whitespace.")

        # urlsplit.hostname performs host parsing without DNS resolution.
        try:
            hostname = parsed.hostname
        except ValueError as exc:
            raise ValueError("URL contains an invalid host.") from exc

        if not hostname:
            raise ValueError("URL must contain a hostname.")

        return value

    @classmethod
    def _normalize_domain(cls, value: str) -> str:
        """
        Normalize a domain/host without DNS resolution.
        """
        if not isinstance(value, str):
            raise ValueError("Domain must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("Domain cannot be empty.")

        if len(value) > cls.MAX_INPUT_LENGTH:
            raise ValueError(
                f"Domain exceeds maximum length of {cls.MAX_INPUT_LENGTH}."
            )

        # Accept accidental scheme prefixes but canonicalize them.
        if "://" in value:
            parsed = urlsplit(value)

            if not parsed.hostname:
                raise ValueError("Invalid domain URL.")

            value = parsed.hostname

        else:
            # Remove a simple path suffix but don't blindly manipulate
            # arbitrary malformed input.
            value = value.split("/", 1)[0]

            # IPv6 literals must remain intact.
            if value.startswith("[") and "]" in value:
                value = value[1:value.index("]")]

            elif value.count(":") == 1:
                host, port = value.rsplit(":", 1)

                if port.isdigit():
                    value = host

        value = value.rstrip(".").lower()

        if not value:
            raise ValueError("Domain cannot be empty.")

        if any(char.isspace() for char in value):
            raise ValueError("Domain cannot contain whitespace.")

        # Reject control characters.
        if any(ord(char) < 32 for char in value):
            raise ValueError("Domain contains control characters.")

        return value

    @staticmethod
    def _normalize_ip(value: str) -> str:
        """
        Canonicalize IPv4/IPv6 without DNS resolution.
        """
        if not isinstance(value, str):
            raise ValueError("IP address must be a string.")

        value = value.strip()

        try:
            return str(ipaddress.ip_address(value))
        except ValueError as exc:
            raise ValueError(
                f"Invalid IP address: {value!r}"
            ) from exc

    # ======================================================================
    # SECURITY / NORMALIZATION HELPERS
    # ======================================================================

    def _require_api_key(self) -> None:
        """
        Enforce URLhaus authentication at request time.
        """
        if not self.api_key:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=401,
                message="URLhaus Auth-Key is not configured.",
            )

    def _get_headers(self) -> Dict[str, str]:
        """
        Build authenticated HTTP headers.

        Never log or expose the API key.
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": "PhishGuard-AI/14.0",
        }

        if self.api_key:
            headers["Auth-Key"] = self.api_key

        return headers

    @staticmethod
    def _parse_retry_after(
        value: Optional[str],
    ) -> Optional[int]:
        """
        Parse integer Retry-After safely.
        """
        if not value:
            return None

        try:
            seconds = int(value.strip())
        except (TypeError, ValueError):
            return None

        return max(0, min(seconds, 3600))

    @classmethod
    def _safe_string(
        cls,
        value: Any,
        max_length: int,
    ) -> str:
        """
        Convert external values into bounded strings.
        """
        if value is None:
            return ""

        try:
            result = str(value).strip()
        except Exception:
            return ""

        if not result:
            return ""

        return result[:max_length]

    @classmethod
    def _normalize_strings(
        cls,
        values: Any,
        max_length: int,
    ) -> List[str]:
        """
        Normalize an arbitrary provider list into bounded strings.
        """
        if not isinstance(values, list):
            return []

        normalized: List[str] = []

        for value in values:
            item = cls._safe_string(
                value,
                max_length,
            )

            if item:
                normalized.append(item)

        return cls._unique_sorted(normalized)

    @staticmethod
    def _unique_sorted(
        values: List[str],
    ) -> List[str]:
        """
        Deduplicate and sort strings deterministically.
        """
        return sorted(
            {
                value.strip()
                for value in values
                if isinstance(value, str) and value.strip()
            }
        )

    def _bounded_raw_response(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Retain a bounded representation of the provider response.

        The primary goal is preventing an unexpectedly large upstream
        response from propagating indefinitely through the pipeline.
        """
        # URLhaus normal responses are normally small. This recursive
        # sanitizer provides a defensive upper bound for pathological data.
        remaining = self.max_raw_response_bytes

        def sanitize(value: Any) -> Any:
            nonlocal remaining

            if remaining <= 0:
                return "[TRUNCATED]"

            if isinstance(value, Mapping):
                result: Dict[str, Any] = {}

                for key, item in value.items():
                    if remaining <= 0:
                        break

                    safe_key = self._safe_string(key, 256)

                    result[safe_key] = sanitize(item)

                    # Approximate memory budget rather than relying on
                    # serialized JSON size.
                    remaining -= len(safe_key) + 32

                return result

            if isinstance(value, list):
                result_list: List[Any] = []

                for item in value:
                    if remaining <= 0:
                        break

                    result_list.append(sanitize(item))
                    remaining -= 16

                return result_list

            if isinstance(value, str):
                result = value[: min(len(value), remaining, 4096)]
                remaining -= len(result)
                return result

            if isinstance(value, (int, float, bool)) or value is None:
                remaining -= 16
                return value

            result = self._safe_string(
                value,
                min(4096, max(1, remaining)),
            )
            remaining -= len(result)
            return result

        sanitized = sanitize(payload)

        if isinstance(sanitized, dict):
            return sanitized

        return {
            "response": sanitized,
        }