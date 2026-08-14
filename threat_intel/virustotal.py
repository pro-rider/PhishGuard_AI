"""
PhishGuard AI - VirusTotal API v3 Threat Intelligence Provider
==============================================================

Version: 14.0.0
File Path: threat_intel/virustotal.py

Production-grade asynchronous VirusTotal API v3 integration.

Security / reliability objectives
---------------------------------
- Strict IOC validation and canonicalization
- Safe URL identifier generation
- API-key isolation
- Explicit HTTPS/TLS usage
- Bounded request timeouts
- Retry handling for transient failures
- Retry-After support
- Response-size protection
- Defensive JSON parsing
- Provider response normalization
- Correct distinction between UNKNOWN and CLEAN
- Deterministic risk scoring
- Safe metadata extraction
- Dependency injection for tests
- AsyncClient lifecycle management
- Optional caller-owned HTTP client
- Health-check support
- Structured provider exceptions
- No provider secrets in exceptions
- Compatibility with Pydantic NormalizedThreatReport
"""

from __future__ import annotations

import asyncio
import base64
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Set

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


class VirusTotalProvider(BaseThreatProvider):
    """
    Production VirusTotal API v3 provider.

    The provider supports:

        - URL
        - Domain
        - IPv4 / IPv6
        - SHA-256 hash

    The provider deliberately does not expose MD5 through the public
    query methods because VirusTotal hash lookup semantics should be
    handled explicitly by a dedicated hash method rather than silently
    treating every arbitrary hash as another IOC category.

    HTTP client ownership
    ---------------------
    If ``custom_client`` is supplied, the caller owns its lifecycle.

    If no client is supplied, this provider creates its own AsyncClient
    and closes it through ``aclose()`` / the async context manager.
    """

    BASE_URL = "https://www.virustotal.com/api/v3"

    DEFAULT_USER_AGENT = "PhishGuard-AI-ThreatIntel/14.0"

    DEFAULT_MAX_RETRIES = 2
    DEFAULT_BACKOFF_BASE_SECONDS = 0.5
    DEFAULT_MAX_BACKOFF_SECONDS = 8.0

    DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MiB

    DEFAULT_CONNECT_TIMEOUT = 3.0
    DEFAULT_READ_TIMEOUT = 5.0
    DEFAULT_WRITE_TIMEOUT = 5.0
    DEFAULT_POOL_TIMEOUT = 5.0

    TRANSIENT_STATUS_CODES = frozenset(
        {
            408,  # Request Timeout
            425,  # Too Early
            429,  # Too Many Requests
            500,
            502,
            503,
            504,
        }
    )

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 5.0,
        custom_client: Optional[httpx.AsyncClient] = None,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        """
        Initialize the VirusTotal provider.

        Parameters
        ----------
        api_key:
            VirusTotal API key.

        timeout_seconds:
            Overall read/request timeout used by the provider.

        custom_client:
            Optional caller-owned AsyncClient. Useful for tests,
            connection pooling, proxies, custom transports, etc.

        max_retries:
            Maximum retry attempts after the initial request.

        backoff_base_seconds:
            Base exponential retry delay.

        max_backoff_seconds:
            Maximum retry delay.

        max_response_bytes:
            Defensive response-size ceiling.

        user_agent:
            HTTP User-Agent sent to VirusTotal.
        """

        normalized_key = str(api_key).strip()

        if not normalized_key:
            raise ValueError("VirusTotal API key must be provided.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds cannot be negative.")

        if max_backoff_seconds <= 0:
            raise ValueError("max_backoff_seconds must be greater than zero.")

        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero.")

        super().__init__(
            api_key=normalized_key,
            timeout_seconds=float(timeout_seconds),
        )

        self._external_client = custom_client
        self._owned_client: Optional[httpx.AsyncClient] = None

        self.max_retries = int(max_retries)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.max_response_bytes = int(max_response_bytes)

        self.user_agent = (
            user_agent.strip()
            if user_agent and user_agent.strip()
            else self.DEFAULT_USER_AGENT
        )

        self._closed = False

    # ==========================================================================
    # PROVIDER CONTRACT
    # ==========================================================================

    @property
    def provider_name(self) -> str:
        """Canonical provider identifier."""
        return "virustotal"

    @property
    def supported_ioc_types(self) -> Set[IOCType]:
        """IOC types supported by this provider."""
        return {
            IOCType.URL,
            IOCType.DOMAIN,
            IOCType.IP,
            IOCType.HASH_SHA256,
        }

    # ==========================================================================
    # CLIENT LIFECYCLE
    # ==========================================================================

    async def __aenter__(self) -> "VirusTotalProvider":
        """Enter async provider context."""
        self._ensure_not_closed()
        self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Close provider-owned resources."""
        await self.aclose()

    async def aclose(self) -> None:
        """
        Close the provider-owned HTTP client.

        Caller-supplied clients are intentionally never closed.
        """
        if self._closed:
            return

        self._closed = True

        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("VirusTotalProvider has already been closed.")

    # ==========================================================================
    # PUBLIC QUERY API
    # ==========================================================================

    async def query_url(self, url: str) -> NormalizedThreatReport:
        """
        Query VirusTotal for URL intelligence.
        """
        self._ensure_not_closed()

        normalized_url = self._normalize_url(url)

        url_id = self._encode_url_id(normalized_url)

        endpoint = f"{self.BASE_URL}/urls/{url_id}"

        return await self._execute_query(
            endpoint=endpoint,
            ioc_value=normalized_url,
            ioc_type=IOCType.URL,
        )

    async def query_domain(self, domain: str) -> NormalizedThreatReport:
        """
        Query VirusTotal for domain intelligence.
        """
        self._ensure_not_closed()

        normalized_domain = self.sanitize_domain(domain)

        if not normalized_domain:
            raise ValueError("Domain cannot be empty.")

        if not self._looks_like_domain(normalized_domain):
            raise ValueError("Invalid domain format.")

        endpoint = f"{self.BASE_URL}/domains/{normalized_domain}"

        return await self._execute_query(
            endpoint=endpoint,
            ioc_value=normalized_domain,
            ioc_type=IOCType.DOMAIN,
        )

    async def query_ip(self, ip_address: str) -> NormalizedThreatReport:
        """
        Query VirusTotal for IPv4/IPv6 intelligence.
        """
        self._ensure_not_closed()

        normalized_ip = str(ip_address).strip()

        if not normalized_ip:
            raise ValueError("IP address cannot be empty.")

        if not self.validate_ip(normalized_ip):
            raise ValueError("Invalid IP address format.")

        endpoint = f"{self.BASE_URL}/ip_addresses/{normalized_ip}"

        return await self._execute_query(
            endpoint=endpoint,
            ioc_value=normalized_ip,
            ioc_type=IOCType.IP,
        )

    async def query_sha256(self, sha256_hash: str) -> NormalizedThreatReport:
        """
        Query VirusTotal for a SHA-256 file hash.
        """
        self._ensure_not_closed()

        normalized_hash = sha256_hash.strip().lower()

        if not self.is_sha256(normalized_hash):
            raise ValueError("Invalid SHA-256 hash.")

        endpoint = f"{self.BASE_URL}/files/{normalized_hash}"

        return await self._execute_query(
            endpoint=endpoint,
            ioc_value=normalized_hash,
            ioc_type=IOCType.HASH_SHA256,
        )

    async def check_health(self) -> bool:
        """
        Perform a lightweight authenticated VirusTotal health check.

        Returns
        -------
        bool
            True only when VirusTotal confirms the request successfully.
        """
        self._ensure_not_closed()

        endpoint = f"{self.BASE_URL}/users/me"

        try:
            response = await self._request(
                endpoint=endpoint,
                method="GET",
            )

            return response.status_code == 200

        except (
            ProviderTimeoutError,
            RateLimitExceededError,
            ThreatIntelAPIError,
        ):
            return False

    # ==========================================================================
    # HTTP REQUEST ENGINE
    # ==========================================================================

    async def _execute_query(
        self,
        endpoint: str,
        ioc_value: str,
        ioc_type: IOCType,
    ) -> NormalizedThreatReport:
        """
        Execute a provider query with retries and defensive response handling.
        """
        start_time = time.perf_counter()

        response = await self._request(
            endpoint=endpoint,
            method="GET",
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # ------------------------------------------------------------------
        # UNKNOWN != CLEAN
        # ------------------------------------------------------------------
        #
        # A 404 only means VirusTotal has no record for the IOC.
        # It does NOT prove that the IOC is benign.
        #
        if response.status_code == 404:
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
                raw_tags=[],
                response_time_ms=elapsed_ms,
                cached_response=False,
                raw_response={
                    "status": "not_found",
                    "provider": self.provider_name,
                },
            )

        if response.status_code != 200:
            self._raise_http_error(response)

        payload = self._decode_json_response(response)

        return self._parse_virustotal_payload(
            payload=payload,
            ioc_value=ioc_value,
            ioc_type=ioc_type,
            elapsed_ms=elapsed_ms,
        )

    async def _request(
        self,
        *,
        endpoint: str,
        method: str = "GET",
    ) -> httpx.Response:
        """
        Execute an HTTP request with bounded retries.

        Only transient failures are retried.
        Authentication and validation failures are not retried.
        """

        client = self._get_client()

        headers = self._build_headers()

        timeout = self._build_timeout()

        last_response: Optional[httpx.Response] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(
                    method=method,
                    url=endpoint,
                    headers=headers,
                    timeout=timeout,
                )

                last_response = response

                if response.status_code not in self.TRANSIENT_STATUS_CODES:
                    return response

                if response.status_code == 429:
                    if attempt >= self.max_retries:
                        self._raise_rate_limit(response)

                    delay = self._calculate_retry_delay(
                        response=response,
                        attempt=attempt,
                    )

                    await asyncio.sleep(delay)
                    continue

                if attempt >= self.max_retries:
                    self._raise_http_error(response)

                delay = self._calculate_retry_delay(
                    response=response,
                    attempt=attempt,
                )

                await asyncio.sleep(delay)

            except httpx.TimeoutException as exc:
                if attempt >= self.max_retries:
                    raise ProviderTimeoutError(
                        provider=self.provider_name,
                        timeout_seconds=self.timeout_seconds,
                    ) from exc

                await asyncio.sleep(
                    self._calculate_backoff(attempt)
                )

            except httpx.RequestError as exc:
                if attempt >= self.max_retries:
                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=503,
                        message="VirusTotal request failed.",
                    ) from exc

                await asyncio.sleep(
                    self._calculate_backoff(attempt)
                )

        if last_response is not None:
            self._raise_http_error(last_response)

        raise ThreatIntelAPIError(
            provider=self.provider_name,
            status_code=503,
            message="VirusTotal request failed without a response.",
        )

    # ==========================================================================
    # RESPONSE HANDLING
    # ==========================================================================

    def _decode_json_response(
        self,
        response: httpx.Response,
    ) -> Dict[str, Any]:
        """
        Decode a JSON response defensively.

        A maximum response size is enforced to prevent accidental
        memory exhaustion from unexpected provider responses.
        """

        content_length = response.headers.get("Content-Length")

        if content_length:
            try:
                declared_size = int(content_length)

                if declared_size > self.max_response_bytes:
                    raise ThreatIntelAPIError(
                        provider=self.provider_name,
                        status_code=502,
                        message="VirusTotal response exceeds configured size limit.",
                    )
            except ValueError:
                # Ignore malformed Content-Length and use actual body size.
                pass

        content = response.content

        if len(content) > self.max_response_bytes:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message="VirusTotal response exceeds configured size limit.",
            )

        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message="VirusTotal returned invalid JSON.",
            ) from exc

        if not isinstance(payload, dict):
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message="VirusTotal returned an unexpected JSON structure.",
            )

        return payload

    def _raise_http_error(
        self,
        response: httpx.Response,
    ) -> None:
        """
        Convert HTTP status codes into safe provider exceptions.

        Response bodies are intentionally not copied into exceptions to
        prevent provider payloads from leaking into application logs.
        """

        status_code = int(response.status_code)

        if status_code == 429:
            self._raise_rate_limit(response)

        if status_code in {401, 403}:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=status_code,
                message="VirusTotal authentication or authorization failed.",
            )

        if status_code == 404:
            return

        if 400 <= status_code < 500:
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=status_code,
                message="VirusTotal rejected the request.",
            )

        raise ThreatIntelAPIError(
            provider=self.provider_name,
            status_code=status_code,
            message="VirusTotal service returned a server error.",
        )

    def _raise_rate_limit(
        self,
        response: httpx.Response,
    ) -> None:
        retry_after = self._parse_retry_after(response)

        raise RateLimitExceededError(
            provider=self.provider_name,
            retry_after=retry_after,
        )

    # ==========================================================================
    # VIRUSTOTAL NORMALIZATION
    # ==========================================================================

    def _parse_virustotal_payload(
        self,
        payload: Dict[str, Any],
        ioc_value: str,
        ioc_type: IOCType,
        elapsed_ms: float,
    ) -> NormalizedThreatReport:
        """
        Convert VirusTotal's provider-specific response into the
        normalized PhishGuard schema.
        """

        data = payload.get("data")

        if not isinstance(data, Mapping):
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message="VirusTotal response missing data object.",
            )

        attributes = data.get("attributes")

        if not isinstance(attributes, Mapping):
            raise ThreatIntelAPIError(
                provider=self.provider_name,
                status_code=502,
                message="VirusTotal response missing attributes object.",
            )

        stats = attributes.get("last_analysis_stats", {})

        if not isinstance(stats, Mapping):
            stats = {}

        malicious = self._safe_nonnegative_int(
            stats.get("malicious", 0)
        )

        suspicious = self._safe_nonnegative_int(
            stats.get("suspicious", 0)
        )

        harmless = self._safe_nonnegative_int(
            stats.get("harmless", 0)
        )

        undetected = self._safe_nonnegative_int(
            stats.get("undetected", 0)
        )

        timeout = self._safe_nonnegative_int(
            stats.get("timeout", 0)
        )

        confirmed_timeout = self._safe_nonnegative_int(
            stats.get("confirmed-timeout", 0)
        )

        failure = self._safe_nonnegative_int(
            stats.get("failure", 0)
        )

        total_engines = (
            malicious
            + suspicious
            + harmless
            + undetected
            + timeout
            + confirmed_timeout
            + failure
        )

        positive_detections = malicious + suspicious

        risk_score = self._calculate_risk_score(
            malicious=malicious,
            suspicious=suspicious,
            total_engines=total_engines,
        )

        severity = NormalizedThreatReport.calculate_severity(
            risk_score
        )

        is_malicious = (
            malicious >= 3
            or risk_score >= 0.40
        )

        threat_categories = self._extract_categories(
            attributes.get("categories")
        )

        raw_tags = self._extract_tags(
            attributes.get("tags")
        )

        return NormalizedThreatReport(
            provider_name=self.provider_name,
            ioc_value=ioc_value,
            ioc_type=ioc_type,
            risk_score=risk_score,
            severity=severity,
            is_malicious=is_malicious,
            positive_detections=positive_detections,
            total_engines=total_engines,
            threat_categories=threat_categories,
            raw_tags=raw_tags,
            response_time_ms=max(0.0, float(elapsed_ms)),
            cached_response=False,
            raw_response=self._sanitize_raw_payload(payload),
        )

    # ==========================================================================
    # RISK SCORING
    # ==========================================================================

    @staticmethod
    def _calculate_risk_score(
        *,
        malicious: int,
        suspicious: int,
        total_engines: int,
    ) -> float:
        """
        Calculate normalized VirusTotal risk score.

        Formula:

            (malicious + 0.5 * suspicious) / total_engines

        This is intentionally deterministic and bounded.

        With no engine observations, the provider cannot establish
        a meaningful risk score, therefore 0.0 is returned while
        the caller can distinguish the situation through UNKNOWN
        severity when appropriate.
        """

        if total_engines <= 0:
            return 0.0

        numerator = float(malicious) + (
            0.5 * float(suspicious)
        )

        score = numerator / float(total_engines)

        if not math.isfinite(score):
            return 0.0

        return max(0.0, min(1.0, score))

    # ==========================================================================
    # SAFE FIELD NORMALIZATION
    # ==========================================================================

    @staticmethod
    def _safe_nonnegative_int(value: Any) -> int:
        """
        Convert an untrusted provider field to a safe non-negative integer.
        """
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

        return max(0, number)

    @staticmethod
    def _extract_categories(value: Any) -> list[str]:
        """
        Normalize VirusTotal category mappings.
        """
        values: list[str] = []

        if isinstance(value, Mapping):
            iterable = value.values()
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            iterable = value
        else:
            iterable = []

        for item in iterable:
            if not isinstance(item, str):
                continue

            cleaned = item.strip()

            if cleaned:
                values.append(cleaned)

        return sorted(set(values))

    @staticmethod
    def _extract_tags(value: Any) -> list[str]:
        """
        Normalize VirusTotal tags into deterministic unique strings.
        """
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return []

        tags: list[str] = []

        for item in value:
            if not isinstance(item, str):
                continue

            cleaned = item.strip()

            if cleaned:
                tags.append(cleaned)

        return sorted(set(tags))

    @staticmethod
    def _sanitize_raw_payload(
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Store provider payload while removing obvious credential-like fields.

        The VirusTotal response itself normally does not contain the
        PhishGuard API key, but defensive sanitization prevents accidental
        propagation if upstream payload structures change.
        """

        blocked_keys = {
            "api_key",
            "apikey",
            "x-apikey",
            "authorization",
            "proxy-authorization",
        }

        def sanitize(value: Any) -> Any:
            if isinstance(value, Mapping):
                result: Dict[str, Any] = {}

                for key, child in value.items():
                    key_string = str(key)

                    if key_string.lower() in blocked_keys:
                        result[key_string] = "[REDACTED]"
                    else:
                        result[key_string] = sanitize(child)

                return result

            if isinstance(value, list):
                return [sanitize(item) for item in value]

            if isinstance(value, tuple):
                return [sanitize(item) for item in value]

            return value

        sanitized = sanitize(payload)

        if not isinstance(sanitized, dict):
            return {}

        return sanitized

    # ==========================================================================
    # URL / DOMAIN NORMALIZATION
    # ==========================================================================

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Normalize a URL without altering its security-relevant semantics.

        Only whitespace is removed. The scheme and URL structure are
        otherwise preserved because aggressive canonicalization can turn
        distinct URLs into the same IOC.
        """
        if not isinstance(url, str):
            raise ValueError("URL must be a string.")

        normalized = url.strip()

        if not normalized:
            raise ValueError("URL cannot be empty.")

        if len(normalized) > 8192:
            raise ValueError("URL exceeds maximum supported length.")

        return normalized

    @staticmethod
    def _looks_like_domain(domain: str) -> bool:
        """
        Conservative domain sanity check.

        This is deliberately not a full RFC hostname parser.
        """
        if len(domain) > 253:
            return False

        if any(
            character.isspace()
            for character in domain
        ):
            return False

        if "/" in domain:
            return False

        if domain.startswith(".") or domain.endswith("."):
            return False

        labels = domain.rstrip(".").split(".")

        if len(labels) < 2:
            return False

        for label in labels:
            if not label:
                return False

            if len(label) > 63:
                return False

            if label.startswith("-") or label.endswith("-"):
                return False

            if not all(
                char.isalnum() or char == "-"
                for char in label
            ):
                return False

        return True

    # ==========================================================================
    # HTTP CONFIGURATION
    # ==========================================================================

    def _build_headers(self) -> Dict[str, str]:
        """
        Build request headers.

        The API key is intentionally never included in logs or exception
        messages by this provider.
        """
        return {
            "x-apikey": self.api_key,
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

    def _build_timeout(self) -> httpx.Timeout:
        """
        Build a bounded HTTP timeout object.
        """
        return httpx.Timeout(
            timeout=self.timeout_seconds,
            connect=min(
                self.DEFAULT_CONNECT_TIMEOUT,
                self.timeout_seconds,
            ),
            read=self.timeout_seconds,
            write=min(
                self.DEFAULT_WRITE_TIMEOUT,
                self.timeout_seconds,
            ),
            pool=min(
                self.DEFAULT_POOL_TIMEOUT,
                self.timeout_seconds,
            ),
        )

    def _get_client(self) -> httpx.AsyncClient:
        """
        Return caller-owned or provider-owned HTTP client.
        """
        self._ensure_not_closed()

        if self._external_client is not None:
            return self._external_client

        if self._owned_client is None:
            self._owned_client = httpx.AsyncClient(
                base_url="",
                follow_redirects=False,
                verify=True,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )

        return self._owned_client

    # ==========================================================================
    # RETRY POLICY
    # ==========================================================================

    def _calculate_backoff(
        self,
        attempt: int,
    ) -> float:
        """
        Calculate capped exponential backoff.

        No jitter is used here so tests remain deterministic.
        Production orchestration can add external jitter if required.
        """
        delay = self.backoff_base_seconds * (
            2 ** max(0, attempt)
        )

        return min(
            self.max_backoff_seconds,
            max(0.0, delay),
        )

    def _calculate_retry_delay(
        self,
        *,
        response: httpx.Response,
        attempt: int,
    ) -> float:
        """
        Prefer provider Retry-After when valid; otherwise use backoff.
        """
        retry_after = self._parse_retry_after(response)

        if retry_after is not None:
            return min(
                self.max_backoff_seconds,
                max(0.0, float(retry_after)),
            )

        return self._calculate_backoff(attempt)

    @staticmethod
    def _parse_retry_after(
        response: httpx.Response,
    ) -> Optional[int]:
        """
        Parse a safe integer Retry-After value.

        HTTP-date based Retry-After is deliberately ignored here because
        clock-dependent parsing is unnecessary for the provider's bounded
        retry strategy.
        """
        value = response.headers.get("Retry-After")

        if value is None:
            return None

        try:
            parsed = int(value.strip())
        except (TypeError, ValueError):
            return None

        if parsed < 0:
            return None

        return parsed

    # ==========================================================================
    # URL IDENTIFIER
    # ==========================================================================

    @staticmethod
    def _encode_url_id(url: str) -> str:
        """
        Convert URL to VirusTotal's URL identifier.

        VirusTotal v3 uses URL-safe Base64 without '=' padding.
        """
        if not isinstance(url, str):
            raise ValueError("URL must be a string.")

        normalized = url.strip()

        if not normalized:
            raise ValueError("URL cannot be empty.")

        encoded = base64.urlsafe_b64encode(
            normalized.encode("utf-8")
        ).decode("ascii")

        return encoded.rstrip("=")