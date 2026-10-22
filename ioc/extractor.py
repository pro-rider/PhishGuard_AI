"""
PhishGuard_AI - Unified IOC Extraction Facade

Aggregates the URL, IP and hash extractors behind a single entry point so a
caller can pull all indicator types from a blob of untrusted text in one pass.

Security / reliability principles:
- Each sub-extractor runs independently and is individually bounded.
- A failure in one extractor type never prevents the others from running.
- Outputs are aggregated with a stable, de-duplicated IOC id per entity.
- No network, filesystem, or subprocess operation is performed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ioc.hash_extractor import HashExtractor, HashIOC
from ioc.ip_extractor import IPExtractor, IPExtractionResult
from ioc.url_extractor import URLExtraction, URLExtractor

# ============================================================================
# AGGREGATED RESULT
# ============================================================================


class AggregatedIOCOutput(BaseModel):
    """Combined output of all IOC extractors."""

    model_config = ConfigDict(extra="forbid")

    urls: List[URLExtraction] = Field(default_factory=list)
    ips: List[Any] = Field(default_factory=list)
    hashes: List[HashIOC] = Field(default_factory=list)
    total_count: int = 0
    truncated: bool = False
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# UNIFIED EXTRACTOR
# ============================================================================


class IOCExtractor:
    """
    Unified IOC extraction entry point.

    Runs URL, IP and hash extraction over the same input and returns an
    aggregated, typed result. Each sub-extractor uses its own safe defaults and
    can be overridden via the constructor.
    """

    def __init__(
        self,
        *,
        url_extractor: Optional[URLExtractor] = None,
        ip_extractor: Optional[IPExtractor] = None,
        hash_extractor: Optional[HashExtractor] = None,
    ) -> None:
        self.url_extractor = url_extractor or URLExtractor()
        self.ip_extractor = ip_extractor or IPExtractor()
        self.hash_extractor = hash_extractor or HashExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> AggregatedIOCOutput:
        """
        Extract all supported IOC types from ``text``.

        Raises :class:`TypeError` for non-string input. Per-extractor failures
        are captured as warnings rather than propagated, keeping the facade
        resilient.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return AggregatedIOCOutput()

        warnings: List[str] = []
        urls: List[URLExtraction] = []
        ips: List[Any] = []
        hashes: List[HashIOC] = []

        try:
            urls = self.url_extractor.extract(text)
        except Exception as exc:  # noqa: BLE001 - facade resilience
            warnings.append(f"url extraction failed: {exc.__class__.__name__}")

        try:
            ip_result = self.ip_extractor.extract(text)
            if isinstance(ip_result, IPExtractionResult):
                ips = list(ip_result.iocs)
            elif isinstance(ip_result, list):
                ips = ip_result
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"ip extraction failed: {exc.__class__.__name__}")

        try:
            hashes = self.hash_extractor.extract(text)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"hash extraction failed: {exc.__class__.__name__}")

        total = len(urls) + len(ips) + len(hashes)
        return AggregatedIOCOutput(
            urls=urls,
            ips=ips,
            hashes=hashes,
            total_count=total,
            truncated=len(hashes) >= self.hash_extractor.config.max_iocs,
            warnings=warnings,
        )

    def extract_text(
        self, text: str, *, kinds: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Return raw indicator values grouped by kind (url/ip/hash)."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return {"url": [], "ip": [], "hash": []}

        kinds = kinds or ["url", "ip", "hash"]
        result: Dict[str, List[str]] = {"url": [], "ip": [], "hash": []}

        if "url" in kinds:
            try:
                result["url"] = [
                    u.raw_value for u in self.url_extractor.extract(text)
                ]
            except Exception:  # noqa: BLE001
                result["url"] = []

        if "ip" in kinds:
            try:
                ip_result = self.ip_extractor.extract(text)
                values = (
                    list(ip_result.iocs)
                    if isinstance(ip_result, IPExtractionResult)
                    else ip_result
                )
                result["ip"] = [
                    getattr(v, "raw_value", None)
                    or getattr(v, "value", None)
                    or str(v)
                    for v in values
                    if v is not None
                ]
            except Exception:  # noqa: BLE001
                result["ip"] = []

        if "hash" in kinds:
            try:
                result["hash"] = [
                    h.digest for h in self.hash_extractor.extract(text)
                ]
            except Exception:  # noqa: BLE001
                result["hash"] = []

        return result


# ============================================================================
# MODULE-LEVEL CONVENIENCE
# ============================================================================

_default_extractor = IOCExtractor()


def extract_iocs(text: str) -> AggregatedIOCOutput:
    """Extract all IOC types from ``text`` using the shared extractor."""
    return _default_extractor.extract(text)


def extract_urls(text: str) -> List[URLExtraction]:
    """Extract validated URLs from ``text``."""
    return _default_extractor.url_extractor.extract(text)


def extract_hashes(text: str) -> List[HashIOC]:
    """Extract validated hash IOCs from ``text``."""
    return _default_extractor.hash_extractor.extract(text)


__all__ = [
    "AggregatedIOCOutput",
    "IOCExtractor",
    "extract_iocs",
    "extract_urls",
    "extract_hashes",
]
