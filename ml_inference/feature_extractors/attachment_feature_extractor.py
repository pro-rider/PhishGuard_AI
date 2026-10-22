"""
PhishGuard_AI - Attachment Feature Extractor

Extracts bounded numeric features from email attachment payloads for the ML
inference pipeline.

Security / reliability principles:
- Features are computed from bounded metadata and in-memory buffers only.
- All values are finite and clamped by the base extractor's sanitizer.
- No file content is executed or logged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from engine.orchestrator import ScanContext, ScanType
from ml_inference.feature_extractors.base_extractor import (
    BaseFeatureExtractor,
    FeatureDefinition,
)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

EXTRACTOR_NAME = "attachment_feature_extractor"
EXTRACTOR_VERSION = "1.0.0"
MAX_NAME_LENGTH = 256
MAX_ENTROPY_WINDOW = 1 << 16

_DANGEROUS_EXTENSIONS = frozenset(
    {".exe", ".dll", ".bat", ".cmd", ".com", ".scr", ".vbs", ".js", ".msi", ".hta"}
)
_DOCUMENT_EXTENSIONS = frozenset(
    {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".rtf"}
)


class AttachmentFeatureExtractor(BaseFeatureExtractor):
    """Numeric feature extractor for email attachments."""

    name: str = EXTRACTOR_NAME
    version: str = EXTRACTOR_VERSION

    def __init__(self) -> None:
        super().__init__()
        self._schema: Mapping[str, FeatureDefinition] = self._build_feature_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _build_feature_schema(self) -> Mapping[str, FeatureDefinition]:
        return {
            "attachment_count": FeatureDefinition(
                name="attachment_count", default=0.0, minimum=0.0, maximum=100.0
            ),
            "has_attachment": FeatureDefinition(
                name="has_attachment", default=0.0, minimum=0.0, maximum=1.0
            ),
            "has_dangerous_extension": FeatureDefinition(
                name="has_dangerous_extension", default=0.0, minimum=0.0, maximum=1.0
            ),
            "has_document_extension": FeatureDefinition(
                name="has_document_extension", default=0.0, minimum=0.0, maximum=1.0
            ),
            "total_attachment_bytes": FeatureDefinition(
                name="total_attachment_bytes",
                default=0.0,
                minimum=0.0,
                maximum=1.0e9,
            ),
            "max_entropy": FeatureDefinition(
                name="max_entropy",
                default=0.0,
                minimum=0.0,
                maximum=8.0,
            ),
            "archive_present": FeatureDefinition(
                name="archive_present", default=0.0, minimum=0.0, maximum=1.0
            ),
            "macros_present": FeatureDefinition(
                name="macros_present", default=0.0, minimum=0.0, maximum=1.0
            ),
        }

    def supports_scan_type(self, scan_type: ScanType) -> bool:
        return scan_type in (ScanType.ATTACHMENT, ScanType.COMPREHENSIVE)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_features(self, context: ScanContext) -> Dict[str, Any]:
        """Compute attachment features from the scan context."""
        attachments = self._resolve_attachments(context)
        if not attachments:
            return dict(self._defaults())

        features: Dict[str, Any] = dict(self._defaults())
        features["attachment_count"] = float(len(attachments))
        features["has_attachment"] = 1.0

        total_bytes = 0.0
        max_entropy = 0.0
        for attachment in attachments:
            filename = str(attachment.get("filename") or "")
            extension = self._extension(filename)
            total_bytes += float(attachment.get("size_bytes", 0) or 0)

            if extension in _DANGEROUS_EXTENSIONS:
                features["has_dangerous_extension"] = 1.0
            if extension in _DOCUMENT_EXTENSIONS:
                features["has_document_extension"] = 1.0

            mime = str(attachment.get("mime_type") or "")
            if "zip" in mime or "rar" in mime or "7z" in mime:
                features["archive_present"] = 1.0

            content = attachment.get("content")
            if isinstance(content, (bytes, bytearray)):
                max_entropy = max(
                    max_entropy,
                    self._shannon_entropy(bytes(content)[:MAX_ENTROPY_WINDOW]),
                )

        features["total_attachment_bytes"] = min(1.0e9, total_bytes)
        features["max_entropy"] = max_entropy
        features["macros_present"] = 1.0 if max_entropy > 5.5 else 0.0
        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_attachments(context: ScanContext) -> list:
        """Resolve attachment descriptors from the payload reference or metadata."""
        if context.attachment_payload is not None:
            return [
                {
                    "filename": context.attachment_payload.filename,
                    "size_bytes": context.attachment_payload.size_bytes,
                    "mime_type": context.attachment_payload.mime_type,
                    "content": None,
                }
            ]
        metadata = context.metadata.data or {}
        raw = metadata.get("attachments")
        if isinstance(raw, list):
            return [a for a in raw if isinstance(a, dict)]
        return []

    def _defaults(self) -> Dict[str, Any]:
        return {
            name: definition.default for name, definition in self._schema.items()
        }

    @staticmethod
    def _extension(filename: str) -> str:
        if not filename or "." not in filename:
            return ""
        return (filename.rsplit(".", 1)[-1].lower())[:32]

    @staticmethod
    def _shannon_entropy(data: bytes) -> float:
        if not data:
            return 0.0
        counts = [0] * 256
        for byte in data:
            counts[byte] += 1
        length = len(data)
        entropy = 0.0
        for count in counts:
            if count == 0:
                continue
            p = count / length
            entropy -= p * math.log2(p)
        return entropy


attachment_feature_extractor = AttachmentFeatureExtractor()


__all__ = [
    "AttachmentFeatureExtractor",
    "attachment_feature_extractor",
    "EXTRACTOR_NAME",
    "EXTRACTOR_VERSION",
]
