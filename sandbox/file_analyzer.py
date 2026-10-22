"""
PhishGuard_AI - Attachment Static File Analyzer

Performs bounded, side-effect-free static analysis of email attachments.

Security / reliability principles:
- Files are analyzed in memory; they are never executed.
- Magic-byte probing, size checks and macro-heuristic detection are bounded.
- Archive/container traversal is depth- and count-limited.
- Malformed or oversized inputs are rejected fail-closed.
"""

from __future__ import annotations

import enum
import io
import struct
import zlib
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024  # 64 MiB
DEFAULT_SNIFF_BYTES = 4096
DEFAULT_MAX_ARCHIVE_ENTRIES = 512
DEFAULT_MAX_ENTRY_BYTES = 16 * 1024 * 1024

# Known magic-byte signatures for common attachment types.
_MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"%PDF", "application/pdf"),
    (b"\x25\x50\x44\x46", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),
    (b"PK\x05\x06", "application/zip"),
    (b"\x1f\x8b\x08", "application/gzip"),
    (b"Rar!\x1a\x07", "application/x-rar-compressed"),
    (b"\x7fELF", "application/x-executable"),
    (b"MZ", "application/x-ms-dos-executable"),
    (b"OLE\x00\x02", "application/x-ole-storage"),  # old .doc/.xls/.ppt
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/x-ole-storage"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"\x00\x00\x01\x00", "image/x-icon"),
    (b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed"),
    (b"\x1f\x8b", "application/gzip"),
)

# Extensions we refuse to analyze statically (fail-closed unknowns allowed).
_DANGEROUS_EXTENSIONS = frozenset(
    {
        ".exe", ".dll", ".bat", ".cmd", ".com", ".scr", ".pif",
        ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".msi",
        ".hta", ".reg", ".ps1", ".psm1", ".scr",
    }
)

# Macro / scripting indicators scanned in text content.
_MACRO_INDICATORS = (
    "sub auto_open",
    "sub autoexec",
    "shell(",
    "powershell",
    "cmd.exe",
    "wscript",
    "mshta",
    "rundll32",
    "regsvr32",
    "createobject",
    "wscript.shell",
    "cscript",
    "certutil",
    "bitsadmin",
)


class FileVerdict(str, enum.Enum):
    """Static verdict for an analyzed file."""

    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class FileAnalyzerError(RuntimeError):
    """Base error for file analysis."""


class FileTooLargeError(FileAnalyzerError):
    """File exceeds the configured size bound."""


@dataclass(frozen=True, slots=True)
class StaticAnalysisResult:
    """Outcome of a static attachment analysis."""

    filename: str
    size_bytes: int
    detected_type: str
    extension: str
    verdict: FileVerdict
    indicators: Tuple[str, ...] = ()
    score: float = 0.0
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "detected_type": self.detected_type,
            "extension": self.extension,
            "verdict": self.verdict.value,
            "indicators": list(self.indicators),
            "score": self.score,
            "truncated": self.truncated,
        }


# ============================================================================
# ANALYZER
# ============================================================================


class FileAnalyzer:
    """
    Static, in-memory file analyzer.

    Detects type via magic bytes, bounds size, inspects containers, and flags
    macro / scripting indicators. Never executes file content.
    """

    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        sniff_bytes: int = DEFAULT_SNIFF_BYTES,
    ) -> None:
        self._max_file_bytes = max_file_bytes
        self._sniff_bytes = sniff_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, filename: str, data: bytes) -> StaticAnalysisResult:
        """Analyze ``data`` and return a static verdict."""
        filename = (filename or "unnamed")[:512]
        extension = self._extension(filename).lower()

        if len(data) > self._max_file_bytes:
            raise FileTooLargeError(
                f"File exceeds {self._max_file_bytes} byte bound."
            )
        if not data:
            return StaticAnalysisResult(
                filename=filename,
                size_bytes=0,
                detected_type="application/octet-stream",
                extension=extension,
                verdict=FileVerdict.UNKNOWN,
                indicators=("empty file",),
                score=0.1,
            )

        indicators: List[str] = []
        score = 0.0
        detected_type = self._detect_type(data)

        if extension in _DANGEROUS_EXTENSIONS:
            indicators.append("dangerous_extension")
            score += 0.45

        if detected_type == "application/x-ole-storage":
            indicators.append("ole_storage_document")
            score += 0.10

        if detected_type in ("application/zip", "application/x-rar-compressed"):
            archive_issues = self._inspect_archive(data, detected_type)
            indicators.extend(archive_issues)
            score += 0.08 * len(archive_issues)

        # Text / script content scan for macro indicators.
        macro_hits = self._scan_text_indicators(data)
        indicators.extend(macro_hits)
        score += 0.15 * len(macro_hits)

        verdict = self._classify(score)
        truncated = len(data) > self._sniff_bytes and detected_type == "application/octet-stream"

        return StaticAnalysisResult(
            filename=filename,
            size_bytes=len(data),
            detected_type=detected_type,
            extension=extension,
            verdict=verdict,
            indicators=tuple(indicators)[:32],
            score=min(1.0, score),
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extension(filename: str) -> str:
        if "." not in filename:
            return ""
        return filename.rsplit(".", 1)[-1]

    def _detect_type(self, data: bytes) -> str:
        head = data[: self._sniff_bytes]
        for signature, mime in _MAGIC_SIGNATURES:
            if head.startswith(signature):
                return mime
        return "application/octet-stream"

    def _inspect_archive(self, data: bytes, detected_type: str) -> List[str]:
        issues: List[str] = []
        if detected_type == "application/zip":
            try:
                import zipfile

                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    infos = archive.infolist()
                    if len(infos) > DEFAULT_MAX_ARCHIVE_ENTRIES:
                        issues.append("archive_entry_count_exceeded")
                    suspicious = [
                        info
                        for info in infos
                        if self._extension(info.filename) in _DANGEROUS_EXTENSIONS
                    ]
                    if suspicious:
                        issues.append("archive_contains_executable")
                    for info in infos:
                        if info.file_size > DEFAULT_MAX_ENTRY_BYTES:
                            issues.append("archive_entry_oversized")
                            break
            except Exception:  # noqa: BLE001
                issues.append("archive_parse_failed")
        elif detected_type == "application/x-rar-compressed":
            issues.append("rar_archive_needs_sandbox")
        return issues

    @staticmethod
    def _scan_text_indicators(data: bytes) -> List[str]:
        hits: List[str] = []
        try:
            text = data[: 1 << 20].decode("utf-8", errors="ignore").lower()
        except Exception:  # noqa: BLE001
            return hits
        for indicator in _MACRO_INDICATORS:
            if indicator in text:
                hits.append(f"macro:{indicator.replace(' ', '_')}")
        return hits[:8]

    @staticmethod
    def _classify(score: float) -> FileVerdict:
        if score >= 0.6:
            return FileVerdict.MALICIOUS
        if score >= 0.3:
            return FileVerdict.SUSPICIOUS
        if score > 0.0:
            return FileVerdict.UNKNOWN
        return FileVerdict.BENIGN


# ============================================================================
# SINGLETON
# ============================================================================

file_analyzer = FileAnalyzer()


__all__ = [
    "FileVerdict",
    "StaticAnalysisResult",
    "FileAnalyzer",
    "FileAnalyzerError",
    "FileTooLargeError",
    "file_analyzer",
]
