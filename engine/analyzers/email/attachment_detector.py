"""
PhishGuard_AI - Email Attachment Security Analyzer
Production-grade sub-analyzer inspecting email attachments for dangerous
extensions, magic-byte / extension mismatches, double-extension deception,
ZipSlip, zip-bombs, nested archives, and Office macro containers.

Key Security Controls
---------------------
1. Safe PayloadReference resolution with hard size limits.
2. Header-only magic-byte engine (PE, ELF, Mach-O, PDF, ZIP, OLE, RAR, 7z, ISO…).
3. In-memory ZIP inspector with entry limits, path-traversal, ratio and nested checks.
4. Double-extension + RTL/Unicode filename deception detection.
5. Office Open XML / OLE macro container heuristics.
6. Calibrated per-category scoring with damping.
7. Strict multi-phase deadline enforcement.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.orchestrator import (
    AnalyzerExecutionStatus,
    AnalyzerResourceClass,
    AnalyzerResult,
    BaseAnalyzer,
    ConstrainedMetadata,
    Finding,
    PayloadReference,
    ScanContext,
    ScanType,
    SecurityVerdict,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & BOUNDS
# ============================================================================

MAX_ATTACHMENT_SIZE_BYTES = 25 * 1024 * 1024          # 25 MB
MAX_HEADER_READ = 512                                 # bytes needed for magic
MAX_ARCHIVE_BYTES_TO_INSPECT = 8 * 1024 * 1024        # 8 MB cap for ZIP inspection
MAX_UNCOMPRESSED_DECLARED = 150 * 1024 * 1024         # 150 MB
MAX_DECOMPRESSION_RATIO = 80.0
MAX_ARCHIVE_ENTRIES = 400
MAX_NESTED_DEPTH = 2
MAX_ATTACHMENTS_TO_PROCESS = 30
MAX_FINDINGS = 20

# High-risk extensions (executables, scripts, containers, macro-enabled Office)
HIGH_RISK_EXTENSIONS: Set[str] = {
    # Native executables / libraries
    ".exe", ".dll", ".scr", ".com", ".msi", ".msp", ".mst",
    ".sys", ".cpl", ".drv", ".ocx", ".ax", ".efi",
    # Scripts
    ".bat", ".cmd", ".ps1", ".psm1", ".psd1", ".vbs", ".vbe",
    ".js", ".jse", ".wsf", ".wsh", ".hta", ".py", ".pyc",
    ".sh", ".bash", ".php", ".pl", ".rb", ".jar",
    # Shortcuts & LOLBins helpers
    ".lnk", ".pif", ".inf", ".scf", ".url", ".library-ms",
    ".search-ms", ".settingcontent-ms", ".appcontent-ms",
    # Disk images / containers
    ".iso", ".img", ".vhd", ".vhdx", ".vmdk", ".dmg",
    # Macro-enabled / old Office
    ".docm", ".xlsm", ".pptm", ".dotm", ".xltm", ".potm",
    ".doc", ".xls", ".ppt",               # still frequently used for macros
    ".rtf", ".pub", ".one", ".chm", ".mht", ".mhtml",
}

# Extensions that look “safe” and are often used for double-extension tricks
DOCUMENT_LIKE_EXTENSIONS: Set[str] = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".csv", ".rtf", ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".svg", ".html", ".htm",
}

# Magic signatures (order matters – longer / more specific first where needed)
MAGIC_SIGNATURES: List[Tuple[bytes, str]] = [
    (b"%PDF", "PDF"),
    (b"PK\x03\x04", "ZIP"),
    (b"PK\x05\x06", "ZIP_EMPTY"),
    (b"PK\x07\x08", "ZIP_SPANNED"),
    (b"\x7fELF", "ELF"),
    (b"MZ", "PE"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "OLE"),          # classic Office / MSI
    (b"Rar!\x1a\x07\x00", "RAR4"),
    (b"Rar!\x1a\x07\x01\x00", "RAR5"),
    (b"7z\xbc\xaf\x27\x1c", "7Z"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"\x00\x00\x01\x00", "ICO"),
    (b"\xca\xfe\xba\xbe", "MACH_O_FAT"),
    (b"\xcf\xfa\xed\xfe", "MACH_O_64"),
    (b"\xce\xfa\xed\xfe", "MACH_O_32"),
    (b"\xbe\xba\xfe\xca", "MACH_O_FAT_BE"),
    (b"CD001", "ISO9660"),                                 # offset 0x8001 usually, but we check header too
    (b"\x00\x00\x00\x18ftyp", "MP4"),
    (b"{\\rtf", "RTF"),
]

# RTL / bidi overrides that can hide the real extension
BIDI_OVERRIDE_RE = re.compile(r"[\u202A-\u202E\u2066-\u2069]")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(frozen=True, slots=True)
class AttachmentMeta:
    filename: str
    declared_mime: str
    size_bytes: int
    payload_ref: Optional[PayloadReference] = None
    # Filled only when we decide to read the header / limited body
    header: bytes = b""
    limited_body: Optional[bytes] = None

# ============================================================================
# HELPERS
# ============================================================================

def safe_filename(name: str) -> str:
    """Normalise and strip dangerous Unicode from filenames."""
    if not name:
        return "unnamed"
    # Strip bidi overrides that hide the real extension
    name = BIDI_OVERRIDE_RE.sub("", name)
    # NFC normalisation
    try:
        import unicodedata
        name = unicodedata.normalize("NFC", name)
    except Exception:
        pass
    return name.strip() or "unnamed"

def get_extensions(filename: str) -> Tuple[str, List[str]]:
    """
    Return (final_ext, all_extensions_from_right).
    Example: 'report.pdf.exe' → ('.exe', ['.exe', '.pdf'])
    """
    base = os.path.basename(filename)
    parts = base.split(".")
    if len(parts) < 2:
        return "", []
    exts = ["." + p.lower() for p in parts[1:] if p]
    final = exts[-1] if exts else ""
    return final, list(reversed(exts))   # right-most first

def detect_magic(header: bytes) -> Optional[str]:
    if not header:
        return None
    for sig, name in MAGIC_SIGNATURES:
        if header.startswith(sig):
            return name
    # Special case: ISO often has CD001 at offset 0x8001
    if len(header) >= 0x8006 and header[0x8001:0x8006] == b"CD001":
        return "ISO9660"
    return None

def is_executable_magic(magic: Optional[str]) -> bool:
    return magic in {"PE", "ELF", "MACH_O_64", "MACH_O_32", "MACH_O_FAT", "MACH_O_FAT_BE"}

# ============================================================================
# SAFE ARCHIVE INSPECTOR
# ============================================================================

class SafeArchiveInspector:
    """In-memory ZIP inspector with hard limits."""

    @classmethod
    def inspect(
        cls,
        data: bytes,
        *,
        depth: int = 0,
        deadline: float,
    ) -> List[str]:
        msgs: List[str] = []
        if depth > MAX_NESTED_DEPTH:
            msgs.append(f"Nested archive depth exceeded ({depth}).")
            return msgs
        if time.monotonic() >= deadline:
            msgs.append("Archive inspection aborted (deadline).")
            return msgs

        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                infos = zf.infolist()
                if len(infos) > MAX_ARCHIVE_ENTRIES:
                    msgs.append(f"Excessive archive entries ({len(infos)}).")
                    return msgs

                total_comp = 0
                total_uncomp = 0
                for info in infos:
                    total_comp += max(0, info.compress_size)
                    total_uncomp += max(0, info.file_size)

                    # Path traversal / absolute paths
                    name = info.filename
                    if (
                        ".." in name
                        or name.startswith(("/", "\\"))
                        or re.match(r"^[a-zA-Z]:[\\/]", name)
                        or "\x00" in name
                    ):
                        msgs.append(f"Path-traversal / absolute path: '{name[:80]}'")

                    # High-risk member
                    member_ext, _ = get_extensions(name)
                    if member_ext in HIGH_RISK_EXTENSIONS:
                        msgs.append(f"High-risk member inside archive: '{name[:80]}'")

                    # Nested ZIP (simple heuristic)
                    if member_ext in {".zip", ".jar", ".apk", ".docx", ".xlsx", ".pptx"} and depth < MAX_NESTED_DEPTH:
                        # We do not extract; just note the nesting
                        msgs.append(f"Nested archive member detected: '{name[:80]}'")

                if total_uncomp > MAX_UNCOMPRESSED_DECLARED:
                    msgs.append(
                        f"Declared uncompressed size too large ({total_uncomp} bytes)."
                    )
                if total_comp > 0:
                    ratio = total_uncomp / float(total_comp)
                    if ratio > MAX_DECOMPRESSION_RATIO:
                        msgs.append(
                            f"Suspicious compression ratio {ratio:.1f}:1 (possible zip-bomb)."
                        )

        except zipfile.BadZipFile:
            msgs.append("Malformed or corrupted ZIP structure.")
        except Exception as exc:
            msgs.append(f"Archive inspection error: {type(exc).__name__}")

        return msgs

# ============================================================================
# PAYLOAD RESOLUTION
# ============================================================================

class EmailPayloadStore:
    """Resolve attachment metadata + limited payload bytes safely."""

    @staticmethod
    def resolve(context: ScanContext) -> List[AttachmentMeta]:
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}
        raw_list = data.get("attachments") or data.get("email_attachments") or []

        results: List[AttachmentMeta] = []
        if not isinstance(raw_list, list):
            return results

        for item in raw_list[:MAX_ATTACHMENTS_TO_PROCESS]:
            if not isinstance(item, dict):
                continue

            filename = safe_filename(str(item.get("filename") or item.get("name") or "unnamed"))
            mime = str(item.get("mime_type") or item.get("content_type") or "application/octet-stream")
            size = int(item.get("size") or item.get("size_bytes") or 0)

            # Prefer PayloadReference
            pref: Optional[PayloadReference] = None
            for key in ("payload_ref", "content_ref", "ref", "payload"):
                candidate = item.get(key)
                if isinstance(candidate, PayloadReference):
                    pref = candidate
                    break

            # Inline bytes (less preferred, size-capped later)
            inline = item.get("payload_bytes") or item.get("content") or item.get("data")
            header = b""
            limited: Optional[bytes] = None

            if pref is not None:
                try:
                    if hasattr(pref, "read"):
                        header = pref.read(max_bytes=MAX_HEADER_READ) or b""  # type: ignore[call-arg]
                        if isinstance(header, str):
                            header = header.encode("utf-8", errors="replace")
                    elif hasattr(pref, "content"):
                        raw = pref.content
                        if isinstance(raw, (bytes, bytearray)):
                            header = bytes(raw[:MAX_HEADER_READ])
                except Exception as exc:
                    logger.debug("Failed reading attachment header: %s", type(exc).__name__)
            elif isinstance(inline, (bytes, bytearray)):
                header = bytes(inline[:MAX_HEADER_READ])
                # Only keep a limited body for ZIP inspection
                if len(inline) <= MAX_ARCHIVE_BYTES_TO_INSPECT:
                    limited = bytes(inline)
                else:
                    limited = bytes(inline[:MAX_ARCHIVE_BYTES_TO_INSPECT])

            results.append(
                AttachmentMeta(
                    filename=filename,
                    declared_mime=mime,
                    size_bytes=size,
                    payload_ref=pref,
                    header=header,
                    limited_body=limited,
                )
            )
        return results

# ============================================================================
# MAIN ANALYZER
# ============================================================================

class AttachmentDetector(BaseAnalyzer):
    """Production email attachment threat analyzer."""

    name: str = "attachment_detector"
    version: str = "2.3.0"
    supported_scan_types: Set[ScanType] = {ScanType.EMAIL, ScanType.COMPREHENSIVE}
    required_inputs: Set[str] = {"email"}
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.CPU_BOUND
    priority: int = 12
    can_hard_override: bool = False

    async def analyze(self, context: ScanContext, deadline_monotonic: float) -> AnalyzerResult:
        start = time.monotonic()

        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        attachments = EmailPayloadStore.resolve(context)
        if not attachments:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.90,
                findings=[],
            )

        findings: List[Finding] = []
        cat_scores: Dict[str, float] = {
            "MALWARE_PAYLOAD": 0.0,
            "DECEPTION": 0.0,
            "OBFUSCATION": 0.0,
            "MACRO_RISK": 0.0,
        }

        def emit(
            rule_id: str,
            category: str,
            severity: SecurityVerdict,
            confidence: float,
            message: str,
            points: float,
            meta: Optional[Dict[str, Any]] = None,
        ) -> None:
            if len(findings) >= MAX_FINDINGS:
                return
            # light damping
            damp = 0.75 ** sum(1 for f in findings if f.rule_id == rule_id)
            cat_scores[category] = cat_scores.get(category, 0.0) + points * damp
            findings.append(
                Finding(
                    rule_id=rule_id,
                    category=category,
                    severity=severity,
                    confidence=confidence,
                    message=message,
                    metadata=ConstrainedMetadata(meta or {}),
                )
            )

        try:
            for att in attachments:
                if time.monotonic() >= deadline_monotonic:
                    return self._timeout(start)

                if att.size_bytes > MAX_ATTACHMENT_SIZE_BYTES:
                    return AnalyzerResult(
                        analyzer_name=self.name,
                        status=AnalyzerExecutionStatus.INVALID_INPUT,
                        execution_time_ms=(time.monotonic() - start) * 1000.0,
                        sanitized_error=(
                            f"Attachment '{att.filename[:48]}' exceeds size limit "
                            f"({att.size_bytes} bytes)."
                        ),
                    )

                filename = att.filename
                final_ext, all_exts = get_extensions(filename)
                magic = detect_magic(att.header)

                # ----- 1. High-risk extension -----
                if final_ext in HIGH_RISK_EXTENSIONS:
                    emit(
                        "EMAIL.ATTACHMENT.HIGH_RISK_EXTENSION",
                        "MALWARE_PAYLOAD",
                        SecurityVerdict.HIGH,
                        0.96,
                        f"High-risk extension '{final_ext}' on '{filename[:64]}'.",
                        38.0,
                        {"filename": filename[:64], "extension": final_ext},
                    )

                # ----- 2. Double-extension / multi-extension deception -----
                if len(all_exts) >= 2:
                    # e.g. .pdf.exe, .docx.js, .txt.scr
                    if final_ext in HIGH_RISK_EXTENSIONS and any(
                        e in DOCUMENT_LIKE_EXTENSIONS for e in all_exts[1:]
                    ):
                        emit(
                            "EMAIL.ATTACHMENT.DOUBLE_EXTENSION_DECEPTION",
                            "DECEPTION",
                            SecurityVerdict.HIGH,
                            0.95,
                            f"Double-extension deception: '{filename[:64]}'.",
                            34.0,
                            {"filename": filename[:64], "extensions": all_exts[:4]},
                        )

                # ----- 3. RTL / bidi filename trick -----
                if BIDI_OVERRIDE_RE.search(att.filename):
                    emit(
                        "EMAIL.ATTACHMENT.BIDI_FILENAME_OBFUSCATION",
                        "DECEPTION",
                        SecurityVerdict.HIGH,
                        0.93,
                        f"Bidirectional override characters in filename '{filename[:64]}'.",
                        30.0,
                        {"filename": filename[:64]},
                    )

                # ----- 4. Magic vs extension mismatch -----
                if magic and is_executable_magic(magic):
                    if final_ext not in HIGH_RISK_EXTENSIONS:
                        emit(
                            "EMAIL.ATTACHMENT.MAGIC_EXTENSION_MISMATCH",
                            "MALWARE_PAYLOAD",
                            SecurityVerdict.CRITICAL,
                            0.98,
                            f"Executable magic ({magic}) but non-executable extension "
                            f"'{final_ext}' on '{filename[:64]}'.",
                            52.0,
                            {
                                "filename": filename[:64],
                                "claimed_ext": final_ext,
                                "detected_magic": magic,
                            },
                        )

                # OLE / classic Office → high macro risk
                if magic == "OLE":
                    emit(
                        "EMAIL.ATTACHMENT.OLE_CONTAINER",
                        "MACRO_RISK",
                        SecurityVerdict.HIGH,
                        0.90,
                        f"OLE compound file (classic Office / MSI) detected: '{filename[:64]}'.",
                        28.0,
                        {"filename": filename[:64], "magic": magic},
                    )

                # ZIP-based Office Open XML with macro-enabled extension
                if magic in {"ZIP", "ZIP_EMPTY"} and final_ext in {
                    ".docm", ".xlsm", ".pptm", ".dotm", ".xltm", ".potm",
                }:
                    emit(
                        "EMAIL.ATTACHMENT.MACRO_ENABLED_OOXML",
                        "MACRO_RISK",
                        SecurityVerdict.HIGH,
                        0.92,
                        f"Macro-enabled Office Open XML: '{filename[:64]}'.",
                        30.0,
                        {"filename": filename[:64], "extension": final_ext},
                    )

                # ----- 5. Archive inspection -----
                if magic in {"ZIP", "ZIP_EMPTY", "ZIP_SPANNED"} or final_ext in {
                    ".zip", ".jar", ".apk", ".docx", ".xlsx", ".pptx", ".odt", ".ods",
                }:
                    body = att.limited_body
                    if body is None and att.payload_ref is not None:
                        # Best-effort limited read for archive
                        try:
                            if hasattr(att.payload_ref, "read"):
                                body = att.payload_ref.read(  # type: ignore[call-arg]
                                    max_bytes=MAX_ARCHIVE_BYTES_TO_INSPECT
                                )
                                if isinstance(body, str):
                                    body = body.encode("utf-8", errors="replace")
                        except Exception:
                            body = None

                    if body:
                        archive_msgs = SafeArchiveInspector.inspect(
                            body, depth=0, deadline=deadline_monotonic
                        )
                        for msg in archive_msgs:
                            severity = SecurityVerdict.HIGH
                            points = 28.0
                            if "zip-bomb" in msg.lower() or "ratio" in msg.lower():
                                points = 36.0
                            elif "malformed" in msg.lower() or "corrupted" in msg.lower():
                                severity = SecurityVerdict.MEDIUM
                                points = 12.0
                            emit(
                                "EMAIL.ATTACHMENT.ARCHIVE_HAZARD",
                                "MALWARE_PAYLOAD",
                                severity,
                                0.91,
                                f"Archive '{filename[:64]}': {msg}",
                                points,
                                {"filename": filename[:64]},
                            )

                if time.monotonic() >= deadline_monotonic:
                    return self._timeout(start)

            # Final calibrated score
            total = (
                min(cat_scores["MALWARE_PAYLOAD"], 55.0)
                + min(cat_scores["DECEPTION"], 40.0)
                + min(cat_scores["MACRO_RISK"], 35.0)
                + min(cat_scores["OBFUSCATION"], 25.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.93 if findings else 0.88,
                findings=findings,
            )

        except Exception as exc:
            logger.exception(
                "AttachmentDetector failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"Attachment analysis error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during attachment processing.",
        )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AttachmentDetector",
    "AttachmentMeta",
    "EmailPayloadStore",
    "SafeArchiveInspector",
    "detect_magic",
    "get_extensions",
    "safe_filename",
]