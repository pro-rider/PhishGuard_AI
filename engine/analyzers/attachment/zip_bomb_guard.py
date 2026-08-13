"""
PhishGuard_AI - Hardened Attachment Decompression & Zip Bomb Guard
Production archive inspector for zip bombs, ZipSlip, dangerous TAR nodes,
encrypted containers, nested depth, and high-risk members.

Key controls
------------
1. Multi-factor bomb detection (ratio + absolute sizes + entry count).
2. Cross-platform path traversal / absolute / UNC / control-char guards.
3. TAR symlink / hardlink / device / FIFO detection.
4. Streaming-oriented inspection with hard bounds (no unbounded extract).
5. Nested archive depth cap with size-capped nested reads.
6. Fail-closed on malformed containers.
7. High-risk extension detection inside archives.
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
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
# CONSTANTS
# ============================================================================

ZIP_BOMB_GUARD_VERSION = "6.2.0"
RULESET_VERSION = "2026.08.13"

MAX_DECOMPRESSION_RATIO = 80.0
CRITICAL_RATIO = 400.0
MAX_UNCOMPRESSED_TOTAL = 120 * 1024 * 1024
MAX_SINGLE_MEMBER = 60 * 1024 * 1024
MAX_ARCHIVE_FILES = 800
MAX_NESTED_DEPTH = 2
MAX_FILENAME_LEN = 255
MAX_METADATA_ENTRIES = 150
MAX_ATTACHMENTS = 15
MAX_INPUT_BYTES = 40 * 1024 * 1024
MAX_NESTED_READ = 4 * 1024 * 1024
MAX_FINDINGS = 25

MAGIC_ZIP = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
MAGIC_GZIP = b"\x1f\x8b"
MAGIC_BZ2 = b"BZh"
MAGIC_XZ = b"\xfd7zXZ\x00"
MAGIC_7Z = b"7z\xbc\xaf\x27\x1c"
MAGIC_RAR = (b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")

RE_WIN_ABS = re.compile(
    r"^(?:[a-zA-Z]:[\\/]|[\\/]{2}[^\\/]+[\\/]+[^\\/]+|\\\\\?\\)",
    re.I,
)
RE_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")

HIGH_RISK_EXTS: Set[str] = {
    ".exe", ".dll", ".scr", ".com", ".bat", ".cmd", ".ps1", ".vbs", ".vbe",
    ".js", ".jse", ".wsf", ".wsh", ".hta", ".msi", ".msp", ".jar", ".lnk",
    ".iso", ".img", ".vhd", ".docm", ".xlsm", ".pptm", ".rtf", ".chm",
}

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(slots=True)
class ArchiveEntryMetadata:
    filename: str
    compressed_size: int
    uncompressed_size: int
    compression_ratio: float
    is_directory: bool = False
    is_encrypted: bool = False
    is_symlink_or_hardlink: bool = False
    is_special_device_node: bool = False
    has_path_traversal: bool = False
    is_high_risk_ext: bool = False


@dataclass(slots=True)
class ArchiveInspectionResult:
    archive_name: str
    archive_type: str
    file_count: int = 0
    total_compressed_size: int = 0
    total_uncompressed_size: int = 0
    overall_compression_ratio: float = 0.0
    nested_depth: int = 1
    has_zip_bomb_signature: bool = False
    has_path_traversal: bool = False
    has_dangerous_links_or_devices: bool = False
    has_duplicate_path_collision: bool = False
    has_encrypted_entries: bool = False
    is_completely_encrypted: bool = False
    has_high_risk_members: bool = False
    exceeds_max_size: bool = False
    exceeds_max_files: bool = False
    exceeds_depth_limit: bool = False
    is_malformed: bool = False
    entries: List[ArchiveEntryMetadata] = field(default_factory=list)
    error_message: Optional[str] = None


# ============================================================================
# PATH GUARD
# ============================================================================

class CrossPlatformPathGuard:
    @classmethod
    def normalize(cls, raw: str) -> str:
        return (raw or "").replace("\\", "/").strip()

    @classmethod
    def validate(cls, raw_path: str) -> Tuple[str, bool, Optional[str]]:
        if not raw_path:
            return "", True, "EMPTY_PATH"
        if RE_CTRL.search(raw_path):
            return raw_path, True, "CONTROL_CHARS"
        if len(raw_path) > MAX_FILENAME_LEN:
            return raw_path, True, "NAME_TOO_LONG"
        if RE_WIN_ABS.match(raw_path) or raw_path.startswith("/"):
            return raw_path, True, "ABSOLUTE_PATH"

        clean = cls.normalize(raw_path)
        depth = 0
        for part in clean.split("/"):
            if part == "..":
                depth -= 1
                if depth < 0:
                    return clean, True, "RELATIVE_TRAVERSAL"
            elif part and part != ".":
                depth += 1
        return clean, False, None

    @classmethod
    def collision_key(cls, path: str) -> str:
        """Normalize for duplicate detection."""
        p = cls.normalize(path)
        parts = [x for x in p.split("/") if x and x != "."]
        return "/".join(parts).lower()


def _high_risk_ext(name: str) -> bool:
    lower = name.lower()
    # handle double extensions: report.pdf.exe
    for ext in HIGH_RISK_EXTS:
        if lower.endswith(ext):
            return True
    return False


# ============================================================================
# ENGINE
# ============================================================================

class ZipBombGuardEngine:
    @classmethod
    def inspect_bytes(
        cls,
        raw: bytes,
        filename: str = "attachment.bin",
        depth: int = 1,
        deadline: Optional[float] = None,
    ) -> ArchiveInspectionResult:
        result = ArchiveInspectionResult(
            archive_name=filename,
            archive_type="UNKNOWN",
            nested_depth=depth,
            total_compressed_size=len(raw) if raw else 0,
        )
        if not raw:
            result.is_malformed = True
            result.error_message = "EMPTY_PAYLOAD"
            return result
        if depth > MAX_NESTED_DEPTH:
            result.exceeds_depth_limit = True
            result.error_message = f"MAX_DEPTH_{MAX_NESTED_DEPTH}"
            return result
        if deadline is not None and time.monotonic() >= deadline:
            result.error_message = "DEADLINE"
            return result

        if raw.startswith(MAGIC_ZIP):
            result.archive_type = "ZIP"
            return cls._inspect_zip(raw, result, depth, deadline)
        if raw.startswith((MAGIC_GZIP, MAGIC_BZ2, MAGIC_XZ)):
            # Try tar.* first; if not tar, treat as single-stream archive
            result.archive_type = "COMPRESSED_STREAM"
            tar_try = cls._inspect_tar(raw, result, depth, deadline)
            if not tar_try.is_malformed:
                tar_try.archive_type = "TAR_COMPRESSED"
                return tar_try
            # pure gzip/bz2/xz without tar → limited inspection
            result.is_malformed = False
            result.error_message = None
            result.file_count = 1
            result.total_uncompressed_size = 0  # unknown without full inflate
            result.entries.append(
                ArchiveEntryMetadata(
                    filename=filename,
                    compressed_size=len(raw),
                    uncompressed_size=0,
                    compression_ratio=0.0,
                )
            )
            return result
        if raw.startswith(MAGIC_7Z):
            result.archive_type = "7Z"
            result.error_message = "7Z_HEADER_PRESENT"
            # Not fully parsed without py7zr; still flag as opaque container
            return result
        if raw.startswith(MAGIC_RAR):
            result.archive_type = "RAR"
            result.error_message = "RAR_HEADER_PRESENT"
            return result

        result.is_malformed = True
        result.error_message = "UNKNOWN_ARCHIVE_HEADER"
        return result

    @classmethod
    def _inspect_zip(
        cls,
        raw: bytes,
        result: ArchiveInspectionResult,
        depth: int,
        deadline: Optional[float],
    ) -> ArchiveInspectionResult:
        try:
            with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
                infos = zf.infolist()
                result.file_count = len(infos)
                if result.file_count > MAX_ARCHIVE_FILES:
                    result.exceeds_max_files = True

                seen: Set[str] = set()
                encrypted = 0

                for info in infos:
                    if deadline is not None and time.monotonic() >= deadline:
                        result.error_message = "DEADLINE"
                        break

                    c_size = max(int(info.compress_size or 0), 0)
                    u_size = max(int(info.file_size or 0), 0)
                    ratio = (u_size / c_size) if c_size > 0 else (float("inf") if u_size else 1.0)

                    if u_size > MAX_SINGLE_MEMBER or (
                        ratio >= CRITICAL_RATIO and u_size > 5 * 1024 * 1024
                    ):
                        result.has_zip_bomb_signature = True

                    clean, trav, _ = CrossPlatformPathGuard.validate(info.filename)
                    if trav:
                        result.has_path_traversal = True
                    key = CrossPlatformPathGuard.collision_key(clean)
                    if key and key in seen and not info.is_dir():
                        result.has_duplicate_path_collision = True
                    if key:
                        seen.add(key)

                    is_enc = bool(info.flag_bits & 0x1)
                    if is_enc:
                        encrypted += 1
                        result.has_encrypted_entries = True

                    risky = _high_risk_ext(info.filename)
                    if risky and not info.is_dir():
                        result.has_high_risk_members = True

                    if len(result.entries) < MAX_METADATA_ENTRIES:
                        result.entries.append(
                            ArchiveEntryMetadata(
                                filename=info.filename[:300],
                                compressed_size=c_size,
                                uncompressed_size=u_size,
                                compression_ratio=float(ratio) if ratio != float("inf") else 9999.0,
                                is_directory=info.is_dir(),
                                is_encrypted=is_enc,
                                has_path_traversal=trav,
                                is_high_risk_ext=risky,
                            )
                        )

                    result.total_uncompressed_size += u_size

                    # Nested: only small members, header sniff, hard read cap
                    if (
                        not info.is_dir()
                        and not is_enc
                        and depth < MAX_NESTED_DEPTH
                        and 0 < u_size <= MAX_NESTED_READ
                        and c_size <= MAX_NESTED_READ
                    ):
                        try:
                            nested = zf.read(info)[:MAX_NESTED_READ]
                            if nested.startswith(MAGIC_ZIP) or nested.startswith(
                                (MAGIC_GZIP, MAGIC_BZ2, MAGIC_XZ)
                            ):
                                nested_res = cls.inspect_bytes(
                                    nested,
                                    filename=info.filename,
                                    depth=depth + 1,
                                    deadline=deadline,
                                )
                                if nested_res.has_zip_bomb_signature:
                                    result.has_zip_bomb_signature = True
                                if nested_res.has_path_traversal:
                                    result.has_path_traversal = True
                                if nested_res.has_high_risk_members:
                                    result.has_high_risk_members = True
                        except Exception:
                            pass

                if encrypted and encrypted == result.file_count and result.file_count > 0:
                    result.is_completely_encrypted = True

                if result.total_compressed_size > 0:
                    result.overall_compression_ratio = (
                        result.total_uncompressed_size / float(result.total_compressed_size)
                    )
                if (
                    result.overall_compression_ratio >= MAX_DECOMPRESSION_RATIO
                    and result.total_uncompressed_size > 8 * 1024 * 1024
                ):
                    result.has_zip_bomb_signature = True
                if result.total_uncompressed_size > MAX_UNCOMPRESSED_TOTAL:
                    result.exceeds_max_size = True

        except zipfile.BadZipFile:
            result.is_malformed = True
            result.error_message = "BAD_ZIP"
        except Exception as exc:
            result.is_malformed = True
            result.error_message = f"ZIP_{type(exc).__name__}"
        return result

    @classmethod
    def _inspect_tar(
        cls,
        raw: bytes,
        result: ArchiveInspectionResult,
        depth: int,
        deadline: Optional[float],
    ) -> ArchiveInspectionResult:
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
                count = 0
                seen: Set[str] = set()
                while True:
                    if deadline is not None and time.monotonic() >= deadline:
                        result.error_message = "DEADLINE"
                        break
                    member = tf.next()
                    if member is None:
                        break
                    count += 1
                    if count > MAX_ARCHIVE_FILES:
                        result.exceeds_max_files = True
                        # stop iterating further members
                        break

                    u_size = max(int(member.size or 0), 0)
                    clean, trav, _ = CrossPlatformPathGuard.validate(member.name)
                    if trav:
                        result.has_path_traversal = True
                    key = CrossPlatformPathGuard.collision_key(clean)
                    if key and key in seen and not member.isdir():
                        result.has_duplicate_path_collision = True
                    if key:
                        seen.add(key)

                    is_link = member.islnk() or member.issym()
                    is_dev = member.ischr() or member.isblk() or member.isfifo()
                    if is_link or is_dev:
                        result.has_dangerous_links_or_devices = True

                    risky = _high_risk_ext(member.name)
                    if risky and not member.isdir():
                        result.has_high_risk_members = True

                    if u_size > MAX_SINGLE_MEMBER:
                        result.has_zip_bomb_signature = True

                    if len(result.entries) < MAX_METADATA_ENTRIES:
                        result.entries.append(
                            ArchiveEntryMetadata(
                                filename=member.name[:300],
                                compressed_size=u_size,  # tar member size
                                uncompressed_size=u_size,
                                compression_ratio=1.0,
                                is_directory=member.isdir(),
                                is_symlink_or_hardlink=is_link,
                                is_special_device_node=is_dev,
                                has_path_traversal=trav,
                                is_high_risk_ext=risky,
                            )
                        )
                    result.total_uncompressed_size += u_size

                result.file_count = count
                if result.total_compressed_size > 0:
                    result.overall_compression_ratio = (
                        result.total_uncompressed_size / float(result.total_compressed_size)
                    )
                if (
                    result.overall_compression_ratio >= MAX_DECOMPRESSION_RATIO
                    and result.total_uncompressed_size > 8 * 1024 * 1024
                ):
                    result.has_zip_bomb_signature = True
                if result.total_uncompressed_size > MAX_UNCOMPRESSED_TOTAL:
                    result.exceeds_max_size = True

        except tarfile.TarError as exc:
            result.is_malformed = True
            result.error_message = f"TAR_{type(exc).__name__}"
        except Exception as exc:
            result.is_malformed = True
            result.error_message = f"TAR_{type(exc).__name__}"
        return result


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class ZipBombPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[Tuple[str, bytes]]:
        out: List[Tuple[str, bytes]] = []
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}
        total = 0

        def add(name: str, content: bytes) -> None:
            nonlocal total
            if not content or total >= MAX_INPUT_BYTES:
                return
            if len(out) >= MAX_ATTACHMENTS:
                return
            chunk = content[: max(0, MAX_INPUT_BYTES - total)]
            if not chunk:
                return
            total += len(chunk)
            out.append((name or "attachment.bin", chunk))

        for item in data.get("attachments") or data.get("email_attachments") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("filename") or item.get("name") or "attachment.bin")
            content = item.get("content_bytes") or item.get("payload_bytes") or item.get("data")
            if isinstance(content, (bytes, bytearray)):
                add(name, bytes(content))
                continue
            # PayloadReference
            for key in ("payload_ref", "content_ref", "ref"):
                ref = item.get(key)
                if isinstance(ref, PayloadReference):
                    try:
                        raw = None
                        if hasattr(ref, "read"):
                            raw = ref.read(max_bytes=MAX_INPUT_BYTES)
                        elif hasattr(ref, "content"):
                            raw = ref.content
                        if isinstance(raw, (bytes, bytearray)):
                            add(name, bytes(raw))
                    except Exception:
                        pass
                    break

        raw_file = data.get("raw_file_bytes")
        file_name = str(data.get("file_name") or data.get("filename") or "uploaded.bin")
        if isinstance(raw_file, (bytes, bytearray)):
            add(file_name, bytes(raw_file))

        return out


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class ZipBombGuardAnalyzer(BaseAnalyzer):
    name: str = "zip_bomb_guard"
    version: str = ZIP_BOMB_GUARD_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.ATTACHMENT, ScanType.EMAIL, ScanType.COMPREHENSIVE
    }
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.COMPUTE
    priority: int = 5
    can_hard_override: bool = True

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()
        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        attachments = ZipBombPayloadStore.resolve(context)
        if not attachments:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.85,
                findings=[],
            )

        findings: List[Finding] = []
        cat = {
            "EXPLOIT_ATTEMPT": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
            "MALWARE_PAYLOAD": 0.0,
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
            cat[category] = cat.get(category, 0.0) + points
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
            for name, content in attachments:
                if time.monotonic() >= deadline_monotonic:
                    break
                # Skip obvious non-archives quickly
                if len(content) < 4:
                    continue

                res = ZipBombGuardEngine.inspect_bytes(
                    content, filename=name, depth=1, deadline=deadline_monotonic
                )

                if res.archive_type in {"7Z", "RAR"}:
                    emit(
                        "ATTACHMENT.ARCHIVE.OPAQUE_CONTAINER",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.85,
                        f"Opaque {res.archive_type} container '{name}' (limited inspection).",
                        18.0,
                        {"filename": name[:120], "type": res.archive_type},
                    )

                if res.is_malformed:
                    emit(
                        "ATTACHMENT.ARCHIVE.MALFORMED",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.HIGH,
                        0.90,
                        f"Malformed archive '{name}' ({res.error_message}).",
                        28.0,
                        {"filename": name[:120], "error": res.error_message or ""},
                    )
                    continue

                if res.has_zip_bomb_signature:
                    emit(
                        "ATTACHMENT.ZIP_BOMB.RATIO_OR_SIZE",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.CRITICAL,
                        0.97,
                        (
                            f"Zip-bomb indicators in '{name}' "
                            f"(ratio≈{res.overall_compression_ratio:.1f}:1, "
                            f"uncompressed≈{res.total_uncompressed_size / (1024*1024):.1f}MB)."
                        ),
                        55.0,
                        {
                            "filename": name[:120],
                            "ratio": round(res.overall_compression_ratio, 2),
                            "uncompressed": res.total_uncompressed_size,
                        },
                    )

                if res.has_path_traversal:
                    emit(
                        "ATTACHMENT.ZIP_SLIP.TRAVERSAL",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.95,
                        f"Path traversal / ZipSlip paths in '{name}'.",
                        42.0,
                        {"filename": name[:120]},
                    )

                if res.has_duplicate_path_collision:
                    emit(
                        "ATTACHMENT.ARCHIVE.PATH_COLLISION",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.MEDIUM,
                        0.88,
                        f"Duplicate/colliding paths in '{name}'.",
                        16.0,
                        {"filename": name[:120]},
                    )

                if res.has_dangerous_links_or_devices:
                    emit(
                        "ATTACHMENT.TAR.DANGEROUS_NODE",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.93,
                        f"Symlink/hardlink/device nodes in '{name}'.",
                        40.0,
                        {"filename": name[:120]},
                    )

                if res.exceeds_max_size or res.exceeds_max_files:
                    emit(
                        "ATTACHMENT.ARCHIVE.LIMIT_EXCEEDED",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.HIGH,
                        0.92,
                        (
                            f"Archive '{name}' exceeds safety limits "
                            f"(files={res.file_count}, "
                            f"uncompressed={res.total_uncompressed_size})."
                        ),
                        30.0,
                        {
                            "filename": name[:120],
                            "files": res.file_count,
                            "uncompressed": res.total_uncompressed_size,
                        },
                    )

                if res.exceeds_depth_limit:
                    emit(
                        "ATTACHMENT.ARCHIVE.NESTING_DEPTH",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.90,
                        f"Nested archive depth exceeded in '{name}'.",
                        18.0,
                        {"filename": name[:120]},
                    )

                if res.is_completely_encrypted:
                    emit(
                        "ATTACHMENT.ARCHIVE.FULLY_ENCRYPTED",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.90,
                        f"Fully encrypted archive '{name}' (content opaque).",
                        22.0,
                        {"filename": name[:120]},
                    )
                elif res.has_encrypted_entries:
                    emit(
                        "ATTACHMENT.ARCHIVE.ENCRYPTED_ENTRIES",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.LOW,
                        0.85,
                        f"Encrypted entries present in '{name}'.",
                        12.0,
                        {"filename": name[:120]},
                    )

                if res.has_high_risk_members:
                    emit(
                        "ATTACHMENT.ARCHIVE.HIGH_RISK_MEMBER",
                        "MALWARE_PAYLOAD",
                        SecurityVerdict.HIGH,
                        0.93,
                        f"High-risk executable/script member(s) inside '{name}'.",
                        36.0,
                        {
                            "filename": name[:120],
                            "samples": [
                                e.filename[:80]
                                for e in res.entries
                                if e.is_high_risk_ext
                            ][:5],
                        },
                    )

            total = (
                min(cat["EXPLOIT_ATTEMPT"], 60.0)
                + min(cat["MALWARE_PAYLOAD"], 40.0)
                + min(cat["SUSPICIOUS_INFRASTRUCTURE"], 35.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.95 if findings else 0.86,
                findings=findings,
            )

        except Exception as exc:
            logger.exception(
                "ZipBombGuard failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"Zip bomb guard error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during archive guard analysis.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ZipBombGuardAnalyzer",
    "ZipBombGuardEngine",
    "CrossPlatformPathGuard",
    "ArchiveInspectionResult",
    "ArchiveEntryMetadata",
    "ZipBombPayloadStore",
    "ZIP_BOMB_GUARD_VERSION",
]