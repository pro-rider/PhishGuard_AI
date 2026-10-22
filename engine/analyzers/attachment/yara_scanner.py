"""
PhishGuard_AI - Hardened YARA Pattern Matching Engine
Non-blocking YARA scanner for attachments and binary payloads.

Key controls
------------
1. Thread offload of yara-python C calls (event-loop safe).
2. Deadline-aligned per-file timeouts.
3. Safe modern/legacy match string extraction.
4. Isolated rule-file compilation + optional .yarc load.
5. Truncation transparency + SHA-256 of scanned window.
6. PayloadReference support + aggregate size caps.
7. Thread-safe ruleset init + calibrated scoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
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

try:
    import yara
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False
    yara = None  # type: ignore

# ============================================================================
# CONSTANTS
# ============================================================================

YARA_SCANNER_VERSION = "7.2.0"
RULESET_VERSION = "2026.08.13"

MAX_SINGLE_FILE_BYTES = 20 * 1024 * 1024
MAX_AGGREGATE_BYTES = 80 * 1024 * 1024
DEFAULT_TIMEOUT_SEC = 3.0
MAX_MATCHES_PER_FILE = 40
MAX_RULE_FILES = 200
MAX_PAYLOADS = 20
MAX_FINDINGS = 40

# Map rule meta categories into project categories
CATEGORY_MAP: Dict[str, str] = {
    "EXPLOIT_ATTEMPT": "EXPLOIT_ATTEMPT",
    "IMPERSONATION": "IMPERSONATION",
    "SUSPICIOUS_INFRASTRUCTURE": "SUSPICIOUS_INFRASTRUCTURE",
    "MALWARE": "MALWARE_PAYLOAD",
    "MALWARE_PAYLOAD": "MALWARE_PAYLOAD",
    "WEBSHELL": "MALWARE_PAYLOAD",
    "CREDENTIAL_HARVESTING": "CREDENTIAL_HARVESTING",
}

EMBEDDED_FALLBACK_YARA_RULES = r"""
rule PG_PE_Header {
    meta:
        description = "Windows PE header at start of file"
        category = "MALWARE_PAYLOAD"
        severity = "HIGH"
    strings:
        $mz = { 4D 5A }
    condition:
        $mz at 0
}

rule PG_ELF_Header {
    meta:
        description = "Linux ELF header at start of file"
        category = "MALWARE_PAYLOAD"
        severity = "HIGH"
    strings:
        $elf = { 7F 45 4C 46 }
    condition:
        $elf at 0
}

rule PG_Weaponized_VBA {
    meta:
        description = "VBA auto-exec + shell/createobject patterns"
        category = "MALWARE_PAYLOAD"
        severity = "CRITICAL"
    strings:
        $a1 = "AutoOpen" nocase
        $a2 = "Workbook_Open" nocase
        $a3 = "Document_Open" nocase
        $a4 = "Auto_Open" nocase
        $s1 = "Shell" nocase
        $s2 = "WScript.Shell" nocase
        $s3 = "CreateObject" nocase
        $s4 = "Powershell" nocase
        $s5 = "cmd.exe" nocase
    condition:
        any of ($a*) and any of ($s*)
}

rule PG_PDF_JS_or_Launch {
    meta:
        description = "PDF with JavaScript or Launch action"
        category = "EXPLOIT_ATTEMPT"
        severity = "HIGH"
    strings:
        $pdf = "%PDF-"
        $js1 = "/JavaScript" nocase
        $js2 = "/JS" nocase
        $launch = "/Launch" nocase
        $aa = "/AA" nocase
        $open = "/OpenAction" nocase
    condition:
        $pdf at 0 and (any of ($js*) or $launch or $aa or $open)
}

rule PG_RTF_Object {
    meta:
        description = "RTF with embedded object package markers"
        category = "EXPLOIT_ATTEMPT"
        severity = "HIGH"
    strings:
        $rtf = "{\\rtf"
        $obj = "\\object" nocase
        $objdata = "\\objdata" nocase
        $ole = "\\objclass" nocase
    condition:
        $rtf at 0 and 2 of ($obj, $objdata, $ole)
}

rule PG_LNK_Header {
    meta:
        description = "Windows Shell Link (LNK) header"
        category = "MALWARE_PAYLOAD"
        severity = "MEDIUM"
    strings:
        $lnk = { 4C 00 00 00 01 14 02 00 }
    condition:
        $lnk at 0
}

rule PG_ISO9660 {
    meta:
        description = "ISO9660 disk image signature"
        category = "SUSPICIOUS_INFRASTRUCTURE"
        severity = "MEDIUM"
    strings:
        $cd001 = "CD001"
    condition:
        $cd001
}
"""

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(slots=True)
class YaraMatchMetadata:
    rule_name: str
    rule_namespace: str = "default"
    category: str = "MALWARE_PAYLOAD"
    severity: SecurityVerdict = SecurityVerdict.HIGH
    description: str = ""
    matched_strings: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class YaraScanFileResult:
    filename: str
    sha256_digest: str
    original_size: int
    scanned_size: int
    is_truncated: bool = False
    matches: List[YaraMatchMetadata] = field(default_factory=list)
    scan_time_ms: float = 0.0
    runtime_warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass(slots=True)
class RulesetHealthStatus:
    is_healthy: bool = False
    source_type: str = "UNINITIALIZED"
    loaded_rules_count: int = 0
    failed_rule_files: List[str] = field(default_factory=list)


# ============================================================================
# RULESET MANAGER (thread-safe)
# ============================================================================

class YaraRulesetManager:
    def __init__(
        self,
        rules_directory: Optional[str] = None,
        trusted_compiled_path: Optional[str] = None,
    ) -> None:
        self.rules_directory = rules_directory
        self.trusted_compiled_path = trusted_compiled_path
        self._compiled_rules: Any = None
        self.health = RulesetHealthStatus()
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        if not YARA_AVAILABLE:
            self.health.source_type = "LIBRARY_MISSING"
            return False

        with self._lock:
            if self._compiled_rules is not None:
                return True

            # 1) Trusted compiled
            if self.trusted_compiled_path and os.path.isfile(self.trusted_compiled_path):
                try:
                    self._compiled_rules = yara.load(self.trusted_compiled_path)  # type: ignore[union-attr]
                    self.health.is_healthy = True
                    self.health.source_type = "TRUSTED_COMPILED_FILE"
                    return True
                except Exception as exc:
                    logger.error("Failed loading compiled YARA rules: %s", exc)

            # 2) Directory, file-by-file isolation
            if self.rules_directory and os.path.isdir(self.rules_directory):
                valid: Dict[str, str] = {}
                count = 0
                for root, _, files in os.walk(self.rules_directory):
                    for file in sorted(files):
                        if count >= MAX_RULE_FILES:
                            break
                        if not file.endswith((".yar", ".yara")):
                            continue
                        full = os.path.join(root, file)
                        try:
                            yara.compile(filepath=full)  # type: ignore[union-attr]
                            valid[f"ns_{count}"] = full
                            count += 1
                        except Exception as exc:
                            logger.warning("Skipping invalid YARA file %s: %s", full, exc)
                            self.health.failed_rule_files.append(full)
                if valid:
                    try:
                        self._compiled_rules = yara.compile(filepaths=valid)  # type: ignore[union-attr]
                        self.health.is_healthy = True
                        self.health.source_type = "CUSTOM_DIRECTORY"
                        self.health.loaded_rules_count = len(valid)
                        return True
                    except Exception as exc:
                        logger.error("Aggregate YARA compile failed: %s", exc)

            # 3) Embedded fallback
            try:
                self._compiled_rules = yara.compile(source=EMBEDDED_FALLBACK_YARA_RULES)  # type: ignore[union-attr]
                self.health.is_healthy = True
                self.health.source_type = "EMBEDDED_FALLBACK"
                self.health.loaded_rules_count = 7
                return True
            except Exception as exc:
                logger.critical("Embedded YARA compile failed: %s", exc)
                self.health.source_type = "COMPILATION_FAILED"
                return False

    def get_rules(self) -> Any:
        if self._compiled_rules is None:
            self.initialize()
        return self._compiled_rules


# ============================================================================
# SCANNER ENGINE
# ============================================================================

class YaraScannerEngine:
    def __init__(self, ruleset_manager: YaraRulesetManager) -> None:
        self.ruleset_manager = ruleset_manager

    @staticmethod
    def _extract_matched_strings(match_strings: Any) -> List[str]:
        extracted: List[str] = []
        if not match_strings:
            return extracted
        for s in match_strings:
            try:
                if hasattr(s, "identifier"):
                    extracted.append(str(s.identifier))
                elif isinstance(s, (tuple, list)) and len(s) >= 2:
                    # legacy: (offset, identifier, data)
                    extracted.append(str(s[1]))
                elif hasattr(s, "instances"):
                    # StringMatch with instances
                    extracted.append(str(getattr(s, "identifier", s)))
            except Exception:
                continue
        # dedupe preserve order
        return list(dict.fromkeys(extracted))[:12]

    @staticmethod
    def _map_severity(raw: str) -> SecurityVerdict:
        r = (raw or "HIGH").upper()
        if r == "CRITICAL":
            return SecurityVerdict.CRITICAL
        if r == "MEDIUM":
            return SecurityVerdict.MEDIUM
        if r == "LOW":
            return SecurityVerdict.LOW
        return SecurityVerdict.HIGH

    @staticmethod
    def _map_category(raw: str) -> str:
        return CATEGORY_MAP.get((raw or "").upper(), "MALWARE_PAYLOAD")

    def scan_bytes_sync(
        self,
        raw_bytes: bytes,
        filename: str,
        timeout_seconds: float,
    ) -> YaraScanFileResult:
        start = time.monotonic()
        orig = len(raw_bytes)
        truncated = orig > MAX_SINGLE_FILE_BYTES
        payload = raw_bytes[:MAX_SINGLE_FILE_BYTES] if truncated else raw_bytes
        digest = hashlib.sha256(payload).hexdigest()

        result = YaraScanFileResult(
            filename=filename,
            sha256_digest=digest,
            original_size=orig,
            scanned_size=len(payload),
            is_truncated=truncated,
        )

        rules = self.ruleset_manager.get_rules()
        if not rules:
            result.error_message = "RULESET_NOT_READY"
            result.scan_time_ms = (time.monotonic() - start) * 1000.0
            return result

        timeout_int = max(1, int(timeout_seconds))
        externals = {
            "filename": filename,
            "extension": os.path.splitext(filename)[1].lower(),
            "filesize": len(payload),
        }

        try:
            # Prefer externals when rules declare them; ignore if unsupported
            try:
                matches = rules.match(
                    data=payload,
                    timeout=timeout_int,
                    externals=externals,
                )
            except TypeError:
                matches = rules.match(data=payload, timeout=timeout_int)
            except Exception as ext_exc:
                # Some rule sets reject unknown externals
                if "external" in str(ext_exc).lower():
                    matches = rules.match(data=payload, timeout=timeout_int)
                else:
                    raise

            for match in (matches or [])[:MAX_MATCHES_PER_FILE]:
                meta = getattr(match, "meta", None) or {}
                category = self._map_category(str(meta.get("category", "MALWARE")))
                severity = self._map_severity(str(meta.get("severity", "HIGH")))
                desc = str(
                    meta.get("description")
                    or meta.get("desc")
                    or f"Matched YARA rule '{match.rule}'"
                )
                tags = list(getattr(match, "tags", []) or [])
                strings = self._extract_matched_strings(getattr(match, "strings", []))

                result.matches.append(
                    YaraMatchMetadata(
                        rule_name=str(match.rule),
                        rule_namespace=str(getattr(match, "namespace", "default")),
                        category=category,
                        severity=severity,
                        description=desc[:400],
                        matched_strings=strings,
                        tags=[str(t) for t in tags[:10]],
                    )
                )

        except Exception as exc:
            name = type(exc).__name__
            if "Timeout" in name or "timeout" in str(exc).lower():
                result.error_message = f"TIMEOUT_{timeout_int}S"
            else:
                result.error_message = f"SCAN_{name}"

        result.scan_time_ms = (time.monotonic() - start) * 1000.0
        return result


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class YaraPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[Tuple[str, bytes]]:
        payloads: List[Tuple[str, bytes]] = []
        cumulative = 0
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        def add(name: str, content: bytes) -> None:
            nonlocal cumulative
            if not content or len(payloads) >= MAX_PAYLOADS:
                return
            if cumulative >= MAX_AGGREGATE_BYTES:
                return
            chunk = content[: max(0, MAX_AGGREGATE_BYTES - cumulative)]
            if not chunk:
                return
            payloads.append((name or "attachment.bin", chunk))
            cumulative += len(chunk)

        for item in data.get("attachments") or data.get("email_attachments") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("filename") or item.get("name") or "attachment.bin")
            content = item.get("content_bytes") or item.get("payload_bytes") or item.get("data")
            if isinstance(content, (bytes, bytearray)):
                add(name, bytes(content))
                continue
            for key in ("payload_ref", "content_ref", "ref"):
                ref = item.get(key)
                if isinstance(ref, PayloadReference):
                    try:
                        raw = None
                        if hasattr(ref, "read"):
                            raw = ref.read(max_bytes=MAX_SINGLE_FILE_BYTES)
                        elif hasattr(ref, "content"):
                            raw = ref.content
                        if isinstance(raw, (bytes, bytearray)):
                            add(name, bytes(raw))
                    except Exception:
                        pass
                    break

        if not payloads:
            raw_file = data.get("raw_file_bytes")
            file_name = str(data.get("file_name") or data.get("filename") or "uploaded.bin")
            if isinstance(raw_file, (bytes, bytearray)):
                add(file_name, bytes(raw_file))

        return payloads


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class YaraScannerAnalyzer(BaseAnalyzer):
    name: str = "yara_scanner"
    version: str = YARA_SCANNER_VERSION
    supported_scan_types: Set[ScanType] = {
        ScanType.ATTACHMENT, ScanType.EMAIL, ScanType.COMPREHENSIVE
    }
    required_inputs: Set[str] = set()
    resource_class: AnalyzerResourceClass = AnalyzerResourceClass.COMPUTE
    priority: int = 6
    can_hard_override: bool = True

    def __init__(
        self,
        rules_directory: Optional[str] = None,
        trusted_compiled_path: Optional[str] = None,
    ) -> None:
        self.ruleset_manager = YaraRulesetManager(rules_directory, trusted_compiled_path)
        self.engine = YaraScannerEngine(self.ruleset_manager)
        # Eager init in background is optional; keep lazy + locked

    async def analyze(
        self, context: ScanContext, deadline_monotonic: float
    ) -> AnalyzerResult:
        start = time.monotonic()

        if not YARA_AVAILABLE:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error="yara-python library missing.",
            )

        if time.monotonic() >= deadline_monotonic:
            return self._timeout(start)

        # Ensure rules ready (thread-safe)
        await asyncio.to_thread(self.ruleset_manager.initialize)

        payloads = YaraPayloadStore.resolve(context)
        if not payloads:
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=0.0,
                confidence=0.85,
                findings=[],
            )

        findings: List[Finding] = []
        seen: Set[str] = set()
        cat = {
            "MALWARE_PAYLOAD": 0.0,
            "EXPLOIT_ATTEMPT": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
            "CREDENTIAL_HARVESTING": 0.0,
        }

        try:
            for name, content in payloads:
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0.15:
                    break
                timeout = min(DEFAULT_TIMEOUT_SEC, max(remaining - 0.05, 0.5))

                res: YaraScanFileResult = await asyncio.to_thread(
                    self.engine.scan_bytes_sync,
                    content,
                    name,
                    timeout,
                )

                if res.is_truncated and len(findings) < MAX_FINDINGS:
                    findings.append(
                        Finding(
                            rule_id="YARA.SCAN.TRUNCATED",
                            category="SUSPICIOUS_INFRASTRUCTURE",
                            severity=SecurityVerdict.LOW,
                            confidence=0.90,
                            message=(
                                f"Attachment '{name}' truncated for scan "
                                f"({res.original_size} → {res.scanned_size} bytes)."
                            ),
                            metadata=ConstrainedMetadata({
                                "filename": name[:120],
                                "sha256": res.sha256_digest,
                                "original_bytes": res.original_size,
                                "scanned_bytes": res.scanned_size,
                            }),
                        )
                    )
                    cat["SUSPICIOUS_INFRASTRUCTURE"] += 4.0

                if res.error_message:
                    if "TIMEOUT" in res.error_message and len(findings) < MAX_FINDINGS:
                        findings.append(
                            Finding(
                                rule_id="YARA.SCAN.TIMEOUT",
                                category="SUSPICIOUS_INFRASTRUCTURE",
                                severity=SecurityVerdict.LOW,
                                confidence=0.80,
                                message=f"YARA timeout on '{name}'.",
                                metadata=ConstrainedMetadata({
                                    "filename": name[:120],
                                    "error": res.error_message,
                                }),
                            )
                        )
                    continue

                for match in res.matches:
                    key = f"{name}:{match.rule_namespace}:{match.rule_name}"
                    if key in seen or len(findings) >= MAX_FINDINGS:
                        continue
                    seen.add(key)

                    points = {
                        SecurityVerdict.CRITICAL: 48.0,
                        SecurityVerdict.HIGH: 36.0,
                        SecurityVerdict.MEDIUM: 18.0,
                        SecurityVerdict.LOW: 8.0,
                    }.get(match.severity, 20.0)

                    cat_name = match.category if match.category in cat else "MALWARE_PAYLOAD"
                    cat[cat_name] = cat.get(cat_name, 0.0) + points

                    findings.append(
                        Finding(
                            rule_id=f"YARA.MATCH.{match.rule_name}",
                            category=cat_name,
                            severity=match.severity,
                            confidence=0.97,
                            message=(
                                f"YARA '{match.rule_name}' matched '{name}': "
                                f"{match.description[:160]}"
                            ),
                            metadata=ConstrainedMetadata({
                                "filename": name[:120],
                                "sha256": res.sha256_digest,
                                "rule": match.rule_name,
                                "namespace": match.rule_namespace,
                                "tags": match.tags[:8],
                                "strings": match.matched_strings[:6],
                                "scan_ms": round(res.scan_time_ms, 1),
                                "ruleset": self.ruleset_manager.health.source_type,
                            }),
                        )
                    )

            total = (
                min(cat["MALWARE_PAYLOAD"], 55.0)
                + min(cat["EXPLOIT_ATTEMPT"], 45.0)
                + min(cat["CREDENTIAL_HARVESTING"], 35.0)
                + min(cat["SUSPICIOUS_INFRASTRUCTURE"], 20.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.97 if any(
                    f.severity in {SecurityVerdict.HIGH, SecurityVerdict.CRITICAL}
                    for f in findings
                ) else (0.90 if findings else 0.85),
                findings=findings,
            )

        except Exception as exc:
            logger.exception(
                "YaraScanner failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"YARA scanner error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during YARA scanning.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "YaraScannerAnalyzer",
    "YaraScannerEngine",
    "YaraRulesetManager",
    "YaraPayloadStore",
    "YaraMatchMetadata",
    "YaraScanFileResult",
    "RulesetHealthStatus",
    "YARA_SCANNER_VERSION",
    "EMBEDDED_FALLBACK_YARA_RULES",
]
