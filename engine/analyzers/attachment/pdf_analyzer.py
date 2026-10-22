"""
PhishGuard_AI - Structural PDF Exploit & Threat Analyzer
Non-blocking PDF inspector for JS, Launch, embedded files, XFA/AcroForm,
filter chains, polyglot/EOF anomalies, and phishing URIs.

Key controls
------------
1. Thread-offloaded parsing with deadline checks.
2. Catalog / page / annotation / action dictionary walk with hex-name unmasking.
3. XFA + AcroForm, EmbeddedFiles, OpenAction/AA coverage.
4. Byte-level fallback for evasive / broken PDFs.
5. Bounds on size, pages, annots, URLs, depth.
6. PayloadReference support + calibrated scoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
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
    import pypdf
    from pypdf.generic import (
        ArrayObject,
        DictionaryObject,
        IndirectObject,
        NameObject,
    )
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    pypdf = None  # type: ignore
    DictionaryObject = ArrayObject = IndirectObject = object  # type: ignore

if PYPDF_AVAILABLE:
    logging.getLogger("pypdf").setLevel(logging.ERROR)

# ============================================================================
# CONSTANTS
# ============================================================================

PDF_ANALYZER_VERSION = "8.2.0"
RULESET_VERSION = "2026.08.13"

MAX_PDF_BYTES = 25 * 1024 * 1024
DEFAULT_TIMEOUT = 3.0
MAX_PAGES = 40
MAX_ANNOTS_PER_PAGE = 25
MAX_URLS = 40
MAX_DEPTH = 12
MAX_PAYLOADS = 12
MAX_FINDINGS = 30

DANGEROUS_ACTIONS: Set[str] = {
    "/JavaScript", "/JS", "/Launch", "/GoToR", "/SubmitForm",
    "/ImportData", "/RichMedia", "/Movie", "/Rendition", "/Sound",
}

SUSPICIOUS_FILTERS: Set[str] = {
    "/ASCIIHexDecode", "/ASCII85Decode", "/LZWDecode", "/Crypt", "/RunLengthDecode",
}

RE_HEX_NAME = re.compile(r"#([0-9a-fA-F]{2})")
RE_EOF = re.compile(rb"%%EOF", re.I)
RE_HEADER = re.compile(rb"%PDF-(\d\.\d)")
RE_ACTION = re.compile(rb"/S\s*/([A-Za-z0-9_#]+)")
RE_FILTER = re.compile(rb"/Filter\s*(?:/([A-Za-z0-9_#]+)|\[([^\]]+)\])")
RE_URI = re.compile(
    rb"/URI\s*(?:\((?:\\.|[^\\)])*\)|<([0-9A-Fa-f\s]+)>)",
    re.I,
)
RE_URI_SIMPLE = re.compile(rb"/URI\s*\((?:\\.|[^\\)])*\)")
RE_JS_TOKEN = re.compile(rb"/JavaScript|/JS[\s/>]", re.I)
RE_LAUNCH = re.compile(rb"/Launch[\s/>]", re.I)
RE_XFA = re.compile(rb"/XFA[\s/>]", re.I)
RE_EMBEDDED = re.compile(rb"/EmbeddedFiles|/Filespec|/EF[\s/>]", re.I)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(slots=True)
class PdfStructuralThreats:
    has_javascript: bool = False
    has_launch_actions: bool = False
    has_dangerous_actions: bool = False
    has_embedded_files: bool = False
    has_open_actions: bool = False
    has_acroform: bool = False
    has_xfa: bool = False
    has_submit_form: bool = False
    has_import_data: bool = False
    has_obfuscated_names: bool = False
    has_suspicious_filters: bool = False
    is_encrypted: bool = False
    missing_eof: bool = False
    multiple_eofs: bool = False
    has_header_anomaly: bool = False
    is_polyglot_or_overlay: bool = False
    incremental_updates: int = 0
    extracted_urls: List[str] = field(default_factory=list)
    action_types: Set[str] = field(default_factory=set)


@dataclass(slots=True)
class PdfParseResult:
    filename: str
    sha256_digest: str
    total_bytes: int
    page_count: int = 0
    threats: PdfStructuralThreats = field(default_factory=PdfStructuralThreats)
    parsed_via: str = "NONE"
    parse_time_ms: float = 0.0
    error_message: Optional[str] = None


# ============================================================================
# ENGINE
# ============================================================================

class PdfInspectorEngine:
    @staticmethod
    def unmask_name(raw: str) -> str:
        if "#" not in raw:
            return raw

        def repl(m: re.Match[str]) -> str:
            try:
                return bytes.fromhex(m.group(1)).decode("latin1", errors="ignore")
            except Exception:
                return m.group(0)

        return RE_HEX_NAME.sub(repl, raw)

    @classmethod
    def inspect_bytes_sync(
        cls, raw_bytes: bytes, filename: str, timeout_seconds: float
    ) -> PdfParseResult:
        start = time.monotonic()
        deadline = start + timeout_seconds

        if len(raw_bytes) > MAX_PDF_BYTES:
            raw_bytes = raw_bytes[:MAX_PDF_BYTES]

        digest = hashlib.sha256(raw_bytes).hexdigest()
        result = PdfParseResult(
            filename=filename,
            sha256_digest=digest,
            total_bytes=len(raw_bytes),
        )
        if not raw_bytes:
            result.error_message = "EMPTY_PAYLOAD"
            return result

        # --- Header / EOF anomalies ---
        hm = RE_HEADER.search(raw_bytes)
        if not hm or hm.start() > 1024:
            result.threats.has_header_anomaly = True
        # PE+PDF polyglot hint
        if raw_bytes.startswith(b"MZ") and hm:
            result.threats.is_polyglot_or_overlay = True

        eofs = list(RE_EOF.finditer(raw_bytes))
        result.threats.incremental_updates = max(0, len(eofs) - 1)
        if not eofs:
            result.threats.missing_eof = True
        elif len(eofs) > 1:
            result.threats.multiple_eofs = True
        if eofs and (len(raw_bytes) - eofs[-1].end()) > 512:
            result.threats.is_polyglot_or_overlay = True

        # --- AST path ---
        if PYPDF_AVAILABLE:
            try:
                reader = pypdf.PdfReader(io.BytesIO(raw_bytes), strict=False)
                result.page_count = len(reader.pages)
                result.parsed_via = "STRUCTURAL_AST"
                if getattr(reader, "is_encrypted", False):
                    result.threats.is_encrypted = True

                root = None
                try:
                    root = reader.trailer.get("/Root")
                    if isinstance(root, IndirectObject):
                        root = root.get_object()
                except Exception:
                    root = None

                if isinstance(root, DictionaryObject):
                    cls._walk_dict(root, result, deadline, 0)
                    # Explicit XFA / AcroForm
                    if "/AcroForm" in root:
                        result.threats.has_acroform = True
                        try:
                            acro = root["/AcroForm"]
                            if isinstance(acro, IndirectObject):
                                acro = acro.get_object()
                            if isinstance(acro, DictionaryObject):
                                if "/XFA" in acro:
                                    result.threats.has_xfa = True
                                cls._walk_dict(acro, result, deadline, 1)
                        except Exception:
                            pass
                    if "/Names" in root:
                        try:
                            names = root["/Names"]
                            if isinstance(names, IndirectObject):
                                names = names.get_object()
                            if isinstance(names, DictionaryObject):
                                if "/EmbeddedFiles" in names or "/JavaScript" in names:
                                    if "/EmbeddedFiles" in names:
                                        result.threats.has_embedded_files = True
                                    if "/JavaScript" in names:
                                        result.threats.has_javascript = True
                                    cls._walk_dict(names, result, deadline, 1)
                        except Exception:
                            pass

                pages = min(result.page_count, MAX_PAGES)
                for i in range(pages):
                    if time.monotonic() >= deadline:
                        break
                    try:
                        page = reader.pages[i]
                        if isinstance(page, DictionaryObject):
                            cls._walk_dict(page, result, deadline, 0)
                            annots = page.get("/Annots")
                            if isinstance(annots, IndirectObject):
                                annots = annots.get_object()
                            if isinstance(annots, ArrayObject):
                                for annot in list(annots)[:MAX_ANNOTS_PER_PAGE]:
                                    if time.monotonic() >= deadline:
                                        break
                                    obj = annot.get_object() if isinstance(annot, IndirectObject) else annot
                                    if isinstance(obj, DictionaryObject):
                                        cls._walk_dict(obj, result, deadline, 0)
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("pypdf failed for %s: %s", filename, type(exc).__name__)

        # --- Byte fallback (always runs as defense-in-depth) ---
        cls._byte_scan(raw_bytes, result, deadline)
        if result.parsed_via == "NONE":
            result.parsed_via = "BYTE_FALLBACK"

        result.parse_time_ms = (time.monotonic() - start) * 1000.0
        return result

    @classmethod
    def _resolve(cls, obj: Any) -> Any:
        try:
            if isinstance(obj, IndirectObject):
                return obj.get_object()
        except Exception:
            return obj
        return obj

    @classmethod
    def _walk_dict(
        cls, dict_obj: Any, result: PdfParseResult, deadline: float, depth: int
    ) -> None:
        if depth > MAX_DEPTH or time.monotonic() >= deadline:
            return
        dict_obj = cls._resolve(dict_obj)
        if not isinstance(dict_obj, DictionaryObject):
            return

        for key, val in list(dict_obj.items()):
            if time.monotonic() >= deadline:
                break
            raw_key = str(key)
            clean_key = cls.unmask_name(raw_key)
            if clean_key != raw_key:
                result.threats.has_obfuscated_names = True

            if clean_key in ("/A", "/OpenAction", "/AA", "/Next"):
                result.threats.has_open_actions = True
                cls._resolve_action(val, result, deadline, depth + 1)
            elif clean_key == "/AcroForm":
                result.threats.has_acroform = True
            elif clean_key == "/XFA":
                result.threats.has_xfa = True
            elif clean_key in ("/EmbeddedFiles", "/EF", "/Filespec"):
                result.threats.has_embedded_files = True
            elif clean_key == "/URI":
                cls._ingest_uri(val, result)
            elif clean_key == "/JS" or clean_key == "/JavaScript":
                result.threats.has_javascript = True
                result.threats.has_dangerous_actions = True
                result.threats.action_types.add(clean_key)
            elif clean_key == "/Filter":
                cls._note_filters(val, result)

            resolved = cls._resolve(val)
            if isinstance(resolved, DictionaryObject):
                cls._walk_dict(resolved, result, deadline, depth + 1)
            elif isinstance(resolved, ArrayObject):
                for item in list(resolved)[:40]:
                    if time.monotonic() >= deadline:
                        break
                    r = cls._resolve(item)
                    if isinstance(r, DictionaryObject):
                        cls._walk_dict(r, result, deadline, depth + 1)

    @classmethod
    def _resolve_action(
        cls, action_obj: Any, result: PdfParseResult, deadline: float, depth: int
    ) -> None:
        if depth > MAX_DEPTH or time.monotonic() >= deadline:
            return
        action_obj = cls._resolve(action_obj)
        if isinstance(action_obj, ArrayObject):
            for item in list(action_obj)[:20]:
                cls._resolve_action(item, result, deadline, depth + 1)
            return
        if not isinstance(action_obj, DictionaryObject):
            return

        s_raw = str(action_obj.get("/S", ""))
        s_type = cls.unmask_name(s_raw)
        if s_type in DANGEROUS_ACTIONS:
            result.threats.has_dangerous_actions = True
            result.threats.action_types.add(s_type)
        if s_type in ("/JavaScript", "/JS"):
            result.threats.has_javascript = True
        elif s_type == "/Launch":
            result.threats.has_launch_actions = True
        elif s_type == "/SubmitForm":
            result.threats.has_submit_form = True
        elif s_type == "/ImportData":
            result.threats.has_import_data = True
        elif s_type == "/URI":
            cls._ingest_uri(action_obj.get("/URI"), result)

        # Nested Next actions
        if "/Next" in action_obj:
            cls._resolve_action(action_obj.get("/Next"), result, deadline, depth + 1)

    @classmethod
    def _ingest_uri(cls, val: Any, result: PdfParseResult) -> None:
        if val is None:
            return
        val = cls._resolve(val)
        text = ""
        try:
            if isinstance(val, DictionaryObject):
                text = str(val.get("/URI", "") or "")
            else:
                text = str(val)
        except Exception:
            return
        text = text.strip()
        if not text:
            return
        if text.startswith(("http://", "https://", "ftp://", "file://", "javascript:")):
            if text not in result.threats.extracted_urls and len(result.threats.extracted_urls) < MAX_URLS:
                result.threats.extracted_urls.append(text[:500])

    @classmethod
    def _note_filters(cls, val: Any, result: PdfParseResult) -> None:
        val = cls._resolve(val)
        tokens: List[str] = []
        if isinstance(val, NameObject) or isinstance(val, str):
            tokens.append(cls.unmask_name(str(val)))
        elif isinstance(val, ArrayObject):
            for x in val:
                tokens.append(cls.unmask_name(str(x)))
        for t in tokens:
            if t in SUSPICIOUS_FILTERS or any(t.endswith(s[1:]) for s in SUSPICIOUS_FILTERS):
                result.threats.has_suspicious_filters = True

    @classmethod
    def _byte_scan(cls, raw: bytes, result: PdfParseResult, deadline: float) -> None:
        if RE_JS_TOKEN.search(raw):
            result.threats.has_javascript = True
            result.threats.has_dangerous_actions = True
            result.threats.action_types.add("/JavaScript")
        if RE_LAUNCH.search(raw):
            result.threats.has_launch_actions = True
            result.threats.has_dangerous_actions = True
            result.threats.action_types.add("/Launch")
        if RE_XFA.search(raw):
            result.threats.has_xfa = True
        if RE_EMBEDDED.search(raw):
            result.threats.has_embedded_files = True

        for m in RE_ACTION.finditer(raw):
            if time.monotonic() >= deadline:
                break
            act = "/" + m.group(1).decode("latin1", errors="ignore")
            unmasked = cls.unmask_name(act)
            if unmasked != act:
                result.threats.has_obfuscated_names = True
            if unmasked in DANGEROUS_ACTIONS:
                result.threats.has_dangerous_actions = True
                result.threats.action_types.add(unmasked)
                if unmasked in ("/JavaScript", "/JS"):
                    result.threats.has_javascript = True
                if unmasked == "/Launch":
                    result.threats.has_launch_actions = True
                if unmasked == "/SubmitForm":
                    result.threats.has_submit_form = True
                if unmasked == "/ImportData":
                    result.threats.has_import_data = True

        for m in RE_FILTER.finditer(raw):
            if time.monotonic() >= deadline:
                break
            blob = (m.group(1) or m.group(2) or b"").decode("latin1", errors="ignore")
            for sf in SUSPICIOUS_FILTERS:
                if sf[1:] in blob or sf in blob:
                    result.threats.has_suspicious_filters = True
                    break

        # Best-effort URI scrape
        if len(result.threats.extracted_urls) < MAX_URLS:
            for m in RE_URI_SIMPLE.finditer(raw):
                if time.monotonic() >= deadline:
                    break
                try:
                    s = m.group(0).decode("latin1", errors="ignore")
                    # /URI (http...)
                    lp = s.find("(")
                    rp = s.rfind(")")
                    if lp >= 0 and rp > lp:
                        uri = s[lp + 1 : rp].replace("\\", "")
                        cls._ingest_uri(uri, result)
                except Exception:
                    continue


# ============================================================================
# PAYLOAD STORE
# ============================================================================

class PdfPayloadStore:
    @staticmethod
    def resolve(context: ScanContext) -> List[Tuple[str, bytes]]:
        out: List[Tuple[str, bytes]] = []
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        def maybe_add(name: str, content: bytes) -> None:
            if not content or len(out) >= MAX_PAYLOADS:
                return
            lower = name.lower()
            if content.startswith(b"%PDF-") or content[:1024].find(b"%PDF-") >= 0 or lower.endswith(".pdf"):
                out.append((name or "document.pdf", content[:MAX_PDF_BYTES]))

        for item in data.get("attachments") or data.get("email_attachments") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("filename") or item.get("name") or "document.pdf")
            content = item.get("content_bytes") or item.get("payload_bytes") or item.get("data")
            if isinstance(content, (bytes, bytearray)):
                maybe_add(name, bytes(content))
                continue
            for key in ("payload_ref", "content_ref", "ref"):
                ref = item.get(key)
                if isinstance(ref, PayloadReference):
                    try:
                        raw = None
                        if hasattr(ref, "read"):
                            raw = ref.read(max_bytes=MAX_PDF_BYTES)
                        elif hasattr(ref, "content"):
                            raw = ref.content
                        if isinstance(raw, (bytes, bytearray)):
                            maybe_add(name, bytes(raw))
                    except Exception:
                        pass
                    break

        if not out:
            raw_file = data.get("raw_file_bytes")
            file_name = str(data.get("file_name") or data.get("filename") or "document.pdf")
            if isinstance(raw_file, (bytes, bytearray)):
                maybe_add(file_name, bytes(raw_file))

        return out


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class PdfAnalyzer(BaseAnalyzer):
    name: str = "pdf_analyzer"
    version: str = PDF_ANALYZER_VERSION
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

        payloads = PdfPayloadStore.resolve(context)
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
            "EXPLOIT_ATTEMPT": 0.0,
            "MALWARE_PAYLOAD": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
            "IMPERSONATION": 0.0,
        }
        collected_urls: List[str] = []

        def emit(
            key: str,
            rule_id: str,
            category: str,
            severity: SecurityVerdict,
            confidence: float,
            message: str,
            points: float,
            meta: Dict[str, Any],
        ) -> None:
            if key in seen or len(findings) >= MAX_FINDINGS:
                return
            seen.add(key)
            cat[category] = cat.get(category, 0.0) + points
            findings.append(
                Finding(
                    rule_id=rule_id,
                    category=category,
                    severity=severity,
                    confidence=confidence,
                    message=message,
                    metadata=ConstrainedMetadata(meta),
                )
            )

        try:
            for name, content in payloads:
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0.2:
                    break
                timeout = min(DEFAULT_TIMEOUT, max(remaining - 0.1, 0.5))

                res: PdfParseResult = await asyncio.to_thread(
                    PdfInspectorEngine.inspect_bytes_sync,
                    content,
                    name,
                    timeout,
                )
                if res.error_message:
                    continue

                t = res.threats
                base_meta = {
                    "filename": name[:120],
                    "sha256": res.sha256_digest,
                    "parsed_via": res.parsed_via,
                    "pages": res.page_count,
                }

                if t.has_launch_actions:
                    emit(
                        f"{name}:LAUNCH",
                        "PDF.EXPLOIT.LAUNCH",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.CRITICAL,
                        0.98,
                        f"PDF '{name}' contains /Launch action (process execution).",
                        50.0,
                        base_meta,
                    )

                if t.has_javascript:
                    emit(
                        f"{name}:JS",
                        "PDF.SCRIPT.JAVASCRIPT",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.95,
                        f"PDF '{name}' embeds JavaScript.",
                        38.0,
                        {**base_meta, "obfuscated_names": t.has_obfuscated_names},
                    )

                if t.has_embedded_files:
                    emit(
                        f"{name}:EMBED",
                        "PDF.PAYLOAD.EMBEDDED_FILE",
                        "MALWARE_PAYLOAD",
                        SecurityVerdict.HIGH,
                        0.92,
                        f"PDF '{name}' contains embedded file specification(s).",
                        34.0,
                        base_meta,
                    )

                if t.has_xfa:
                    emit(
                        f"{name}:XFA",
                        "PDF.FORM.XFA",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.90,
                        f"PDF '{name}' contains XFA form (high-risk vector).",
                        30.0,
                        base_meta,
                    )
                elif t.has_acroform and (t.has_javascript or t.has_open_actions):
                    emit(
                        f"{name}:ACRO",
                        "PDF.FORM.ACROFORM_ACTIVE",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.88,
                        f"PDF '{name}' has AcroForm with active actions.",
                        16.0,
                        base_meta,
                    )

                if t.has_submit_form:
                    emit(
                        f"{name}:SUBMIT",
                        "PDF.FORM.SUBMIT",
                        "CREDENTIAL_HARVESTING"
                        if "CREDENTIAL_HARVESTING" in cat
                        else "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.91,
                        f"PDF '{name}' uses /SubmitForm action.",
                        28.0,
                        base_meta,
                    )
                    # keep category bucket stable
                    cat["EXPLOIT_ATTEMPT"] = cat.get("EXPLOIT_ATTEMPT", 0.0)

                if t.has_import_data:
                    emit(
                        f"{name}:IMPORT",
                        "PDF.FORM.IMPORT_DATA",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.MEDIUM,
                        0.88,
                        f"PDF '{name}' uses /ImportData action.",
                        18.0,
                        base_meta,
                    )

                if t.has_suspicious_filters:
                    emit(
                        f"{name}:FILTER",
                        "PDF.EVASION.FILTER_CHAIN",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.86,
                        f"PDF '{name}' uses suspicious stream filters.",
                        14.0,
                        base_meta,
                    )

                if t.is_polyglot_or_overlay:
                    emit(
                        f"{name}:POLYGLOT",
                        "PDF.EVASION.POLYGLOT_OR_OVERLAY",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.88,
                        f"PDF '{name}' shows polyglot/trailing-overlay characteristics.",
                        18.0,
                        base_meta,
                    )

                if t.has_header_anomaly or t.missing_eof:
                    emit(
                        f"{name}:STRUCT",
                        "PDF.STRUCTURE.ANOMALY",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.LOW,
                        0.80,
                        f"PDF '{name}' has header/EOF structural anomalies.",
                        8.0,
                        {
                            **base_meta,
                            "header_anomaly": t.has_header_anomaly,
                            "missing_eof": t.missing_eof,
                            "multiple_eofs": t.multiple_eofs,
                        },
                    )

                if t.is_encrypted and (t.has_open_actions or t.has_javascript):
                    emit(
                        f"{name}:ENC",
                        "PDF.CRYPTO.ENCRYPTED_ACTIVE",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.85,
                        f"Encrypted PDF '{name}' still exposes active action indicators.",
                        15.0,
                        base_meta,
                    )

                if t.extracted_urls:
                    collected_urls.extend(t.extracted_urls)
                    emit(
                        f"{name}:URI",
                        "PDF.LINK.EXTRACTED_URI",
                        "IMPERSONATION",
                        SecurityVerdict.LOW,
                        0.90,
                        f"Extracted {len(t.extracted_urls)} URI(s) from PDF '{name}'.",
                        6.0,
                        {
                            **base_meta,
                            "url_count": len(t.extracted_urls),
                            "sample_urls": t.extracted_urls[:5],
                        },
                    )

            # Hand off URLs to downstream URL analyzers
            if collected_urls:
                meta = getattr(context, "metadata", None)
                if meta is not None and hasattr(meta, "data"):
                    existing = meta.data.get("extracted_urls") or []
                    if not isinstance(existing, list):
                        existing = []
                    merged = list(dict.fromkeys(list(existing) + collected_urls))[:100]
                    meta.data["extracted_urls"] = merged

            total = (
                min(cat["EXPLOIT_ATTEMPT"], 55.0)
                + min(cat["MALWARE_PAYLOAD"], 40.0)
                + min(cat["SUSPICIOUS_INFRASTRUCTURE"], 30.0)
                + min(cat["IMPERSONATION"], 15.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.95 if findings else 0.85,
                findings=findings,
            )

        except Exception as exc:
            logger.exception(
                "PdfAnalyzer failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"PDF analyzer error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during PDF analysis.",
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PdfAnalyzer",
    "PdfInspectorEngine",
    "PdfParseResult",
    "PdfStructuralThreats",
    "PdfPayloadStore",
    "PDF_ANALYZER_VERSION",
]
