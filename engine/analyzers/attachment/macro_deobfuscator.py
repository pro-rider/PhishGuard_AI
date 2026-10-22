"""
PhishGuard_AI - VBA Macro Deobfuscator & Office Malware Analyzer
Production static analysis of OLE/OOXML macro attachments.

Security properties
-------------------
- Never executes macro code.
- Zip-bomb / size / entry bounds on OOXML.
- Bounded VBA text + deobfuscation passes.
- Auto-exec correlation with download/execution.
- URL extraction + hand-off to URL pipeline.
- oletools when available; safe string fallbacks.
- PayloadReference support + deadline awareness.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import logging
import re
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit

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
    from oletools.olevba import VBA_Parser
    OLEVBA_AVAILABLE = True
except ImportError:
    VBA_Parser = None  # type: ignore
    OLEVBA_AVAILABLE = False

# ============================================================================
# CONSTANTS
# ============================================================================

MACRO_DEOBFUSCATOR_VERSION = "9.2.0"
RULESET_VERSION = "2026.08.13"

MAX_MACRO_TEXT_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 6 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 40 * 1024 * 1024
MAX_ZIP_ENTRIES = 400
MAX_ZIP_MEMBER_BYTES = 12 * 1024 * 1024
MAX_TOTAL_ZIP_UNCOMPRESSED = 25 * 1024 * 1024
MAX_VBA_PROJECT_BYTES = 12 * 1024 * 1024
MAX_DEOBFUSCATED_STR_LEN = 8_000
MAX_DEOBFUSCATION_RESULTS = 80
MAX_URLS = 40
MAX_FINDINGS = 40
MAX_ATTACHMENTS = 20
DEFAULT_TIMEOUT = 3.0

OOXML_MACRO_EXTS = {
    ".docm", ".dotm", ".xlsm", ".xltm", ".xlam",
    ".pptm", ".potm", ".ppsm", ".ppam",
}
OLE_EXTS = {".doc", ".dot", ".xls", ".xlt", ".ppt", ".pot", ".pps", ".ppa"}
SOURCE_EXTS = {".bas", ".cls", ".frm", ".vba", ".vbs"}

OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

# ============================================================================
# REGEXES
# ============================================================================

RE_AUTO_EXEC = re.compile(
    r"(?ix)\b(?:"
    r"autoopen|autoexec|auto_open|auto_close|auto_exit|"
    r"workbook_open|workbook_activate|workbook_beforeclose|"
    r"document_open|document_close|document_new|document_beforeclose|"
    r"presentation_open|presentation_close|presentation_save"
    r")\b"
)

RE_SUSPICIOUS_API = re.compile(
    r"(?ix)\b(?:"
    r"shell|wscript(?:\.shell)?|createobject|getobject|"
    r"powershell|pwsh|cmd(?:\.exe)?|rundll32|regsvr32|mshta|"
    r"cscript|certutil|bitsadmin|msiexec|"
    r"urldownloadtofile|winhttp|xmlhttp|adodb\.stream|"
    r"virtualalloc|virtualprotect|createthread|"
    r"writeprocessmemory|readprocessmemory|openprocess|"
    r"winexec|shellexecute|regwrite|regread|"
    r"win32_process|managementobject|wmi|callbyname|eval"
    r")\b"
)

RE_SANDBOX_EVASION = re.compile(
    r"(?ix)\b(?:"
    r"application\.wait|application\.ontime|sleep|"
    r"isdebuggerpresent|checkremotedebuggerpresent|"
    r"getcursorpos|getasynckeystate|gettickcount|"
    r"queryperformancecounter|vmware|virtualbox|vbox|qemu|hyperv|sandbox"
    r")\b"
)

RE_DDE = re.compile(r"(?ix)\b(?:DDE(?:AUTO|LINK|REQUEST)?|DDEAUTO|DDELINK)\b")
RE_FILE_DOWNLOAD = re.compile(
    r"(?ix)(?:URLDownloadToFile|WinHttpRequest|XMLHTTP|ServerXMLHTTP|ADODB\.Stream)"
)
RE_EXECUTION = re.compile(
    r"(?ix)\b(?:Shell|Execute|Eval|CallByName|Application\.Run)\s*\("
)
RE_DECLARE_API = re.compile(
    r"(?ix)\bDeclare\s+(?:PtrSafe\s+)?(?:Function|Sub)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
RE_CREATEOBJECT_TARGET = re.compile(
    r"""(?ix)CreateObject\s*\(\s*["']([^"']{1,256})["']"""
)
RE_VBA_CHR = re.compile(r"(?ix)\bChrW?\s*\(\s*(\d{1,6})\s*\)")
RE_STRREVERSE = re.compile(r"""(?ix)\bStrReverse\s*\(\s*["']([^"']{1,8000})["']\s*\)""")
RE_HEX_STRING = re.compile(r"""(?ix)["']([0-9A-F]{12,})["']""")
RE_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{40,}={0,2})(?![A-Za-z0-9+/=])")
RE_PS_ENC = re.compile(
    r"(?ix)powershell(?:\.exe)?[^\n]{0,200}(?:-enc|-e|-encodedcommand)\s+([A-Za-z0-9+/=]{20,})"
)
RE_URL = re.compile(r"""(?ix)\bhttps?://[^\s"'<>]{1,2048}""")
RE_XOR_LIT = re.compile(
    r"(?ix)(&H[0-9A-F]+|\d+)\s+Xor\s+(&H[0-9A-F]+|\d+)"
)
RE_ENVIRON = re.compile(r"(?ix)\bEnviron\s*\(")
RE_NULL = re.compile(r"\x00+")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(slots=True)
class DeobfuscatedPayload:
    original_snippet: str
    deobfuscated_text: str
    technique: str


@dataclass(slots=True)
class MacroAnalysisResult:
    filename: str
    sha256_digest: str
    has_macros: bool = False
    extraction_method: str = "NONE"
    auto_exec_triggers: List[str] = field(default_factory=list)
    suspicious_apis: List[str] = field(default_factory=list)
    evasion_triggers: List[str] = field(default_factory=list)
    download_indicators: List[str] = field(default_factory=list)
    execution_indicators: List[str] = field(default_factory=list)
    createobject_targets: List[str] = field(default_factory=list)
    declared_apis: List[str] = field(default_factory=list)
    dde_indicators: List[str] = field(default_factory=list)
    environment_discovery: List[str] = field(default_factory=list)
    deobfuscated_items: List[DeobfuscatedPayload] = field(default_factory=list)
    extracted_urls: List[str] = field(default_factory=list)
    macro_text_truncated: bool = False
    parse_time_ms: float = 0.0
    error_message: Optional[str] = None


# ============================================================================
# UTILS
# ============================================================================

def _safe_text(value: Any, limit: int = 1024) -> str:
    try:
        text = str(value)
    except Exception:
        return ""
    return text.replace("\x00", "")[:limit]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_url(url: str) -> Optional[str]:
    url = url.strip().rstrip(".,;:)]}>\"'")
    if len(url) > 2048:
        return None
    try:
        p = urlsplit(url)
        if p.scheme.lower() not in {"http", "https"} or not p.netloc:
            return None
        return url
    except Exception:
        return None


def _unique_bounded(values: Iterable[str], limit: int) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for v in values:
        v = _safe_text(v, 2048)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _interesting_text(text: str) -> bool:
    lower = text.lower()
    return any(
        x in lower
        for x in (
            "http://", "https://", "powershell", "cmd.exe", "wscript",
            "shell", "rundll32", "mshta", "createobject", "download",
            "invoke-", "bypass", "frombase64",
        )
    )


# ============================================================================
# DEOBFUSCATION
# ============================================================================

class MacroDeobfuscationEngine:
    @classmethod
    def deobfuscate_chr_calls(cls, text: str) -> List[DeobfuscatedPayload]:
        results: List[DeobfuscatedPayload] = []
        matches = list(RE_VBA_CHR.finditer(text))
        if not matches:
            return results

        current: List[re.Match[str]] = []
        prev_end = -1
        for m in matches:
            # Allow small gaps for & / + / whitespace concatenation
            if current and m.start() - prev_end > 48:
                cls._flush_chr(current, results)
                current = []
            current.append(m)
            prev_end = m.end()
        if current:
            cls._flush_chr(current, results)
        return results[:MAX_DEOBFUSCATION_RESULTS]

    @staticmethod
    def _flush_chr(matches: List[re.Match[str]], results: List[DeobfuscatedPayload]) -> None:
        chars: List[str] = []
        for m in matches:
            try:
                val = int(m.group(1))
                if 0 <= val <= 0x10FFFF:
                    chars.append(chr(val))
            except (ValueError, OverflowError):
                continue
        decoded = "".join(chars)
        if len(decoded) < 3:
            return
        results.append(
            DeobfuscatedPayload(
                original_snippet=matches[0].group(0)[:200],
                deobfuscated_text=decoded[:MAX_DEOBFUSCATED_STR_LEN],
                technique="VBA_CHR_CHAIN",
            )
        )

    @classmethod
    def deobfuscate_strreverse(cls, text: str) -> List[DeobfuscatedPayload]:
        results: List[DeobfuscatedPayload] = []
        for m in RE_STRREVERSE.finditer(text):
            original = m.group(1)
            if not original:
                continue
            decoded = original[::-1]
            if decoded != original:
                results.append(
                    DeobfuscatedPayload(
                        original_snippet=m.group(0)[:200],
                        deobfuscated_text=decoded[:MAX_DEOBFUSCATED_STR_LEN],
                        technique="VBA_STRREVERSE",
                    )
                )
            if len(results) >= MAX_DEOBFUSCATION_RESULTS:
                break
        return results

    @classmethod
    def deobfuscate_base64(cls, text: str) -> List[DeobfuscatedPayload]:
        results: List[DeobfuscatedPayload] = []
        for m in RE_BASE64.finditer(text):
            cand = m.group(1)
            if len(cand) > 180_000:
                continue
            pad = (-len(cand)) % 4
            try:
                decoded = base64.b64decode(cand + ("=" * pad), validate=False)
            except (binascii.Error, ValueError):
                continue
            if len(decoded) < 8:
                continue
            out = decoded.decode("utf-8", errors="replace")
            printable = sum(1 for c in out if c.isprintable() or c in "\r\n\t") / max(len(out), 1)
            if printable < 0.50 and not _interesting_text(out):
                continue
            results.append(
                DeobfuscatedPayload(
                    original_snippet=cand[:200],
                    deobfuscated_text=out[:MAX_DEOBFUSCATED_STR_LEN],
                    technique="BASE64_DECODE",
                )
            )
            if len(results) >= MAX_DEOBFUSCATION_RESULTS:
                break
        return results

    @classmethod
    def deobfuscate_powershell_enc(cls, text: str) -> List[DeobfuscatedPayload]:
        results: List[DeobfuscatedPayload] = []
        for m in RE_PS_ENC.finditer(text):
            cand = m.group(1)
            pad = (-len(cand)) % 4
            try:
                # PowerShell -enc is UTF-16LE base64
                decoded = base64.b64decode(cand + ("=" * pad), validate=False)
                out = decoded.decode("utf-16-le", errors="replace")
            except Exception:
                continue
            if len(out) < 4:
                continue
            results.append(
                DeobfuscatedPayload(
                    original_snippet=m.group(0)[:200],
                    deobfuscated_text=out[:MAX_DEOBFUSCATED_STR_LEN],
                    technique="POWERSHELL_ENC",
                )
            )
            if len(results) >= MAX_DEOBFUSCATION_RESULTS:
                break
        return results

    @classmethod
    def deobfuscate_hex(cls, text: str) -> List[DeobfuscatedPayload]:
        results: List[DeobfuscatedPayload] = []
        for m in RE_HEX_STRING.finditer(text):
            hx = m.group(1)
            if len(hx) % 2:
                continue
            try:
                decoded = bytes.fromhex(hx)
            except ValueError:
                continue
            out = decoded.decode("utf-8", errors="replace")
            if not out or not any(c.isprintable() for c in out):
                continue
            results.append(
                DeobfuscatedPayload(
                    original_snippet=m.group(0)[:200],
                    deobfuscated_text=out[:MAX_DEOBFUSCATED_STR_LEN],
                    technique="HEX_STRING_DECODE",
                )
            )
            if len(results) >= MAX_DEOBFUSCATION_RESULTS:
                break
        return results

    @classmethod
    def deobfuscate_xor_literals(cls, text: str) -> List[DeobfuscatedPayload]:
        results: List[DeobfuscatedPayload] = []
        for m in RE_XOR_LIT.finditer(text):
            try:
                left_raw, right_raw = m.group(1), m.group(2)
                left = int(left_raw[2:], 16) if left_raw.lower().startswith("&h") else int(left_raw)
                right = int(right_raw[2:], 16) if right_raw.lower().startswith("&h") else int(right_raw)
                value = left ^ right
                if not 32 <= value <= 126:
                    continue
                results.append(
                    DeobfuscatedPayload(
                        original_snippet=m.group(0)[:200],
                        deobfuscated_text=chr(value),
                        technique="VBA_LITERAL_XOR",
                    )
                )
            except Exception:
                continue
            if len(results) >= MAX_DEOBFUSCATION_RESULTS:
                break
        return results

    @classmethod
    def run_all(cls, text: str) -> List[DeobfuscatedPayload]:
        results: List[DeobfuscatedPayload] = []
        for fn in (
            cls.deobfuscate_chr_calls,
            cls.deobfuscate_strreverse,
            cls.deobfuscate_powershell_enc,
            cls.deobfuscate_base64,
            cls.deobfuscate_hex,
            cls.deobfuscate_xor_literals,
        ):
            try:
                for item in fn(text):
                    results.append(item)
                    if len(results) >= MAX_DEOBFUSCATION_RESULTS:
                        return results
            except Exception:
                logger.debug("deobfuscation pass failed: %s", getattr(fn, "__name__", "?"))
        return results


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================

class MacroAnalysisEngine:
    @classmethod
    def analyze_sync(
        cls, macro_text: str, filename: str, timeout_seconds: float
    ) -> MacroAnalysisResult:
        start = time.monotonic()
        digest = hashlib.sha256(macro_text.encode("utf-8", errors="ignore")).hexdigest()
        result = MacroAnalysisResult(
            filename=_safe_text(filename, 512),
            sha256_digest=digest,
            has_macros=bool(macro_text.strip()),
        )
        if not result.has_macros:
            return result

        if len(macro_text) > MAX_MACRO_TEXT_BYTES:
            macro_text = macro_text[:MAX_MACRO_TEXT_BYTES]
            result.macro_text_truncated = True

        macro_text = macro_text.replace("\r\n", "\n")
        macro_text = RE_NULL.sub("", macro_text)

        def timed_out() -> bool:
            return (time.monotonic() - start) >= max(timeout_seconds, 0.1)

        result.auto_exec_triggers = _unique_bounded(RE_AUTO_EXEC.findall(macro_text), 32)
        if timed_out():
            result.error_message = "BUDGET"
            return result

        result.suspicious_apis = _unique_bounded(RE_SUSPICIOUS_API.findall(macro_text), 64)
        result.evasion_triggers = _unique_bounded(RE_SANDBOX_EVASION.findall(macro_text), 32)
        result.dde_indicators = _unique_bounded(RE_DDE.findall(macro_text), 16)
        result.download_indicators = _unique_bounded(RE_FILE_DOWNLOAD.findall(macro_text), 32)
        result.execution_indicators = _unique_bounded(RE_EXECUTION.findall(macro_text), 32)
        result.declared_apis = _unique_bounded(RE_DECLARE_API.findall(macro_text), 32)
        result.createobject_targets = _unique_bounded(RE_CREATEOBJECT_TARGET.findall(macro_text), 32)
        if RE_ENVIRON.search(macro_text):
            result.environment_discovery = ["Environ"]

        if timed_out():
            result.error_message = "BUDGET"
            return result

        result.deobfuscated_items = MacroDeobfuscationEngine.run_all(macro_text)

        # Re-scan decoded material for APIs / downloads
        decoded_blob = "\n".join(i.deobfuscated_text for i in result.deobfuscated_items[:30])
        if decoded_blob:
            result.suspicious_apis = _unique_bounded(
                result.suspicious_apis + RE_SUSPICIOUS_API.findall(decoded_blob), 64
            )
            result.download_indicators = _unique_bounded(
                result.download_indicators + RE_FILE_DOWNLOAD.findall(decoded_blob), 32
            )
            result.execution_indicators = _unique_bounded(
                result.execution_indicators + RE_EXECUTION.findall(decoded_blob), 32
            )

        if timed_out():
            result.error_message = "BUDGET"
            return result

        urls: List[str] = []
        for m in RE_URL.finditer(macro_text):
            n = _normalize_url(m.group(0))
            if n:
                urls.append(n)
            if len(urls) >= MAX_URLS:
                break
        for item in result.deobfuscated_items:
            for m in RE_URL.finditer(item.deobfuscated_text):
                n = _normalize_url(m.group(0))
                if n:
                    urls.append(n)
                if len(urls) >= MAX_URLS:
                    break
            if len(urls) >= MAX_URLS:
                break
        result.extracted_urls = _unique_bounded(urls, MAX_URLS)

        result.parse_time_ms = (time.monotonic() - start) * 1000.0
        return result


# ============================================================================
# EXTRACTION
# ============================================================================

class MacroPayloadStore:
    @classmethod
    def _safe_zip_vba_members(cls, content: bytes) -> List[Tuple[str, bytes]]:
        out: List[Tuple[str, bytes]] = []
        if len(content) > MAX_ATTACHMENT_BYTES or not content.startswith(ZIP_MAGICS):
            return out
        try:
            with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
                infos = zf.infolist()
                if len(infos) > MAX_ZIP_ENTRIES:
                    return out
                total = 0
                seen: Set[str] = set()
                for info in infos:
                    name = (info.filename or "").replace("\\", "/")
                    if not name or name.startswith("/") or ".." in name.split("/"):
                        continue
                    low = name.lower()
                    if low in seen or info.is_dir():
                        continue
                    seen.add(low)
                    size = int(info.file_size or 0)
                    if size < 0 or size > MAX_ZIP_MEMBER_BYTES:
                        continue
                    total += size
                    if total > MAX_TOTAL_ZIP_UNCOMPRESSED:
                        break
                    if "vbaproject.bin" not in low:
                        continue
                    if size > MAX_VBA_PROJECT_BYTES:
                        continue
                    try:
                        data = zf.read(info)
                    except Exception:
                        continue
                    if 0 < len(data) <= MAX_VBA_PROJECT_BYTES:
                        out.append((name, data))
        except Exception:
            return []
        return out

    @classmethod
    def _extract_olevba(cls, filename: str, content: bytes) -> Tuple[str, bool]:
        if not OLEVBA_AVAILABLE or VBA_Parser is None:
            return "", False
        parser = None
        try:
            parser = VBA_Parser(filename=filename, data=content)
            try:
                has = bool(parser.detect_vba_macros())
            except Exception:
                has = False
            if not has:
                return "", False
            blocks: List[str] = []
            total = 0
            try:
                for item in parser.extract_macros():
                    code = None
                    if isinstance(item, (tuple, list)) and len(item) >= 4:
                        code = item[3]
                    elif isinstance(item, (tuple, list)) and len(item) >= 1:
                        code = item[-1]
                    if not code:
                        continue
                    text = str(code)
                    remain = MAX_MACRO_TEXT_BYTES - total
                    if remain <= 0:
                        break
                    text = text[:remain]
                    blocks.append(text)
                    total += len(text)
            except Exception:
                pass
            combined = "\n".join(blocks)
            return combined[:MAX_MACRO_TEXT_BYTES], bool(combined.strip())
        except Exception as exc:
            logger.debug("olevba failed for %s: %s", _safe_text(filename, 120), type(exc).__name__)
            return "", False
        finally:
            if parser is not None:
                try:
                    parser.close()
                except Exception:
                    pass

    @classmethod
    def _fallback_strings(cls, raw: bytes) -> str:
        if not raw or len(raw) > MAX_VBA_PROJECT_BYTES:
            return ""
        strings = re.findall(rb"[\x20-\x7E\t\r\n]{6,}", raw[:MAX_VBA_PROJECT_BYTES])
        out: List[str] = []
        total = 0
        for s in strings:
            try:
                t = s.decode("latin1", errors="ignore")
            except Exception:
                continue
            remain = MAX_MACRO_TEXT_BYTES - total
            if remain <= 0:
                break
            t = t[:remain]
            out.append(t)
            total += len(t)
        return "\n".join(out)

    @classmethod
    def extract_macro_text(cls, filename: str, content: bytes) -> Tuple[str, str]:
        if not content or len(content) > MAX_ATTACHMENT_BYTES:
            return "", "NONE" if content else "EMPTY"

        name = _safe_text(filename, 512)
        lower = name.lower()

        if any(lower.endswith(ext) for ext in SOURCE_EXTS):
            data = content[:MAX_SOURCE_BYTES]
            try:
                text = data.decode("utf-8", errors="replace")
                if text.strip():
                    return text[:MAX_MACRO_TEXT_BYTES], "SOURCE_TEXT"
            except Exception:
                return "", "SOURCE_DECODE_ERROR"

        text, ok = cls._extract_olevba(name, content)
        if ok and text:
            return text, "OLETOOLS"

        if content.startswith(ZIP_MAGICS):
            members = cls._safe_zip_vba_members(content)
            blocks: List[str] = []
            for member_name, raw in members:
                # Prefer olevba on vbaProject.bin alone
                t2, ok2 = cls._extract_olevba(member_name, raw)
                if ok2 and t2:
                    blocks.append(t2)
                else:
                    fb = cls._fallback_strings(raw)
                    if fb:
                        blocks.append(fb)
            combined = "\n".join(blocks)
            if combined.strip():
                return combined[:MAX_MACRO_TEXT_BYTES], "OOXML_EXTRACT"

        if content.startswith(OLE_MAGIC):
            fb = cls._fallback_strings(content[:MAX_VBA_PROJECT_BYTES])
            if fb.strip():
                return fb[:MAX_MACRO_TEXT_BYTES], "OLE_STRING_FALLBACK"

        return "", "NONE"

    @classmethod
    def resolve(cls, context: ScanContext) -> List[Tuple[str, str, str]]:
        payloads: List[Tuple[str, str, str]] = []
        seen: Set[str] = set()
        meta = getattr(context, "metadata", None)
        data: Dict[str, Any] = getattr(meta, "data", {}) if meta else {}

        def add_bytes(name: str, content: bytes) -> None:
            if not content or len(content) > MAX_ATTACHMENT_BYTES:
                return
            digest = _sha256(content[: min(len(content), 1024 * 1024)] + str(len(content)).encode())
            if digest in seen:
                return
            seen.add(digest)
            text, method = cls.extract_macro_text(name, content)
            if text:
                payloads.append((name, text[:MAX_MACRO_TEXT_BYTES], method))

        for item in (data.get("attachments") or data.get("email_attachments") or [])[:MAX_ATTACHMENTS]:
            if not isinstance(item, dict):
                continue
            name = _safe_text(item.get("filename") or item.get("name") or "document.docm", 512)
            content = item.get("content_bytes") or item.get("payload_bytes") or item.get("data")
            if isinstance(content, (bytes, bytearray)):
                add_bytes(name, bytes(content))
                continue
            for key in ("payload_ref", "content_ref", "ref"):
                ref = item.get(key)
                if isinstance(ref, PayloadReference):
                    try:
                        raw = None
                        if hasattr(ref, "read"):
                            raw = ref.read(max_bytes=MAX_ATTACHMENT_BYTES)
                        elif hasattr(ref, "content"):
                            raw = ref.content
                        if isinstance(raw, (bytes, bytearray)):
                            add_bytes(name, bytes(raw))
                    except Exception:
                        pass
                    break

        if not payloads:
            raw_file = data.get("raw_file_bytes")
            file_name = _safe_text(data.get("file_name") or data.get("filename") or "document.docm", 512)
            if isinstance(raw_file, (bytes, bytearray)):
                add_bytes(file_name, bytes(raw_file))

        return payloads


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class MacroDeobfuscatorAnalyzer(BaseAnalyzer):
    name: str = "macro_deobfuscator"
    version: str = MACRO_DEOBFUSCATOR_VERSION
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

        payloads = MacroPayloadStore.resolve(context)
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
        seen_rules: Set[str] = set()
        cat = {
            "EXPLOIT_ATTEMPT": 0.0,
            "MALWARE_PAYLOAD": 0.0,
            "SUSPICIOUS_INFRASTRUCTURE": 0.0,
            "IMPERSONATION": 0.0,
        }
        all_urls: List[str] = []

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
            if key in seen_rules or len(findings) >= MAX_FINDINGS:
                return
            seen_rules.add(key)
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
            for name, macro_code, method in payloads:
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0.2:
                    break
                timeout = min(DEFAULT_TIMEOUT, max(remaining - 0.1, 0.25))

                result: MacroAnalysisResult = await asyncio.to_thread(
                    MacroAnalysisEngine.analyze_sync,
                    macro_code,
                    name,
                    timeout,
                )
                if result.error_message == "BUDGET" or not result.has_macros:
                    continue

                base = {
                    "filename": _safe_text(name, 160),
                    "sha256": result.sha256_digest,
                    "extraction_method": method,
                }

                # Critical: autoexec + execution + (api|download)
                if (
                    result.auto_exec_triggers
                    and result.execution_indicators
                    and (result.suspicious_apis or result.download_indicators)
                ):
                    emit(
                        f"{name}:WEAPONIZED",
                        "MACRO.WEAPONIZED.AUTOEXEC_EXECUTION",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.CRITICAL,
                        0.99,
                        f"Macro '{_safe_text(name, 120)}' combines auto-exec with execution/download capability.",
                        52.0,
                        {
                            **base,
                            "auto_exec": result.auto_exec_triggers[:8],
                            "execution": result.execution_indicators[:8],
                            "apis": result.suspicious_apis[:10],
                        },
                    )

                if result.download_indicators and (
                    result.execution_indicators or result.suspicious_apis
                ):
                    emit(
                        f"{name}:DL_EXEC",
                        "MACRO.DOWNLOAD.EXECUTION_CHAIN",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.97,
                        f"Macro '{_safe_text(name, 120)}' shows download + execution indicators.",
                        40.0,
                        {
                            **base,
                            "download": result.download_indicators[:8],
                            "execution": result.execution_indicators[:8],
                        },
                    )

                if result.deobfuscated_items:
                    techniques = _unique_bounded(
                        (i.technique for i in result.deobfuscated_items), 12
                    )
                    decoded = " ".join(i.deobfuscated_text for i in result.deobfuscated_items[:10])
                    malicious = _interesting_text(decoded)
                    # Escalate if auto-exec present with malicious decode
                    if malicious and result.auto_exec_triggers:
                        emit(
                            f"{name}:DEOBF_AUTO",
                            "MACRO.DEOBFUSCATION.AUTOEXEC_PAYLOAD",
                            "EXPLOIT_ATTEMPT",
                            SecurityVerdict.HIGH,
                            0.96,
                            f"Auto-exec macro with reconstructed malicious strings ({', '.join(techniques[:5])}).",
                            36.0,
                            {**base, "techniques": techniques},
                        )
                    else:
                        emit(
                            f"{name}:DEOBF",
                            "MACRO.DEOBFUSCATION.STATIC_RECONSTRUCTION",
                            "EXPLOIT_ATTEMPT" if malicious else "SUSPICIOUS_INFRASTRUCTURE",
                            SecurityVerdict.HIGH if malicious else SecurityVerdict.MEDIUM,
                            0.97 if malicious else 0.90,
                            f"Statically reconstructed obfuscated strings: {', '.join(techniques[:6])}.",
                            28.0 if malicious else 16.0,
                            {
                                **base,
                                "techniques": techniques,
                                "decoded_count": len(result.deobfuscated_items),
                            },
                        )

                if result.dde_indicators:
                    emit(
                        f"{name}:DDE",
                        "MACRO.DDE.ACTIVE_CONTENT",
                        "EXPLOIT_ATTEMPT",
                        SecurityVerdict.HIGH,
                        0.94,
                        f"DDE indicators in '{_safe_text(name, 120)}'.",
                        32.0,
                        {**base, "indicators": result.dde_indicators[:8]},
                    )

                if result.evasion_triggers:
                    emit(
                        f"{name}:EVASION",
                        "MACRO.EVASION.ANTI_ANALYSIS",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.90,
                        "Anti-analysis / sandbox-evasion indicators in macro.",
                        14.0,
                        {**base, "indicators": result.evasion_triggers[:12]},
                    )

                if result.declared_apis:
                    emit(
                        f"{name}:DECLARE",
                        "MACRO.WIN32.DECLARE_API",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.91,
                        "Native Windows API declarations in macro.",
                        14.0,
                        {
                            **base,
                            "declared_apis": result.declared_apis[:12],
                            "count": len(result.declared_apis),
                        },
                    )

                suspicious_co = [
                    t
                    for t in result.createobject_targets
                    if any(
                        k in t.lower()
                        for k in (
                            "wscript", "shell", "xmlhttp", "winhttp",
                            "adodb", "management", "filesystem", "scripting",
                        )
                    )
                ]
                if suspicious_co:
                    emit(
                        f"{name}:CO",
                        "MACRO.CREATEOBJECT.SUSPICIOUS",
                        "SUSPICIOUS_INFRASTRUCTURE",
                        SecurityVerdict.MEDIUM,
                        0.94,
                        "Suspicious CreateObject targets in macro.",
                        16.0,
                        {**base, "targets": suspicious_co[:12]},
                    )

                if result.extracted_urls:
                    all_urls.extend(result.extracted_urls)
                    emit(
                        f"{name}:URL",
                        "MACRO.LINK.EXTRACTED_URI",
                        "IMPERSONATION",
                        SecurityVerdict.LOW,
                        0.93,
                        f"Extracted {len(result.extracted_urls)} URL(s) from macro.",
                        6.0,
                        {
                            **base,
                            "url_count": len(result.extracted_urls),
                            "sample_urls": result.extracted_urls[:5],
                        },
                    )

            if all_urls:
                meta = getattr(context, "metadata", None)
                if meta is not None and hasattr(meta, "data"):
                    existing = meta.data.get("extracted_urls") or []
                    if not isinstance(existing, list):
                        existing = []
                    merged = list(dict.fromkeys(list(existing) + all_urls))[:100]
                    meta.data["extracted_urls"] = merged

            total = (
                min(cat["EXPLOIT_ATTEMPT"], 55.0)
                + min(cat["MALWARE_PAYLOAD"], 35.0)
                + min(cat["SUSPICIOUS_INFRASTRUCTURE"], 30.0)
                + min(cat["IMPERSONATION"], 15.0)
            )
            score = min(max(total, 0.0), 100.0)

            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.SUCCESS,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                score_contribution=score,
                confidence=0.97 if findings else 0.86,
                findings=findings,
            )

        except Exception as exc:
            logger.exception(
                "MacroDeobfuscator failure scan_id=%s", getattr(context, "scan_id", "?")
            )
            return AnalyzerResult(
                analyzer_name=self.name,
                status=AnalyzerExecutionStatus.FAILED,
                execution_time_ms=(time.monotonic() - start) * 1000.0,
                sanitized_error=f"Macro analyzer error: {type(exc).__name__}",
            )

    def _timeout(self, start: float) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name=self.name,
            status=AnalyzerExecutionStatus.TIMEOUT,
            execution_time_ms=(time.monotonic() - start) * 1000.0,
            sanitized_error="Deadline exceeded during VBA macro analysis.",
        )


# Compatibility alias
MacroDeobfuscatorEngine = MacroAnalysisEngine

__all__ = [
    "MacroDeobfuscatorAnalyzer",
    "MacroAnalysisEngine",
    "MacroDeobfuscationEngine",
    "MacroDeobfuscatorEngine",
    "MacroPayloadStore",
    "MacroAnalysisResult",
    "DeobfuscatedPayload",
    "MACRO_DEOBFUSCATOR_VERSION",
]
