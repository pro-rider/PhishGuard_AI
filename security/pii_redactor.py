"""
PhishGuard_AI - Enterprise PII & Secret Sanitization Engine

Defense-in-depth sanitization pipeline for scrubbing credentials, PII,
API keys, JWTs, and dangerous URLs from strings, structured models, and nested object graphs.

Pipeline Architecture:
  Input Object
    │
    ├── 1. Circular Reference & Depth Protection
    ├── 2. SecretStr / Primitive / Container Type Dispatch
    ├── 3. Sensitive Key Normalized Matching
    ├── 4. Complex Object Traversal (Dict, List, Set, Dataclass, Pydantic)
    ├── 5. String-Level Secret / JWT / Basic Auth / PII Redaction
    ├── 6. Comprehensive URL Sanitization (Userinfo, Query, Fragment)
    └── 7. Final Length Truncation (Sanitize-Before-Truncate)
          │
          ▼
    Safe Output
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, SecretStr

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & SAFETY BOUNDS
# ============================================================================

MAX_LOG_STRING_LENGTH = 4096
MAX_DEPTH = 10
MAX_COLLECTION_ITEMS = 100

# ============================================================================
# HARDENED REGEX PATTERNS
# ============================================================================

# 1. PII Patterns
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
PHONE_REGEX = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# 2. Authentication & Credential Patterns
BEARER_TOKEN_REGEX = re.compile(
    r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",
    re.IGNORECASE,
)
BASIC_AUTH_REGEX = re.compile(
    r"\bBasic\s+[A-Za-z0-9+/=]{8,}\b",
    re.IGNORECASE,
)
JWT_REGEX = re.compile(
    r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b"
)
AWS_KEY_REGEX = re.compile(
    r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"
)
PRIVATE_KEY_REGEX = re.compile(
    r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
    re.IGNORECASE,
)

# 3. Generic Key=Value / Key:Value Patterns in Raw Text
GENERIC_SECRET_REGEX = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|token|access[_-]?token|refresh[_-]?token|auth|credentials)\s*[:=]\s*['\"]?([^'\"\s&;]+)",
    re.IGNORECASE,
)

# 4. Embedded URL Detection Regex
URL_REGEX = re.compile(
    r"https?://[^\s\"'>]+",
    re.IGNORECASE,
)

# ============================================================================
# SENSITIVE KEY MATCHING DATA STRUCTURES
# ============================================================================

EXACT_SENSITIVE_KEYS: Set[str] = {
    "password",
    "passwd",
    "secret",
    "secret_key",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
    "session",
    "postgres_password",
    "redis_password",
    "aws_secret_access_key",
    "private_key",
    "x_api_key",
}

# Word boundary sensitive fragments to prevent over-aggressive false positives
SENSITIVE_KEY_SUFFIXES: Tuple[str, ...] = (
    "_password",
    "_passwd",
    "_secret",
    "_key",
    "_token",
    "_auth",
    "_credentials",
    "_cookie",
    "_session",
)

SENSITIVE_KEY_PREFIXES: Tuple[str, ...] = (
    "secret_",
    "api_key_",
    "auth_",
    "session_",
    "token_",
    "private_",
)


class PIIRedactor:
    """Enterprise sanitization engine for scrubbing PII, tokens, and secrets."""

    @classmethod
    def is_sensitive_key(cls, key: str) -> bool:
        """
        Determines whether a dictionary or model field key represents a secret.
        Uses normalized exact checks and structural prefixes/suffixes to avoid
        false positives on keys like 'training_session' or 'tokenizer_config'.
        """
        if not key or not isinstance(key, str):
            return False

        norm_key = key.lower().replace("-", "_").strip()

        if norm_key in EXACT_SENSITIVE_KEYS:
            return True

        if norm_key.endswith(SENSITIVE_KEY_SUFFIXES) or norm_key.startswith(
            SENSITIVE_KEY_PREFIXES
        ):
            return True

        return False

    @classmethod
    def sanitize_url_string(cls, url_str: str) -> str:
        """
        Scrubs userinfo (credentials), query parameters, and fragments from a URL.
        Example: https://user:pass@example.com/login?token=xyz#access_token=123
        Returns: https://user:[REDACTED]@example.com/login?token=[REDACTED_SECRET]#access_token=[REDACTED_SECRET]
        """
        try:
            parsed = urlparse(url_str)
            if not parsed.scheme or not parsed.netloc:
                return url_str

            # 1. Sanitize Userinfo Credentials (user:password@host)
            netloc = parsed.netloc
            if "@" in netloc:
                userinfo, host = netloc.rsplit("@", 1)
                if ":" in userinfo:
                    username, _ = userinfo.split(":", 1)
                    netloc = f"{username}:[REDACTED_SECRET]@{host}"
                else:
                    netloc = f"[REDACTED_SECRET]@{host}"

            # 2. Sanitize Query Parameters
            sanitized_query = ""
            if parsed.query:
                query_params = parse_qsl(parsed.query, keep_blank_values=True)
                sanitized_params: List[Tuple[str, str]] = []

                for k, v in query_params:
                    if cls.is_sensitive_key(k):
                        sanitized_params.append((k, "[REDACTED_SECRET]"))
                    else:
                        sanitized_val = EMAIL_REGEX.sub("[REDACTED_EMAIL]", v)
                        sanitized_params.append((k, sanitized_val))

                sanitized_query = urlencode(sanitized_params)

            # 3. Sanitize URL Fragments (#access_token=secret)
            sanitized_fragment = ""
            if parsed.fragment:
                if "=" in parsed.fragment:
                    frag_parts = parse_qsl(parsed.fragment, keep_blank_values=True)
                    sanitized_frag_parts: List[Tuple[str, str]] = []
                    for k, v in frag_parts:
                        if cls.is_sensitive_key(k):
                            sanitized_frag_parts.append((k, "[REDACTED_SECRET]"))
                        else:
                            sanitized_frag_parts.append((k, v))
                    sanitized_fragment = urlencode(sanitized_frag_parts)
                else:
                    sanitized_fragment = cls.sanitize_text_content(parsed.fragment)

            return urlunparse(
                (
                    parsed.scheme,
                    netloc,
                    parsed.path,
                    parsed.params,
                    sanitized_query,
                    sanitized_fragment,
                )
            )
        except Exception as exc:
            logger.debug("URL parsing failed during sanitization: %s", exc)
            return EMAIL_REGEX.sub("[REDACTED_EMAIL]", url_str)

    @classmethod
    def sanitize_text_content(cls, text: str) -> str:
        """
        Applies primary text scrubbing (secrets, tokens, PII, embedded URLs).
        Note: Truncation is deferred to the caller to ensure complete coverage.
        """
        if not text:
            return text

        # 1. Private Keys
        text = PRIVATE_KEY_REGEX.sub("[REDACTED_PRIVATE_KEY]", text)

        # 2. JWT Tokens
        text = JWT_REGEX.sub("[REDACTED_JWT]", text)

        # 3. Bearer & Basic Auth
        text = BEARER_TOKEN_REGEX.sub("Bearer [REDACTED_TOKEN]", text)
        text = BASIC_AUTH_REGEX.sub("Basic [REDACTED_CREDENTIALS]", text)

        # 4. AWS Keys
        text = AWS_KEY_REGEX.sub("[REDACTED_AWS_KEY]", text)

        # 5. Generic inline secret assignments ("password=super-secret")
        text = GENERIC_SECRET_REGEX.sub(r"\1=[REDACTED_SECRET]", text)

        # 6. Emails and Phone Numbers
        text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
        text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)

        # 7. Embedded URLs
        if "http://" in text or "https://" in text:

            def _url_replacer(match: re.Match[str]) -> str:
                return cls.sanitize_url_string(match.group(0))

            text = URL_REGEX.sub(_url_replacer, text)

        # 8. Neutralize CRLF Injection
        return text.replace("\r", "\\r").replace("\n", "\\n")

    @classmethod
    def sanitize_value(
        cls,
        key: str | None,
        val: Any,
        depth: int = 0,
        visited: Set[int] | None = None,
    ) -> Any:
        """
        Recursively scrubs secrets and PII across complex structures.
        Includes depth bounding, circular reference protection, and collection size limits.
        """
        if visited is None:
            visited = set()

        # 1. Handle Pydantic SecretStr explicitly
        if isinstance(val, SecretStr):
            return "[REDACTED_SECRET]"

        # 2. Depth Limit Protection
        if depth > MAX_DEPTH:
            return "<MAX_DEPTH_EXCEEDED>"

        # 3. Key-based Redaction Check
        if key and cls.is_sensitive_key(key):
            return "[REDACTED_SECRET]"

        # 4. Handle None and Booleans
        if val is None or isinstance(val, bool):
            return val

        # 5. Handle Strings (Sanitize FIRST, Truncate LAST)
        if isinstance(val, str):
            sanitized_str = cls.sanitize_text_content(val)
            if len(sanitized_str) > MAX_LOG_STRING_LENGTH:
                return sanitized_str[:MAX_LOG_STRING_LENGTH] + "...[TRUNCATED]"
            return sanitized_str

        # 6. Handle Primitive Numbers
        if isinstance(val, (int, float)):
            return val

        # 7. Handle Bytes
        if isinstance(val, bytes):
            return f"<RAW_BYTES_LENGTH_{len(val)}>"

        # 8. Circular Reference Protection for Reference Types
        val_id = id(val)
        if val_id in visited:
            return "<CIRCULAR_REFERENCE>"
        visited.add(val_id)

        try:
            # 9. Handle Pydantic Models
            if isinstance(val, BaseModel):
                return cls.sanitize_value(
                    key, val.model_dump(), depth + 1, visited
                )

            # 10. Handle Dataclasses
            if dataclasses.is_dataclass(val):
                return cls.sanitize_value(
                    key, dataclasses.asdict(val), depth + 1, visited
                )

            # 11. Handle Dictionaries
            if isinstance(val, dict):
                sanitized_dict: Dict[str, Any] = {}
                for idx, (k, v) in enumerate(val.items()):
                    if idx >= MAX_COLLECTION_ITEMS:
                        sanitized_dict["_truncated"] = (
                            f"<{len(val) - MAX_COLLECTION_ITEMS}_ITEMS_OMITTED>"
                        )
                        break
                    str_k = str(k)
                    sanitized_dict[str_k] = cls.sanitize_value(
                        str_k, v, depth + 1, visited
                    )
                return sanitized_dict

            # 12. Handle Lists, Tuples, and Sets
            if isinstance(val, (list, tuple, set)):
                sanitized_items: List[Any] = []
                for idx, item in enumerate(val):
                    if idx >= MAX_COLLECTION_ITEMS:
                        sanitized_items.append(
                            f"<{len(val) - MAX_COLLECTION_ITEMS}_ITEMS_OMITTED>"
                        )
                        break
                    sanitized_items.append(
                        cls.sanitize_value(key, item, depth + 1, visited)
                    )

                if isinstance(val, tuple):
                    return tuple(sanitized_items)
                if isinstance(val, set):
                    try:
                        return set(sanitized_items)
                    except TypeError:
                        # Fallback if sanitized items are unhashable
                        return sanitized_items
                return sanitized_items

            # 13. Fallback for Arbitrary Custom Objects
            try:
                obj_str = str(val)
                sanitized_obj_str = cls.sanitize_text_content(obj_str)
                if len(sanitized_obj_str) > MAX_LOG_STRING_LENGTH:
                    return (
                        sanitized_obj_str[:MAX_LOG_STRING_LENGTH]
                        + "...[TRUNCATED]"
                    )
                return sanitized_obj_str
            except Exception as exc:
                logger.debug("Failed to stringify custom object: %s", exc)
                return f"<{val.__class__.__name__}_UNSTRINGIFIABLE>"

        finally:
            # Remove from visited set to allow clean sibling traversal
            visited.remove(val_id)
