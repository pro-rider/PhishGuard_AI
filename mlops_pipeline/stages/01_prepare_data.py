"""
PhishGuard AI - Stage 01: Data Preparation & Feature Extraction
================================================================

Version: 14.0.0
File Path: src/stages/01_prepare_data.py

Production-grade data ingestion, validation, normalization, feature extraction,
leakage prevention, stratified/group-aware dataset splitting, artifact generation,
quality gates, reproducibility metadata, and dataset fingerprinting.

Security goals
--------------
1. Fail closed on malformed or ambiguous input.
2. Prevent exact and normalized duplicate leakage across train/validation/test.
3. Support group-aware splitting by normalized hostname/domain.
4. Preserve only model-safe numerical features in model artifacts.
5. Generate reproducible manifests and SHA-256 fingerprints.
6. Keep raw URL payloads out of processed model matrices.
7. Detect class imbalance and insufficient class cardinality.
8. Provide deterministic feature ordering.
9. Produce machine-readable preparation metadata for downstream MLOps stages.

Important
---------
This module performs FEATURE EXTRACTION only. It does not fetch URLs, resolve DNS,
connect to external hosts, execute downloaded content, or perform active scanning.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl, unquote, urlsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger("src.stages.01_prepare_data")


# =============================================================================
# VERSION / CONSTANTS
# =============================================================================

STAGE_VERSION = "14.0.0"
SCHEMA_VERSION = "1.0"

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_SIZE = 0.10

SUPPORTED_INPUT_SUFFIXES = {".csv", ".parquet"}

# These are intentionally heuristic features, NOT authoritative threat intel.
SUSPICIOUS_TLDS = frozenset(
    {
        ".xyz",
        ".top",
        ".work",
        ".gq",
        ".ml",
        ".ga",
        ".cf",
        ".tk",
        ".club",
        ".info",
    }
)

SUSPICIOUS_KEYWORDS = (
    "login",
    "verify",
    "account",
    "update",
    "banking",
    "secure",
    "signin",
    "paypal",
    "credential",
    "confirm",
    "password",
    "wallet",
    "invoice",
    "payment",
    "authorize",
    "authentication",
)

# Unicode controls frequently relevant to obfuscation.
ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",
        "\u200c",
        "\u200d",
        "\u2060",
        "\ufeff",
    }
)

# Characters commonly involved in URL/hostname obfuscation.
SUSPICIOUS_UNICODE_CATEGORIES = frozenset(
    {
        "Cf",  # Format/control-like invisible characters
        "Cs",  # Surrogates
    }
)

# Conservative maximum URL length used for quality validation.
DEFAULT_MAX_URL_LENGTH = 8192

# Regexes are used only for feature extraction/validation.
HEX_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
MULTIPLE_SLASH_RE = re.compile(r"//+")
DIGIT_RE = re.compile(r"\d")
LETTER_RE = re.compile(r"[A-Za-z]")
SPECIAL_CHAR_RE = re.compile(r"[^A-Za-z0-9]")

# Conservative domain-like pattern.
DOMAIN_LABEL_RE = re.compile(
    r"^(?=.{1,253}$)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)


# =============================================================================
# DATA CONTRACTS
# =============================================================================


@dataclass(frozen=True)
class PreparationConfig:
    """
    Immutable configuration used by the preparation stage.
    """

    raw_data_path: str
    output_dir: str
    target_column: str = "label"
    url_column: str = "url"

    test_size: float = DEFAULT_TEST_SIZE
    val_size: float = DEFAULT_VAL_SIZE

    random_state: int = DEFAULT_RANDOM_STATE

    group_split: bool = True
    group_column: Optional[str] = None

    min_samples: int = 100
    min_samples_per_class: int = 2

    min_positive_ratio: float = 0.001
    max_positive_ratio: float = 0.999

    max_url_length: int = DEFAULT_MAX_URL_LENGTH

    fail_on_invalid_rows: bool = True
    drop_exact_duplicates: bool = True
    drop_normalized_duplicates: bool = True

    write_parquet: bool = True
    write_csv: bool = True

    artifact_prefix: str = ""

    def validate(self) -> None:
        """Validate stage configuration before touching the dataset."""

        if not self.raw_data_path:
            raise ValueError("raw_data_path must not be empty.")

        if not self.output_dir:
            raise ValueError("output_dir must not be empty.")

        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if not 0 <= self.val_size < 1:
            raise ValueError("val_size must be between 0 and 1.")

        if self.test_size + self.val_size >= 1:
            raise ValueError(
                "test_size + val_size must be strictly less than 1."
            )

        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1.")

        if self.min_samples_per_class < 1:
            raise ValueError("min_samples_per_class must be >= 1.")

        if not 0 <= self.min_positive_ratio <= 1:
            raise ValueError("min_positive_ratio must be between 0 and 1.")

        if not 0 <= self.max_positive_ratio <= 1:
            raise ValueError("max_positive_ratio must be between 0 and 1.")

        if self.min_positive_ratio > self.max_positive_ratio:
            raise ValueError(
                "min_positive_ratio cannot exceed max_positive_ratio."
            )

        if self.max_url_length < 64:
            raise ValueError("max_url_length is unrealistically small.")


@dataclass(frozen=True)
class DatasetFingerprint:
    """
    Reproducibility identity for an input dataset.
    """

    sha256: str
    row_count: int
    column_count: int
    columns: Tuple[str, ...]


@dataclass(frozen=True)
class SplitSummary:
    """
    Summary statistics for one dataset partition.
    """

    name: str
    samples: int
    positives: int
    negatives: int
    positive_ratio: float
    groups: int


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def _json_default(value: Any) -> Any:
    """JSON serializer for numpy/pandas scalar values."""

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return list(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _atomic_write_text(path: Path, content: str) -> None:
    """
    Atomically write text to disk.

    The temporary file is created in the destination directory to keep the
    final replace operation on the same filesystem.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_write_dataframe(
    df: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
) -> None:
    """Atomically write a DataFrame as CSV or Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        if path.suffix.lower() == ".csv":
            df.to_csv(tmp_path, index=index)

        elif path.suffix.lower() == ".parquet":
            df.to_parquet(tmp_path, index=index)

        else:
            raise ValueError(
                f"Unsupported artifact format: {path.suffix}"
            )

        os.replace(tmp_path, path)

    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate SHA-256 fingerprint of a file without loading the entire file
    into memory.
    """

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Deterministically fingerprint a DataFrame.

    This is intended for pipeline reproducibility and leakage auditing.
    """

    normalized = df.copy()

    normalized = normalized.reindex(sorted(normalized.columns), axis=1)

    for column in normalized.columns:
        if pd.api.types.is_object_dtype(normalized[column]):
            normalized[column] = (
                normalized[column]
                .astype("string")
                .fillna("<NA>")
            )

    hashed = pd.util.hash_pandas_object(
        normalized,
        index=True,
        categorize=True,
    )

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            list(normalized.columns),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(hashed.to_numpy(dtype=np.uint64).tobytes())

    return digest.hexdigest()


# =============================================================================
# URL NORMALIZATION
# =============================================================================


def normalize_url(raw_url: Any) -> str:
    """
    Produce a deterministic representation of a URL for deduplication.

    This function does NOT resolve DNS or make network requests.

    Normalization intentionally remains conservative:
    - Unicode normalization
    - whitespace trimming
    - lower-case scheme/hostname
    - removal of default ports
    - fragment removal
    - stable query ordering
    - trailing-dot removal from hostname

    The original URL is never modified in-place.
    """

    if raw_url is None or pd.isna(raw_url):
        return ""

    value = str(raw_url).strip()

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)

    # Remove invisible Unicode characters for deduplication purposes.
    for char in ZERO_WIDTH_CHARS:
        value = value.replace(char, "")

    if "://" not in value:
        value = f"http://{value}"

    try:
        parsed = urlsplit(value)

        scheme = (parsed.scheme or "http").lower()

        hostname = parsed.hostname or ""
        hostname = hostname.rstrip(".").lower()

        # Preserve IDN semantics but normalize Unicode.
        hostname = unicodedata.normalize("NFKC", hostname)

        try:
            hostname_ascii = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError:
            hostname_ascii = hostname

        username = parsed.username or ""
        password = parsed.password or ""

        # Credentials are normalized but not exposed in generated features.
        auth = ""
        if username:
            auth = username
            if password:
                auth += f":{password}"
            auth += "@"

        port = parsed.port

        port_text = ""
        if port is not None:
            default_port = (
                (scheme == "http" and port == 80)
                or (scheme == "https" and port == 443)
            )
            if not default_port:
                port_text = f":{port}"

        path = parsed.path or "/"

        # Percent encoding is normalized conservatively.
        path = re.sub(
            r"%[0-9A-Fa-f]{2}",
            lambda m: m.group(0).upper(),
            path,
        )

        query_pairs = parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=False,
        )

        query_pairs.sort(key=lambda item: (item[0], item[1]))

        query = "&".join(
            f"{key}={value}"
            for key, value in query_pairs
        )

        normalized = f"{scheme}://{auth}{hostname_ascii}{port_text}{path}"

        if query:
            normalized += f"?{query}"

        # Deliberately remove fragments because they are not sent to servers.
        return normalized

    except (ValueError, UnicodeError):
        return value.lower()


def extract_group_key(raw_url: Any) -> str:
    """
    Extract a grouping key suitable for leakage-resistant dataset splitting.

    Prefer registered hostname. Fall back to normalized URL when hostname
    cannot be parsed.

    This function performs no network activity.
    """

    normalized = normalize_url(raw_url)

    if not normalized:
        return "<EMPTY>"

    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname

        if not hostname:
            return normalized

        hostname = hostname.lower().rstrip(".")

        # Remove a leading www. because www.example.com and
        # example.com should normally remain in the same evaluation group.
        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except ValueError:
        return normalized


# =============================================================================
# ENTROPY / URL ANALYTICS
# =============================================================================


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy in bits per character.

    Returns 0 for empty input.
    """

    if not text:
        return 0.0

    value = str(text)

    counts = pd.Series(list(value), dtype="string").value_counts(
        normalize=True
    )

    entropy = -float(
        np.sum(
            counts.to_numpy(dtype=float)
            * np.log2(counts.to_numpy(dtype=float))
        )
    )

    return entropy


def _safe_hostname(hostname: Optional[str]) -> str:
    """Normalize hostname representation without network access."""

    if not hostname:
        return ""

    hostname = unicodedata.normalize("NFKC", hostname).strip().rstrip(".")

    return hostname.lower()


def _is_valid_ipv4_or_ipv6(hostname: str) -> bool:
    """Return True when hostname is a syntactically valid IP address."""

    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _contains_unicode(text: str) -> bool:
    """Detect non-ASCII Unicode characters."""

    return any(ord(char) > 127 for char in text)


def _unicode_control_count(text: str) -> int:
    """Count invisible/control-like Unicode characters."""

    return sum(
        1
        for char in text
        if char in ZERO_WIDTH_CHARS
        or unicodedata.category(char) in SUSPICIOUS_UNICODE_CATEGORIES
    )


def _digit_ratio(text: str) -> float:
    """Calculate the proportion of digits in a string."""

    if not text:
        return 0.0

    return float(sum(char.isdigit() for char in text) / len(text))


def _letter_ratio(text: str) -> float:
    """Calculate the proportion of alphabetic characters."""

    if not text:
        return 0.0

    return float(sum(char.isalpha() for char in text) / len(text))


def _special_ratio(text: str) -> float:
    """Calculate the proportion of non-alphanumeric characters."""

    if not text:
        return 0.0

    return float(
        sum(not char.isalnum() for char in text) / len(text)
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def extract_url_features(url: Any) -> Dict[str, Union[int, float]]:
    """
    Extract security-oriented lexical and structural features from a URL.

    Important:
        These are heuristic telemetry features. They are not proof that a URL
        is malicious or benign.

    No DNS lookup, HTTP request, URL fetch, or external API call is performed.
    """

    if url is None or pd.isna(url):
        raw = ""
    else:
        raw = str(url).strip()

    normalized = unicodedata.normalize("NFKC", raw)

    parse_target = normalized

    if parse_target and "://" not in parse_target:
        parse_target = f"http://{parse_target}"

    try:
        parsed = urlsplit(parse_target)
    except ValueError:
        parsed = urlsplit("http://invalid.local")

    hostname = _safe_hostname(parsed.hostname)

    try:
        port = parsed.port
    except ValueError:
        port = None

    scheme = (parsed.scheme or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # -------------------------------------------------------------------------
    # Host / domain structure
    # -------------------------------------------------------------------------

    host_labels = [
        label for label in hostname.split(".") if label
    ]

    subdomain_count = max(len(host_labels) - 2, 0)

    tld = ""
    if len(host_labels) >= 2:
        tld = f".{host_labels[-1]}"

    has_suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    is_ip = int(_is_valid_ipv4_or_ipv6(hostname))

    # -------------------------------------------------------------------------
    # URL lexical features
    # -------------------------------------------------------------------------

    url_length = len(normalized)
    domain_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    dot_count = normalized.count(".")
    hyphen_count = normalized.count("-")
    underscore_count = normalized.count("_")
    slash_count = normalized.count("/")
    backslash_count = normalized.count("\\")
    at_symbol_present = int("@" in normalized)

    # A second "//" after the scheme is often useful as a heuristic.
    path_like = f"{path}?{query}" if query else path
    double_slash_redirect = int("//" in path_like)

    # -------------------------------------------------------------------------
    # Encodings / Unicode / obfuscation
    # -------------------------------------------------------------------------

    percent_encoded_count = len(
        HEX_ESCAPE_RE.findall(normalized)
    )

    unicode_present = int(_contains_unicode(normalized))
    unicode_control_count = _unicode_control_count(normalized)

    punycode_present = int(
        any(label.startswith("xn--") for label in host_labels)
    )

    non_ascii_count = sum(
        1 for char in normalized if ord(char) > 127
    )

    # -------------------------------------------------------------------------
    # Authentication / port indicators
    # -------------------------------------------------------------------------

    has_credentials = int(
        parsed.username is not None or parsed.password is not None
    )

    has_non_default_port = int(
        port is not None
        and not (
            (scheme == "http" and port == 80)
            or (scheme == "https" and port == 443)
        )
    )

    # -------------------------------------------------------------------------
    # Suspicious lexical keywords
    # -------------------------------------------------------------------------

    lowered = normalized.lower()

    keyword_hits = [
        keyword
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lowered
    ]

    keyword_count = len(keyword_hits)

    # -------------------------------------------------------------------------
    # Character composition
    # -------------------------------------------------------------------------

    digit_count = sum(char.isdigit() for char in normalized)
    letter_count = sum(char.isalpha() for char in normalized)

    # -------------------------------------------------------------------------
    # Query structure
    # -------------------------------------------------------------------------

    query_parameter_count = (
        len(
            parse_qsl(
                query,
                keep_blank_values=True,
                strict_parsing=False,
            )
        )
        if query
        else 0
    )

    # -------------------------------------------------------------------------
    # Hostname validation
    # -------------------------------------------------------------------------

    hostname_is_domain = int(
        bool(DOMAIN_LABEL_RE.match(hostname))
    )

    # -------------------------------------------------------------------------
    # Entropy
    # -------------------------------------------------------------------------

    url_entropy = calculate_entropy(normalized)
    hostname_entropy = calculate_entropy(hostname)

    return {
        # Basic dimensions
        "url_length": int(url_length),
        "domain_length": int(domain_length),
        "path_length": int(path_length),
        "query_length": int(query_length),

        # Structural counts
        "dot_count": int(dot_count),
        "hyphen_count": int(hyphen_count),
        "underscore_count": int(underscore_count),
        "slash_count": int(slash_count),
        "backslash_count": int(backslash_count),
        "subdomain_count": int(subdomain_count),

        # Suspicious syntax
        "at_symbol_present": int(at_symbol_present),
        "double_slash_redirect": int(double_slash_redirect),
        "is_ip_address": int(is_ip),
        "has_suspicious_tld": int(has_suspicious_tld),

        # Authentication / ports
        "has_credentials": int(has_credentials),
        "has_non_default_port": int(has_non_default_port),

        # Encoding / Unicode
        "percent_encoded_count": int(percent_encoded_count),
        "unicode_present": int(unicode_present),
        "unicode_control_count": int(unicode_control_count),
        "punycode_present": int(punycode_present),
        "non_ascii_count": int(non_ascii_count),

        # Lexical
        "keyword_count": int(keyword_count),

        # Composition
        "digit_count": int(digit_count),
        "letter_count": int(letter_count),
        "digit_ratio": _digit_ratio(normalized),
        "letter_ratio": _letter_ratio(normalized),
        "special_char_ratio": _special_ratio(normalized),

        # Query
        "query_parameter_count": int(query_parameter_count),

        # Host validation
        "hostname_is_domain": int(hostname_is_domain),

        # Entropy
        "url_entropy": float(url_entropy),
        "hostname_entropy": float(hostname_entropy),

        # URL semantics
        "has_fragment": int(bool(fragment)),
        "uses_https": int(scheme == "https"),
        "uses_http": int(scheme == "http"),
        "unknown_scheme": int(
            bool(scheme) and scheme not in {"http", "https"}
        ),
    }


# =============================================================================
# DATASET LOADING
# =============================================================================


def _load_raw_dataset(path: Path) -> pd.DataFrame:
    """
    Load supported dataset format.

    CSV parsing intentionally fails on malformed records rather than silently
    dropping potentially important security examples.
    """

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported input file format '{suffix}'. "
            f"Expected one of: {sorted(SUPPORTED_INPUT_SUFFIXES)}"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset does not exist: '{path}'"
        )

    if not path.is_file():
        raise ValueError(
            f"Raw dataset path is not a regular file: '{path}'"
        )

    logger.info(
        "Loading raw dataset: %s (%s)",
        path,
        suffix,
    )

    if suffix == ".parquet":
        df = pd.read_parquet(path)

    else:
        df = pd.read_csv(
            path,
            on_bad_lines="error",
            low_memory=False,
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset loader did not return a pandas DataFrame.")

    return df


# =============================================================================
# DATA VALIDATION
# =============================================================================


def _validate_columns(
    df: pd.DataFrame,
    target_column: str,
    url_column: str,
) -> None:
    """Validate required input columns."""

    missing = [
        column
        for column in (target_column, url_column)
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Required dataset columns are missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _coerce_binary_target(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """
    Convert target labels into strict binary integers.

    Accepted representations:
        0 / 1
        True / False
        common textual benign/phishing representations
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")

    # Numeric path.
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        unique = set(numeric.astype(int).unique())

        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Target column '{column_name}' must contain only 0/1. "
                f"Observed values: {sorted(unique)}"
            )

        return numeric.astype("int8")

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "benign": 0,
        "legitimate": 0,
        "legit": 0,
        "safe": 0,
        "normal": 0,
        "phishing": 1,
        "phish": 1,
        "malicious": 1,
        "malware": 1,
        "fraud": 1,
        "unsafe": 1,
    }

    converted = normalized.map(mapping)

    if converted.isna().any():
        invalid = (
            normalized[converted.isna()]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Target column '{column_name}' contains unsupported labels: "
            f"{invalid[:20]}"
        )

    return converted.astype("int8")


def _validate_url_column(
    series: pd.Series,
    *,
    max_url_length: int,
) -> pd.Series:
    """
    Normalize URL input and reject obviously invalid records.
    """

    values = (
        series.astype("string")
        .fillna("")
        .str.strip()
    )

    too_long = values.str.len() > max_url_length

    if too_long.any():
        count = int(too_long.sum())

        raise ValueError(
            f"{count} URL records exceed max_url_length="
            f"{max_url_length}."
        )

    empty_count = int((values == "").sum())

    if empty_count:
        logger.warning(
            "Dataset contains %d empty URL values.",
            empty_count,
        )

    return values


# =============================================================================
# DUPLICATE / LEAKAGE AUDITING
# =============================================================================


def _remove_duplicate_rows(
    df: pd.DataFrame,
    *,
    target_column: str,
    url_column: str,
    remove_exact: bool,
    remove_normalized: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove exact and normalized URL duplicates.

    For phishing detection, normalized URL duplicates are particularly
    important because exact-row deduplication is insufficient.
    """

    working = df.copy()

    stats = {
        "initial_rows": len(working),
        "exact_duplicates_removed": 0,
        "normalized_url_duplicates_removed": 0,
    }

    if remove_exact:
        before = len(working)

        working = working.drop_duplicates(
            keep="first"
        ).reset_index(drop=True)

        stats["exact_duplicates_removed"] = before - len(working)

    if remove_normalized:
        working["_normalized_url"] = working[url_column].map(
            normalize_url
        )

        before = len(working)

        # Empty normalized URLs should not collapse all empty records into one.
        non_empty_mask = working["_normalized_url"].ne("")

        duplicate_mask = (
            non_empty_mask
            & working["_normalized_url"].duplicated(
                keep="first"
            )
        )

        working = working.loc[~duplicate_mask].copy()

        stats["normalized_url_duplicates_removed"] = (
            before - len(working)
        )

    return working.reset_index(drop=True), stats


def _audit_label_conflicts(
    df: pd.DataFrame,
    *,
    target_column: str,
    url_column: str,
) -> Dict[str, Any]:
    """
    Detect normalized URLs that occur with conflicting labels.

    Conflicting labels are a major data-quality signal and should not simply
    be silently resolved.
    """

    temp = df.copy()

    temp["_normalized_url"] = temp[url_column].map(
        normalize_url
    )

    non_empty = temp[temp["_normalized_url"].ne("")]

    if non_empty.empty:
        return {
            "conflicting_normalized_urls": 0,
            "conflicting_rows": 0,
        }

    grouped = (
        non_empty.groupby("_normalized_url")[target_column]
        .nunique()
    )

    conflicting = grouped[grouped > 1]

    if conflicting.empty:
        return {
            "conflicting_normalized_urls": 0,
            "conflicting_rows": 0,
        }

    conflict_keys = set(conflicting.index)

    conflicting_rows = int(
        non_empty["_normalized_url"].isin(conflict_keys).sum()
    )

    return {
        "conflicting_normalized_urls": int(len(conflicting)),
        "conflicting_rows": conflicting_rows,
        "sample_conflicting_urls": list(conflict_keys)[:10],
    }


# =============================================================================
# FEATURE MATRIX CONSTRUCTION
# =============================================================================


def _build_feature_matrix(
    df: pd.DataFrame,
    *,
    target_column: str,
    url_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build deterministic numerical feature matrix.

    Returns
    -------
    X:
        Numerical model features only.
    y:
        Binary labels.
    groups:
        Leakage-control grouping key.
    """

    url_features = pd.DataFrame(
        df[url_column].map(extract_url_features).tolist(),
        index=df.index,
    )

    # Preserve any existing numerical telemetry columns supplied by the dataset.
    numeric_candidates = df.drop(
        columns=[target_column, url_column],
        errors="ignore",
    ).select_dtypes(
        include=[np.number, "bool"]
    )

    if not numeric_candidates.empty:
        numeric_candidates = numeric_candidates.copy()

        # Ensure feature names do not collide with URL features.
        collisions = set(numeric_candidates.columns).intersection(
            url_features.columns
        )

        if collisions:
            rename_map = {
                column: f"input_{column}"
                for column in collisions
            }
            numeric_candidates = numeric_candidates.rename(
                columns=rename_map
            )

    X = pd.concat(
        [
            numeric_candidates,
            url_features,
        ],
        axis=1,
    )

    # Remove columns with no variance.
    nunique = X.nunique(dropna=False)

    constant_columns = [
        column
        for column in X.columns
        if nunique[column] <= 1
    ]

    if constant_columns:
        logger.warning(
            "Removing %d constant feature(s): %s",
            len(constant_columns),
            constant_columns,
        )

        X = X.drop(columns=constant_columns)

    # Strict numeric conversion.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Replace infinities before missing-value checks.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # The feature extractor should not create missing values.
    # Existing telemetry columns may.
    missing_counts = X.isna().sum()

    missing_features = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    if missing_features:
        logger.warning(
            "Missing numerical values detected: %s",
            missing_features,
        )

        # Tree models can differ in NaN handling across versions.
        # Fail closed rather than allowing silent downstream behavior.
        raise ValueError(
            "Feature matrix contains missing values. "
            "Clean/impute source numerical telemetry before training. "
            f"Missing counts: {missing_features}"
        )

    # Stable deterministic feature ordering.
    X = X.reindex(
        sorted(X.columns),
        axis=1,
    )

    y = df[target_column].astype("int8")

    groups = df[url_column].map(
        extract_group_key
    )

    return X, y, groups


# =============================================================================
# SPLITTING
# =============================================================================


def _validate_split_class_counts(
    y: pd.Series,
    *,
    min_samples_per_class: int,
) -> None:
    """Validate class cardinality before stratification."""

    counts = y.value_counts().to_dict()

    for class_value in (0, 1):
        count = int(counts.get(class_value, 0))

        if count < min_samples_per_class:
            raise ValueError(
                f"Class {class_value} contains only {count} sample(s). "
                f"At least {min_samples_per_class} are required."
            )


def _split_with_stratification(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """
    Standard stratified split.

    The initial test split is followed by a validation split from the remaining
    training data.
    """

    indices = np.arange(len(X))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y.to_numpy(),
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y.iloc[train_val_idx].to_numpy(),
    )

    return {
        "train": np.asarray(train_idx),
        "val": np.asarray(val_idx),
        "test": np.asarray(test_idx),
    }


def _split_with_groups(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """
    Group-aware split to prevent the same host/domain from crossing partitions.

    GroupShuffleSplit is intentionally used rather than ordinary row-level
    splitting because phishing datasets commonly contain many URLs belonging
    to the same domain.
    """

    indices = np.arange(len(X))
    group_values = groups.to_numpy()

    if len(np.unique(group_values)) < 3:
        raise ValueError(
            "Group-aware splitting requires at least 3 unique groups."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_val_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=group_values,
        )
    )

    remaining_groups = group_values[train_val_idx]

    relative_val_size = val_size / (1.0 - test_size)

    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state,
    )

    train_local_idx, val_local_idx = next(
        val_splitter.split(
            X.iloc[train_val_idx],
            y.iloc[train_val_idx],
            groups=remaining_groups,
        )
    )

    train_idx = train_val_idx[train_local_idx]
    val_idx = train_val_idx[val_local_idx]

    return {
        "train": np.asarray(train_idx),
        "val": np.asarray(val_idx),
        "test": np.asarray(test_idx),
    }


def _assert_split_integrity(
    split_indices: Mapping[str, np.ndarray],
    groups: pd.Series,
) -> None:
    """
    Fail if samples or grouping keys overlap across partitions.
    """

    names = ("train", "val", "test")

    # Sample-level overlap.
    for i, left_name in enumerate(names):
        left = set(split_indices[left_name].tolist())

        for right_name in names[i + 1 :]:
            right = set(split_indices[right_name].tolist())

            overlap = left.intersection(right)

            if overlap:
                raise RuntimeError(
                    f"Sample leakage detected between '{left_name}' "
                    f"and '{right_name}'. Overlap count={len(overlap)}"
                )

    # Group-level overlap.
    group_sets = {
        name: set(groups.iloc[idx].astype(str).tolist())
        for name, idx in split_indices.items()
    }

    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            overlap = group_sets[left_name].intersection(
                group_sets[right_name]
            )

            if overlap:
                raise RuntimeError(
                    f"Group leakage detected between '{left_name}' "
                    f"and '{right_name}'. "
                    f"Overlap count={len(overlap)}; "
                    f"examples={list(overlap)[:10]}"
                )


# =============================================================================
# SPLIT QUALITY
# =============================================================================


def _summarize_split(
    name: str,
    y: pd.Series,
    groups: pd.Series,
) -> SplitSummary:
    """Generate partition-level statistics."""

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    return SplitSummary(
        name=name,
        samples=len(y),
        positives=positives,
        negatives=negatives,
        positive_ratio=float(y.mean()) if len(y) else 0.0,
        groups=int(groups.nunique(dropna=False)),
    )


def _validate_split_distribution(
    summaries: Sequence[SplitSummary],
    *,
    min_positive_ratio: float,
    max_positive_ratio: float,
) -> None:
    """
    Ensure every partition has a sensible class distribution.

    Group-aware splitting can legitimately produce more variation than
    stratified row splitting, so this is a quality gate rather than a
    requirement for exact equality.
    """

    for summary in summaries:
        if summary.samples == 0:
            raise RuntimeError(
                f"Split '{summary.name}' is empty."
            )

        if not (
            min_positive_ratio
            <= summary.positive_ratio
            <= max_positive_ratio
        ):
            raise RuntimeError(
                f"Split '{summary.name}' has an unacceptable positive "
                f"class ratio of {summary.positive_ratio:.4f}. "
                f"Expected [{min_positive_ratio}, {max_positive_ratio}]."
            )


# =============================================================================
# ARTIFACT MANIFEST
# =============================================================================


def _feature_manifest(
    X: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate deterministic feature metadata."""

    features: List[Dict[str, Any]] = []

    for position, column in enumerate(X.columns):
        series = X[column]

        features.append(
            {
                "position": position,
                "name": str(column),
                "dtype": str(series.dtype),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "n_unique": int(series.nunique(dropna=False)),
            }
        )

    return {
        "feature_count": len(features),
        "features": features,
    }


def _write_json(
    data: Mapping[str, Any],
    path: Path,
) -> None:
    """Write deterministic JSON metadata."""

    payload = json.dumps(
        data,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )

    _atomic_write_text(
        path,
        payload + "\n",
    )


# =============================================================================
# MAIN PREPARATION PIPELINE
# =============================================================================


def prepare_dataset(
    raw_data_path: Union[str, Path],
    output_dir: Union[str, Path],
    target_column: str = "label",
    url_column: str = "url",
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    group_split: bool = True,
    group_column: Optional[str] = None,
    min_samples: int = 100,
    min_samples_per_class: int = 2,
    min_positive_ratio: float = 0.001,
    max_positive_ratio: float = 0.999,
    max_url_length: int = DEFAULT_MAX_URL_LENGTH,
    fail_on_invalid_rows: bool = True,
    drop_exact_duplicates: bool = True,
    drop_normalized_duplicates: bool = True,
    write_parquet: bool = True,
    write_csv: bool = True,
    artifact_prefix: str = "",
) -> Dict[str, Any]:
    """
    Execute the complete Stage 01 preparation pipeline.

    Parameters
    ----------
    raw_data_path:
        Raw CSV or Parquet dataset.

    output_dir:
        Destination directory for processed artifacts.

    target_column:
        Binary target column.

    url_column:
        URL/text input column.

    test_size:
        Fraction of the complete dataset assigned to test.

    val_size:
        Fraction of the complete dataset assigned to validation.

    random_state:
        Deterministic random seed.

    group_split:
        When True, split by normalized hostname/domain to reduce leakage.

    group_column:
        Optional explicit grouping column. If supplied, it must exist in the
        dataset and is preferred over hostname grouping.

    min_samples:
        Minimum total dataset size.

    min_samples_per_class:
        Minimum examples required for each binary class.

    min_positive_ratio / max_positive_ratio:
        Dataset class-balance quality gates.

    max_url_length:
        Maximum permitted URL/string length.

    fail_on_invalid_rows:
        Reserved for future row-level quarantine handling. Current production
        behavior is fail-closed.

    drop_exact_duplicates:
        Remove exact duplicate rows.

    drop_normalized_duplicates:
        Remove duplicate normalized URLs.

    write_parquet:
        Write Parquet artifacts.

    write_csv:
        Write CSV artifacts.

    artifact_prefix:
        Optional artifact filename prefix.

    Returns
    -------
    Dict[str, Any]
        Complete preparation summary and artifact metadata.
    """

    config = PreparationConfig(
        raw_data_path=str(raw_data_path),
        output_dir=str(output_dir),
        target_column=target_column,
        url_column=url_column,
        test_size=float(test_size),
        val_size=float(val_size),
        random_state=int(random_state),
        group_split=bool(group_split),
        group_column=group_column,
        min_samples=int(min_samples),
        min_samples_per_class=int(min_samples_per_class),
        min_positive_ratio=float(min_positive_ratio),
        max_positive_ratio=float(max_positive_ratio),
        max_url_length=int(max_url_length),
        fail_on_invalid_rows=bool(fail_on_invalid_rows),
        drop_exact_duplicates=bool(drop_exact_duplicates),
        drop_normalized_duplicates=bool(drop_normalized_duplicates),
        write_parquet=bool(write_parquet),
        write_csv=bool(write_csv),
        artifact_prefix=str(artifact_prefix),
    )

    config.validate()

    raw_path = Path(config.raw_data_path).resolve()
    out_path = Path(config.output_dir).resolve()

    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("=" * 78)
    logger.info(
        "PHISHGUARD AI STAGE 01 - DATA PREPARATION v%s",
        STAGE_VERSION,
    )
    logger.info("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Load
    # -------------------------------------------------------------------------

    source_fingerprint = DatasetFingerprint(
        sha256=sha256_file(raw_path),
        row_count=0,
        column_count=0,
        columns=(),
    )

    df = _load_raw_dataset(raw_path)

    source_fingerprint = DatasetFingerprint(
        sha256=source_fingerprint.sha256,
        row_count=len(df),
        column_count=len(df.columns),
        columns=tuple(map(str, df.columns)),
    )

    logger.info(
        "Loaded dataset: rows=%d columns=%d",
        len(df),
        len(df.columns),
    )

    # -------------------------------------------------------------------------
    # 2. Minimum dataset checks
    # -------------------------------------------------------------------------

    if len(df) < config.min_samples:
        raise ValueError(
            f"Dataset contains only {len(df)} rows. "
            f"Minimum required: {config.min_samples}."
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df.columns.duplicated().any():
        duplicated_columns = (
            df.columns[df.columns.duplicated()]
            .tolist()
        )

        raise ValueError(
            f"Duplicate column names detected: {duplicated_columns}"
        )

    _validate_columns(
        df,
        config.target_column,
        config.url_column,
    )

    # -------------------------------------------------------------------------
    # 3. Target normalization
    # -------------------------------------------------------------------------

    df[config.target_column] = _coerce_binary_target(
        df[config.target_column],
        column_name=config.target_column,
    )

    # -------------------------------------------------------------------------
    # 4. URL validation
    # -------------------------------------------------------------------------

    df[config.url_column] = _validate_url_column(
        df[config.url_column],
        max_url_length=config.max_url_length,
    )

    # -------------------------------------------------------------------------
    # 5. Duplicate removal
    # -------------------------------------------------------------------------

    df, duplicate_stats = _remove_duplicate_rows(
        df,
        target_column=config.target_column,
        url_column=config.url_column,
        remove_exact=config.drop_exact_duplicates,
        remove_normalized=config.drop_normalized_duplicates,
    )

    logger.info(
        "Duplicate audit: exact_removed=%d normalized_removed=%d",
        duplicate_stats["exact_duplicates_removed"],
        duplicate_stats["normalized_url_duplicates_removed"],
    )

    if len(df) < config.min_samples:
        raise ValueError(
            f"Dataset fell below minimum size after deduplication: "
            f"{len(df)} < {config.min_samples}."
        )

    # -------------------------------------------------------------------------
    # 6. Label consistency audit
    # -------------------------------------------------------------------------

    conflict_report = _audit_label_conflicts(
        df,
        target_column=config.target_column,
        url_column=config.url_column,
    )

    if conflict_report["conflicting_normalized_urls"] > 0:
        logger.error(
            "Detected %d normalized URLs with conflicting labels.",
            conflict_report["conflicting_normalized_urls"],
        )

        raise ValueError(
            "Label conflict detected for normalized URLs. "
            "Resolve conflicting annotations before training. "
            f"Report: {conflict_report}"
        )

    # -------------------------------------------------------------------------
    # 7. Class distribution
    # -------------------------------------------------------------------------

    y = df[config.target_column]

    _validate_split_class_counts(
        y,
        min_samples_per_class=config.min_samples_per_class,
    )

    overall_positive_ratio = float(y.mean())

    if not (
        config.min_positive_ratio
        <= overall_positive_ratio
        <= config.max_positive_ratio
    ):
        raise ValueError(
            f"Dataset positive ratio {overall_positive_ratio:.4f} "
            f"is outside allowed range "
            f"[{config.min_positive_ratio}, {config.max_positive_ratio}]."
        )

    # -------------------------------------------------------------------------
    # 8. Build model features
    # -------------------------------------------------------------------------

    X, y, generated_groups = _build_feature_matrix(
        df,
        target_column=config.target_column,
        url_column=config.url_column,
    )

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "Feature extraction produced zero model features."
        )

    # -------------------------------------------------------------------------
    # 9. Determine grouping strategy
    # -------------------------------------------------------------------------

    if config.group_column:
        if config.group_column not in df.columns:
            raise KeyError(
                f"Configured group_column '{config.group_column}' "
                "does not exist in the dataset."
            )

        groups = (
            df[config.group_column]
            .astype("string")
            .fillna("<MISSING_GROUP>")
            .astype(str)
        )

        grouping_source = config.group_column

    else:
        groups = generated_groups
        grouping_source = "normalized_hostname"

    # -------------------------------------------------------------------------
    # 10. Split
    # -------------------------------------------------------------------------

    if config.group_split:
        logger.info(
            "Using group-aware splitting. Group source='%s'.",
            grouping_source,
        )

        split_indices = _split_with_groups(
            X,
            y,
            groups,
            test_size=config.test_size,
            val_size=config.val_size,
            random_state=config.random_state,
        )

    else:
        logger.warning(
            "Group-aware splitting disabled. "
            "Random row-level splitting may permit domain leakage."
        )

        split_indices = _split_with_stratification(
            X,
            y,
            groups,
            test_size=config.test_size,
            val_size=config.val_size,
            random_state=config.random_state,
        )

    # -------------------------------------------------------------------------
    # 11. Leakage gate
    # -------------------------------------------------------------------------

    _assert_split_integrity(
        split_indices,
        groups,
    )

    # -------------------------------------------------------------------------
    # 12. Construct partitions
    # -------------------------------------------------------------------------

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups.iloc[train_idx]
    groups_val = groups.iloc[val_idx]
    groups_test = groups.iloc[test_idx]

    summaries = [
        _summarize_split(
            "train",
            y_train,
            groups_train,
        ),
        _summarize_split(
            "val",
            y_val,
            groups_val,
        ),
        _summarize_split(
            "test",
            y_test,
            groups_test,
        ),
    ]

    _validate_split_distribution(
        summaries,
        min_positive_ratio=config.min_positive_ratio,
        max_positive_ratio=config.max_positive_ratio,
    )

    # -------------------------------------------------------------------------
    # 13. Build final model artifacts
    # -------------------------------------------------------------------------

    train_df = X_train.copy()
    train_df[config.target_column] = y_train.to_numpy()

    val_df = X_val.copy()
    val_df[config.target_column] = y_val.to_numpy()

    test_df = X_test.copy()
    test_df[config.target_column] = y_test.to_numpy()

    prefix = (
        f"{config.artifact_prefix}_"
        if config.artifact_prefix
        else ""
    )

    artifacts: Dict[str, Dict[str, str]] = {}

    # -------------------------------------------------------------------------
    # 14. Persist CSV
    # -------------------------------------------------------------------------

    if config.write_csv:
        csv_paths = {
            "train": out_path / f"{prefix}train.csv",
            "val": out_path / f"{prefix}val.csv",
            "test": out_path / f"{prefix}test.csv",
        }

        for name, frame in (
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ):
            path = csv_paths[name]

            _atomic_write_dataframe(
                frame,
                path,
                index=False,
            )

            artifacts.setdefault(name, {})["csv"] = str(path)

    # -------------------------------------------------------------------------
    # 15. Persist Parquet
    # -------------------------------------------------------------------------

    if config.write_parquet:
        parquet_paths = {
            "train": out_path / f"{prefix}train.parquet",
            "val": out_path / f"{prefix}val.parquet",
            "test": out_path / f"{prefix}test.parquet",
        }

        for name, frame in (
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ):
            path = parquet_paths[name]

            _atomic_write_dataframe(
                frame,
                path,
                index=False,
            )

            artifacts.setdefault(name, {})["parquet"] = str(path)

    # -------------------------------------------------------------------------
    # 16. Feature manifest
    # -------------------------------------------------------------------------

    feature_manifest = _feature_manifest(X)

    feature_manifest_path = (
        out_path / f"{prefix}feature_manifest.json"
    )

    _write_json(
        feature_manifest,
        feature_manifest_path,
    )

    # -------------------------------------------------------------------------
    # 17. Artifact fingerprints
    # -------------------------------------------------------------------------

    artifact_fingerprints: Dict[str, str] = {}

    for split_name, split_artifacts in artifacts.items():
        for fmt, artifact_path in split_artifacts.items():
            artifact_fingerprints[
                f"{split_name}.{fmt}"
            ] = sha256_file(artifact_path)

    feature_fingerprint = dataframe_fingerprint(X)

    # -------------------------------------------------------------------------
    # 18. Preparation manifest
    # -------------------------------------------------------------------------

    split_dicts = [
        asdict(summary)
        for summary in summaries
    ]

    preparation_manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": {
            "name": "01_prepare_data",
            "version": STAGE_VERSION,
        },
        "source": {
            "path": str(raw_path),
            "sha256": source_fingerprint.sha256,
            "row_count_before_processing": source_fingerprint.row_count,
            "column_count": source_fingerprint.column_count,
            "columns": list(source_fingerprint.columns),
        },
        "processing": {
            "target_column": config.target_column,
            "url_column": config.url_column,
            "test_size": config.test_size,
            "validation_size": config.val_size,
            "random_state": config.random_state,
            "group_split_enabled": config.group_split,
            "group_source": grouping_source,
            "exact_duplicate_removal": config.drop_exact_duplicates,
            "normalized_duplicate_removal": config.drop_normalized_duplicates,
            "max_url_length": config.max_url_length,
        },
        "quality": {
            "rows_after_processing": len(df),
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "positive_ratio": overall_positive_ratio,
            "duplicate_stats": duplicate_stats,
            "label_conflict_report": conflict_report,
            "group_count": int(groups.nunique()),
            "feature_count": X.shape[1],
            "feature_fingerprint": feature_fingerprint,
        },
        "splits": split_dicts,
        "artifacts": artifacts,
        "artifact_sha256": artifact_fingerprints,
        "feature_manifest": str(feature_manifest_path),
    }

    manifest_path = (
        out_path / f"{prefix}preparation_manifest.json"
    )

    _write_json(
        preparation_manifest,
        manifest_path,
    )

    # -------------------------------------------------------------------------
    # 19. Final summary
    # -------------------------------------------------------------------------

    summary: Dict[str, Any] = {
        "status": "COMPLETED",
        "stage": "01_prepare_data",
        "stage_version": STAGE_VERSION,

        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),

        "feature_count": X.shape[1],

        "train_phishing_ratio": float(y_train.mean()),
        "val_phishing_ratio": float(y_val.mean()),
        "test_phishing_ratio": float(y_test.mean()),

        "group_split_enabled": config.group_split,
        "group_source": grouping_source,

        "dataset_fingerprint": source_fingerprint.sha256,
        "feature_fingerprint": feature_fingerprint,

        "duplicates_removed": duplicate_stats,

        "artifacts": artifacts,

        "feature_manifest": str(feature_manifest_path),
        "preparation_manifest": str(manifest_path),

        "quality_gates": {
            "label_conflicts": False,
            "sample_leakage": False,
            "group_leakage": False,
            "class_distribution_valid": True,
            "feature_matrix_valid": True,
        },
    }

    logger.info("=" * 78)
    logger.info(
        "STAGE 01 COMPLETED SUCCESSFULLY"
    )
    logger.info(
        "Train: %d samples | phishing=%0.2f%% | groups=%d",
        len(train_df),
        y_train.mean() * 100,
        groups_train.nunique(),
    )
    logger.info(
        "Val:   %d samples | phishing=%0.2f%% | groups=%d",
        len(val_df),
        y_val.mean() * 100,
        groups_val.nunique(),
    )
    logger.info(
        "Test:  %d samples | phishing=%0.2f%% | groups=%d",
        len(test_df),
        y_test.mean() * 100,
        groups_test.nunique(),
    )
    logger.info(
        "Features: %d | Group split: %s",
        X.shape[1],
        config.group_split,
    )
    logger.info(
        "Dataset SHA-256: %s",
        source_fingerprint.sha256,
    )
    logger.info("=" * 78)

    return summary


# =============================================================================
# STAGE ENTRYPOINT
# =============================================================================


def main(
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Stage 01 programmatic entrypoint.

    Supports:
        run_stage("01_prepare_data", config_path="params.yaml")

    and direct Python invocation.
    """

    config = load_config(config_path) if config_path else {}

    prep_cfg = config.get(
        "data_preparation",
        {},
    )

    main_cfg = config.get(
        "main",
        {},
    )

    raw_path = kwargs.get(
        "raw_data_path",
        prep_cfg.get(
            "raw_data_path",
            "data/raw/phishing_dataset.csv",
        ),
    )

    output_dir = kwargs.get(
        "output_dir",
        prep_cfg.get(
            "output_dir",
            "data/processed",
        ),
    )

    target_col = kwargs.get(
        "target_column",
        prep_cfg.get(
            "target_column",
            "label",
        ),
    )

    url_col = kwargs.get(
        "url_column",
        prep_cfg.get(
            "url_column",
            "url",
        ),
    )

    test_sz = kwargs.get(
        "test_size",
        prep_cfg.get(
            "test_size",
            DEFAULT_TEST_SIZE,
        ),
    )

    val_sz = kwargs.get(
        "val_size",
        prep_cfg.get(
            "val_size",
            DEFAULT_VAL_SIZE,
        ),
    )

    seed = kwargs.get(
        "random_state",
        main_cfg.get(
            "random_seed",
            DEFAULT_RANDOM_STATE,
        ),
    )

    return prepare_dataset(
        raw_data_path=raw_path,
        output_dir=output_dir,
        target_column=target_col,
        url_column=url_col,
        test_size=test_sz,
        val_size=val_sz,
        random_state=seed,
        group_split=kwargs.get(
            "group_split",
            prep_cfg.get(
                "group_split",
                True,
            ),
        ),
        group_column=kwargs.get(
            "group_column",
            prep_cfg.get(
                "group_column",
            ),
        ),
        min_samples=kwargs.get(
            "min_samples",
            prep_cfg.get(
                "min_samples",
                100,
            ),
        ),
        min_samples_per_class=kwargs.get(
            "min_samples_per_class",
            prep_cfg.get(
                "min_samples_per_class",
                2,
            ),
        ),
        min_positive_ratio=kwargs.get(
            "min_positive_ratio",
            prep_cfg.get(
                "min_positive_ratio",
                0.001,
            ),
        ),
        max_positive_ratio=kwargs.get(
            "max_positive_ratio",
            prep_cfg.get(
                "max_positive_ratio",
                0.999,
            ),
        ),
        max_url_length=kwargs.get(
            "max_url_length",
            prep_cfg.get(
                "max_url_length",
                DEFAULT_MAX_URL_LENGTH,
            ),
        ),
        fail_on_invalid_rows=kwargs.get(
            "fail_on_invalid_rows",
            prep_cfg.get(
                "fail_on_invalid_rows",
                True,
            ),
        ),
        drop_exact_duplicates=kwargs.get(
            "drop_exact_duplicates",
            prep_cfg.get(
                "drop_exact_duplicates",
                True,
            ),
        ),
        drop_normalized_duplicates=kwargs.get(
            "drop_normalized_duplicates",
            prep_cfg.get(
                "drop_normalized_duplicates",
                True,
            ),
        ),
        write_parquet=kwargs.get(
            "write_parquet",
            prep_cfg.get(
                "write_parquet",
                True,
            ),
        ),
        write_csv=kwargs.get(
            "write_csv",
            prep_cfg.get(
                "write_csv",
                True,
            ),
        ),
        artifact_prefix=kwargs.get(
            "artifact_prefix",
            prep_cfg.get(
                "artifact_prefix",
                "",
            ),
        ),
    )


# =============================================================================
# CLI
# =============================================================================


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line parser for Stage 01."""

    parser = argparse.ArgumentParser(
        description=(
            "PhishGuard AI Stage 01 - "
            "Data Preparation & Feature Extraction"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to pipeline YAML configuration.",
    )

    parser.add_argument(
        "--raw-data",
        type=str,
        default=None,
        help="Optional raw dataset override.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )

    parser.add_argument(
        "--no-group-split",
        action="store_true",
        help=(
            "Disable domain/group-aware splitting. "
            "Not recommended for production evaluation."
        ),
    )

    return parser


def cli() -> int:
    """CLI entrypoint."""

    parser = _build_argument_parser()
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}

    if args.raw_data:
        overrides["raw_data_path"] = args.raw_data

    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    if args.no_group_split:
        overrides["group_split"] = False

    try:
        result = main(
            config_path=args.config,
            **overrides,
        )

        print(
            json.dumps(
                result,
                indent=2,
                default=_json_default,
            )
        )

        return 0

    except Exception as exc:
        logger.error(
            "Stage 01 failed: %s",
            exc,
            exc_info=True,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(cli())


__all__ = [
    "PreparationConfig",
    "DatasetFingerprint",
    "SplitSummary",
    "STAGE_VERSION",
    "calculate_entropy",
    "normalize_url",
    "extract_group_key",
    "extract_url_features",
    "sha256_file",
    "dataframe_fingerprint",
    "prepare_dataset",
    "main",
    "cli",
]