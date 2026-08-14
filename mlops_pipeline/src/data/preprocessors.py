"""
PhishGuard AI - Enterprise Multi-Modal Feature Preprocessing Engine
===================================================================

Version: 14.0.0
File Path: src/data/preprocessors.py

Purpose
-------
Production-grade preprocessing layer for the PhishGuard AI MLOps platform.

The module provides:

    * Strict fit/transform lifecycle enforcement
    * Training-schema locking
    * Data leakage prevention
    * Finite-value validation
    * Resource and dimensionality protection
    * Sparse-safe TF-IDF processing
    * Security-aware email normalization
    * URL lexical preprocessing
    * Attachment metadata preprocessing
    * Multi-modal numerical scaling
    * Feature-name contracts
    * Training metadata / provenance
    * Artifact fingerprints
    * Atomic artifact persistence
    * Trusted-artifact loading controls
    * Backward-compatible public classes
    * Operational diagnostics
    * Defensive validation
    * Deterministic configuration fingerprints

Security model
--------------
Preprocessors operate on potentially attacker-controlled data. Therefore:

    1. Input size is bounded.
    2. Unexpected columns are detected.
    3. Non-finite values are rejected or safely imputed.
    4. Dense conversion of large sparse matrices is prevented.
    5. Training and inference schemas are separated explicitly.
    6. Serialized artifacts are treated as trusted-only objects.
    7. Artifact fingerprints can be verified before loading.
    8. Pickle/joblib loading is never considered a security boundary.

Important
---------
joblib/pickle artifacts MUST only be loaded from a trusted artifact registry
or another authenticated/integrity-verified source.

For highly security-sensitive deployment, consider a safer model/artifact
format such as skops.io or ONNX where applicable.
"""

from __future__ import annotations

# ============================================================================
# STANDARD LIBRARY
# ============================================================================

import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import re
import tempfile
import threading
import warnings

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ============================================================================
# THIRD-PARTY
# ============================================================================

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler, StandardScaler

# ============================================================================
# PROJECT
# ============================================================================

from src.utils.logger import get_logger

logger = get_logger("src.data.preprocessors")


# ============================================================================
# MODULE METADATA
# ============================================================================

MODULE_NAME = "phishguard.data.preprocessors"
MODULE_VERSION = "14.0.0"

DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_MAX_TEXT_LENGTH = 1_000_000
DEFAULT_MAX_DENSE_ELEMENTS = 20_000_000
DEFAULT_MAX_FEATURES = 100_000

ARTIFACT_FORMAT_VERSION = "1.0"

SUPPORTED_TEXT_TYPES = (str, bytes)

_THREAD_LOCK = threading.RLock()


# ============================================================================
# EXCEPTIONS
# ============================================================================


class PreprocessorError(RuntimeError):
    """Base exception for preprocessing failures."""


class PreprocessorNotFittedError(PreprocessorError):
    """Raised when transform is called before fit."""


class PreprocessorValidationError(PreprocessorError):
    """Raised when input data violates a preprocessing contract."""


class PreprocessorArtifactError(PreprocessorError):
    """Raised when a persisted preprocessor is invalid or unsafe."""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _require_positive_int(
    value: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> int:
    """Validate integer configuration values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    minimum = 0 if allow_zero else 1

    if value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}; received {value!r}."
        )

    return value


def _require_probability(value: float, name: str) -> float:
    """Validate a probability/rate in [0, 1]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1; received {value}."
        )

    return value


def _canonical_json(value: Any) -> str:
    """
    Produce a deterministic JSON representation.

    This intentionally only handles configuration/provenance values and
    converts unsupported objects to repr strings.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _sha256_text(value: str) -> str:
    """Return SHA-256 hexadecimal digest for UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configuration_fingerprint(config: Mapping[str, Any]) -> str:
    """Create deterministic configuration fingerprint."""
    return _sha256_text(_canonical_json(dict(config)))


def _safe_package_version(package: str) -> Optional[str]:
    """Return installed package version without raising."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _runtime_metadata() -> Dict[str, Any]:
    """Collect non-sensitive runtime provenance."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": _safe_package_version("scikit-learn"),
        "joblib_version": _safe_package_version("joblib"),
        "scipy_version": _safe_package_version("scipy"),
    }


def _normalize_columns(columns: Iterable[Any]) -> List[str]:
    """
    Normalize DataFrame column names to strings.

    Column names are part of the model contract and should therefore have
    deterministic representations.
    """
    result: List[str] = []

    for column in columns:
        if isinstance(column, str):
            name = column
        else:
            name = str(column)

        name = name.strip()

        if not name:
            raise PreprocessorValidationError(
                "DataFrame contains an empty column name."
            )

        result.append(name)

    if len(result) != len(set(result)):
        duplicates = sorted(
            {
                name
                for name in result
                if result.count(name) > 1
            }
        )

        raise PreprocessorValidationError(
            f"Duplicate column names detected: {duplicates!r}"
        )

    return result


def _validate_dataframe(
    X: pd.DataFrame,
    *,
    name: str = "X",
    allow_empty_rows: bool = False,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> pd.DataFrame:
    """Validate and shallow-copy a DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame; "
            f"received {type(X).__name__}."
        )

    if not allow_empty_rows and X.empty:
        raise PreprocessorValidationError(
            f"{name} must contain at least one row."
        )

    if len(X) > max_rows:
        raise PreprocessorValidationError(
            f"{name} contains {len(X):,} rows, exceeding the configured "
            f"maximum of {max_rows:,}."
        )

    normalized_columns = _normalize_columns(X.columns)

    if list(X.columns) != normalized_columns:
        X = X.copy()
        X.columns = normalized_columns

    return X


def _validate_text_sequence(
    X: Union[pd.Series, Sequence[Any]],
    *,
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
) -> List[str]:
    """
    Normalize text inputs without silently converting arbitrary objects
    into huge or misleading strings.
    """
    if isinstance(X, pd.Series):
        values = X.tolist()
    elif isinstance(X, (list, tuple, np.ndarray)):
        values = list(X)
    else:
        raise TypeError(
            "Text input must be a pandas Series, list, tuple, or NumPy array."
        )

    normalized: List[str] = []

    for index, value in enumerate(values):
        if value is None:
            normalized.append("")
            continue

        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception as exc:
                raise PreprocessorValidationError(
                    f"Unable to decode text item at index {index}."
                ) from exc

        elif not isinstance(value, str):
            if pd.isna(value):
                normalized.append("")
                continue

            raise PreprocessorValidationError(
                f"Text item at index {index} has unsupported type "
                f"{type(value).__name__}."
            )

        if len(value) > max_text_length:
            raise PreprocessorValidationError(
                f"Text item at index {index} exceeds the maximum allowed "
                f"length of {max_text_length:,} characters."
            )

        normalized.append(value)

    return normalized


def _validate_numeric_array(
    X: Any,
    *,
    name: str = "X",
    max_rows: int = DEFAULT_MAX_ROWS,
) -> np.ndarray:
    """Validate numerical matrix and reject unsafe values."""
    if sparse.issparse(X):
        matrix = X

        if matrix.ndim != 2:
            raise PreprocessorValidationError(
                f"{name} must be a 2-dimensional sparse matrix."
            )

        if matrix.shape[0] > max_rows:
            raise PreprocessorValidationError(
                f"{name} contains too many rows."
            )

        if matrix.dtype.kind not in "fiu":
            raise PreprocessorValidationError(
                f"{name} must contain numeric sparse values."
            )

        return matrix

    array = np.asarray(X)

    if array.ndim != 2:
        raise PreprocessorValidationError(
            f"{name} must be a 2-dimensional numerical matrix; "
            f"received shape {array.shape}."
        )

    if array.shape[0] > max_rows:
        raise PreprocessorValidationError(
            f"{name} contains {array.shape[0]:,} rows, exceeding "
            f"the configured maximum."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise PreprocessorValidationError(
            f"{name} must contain numerical values."
        )

    if not np.isfinite(array).all():
        raise PreprocessorValidationError(
            f"{name} contains NaN or infinite values."
        )

    return array


def _ensure_finite_dense(
    array: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Reject non-finite dense output."""
    if not np.isfinite(array).all():
        raise PreprocessorValidationError(
            f"{name} produced NaN or infinite values."
        )

    return array


def _ensure_sparse_finite(
    matrix: sparse.spmatrix,
    *,
    name: str,
) -> sparse.spmatrix:
    """Reject non-finite sparse values."""
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise PreprocessorValidationError(
            f"{name} produced NaN or infinite sparse values."
        )

    return matrix


def _safe_dense(
    matrix: Any,
    *,
    max_elements: int,
    name: str,
) -> np.ndarray:
    """
    Convert sparse matrix to dense only when explicitly within resource
    limits.
    """
    if sparse.issparse(matrix):
        rows, columns = matrix.shape
        elements = rows * columns

        if elements > max_elements:
            raise PreprocessorValidationError(
                f"Refusing to densify {name}: requested {elements:,} elements "
                f"exceeds safety limit {max_elements:,}."
            )

        matrix = matrix.toarray()

    array = np.asarray(matrix)

    if array.size > max_elements:
        raise PreprocessorValidationError(
            f"{name} contains {array.size:,} elements, exceeding "
            f"the dense safety limit of {max_elements:,}."
        )

    return _ensure_finite_dense(array, name=name)


def _atomic_joblib_dump(
    obj: Any,
    path: Path,
    *,
    compress: Union[int, Tuple[str, int]] = 3,
) -> None:
    """
    Persist a joblib artifact using a same-directory temporary file followed
    by atomic replacement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    os.close(fd)

    temporary_path = Path(temporary_name)

    try:
        joblib.dump(
            obj,
            temporary_path,
            compress=compress,
            protocol=5,
        )

        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())

        os.replace(temporary_path, path)

    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for an artifact without loading it into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class PreprocessorMetadata:
    """Operational metadata associated with a fitted preprocessor."""

    module: str = MODULE_NAME
    module_version: str = MODULE_VERSION
    artifact_format_version: str = ARTIFACT_FORMAT_VERSION

    class_name: str = ""
    fitted: bool = False

    input_rows_seen: int = 0
    input_columns: List[str] = field(default_factory=list)
    output_feature_count: int = 0

    configuration_fingerprint: str = ""

    runtime: Dict[str, Any] = field(default_factory=dict)

    created_at_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible representation."""
        return asdict(self)


# ============================================================================
# BASE PREPROCESSOR
# ============================================================================


class BasePreprocessor(ABC):
    """
    Enterprise base class for all PhishGuard preprocessing components.

    Guarantees:

        * explicit fit state
        * schema capture
        * configuration fingerprint
        * resource validation
        * atomic artifact persistence
        * artifact metadata
        * lifecycle locking
    """

    artifact_type = "preprocessor"

    def __init__(
        self,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_dense_elements: int = DEFAULT_MAX_DENSE_ELEMENTS,
    ) -> None:
        self.max_rows = _require_positive_int(
            max_rows,
            "max_rows",
        )

        self.max_dense_elements = _require_positive_int(
            max_dense_elements,
            "max_dense_elements",
        )

        self.is_fitted: bool = False

        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

        self.n_features_in_: Optional[int] = None
        self.n_features_out_: Optional[int] = None

        self.n_samples_seen_: int = 0

        self.metadata_: Optional[PreprocessorMetadata] = None

        self.configuration_fingerprint_: Optional[str] = None

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # ABSTRACT CONTRACT
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Optional[Any] = None,
    ) -> "BasePreprocessor":
        """Fit preprocessing state using training data."""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray, sparse.spmatrix]:
        """Transform previously unseen data using fitted state."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Optional[Any] = None,
    ) -> Union[pd.DataFrame, np.ndarray, sparse.spmatrix]:
        """Fit the preprocessor and transform the same training data."""
        return self.fit(X, y).transform(X)

    def _require_fitted(self) -> None:
        """Ensure transform is only used after fitting."""
        if not self.is_fitted:
            raise PreprocessorNotFittedError(
                f"{self.__class__.__name__} must be fitted before transform()."
            )

    def _mark_fitted(
        self,
        *,
        input_rows: int,
        input_features: int,
        output_features: int,
        input_columns: Optional[Sequence[str]] = None,
        output_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Finalize fitted state and capture immutable operational metadata."""
        self.is_fitted = True

        self.n_samples_seen_ = int(input_rows)
        self.n_features_in_ = int(input_features)
        self.n_features_out_ = int(output_features)

        self.feature_names_in_ = (
            list(input_columns)
            if input_columns is not None
            else []
        )

        self.feature_names_out_ = (
            list(output_columns)
            if output_columns is not None
            else []
        )

        self.configuration_fingerprint_ = self._configuration_fingerprint()

        self.metadata_ = PreprocessorMetadata(
            class_name=self.__class__.__name__,
            fitted=True,
            input_rows_seen=self.n_samples_seen_,
            input_columns=list(self.feature_names_in_),
            output_feature_count=self.n_features_out_ or 0,
            configuration_fingerprint=self.configuration_fingerprint_,
            runtime=_runtime_metadata(),
            created_at_utc=pd.Timestamp.utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    def _configuration(self) -> Dict[str, Any]:
        """
        Return serializable configuration.

        Subclasses should override when they have custom parameters.
        """
        return {
            "class_name": self.__class__.__name__,
            "max_rows": self.max_rows,
            "max_dense_elements": self.max_dense_elements,
        }

    def _configuration_fingerprint(self) -> str:
        """Return deterministic hash of preprocessing configuration."""
        return _configuration_fingerprint(self._configuration())

    # ------------------------------------------------------------------
    # SCHEMA CONTRACT
    # ------------------------------------------------------------------

    def _validate_dataframe_schema(
        self,
        X: pd.DataFrame,
        *,
        allow_extra_columns: bool = False,
    ) -> pd.DataFrame:
        """Validate inference DataFrame against fitted schema."""
        X = _validate_dataframe(
            X,
            max_rows=self.max_rows,
        )

        if not self.feature_names_in_:
            return X

        expected = self.feature_names_in_
        actual = list(X.columns)

        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]

        if missing:
            raise PreprocessorValidationError(
                f"{self.__class__.__name__}: missing fitted columns: {missing!r}"
            )

        if extra and not allow_extra_columns:
            raise PreprocessorValidationError(
                f"{self.__class__.__name__}: unexpected columns during "
                f"transform: {extra!r}"
            )

        return X[expected]

    # ------------------------------------------------------------------
    # FEATURE NAMES
    # ------------------------------------------------------------------

    def get_feature_names_out(self) -> np.ndarray:
        """Return output feature names."""
        self._require_fitted()

        return np.asarray(
            self.feature_names_out_,
            dtype=object,
        )

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Return preprocessing metadata."""
        self._require_fitted()

        if self.metadata_ is None:
            raise PreprocessorArtifactError(
                "Fitted preprocessor has no metadata."
            )

        return self.metadata_.to_dict()

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def save(
        self,
        filepath: Union[str, Path],
        *,
        compress: Union[int, Tuple[str, int]] = 3,
        overwrite: bool = True,
    ) -> Path:
        """
        Persist a trusted preprocessor artifact atomically.

        WARNING
        -------
        joblib uses pickle semantics. The resulting artifact must never be
        loaded from an untrusted source.
        """
        self._require_fitted()

        path = Path(filepath).expanduser().resolve()

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Preprocessor artifact already exists: '{path}'."
            )

        artifact = {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "module": MODULE_NAME,
            "module_version": MODULE_VERSION,
            "class_name": self.__class__.__name__,
            "configuration": self._configuration(),
            "configuration_fingerprint": self.configuration_fingerprint_,
            "metadata": self.get_metadata(),
            "preprocessor": self,
        }

        with _THREAD_LOCK:
            _atomic_joblib_dump(
                artifact,
                path,
                compress=compress,
            )

        digest = _sha256_file(path)

        logger.info(
            "Preprocessor artifact saved.",
            extra={
                "extra_fields": {
                    "artifact": str(path),
                    "artifact_sha256": digest,
                    "class_name": self.__class__.__name__,
                }
            },
        )

        return path

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        *,
        expected_sha256: Optional[str] = None,
        max_artifact_size_bytes: int = 512 * 1024 * 1024,
    ) -> "BasePreprocessor":
        """
        Load a trusted joblib preprocessor artifact.

        The SHA-256 check occurs before deserialization when provided.

        SECURITY
        --------
        This function does NOT make joblib safe for untrusted files.
        """
        path = Path(filepath).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: '{path}'."
            )

        size = path.stat().st_size

        if size > max_artifact_size_bytes:
            raise PreprocessorArtifactError(
                f"Artifact size {size:,} bytes exceeds configured "
                f"maximum of {max_artifact_size_bytes:,} bytes."
            )

        actual_sha256 = _sha256_file(path)

        if expected_sha256 is not None:
            expected = expected_sha256.strip().lower()

            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(
                    "expected_sha256 must be a 64-character hexadecimal digest."
                )

            if not __import__("hmac").compare_digest(
                actual_sha256.lower(),
                expected,
            ):
                raise PreprocessorArtifactError(
                    "Preprocessor artifact integrity verification failed."
                )

        # Explicit security warning in logs.
        logger.warning(
            "Loading a joblib preprocessor artifact. "
            "Artifact MUST originate from a trusted source."
        )

        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise PreprocessorArtifactError(
                f"Unable to deserialize preprocessor artifact '{path}'."
            ) from exc

        if not isinstance(payload, dict):
            raise PreprocessorArtifactError(
                "Invalid preprocessor artifact structure."
            )

        if payload.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
            raise PreprocessorArtifactError(
                "Unsupported preprocessor artifact format version: "
                f"{payload.get('artifact_format_version')!r}."
            )

        obj = payload.get("preprocessor")

        if not isinstance(obj, BasePreprocessor):
            raise PreprocessorArtifactError(
                "Artifact does not contain a valid BasePreprocessor instance."
            )

        if not obj.is_fitted:
            raise PreprocessorArtifactError(
                "Persisted preprocessor is not marked as fitted."
            )

        stored_fingerprint = payload.get(
            "configuration_fingerprint"
        )

        if (
            stored_fingerprint
            and obj.configuration_fingerprint_
            and stored_fingerprint != obj.configuration_fingerprint_
        ):
            raise PreprocessorArtifactError(
                "Preprocessor configuration fingerprint mismatch."
            )

        logger.info(
            "Trusted preprocessor artifact loaded.",
            extra={
                "extra_fields": {
                    "artifact": str(path),
                    "artifact_sha256": actual_sha256,
                    "class_name": obj.__class__.__name__,
                }
            },
        )

        return obj


# ============================================================================
# EMAIL TEXT PREPROCESSOR
# ============================================================================


class EmailTextPreprocessor(BasePreprocessor):
    """
    Security-aware email text preprocessing and TF-IDF vectorization.

    Design principles:

        * Preserve security-relevant lexical signals.
        * Normalize HTML safely.
        * Replace URLs instead of deleting URL presence.
        * Prevent oversized documents.
        * Prevent accidental sparse-to-dense conversion.
        * Lock vectorizer vocabulary after training.
        * Support both word and character analysis.
    """

    def __init__(
        self,
        max_features: int = 5_000,
        ngram_range: Tuple[int, int] = (1, 2),
        *,
        analyzer: str = "word",
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        sublinear_tf: bool = True,
        stop_words: Optional[str] = None,
        lowercase: bool = True,
        strip_accents: Optional[str] = "unicode",
        dtype: Any = np.float32,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        max_dense_elements: int = DEFAULT_MAX_DENSE_ELEMENTS,
        return_dense: bool = False,
        url_placeholder: str = " urltoken ",
        html_placeholder: str = " htmltoken ",
    ) -> None:
        super().__init__(
            max_rows=DEFAULT_MAX_ROWS,
            max_dense_elements=max_dense_elements,
        )

        self.max_features = _require_positive_int(
            max_features,
            "max_features",
        )

        if max_features > DEFAULT_MAX_FEATURES:
            raise ValueError(
                f"max_features cannot exceed {DEFAULT_MAX_FEATURES:,}."
            )

        if (
            not isinstance(ngram_range, tuple)
            or len(ngram_range) != 2
            or ngram_range[0] < 1
            or ngram_range[1] < ngram_range[0]
        ):
            raise ValueError(
                "ngram_range must be a tuple such as (1, 2)."
            )

        if analyzer not in {"word", "char", "char_wb"}:
            raise ValueError(
                "analyzer must be 'word', 'char', or 'char_wb'."
            )

        if isinstance(min_df, float) and not 0.0 < min_df <= 1.0:
            raise ValueError("float min_df must be in (0, 1].")

        if isinstance(max_df, float) and not 0.0 < max_df <= 1.0:
            raise ValueError("float max_df must be in (0, 1].")

        if (
            isinstance(min_df, int)
            and min_df < 1
        ):
            raise ValueError("integer min_df must be >= 1.")

        if (
            isinstance(max_df, int)
            and max_df < 1
        ):
            raise ValueError("integer max_df must be >= 1.")

        if min_df > max_df:
            raise ValueError(
                "min_df cannot be greater than max_df."
            )

        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = bool(sublinear_tf)
        self.stop_words = stop_words
        self.lowercase = bool(lowercase)
        self.strip_accents = strip_accents
        self.dtype = dtype

        self.max_text_length = _require_positive_int(
            max_text_length,
            "max_text_length",
        )

        self.return_dense = bool(return_dense)

        self.url_placeholder = url_placeholder
        self.html_placeholder = html_placeholder

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self.stop_words,
            sublinear_tf=self.sublinear_tf,
            lowercase=self.lowercase,
            strip_accents=self.strip_accents,
            dtype=self.dtype,
        )

        self.training_document_count_: Optional[int] = None
        self.vocabulary_size_: Optional[int] = None

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def _configuration(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "analyzer": self.analyzer,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "stop_words": self.stop_words,
            "lowercase": self.lowercase,
            "strip_accents": self.strip_accents,
            "dtype": str(self.dtype),
            "max_text_length": self.max_text_length,
            "return_dense": self.return_dense,
            "url_placeholder": self.url_placeholder,
            "html_placeholder": self.html_placeholder,
            "max_dense_elements": self.max_dense_elements,
        }

    # ------------------------------------------------------------------
    # TEXT NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(
        raw_text: Any,
        *,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        url_placeholder: str = " urltoken ",
        html_placeholder: str = " htmltoken ",
    ) -> str:
        """
        Normalize email text while retaining phishing-relevant signals.

        This intentionally does not aggressively remove punctuation,
        encoded strings, or suspicious lexical structures.
        """
        if raw_text is None:
            return ""

        if isinstance(raw_text, bytes):
            raw_text = raw_text.decode(
                "utf-8",
                errors="replace",
            )

        if not isinstance(raw_text, str):
            if pd.isna(raw_text):
                return ""

            raise TypeError(
                f"Email text must be str/bytes/None, got "
                f"{type(raw_text).__name__}."
            )

        if len(raw_text) > max_text_length:
            raise PreprocessorValidationError(
                "Email text exceeds configured maximum length."
            )

        text = html.unescape(raw_text)

        # Replace scripts/styles rather than allowing their entire contents
        # to pollute the lexical model.
        text = re.sub(
            r"(?is)<(?:script|style).*?>.*?</(?:script|style)>",
            f" {html_placeholder} ",
            text,
        )

        # Preserve existence of HTML markup.
        if re.search(r"<[a-zA-Z][^>]*>", text):
            text = re.sub(
                r"<[^>]+>",
                " ",
                text,
            )
            text += f" {html_placeholder} "

        # Preserve URL presence as a model-visible token.
        text = re.sub(
            r"(?i)\b(?:https?://|www\.)[^\s<>'\"]+",
            f" {url_placeholder} ",
            text,
        )

        # Normalize common control characters.
        text = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
            " ",
            text,
        )

        # Normalize whitespace but don't destroy punctuation.
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text.lower()

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.Series, Sequence[Any]],
        y: Optional[Any] = None,
    ) -> "EmailTextPreprocessor":
        """Fit TF-IDF vocabulary strictly on training text."""
        with self._lock:
            texts = _validate_text_sequence(
                X,
                max_text_length=self.max_text_length,
            )

            if not texts:
                raise PreprocessorValidationError(
                    "Cannot fit EmailTextPreprocessor on empty input."
                )

            cleaned = [
                self.clean_text(
                    value,
                    max_text_length=self.max_text_length,
                    url_placeholder=self.url_placeholder,
                    html_placeholder=self.html_placeholder,
                )
                for value in texts
            ]

            # Prevent an all-empty corpus from producing obscure sklearn errors.
            if not any(cleaned):
                raise PreprocessorValidationError(
                    "Training corpus contains no usable textual content."
                )

            logger.info(
                "Fitting EmailTextPreprocessor.",
                extra={
                    "extra_fields": {
                        "documents": len(cleaned),
                        "max_features": self.max_features,
                        "analyzer": self.analyzer,
                    }
                },
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.vectorizer.fit(cleaned)

            vocabulary = getattr(
                self.vectorizer,
                "vocabulary_",
                {},
            )

            if not vocabulary:
                raise PreprocessorValidationError(
                    "TF-IDF vocabulary is empty after fitting."
                )

            self.training_document_count_ = len(cleaned)
            self.vocabulary_size_ = len(vocabulary)

            feature_names = [
                str(name)
                for name in self.vectorizer.get_feature_names_out()
            ]

            self._mark_fitted(
                input_rows=len(cleaned),
                input_features=1,
                output_features=len(feature_names),
                input_columns=["text"],
                output_columns=feature_names,
            )

            return self

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------

    def transform(
        self,
        X: Union[pd.Series, Sequence[Any]],
    ) -> Union[np.ndarray, sparse.csr_matrix]:
        """Transform email text into sparse or explicitly bounded dense TF-IDF."""
        self._require_fitted()

        texts = _validate_text_sequence(
            X,
            max_text_length=self.max_text_length,
        )

        cleaned = [
            self.clean_text(
                value,
                max_text_length=self.max_text_length,
                url_placeholder=self.url_placeholder,
                html_placeholder=self.html_placeholder,
            )
            for value in texts
        ]

        matrix = self.vectorizer.transform(cleaned)

        _ensure_sparse_finite(
            matrix,
            name="TF-IDF transformation",
        )

        matrix = matrix.tocsr()

        if not self.return_dense:
            return matrix

        return _safe_dense(
            matrix,
            max_elements=self.max_dense_elements,
            name="TF-IDF output",
        )


# ============================================================================
# URL FEATURE PREPROCESSOR
# ============================================================================


class URLFeaturePreprocessor(BasePreprocessor):
    """
    Production URL numerical feature preprocessor.

    Important behavior:

        * Training medians are frozen.
        * Numerical feature names are frozen.
        * Non-finite values are converted to missing values.
        * Log transformation is applied only to valid non-negative columns.
        * Negative values are not silently clamped.
        * Unknown inference columns are rejected.
        * Numerical output remains a DataFrame with stable names.
    """

    def __init__(
        self,
        skew_columns: Optional[List[str]] = None,
        *,
        scaler: str = "robust",
        max_rows: int = DEFAULT_MAX_ROWS,
        allow_negative_log_values: bool = False,
    ) -> None:
        super().__init__(
            max_rows=max_rows,
        )

        self.skew_columns = list(
            skew_columns
            if skew_columns is not None
            else [
                "url_length",
                "domain_age_days",
            ]
        )

        if scaler not in {"robust", "standard"}:
            raise ValueError(
                "scaler must be either 'robust' or 'standard'."
            )

        self.scaler_type = scaler
        self.allow_negative_log_values = bool(
            allow_negative_log_values
        )

        self.medians_: Dict[str, float] = {}
        self.numeric_columns_: List[str] = []

        self.scaler = (
            RobustScaler()
            if scaler == "robust"
            else StandardScaler()
        )

        self.invalid_value_counts_: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def _configuration(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "skew_columns": self.skew_columns,
            "scaler": self.scaler_type,
            "max_rows": self.max_rows,
            "allow_negative_log_values": self.allow_negative_log_values,
        }

    # ------------------------------------------------------------------
    # CLEANING
    # ------------------------------------------------------------------

    def _prepare_numeric(
        self,
        X: pd.DataFrame,
        *,
        fitting: bool,
    ) -> pd.DataFrame:
        """Prepare numerical URL features."""
        if fitting:
            numeric = X.select_dtypes(
                include=[np.number]
            ).copy()

            if numeric.empty:
                raise PreprocessorValidationError(
                    "URL dataset contains no numerical features."
                )

            self.numeric_columns_ = list(
                numeric.columns
            )

        else:
            numeric = X[
                self.numeric_columns_
            ].copy()

        # Replace inf/-inf with NaN instead of allowing scaler failures.
        for column in self.numeric_columns_:
            values = pd.to_numeric(
                numeric[column],
                errors="coerce",
            )

            invalid_count = int(
                (~np.isfinite(values.fillna(0))).sum()
            )

            if invalid_count:
                self.invalid_value_counts_[column] = (
                    self.invalid_value_counts_.get(column, 0)
                    + invalid_count
                )

            numeric[column] = values.replace(
                [np.inf, -np.inf],
                np.nan,
            )

        if fitting:
            for column in self.numeric_columns_:
                median = numeric[column].median()

                if pd.isna(median) or not np.isfinite(median):
                    raise PreprocessorValidationError(
                        f"Column '{column}' contains no finite values "
                        "from which a training median can be learned."
                    )

                self.medians_[column] = float(median)

        numeric = numeric.fillna(
            self.medians_
        )

        # Apply log1p only to columns explicitly declared as non-negative.
        for column in self.skew_columns:
            if column not in numeric.columns:
                continue

            values = numeric[column].to_numpy(
                dtype=np.float64,
                copy=True,
            )

            if not self.allow_negative_log_values:
                negative_mask = values < 0

                if negative_mask.any():
                    raise PreprocessorValidationError(
                        f"URL feature '{column}' contains "
                        f"{int(negative_mask.sum())} negative values. "
                        "Refusing silent clipping before log1p."
                    )

            values = np.log1p(values)

            if not np.isfinite(values).all():
                raise PreprocessorValidationError(
                    f"Log transformation produced non-finite values "
                    f"for column '{column}'."
                )

            numeric[column] = values

        return numeric

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Any] = None,
    ) -> "URLFeaturePreprocessor":
        with self._lock:
            X = _validate_dataframe(
                X,
                max_rows=self.max_rows,
            )

            prepared = self._prepare_numeric(
                X,
                fitting=True,
            )

            self.scaler.fit(
                prepared[self.numeric_columns_]
            )

            self._mark_fitted(
                input_rows=len(X),
                input_features=len(self.numeric_columns_),
                output_features=len(self.numeric_columns_),
                input_columns=list(self.numeric_columns_),
                output_columns=list(self.numeric_columns_),
            )

            logger.info(
                "URLFeaturePreprocessor fitted.",
                extra={
                    "extra_fields": {
                        "rows": len(X),
                        "numeric_features": len(self.numeric_columns_),
                        "scaler": self.scaler_type,
                    }
                },
            )

            return self

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        self._require_fitted()

        X = self._validate_dataframe_schema(
            X,
            allow_extra_columns=False,
        )

        prepared = self._prepare_numeric(
            X,
            fitting=False,
        )

        values = self.scaler.transform(
            prepared[self.numeric_columns_]
        )

        values = _ensure_finite_dense(
            np.asarray(values),
            name="URL scaled output",
        )

        return pd.DataFrame(
            values,
            columns=self.numeric_columns_,
            index=X.index,
        )


# ============================================================================
# ATTACHMENT FEATURE PREPROCESSOR
# ============================================================================


class AttachmentFeaturePreprocessor(BasePreprocessor):
    """
    Production attachment metadata preprocessor.

    Handles:

        * numerical metadata
        * optional categorical metadata
        * missing-value imputation
        * stable training schema
        * unknown-category handling
        * scaling
        * bounded dense output

    The class intentionally keeps categorical processing optional to preserve
    compatibility with existing pipelines that only expect numerical output.
    """

    def __init__(
        self,
        *,
        scaler: str = "standard",
        categorical_columns: Optional[List[str]] = None,
        max_categories_per_column: int = 500,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        super().__init__(
            max_rows=max_rows,
        )

        if scaler not in {"standard", "robust"}:
            raise ValueError(
                "scaler must be 'standard' or 'robust'."
            )

        self.scaler_type = scaler

        self.categorical_columns = list(
            categorical_columns or []
        )

        self.max_categories_per_column = _require_positive_int(
            max_categories_per_column,
            "max_categories_per_column",
        )

        self.numeric_columns_: List[str] = []
        self.numeric_fill_values_: Dict[str, float] = {}

        self.category_values_: Dict[str, List[str]] = {}
        self.category_maps_: Dict[str, Dict[str, int]] = {}

        self.scaler = (
            StandardScaler()
            if scaler == "standard"
            else RobustScaler()
        )

        self.output_columns_: List[str] = []

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def _configuration(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "scaler": self.scaler_type,
            "categorical_columns": self.categorical_columns,
            "max_categories_per_column": self.max_categories_per_column,
            "max_rows": self.max_rows,
        }

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Any] = None,
    ) -> "AttachmentFeaturePreprocessor":
        with self._lock:
            X = _validate_dataframe(
                X,
                max_rows=self.max_rows,
            )

            self.numeric_columns_ = list(
                X.select_dtypes(
                    include=[np.number]
                ).columns
            )

            if not self.numeric_columns_:
                raise PreprocessorValidationError(
                    "Attachment dataset contains no numerical features."
                )

            numeric = X[
                self.numeric_columns_
            ].copy()

            numeric = numeric.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            for column in self.numeric_columns_:
                median = numeric[column].median()

                if pd.isna(median) or not np.isfinite(median):
                    raise PreprocessorValidationError(
                        f"Cannot derive finite median for '{column}'."
                    )

                self.numeric_fill_values_[column] = float(
                    median
                )

            numeric = numeric.fillna(
                self.numeric_fill_values_
            )

            self.scaler.fit(numeric)

            self.output_columns_ = list(
                self.numeric_columns_
            )

            # Optional categorical vocabulary.
            for column in self.categorical_columns:
                if column not in X.columns:
                    raise PreprocessorValidationError(
                        f"Configured categorical column '{column}' "
                        "does not exist in attachment dataset."
                    )

                values = (
                    X[column]
                    .fillna("__MISSING__")
                    .astype(str)
                    .value_counts()
                    .head(self.max_categories_per_column)
                    .index
                    .tolist()
                )

                if "__UNKNOWN__" not in values:
                    values.append("__UNKNOWN__")

                self.category_values_[column] = values

                self.category_maps_[column] = {
                    value: index
                    for index, value in enumerate(values)
                }

                for value in values:
                    self.output_columns_.append(
                        f"{column}__{value}"
                    )

            self._mark_fitted(
                input_rows=len(X),
                input_features=len(
                    self.numeric_columns_
                ),
                output_features=len(
                    self.output_columns_
                ),
                input_columns=list(X.columns),
                output_columns=list(
                    self.output_columns_
                ),
            )

            logger.info(
                "AttachmentFeaturePreprocessor fitted.",
                extra={
                    "extra_fields": {
                        "rows": len(X),
                        "numeric_features": len(
                            self.numeric_columns_
                        ),
                        "categorical_features": len(
                            self.categorical_columns
                        ),
                    }
                },
            )

            return self

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        self._require_fitted()

        X = self._validate_dataframe_schema(
            X,
            allow_extra_columns=False,
        )

        numeric = X[
            self.numeric_columns_
        ].copy()

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        numeric = numeric.fillna(
            self.numeric_fill_values_
        )

        scaled = self.scaler.transform(
            numeric
        )

        scaled = _ensure_finite_dense(
            np.asarray(scaled),
            name="Attachment scaled output",
        )

        result = pd.DataFrame(
            scaled,
            columns=self.numeric_columns_,
            index=X.index,
        )

        for column in self.categorical_columns:
            series = (
                X[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            known = self.category_maps_[column]

            encoded = np.zeros(
                (
                    len(X),
                    len(self.category_values_[column]),
                ),
                dtype=np.float32,
            )

            for row_index, value in enumerate(series):
                mapped = known.get(
                    value,
                    known["__UNKNOWN__"],
                )

                encoded[
                    row_index,
                    mapped,
                ] = 1.0

            category_frame = pd.DataFrame(
                encoded,
                columns=[
                    f"{column}__{value}"
                    for value in self.category_values_[column]
                ],
                index=X.index,
            )

            result = pd.concat(
                [
                    result,
                    category_frame,
                ],
                axis=1,
            )

        return result


# ============================================================================
# MULTI-MODAL FEATURE SCALER
# ============================================================================


class MultiModalFeatureScaler(BasePreprocessor):
    """
    Numerical scaler for already-combined multi-modal features.

    Designed for vectors produced after separate modality-specific
    preprocessing.

    Strong contracts:

        * input dimensionality locked during fit
        * finite-value enforcement
        * optional sparse support
        * output dimensionality verification
        * feature-name preservation
    """

    def __init__(
        self,
        *,
        scaler: str = "standard",
        max_rows: int = DEFAULT_MAX_ROWS,
        allow_sparse: bool = True,
    ) -> None:
        super().__init__(
            max_rows=max_rows,
        )

        if scaler not in {"standard", "robust"}:
            raise ValueError(
                "scaler must be 'standard' or 'robust'."
            )

        self.scaler_type = scaler
        self.allow_sparse = bool(allow_sparse)

        self.scaler = (
            StandardScaler(
                with_mean=not allow_sparse
            )
            if scaler == "standard"
            else RobustScaler(
                with_centering=not allow_sparse
            )
        )

        self.input_shape_: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def _configuration(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "scaler": self.scaler_type,
            "max_rows": self.max_rows,
            "allow_sparse": self.allow_sparse,
        }

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, sparse.spmatrix],
        y: Optional[Any] = None,
    ) -> "MultiModalFeatureScaler":
        with self._lock:
            array = _validate_numeric_array(
                X,
                max_rows=self.max_rows,
            )

            if sparse.issparse(array):
                if not self.allow_sparse:
                    raise PreprocessorValidationError(
                        "Sparse input is disabled for this scaler."
                    )

                _ensure_sparse_finite(
                    array,
                    name="multimodal input",
                )

            else:
                _ensure_finite_dense(
                    array,
                    name="multimodal input",
                )

            self.input_shape_ = tuple(
                array.shape
            )

            self.scaler.fit(array)

            feature_count = int(
                array.shape[1]
            )

            names = [
                f"feature_{index}"
                for index in range(feature_count)
            ]

            self._mark_fitted(
                input_rows=array.shape[0],
                input_features=feature_count,
                output_features=feature_count,
                input_columns=names,
                output_columns=names,
            )

            logger.info(
                "MultiModalFeatureScaler fitted.",
                extra={
                    "extra_fields": {
                        "rows": array.shape[0],
                        "features": feature_count,
                        "scaler": self.scaler_type,
                        "sparse": sparse.issparse(array),
                    }
                },
            )

            return self

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------

    def transform(
        self,
        X: Union[np.ndarray, sparse.spmatrix],
    ) -> Union[np.ndarray, sparse.spmatrix]:
        self._require_fitted()

        array = _validate_numeric_array(
            X,
            max_rows=self.max_rows,
        )

        if (
            self.input_shape_ is None
            or array.shape[1] != self.input_shape_[1]
        ):
            raise PreprocessorValidationError(
                "Multi-modal feature dimensionality changed after fitting: "
                f"expected {self.input_shape_[1] if self.input_shape_ else None}, "
                f"received {array.shape[1]}."
            )

        if sparse.issparse(array):
            if not self.allow_sparse:
                raise PreprocessorValidationError(
                    "Sparse input is disabled."
                )

            output = self.scaler.transform(
                array
            )

            return _ensure_sparse_finite(
                output,
                name="multimodal transformed output",
            )

        output = self.scaler.transform(
            array
        )

        return _ensure_finite_dense(
            np.asarray(output),
            name="multimodal transformed output",
        )


# ============================================================================
# GENERIC PREPROCESSING UTILITIES
# ============================================================================


def validate_preprocessor_artifact(
    filepath: Union[str, Path],
    *,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate the cryptographic integrity and metadata envelope of a
    preprocessor artifact without loading the contained Python object.

    Note
    ----
    This function intentionally uses joblib only after the caller has
    established trust in the artifact. It is not a sandbox.
    """
    path = Path(filepath).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact not found: '{path}'."
        )

    digest = _sha256_file(path)

    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            expected,
        ):
            raise ValueError(
                "expected_sha256 must be a valid SHA-256 digest."
            )

        if not __import__("hmac").compare_digest(
            digest,
            expected,
        ):
            raise PreprocessorArtifactError(
                "Artifact SHA-256 verification failed."
            )

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def get_preprocessor_environment() -> Dict[str, Any]:
    """
    Return runtime information useful for experiment provenance and
    reproducibility reports.
    """
    return {
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "runtime": _runtime_metadata(),
    }


# ============================================================================
# COMPATIBILITY / SELF-DIAGNOSTIC HELPERS
# ============================================================================


def check_preprocessor_compatibility(
    preprocessor: BasePreprocessor,
) -> Dict[str, Any]:
    """
    Inspect a preprocessor for operational readiness.
    """
    if not isinstance(
        preprocessor,
        BasePreprocessor,
    ):
        raise TypeError(
            "preprocessor must inherit from BasePreprocessor."
        )

    checks = {
        "is_fitted": bool(
            preprocessor.is_fitted
        ),
        "has_input_schema": bool(
            preprocessor.feature_names_in_
        ),
        "has_output_schema": bool(
            preprocessor.feature_names_out_
        ),
        "has_configuration_fingerprint": bool(
            preprocessor.configuration_fingerprint_
        ),
        "has_metadata": preprocessor.metadata_ is not None,
    }

    checks["ready"] = all(
        checks.values()
    )

    return checks


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    # Exceptions
    "PreprocessorError",
    "PreprocessorNotFittedError",
    "PreprocessorValidationError",
    "PreprocessorArtifactError",

    # Metadata
    "PreprocessorMetadata",

    # Base
    "BasePreprocessor",

    # Modality preprocessors
    "EmailTextPreprocessor",
    "URLFeaturePreprocessor",
    "AttachmentFeaturePreprocessor",
    "MultiModalFeatureScaler",

    # Utilities
    "validate_preprocessor_artifact",
    "get_preprocessor_environment",
    "check_preprocessor_compatibility",
]