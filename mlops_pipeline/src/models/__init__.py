"""
PhishGuard AI - Model Architecture & Classifier Registry
=========================================================

Version: 14.0.0
File Path: src/models/__init__.py

Purpose
-------
Public package interface and controlled architecture registry for the
PhishGuard AI model layer.

This module provides:

    * Public exports for model implementations.
    * Abstract model contract exposure.
    * Thread-safe architecture registration.
    * Canonical model identifiers and aliases.
    * Duplicate-registration protection.
    * Model-class contract validation.
    * Safe model factory instantiation.
    * Registry introspection.
    * Registry snapshots and deterministic fingerprints.
    * Registry health/audit utilities.
    * Optional explicit unregistration for controlled test/plugin usage.
    * Metadata describing registered architectures.
    * Protection against accidentally registering arbitrary classes.
    * Separation between architecture identity and trained artifact identity.

Important Architectural Boundary
---------------------------------
This registry manages MODEL ARCHITECTURES / CLASSES.

It does NOT manage:

    * trained model weights
    * ONNX files
    * model artifacts
    * model checksums
    * model deployment versions
    * model signatures
    * model promotion
    * artifact trust

Those responsibilities belong to the dedicated model registry/artifact
management subsystem.

Security Philosophy
-------------------
Model architecture names may originate from configuration files, CLI
arguments, API requests, or model manifests. Therefore this module treats
model names as untrusted identifiers.

No arbitrary Python import path is accepted here.

Only explicitly registered classes can be instantiated through this factory.

Thread Safety
-------------
The registry is protected by a re-entrant lock so concurrent initialization,
plugin registration, inspection, and model creation do not corrupt registry
state.

Compatibility
-------------
Designed for Python 3.10+ and modern typing semantics.

"""

from __future__ import annotations

# ==============================================================================
# STANDARD LIBRARY
# ==============================================================================

import hashlib
import inspect
import json
import re
import threading
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

# ==============================================================================
# INTERNAL MODEL IMPLEMENTATIONS
# ==============================================================================

from src.models.base_model import BasePhishingModel
from src.models.attachment_classifier import AttachmentPhishingClassifier
from src.models.email_classifier import EmailPhishingClassifier
from src.models.url_classifier import URLPhishingClassifier

from src.utils.logger import get_logger


logger = get_logger("src.models")


# ==============================================================================
# PACKAGE METADATA
# ==============================================================================

PACKAGE_NAME: str = "phishguard_ai.models"

PACKAGE_VERSION: str = "14.0.0"

MODEL_REGISTRY_VERSION: str = "2.0"

DEFAULT_MODEL_FAMILY: str = "phishing"

SUPPORTED_MODEL_DOMAINS: Tuple[str, ...] = (
    "url",
    "email",
    "attachment",
)


# ==============================================================================
# REGISTRY CONFIGURATION
# ==============================================================================

# Maximum identifier length prevents pathological input from being accepted.
MAX_MODEL_NAME_LENGTH: int = 128

# Model names should remain simple identifiers.
#
# Examples:
#
#   url
#   url_classifier
#   url_phishing_v14
#
# We intentionally do not allow arbitrary import paths such as:
#
#   package.module.Class
#
# because this registry is not a dynamic Python import mechanism.
_MODEL_NAME_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_.:-]{0,127}$"
)


# ==============================================================================
# REGISTRY INTERNAL STATE
# ==============================================================================

# Canonical name -> class
_MODEL_REGISTRY: Dict[str, Type[BasePhishingModel]] = {}

# Alias -> canonical model name
_MODEL_ALIAS_MAP: Dict[str, str] = {}

# Canonical model name -> metadata
_MODEL_METADATA: Dict[str, "ModelRegistration"] = {}

# Protect all registry mutations and reads.
_REGISTRY_LOCK = threading.RLock()


# ==============================================================================
# MODEL REGISTRATION METADATA
# ==============================================================================

@dataclass(frozen=True)
class ModelRegistration:
    """
    Immutable metadata describing a registered model architecture.

    This object describes the architecture registration only. It must not be
    confused with trained-model artifact metadata.
    """

    canonical_name: str
    class_name: str
    module: str
    aliases: Tuple[str, ...]
    version: str
    domain: str
    family: str
    abstract: bool
    constructor_parameters: Tuple[str, ...]
    registration_source: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible metadata representation."""
        return asdict(self)


# ==============================================================================
# IDENTIFIER NORMALIZATION
# ==============================================================================

def _normalize_model_name(model_name: str) -> str:
    """
    Normalize and validate a model architecture identifier.

    Parameters
    ----------
    model_name:
        User/configuration supplied architecture name.

    Returns
    -------
    str
        Canonical normalized identifier.

    Raises
    ------
    TypeError
        If model_name is not a string.

    ValueError
        If identifier is empty, too long, or contains invalid characters.
    """

    if not isinstance(model_name, str):
        raise TypeError(
            "Model architecture name must be a string."
        )

    normalized = model_name.strip().lower()

    if not normalized:
        raise ValueError(
            "Model architecture name cannot be empty."
        )

    if len(normalized) > MAX_MODEL_NAME_LENGTH:
        raise ValueError(
            f"Model architecture name exceeds the maximum allowed length "
            f"of {MAX_MODEL_NAME_LENGTH} characters."
        )

    if not _MODEL_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Invalid model architecture identifier '{model_name}'. "
            "Only lowercase letters, digits, '.', '_', ':', and '-' "
            "are permitted."
        )

    return normalized


# ==============================================================================
# MODEL CLASS CONTRACT VALIDATION
# ==============================================================================

def _validate_model_class(
    model_cls: Type[BasePhishingModel],
) -> None:
    """
    Validate that a class satisfies the PhishGuard model architecture contract.

    This prevents accidental registration of arbitrary classes.

    Parameters
    ----------
    model_cls:
        Candidate model class.

    Raises
    ------
    TypeError
        If the class violates the expected architecture contract.
    """

    if not inspect.isclass(model_cls):
        raise TypeError(
            "Registered model must be a class."
        )

    if not issubclass(model_cls, BasePhishingModel):
        raise TypeError(
            f"Model class '{model_cls.__name__}' must inherit from "
            "BasePhishingModel."
        )

    # Verify that the class belongs to a valid Python module.
    module_name = getattr(model_cls, "__module__", None)

    if not module_name or not isinstance(module_name, str):
        raise TypeError(
            f"Model class '{model_cls.__name__}' does not expose a valid "
            "Python module."
        )


# ==============================================================================
# CONSTRUCTOR INTROSPECTION
# ==============================================================================

def _get_constructor_parameters(
    model_cls: Type[BasePhishingModel],
) -> Tuple[str, ...]:
    """
    Extract constructor parameter names for introspection and diagnostics.

    Internal/private ``self`` is excluded.

    Returns
    -------
    Tuple[str, ...]
        Sorted constructor parameter names.
    """

    try:
        signature = inspect.signature(model_cls.__init__)

        parameters = []

        for name, parameter in signature.parameters.items():

            if name == "self":
                continue

            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            parameters.append(name)

        return tuple(sorted(parameters))

    except (TypeError, ValueError):
        # Introspection failure should not prevent registration.
        logger.warning(
            "Unable to inspect constructor parameters for model class '%s'.",
            model_cls.__name__,
        )
        return tuple()


# ==============================================================================
# REGISTRATION SOURCE
# ==============================================================================

def _get_registration_source(
    model_cls: Type[BasePhishingModel],
) -> str:
    """
    Return the module path where the model class was defined.
    """

    return getattr(
        model_cls,
        "__module__",
        "unknown",
    )


# ==============================================================================
# MODEL VERSION RESOLUTION
# ==============================================================================

def _get_model_version(
    model_cls: Type[BasePhishingModel],
) -> str:
    """
    Resolve model architecture version.

    Classes may optionally expose:

        MODEL_VERSION = "14.0.0"

    Otherwise the package version is used.
    """

    version = getattr(
        model_cls,
        "MODEL_VERSION",
        PACKAGE_VERSION,
    )

    if not isinstance(version, str) or not version.strip():
        return PACKAGE_VERSION

    return version.strip()


# ==============================================================================
# MODEL DOMAIN RESOLUTION
# ==============================================================================

def _get_model_domain(
    model_cls: Type[BasePhishingModel],
    canonical_name: str,
) -> str:
    """
    Resolve the logical domain of a model.

    A model may optionally expose:

        MODEL_DOMAIN = "url"

    Otherwise the canonical identifier is inspected.
    """

    explicit_domain = getattr(
        model_cls,
        "MODEL_DOMAIN",
        None,
    )

    if isinstance(explicit_domain, str) and explicit_domain.strip():
        return explicit_domain.strip().lower()

    for domain in SUPPORTED_MODEL_DOMAINS:
        if canonical_name == domain or canonical_name.startswith(
            f"{domain}_"
        ):
            return domain

    return "unknown"


# ==============================================================================
# REGISTRATION DECORATOR
# ==============================================================================

def register_model(
    name_or_aliases: Union[str, Sequence[str]],
    *,
    canonical_name: Optional[str] = None,
    version: Optional[str] = None,
    domain: Optional[str] = None,
    family: str = DEFAULT_MODEL_FAMILY,
    allow_override: bool = False,
) -> Callable[[Type[BasePhishingModel]], Type[BasePhishingModel]]:
    """
    Decorator used to register a model architecture.

    Parameters
    ----------
    name_or_aliases:
        One identifier or multiple identifiers.

    canonical_name:
        Stable primary architecture identifier. If omitted, the first
        supplied name becomes canonical.

    version:
        Optional architecture version.

    domain:
        Logical model domain, e.g. ``url`` or ``email``.

    family:
        Model family identifier.

    allow_override:
        Explicitly permit replacing an existing registration.

    Notes
    -----
    Silent registration replacement is disabled by default.

    This is intentional because accidental duplicate registration can create
    non-deterministic model resolution.
    """

    if isinstance(name_or_aliases, str):
        raw_aliases = [name_or_aliases]
    else:
        raw_aliases = list(name_or_aliases)

    if not raw_aliases:
        raise ValueError(
            "At least one model name or alias must be provided."
        )

    normalized_aliases = tuple(
        dict.fromkeys(
            _normalize_model_name(alias)
            for alias in raw_aliases
        )
    )

    if canonical_name is None:
        resolved_canonical = normalized_aliases[0]
    else:
        resolved_canonical = _normalize_model_name(canonical_name)

    if resolved_canonical not in normalized_aliases:
        normalized_aliases = (
            resolved_canonical,
            *normalized_aliases,
        )

    if domain is not None:
        normalized_domain = _normalize_model_name(domain)
    else:
        normalized_domain = ""

    if not isinstance(family, str) or not family.strip():
        raise ValueError(
            "Model family must be a non-empty string."
        )

    def decorator(
        model_cls: Type[BasePhishingModel],
    ) -> Type[BasePhishingModel]:

        _validate_model_class(model_cls)

        resolved_version = (
            version.strip()
            if isinstance(version, str) and version.strip()
            else _get_model_version(model_cls)
        )

        resolved_domain = (
            normalized_domain
            if normalized_domain
            else _get_model_domain(
                model_cls,
                resolved_canonical,
            )
        )

        constructor_parameters = _get_constructor_parameters(
            model_cls
        )

        registration_source = _get_registration_source(
            model_cls
        )

        metadata = ModelRegistration(
            canonical_name=resolved_canonical,
            class_name=model_cls.__name__,
            module=registration_source,
            aliases=normalized_aliases,
            version=resolved_version,
            domain=resolved_domain,
            family=family.strip(),
            abstract=inspect.isabstract(model_cls),
            constructor_parameters=constructor_parameters,
            registration_source=registration_source,
        )

        with _REGISTRY_LOCK:

            # ------------------------------------------------------------------
            # Never allow abstract architectures to become factory targets.
            # ------------------------------------------------------------------

            if inspect.isabstract(model_cls):
                raise TypeError(
                    f"Cannot register abstract model class "
                    f"'{model_cls.__name__}' as an instantiable architecture."
                )

            # ------------------------------------------------------------------
            # Validate canonical-name collision.
            # ------------------------------------------------------------------

            existing_class = _MODEL_REGISTRY.get(
                resolved_canonical
            )

            if (
                existing_class is not None
                and existing_class is not model_cls
                and not allow_override
            ):
                raise RuntimeError(
                    f"Model architecture '{resolved_canonical}' is already "
                    f"registered to '{existing_class.__name__}'. "
                    "Use a unique canonical name or explicitly set "
                    "allow_override=True."
                )

            # ------------------------------------------------------------------
            # Validate every alias collision.
            # ------------------------------------------------------------------

            for alias in normalized_aliases:

                existing_canonical = _MODEL_ALIAS_MAP.get(alias)

                if (
                    existing_canonical is not None
                    and existing_canonical != resolved_canonical
                    and not allow_override
                ):
                    raise RuntimeError(
                        f"Model alias '{alias}' is already assigned to "
                        f"architecture '{existing_canonical}'."
                    )

            # ------------------------------------------------------------------
            # Remove old aliases during explicit replacement.
            # ------------------------------------------------------------------

            if allow_override:

                old_aliases = []

                for alias, registered_canonical in list(
                    _MODEL_ALIAS_MAP.items()
                ):
                    if registered_canonical == resolved_canonical:
                        old_aliases.append(alias)

                for alias in old_aliases:
                    del _MODEL_ALIAS_MAP[alias]

            # ------------------------------------------------------------------
            # Register canonical architecture.
            # ------------------------------------------------------------------

            _MODEL_REGISTRY[
                resolved_canonical
            ] = model_cls

            # ------------------------------------------------------------------
            # Register aliases.
            # ------------------------------------------------------------------

            for alias in normalized_aliases:

                _MODEL_ALIAS_MAP[alias] = resolved_canonical

            # ------------------------------------------------------------------
            # Register metadata.
            # ------------------------------------------------------------------

            _MODEL_METADATA[
                resolved_canonical
            ] = metadata

        logger.info(
            "Registered model architecture '%s' -> %s.%s "
            "(version=%s, domain=%s, aliases=%s).",
            resolved_canonical,
            registration_source,
            model_cls.__name__,
            resolved_version,
            resolved_domain,
            list(normalized_aliases),
        )

        return model_cls

    return decorator


# ==============================================================================
# MODEL RESOLUTION
# ==============================================================================

def resolve_model_name(
    model_name: str,
) -> str:
    """
    Resolve an alias to its canonical architecture identifier.

    Parameters
    ----------
    model_name:
        Architecture name or alias.

    Returns
    -------
    str
        Canonical model identifier.

    Raises
    ------
    ValueError
        If the model is not registered.
    """

    normalized = _normalize_model_name(model_name)

    with _REGISTRY_LOCK:

        canonical = _MODEL_ALIAS_MAP.get(
            normalized
        )

        if canonical is None:
            available = sorted(
                _MODEL_ALIAS_MAP.keys()
            )

            raise ValueError(
                f"Unknown model architecture '{model_name}'. "
                f"Available identifiers: {available}"
            )

        return canonical


# ==============================================================================
# MODEL CLASS LOOKUP
# ==============================================================================

def get_model_class(
    model_name: str,
) -> Type[BasePhishingModel]:
    """
    Retrieve the registered class without instantiating it.

    This is useful for:

    * dependency inspection
    * capability inspection
    * test frameworks
    * model validation
    * registry health checks
    """

    canonical = resolve_model_name(
        model_name
    )

    with _REGISTRY_LOCK:

        model_cls = _MODEL_REGISTRY.get(
            canonical
        )

        if model_cls is None:
            raise RuntimeError(
                f"Registry inconsistency detected: canonical architecture "
                f"'{canonical}' has no registered class."
            )

        return model_cls


# ==============================================================================
# MODEL FACTORY
# ==============================================================================

def get_model_by_name(
    model_name: str,
    **model_kwargs: Any,
) -> BasePhishingModel:
    """
    Instantiate a registered model architecture.

    Parameters
    ----------
    model_name:
        Canonical architecture identifier or registered alias.

    **model_kwargs:
        Constructor parameters for the selected model.

    Returns
    -------
    BasePhishingModel
        Instantiated model.

    Raises
    ------
    ValueError
        If architecture is unknown.

    TypeError
        If constructor arguments are invalid.

    RuntimeError
        If registry integrity is compromised.

    Security Note
    ------------
    This function does not dynamically import classes from user-provided
    module paths. Only trusted classes explicitly registered in this module
    can be instantiated.
    """

    canonical = resolve_model_name(
        model_name
    )

    model_cls = get_model_class(
        canonical
    )

    metadata = get_model_metadata(
        canonical
    )

    # --------------------------------------------------------------------------
    # Avoid logging potentially sensitive constructor values.
    #
    # Example:
    #
    # password="..."
    # api_key="..."
    #
    # should never appear in normal production logs.
    # --------------------------------------------------------------------------

    if model_kwargs:

        safe_keys = sorted(
            str(key)
            for key in model_kwargs.keys()
        )

        logger.debug(
            "Instantiating model architecture '%s' "
            "with constructor parameters: %s",
            canonical,
            safe_keys,
        )
    else:

        logger.debug(
            "Instantiating model architecture '%s' "
            "with default constructor parameters.",
            canonical,
        )

    try:

        instance = model_cls(
            **model_kwargs
        )

    except TypeError as exc:

        logger.error(
            "Invalid constructor arguments for model architecture '%s': %s",
            canonical,
            exc,
        )

        raise

    except Exception as exc:

        logger.error(
            "Failed to instantiate model architecture '%s': %s",
            canonical,
            exc,
        )

        raise RuntimeError(
            f"Failed to instantiate model architecture "
            f"'{canonical}'."
        ) from exc

    if not isinstance(
        instance,
        BasePhishingModel,
    ):
        raise RuntimeError(
            f"Registry contract violation: architecture '{canonical}' "
            "did not produce a BasePhishingModel instance."
        )

    logger.info(
        "Instantiated model architecture '%s' "
        "(class=%s, version=%s, domain=%s).",
        canonical,
        metadata.class_name,
        metadata.version,
        metadata.domain,
    )

    return instance


# ==============================================================================
# MODEL METADATA
# ==============================================================================

def get_model_metadata(
    model_name: str,
) -> ModelRegistration:
    """
    Return immutable registration metadata for a model architecture.
    """

    canonical = resolve_model_name(
        model_name
    )

    with _REGISTRY_LOCK:

        metadata = _MODEL_METADATA.get(
            canonical
        )

        if metadata is None:
            raise RuntimeError(
                f"Registry metadata missing for architecture "
                f"'{canonical}'."
            )

        return metadata


# ==============================================================================
# LIST REGISTERED MODELS
# ==============================================================================

def list_registered_models() -> List[str]:
    """
    Return canonical model architecture identifiers.

    Returns
    -------
    List[str]
        Sorted canonical identifiers.
    """

    with _REGISTRY_LOCK:
        return sorted(
            _MODEL_REGISTRY.keys()
        )


# ==============================================================================
# LIST ALL ALIASES
# ==============================================================================

def list_model_aliases() -> Dict[str, str]:
    """
    Return a copy of the alias -> canonical-name mapping.

    A copy is returned so callers cannot mutate registry state.
    """

    with _REGISTRY_LOCK:
        return dict(
            sorted(
                _MODEL_ALIAS_MAP.items()
            )
        )


# ==============================================================================
# MODEL REGISTRY SNAPSHOT
# ==============================================================================

def get_registry_snapshot() -> Dict[str, Dict[str, Any]]:
    """
    Return a complete registry metadata snapshot.

    The returned object is detached from internal registry state.
    """

    with _REGISTRY_LOCK:

        return {
            canonical: metadata.to_dict()
            for canonical, metadata
            in sorted(
                _MODEL_METADATA.items()
            )
        }


# ==============================================================================
# PUBLIC READ-ONLY REGISTRY VIEW
# ==============================================================================

def get_registry_view() -> Mapping[str, Type[BasePhishingModel]]:
    """
    Return a read-only mapping of canonical model names to classes.

    This is useful for diagnostics without exposing mutable registry state.
    """

    with _REGISTRY_LOCK:

        return MappingProxyType(
            dict(_MODEL_REGISTRY)
        )


# ==============================================================================
# MODEL AVAILABILITY CHECK
# ==============================================================================

def is_model_registered(
    model_name: str,
) -> bool:
    """
    Check whether an architecture or alias is registered.

    Invalid identifiers return False rather than raising an exception.
    """

    try:
        normalized = _normalize_model_name(
            model_name
        )
    except (TypeError, ValueError):
        return False

    with _REGISTRY_LOCK:
        return normalized in _MODEL_ALIAS_MAP


# ==============================================================================
# DOMAIN FILTERING
# ==============================================================================

def list_models_by_domain(
    domain: str,
) -> List[str]:
    """
    Return canonical architectures belonging to a logical domain.
    """

    normalized_domain = _normalize_model_name(
        domain
    )

    with _REGISTRY_LOCK:

        return sorted(
            canonical
            for canonical, metadata
            in _MODEL_METADATA.items()
            if metadata.domain == normalized_domain
        )


# ==============================================================================
# VERSION LOOKUP
# ==============================================================================

def get_model_version(
    model_name: str,
) -> str:
    """
    Return registered architecture version.
    """

    return get_model_metadata(
        model_name
    ).version


# ==============================================================================
# MODEL CLASS NAME LOOKUP
# ==============================================================================

def get_model_class_name(
    model_name: str,
) -> str:
    """
    Return implementation class name for an architecture.
    """

    return get_model_metadata(
        model_name
    ).class_name


# ==============================================================================
# CONSTRUCTOR CONTRACT INSPECTION
# ==============================================================================

def get_model_constructor_parameters(
    model_name: str,
) -> Tuple[str, ...]:
    """
    Return supported constructor parameter names.

    This can be used by configuration validation before attempting
    instantiation.
    """

    return get_model_metadata(
        model_name
    ).constructor_parameters


# ==============================================================================
# SAFE CONSTRUCTOR ARGUMENT VALIDATION
# ==============================================================================

def validate_model_kwargs(
    model_name: str,
    model_kwargs: Mapping[str, Any],
) -> None:
    """
    Validate that supplied keyword arguments correspond to the model's
    explicit constructor parameters.

    Parameters
    ----------
    model_name:
        Registered architecture.

    model_kwargs:
        Constructor keyword arguments.

    Raises
    ------
    TypeError
        If unknown constructor parameters are supplied.

    Notes
    -----
    This function does not validate semantic parameter values. That remains
    the responsibility of the model implementation.
    """

    if not isinstance(
        model_kwargs,
        Mapping,
    ):
        raise TypeError(
            "model_kwargs must be a mapping."
        )

    model_cls = get_model_class(
        model_name
    )

    try:

        signature = inspect.signature(
            model_cls.__init__
        )

    except (TypeError, ValueError) as exc:

        raise RuntimeError(
            f"Unable to inspect constructor for model "
            f"'{model_name}'."
        ) from exc

    parameters = signature.parameters

    accepts_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_kwargs:
        return

    unknown = sorted(
        key
        for key in model_kwargs.keys()
        if key not in parameters
    )

    if unknown:

        raise TypeError(
            f"Unsupported constructor parameters for model "
            f"'{model_name}': {unknown}"
        )


# ==============================================================================
# REGISTRY INTEGRITY AUDIT
# ==============================================================================

def audit_model_registry() -> Dict[str, Any]:
    """
    Perform an internal consistency audit.

    Returns
    -------
    Dict[str, Any]
        Audit report suitable for health checks and diagnostics.

    The registry is considered healthy only if:

        * Every canonical name has a class.
        * Every class inherits from BasePhishingModel.
        * Every alias resolves to an existing canonical architecture.
        * Metadata exists for every canonical architecture.
        * Metadata aliases match registry aliases.
        * No registered class is abstract.
    """

    errors: List[str] = []
    warnings: List[str] = []

    with _REGISTRY_LOCK:

        canonical_names = set(
            _MODEL_REGISTRY.keys()
        )

        metadata_names = set(
            _MODEL_METADATA.keys()
        )

        if canonical_names != metadata_names:

            errors.append(
                "Model registry and metadata registry contain "
                "different canonical model sets."
            )

        for canonical, model_cls in _MODEL_REGISTRY.items():

            try:
                _validate_model_class(
                    model_cls
                )
            except Exception as exc:
                errors.append(
                    f"Invalid model class for '{canonical}': {exc}"
                )

            if inspect.isabstract(model_cls):

                errors.append(
                    f"Abstract model class registered as "
                    f"instantiable architecture: '{canonical}'."
                )

            if canonical not in _MODEL_ALIAS_MAP:

                errors.append(
                    f"Canonical architecture '{canonical}' "
                    "has no alias mapping."
                )

        for alias, canonical in _MODEL_ALIAS_MAP.items():

            if canonical not in _MODEL_REGISTRY:

                errors.append(
                    f"Alias '{alias}' points to missing "
                    f"architecture '{canonical}'."
                )

        for canonical, metadata in _MODEL_METADATA.items():

            for alias in metadata.aliases:

                if _MODEL_ALIAS_MAP.get(alias) != canonical:

                    errors.append(
                        f"Metadata alias mismatch: '{alias}' "
                        f"does not resolve to '{canonical}'."
                    )

        if not _MODEL_REGISTRY:

            warnings.append(
                "Model registry is empty."
            )

        return {
            "healthy": not errors,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "canonical_model_count": len(
                _MODEL_REGISTRY
            ),
            "alias_count": len(
                _MODEL_ALIAS_MAP
            ),
        }


# ==============================================================================
# REGISTRY FINGERPRINT
# ==============================================================================

def get_registry_fingerprint(
    algorithm: str = "sha256",
) -> str:
    """
    Generate a deterministic fingerprint of the architecture registry.

    This fingerprint can be recorded alongside training/evaluation metadata
    to identify exactly which architecture registry configuration was active.

    Parameters
    ----------
    algorithm:
        Hash algorithm supported by hashlib.

    Returns
    -------
    str
        Hexadecimal registry fingerprint.
    """

    try:
        hasher = hashlib.new(
            algorithm
        )
    except ValueError as exc:

        raise ValueError(
            f"Unsupported hashing algorithm '{algorithm}'."
        ) from exc

    snapshot = get_registry_snapshot()

    canonical_json = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    hasher.update(
        canonical_json.encode(
            "utf-8"
        )
    )

    return hasher.hexdigest()


# ==============================================================================
# REGISTRY HEALTH CHECK
# ==============================================================================

def ensure_registry_healthy() -> None:
    """
    Raise RuntimeError if the architecture registry is inconsistent.

    This is useful during application startup, CI, and deployment validation.
    """

    report = audit_model_registry()

    if not report["healthy"]:

        formatted_errors = "\n".join(
            f"- {error}"
            for error in report["errors"]
        )

        raise RuntimeError(
            "PhishGuard AI model architecture registry failed "
            f"integrity validation:\n{formatted_errors}"
        )


# ==============================================================================
# CONTROLLED UNREGISTRATION
# ==============================================================================

def unregister_model(
    model_name: str,
) -> None:
    """
    Remove a model architecture from the registry.

    WARNING
    -------
    This function should generally be used only by:

        * test suites
        * controlled plugin lifecycle management
        * development tooling

    Production code should normally treat architecture registrations as
    immutable after application startup.
    """

    canonical = resolve_model_name(
        model_name
    )

    with _REGISTRY_LOCK:

        if canonical not in _MODEL_REGISTRY:

            raise KeyError(
                f"Model architecture '{canonical}' "
                "is not registered."
            )

        aliases_to_remove = [
            alias
            for alias, registered_canonical
            in _MODEL_ALIAS_MAP.items()
            if registered_canonical == canonical
        ]

        for alias in aliases_to_remove:
            del _MODEL_ALIAS_MAP[alias]

        del _MODEL_REGISTRY[
            canonical
        ]

        _MODEL_METADATA.pop(
            canonical,
            None,
        )

    logger.warning(
        "Unregistered model architecture '%s'.",
        canonical,
    )


# ==============================================================================
# BULK REGISTRY INITIALIZATION
# ==============================================================================

def _register_builtin_models() -> None:
    """
    Register trusted built-in PhishGuard model architectures.

    This function is intentionally explicit rather than dynamically importing
    arbitrary model classes.
    """

    builtins = (
        (
            BasePhishingModel,
            (
                "base",
                "base_model",
            ),
        ),
        (
            URLPhishingClassifier,
            (
                "url",
                "url_classifier",
                "url_phishing_v13",
                "url_phishing",
            ),
        ),
        (
            EmailPhishingClassifier,
            (
                "email",
                "email_classifier",
                "email_phishing_v13",
                "email_phishing",
            ),
        ),
        (
            AttachmentPhishingClassifier,
            (
                "attachment",
                "attachment_classifier",
                "attachment_phishing_v13",
                "attachment_phishing",
            ),
        ),
    )

    for model_cls, aliases in builtins:

        # ----------------------------------------------------------------------
        # The abstract base class is a contract and NOT an instantiable model.
        # Therefore it is exposed publicly but deliberately excluded from the
        # factory registry.
        # ----------------------------------------------------------------------

        if inspect.isabstract(model_cls):

            logger.debug(
                "Skipping abstract model contract '%s' "
                "from instantiable architecture registry.",
                model_cls.__name__,
            )

            continue

        register_model(
            aliases,
            canonical_name=aliases[0],
        )(model_cls)


# ==============================================================================
# INITIALIZE TRUSTED BUILTIN REGISTRY
# ==============================================================================

_register_builtin_models()


# ==============================================================================
# STARTUP REGISTRY VALIDATION
# ==============================================================================

try:
    ensure_registry_healthy()
except Exception:
    # Do not silently continue with an inconsistent registry.
    logger.exception(
        "Fatal model architecture registry initialization failure."
    )
    raise


# ==============================================================================
# PUBLIC PACKAGE CONTRACT
# ==============================================================================

__all__ = [
    # --------------------------------------------------------------------------
    # Package metadata
    # --------------------------------------------------------------------------
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "MODEL_REGISTRY_VERSION",
    "DEFAULT_MODEL_FAMILY",

    # --------------------------------------------------------------------------
    # Abstract model contract
    # --------------------------------------------------------------------------
    "BasePhishingModel",

    # --------------------------------------------------------------------------
    # Specialized classifiers
    # --------------------------------------------------------------------------
    "URLPhishingClassifier",
    "EmailPhishingClassifier",
    "AttachmentPhishingClassifier",

    # --------------------------------------------------------------------------
    # Registry metadata
    # --------------------------------------------------------------------------
    "ModelRegistration",

    # --------------------------------------------------------------------------
    # Registration
    # --------------------------------------------------------------------------
    "register_model",

    # --------------------------------------------------------------------------
    # Resolution / factory
    # --------------------------------------------------------------------------
    "resolve_model_name",
    "get_model_class",
    "get_model_by_name",

    # --------------------------------------------------------------------------
    # Metadata / introspection
    # --------------------------------------------------------------------------
    "get_model_metadata",
    "get_model_version",
    "get_model_class_name",
    "get_model_constructor_parameters",

    # --------------------------------------------------------------------------
    # Registry inspection
    # --------------------------------------------------------------------------
    "list_registered_models",
    "list_model_aliases",
    "list_models_by_domain",
    "get_registry_snapshot",
    "get_registry_view",
    "get_registry_fingerprint",

    # --------------------------------------------------------------------------
    # Validation / health
    # --------------------------------------------------------------------------
    "is_model_registered",
    "validate_model_kwargs",
    "audit_model_registry",
    "ensure_registry_healthy",

    # --------------------------------------------------------------------------
    # Controlled lifecycle
    # --------------------------------------------------------------------------
    "unregister_model",
]