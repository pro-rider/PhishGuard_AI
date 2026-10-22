"""
PhishGuard_AI - Deterministic Rule Engine & Policy Compiler
Production Version 9.1.0
Security policy compiler/evaluator for declarative YAML rules.
"""

from __future__ import annotations
import enum
import hashlib
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

try:
    import yaml
    try:
        from yaml import CSafeLoader as SafeLoader
    except ImportError:
        from yaml import SafeLoader # type: ignore
    YAML_AVAILABLE = True
except ImportError:
    yaml = None # type: ignore
    SafeLoader = Any # type: ignore
    YAML_AVAILABLE = False

# Optional third-party regex engine.
try:
    import regex as regex_module # type: ignore
    REGEX_TIMEOUT_AVAILABLE = True
except ImportError:
    regex_module = None # type: ignore
    REGEX_TIMEOUT_AVAILABLE = False

from engine.orchestrator import (
    Finding,
    ScanContext,
    SecurityVerdict,
)

logger = logging.getLogger(__name__)

# ============================================================================
# VERSION / LIMITS (UPDATED)
# ============================================================================
RULE_ENGINE_VERSION = "9.1.0"
RULESET_SCHEMA_VERSION = 1

DEFAULT_MAX_EVALUATION_TIME_MS = 50.0
MAX_RULES = 2_000
MAX_RULE_ID_LENGTH = 128
MAX_RULE_NAME_LENGTH = 256
MAX_CATEGORY_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 4_096
MAX_TAGS_PER_RULE = 32
MAX_TAG_LENGTH = 64
MAX_AST_DEPTH = 12
MAX_AST_NODES = 128
MAX_TOTAL_NODES = 1_000          # NEW: global AST node limit
MAX_REGEX_LENGTH = 1_024
MAX_REGEX_INPUT_LENGTH = 64 * 1024
MAX_PROPERTY_DEPTH = 12          # NEW: property resolver depth cap
REGEX_TIMEOUT_MS = 50.0          # UPDATED: safer default

MAX_LIST_ITEMS = 256
MAX_TARGET_STRING_LENGTH = 4_096
MAX_FINDINGS_TO_SCAN = 2_000
MIN_RULE_SCORE = 0.0
MAX_RULE_SCORE = 100.0
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# ============================================================================
# DANGEROUS REGEX PATTERNS (UPDATED)
# ============================================================================
DANGEROUS_REGEX_PATTERNS = (
    re.compile(r"\([^)]*[+*][^)]*\)[+*]"),
    re.compile(r"\([^)]*\{\d+,\}[^)]*\)[+*]"),
    re.compile(r"\.\*.*\.\*"),
    re.compile(r"\.\+.*\.\+"),
    re.compile(r"\[\s*[:\w]+\s*-\s*[:\w]+\s*\]"),   # NEW: character-range ReDoS
    re.compile(r"(\w+)\1+"),                        # NEW: exponential backtracking
)

# ============================================================================
# ENUMERATIONS
# ============================================================================
class RuleOperator(str, enum.Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    HAS_FINDING = "has_finding"
    HAS_CATEGORY = "has_category"
    HAS_SEVERITY = "has_severity"
    FINDING_COUNT_GREATER_THAN = "finding_count_greater_than"
    FINDING_COUNT_AT_LEAST = "finding_count_at_least"
    IS_IN = "is_in"                                 # NEW
    ALL = "all"                                     # NEW

class LogicOperator(str, enum.Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

# ============================================================================
# EXCEPTIONS
# ============================================================================
class RuleEngineError(Exception): pass
class RuleCompilationError(RuleEngineError): pass
class RuleSetValidationError(RuleEngineError): pass
class RuleEvaluationTimeout(RuleEngineError): pass

# ============================================================================
# AST DATA STRUCTURES
# ============================================================================
@dataclass(frozen=True, slots=True)
class ConditionNode:
    field_path: Optional[str]
    operator: RuleOperator
    target_value: Any
    compiled_regex: Optional[Any] = None

@dataclass(frozen=True, slots=True)
class LogicNode:
    logic: LogicOperator
    conditions: Tuple[Union["LogicNode", ConditionNode], ...] = ()

@dataclass(frozen=True, slots=True)
class CompiledRule:
    rule_id: str
    name: str
    severity: SecurityVerdict
    score: float
    score_damping: float = 1.0                       # NEW: per-rule damping
    confidence: float
    category: str
    description: str
    root_condition: LogicNode
    enabled: bool = True
    tags: Tuple[str, ...] = ()
    version: str = RULESET_SCHEMA_VERSION.__str__()

@dataclass(frozen=True, slots=True)
class RuleMatchResult:
    rule_id: str
    rule_name: str
    severity: SecurityVerdict
    score: float
    confidence: float
    category: str
    description: str
    matched_field: Optional[str] = None
    matched_value: Optional[Any] = None
    tags: Tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class RuleSetSnapshot:
    rules: Tuple[CompiledRule, ...]
    version: str
    fingerprint: str
    loaded_at_monotonic: float

# ============================================================================
# PROPERTY RESOLVER (UPDATED)
# ============================================================================
class PropertyResolver:
    _SAFE_ATTRIBUTE_NAMES: Set[str] = { ... }  # unchanged (kept for brevity)

    @classmethod
    def resolve(cls, target: Any, path: str) -> Any:
        if target is None or not path or len(path) > 256:
            return None
        parts = path.split(".")
        if len(parts) > MAX_PROPERTY_DEPTH:
            return None
        current = target
        for part in parts:
            if not part:
                return None
            current = cls._resolve_one(current, part)
            if current is None:
                return None
        return current

    @classmethod
    def _resolve_one(cls, current: Any, part: str) -> Any:
        if current is None:
            return None
        if isinstance(current, Mapping):
            return current.get(part)
        if isinstance(current, (list, tuple, set)):
            if part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    return current[index]
            return None
        if part not in cls._SAFE_ATTRIBUTE_NAMES:
            return None
        try:
            return getattr(current, part, None)
        except Exception:
            return None

# ============================================================================
# NORMALIZATION / VALIDATION HELPERS (UPDATED)
# ============================================================================
class RuleValidation:
    # ... (all original validators unchanged) ...

    @staticmethod
    def bounded_string(value: Any, *, name: str, maximum: int, allow_empty: bool = True) -> str:
        # ... (original) ...
        pass

    @staticmethod
    def validate_target(value: Any, *, name: str = "condition.value") -> Any:
        # ... (original) ...
        if isinstance(value, list):
            if len(value) > MAX_LIST_ITEMS:
                raise RuleCompilationError(f"{name} exceeds maximum list size")
            return [RuleValidation.validate_target(item, name=f"{name}[]") for item in value]
        return value

# ============================================================================
# REGEX COMPILER (UPDATED)
# ============================================================================
class SafeRegexCompiler:
    @classmethod
    def compile(cls, pattern: str) -> Any:
        if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_REGEX_LENGTH:
            raise RuleCompilationError("Invalid regex pattern")
        if REGEX_TIMEOUT_AVAILABLE:
            try:
                return regex_module.compile(pattern, regex_module.IGNORECASE)
            except Exception as exc:
                raise RuleCompilationError(f"Invalid regex pattern: {type(exc).__name__}") from exc
        # conservative fallback + new dangerous patterns
        for dangerous_pattern in DANGEROUS_REGEX_PATTERNS:
            if dangerous_pattern.search(pattern):
                raise RuleCompilationError("Regex rejected by ReDoS safety policy")
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise RuleCompilationError(f"Invalid regex pattern: {exc}") from exc

    @classmethod
    def search(cls, compiled: Any, value: str) -> bool:
        if len(value) > MAX_REGEX_INPUT_LENGTH:
            value = value[:MAX_REGEX_INPUT_LENGTH]
        if REGEX_TIMEOUT_AVAILABLE:
            try:
                result = compiled.search(value, timeout=REGEX_TIMEOUT_MS / 1000.0)
                return bool(result)
            except TimeoutError:
                return False
            except Exception:
                return False
        try:
            return bool(compiled.search(value))
        except (re.error, RecursionError):
            return False

# ============================================================================
# AST COMPILER (UPDATED)
# ============================================================================
class RuleCompiler:
    @classmethod
    def compile_rule(cls, raw_rule: Mapping[str, Any]) -> CompiledRule:
        # ... (all original compilation logic unchanged) ...

        # NEW: global node limit check
        if node_counter[0] > MAX_TOTAL_NODES:
            raise RuleCompilationError("Maximum total rule nodes exceeded")

        return CompiledRule(
            rule_id=rule_id,
            name=name,
            severity=severity,
            score=score,
            score_damping=1.0,
            confidence=confidence,
            category=category,
            description=description,
            root_condition=root_node,
            enabled=enabled,
            tags=tuple(tags),
            version=str(raw_rule.get("version", RULESET_SCHEMA_VERSION)),
        )

    @classmethod
    def _compile_logic_node(cls, raw_logic: Any, *, depth: int, node_counter: List[int]) -> LogicNode:
        # ... (original) ...
        if depth > MAX_AST_DEPTH or node_counter[0] > MAX_TOTAL_NODES:
            raise RuleCompilationError("Maximum rule AST depth/nodes exceeded")

        # NEW: support for ALL operator
        if logic == LogicOperator.ALL:
            if len(raw_items) < 2:
                raise RuleCompilationError("ALL must contain at least two conditions")

        # ... rest of original _compile_logic_node ...

    @classmethod
    def _compile_condition_node(cls, raw_cond: Mapping[str, Any], *, node_counter: List[int]) -> ConditionNode:
        # ... (original) ...
        if operator == RuleOperator.ALL:  # NEW
            raise RuleCompilationError("ALL operator cannot be used on conditions")
        return ConditionNode(
            field_path=field_path,
            operator=operator,
            target_value=target_value,
            compiled_regex=compiled_regex,
        )

# ============================================================================
# EVALUATION ENGINE (UPDATED)
# ============================================================================
class RuleEvaluator:
    @classmethod
    def evaluate_rule(cls, rule: CompiledRule, context: ScanContext, findings: Sequence[Finding], deadline_monotonic: float) -> Optional[RuleMatchResult]:
        if not rule.enabled:
            return None
        cls._check_deadline(deadline_monotonic)
        matched = cls._eval_node(rule.root_condition, context, findings, deadline_monotonic, depth=0)
        if not matched:
            return None
        return RuleMatchResult(...)  # original

    @classmethod
    def _check_deadline(cls, deadline_monotonic: float) -> None:
        if time.monotonic() >= deadline_monotonic:
            raise RuleEvaluationTimeout()

    @classmethod
    def _eval_node(cls, node: Union[LogicNode, ConditionNode], context: ScanContext, findings: Sequence[Finding], deadline_monotonic: float, depth: int) -> bool:
        cls._check_deadline(deadline_monotonic)
        if depth > MAX_AST_DEPTH:
            return False
        if isinstance(node, ConditionNode):
            return cls._eval_condition(node, context, findings, deadline_monotonic)
        children = node.conditions
        if node.logic == LogicOperator.AND:
            for child in children:
                if not cls._eval_node(child, context, findings, deadline_monotonic, depth + 1):
                    return False
            return True
        if node.logic == LogicOperator.OR:
            for child in children:
                if cls._eval_node(child, context, findings, deadline_monotonic, depth + 1):
                    return True
            return False
        if node.logic == LogicOperator.NOT:
            return not cls._eval_node(children[0], context, findings, deadline_monotonic, depth + 1)
        if node.logic == LogicOperator.ALL:  # NEW
            for child in children:
                if not cls._eval_node(child, context, findings, deadline_monotonic, depth + 1):
                    return False
            return True
        return False

    @classmethod
    def _eval_condition(cls, cond: ConditionNode, context: ScanContext, findings: Sequence[Finding], deadline_monotonic: float) -> bool:
        cls._check_deadline(deadline_monotonic)
        op = cond.operator

        # ... (all original operators unchanged) ...

        if op == RuleOperator.IS_IN:  # NEW
            if cond.field_path:
                actual = PropertyResolver.resolve(context, cond.field_path)
                return actual in cond.target_value
            return False

        if op == RuleOperator.ALL:  # NEW
            if not cond.field_path:
                return False
            actual = PropertyResolver.resolve(context, cond.field_path)
            if isinstance(actual, (list, tuple, set)):
                return all(item in cond.target_value for item in actual)
            return False

        # ... (rest of original _eval_condition) ...

    # _contains, _numeric_compare, finding accessors unchanged (with MAX_FINDINGS_TO_SCAN caps)

# ============================================================================
# MAIN RULE ENGINE (UPDATED)
# ============================================================================
class RuleEngine:
    def __init__(self, *, max_rules: int = MAX_RULES) -> None:
        # ... (original) ...
        pass

    # ... (all original methods) ...

    def load_rules_from_yaml(self, yaml_content: str, *, source_name: str = "<memory>") -> int:
        # ... (original parsing) ...

        # UPDATED: global node limit check
        if node_counter[0] > MAX_TOTAL_NODES:
            raise RuleSetValidationError(f"Total rule nodes exceed {MAX_TOTAL_NODES}")

        # ... rest of original load_rules_from_yaml ...

        # NEW: findings metadata now always includes fingerprint
        metadata: Dict[str, Any] = {
            "rule_id": match.rule_id,
            "rule_name": match.rule_name,
            "rule_version": match.rule_id,
            "ruleset_version": snapshot.version,
            "ruleset_fingerprint": snapshot.fingerprint,   # ALWAYS included
            "tags": ",".join(match.tags),
            "confidence": match.confidence,
        }
        # ... rest unchanged ...

    def clear(self) -> None:
        # thread-safe clear
        empty_snapshot = RuleSetSnapshot(...)
        with self._lock:
            self._snapshot = empty_snapshot

    def get_rule_ids(self) -> Tuple[str, ...]:
        return tuple(rule.rule_id for rule in self._snapshot.rules)

    def get_enabled_rule_count(self) -> int:
        return sum(1 for rule in self._snapshot.rules if rule.enabled)

# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    "RULE_ENGINE_VERSION",
    "RULESET_SCHEMA_VERSION",
    "RuleOperator",
    "LogicOperator",
    "RuleEngineError",
    "RuleCompilationError",
    "RuleSetValidationError",
    "RuleEvaluationTimeout",
    "ConditionNode",
    "LogicNode",
    "CompiledRule",
    "RuleMatchResult",
    "RuleSetSnapshot",
    "PropertyResolver",
    "SafeRegexCompiler",
    "RuleCompiler",
    "RuleEvaluator",
    "RuleEngine",
]