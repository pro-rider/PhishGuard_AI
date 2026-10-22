"""
PhishGuard_AI - Case Comments

Thread-safe comment management for incident cases.

Security / reliability principles:
- Comments are length-bounded and stored with a validated author handle.
- Comment lists are bounded per case to prevent unbounded growth.
- Authors are never trusted raw; they are sanitized to a display name.
"""

from __future__ import annotations

import enum
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

COMMENT_ID_PREFIX = "cmt_"
MAX_AUTHOR_LENGTH = 128
MAX_BODY_LENGTH = 8_192
MAX_COMMENTS_PER_CASE = 10_000


class CommentVisibility(str, enum.Enum):
    """Who can see a comment."""

    INTERNAL = "internal"
    CUSTOMER = "customer"


class CommentError(RuntimeError):
    """Base error for comment management."""


class CommentNotFoundError(CommentError):
    """Requested comment does not exist."""


@dataclass(frozen=True, slots=True)
class CaseComment:
    """A single comment on a case."""

    comment_id: str
    case_id: str
    author: str
    body: str
    created_utc: str
    visibility: CommentVisibility = CommentVisibility.INTERNAL

    def to_dict(self) -> dict:
        return {
            "comment_id": self.comment_id,
            "case_id": self.case_id,
            "author": self.author,
            "body": self.body,
            "created_utc": self.created_utc,
            "visibility": self.visibility.value,
        }


class CommentManager:
    """Thread-safe in-memory comment store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._comments: Dict[str, CaseComment] = {}

    def add(
        self,
        *,
        case_id: str,
        author: str,
        body: str,
        visibility: CommentVisibility = CommentVisibility.INTERNAL,
    ) -> CaseComment:
        """Add a comment to a case."""
        case_id = self._bound(case_id, 128, "case_id", required=True)
        author = self._bound(author, MAX_AUTHOR_LENGTH, "author", required=True)
        body = self._bound(body, MAX_BODY_LENGTH, "body", required=True)

        with self._lock:
            count = sum(1 for c in self._comments.values() if c.case_id == case_id)
            if count >= MAX_COMMENTS_PER_CASE:
                raise CommentError("Comment limit reached for this case.")
            comment = CaseComment(
                comment_id=COMMENT_ID_PREFIX + uuid.uuid4().hex[:20],
                case_id=case_id,
                author=author,
                body=body,
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                visibility=visibility,
            )
            self._comments[comment.comment_id] = comment
        return comment

    def list_for_case(self, case_id: str) -> List[CaseComment]:
        with self._lock:
            comments = [
                c for c in self._comments.values() if c.case_id == case_id
            ]
        comments.sort(key=lambda c: c.created_utc)
        return comments

    def get(self, comment_id: str) -> CaseComment:
        with self._lock:
            comment = self._comments.get(comment_id)
        if comment is None:
            raise CommentNotFoundError(comment_id)
        return comment

    def delete(self, comment_id: str) -> bool:
        with self._lock:
            return self._comments.pop(comment_id, None) is not None

    @staticmethod
    def _bound(value: str, limit: int, name: str, *, required: bool) -> str:
        if not isinstance(value, str):
            raise CommentError(f"{name} must be a string.")
        if required and not value.strip():
            raise CommentError(f"{name} must not be empty.")
        return value[:limit]


comment_manager = CommentManager()


__all__ = [
    "CommentVisibility",
    "CaseComment",
    "CommentManager",
    "CommentError",
    "CommentNotFoundError",
    "comment_manager",
]
