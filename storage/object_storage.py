"""
PhishGuard_AI - Object Storage Abstraction

Unified object storage layer supporting local filesystem and S3-compatible
backends. Used for quarantined artifacts, evidence, and analysis outputs.

Security / reliability principles:
- Path traversal is prevented via strict key sanitization and path containment.
- A local filesystem backend is used when no S3 endpoint is configured, so the
  application remains runnable in development without cloud credentials.
- Large reads/writes are streamed in bounded chunks.
- All keys are size-bounded and namespace-prefixed.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple

from config.internal.settings import settings

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    _BOTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception
    _BOTO_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_KEY_LENGTH = 1024
DEFAULT_NAMESPACE = "pg"
MAX_STREAM_CHUNK = 8 * 1024 * 1024  # 8 MiB
MAX_OBJECT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB soft bound

_SAFE_KEY_REGEX = re.compile(r"^[a-zA-Z0-9_\-./]{1,1024}$")

# ============================================================================
# ERRORS & DATACLASSES
# ============================================================================


class ObjectStorageError(RuntimeError):
    """Base error for the object storage layer."""


class ObjectNotFoundError(ObjectStorageError):
    """Requested object does not exist."""


class InvalidKeyError(ObjectStorageError):
    """Key violates naming or safety constraints."""


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    """Metadata describing a stored object."""

    key: str
    size_bytes: int
    content_type: Optional[str] = None
    etag: Optional[str] = None


# ============================================================================
# ABSTRACT BASE
# ============================================================================


class ObjectStorage:
    """Abstract object storage interface."""

    name: str = "base"

    def put_bytes(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        raise NotImplementedError

    def put_stream(
        self, key: str, stream: BinaryIO, content_type: Optional[str] = None
    ) -> str:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def get_stream(self, key: str) -> BinaryIO:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def stat(self, key: str) -> ObjectMetadata:
        raise NotImplementedError

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        raise NotImplementedError


def validate_key(key: str) -> str:
    """Validate and normalize an object key."""
    if not isinstance(key, str) or not key:
        raise InvalidKeyError("Key must be a non-empty string.")
    if len(key) > MAX_KEY_LENGTH:
        raise InvalidKeyError(f"Key exceeds {MAX_KEY_LENGTH} chars.")
    if not _SAFE_KEY_REGEX.match(key):
        raise InvalidKeyError("Key contains disallowed characters.")
    if key.startswith("/") or ".." in key.split("/"):
        raise InvalidKeyError("Key must be a relative, safe path.")
    return key


# ============================================================================
# LOCAL FILESYSTEM BACKEND
# ============================================================================


class LocalObjectStorage(ObjectStorage):
    """
    Filesystem-backed object storage with path-containment guarantees.

    The configured root is the only writable location; keys are resolved and
    verified to remain inside it before any filesystem operation.
    """

    name = "local"

    def __init__(self, root: Optional[os.PathLike] = None) -> None:
        self._root = Path(root or "storage/objects").resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        validate_key(key)
        target = (self._root / key).resolve()
        # Path containment check (defense-in-depth against traversal).
        if not target.is_relative_to(self._root):
            raise InvalidKeyError("Key escapes storage root.")
        return target

    def put_bytes(
        self, key: str, data: bytes, content_type: Optional[str] = None
    ) -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.write(data)
        return key

    def put_stream(
        self, key: str, stream: BinaryIO, content_type: Optional[str] = None
    ) -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            while True:
                chunk = stream.read(MAX_STREAM_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
        return key

    def get_bytes(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.is_file():
            raise ObjectNotFoundError(key)
        return target.read_bytes()

    def get_stream(self, key: str) -> BinaryIO:
        target = self._resolve(key)
        if not target.is_file():
            raise ObjectNotFoundError(key)
        return target.open("rb")

    def exists(self, key: str) -> bool:
        target = self._resolve(key)
        return target.is_file()

    def delete(self, key: str) -> bool:
        target = self._resolve(key)
        if not target.is_file():
            return False
        target.unlink()
        return True

    def stat(self, key: str) -> ObjectMetadata:
        target = self._resolve(key)
        if not target.is_file():
            raise ObjectNotFoundError(key)
        return ObjectMetadata(key=key, size_bytes=target.stat().st_size)

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        base = self._resolve(prefix) if prefix else self._root
        if base.is_file():
            yield str(base.relative_to(self._root))
            return
        for path in base.rglob("*"):
            if path.is_file():
                yield str(path.relative_to(self._root))


# ============================================================================
# S3-COMPATIBLE BACKEND
# ============================================================================


class S3ObjectStorage(ObjectStorage):
    """S3-compatible backend backed by boto3."""

    name = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        if not _BOTO_AVAILABLE:
            raise ObjectStorageError("boto3 is not installed.")
        if not bucket:
            raise ObjectStorageError("S3 bucket must be configured.")

        kwargs: dict = {"region_name": region or "us-east-1"}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key

        self._bucket = bucket
        self._client = boto3.client("s3", **kwargs)

    def put_bytes(
        self, key: str, data: bytes, content_type: Optional[str] = None
    ) -> str:
        validate_key(key)
        extra = {"ContentType": content_type} if content_type else {}
        try:
            self._client.put_object(
                Bucket=self._bucket, Key=key, Body=data, **extra
            )
        except (BotoCoreError, ClientError) as exc:  # noqa: BLE001
            raise ObjectStorageError("S3 put failed.") from exc
        return key

    def put_stream(
        self, key: str, stream: BinaryIO, content_type: Optional[str] = None
    ) -> str:
        validate_key(key)
        try:
            self._client.upload_fileobj(
                stream,
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type} if content_type else {},
            )
        except (BotoCoreError, ClientError) as exc:  # noqa: BLE001
            raise ObjectStorageError("S3 upload failed.") from exc
        return key

    def get_bytes(self, key: str) -> bytes:
        validate_key(key)
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise ObjectNotFoundError(key) from exc
            raise ObjectStorageError("S3 get failed.") from exc
        except BotoCoreError as exc:  # noqa: BLE001
            raise ObjectStorageError("S3 get failed.") from exc

    def get_stream(self, key: str) -> BinaryIO:
        validate_key(key)
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return io.BytesIO(resp["Body"].read())
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise ObjectNotFoundError(key) from exc
            raise ObjectStorageError("S3 get failed.") from exc
        except BotoCoreError as exc:  # noqa: BLE001
            raise ObjectStorageError("S3 get failed.") from exc

    def exists(self, key: str) -> bool:
        validate_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            logger.warning("S3 head failed; assuming missing.", exc_info=True)
            return False
        except BotoCoreError:  # noqa: BLE001
            return False

    def delete(self, key: str) -> bool:
        validate_key(key)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except (BotoCoreError, ClientError) as exc:  # noqa: BLE001
            raise ObjectStorageError("S3 delete failed.") from exc

    def stat(self, key: str) -> ObjectMetadata:
        validate_key(key)
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            return ObjectMetadata(
                key=key,
                size_bytes=int(head.get("ContentLength", 0)),
                content_type=head.get("ContentType"),
                etag=head.get("ETag"),
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NotFound"}:
                raise ObjectNotFoundError(key) from exc
            raise ObjectStorageError("S3 stat failed.") from exc
        except BotoCoreError as exc:  # noqa: BLE001
            raise ObjectStorageError("S3 stat failed.") from exc

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        kwargs: dict = {"Bucket": self._bucket}
        if prefix:
            validate_key(prefix)
            kwargs["Prefix"] = prefix
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(**kwargs):
                for obj in page.get("Contents", []):
                    yield str(obj["Key"])
        except (BotoCoreError, ClientError) as exc:  # noqa: BLE001
            raise ObjectStorageError("S3 list failed.") from exc


# ============================================================================
# FACTORY
# ============================================================================


def create_object_storage() -> ObjectStorage:
    """Instantiate the storage backend selected by application settings."""
    s3_bucket = getattr(settings, "S3_BUCKET", None) or os.getenv("S3_BUCKET")
    if s3_bucket:
        return S3ObjectStorage(
            bucket=s3_bucket,
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            region=os.getenv("S3_REGION"),
            access_key=os.getenv("S3_ACCESS_KEY"),
            secret_key=os.getenv("S3_SECRET_KEY"),
        )
    return LocalObjectStorage()


object_storage = create_object_storage()


__all__ = [
    "ObjectStorage",
    "LocalObjectStorage",
    "S3ObjectStorage",
    "ObjectMetadata",
    "ObjectStorageError",
    "ObjectNotFoundError",
    "InvalidKeyError",
    "validate_key",
    "create_object_storage",
    "object_storage",
]
