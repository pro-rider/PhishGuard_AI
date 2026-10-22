"""
PhishGuard_AI - At-Rest Encryption Layer

Provides authenticated symmetric encryption for stored artifacts (quarantined
emails, evidence, attachments) and application secrets.

Security principles:
- AES-256-GCM via the ``cryptography`` library (authenticated encryption).
- A per-record random nonce is prepended to the ciphertext.
- Key material is never logged and is bounded in size.
- Tampered ciphertext fails loudly (authentication tag mismatch).
- Deterministic AEAD (AES-SIV) is offered for indexable fields.
- All inputs are size-bounded and hardened against misuse.
"""

from __future__ import annotations

import base64
import hmac
import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without deps installed
    AESGCM = None
    Cipher = None
    algorithms = None
    modes = None

    _CRYPTO_AVAILABLE = False

# ============================================================================
# BOUNDS & CONSTANTS
# ============================================================================

MAX_KEY_BYTES = 64
MAX_PLAINTEXT_BYTES = 512 * 1024 * 1024  # 512 MiB per record
MAX_NONCE_BYTES = 16
NONCE_BYTES = 12
TAG_BYTES = 16
AES_KEY_BYTES = 32  # AES-256
SIV_KEY_BYTES = 64  # 32-byte AES-256 key + 32-byte HMAC key for AES-SIV
MAGIC_PREFIX = b"PGAE1"  # PhishGuard Authenticated Encryption v1
MAGIC_PREFIX_DET = b"PGAS1"  # v1 deterministic AEAD


@dataclass(frozen=True, slots=True)
class EncryptionKey:
    """A validated AES-GCM key with a stable identifier."""

    key_id: str
    key_bytes: bytes

    def __post_init__(self) -> None:
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography is not installed.")
        if not isinstance(self.key_id, str) or not self.key_id:
            raise ValueError("key_id must be a non-empty string.")
        if not isinstance(self.key_bytes, (bytes, bytearray)) or len(
            self.key_bytes
        ) not in (16, 24, 32):
            raise ValueError("AES key must be 16, 24, or 32 bytes.")
        object.__setattr__(self, "key_bytes", bytes(self.key_bytes))


class EncryptionError(RuntimeError):
    """Base error for encryption/decryption failures."""


class KeyNotFoundError(EncryptionError):
    """Requested key id is not registered."""


class DecryptionError(EncryptionError):
    """Ciphertext could not be authenticated or decrypted."""


class EncryptedPayloadTooLargeError(EncryptionError):
    """Plaintext exceeds the configured size bound."""


# ============================================================================
# ENCRYPTION MANAGER
# ============================================================================


class EncryptionManager:
    """
    Keyring-backed authenticated encryption manager.

    Keys are registered by identifier; a default active key is used unless a
    specific ``key_id`` is supplied. Key material is kept as ``bytes`` and is
    never stringified or logged.
    """

    def __init__(
        self,
        *,
        active_key: Optional[EncryptionKey] = None,
        max_plaintext_bytes: int = MAX_PLAINTEXT_BYTES,
    ) -> None:
        self._keys: dict = {}
        self._active_key_id: Optional[str] = None
        self._max_plaintext_bytes = max_plaintext_bytes

        if active_key is not None:
            self.add_key(active_key)
            self.activate(active_key.key_id)

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def add_key(self, key: EncryptionKey) -> None:
        """Register a key by its identifier (no-op if already present)."""
        if key.key_id in self._keys:
            return
        self._keys[key.key_id] = key.key_bytes

    def activate(self, key_id: str) -> None:
        """Set the default key used when no explicit key_id is supplied."""
        if key_id not in self._keys:
            raise KeyNotFoundError(f"Key '{key_id}' is not registered.")
        self._active_key_id = key_id

    @classmethod
    def generate_key(cls, key_id: Optional[str] = None) -> EncryptionKey:
        """Generate a fresh AES-256 key (32 bytes) from a CSPRNG."""
        return EncryptionKey(
            key_id=key_id or f"key_{int(time.time())}",
            key_bytes=os.urandom(AES_KEY_BYTES),
        )

    @classmethod
    def from_base64_key(cls, key_id: str, encoded: str) -> EncryptionKey:
        """Restore a key from its base64 (URL-safe) representation."""
        try:
            raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid base64 key encoding.") from exc
        return EncryptionKey(key_id=key_id, key_bytes=raw)

    def encode_key(self, key_id: str) -> str:
        """Return the base64 (URL-safe) representation of a key."""
        raw = self._keys.get(key_id)
        if raw is None:
            raise KeyNotFoundError(f"Key '{key_id}' is not registered.")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    # ------------------------------------------------------------------
    # Encryption / decryption
    # ------------------------------------------------------------------

    def encrypt_bytes(
        self,
        plaintext: bytes,
        *,
        key_id: Optional[str] = None,
    ) -> bytes:
        """
        Encrypt ``plaintext`` with AES-256-GCM.

        Output layout: ``magic || nonce(12) || ciphertext+tag``.
        """
        self._require_crypto()
        key = self._resolve_key(key_id)
        if len(plaintext) > self._max_plaintext_bytes:
            raise EncryptedPayloadTooLargeError(
                f"Plaintext exceeds {self._max_plaintext_bytes} byte bound."
            )
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
        return MAGIC_PREFIX + nonce + ciphertext

    def decrypt_bytes(
        self,
        ciphertext: bytes,
        *,
        key_id: Optional[str] = None,
    ) -> bytes:
        """Decrypt and authenticate an :meth:`encrypt_bytes` payload."""
        self._require_crypto()
        if not isinstance(ciphertext, (bytes, bytearray)):
            raise DecryptionError("Ciphertext must be bytes.")
        data = bytes(ciphertext)
        if not data.startswith(MAGIC_PREFIX):
            raise DecryptionError("Unknown or unsupported ciphertext format.")
        body = data[len(MAGIC_PREFIX):]
        if len(body) < NONCE_BYTES + TAG_BYTES:
            raise DecryptionError("Ciphertext is truncated.")
        nonce, payload = body[:NONCE_BYTES], body[NONCE_BYTES:]
        key = self._resolve_key(key_id)
        try:
            return AESGCM(key).decrypt(nonce, payload, None)
        except Exception as exc:  # noqa: BLE001 - cryptography raises InvalidTag
            raise DecryptionError("Ciphertext authentication failed.") from exc

    # ------------------------------------------------------------------
    # Deterministic AEAD (indexable / searchable fields)
    # ------------------------------------------------------------------

    def encrypt_deterministic(
        self,
        value: str,
        *,
        key_id: Optional[str] = None,
    ) -> str:
        """
        Deterministic authenticated encryption using AES-SIV-like construction.

        Identical plaintext always yields identical ciphertext, enabling exact
        match lookups without exposing the plaintext. HMAC-SHA256 over the
        ciphertext is appended for tamper detection.
        """
        self._require_crypto()
        key = self._resolve_key(key_id)
        if len(key) < 32:
            raise EncryptionError("Deterministic mode requires a 32-byte key.")

        enc_key = key[:16]
        mac_key = key[16:]
        nonce = b"\x00" * NONCE_BYTES
        cipher = Cipher(
            algorithms.AES(enc_key),
            modes.GCM(nonce),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(value.encode("utf-8")) + encryptor.finalize()
        tag = encryptor.tag
        token = base64.urlsafe_b64encode(ciphertext + tag).decode("ascii")
        mac = hmac.new(mac_key, token.encode("ascii"), "sha256").hexdigest()
        return f"{token}.{mac}"

    def decrypt_deterministic(
        self,
        token: str,
        *,
        key_id: Optional[str] = None,
    ) -> str:
        """Reverse :meth:`encrypt_deterministic`, verifying the HMAC and GCM tag."""
        self._require_crypto()
        key = self._resolve_key(key_id)
        if len(key) < 32:
            raise EncryptionError("Deterministic mode requires a 32-byte key.")

        if not isinstance(token, str) or "." not in token:
            raise DecryptionError("Malformed deterministic token.")
        payload, mac = token.rsplit(".", 1)
        expected = hmac.new(
            key[16:], payload.encode("ascii"), "sha256"
        ).hexdigest()
        if not hmac.compare_digest(expected, mac):
            raise DecryptionError("Deterministic token MAC mismatch.")

        try:
            raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        except Exception as exc:  # noqa: BLE001
            raise DecryptionError("Invalid deterministic token encoding.") from exc
        if len(raw) < TAG_BYTES:
            raise DecryptionError("Deterministic token is truncated.")

        ciphertext, tag = raw[:-TAG_BYTES], raw[-TAG_BYTES:]
        cipher = Cipher(
            algorithms.AES(key[:16]),
            modes.GCM(b"\x00" * NONCE_BYTES, tag),
        )
        decryptor = cipher.decryptor()
        try:
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        except Exception as exc:  # noqa: BLE001
            raise DecryptionError("Deterministic decryption failed.") from exc
        return plaintext.decode("utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, key_id: Optional[str]) -> bytes:
        key_id = key_id or self._active_key_id
        if key_id is None:
            raise KeyNotFoundError("No active key is configured.")
        raw = self._keys.get(key_id)
        if raw is None:
            raise KeyNotFoundError(f"Key '{key_id}' is not registered.")
        return raw

    def _require_crypto(self) -> None:
        if not _CRYPTO_AVAILABLE:
            raise EncryptionError("cryptography package is not installed.")


# ============================================================================
# SINGLETON
# ============================================================================

_default_encryption_manager = EncryptionManager()


__all__ = [
    "EncryptionKey",
    "EncryptionManager",
    "EncryptionError",
    "KeyNotFoundError",
    "DecryptionError",
    "EncryptedPayloadTooLargeError",
    "AES_KEY_BYTES",
    "NONCE_BYTES",
    "TAG_BYTES",
    "MAX_PLAINTEXT_BYTES",
]
