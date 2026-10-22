"""
Unit tests for security.pii_redactor.PIIRedactor.
"""

from __future__ import annotations

import pytest

from security.pii_redactor import PIIRedactor


@pytest.mark.unit
class TestPIIRedactor:
    def test_is_sensitive_key_true(self) -> None:
        for key in ("password", "secret_key", "token", "Authorization", "api_key"):
            assert PIIRedactor.is_sensitive_key(key) is True

    def test_is_sensitive_key_false(self) -> None:
        for key in ("display_name", "subject", "url", "message"):
            assert PIIRedactor.is_sensitive_key(key) is False

    def test_sanitize_url_redacts_userinfo(self) -> None:
        sanitized = PIIRedactor.sanitize_url_string(
            "https://alice:secret@example.com/path"
        )
        assert "alice" not in sanitized
        assert "secret" not in sanitized
        assert "example.com/path" in sanitized

    def test_sanitize_text_redacts_email(self) -> None:
        sanitized = PIIRedactor.sanitize_text_content(
            "Reach me at alice@example.com today."
        )
        assert "alice@example.com" not in sanitized

    def test_sanitize_text_redacts_phone(self) -> None:
        sanitized = PIIRedactor.sanitize_text_content("Call +1-555-0100 now")
        assert "555-0100" not in sanitized
