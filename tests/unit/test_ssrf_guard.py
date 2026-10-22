"""
Unit tests for security.ssrf_guard.SSRFGuard.
"""

from __future__ import annotations

import pytest

from security.ssrf_guard import SSRFGuard, ssrf_guard


@pytest.mark.unit
class TestSSRFGuard:
    def test_public_singleton_is_guard_instance(self) -> None:
        assert isinstance(ssrf_guard, SSRFGuard)

    def test_allows_global_ip_literal(self) -> None:
        # DNS-independent: a global, non-private IP literal must be allowed.
        result = ssrf_guard.validate("https://1.1.1.1/path?q=1")
        assert result.allowed is True
        assert result.scheme == "https"
        assert result.host == "1.1.1.1"

    def test_blocks_loopback(self) -> None:
        result = ssrf_guard.validate("http://127.0.0.1:8080/admin")
        assert result.allowed is False
        assert result.reason

    def test_blocks_private_range(self) -> None:
        result = ssrf_guard.validate("http://10.0.0.5/")
        assert result.allowed is False

    def test_blocks_ipv6_loopback(self) -> None:
        result = ssrf_guard.validate("http://[::1]/")
        assert result.allowed is False

    def test_rejects_unparseable_url(self) -> None:
        result = ssrf_guard.validate("not-a-url")
        assert result.allowed is False
