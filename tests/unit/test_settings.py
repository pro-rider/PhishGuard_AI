"""
Unit tests for config.internal.settings.Settings.
"""

from __future__ import annotations

import pytest

from config.internal.settings import (
    EnvironmentOption,
    LogLevel,
    Settings,
)


@pytest.mark.unit
class TestSettings:
    def test_default_environment_is_development(self) -> None:
        assert Settings.ENVIRONMENT is EnvironmentOption
        assert EnvironmentOption.DEVELOPMENT.value == "development"

    def test_enums_expose_expected_members(self) -> None:
        assert hasattr(LogLevel, "INFO")
        assert hasattr(LogLevel, "DEBUG")
        assert hasattr(LogLevel, "ERROR")
        assert hasattr(EnvironmentOption, "PRODUCTION")
        assert hasattr(EnvironmentOption, "DEVELOPMENT")

    def test_settings_declares_security_fields(self) -> None:
        annotations = Settings.model_fields
        for field in ("SECRET_KEY", "JWT_ALGORITHM", "APP_NAME", "APP_VERSION"):
            assert field in annotations

    def test_settings_constructs_with_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("SECRET_KEY", "override-secret-123")
        settings = Settings(_env_file=None)
        assert settings.ENVIRONMENT.value == "production"
        assert settings.SECRET_KEY.get_secret_value() == "override-secret-123"
