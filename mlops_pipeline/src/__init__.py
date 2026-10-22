"""
PhishGuard_AI - MLOps Source Package
====================================

Machine-learning pipeline for phishing model training, evaluation, and export.
"""

from __future__ import annotations

__version__ = "14.0.0"
__package_name__ = "phishguard-mlops"


def get_package_version() -> str:
    """Return the pipeline package version string."""
    return __version__


__all__ = ["get_package_version", "__version__"]
