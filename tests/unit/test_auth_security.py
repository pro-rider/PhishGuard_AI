"""
PhishGuard_AI - Unit tests for the secure-by-default auth posture.

Covers:
- Open mode is FAIL-CLOSED by default (rejected unless DEV_INSECURE=True).
- API keys default to the least-privilege `viewer` role.
- The login throttle locks an account/IP after repeated failures and resets
  on success.

Run from the project root:
    python tests/unit/test_auth_security.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Allow the app module to import without a real secret (we test the guard
# directly with explicit `dev_insecure` flags).
os.environ.setdefault("DEV_INSECURE", "true")
os.environ.pop("AUTH_USERS", None)
os.environ.pop("API_KEYS", None)
os.environ.pop("SECRET_KEY", None)

from api_gateway.middleware.auth_guard import (  # noqa: E402
    AuthenticationError,
    AuthenticationGuard,
    _parse_key_entry,
)
from api_gateway.main import _LoginThrottle  # noqa: E402
from security.identity import ROLE_VIEWER, ROLE_ANALYST  # noqa: E402

PASS = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {extra}" if extra else ""))
    if not cond:
        PASS.append(name)


class _Req:
    def __init__(self, path="/scan", headers=None):
        from types import SimpleNamespace

        self.url = SimpleNamespace(path=path)
        self.headers = headers or {}
        self.state = type("S", (), {})()


def main() -> int:
    # 1. Open mode is fail-closed by default.
    guard_closed = AuthenticationGuard(dev_insecure=False)
    r = _Req("/scan")
    try:
        guard_closed.authenticate(r)
        check("open_mode_refused_default", False, "no AuthenticationError raised")
    except AuthenticationError:
        check("open_mode_refused_default", True)

    # 2. Open mode allowed only with DEV_INSECURE=True.
    guard_open = AuthenticationGuard(dev_insecure=True)
    ro = _Req("/scan")
    try:
        guard_open.authenticate(ro)
        check("open_mode_allowed_dev_insecure", getattr(ro.state, "authenticated", None) is False)
    except AuthenticationError:
        check("open_mode_allowed_dev_insecure", False, "raised")

    # 3. Exempt paths bypass auth regardless of mode.
    guard_closed.authenticate(_Req("/health"))
    check("exempt_path_ok", True)

    # 4. Bare API key defaults to viewer (least privilege).
    guard_keys = AuthenticationGuard(api_keys=["secretkey"], dev_insecure=False)
    rk = _Req("/cases", {"X-API-Key": "secretkey"})
    guard_keys.authenticate(rk)
    check(
        "apikey_defaults_viewer",
        "viewer" in rk.state.roles and "analyst" not in rk.state.roles,
        str(rk.state.roles),
    )

    # 5. _parse_key_entry: bare entry -> viewer.
    key, tenant, roles = _parse_key_entry("abc")
    check("parse_default_viewer", roles == frozenset({ROLE_VIEWER}), str(roles))
    key, tenant, roles = _parse_key_entry("k|analyst|acme")
    check(
        "parse_explicit_role_analyst",
        roles == frozenset({ROLE_ANALYST}) and tenant == "acme",
        f"{roles}@{tenant}",
    )

    # 6. Login throttle locks after 5 failures, resets on success.
    th = _LoginThrottle()
    for _ in range(5):
        th.record_failure("user", "1.2.3.4")
    check("throttle_locks_after_failures", th.is_locked("user", "1.2.3.4"))
    th.record_success("user", "1.2.3.4")
    check("throttle_resets_on_success", not th.is_locked("user", "1.2.3.4"))
    check("throttle_no_cross_leak", not th.is_locked("other", "1.2.3.4"))

    print(f"\n{'='*50}")
    if PASS:
        print(f"FAILED: {PASS}")
        return 1
    print("ALL AUTH-SECURITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
