"""
PhishGuard_AI - Unit tests for the email header security analyzer.

Directly exercises HeaderAnalyzer against crafted RFC 5322 messages to verify it
emits real, high-value spoofing signals (Reply-To mismatch, alignment failures)
and degrades gracefully (INVALID_INPUT) when no headers are supplied.

Run from the project root:
    python tests/unit/test_header_analyzer.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("DEV_INSECURE", "true")
os.environ.pop("AUTH_USERS", None)
os.environ.pop("API_KEYS", None)

from engine.analyzers.email.header_analyzer import HeaderAnalyzer  # noqa: E402
from engine.orchestrator import (  # noqa: E402
    AnalyzerExecutionStatus,
    ConstrainedMetadata,
    ScanContext,
    ScanType,
)

PASS = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {extra}" if extra else ""))
    if not cond:
        PASS.append(name)


SPOOFED = (
    "Return-Path: <bounce@evil-attacker.com>\r\n"
    "Received: from mx.evil.com (mx.evil.com [203.0.113.9]) by mail.gateway.local\r\n"
    'From: "PayPal Security" <security@paypal.com>\r\n'
    'Reply-To: "PayPal Account Team" <account-verify@evil-attacker.com>\r\n'
    "Subject: Your account has been limited\r\n"
    "To: victim@example.com\r\n"
    "Date: Wed, 21 Oct 2026 10:00:00 +0000\r\n"
    "Message-ID: <abc123@evil-attacker.com>\r\n"
    "Content-Type: text/plain\r\n"
    "\r\n"
    "URGENT: click https://verify-paypal.evil-attacker.com to restore access.\r\n"
)

BENIGN = (
    "Return-Path: <bounce@example.com>\r\n"
    "Received: from mx.example.com (mx.example.com [198.51.100.4]) by mx.example.com\r\n"
    'From: "Example Support" <support@example.com>\r\n'
    "Reply-To: <help@example.com>\r\n"
    "Subject: Monthly statement\r\n"
    "To: user@example.com\r\n"
    "Date: Wed, 21 Oct 2026 10:00:00 +0000\r\n"
    "Message-ID: <stmt-1@example.com>\r\n"
    "Content-Type: text/plain\r\n"
    "\r\n"
    "Your statement for October is attached.\r\n"
)


def run(analyzer: HeaderAnalyzer, eml: str):
    ctx = ScanContext(
        scan_type=ScanType.EMAIL,
        metadata=ConstrainedMetadata({"raw_headers": eml}),
    )
    return asyncio.run(analyzer.analyze(ctx, time.monotonic() + 10.0))


def main() -> int:
    analyzer = HeaderAnalyzer()

    # 1. No headers -> graceful INVALID_INPUT (must not crash, must not be SUCCESS).
    empty = run(analyzer, "")
    check(
        "no_headers_graceful",
        empty.status == AnalyzerExecutionStatus.INVALID_INPUT,
        str(getattr(empty, "status", None)),
    )

    # 2. Spoofed email -> real HIGH Reply-To mismatch finding + score.
    spoofed = run(analyzer, SPOOFED)
    check("spoofed_success", spoofed.status == AnalyzerExecutionStatus.SUCCESS)
    rules = {(f.rule_id, f.severity.value) for f in (spoofed.findings or [])}
    check(
        "spoofed_replyto_mismatch",
        ("EMAIL.ALIGNMENT.REPLY_TO_MISMATCH", "HIGH") in rules,
        str(rules),
    )
    check("spoofed_score_positive", getattr(spoofed, "score_contribution", 0.0) > 0.0)

    # 3. Benign aligned email -> SUCCESS with no HIGH findings.
    benign = run(analyzer, BENIGN)
    check("benign_success", benign.status == AnalyzerExecutionStatus.SUCCESS)
    high = [f for f in (benign.findings or []) if f.severity.value == "HIGH"]
    check("benign_no_high", not high, str([f.rule_id for f in high]))

    print(f"\n{'='*50}")
    if PASS:
        print(f"FAILED: {PASS}")
        return 1
    print("ALL HEADER-ANALYZER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
