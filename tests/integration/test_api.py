"""
PhishGuard_AI - API Integration Test Suite

Runs the public API end-to-end against a throwaway SQLite database (set via the
PHISHGUARD_DB env before import) using FastAPI's TestClient. Covers:

- DB-aware readiness probe
- JWT auth + RBAC (admin writes, viewer read-only)
- Sync ``/scan`` + audit hook
- Async ``/scan/async`` job lifecycle + SSE events
- Cases, quarantine, audit trail

Run from the project root:
    python tests/integration/test_api.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure the project root is importable regardless of the invocation directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Use a throwaway DB for every run so tests never touch real data.
os.environ["PHISHGUARD_DB"] = tempfile.mktemp(suffix=".db")
os.environ["AUTH_USERS"] = "admin:admin123:admin:system;viewer:view123:viewer:system"
os.environ["SECRET_KEY"] = "test-secret-key-at-least-thirty-two-characters-long"
os.environ.pop("API_KEYS", None)
os.environ.pop("SQLALCHEMY_DATABASE_URI", None)

from fastapi.testclient import TestClient  # noqa: E402
from api_gateway.main import app  # noqa: E402

PHISHING_TEXT = "URGENT verify your account now https://x.tk"
PASS = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {extra}" if extra else ""))
    if not cond:
        PASS.append(name)


def main() -> int:
    c = TestClient(app, raise_server_exceptions=False)

    # 1. Readiness probe actually checks the DB.
    r = c.get("/ready")
    check("ready.db_ok", r.status_code == 200 and r.json().get("db") == "ok", str(r.json()))

    # 2. Auth: admin token + RBAC.
    tr = c.post("/auth/token", json={"username": "admin", "password": "admin123"})
    check("admin_login", tr.status_code == 200, tr.text[:80])
    admin_h = {"Authorization": f"Bearer {tr.json()['access_token']}"}

    vr = c.post("/auth/token", json={"username": "viewer", "password": "view123"})
    viewer_h = {"Authorization": f"Bearer {vr.json()['access_token']}"}

    # 3. Sync scan + audit hook + explainability signals.
    s = c.post("/scan", json={"email_text": PHISHING_TEXT}, headers=admin_h)
    check("sync_scan", s.status_code == 200 and "scan_id" in s.json(), str(s.json()))
    sig = s.json().get("signals", [])
    check(
        "scan_has_signals",
        isinstance(sig, list) and all(
            {"analyzer", "score", "confidence", "findings"} <= set(x) for x in sig
        ),
        str(sig)[:200],
    )

    # 4. Async job lifecycle.
    a = c.post("/scan/async", json={"email_text": PHISHING_TEXT}, headers=admin_h)
    check("async_submit", a.status_code == 200 and a.json()["status"] == "pending", str(a.json()))
    job_id = a.json()["job_id"]
    job = None
    for _ in range(200):
        job = c.get(f"/scan/jobs/{job_id}", headers=admin_h)
        if job.json().get("status") in ("completed", "failed"):
            break
        time.sleep(0.05)
    check("job_completed", job.json().get("status") == "completed", str(job.json().get("status")))
    check("job_has_result", bool(job.json().get("result", {}).get("scan_id")), str(job.json()))

    # 5. SSE events stream.
    with c.stream("GET", f"/scan/jobs/{job_id}/events", headers=admin_h) as ev:
        text = "".join(ev.iter_text())
    check("sse_result_event", "event: result" in text)
    check("sse_done_event", "event: done" in text)

    # 6. Cases read for viewer is allowed; write is denied (RBAC).
    check("viewer_read_cases", c.get("/cases", headers=viewer_h).status_code == 200)
    denied = c.post("/cases", json={"title": "denied"}, headers=viewer_h)
    check("viewer_write_denied_403", denied.status_code == 403, str(denied.json()))

    # 6a. Case workflow: create -> update -> transition -> comments -> evidence -> timeline.
    cc = c.post(
        "/cases",
        json={"title": "Phishing campaign A", "severity": "high"},
        headers=admin_h,
    )
    check("case_create", cc.status_code == 200 and cc.json()["status"] == "open", str(cc.json()))
    case_id = cc.json()["case_id"]
    check("case_get_own", c.get(f"/cases/{case_id}", headers=admin_h).status_code == 200)
    check(
        "case_get_viewer_ok",
        c.get(f"/cases/{case_id}", headers=viewer_h).status_code == 200,
    )
    check(
        "case_viewer_transition_denied",
        c.post(
            f"/cases/{case_id}/transition",
            json={"new_status": "in_progress"},
            headers=viewer_h,
        ).status_code == 403,
    )
    upd = c.patch(f"/cases/{case_id}", json={"assignee": "soc-1"}, headers=admin_h)
    check("case_update", upd.status_code == 200 and upd.json()["assignee"] == "soc-1", str(upd.json()))
    tr = c.post(
        f"/cases/{case_id}/transition",
        json={"new_status": "in_progress", "comment": "assigning to SOC"},
        headers=admin_h,
    )
    check("case_transition_ok", tr.status_code == 200 and tr.json()["status"] == "in_progress", str(tr.json()))
    bad = c.post(
        f"/cases/{case_id}/transition",
        json={"new_status": "archived"},
        headers=admin_h,
    )
    check("case_transition_illegal_409", bad.status_code == 409, str(bad.json()))
    cmt = c.post(
        f"/cases/{case_id}/comments",
        json={"body": "verified via header analysis", "visibility": "internal"},
        headers=admin_h,
    )
    check("case_comment_add", cmt.status_code == 200, str(cmt.json()))
    check(
        "case_comments_list",
        len(c.get(f"/cases/{case_id}/comments", headers=admin_h).json()) >= 2,
    )
    ev = c.post(
        f"/cases/{case_id}/evidence",
        json={
            "kind": "ioc",
            "source": "url_parser",
            "storage_key": "ioc:https://x.tk",
            "metadata": {"score": "44"},
        },
        headers=admin_h,
    )
    check("case_evidence_attach", ev.status_code == 200, str(ev.json()))
    check(
        "case_evidence_list",
        len(c.get(f"/cases/{case_id}/evidence", headers=admin_h).json()) >= 1,
    )
    tl = c.get(f"/cases/{case_id}/timeline", headers=admin_h).json()
    kinds = {evt["type"] for evt in tl}
    check(
        "case_timeline_mixed",
        {"comment", "evidence", "audit"} <= kinds,
        str(kinds),
    )
    check(
        "case_cross_tenant_hidden",
        c.get("/cases/not-a-real-id", headers=admin_h).status_code == 404,
    )

    # 6b. Quarantine workflow: create -> get -> preview -> release -> delete.
    q = c.post(
        "/quarantine",
        json={"message_id": "msg-1", "body": "FAKE_INVOICE_PAYMENT", "risk_score": 0.9},
        headers=admin_h,
    )
    check("quarantine_create", q.status_code == 200, str(q.json()))
    qid = q.json()["quarantine_id"]
    check("quarantine_get_viewer_ok", c.get(f"/quarantine/{qid}", headers=viewer_h).status_code == 200)
    check(
        "quarantine_viewer_release_denied",
        c.post(f"/quarantine/{qid}/release", json={}, headers=viewer_h).status_code == 403,
    )
    prev = c.get(f"/quarantine/{qid}/preview", headers=admin_h)
    check(
        "quarantine_preview_body",
        prev.status_code == 200 and "FAKE_INVOICE_PAYMENT" in prev.json()["body"],
        prev.text[:120],
    )
    rel = c.post(f"/quarantine/{qid}/release", json={"reason": "fp"}, headers=admin_h)
    check("quarantine_release", rel.status_code == 200 and rel.json()["status"] == "released", str(rel.json()))
    rel2 = c.post(f"/quarantine/{qid}/release", json={}, headers=admin_h)
    check("quarantine_release_again_409", rel2.status_code == 409, str(rel2.json()))
    dl = c.post(f"/quarantine/{qid}/delete", json={}, headers=admin_h)
    check("quarantine_delete", dl.status_code == 200 and dl.json()["status"] == "deleted", str(dl.json()))

    # 7. Audit trail populated with auth + scan actions.
    aud = c.get("/audit", headers=admin_h)
    actions = [x["action"] for x in aud.json()]
    check("audit_present", aud.status_code == 200 and len(actions) >= 3, str(actions))

    # 8. Prometheus /metrics renders counters in text format.
    m = c.get("/metrics")
    mtext = m.text
    check("metrics_200", m.status_code == 200, str(m.status_code))
    check(
        "metrics_text_format",
        m.headers.get("content-type", "").startswith("text/plain"),
        m.headers.get("content-type", ""),
    )
    check(
        "metrics_has_scan_total",
        "phishguard_scan_total" in mtext,
        [ln for ln in mtext.splitlines() if "scan_total" in ln][:2],
    )

    print(f"\n{'='*50}")
    if PASS:
        print(f"FAILED: {PASS}")
        return 1
    print("ALL INTEGRATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
