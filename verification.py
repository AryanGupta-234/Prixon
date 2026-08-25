"""Verification engine (spec section 18).

Never assume a tool succeeded just because the underlying call didn't raise.
This wraps the two things the current tool registry can actually do --
launching an executable, and running a read-only diagnostic -- with real
post-execution checks. It intentionally does NOT claim to verify things it
can't: os.startfile() for a URI is a fire-and-forget shell handoff with no
process handle to poll, so that path is reported as genuinely unverifiable
rather than silently marked "success". Lying about verification is worse
than admitting there isn't any.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class VerificationResult:
    verified: bool               # could this be checked at all?
    confirmed: Optional[bool]    # True/False if checked, None if unverifiable
    evidence: str

    def to_dict(self):
        return {"verified": self.verified, "confirmed": self.confirmed, "evidence": self.evidence}


def verify_process_launch(executable: str, timeout: float = 2.0) -> VerificationResult:
    """Polls for a process whose image name matches `executable` for up to
    `timeout` seconds. Requires psutil; degrades to "unverifiable" without
    it rather than pretending the launch was confirmed."""
    if not _HAS_PSUTIL:
        return VerificationResult(False, None, "psutil not installed -- launch not independently verified")

    target = os.path.basename(executable).lower()
    deadline = time.time() + timeout
    while time.time() < deadline:
        for proc in psutil.process_iter(["name"]):
            try:
                if (proc.info.get("name") or "").lower() == target:
                    return VerificationResult(True, True, f"found running process '{target}'")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(0.2)
    return VerificationResult(True, False, f"no process named '{target}' found within {timeout}s")


def verify_uri_open() -> VerificationResult:
    """os.startfile() hands off to the shell and returns immediately with no
    handle to poll -- there is nothing to verify against, so this is
    reported as unverifiable rather than assumed successful."""
    return VerificationResult(False, None, "URI handoff has no process handle to verify against")


def verify_diagnostic(ok: bool, data) -> VerificationResult:
    """Diagnostics are synchronous PowerShell calls -- return code plus
    non-empty output already IS the verification, unlike the fire-and-forget
    launch/URI paths above."""
    confirmed = bool(ok and data)
    evidence = "diagnostic returned data" if confirmed else "diagnostic returned no usable data"
    return VerificationResult(True, confirmed, evidence)
