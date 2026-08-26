"""Resource-aware local-vs-cloud policy (spec section 9).

Deliberately simple, rule-based thresholds for this slice -- spec section 55
("model routing should learn") explicitly defers empirical/measured routing
to a later phase once real model_usage data exists. This is the rules this
codebase can act on today, not a placeholder.

Kept as pure functions over a SystemSnapshot (no I/O, no network) so this is
trivially unit-testable without mocking psutil or touching real hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config
from system.snapshot import SystemSnapshot


@dataclass
class RoutingAdvice:
    avoid_local: bool
    prefer_local: bool
    reason: str


def advise(snapshot: SystemSnapshot) -> RoutingAdvice:
    """Returns advice, not a hard decision -- brain/router.py combines this
    with provider availability itself. Two independent signals:

    avoid_local: resources are tight enough that starting/continuing local
    inference would likely hurt (spec section 9's "Available RAM = 1.5 GB...
    the router should avoid starting an expensive local inference
    operation" example).

    prefer_local: there's a concrete reason to prefer local even when cloud
    would also work -- currently just "offline," per spec section 9's
    "If Network = offline, then LOCAL if available."
    """
    mem = snapshot.memory
    cpu = snapshot.cpu

    if mem.available_mb is not None and mem.available_mb < config.MIN_AVAILABLE_RAM_MB_FOR_LOCAL:
        return RoutingAdvice(
            avoid_local=True, prefer_local=False,
            reason=f"available RAM {mem.available_mb:.0f}MB below {config.MIN_AVAILABLE_RAM_MB_FOR_LOCAL}MB threshold",
        )

    if cpu.usage_percent is not None and cpu.usage_percent > config.MAX_CPU_PERCENT_FOR_LOCAL:
        return RoutingAdvice(
            avoid_local=True, prefer_local=False,
            reason=f"CPU usage {cpu.usage_percent:.0f}% above {config.MAX_CPU_PERCENT_FOR_LOCAL}% threshold",
        )

    if snapshot.network.online is False:
        return RoutingAdvice(
            avoid_local=False, prefer_local=True,
            reason="network appears offline -- cloud providers are unreachable anyway",
        )

    return RoutingAdvice(avoid_local=False, prefer_local=False, reason="resources nominal, no routing preference")
