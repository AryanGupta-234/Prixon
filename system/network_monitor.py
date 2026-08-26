"""Network monitor (spec section 11).

Interface *role* classification is name-pattern-based -- psutil doesn't
expose "is this a VPN/virtual adapter" as a flag, and the spec is explicit
that a disconnected VirtualBox/TAP/Bluetooth-PAN adapter must never read as
"the internet is down." This errs toward under-claiming role ("unknown"
beats a wrong guess) since bias_candidates-style callers should treat
"unknown" conservatively too.

`online` is a real (short-timeout) reachability probe, not just "some
adapter is up" -- an adapter can be up with no actual route out. It's
opt-in per read() call (not fired on every lightweight status check)
because it's the one thing here that touches the network at all; callers
that only need adapter inventory can skip it.
"""
from __future__ import annotations

import re
import socket

from system.snapshot import NetworkAdapterState, NetworkState

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_VIRTUAL_PATTERNS = re.compile(
    r"virtualbox|vmware|vethernet|hyper-v|tap|tun|docker|wsl|npcap|loopback",
    re.I,
)
_BLUETOOTH_PATTERN = re.compile(r"bluetooth", re.I)
_LOOPBACK_PATTERN = re.compile(r"^lo$|loopback", re.I)


def _classify_role(name: str) -> str:
    if _LOOPBACK_PATTERN.search(name):
        return "loopback"
    if _VIRTUAL_PATTERNS.search(name):
        return "virtual"
    if _BLUETOOTH_PATTERN.search(name):
        return "virtual"  # Bluetooth PAN adapters behave like virtual ones for connectivity purposes
    if re.search(r"wi-?fi|wlan|ethernet|eth\d|en\d", name, re.I):
        return "physical"
    return "unknown"


def _probe_reachable(timeout: float = 1.5) -> bool:
    """A single short-timeout TCP connect attempt. Deliberately not an
    HTTP request (no interpretation of response needed) and deliberately
    not repeated/retried -- this exists to distinguish 'genuinely offline'
    from 'has some adapter up' for the router's offline->local-preferred
    rule (spec section 9), not to be a general connectivity diagnostic."""
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            return True
    except OSError:
        return False


def read(probe_internet: bool = True) -> NetworkState:
    adapters = []
    if _HAS_PSUTIL:
        try:
            stats = psutil.net_if_stats()
            for name, s in stats.items():
                adapters.append(NetworkAdapterState(name=name, is_up=s.isup, role=_classify_role(name)))
        except Exception:
            pass

    online = None
    if probe_internet:
        try:
            online = _probe_reachable()
        except Exception:
            online = None  # unknown, not "offline" -- see module docstring

    return NetworkState(adapters=adapters, online=online)
