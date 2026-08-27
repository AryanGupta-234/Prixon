"""Process monitor (spec sections 10-11, 15).

Deliberately narrow: only the top-N heaviest processes by CPU and by
memory, not a full process table -- spec section 11 explicitly says "do
not continuously store every process snapshot," and spec section 51 puts
"full process analysis" in the "on demand + periodic lightweight sampling"
bucket, not "poll everything every cycle."

This exists specifically so anomaly_detector.py can name a top_process in
its events (spec section 15's example event has "top_process": "python.exe",
"process_cpu": 79) -- before this module, an alert could say CPU was high
but never *what* was using it, which is the whole point of spec section
18's alert examples ("Python is currently using most of it").

psutil.process_iter with cpu_percent requires a prior baseline sample per
process to be meaningful (the first call after a process starts always
reads ~0) -- see the module-level _cpu_percent_primed flag below.
"""
from __future__ import annotations

from typing import List

from system.snapshot import ProcessInfo, ProcessesState

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_TOP_N = 5
_cpu_percent_primed = False  # see module docstring


def read() -> ProcessesState:
    global _cpu_percent_primed
    if not _HAS_PSUTIL:
        return ProcessesState()

    try:
        procs: List[ProcessInfo] = []
        for p in psutil.process_iter(["pid", "name"]):
            try:
                # cpu_percent(None) is non-blocking and returns the usage
                # since psutil last checked this PID -- 0.0 on a process's
                # first-ever read, which is why the whole snapshot for this
                # poll is deliberately not trusted until _cpu_percent_primed
                # (see below), rather than reporting misleadingly-low
                # numbers on process_monitor's very first call.
                cpu = p.cpu_percent(None)
                mem_mb = p.memory_info().rss / (1024 * 1024)
                procs.append(ProcessInfo(pid=p.pid, name=p.info.get("name") or "unknown",
                                          cpu_percent=round(cpu, 1), memory_mb=round(mem_mb, 1)))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue  # process exited or is unreadable mid-iteration -- not an error, just skip it

        if not _cpu_percent_primed:
            # First-ever call: every cpu_percent above is meaningless (no
            # prior sample to diff against). Prime it and return an empty
            # result rather than a snapshot full of false zeros -- the next
            # poll (SystemAgent's regular interval) will have real numbers.
            _cpu_percent_primed = True
            return ProcessesState()

        top_by_cpu = sorted(procs, key=lambda x: x.cpu_percent, reverse=True)[:_TOP_N]
        top_by_memory = sorted(procs, key=lambda x: x.memory_mb, reverse=True)[:_TOP_N]
        return ProcessesState(top_by_cpu=top_by_cpu, top_by_memory=top_by_memory)
    except Exception:
        return ProcessesState()
