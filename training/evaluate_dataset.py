#!/usr/bin/env python3
"""Validate Prixon's generated JSONL corpus before training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


def rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield row


def validate(path: Path) -> Dict[str, Any]:
    count = 0
    roles = Counter()
    tracks = Counter()
    sources = Counter()
    failures = []

    for row in rows(path):
        count += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            failures.append(f"example {count}: missing messages")
            continue
        for message in messages:
            if isinstance(message, dict):
                roles[message.get("role", "unknown")] += 1
        metadata = row.get("metadata") or {}
        tracks[str(metadata.get("track", "unknown"))] += 1
        sources[str(metadata.get("source", "unknown"))] += 1
        if not any(isinstance(m, dict) and m.get("role") == "user" and str(m.get("content", "")).strip() for m in messages):
            failures.append(f"example {count}: no non-empty user message")
        if not any(isinstance(m, dict) and m.get("role") == "assistant" and (str(m.get("content", "")).strip() or m.get("tool_calls")) for m in messages):
            failures.append(f"example {count}: no assistant response/tool call")

    return {
        "file": str(path),
        "examples": count,
        "roles": dict(roles),
        "tracks": dict(tracks),
        "sources": dict(sources),
        "failures": failures[:50],
        "valid": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    if not args.files:
        raise SystemExit("Provide one or more JSONL files")

    results = [validate(path) for path in args.files]
    print(json.dumps(results, indent=2))
    if any(not result["valid"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
