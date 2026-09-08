"""Render a concise Markdown report from benchmark_results.json."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def render(payload: dict) -> str:
    rows = payload["results"]
    categories = Counter(row["category"] for row in rows)
    passed = Counter(row["category"] for row in rows if row["overall_pass"])
    lines = ["# Prixon benchmark report", "", f"Cases: {payload['case_count']}",
             f"Evaluated: {payload['evaluated_count']}", f"Passed: {payload['passed_count']}",
             "", "| Category | Cases | Passed |", "| --- | ---: | ---: |"]
    lines += [f"| {name} | {count} | {passed[name]} |" for name, count in sorted(categories.items())]
    failures = [row for row in rows if not row["overall_pass"]]
    if failures:
        lines += ["", "## Unmet cases", "", "| ID | Request | Expected target | Actual target |", "| --- | --- | --- | --- |"]
        lines += [f"| {row['id']} | {row['request']} | {row['expected_target']} | {row['actual_target']} |" for row in failures[:25]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("benchmark_results.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmark_report.md"))
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.results.read_text(encoding="utf-8"))), encoding="utf-8")


if __name__ == "__main__":
    main()
