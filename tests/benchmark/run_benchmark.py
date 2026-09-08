"""Implementation-independent Prixon benchmark runner.

Predictions are supplied by an adapter as JSONL.  This keeps benchmark
expectations out of the runtime and makes the same cases usable by Prixon,
the base model, and a fine-tuned model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_CASE_FIELDS = {"id", "category", "request", "expected_target", "expected_intent"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        missing = REQUIRED_CASE_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{path}:{number}: missing {sorted(missing)}")
        rows.append(row)
    return rows


def _matches(expected: Any, actual: Any) -> bool | None:
    return None if expected is None else expected == actual


def _parameter_accuracy(expected: dict[str, Any], actual: Any) -> bool:
    if not expected:
        return True
    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def score_case(case: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    prediction = prediction or {}
    expected_parameters = case.get("expected_parameters", {})
    answer = prediction.get("final_answer", "")
    target_ok = _matches(case["expected_target"], prediction.get("actual_target"))
    intent_ok = _matches(case["expected_intent"], prediction.get("actual_intent"))
    tool_ok = _matches(case.get("expected_tool_success"), prediction.get("tool_success"))
    verification_ok = _matches(case.get("expected_verification_success"), prediction.get("verification_success"))
    parameter_ok = _parameter_accuracy(expected_parameters, prediction.get("parameters"))
    answer_ok = isinstance(answer, str) and bool(answer.strip())
    checks = [x for x in (target_ok, intent_ok, tool_ok, verification_ok) if x is not None]
    overall = bool(prediction) and all(checks) and parameter_ok and answer_ok
    return {
        "id": case["id"], "category": case["category"], "request": case["request"],
        "expected_target": case["expected_target"], "actual_target": prediction.get("actual_target"),
        "expected_intent": case["expected_intent"], "actual_intent": prediction.get("actual_intent"),
        "parameter_accuracy": parameter_ok, "tool_success": prediction.get("tool_success"),
        "verification_success": prediction.get("verification_success"),
        "final_answer_valid": answer_ok, "overall_pass": overall,
    }


def load_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    return {row["id"]: row for row in rows}


def run(cases_path: Path, predictions_path: Path | None, output_path: Path) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    if len(cases) < 200:
        raise ValueError(f"benchmark requires at least 200 cases, found {len(cases)}")
    predictions = load_predictions(predictions_path)
    results = [score_case(case, predictions.get(case["id"])) for case in cases]
    evaluated = sum(1 for case in results if case["id"] in predictions)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases), "evaluated_count": evaluated,
        "passed_count": sum(row["overall_pass"] for row in results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmark_results.json"))
    args = parser.parse_args()
    result = run(args.cases, args.predictions, args.output)
    print(f"wrote {args.output}: {result['evaluated_count']}/{result['case_count']} cases evaluated")


if __name__ == "__main__":
    main()
