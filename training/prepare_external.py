#!/usr/bin/env python3
"""Convert a generic Hugging Face chat/tool dataset into Prixon's JSONL contract.

Nothing about a particular dataset, tool name, path, model, or provider is
embedded here. Dataset identifiers, subsets, limits, and output paths are
runtime configuration/CLI inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    value = parse_jsonish(value)
    if isinstance(value, dict):
        value = value.get("messages") or value.get("conversation") or value.get("input")
    if not isinstance(value, list):
        return []

    aliases = {
        "human": "user", "user": "user", "gpt": "assistant",
        "assistant": "assistant", "model": "assistant",
        "system": "system", "tool": "tool", "function": "tool",
    }
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = aliases.get(str(item.get("role") or item.get("from") or item.get("speaker") or "").lower())
        if not role:
            continue
        message: dict[str, Any] = {"role": role, "content": item.get("content", item.get("value", item.get("text", "")))}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in item:
                message[key] = parse_jsonish(item[key])
        output.append(message)
    return output


def normalize_tools(value: Any) -> list[dict[str, Any]] | None:
    value = parse_jsonish(value)
    if not isinstance(value, list):
        return None
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            output.append(item)
            continue
        name = item.get("name")
        if not name:
            continue
        parameters = item.get("parameters") or {"type": "object", "properties": {}}
        output.append({
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(item.get("description") or ""),
                "parameters": parameters,
            },
        })
    return output or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--tools-field", default="available_tools")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install training/requirements.txt first") from exc

    kwargs: dict[str, Any] = {"path": args.dataset, "split": args.split, "streaming": args.streaming}
    if args.subset:
        kwargs["name"] = args.subset

    rows = load_dataset(**kwargs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if args.max_rows is not None and written >= args.max_rows:
                break
            messages = normalize_messages(row.get(args.messages_field))
            if not messages or not any(m["role"] == "assistant" for m in messages):
                skipped += 1
                continue
            record = {"messages": messages}
            tools = normalize_tools(row.get(args.tools_field))
            if tools:
                record["tools"] = tools
            record["metadata"] = {"source_dataset": args.dataset, "source_split": args.split}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(json.dumps({"written": written, "skipped": skipped, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
