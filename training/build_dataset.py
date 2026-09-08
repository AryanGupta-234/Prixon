#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def env_value(name: Optional[str], default: Any = None) -> Any:
    if name:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fingerprint(example: Dict[str, Any]) -> str:
    payload = {"messages": example.get("messages"), "tools": example.get("tools")}
    return hashlib.sha256(compact(payload).encode("utf-8")).hexdigest()


def load_python_attribute(module_path: str, attribute: str) -> Any:
    path = (ROOT / module_path).resolve()
    spec = importlib.util.spec_from_file_location("dataset_runtime_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


def system_prompt(cfg: Dict[str, Any]) -> str:
    spec = cfg["project"]["system_prompt"]
    source = spec["source"]
    if source == "python_attribute":
        return str(load_python_attribute(spec["module_path"], spec["attribute"]))
    if source == "file":
        return (ROOT / spec["path"]).read_text(encoding="utf-8")
    if source == "env":
        value = os.getenv(spec["env"])
        if not value:
            raise RuntimeError(f"Missing {spec['env']}")
        return value
    raise ValueError(f"Unsupported system prompt source: {source}")


def assistant_name(cfg: Dict[str, Any]) -> str:
    project = cfg["project"]
    return str(env_value(project.get("name_env"), project.get("name_default", "")))


def resolve_path(source: Dict[str, Any]) -> Path:
    raw = env_value(source.get("path_env"), source.get("path_default"))
    if not raw:
        raise RuntimeError(f"Missing path for {source['name']}")
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def hf_rows(source: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install training/requirements.txt first") from exc
    kwargs: Dict[str, Any] = {
        "path": source["dataset"],
        "split": source.get("split", "train"),
        "streaming": bool(source.get("streaming", True)),
    }
    if source.get("subset"):
        kwargs["name"] = source["subset"]
    if source.get("token_env") and os.getenv(source["token_env"]):
        kwargs["token"] = os.getenv(source["token_env"])
    return load_dataset(**kwargs)


def normalize_message(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    raw_role = value.get("role") or value.get("from") or value.get("speaker")
    aliases = {
        "human": "user", "user": "user", "gpt": "assistant",
        "assistant": "assistant", "model": "assistant",
        "system": "system", "tool": "tool", "function": "tool",
    }
    role = aliases.get(str(raw_role).lower()) if raw_role is not None else None
    if role is None:
        return None
    content = value.get("content")
    if content is None:
        content = value.get("value")
    if content is None:
        content = value.get("text")
    out: Dict[str, Any] = {"role": role, "content": "" if content is None else content}
    for key in ("tool_calls", "tool_call_id", "name"):
        if value.get(key) is not None:
            out[key] = parse_jsonish(value[key])
    return out


def normalize_messages(value: Any) -> List[Dict[str, Any]]:
    value = parse_jsonish(value)
    if isinstance(value, dict):
        value = value.get("messages") or value.get("input")
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        normalized = normalize_message(item)
        if normalized:
            out.append(normalized)
    return out


def normalize_tools(value: Any) -> Optional[List[Dict[str, Any]]]:
    value = parse_jsonish(value)
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            out.append(item)
            continue
        name = item.get("name")
        if not name:
            continue
        params = item.get("parameters") or {"type": "object", "properties": {}}
        if isinstance(params, dict) and params.get("type") == "dict":
            params = dict(params)
            params["type"] = "object"
        out.append({
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(item.get("description") or ""),
                "parameters": params,
            },
        })
    return out or None


def tool_calls(value: Any) -> List[Dict[str, Any]]:
    value = parse_jsonish(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for i, action in enumerate(value):
        if not isinstance(action, dict):
            continue
        name = action.get("name") or action.get("tool") or action.get("function")
        args = action.get("arguments") or action.get("args") or action.get("parameters") or {}
        args = parse_jsonish(args)
        if not isinstance(args, dict):
            args = {"value": args}
        if name:
            out.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": str(name), "arguments": compact(args)},
            })
    return out


def adapt_openai(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = normalize_messages(row.get("messages") or row.get("conversation"))
    if not messages:
        query = row.get("query") or row.get("prompt") or row.get("instruction")
        calls = tool_calls(row.get("answers") or row.get("answer") or row.get("tool_calls"))
        if query and calls:
            messages = [{"role": "user", "content": str(query)}, {"role": "assistant", "content": "", "tool_calls": calls}]
    if not messages:
        return None
    return {"messages": messages, "tools": normalize_tools(row.get("tools") or row.get("tools_json") or row.get("functions"))}


def adapt_hermes(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = normalize_messages(row.get("conversations") or row.get("messages"))
    if not messages:
        return None
    return {"messages": messages, "tools": normalize_tools(row.get("tools") or row.get("functions"))}


def adapt_nemotron(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    params = parse_jsonish(row.get("responses_create_params"))
    params = params if isinstance(params, dict) else {}
    messages = normalize_messages(params.get("input") or params.get("messages") or row.get("messages"))
    tools = normalize_tools(params.get("tools") or row.get("tools"))
    expected = row.get("expected_action") or row.get("expected_answer") or row.get("answer")
    if expected and not any(m.get("role") == "assistant" for m in messages):
        calls = tool_calls(expected)
        if calls:
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        elif isinstance(expected, str):
            messages.append({"role": "assistant", "content": expected})
    return {"messages": messages, "tools": tools} if messages else None


ADAPTERS = {
    "openai_tool_messages": adapt_openai,
    "hermes_conversations": adapt_hermes,
    "nemotron_responses": adapt_nemotron,
}


def catalog_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    examples: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        target = str(row.get("target") or "").strip()
        if not target:
            continue
        groups.setdefault(target, {
            "target": target,
            "target_name": row.get("target_name") or target,
            "intent": row.get("intent") or "unknown",
            "action": row.get("action") or "",
            "risk": row.get("risk") or "low",
        })
        instruction = str(row.get("instruction") or "").strip()
        if instruction and instruction not in examples[target]:
            examples[target].append(instruction)
    for target, group in groups.items():
        group["examples"] = examples[target]
    return groups


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def candidate(group: Dict[str, Any], score: float) -> Dict[str, Any]:
    return {
        "target": group["target"], "name": group["target_name"],
        "intent": group["intent"], "action": group["action"],
        "risk": group.get("risk", "low"), "retrieval_score": round(score, 4),
        "examples": list(group.get("examples") or [])[:4],
    }


def candidate_set(instruction: str, correct: str, groups: Dict[str, Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    q = tokens(instruction)
    ranked: List[Tuple[float, str]] = []
    for target, group in groups.items():
        corpus = " ".join([str(group["target_name"]), str(group["intent"]), " ".join(group.get("examples") or [])])
        t = tokens(corpus)
        score = len(q & t) / max(1, len(q | t))
        if target == correct:
            score += 1.0
        ranked.append((score, target))
    rng.shuffle(ranked)
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = ranked[:max(1, count)]
    if correct not in {target for _, target in selected}:
        selected[-1] = (1.0, correct)
    return [candidate(groups[target], score) for score, target in selected]


def runtime_user_prompt(name: str, request: str, allow_list: List[Dict[str, Any]], conversation: Optional[Dict[str, Any]] = None) -> str:
    context = {
        "assistant_name": name,
        "request": request,
        "conversation": conversation or {"recent_turns": [], "active_slots": {}},
        "retrieval": "shortlist",
        "allow_list": allow_list,
    }
    return (
        "Understand the user's request. Resolve references using recent conversation. "
        "Select at most ONE allow-listed target. Extract useful parameters, but never invent a capability.\n\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )


def nlu_answer(row: Dict[str, Any], aug: Dict[str, Any], matched: bool = True, reference: str = "none") -> str:
    return compact({
        "match_target": str(row.get("target") or "") if matched else None,
        "confidence": "high" if matched else "none",
        "intent": str(row.get("intent") or "unknown") if matched else "unknown",
        "parameters": {},
        "reference": reference,
        "reply": aug["success_reply"] if matched else aug["no_match_reply"],
        "reason": "Matched an allow-listed capability." if matched else "No supplied candidate matches safely.",
    })


def prixon_examples(rows: Sequence[Dict[str, Any]], source: Dict[str, Any], cfg: Dict[str, Any], prompt: str, rng: random.Random) -> Iterator[Dict[str, Any]]:
    groups = catalog_groups(rows)
    aug = cfg["augmentation"]["local_catalog"]
    count = int(cfg["build"]["candidate_count"])
    name = assistant_name(cfg)
    for row in rows:
        instruction = str(row.get("instruction") or "").strip()
        target = str(row.get("target") or "").strip()
        if not instruction or target not in groups:
            continue
        candidates = candidate_set(instruction, target, groups, count, rng)
        yield {
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": runtime_user_prompt(name, instruction, candidates)}, {"role": "assistant", "content": nlu_answer(row, aug)}],
            "metadata": {"source": source["name"], "track": "prixon_nlu", "target": target},
        }
        if aug.get("create_negative_examples"):
            wrong = [item for item in candidates if item["target"] != target]
            if wrong:
                yield {
                    "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": runtime_user_prompt(name, instruction, wrong)}, {"role": "assistant", "content": nlu_answer(row, aug, matched=False)}],
                    "metadata": {"source": source["name"], "track": "prixon_nlu_negative", "target": target},
                }
        if aug.get("create_context_examples"):
            request = str(aug["reference_request"])
            conversation = {
                "recent_turns": [{"user": instruction, "target": target, "target_name": groups[target]["target_name"], "intent": groups[target]["intent"], "parameters": {}, "reply": aug["success_reply"]}],
                "active_slots": {"last_target": target, "last_target_name": groups[target]["target_name"]},
            }
            yield {
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": runtime_user_prompt(name, request, candidates, conversation)}, {"role": "assistant", "content": nlu_answer(row, aug, reference="recent_target")}],
                "metadata": {"source": source["name"], "track": "prixon_context", "target": target},
            }


def source_examples(source: Dict[str, Any], cfg: Dict[str, Any], prompt: str, rng: random.Random) -> Iterator[Dict[str, Any]]:
    limit = source.get("max_examples")
    if limit is None:
        limit = cfg["build"].get("max_examples_per_source")
    if source["adapter"] == "prixon_nlu":
        iterator: Iterable[Dict[str, Any]] = prixon_examples(list(read_jsonl(resolve_path(source))), source, cfg, prompt, rng)
    else:
        rows = read_jsonl(resolve_path(source)) if source["kind"] == "local_jsonl" else hf_rows(source)
        adapter = ADAPTERS[source["adapter"]]
        def converted() -> Iterator[Dict[str, Any]]:
            for row in rows:
                example = adapter(dict(row))
                if example:
                    example["metadata"] = {"source": source["name"], "track": "general_agent"}
                    yield example
        iterator = converted()
    for idx, example in enumerate(iterator):
        if limit not in (None, "") and idx >= int(limit):
            break
        yield example


def valid(example: Dict[str, Any], quality: Dict[str, Any]) -> bool:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    roles = {m.get("role") for m in messages if isinstance(m, dict)}
    if any(role not in roles for role in quality.get("required_roles", [])):
        return False
    user_chars = sum(len(str(m.get("content") or "")) for m in messages if m.get("role") == "user")
    assistant_chars = sum(len(str(m.get("content") or "")) for m in messages if m.get("role") == "assistant")
    has_tool_call = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
    return (
        user_chars >= int(quality["min_user_chars"])
        and (assistant_chars >= int(quality["min_assistant_chars"]) or has_tool_call)
        and len(compact(example)) <= int(quality["max_chars_per_example"])
    )


def write_jsonl(path: Path, examples: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "training" / "config" / "dataset.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--source", action="append")
    parser.add_argument("--no-external", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    build = cfg["build"]
    seed = int(env_value(build.get("seed_env"), build["seed_default"]))
    rng = random.Random(seed)
    prompt = system_prompt(cfg)
    selected = set(args.source or [])
    seen: set[str] = set()
    examples: List[Dict[str, Any]] = []
    stats: Dict[str, int] = defaultdict(int)

    for source in cfg["sources"]:
        if not source.get("enabled", True):
            continue
        if selected and source["name"] not in selected:
            continue
        if args.no_external and source["kind"] == "huggingface":
            continue
        weight = float(source.get("weight", 1.0))
        for example in source_examples(source, cfg, prompt, rng):
            if not valid(example, cfg["quality"]):
                stats[f"rejected:{source['name']}"] += 1
                continue
            key = fingerprint(example)
            if build.get("deduplicate", True) and key in seen:
                stats[f"duplicate:{source['name']}"] += 1
                continue
            seen.add(key)
            whole = int(weight)
            for _ in range(whole):
                examples.append(example)
                stats[source["name"]] += 1
            if rng.random() < max(0.0, weight - whole):
                examples.append(example)
                stats[source["name"]] += 1

    if build.get("shuffle", True):
        rng.shuffle(examples)

    ratios = [float(build["train_ratio"]), float(build["validation_ratio"]), float(build["eval_ratio"])]
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/validation/eval ratios must sum to 1.0")
    total = len(examples)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    splits = {"train": examples[:train_end], "validation": examples[train_end:val_end], "eval": examples[val_end:]}

    raw_output = args.output or env_value(build.get("output_dir_env"), build["output_dir_default"])
    output = Path(raw_output)
    if not output.is_absolute():
        output = ROOT / output
    for split, rows in splits.items():
        write_jsonl(output / f"prixon_{split}.jsonl", rows)

    manifest = {
        "seed": seed,
        "unique_examples": len(seen),
        "weighted_examples": total,
        "splits": {k: len(v) for k, v in splits.items()},
        "source_stats": dict(sorted(stats.items())),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
