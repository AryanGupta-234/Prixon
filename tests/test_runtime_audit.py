"""Regression coverage for the pre-training runtime audit.

All OS, model, and process effects are mocked: these tests assert the
allow-list and evidence boundaries without opening applications or calling an
LLM provider.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import config
import executor
import main
import nlu
import reference_resolver
from agent_state import AgentState
from brain.provider_base import LLMProvider
from brain.router import ModelRouter
from data_store import ActionGroup, normalize
from memory import UnifiedMemory
from tool_router import CapabilityRegistry, ToolRouter
from verification import VerificationResult


def group(target="calculator", action="open_app", executable="calc.exe", risk="low"):
    return ActionGroup(target, target.title(), "open_app", action, "", executable,
                       "", risk, executable=executable)


def test_normalization_and_empty_input():
    assert normalize("  Wi-Fi pls  ") == "wifi please"
    assert nlu._extract_json("") == {}
    assert nlu._extract_json("not json") == {}


def test_health_report_is_renderable_text():
    import diagnostics
    assert "HEALTH" in diagnostics.format_report([])


def test_malformed_llm_output_and_invented_target_are_rejected(monkeypatch):
    candidates = [group().to_candidate()]
    monkeypatch.setattr(nlu, "call_llm", lambda *args, **kwargs: '{"match_target":"powershell","confidence":"high"}')
    result = nlu.resolve("run arbitrary shell command", candidates)
    assert result.match_target is None
    assert result.confidence == "none"
    monkeypatch.setattr(nlu, "call_llm", lambda *args, **kwargs: "{broken")
    assert nlu.resolve("open calculator", candidates).match_target is None


def test_reference_resolution_handles_repeat_and_ambiguous_pronoun():
    state = AgentState(last_target="calculator", last_target_name="Calculator", last_intent="open_app")
    assert reference_resolver.resolve("do that again", state).target == "calculator"
    assert not reference_resolver.resolve("open it", AgentState()).resolved


def test_memory_skips_corruption_and_persists_future_context(tmp_path):
    path = tmp_path / "episodes.jsonl"
    valid = {"event_id": "1", "event_type": "task_completed", "intent": "open_app",
             "target": "calculator", "target_name": "Calculator", "parameters": {},
             "success": True, "timestamp": 1.0}
    path.write_text("{bad json}\n" + json.dumps(valid) + "\n", encoding="utf-8")
    memory = UnifiedMemory(str(path))
    assert memory.last_successful_task().target == "calculator"
    memory.record_event("task_completed", target="notepad", success=True)
    assert UnifiedMemory(str(path)).last_successful_task().target == "notepad"


def test_execution_policy_rejects_unknown_and_never_passes_user_shell(monkeypatch):
    unknown = group(target="invented", action="run_command", executable="")
    monkeypatch.setattr(executor.sys, "platform", "win32")
    assert not executor.run(unknown, {"command": "Remove-Item C:\\ -Recurse"}).ok
    launch = group()
    launched = Mock(return_value=executor.ToolResult(True, "Launched."))
    monkeypatch.setattr(executor, "launch_process", launched)
    assert executor._run_allowlisted(launch, {"command": "cmd.exe /c whoami"}).ok
    launched.assert_called_once_with("calc.exe", "")


def test_tool_router_attaches_failed_verification(monkeypatch):
    registry = Mock(spec=CapabilityRegistry)
    router = ToolRouter(registry)
    monkeypatch.setattr(executor, "run", lambda *args: executor.ExecutionResult(True, "Launched."))
    monkeypatch.setattr("verification.verify_process_launch", lambda *args: VerificationResult(True, False, "not found"))
    result = router.dispatch(group())
    assert result.ok and result.verification.confirmed is False


class _Provider(LLMProvider):
    def __init__(self, name, response=None, error=None):
        self.name, self.response, self.error = name, response, error
    def available(self): return True
    def chat(self, *args, **kwargs):
        if self.error: raise self.error
        return self.response


def test_provider_fallback_after_timeout(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_ONLY_MODE", False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "local")
    monkeypatch.setattr(config, "PROVIDER_FALLBACK_ORDER", ["local", "cerebras"])
    router = ModelRouter([_Provider("local", error=TimeoutError("timed out")), _Provider("cerebras", response="{}")])
    assert router.call("s", "u", max_tokens=1, temperature=0) == "{}"


def test_provider_fallback_after_auth_error(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_ONLY_MODE", False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "local")
    monkeypatch.setattr(config, "PROVIDER_FALLBACK_ORDER", ["local", "cerebras"])
    router = ModelRouter([_Provider("local", error=RuntimeError("401 unauthorized")), _Provider("cerebras", response="{}")])
    assert router.call("s", "u", max_tokens=1, temperature=0) == "{}"
    assert "local" in router._exhausted


def test_unverified_execution_is_not_recorded_as_success(monkeypatch, tmp_path):
    action = group()
    index = Mock(groups={"calculator": action})
    index.get_group.return_value = action
    result = nlu.NLUResult("calculator", "high", "Opening Calculator", "open_app")
    routed = Mock(result=result, tier="tier3-qwen-semantic", debug={})
    dispatched = Mock(ok=True, data=None, message="Launched.", verification=VerificationResult(False, None, "no handle"))
    memory = UnifiedMemory(str(tmp_path / "memory.jsonl"))
    state = AgentState()
    router = Mock(dispatch=Mock(return_value=dispatched))
    said = []
    monkeypatch.setattr(main.context_engine, "route", lambda *args, **kwargs: routed)
    monkeypatch.setattr(main, "say", lambda text, voice: said.append(text))
    main.handle_command("open calculator", index, False, state, memory, router, None, Mock(), Mock())
    assert state.last_target is None
    assert memory.last_successful_task() is None
    assert "couldn't verify" in said[-1]
