"""Entry point for the LLM-first Windows assistant."""
import argparse
import json
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import config
import context_engine
import diagnostics
import executor
import goal_engine
import persona
import small_talk
import tier2
import tools
import voice
from agent_state import AgentState
from brain.router import get_router
from cognition.experience import ExperienceModel
from cognition.patterns import PatternMemory
from cognition import training_log
from data_store import ActionIndex
from embeddings import SemanticIndex
from memory import UnifiedMemory
from system import system_agent
from tool_router import CapabilityRegistry, ToolRouter

EXIT_WORDS = {"exit", "quit", "goodbye", "bye", "shutdown assistant", "stop listening"}
YES = {"yes", "y", "yeah", "yep", "sure", "go ahead", "do it", "proceed"}
HEALTH_WORDS = {"health", "status", "diagnostics", "are you ok", "are you okay"}
DEBUG = "--debug" in sys.argv
config.DEBUG = DEBUG


def say(text, use_voice):
    print(f"{config.ASSISTANT_NAME}: {text}", flush=True)
    if use_voice:
        voice.speak(text)


def get_input(use_voice):
    if use_voice:
        text = voice.listen_once()
        if text:
            print(f"You (heard): {text}", flush=True)
        return text or ""
    try:
        return input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"


def _trace(label, payload):
    if DEBUG:
        print(f"[{label}] {payload}", flush=True)


def _format_alert(event) -> str:
    minutes = event.duration_seconds / 60.0
    duration_txt = f"{minutes:.0f} minute{'s' if minutes >= 2 else ''}" if minutes >= 1 else f"{event.duration_seconds:.0f} seconds"
    metric_txt = {"cpu_usage_percent": "CPU usage", "memory_percent": "memory usage", "disk_used_percent": "disk usage"}.get(event.metric, event.metric)
    base = f"Heads up -- {metric_txt} has been around {event.value:.0f}% for about {duration_txt}."
    if event.top_process:
        base += f" {event.top_process} is using the most of it."
    return base + " Worth a look?"


def _make_alert_handler(use_voice):
    def _on_event(event):
        print()
        say(_format_alert(event), use_voice)
    return _on_event


def _diagnostic_reply(user_text: str, group, data, state: AgentState, parameters=None):
    """Turn diagnostic data into a concise answer using current context.

    `parameters` is important for reference-only turns such as "check again":
    the user's new utterance may contain no entity name, but the situation
    state has already resolved the concrete entity for this execution.
    """
    if data is None:
        return None
    parameters = parameters or {}
    parsed = data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception:
            parsed = None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        hint = parameters.get("app_name_hint") or tier2.extract_app_name_hint(user_text) or tier2.extract_running_app_hint(user_text)
        if hint:
            resolved = parameters.get("resolved_process") or tools.find_running_app(hint)
            if resolved:
                matching = [row for row in parsed if str(row.get("Name", "")).lower() == resolved.lower()]
                # Some Windows process output can normalize names differently;
                # if the live resolver found the process, a verified process
                # lookup is still authoritative even when the diagnostic JSON
                # omitted it between samples.
                if matching or resolved:
                    state.note_referenced_app(resolved, hint, operation="application_status")
                    count = len(matching) or 1
                    return f"Yes — {hint} is running. I found {count} {hint} process{'es' if count != 1 else ''}."
            return f"No — I don't see {hint} running right now."
        if all("Name" in row and "CPU" in row for row in parsed):
            top = parsed[:5]
            parts = [f"{r.get('Name')} ({float(r.get('CPU') or 0):.1f} CPU)" for r in top]
            return "Top processes right now: " + ", ".join(parts) + "."
    return None


def handle_command(user_text, index, use_voice, state: AgentState, memory: UnifiedMemory,
                   router: ToolRouter, semantic_index: SemanticIndex, experience: ExperienceModel,
                   patterns: PatternMemory):
    chit = small_talk.resolve(user_text)
    if chit.handled:
        _trace("TIER", "tier0-smalltalk")
        memory.conversation.turns.append(__import__('nlu').Turn(user=user_text, reply=chit.reply))
        say(chit.reply, use_voice)
        return

    # Semantic retrieval is the sole candidate source. The historical TF-IDF
    # shim remains retired and is never used to decide an action.
    candidates = []
    broad = False
    if not DEBUG:
        print("(thinking...)", flush=True)
    routed = context_engine.route(user_text, candidates, state, memory, index.groups, config.ASSISTANT_NAME, broad,
                                   semantic_index=semantic_index, patterns=patterns)
    result = routed.result
    _trace("TIER", routed.tier)
    _trace("CONTEXT", routed.debug)
    if not result.match_target or result.confidence in {"none", "low"}:
        say(result.reply or persona.failure_fallback(), use_voice)
        return
    group = index.get_group(result.match_target)
    if group is None:
        say(persona.failure_fallback(), use_voice)
        return

    memory.record_event("task_started", intent=result.intent, target=result.match_target, target_name=group.target_name, parameters=result.parameters)
    resolved_process = None
    if (group.action or "").lower() == "close_app_dynamic":
        hint = result.parameters.get("app_name_hint") or tier2.extract_app_name_hint(user_text) or group.target_name
        resolved_process = tools.find_running_app(hint)
        if not resolved_process:
            say(f"I don't see anything matching '{hint}' currently running.", use_voice)
            memory.record_event("task_failed", intent=result.intent, target=result.match_target, target_name=group.target_name, success=False)
            experience.observe("task_failed", result.match_target, group.target_name, False)
            patterns.observe_action(group.target_name, result.intent, False)
            return
        result.parameters = {**result.parameters, "resolved_process": resolved_process, "app_name_hint": hint}
        result.reply = f"Found {resolved_process} running."

    if executor.needs_confirmation(group.risk):
        if (group.action or "").lower() == "close_app_dynamic":
            prompt = f"{result.reply} Close {result.parameters['resolved_process']}? {persona.confirm_prompt()}"
        else:
            prompt = f"{result.reply} {persona.confirm_prompt()}"
        say(prompt, use_voice)
        confirmation = get_input(use_voice).lower().strip()
        if confirmation not in YES:
            say(persona.cancelled(), use_voice)
            memory.record_event("task_failed", intent=result.intent, target=result.match_target, target_name=group.target_name, success=False)
            experience.observe("task_failed", result.match_target, group.target_name, False)
            patterns.observe_action(group.target_name, result.intent, False)
            return
    else:
        say(result.reply, use_voice)

    dispatched = router.dispatch(group, result.parameters)
    v = dispatched.verification
    _trace("VERIFICATION", f"ok={dispatched.ok} " + str(v.to_dict() if v else "not attempted"))
    if not dispatched.ok:
        say(dispatched.message, use_voice)
        memory.record_event("task_failed", intent=result.intent, target=result.match_target, target_name=group.target_name, success=False,
                            parameters={**result.parameters, "verification": v.to_dict() if v else None})
        memory.remember_turn(user_text, result, group.target_name)
        experience.observe("task_failed", result.match_target, group.target_name, False)
        patterns.observe_action(group.target_name, result.intent, False)
        return

    verified_ok = dispatched.ok and (v is None or v.confirmed is not False)
    concrete_name = resolved_process or result.parameters.get("app_name_hint") or group.target_name
    previous = state.last_target_name
    state.note_successful_task(result.match_target, concrete_name, result.intent, resolved_name=resolved_process)
    if verified_ok:
        state.active_goal = goal_engine.topic_for_group(group)
        _trace("GOAL", state.active_goal)
    memory.record_event("task_completed", intent=result.intent, target=result.match_target, target_name=concrete_name, success=verified_ok,
                        parameters={**result.parameters, "verification": v.to_dict() if v else None})
    memory.remember_turn(user_text, result, concrete_name)
    experience.observe("task_completed", result.match_target, concrete_name, verified_ok)
    patterns.observe_action(concrete_name, result.intent, verified_ok)
    patterns.observe_transition(previous, concrete_name, verified_ok)
    state.learned_context = experience.context()
    training_log.log_verified_interaction(user_text, routed.tier, result.raw, verified_ok)
    _trace("MEMORY", "episode + experience + patterns stored")

    if v and v.verified and v.confirmed is False:
        say(f"I tried, but I couldn't confirm it actually opened ({v.evidence}).", use_voice)
    elif dispatched.data:
        concise = _diagnostic_reply(user_text, group, dispatched.data, state, result.parameters)
        if concise:
            say(concise, use_voice)
        else:
            text = str(dispatched.data)
            if len(text) > 2500:
                text = text[:2500] + " …"
            say(f"Done. {text}", use_voice)


def main():
    parser = argparse.ArgumentParser(description="LLM-first Windows personal assistant")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    use_voice = args.voice
    if use_voice and not voice.voice_available():
        print("Voice dependencies are unavailable; falling back to text mode.\n")
        use_voice = False

    print(f"Loading {config.ASSISTANT_NAME}'s action brain...")
    index = ActionIndex()
    state = AgentState()
    memory = UnifiedMemory()
    experience = ExperienceModel()
    patterns = PatternMemory()
    state.learned_context = experience.context()
    registry = CapabilityRegistry(index)
    router = ToolRouter(registry)
    semantic_index = SemanticIndex(index.groups)
    print(f"Loaded {len(index.entries)} language examples across {len(index.groups)} executable actions.")
    _trace("CAPABILITIES", registry.summary())
    _trace("SEMANTIC", "loading in background" if not semantic_index.ready else "ready")
    _trace("LEARNING", {"experience": state.learned_context, "patterns": patterns.context()})

    agent = system_agent.start_default_agent(memory=memory, on_event=_make_alert_handler(use_voice))
    _trace("SYSTEM_AGENT", f"started, polling every {agent.poll_interval}s")
    if args.healthcheck:
        deadline = time.time() + 2.0
        while not agent.ready and time.time() < deadline:
            time.sleep(0.1)
        checks = diagnostics.run(get_router(), memory, semantic_index, registry)
        print(diagnostics.format_report(checks))
        agent.stop()
        return

    say(persona.greeting(), use_voice)
    while True:
        text = get_input(use_voice)
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            say(persona.exit_line(), use_voice)
            agent.stop()
            break
        if text.lower() in HEALTH_WORDS:
            checks = diagnostics.run(get_router(), memory, semantic_index, registry)
            print(diagnostics.format_report(checks))
            continue
        try:
            handle_command(text, index, use_voice, state, memory, router, semantic_index, experience, patterns)
        except RuntimeError as exc:
            say(f"I'm having trouble reaching my language model right now: {exc}", use_voice)
        except Exception as exc:
            say(f"Something went wrong on my side: {exc}", use_voice)


if __name__ == "__main__":
    main()
