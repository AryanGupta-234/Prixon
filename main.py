"""Entry point for the LLM-first Windows assistant."""
import argparse
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
import voice
from agent_state import AgentState
from brain.router import get_router
from data_store import ActionIndex
from embeddings import SemanticIndex
from memory import UnifiedMemory
from system import system_agent
from tool_router import CapabilityRegistry, ToolRouter

EXIT_WORDS = {"exit", "quit", "goodbye", "bye", "shutdown assistant", "stop listening"}
YES = {"yes", "y", "yeah", "yep", "sure", "go ahead", "do it", "proceed"}
HEALTH_WORDS = {"health", "status", "diagnostics", "are you ok", "are you okay"}
DEBUG = "--debug" in sys.argv
config.DEBUG = DEBUG  # let brain/router.py (and anything else) trace without a circular import


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
    """Turns a raw AnomalyEvent into roughly the natural-language shape
    spec section 18 shows. This is the deterministic fallback -- true
    natural phrasing that reasons over *why* (spec section 38's "Chrome and
    Python are using most of it") is Tier 3/LLM territory and needs the
    conversational brain in the loop, which this background thread
    shouldn't block on. This keeps alerts working even with every LLM
    provider down (spec section 50's failover non-negotiable)."""
    minutes = event.duration_seconds / 60.0
    duration_txt = f"{minutes:.0f} minute{'s' if minutes >= 2 else ''}" if minutes >= 1 else f"{event.duration_seconds:.0f} seconds"
    metric_txt = {
        "cpu_usage_percent": "CPU usage",
        "memory_percent": "memory usage",
        "disk_used_percent": "disk usage",
    }.get(event.metric, event.metric)
    return f"Heads up -- {metric_txt} has been around {event.value:.0f}% for about {duration_txt}. Worth a look?"


def _make_alert_handler(use_voice):
    def _on_event(event):
        print()  # separate the alert from whatever's mid-typed in the REPL
        say(_format_alert(event), use_voice)
    return _on_event


def handle_command(user_text, index, use_voice, state: AgentState, memory: UnifiedMemory, router: ToolRouter,
                    semantic_index: SemanticIndex):
    chit = small_talk.resolve(user_text)
    if chit.handled:
        _trace("TIER", "tier0-smalltalk")
        say(chit.reply, use_voice)
        return

    # Retrieval is only a fast candidate generator. The LLM (or Tier 1) gets
    # a shortlist, and when lexical retrieval is weak it gets the whole catalog.
    candidates = index.search(user_text)
    broad = not candidates or candidates[0]["score"] < config.MIN_TFIDF_SCORE
    if broad:
        candidates = index.full_catalog()

    if not DEBUG:
        print("(thinking...)", flush=True)
    routed = context_engine.route(user_text, candidates, state, memory, index.groups, config.ASSISTANT_NAME, broad,
                                   semantic_index=semantic_index)
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

    memory.record_event("task_started", intent=result.intent, target=result.match_target,
                         target_name=group.target_name, parameters=result.parameters)

    if executor.needs_confirmation(group.risk):
        say(f"{result.reply} {persona.confirm_prompt()}", use_voice)
        confirmation = get_input(use_voice).lower().strip()
        if confirmation not in YES:
            say(persona.cancelled(), use_voice)
            memory.record_event("task_failed", intent=result.intent, target=result.match_target,
                                 target_name=group.target_name, success=False)
            return
    else:
        say(result.reply, use_voice)

    dispatched = router.dispatch(group, result.parameters)
    v = dispatched.verification
    _trace("VERIFICATION", f"ok={dispatched.ok} " + str(v.to_dict() if v else "not attempted"))

    if not dispatched.ok:
        say(dispatched.message, use_voice)
        memory.record_event("task_failed", intent=result.intent, target=result.match_target,
                             target_name=group.target_name, success=False,
                             parameters={**result.parameters, "verification": v.to_dict() if v else None})
        memory.remember_turn(user_text, result, group.target_name)
        return

    # exec_result.ok means "the call didn't raise", not "it's confirmed
    # working" -- only treat the task as genuinely successful, and only
    # update state/say "Done" as such, when verification agrees (or the
    # action has no way to be verified at all, in which case we say so).
    verified_ok = dispatched.ok and (v is None or v.confirmed is not False)
    state.note_successful_task(result.match_target, group.target_name, result.intent)
    if verified_ok:
        state.active_goal = goal_engine.topic_for_group(group)
        _trace("GOAL", state.active_goal)
    memory.record_event("task_completed", intent=result.intent, target=result.match_target,
                         target_name=group.target_name, success=verified_ok,
                         parameters={**result.parameters, "verification": v.to_dict() if v else None})
    memory.remember_turn(user_text, result, group.target_name)
    _trace("MEMORY", "episode stored")

    if v and v.verified and v.confirmed is False:
        say(f"I tried, but I couldn't confirm it actually opened ({v.evidence}).", use_voice)
    elif dispatched.data:
        # Diagnostics return data; keep it readable without dumping huge JSON.
        text = str(dispatched.data)
        if len(text) > 2500:
            text = text[:2500] + " …"
        say(f"Done. {text}", use_voice)


def main():
    parser = argparse.ArgumentParser(description="LLM-first personal Windows assistant")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--debug", action="store_true",
                         help="Print tier routing / context / verification traces (section 29)")
    parser.add_argument("--healthcheck", action="store_true",
                         help="Run self-diagnostics (spec section 49) once and exit, without starting the REPL")
    args = parser.parse_args()

    use_voice = args.voice
    if use_voice and not voice.voice_available():
        print("Voice dependencies are unavailable; falling back to text mode.\n")
        use_voice = False

    print(f"Loading {config.ASSISTANT_NAME}'s action brain...")
    index = ActionIndex()
    state = AgentState()
    memory = UnifiedMemory()
    registry = CapabilityRegistry(index)
    router = ToolRouter(registry)
    semantic_index = SemanticIndex(index.groups)  # loads in background; may still be unready for a while
    print(f"Loaded {len(index.entries)} language examples across {len(index.groups)} executable actions.")
    _trace("CAPABILITIES", registry.summary())
    _trace("SEMANTIC", "loading in background" if not semantic_index.ready else "ready")

    agent = system_agent.start_default_agent(memory=memory, on_event=_make_alert_handler(use_voice))
    _trace("SYSTEM_AGENT", f"started, polling every {agent.poll_interval}s")

    if args.healthcheck:
        # Give the agent a brief chance to complete its first poll so the
        # report reflects real numbers rather than "no poll completed yet"
        # purely due to startup timing -- capped short since this path is
        # meant to be a quick one-shot check, not a wait for a full cycle.
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
            handle_command(text, index, use_voice, state, memory, router, semantic_index)
        except RuntimeError as exc:
            # nlu._call_llm() raises RuntimeError for both "nothing configured"
            # and "this one request timed out/failed." Neither should kill the
            # whole session (spec section 25/32: stay operational on provider
            # failure) -- report it and keep the loop alive so the next message
            # gets a fresh attempt instead of the app just exiting.
            say(f"I'm having trouble reaching my language model right now: {exc}", use_voice)
        except Exception as exc:
            say(f"Something went wrong on my side: {exc}", use_voice)


if __name__ == "__main__":
    main()
