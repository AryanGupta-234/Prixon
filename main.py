"""Entry point for the LLM-first Windows assistant."""
import argparse
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import config
import nlu
from ollama_provider import install as install_ollama_provider

# Register the local Ollama provider into the existing NLU provider chain.
# This keeps the current NLU architecture intact while making the local model
# the normal first choice and retaining cloud failover.
install_ollama_provider(nlu)

import context_engine
import executor
import goal_engine
import persona
import small_talk
import voice
from agent_state import AgentState
from data_store import ActionIndex
from embeddings import SemanticIndex
from memory import UnifiedMemory
from tool_router import CapabilityRegistry, ToolRouter

EXIT_WORDS = {"exit", "quit", "goodbye", "bye", "shutdown assistant", "stop listening"}
YES = {"yes", "y", "yeah", "yep", "sure", "go ahead", "do it", "proceed"}
DEBUG = "--debug" in sys.argv


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


def handle_command(user_text, index, use_voice, state: AgentState, memory: UnifiedMemory, router: ToolRouter,
                    semantic_index: SemanticIndex):
    chit = small_talk.resolve(user_text)
    if chit.handled:
        _trace("TIER", "tier0-smalltalk")
        say(chit.reply, use_voice)
        return

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
        text = str(dispatched.data)
        if len(text) > 2500:
            text = text[:2500] + " …"
        say(f"Done. {text}", use_voice)


def main():
    parser = argparse.ArgumentParser(description="LLM-first personal Windows assistant")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--debug", action="store_true",
                         help="Print tier routing / context / verification traces")
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
    semantic_index = SemanticIndex(index.groups)
    print(f"Loaded {len(index.entries)} language examples across {len(index.groups)} executable actions.")
    _trace("CAPABILITIES", registry.summary())
    _trace("SEMANTIC", "loading in background" if not semantic_index.ready else "ready")
    _trace("LLM", f"preferred={config.LLM_PROVIDER} local={config.OLLAMA_MODEL}")
    say(persona.greeting(), use_voice)

    while True:
        text = get_input(use_voice)
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            say(persona.exit_line(), use_voice)
            break
        try:
            handle_command(text, index, use_voice, state, memory, router, semantic_index)
        except RuntimeError as exc:
            # Provider failure must not kill the assistant loop. The NLU layer
            # already tries the configured provider chain; this is the final
            # guard for an unexpected all-provider failure.
            say(f"I'm having trouble reaching my language model right now: {exc}", use_voice)
        except Exception as exc:
            say(f"Something went wrong on my side: {exc}", use_voice)


if __name__ == "__main__":
    main()
