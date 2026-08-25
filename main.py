"""Entry point for the LLM-first Windows assistant."""
import argparse
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import config
import executor
import persona
import voice
from data_store import ActionIndex
from nlu import ConversationState, resolve

EXIT_WORDS = {"exit", "quit", "goodbye", "bye", "shutdown assistant", "stop listening"}
YES = {"yes", "y", "yeah", "yep", "sure", "go ahead", "do it", "proceed"}


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


def handle_command(user_text, index, use_voice, state):
    # Retrieval is only a fast candidate generator. The LLM gets a shortlist,
    # and when lexical retrieval is weak it gets the whole catalog.
    candidates = index.search(user_text)
    broad = not candidates or candidates[0]["score"] < config.MIN_TFIDF_SCORE
    if broad:
        candidates = index.full_catalog()

    print("(thinking...)", flush=True)
    result = resolve(user_text, candidates, config.ASSISTANT_NAME, broad, state)

    if not result.match_target or result.confidence in {"none", "low"}:
        say(result.reply or persona.failure_fallback(), use_voice)
        return

    group = index.get_group(result.match_target)
    if group is None:
        say(persona.failure_fallback(), use_voice)
        return

    if executor.needs_confirmation(group.risk):
        say(f"{result.reply} {persona.confirm_prompt()}", use_voice)
        confirmation = get_input(use_voice).lower().strip()
        if confirmation not in YES:
            say(persona.cancelled(), use_voice)
            return
    else:
        say(result.reply, use_voice)

    result_exec = executor.run(group, result.parameters)
    if not result_exec.ok:
        say(result_exec.message, use_voice)
    elif result_exec.data:
        # Diagnostics return data; keep it readable without dumping huge JSON.
        text = str(result_exec.data)
        if len(text) > 2500:
            text = text[:2500] + " …"
        say(f"Done. {text}", use_voice)


def main():
    parser = argparse.ArgumentParser(description="LLM-first personal Windows assistant")
    parser.add_argument("--voice", action="store_true")
    args = parser.parse_args()

    use_voice = args.voice
    if use_voice and not voice.voice_available():
        print("Voice dependencies are unavailable; falling back to text mode.\n")
        use_voice = False

    print(f"Loading {config.ASSISTANT_NAME}'s action brain...")
    index = ActionIndex()
    state = ConversationState()
    print(f"Loaded {len(index.entries)} language examples across {len(index.groups)} executable actions.\n")
    say(persona.greeting(), use_voice)

    while True:
        text = get_input(use_voice)
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            say(persona.exit_line(), use_voice)
            break
        try:
            handle_command(text, index, use_voice, state)
        except RuntimeError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            break
        except Exception as exc:
            say(f"Something went wrong on my side: {exc}", use_voice)


if __name__ == "__main__":
    main()
