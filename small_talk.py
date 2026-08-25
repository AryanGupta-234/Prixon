"""Tier 0: zero-LLM, zero-retrieval handling for pure conversational chit-chat.

Greetings, thanks, "how are you", "who are you" etc. don't correspond to any
catalogued action. Without this, a bare "hi" still goes through retrieval
(TF-IDF score ~0 against a catalog of Windows settings -> broad_search=True
-> the FULL 207-action catalog gets serialized into the LLM prompt) just to
have the model come back and say "that's not an action." That's exactly the
wasted round-trip spec section 26 warns against ("the intelligence should
come from routing, not simply using a larger model"), and it's the concrete
cause of "hi" being slow -- a big prompt on a free-tier remote model is slow
even when the answer is "no match." This runs BEFORE Tier 1 in main.py's
dispatch order specifically to avoid ever reaching that broad_search path.

Deliberately conservative: only fires when the ENTIRE utterance is chit-chat
with nothing else in it ("hi") -- never a prefix match, so "hi, can you open
wifi settings" still goes through the normal pipeline untouched.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

GREETINGS = {"hi", "hello", "hey", "yo", "sup", "hiya", "morning", "good morning",
             "good afternoon", "good evening", "howdy"}
THANKS = {"thanks", "thank you", "thx", "ty", "appreciate it", "cheers"}
STATUS_QUESTIONS = {"how are you", "hows it going", "you there", "are you there",
                     "you good", "whats up", "sup jarvis"}
IDENTITY_QUESTIONS = {"who are you", "what are you", "what can you do", "what do you do"}
ACKNOWLEDGEMENTS = {"ok", "okay", "cool", "nice", "alright", "got it", "sounds good", "great"}

GREETING_REPLIES = ["Hey. What do you need?", "Hi there. Go ahead.", "Hey, I'm listening."]
THANKS_REPLIES = ["Anytime.", "You got it.", "No problem."]
STATUS_REPLIES = ["Running fine. What can I do?", "All good here. What's next?"]
IDENTITY_REPLIES = ["I handle Windows settings, apps, and quick diagnostics -- just tell me what you need."]
ACK_REPLIES = ["Noted.", "Alright.", "Got it."]


@dataclass
class SmallTalkResult:
    handled: bool
    reply: Optional[str] = None


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s']", "", text.lower()).strip()


def resolve(user_text: str) -> SmallTalkResult:
    text = _normalize(user_text)
    if not text or len(text.split()) > 5:
        return SmallTalkResult(False)  # too long to be pure chit-chat -- let it through untouched

    if text in GREETINGS:
        return SmallTalkResult(True, random.choice(GREETING_REPLIES))
    if text in THANKS:
        return SmallTalkResult(True, random.choice(THANKS_REPLIES))
    if text in STATUS_QUESTIONS:
        return SmallTalkResult(True, random.choice(STATUS_REPLIES))
    if text in IDENTITY_QUESTIONS:
        return SmallTalkResult(True, random.choice(IDENTITY_REPLIES))
    if text in ACKNOWLEDGEMENTS:
        return SmallTalkResult(True, random.choice(ACK_REPLIES))
    return SmallTalkResult(False)
