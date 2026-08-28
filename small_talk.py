"""Tier 0: zero-LLM, zero-retrieval handling for pure conversational chit-chat."""
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import Optional

GREETINGS={"hi","hello","hey","yo","yoo","sup","wassup","what's up","whats up","hiya","howdy","hey man","hey bro","yo man","yoo man","morning","good morning","good afternoon","good evening","good night"}
THANKS={"thanks","thank you","thx","ty","appreciate it","cheers"}
STATUS_QUESTIONS={"how are you","how are u","how's it going","hows it going","how is it going","you there","are you there","you good","are you good","whats up","what's up","wassup","sup jarvis","sup prixon"}
IDENTITY_QUESTIONS={"who are you","what are you","what can you do","what do you do"}
ACKNOWLEDGEMENTS={"ok","okay","cool","nice","alright","got it","sounds good","great","oke","k"}
GREETINGS_REPLIES=["Hey. What do you need?","Hey, I'm listening.","Yo. What's up?"]
THANKS_REPLIES=["Anytime.","You got it.","No problem."]
STATUS_REPLIES=["Doing good. What's next?","All good here. What are we doing?"]
IDENTITY_REPLIES=["I handle your apps, Windows, system state, and whatever else I can help with."]
ACK_REPLIES=["Noted.","Alright.","Got it."]

GREETING_CORE_WORDS={"hi","hello","hey","yo","yoo","sup","wassup","hiya","howdy","morning"}
STATUS_CORE_WORDS={"how","wassup","sup","good","there","going"}
_FILLER_WORDS={"man","bro","dude","there","u","you","whats","what's","up","good","evening","afternoon","night","jarvis","prixon","yo","hey","today","doing","going","it"}

@dataclass
class SmallTalkResult:
    handled: bool
    reply: Optional[str]=None

def _normalize(text:str)->str:
    text=re.sub(r"[^\w\s']","",text.lower().strip())
    return re.sub(r"\s+"," ",text)

def _word_match(words:list,core:set)->bool:
    if not words or len(words)>7:return False
    if not any(w in core for w in words):return False
    return all(w in core or w in _FILLER_WORDS for w in words)

def resolve(user_text:str)->SmallTalkResult:
    text=_normalize(user_text)
    if not text or len(text.split())>7:return SmallTalkResult(False)
    if text in GREETINGS:return SmallTalkResult(True,random.choice(GREETINGS_REPLIES))
    if text in THANKS:return SmallTalkResult(True,random.choice(THANKS_REPLIES))
    if text in STATUS_QUESTIONS:return SmallTalkResult(True,random.choice(STATUS_REPLIES))
    if text in IDENTITY_QUESTIONS:return SmallTalkResult(True,random.choice(IDENTITY_REPLIES))
    if text in ACKNOWLEDGEMENTS:return SmallTalkResult(True,random.choice(ACK_REPLIES))
    words=text.split()
    if _word_match(words,STATUS_CORE_WORDS):return SmallTalkResult(True,random.choice(STATUS_REPLIES))
    if _word_match(words,GREETING_CORE_WORDS):return SmallTalkResult(True,random.choice(GREETINGS_REPLIES))
    return SmallTalkResult(False)
