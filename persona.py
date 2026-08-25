"""Short, natural voice/personality responses."""
import random
import config

GREETINGS = [
    "{name} online. What are we doing?",
    "I'm listening.",
    "{name} here. Go ahead.",
    "Ready when you are.",
    "Systems are up. What's next?",
]
CONFIRM_PROMPTS = [
    "That needs a quick confirmation. Proceed?",
    "Before I do that, want me to continue?",
]
CANCELLED = ["Alright, cancelled.", "No problem. Leaving it alone."]
FAILURE_FALLBACK = ["I couldn't place that yet. Try saying it another way?", "I'm not quite sure what you mean yet."]
EXIT_LINES = ["Signing off. Call me if you need me.", "Alright, stepping back. Catch you later."]

def greeting(): return random.choice(GREETINGS).format(name=config.ASSISTANT_NAME)
def confirm_prompt(): return random.choice(CONFIRM_PROMPTS)
def cancelled(): return random.choice(CANCELLED)
def failure_fallback(): return random.choice(FAILURE_FALLBACK)
def exit_line(): return random.choice(EXIT_LINES)
