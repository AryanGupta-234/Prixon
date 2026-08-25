<<<<<<< HEAD
# PC Assistant — LLM-First Windows Agent

This version treats the LLM as the **semantic brain** rather than a keyword classifier.

## Flow

User → normalization/retrieval → LLM reasoning → allow-listed target → risk policy → fixed tool → verification → natural response

### The LLM handles
- natural English and indirect requests
- synonyms, slang and speech-to-text noise
- conversational references (`it`, `that one`, `same thing`, `there`)
- parameter extraction
- intent vs target separation
- short-term conversational context
- deciding between opening settings, launching an app, or using a diagnostic candidate

### Python handles
- candidate retrieval
- execution policy
- allow-list enforcement
- fixed PowerShell diagnostics
- Windows URI launching
- process launching
- confirmation for medium/high-risk actions

**The LLM never receives permission to invent or execute arbitrary shell commands.**

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Put your Hugging Face token in `.env`, or switch to Anthropic:

```text
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-haiku-4-5-20251001
```

Run:

```powershell
python main.py
```

Voice mode:

```powershell
python main.py --voice
```

## Example requests

```text
Open the thing for Wi-Fi.
My eyes are killing me, dim it.
What's eating all my storage?
Show me what's running.
Launch Calculator.
Can you take me to where I change my mouse?
That one.
Same thing, but Bluetooth.
```

For best results, use a capable instruction-following model. Retrieval is only a fast hint; it is not the assistant's source of truth.
=======
# Prixon
>>>>>>> aa3fe142e68e7e20e0ade9f87df1982ca48aac40
