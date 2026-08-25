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

Put a free API key in `.env`:

```text
LLM_PROVIDER=cerebras
CEREBRAS_API_KEY=...     # free at https://cloud.cerebras.ai
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

## Choosing an LLM provider

`openai/gpt-oss-120b` (OpenAI's open-weight model) is a strong, free choice
for this -- but *where* you run it matters a lot, because "free" means very
different things depending on the host:

| Provider | Free limit for gpt-oss-120b | Card needed? |
|---|---|---|
| **Cerebras** (default) | 1,000,000 tokens/day | No |
| **Groq** | 1,000 requests/day, 200,000 tokens/day | No |
| **Hugging Face** | ~$0.10/month in credits (tiny) | No |

Since every request here sends the full 207-action catalog (~4-5K tokens),
Cerebras's much larger daily token allowance is the practical choice for
sustained personal use -- Hugging Face's own free tier is really only enough
to sample the app, not run it day to day. Groq is faster but its lower daily
token cap will run out sooner for this specific pattern.

Set as many provider keys as you want in `.env` -- see the comments there.
You don't have to pick just one.

## Automatic failover when a provider runs out of quota

Every free tier eventually hits a rate limit or daily/monthly cap. When
that happens mid-session, the app:

1. Detects it specifically (distinct from "model not found" or "still
   warming up").
2. Automatically moves to the next provider that has credentials configured,
   in this order: Cerebras -> Groq -> Hugging Face.
3. Remembers which providers are exhausted for the rest of the session, so
   it doesn't waste a request re-discovering that on every command -- but
   still retries a provider fresh the next time you launch the app.
4. If every configured provider is exhausted, you get a clear message
   naming each one and where to check/top up, instead of a cryptic error.

Setting two or three of these keys costs nothing and means you're very
unlikely to get stuck mid-task -- e.g. `CEREBRAS_API_KEY` as your daily
driver with `GROQ_API_KEY` as a same-model backup.
