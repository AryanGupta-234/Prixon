"""Configuration for the Windows personal assistant."""
import os

try:
    from dotenv import load_dotenv, find_dotenv
    here = os.path.dirname(os.path.abspath(__file__))
    local_env = os.path.join(here, ".env")
    load_dotenv(local_env if os.path.exists(local_env) else find_dotenv(usecwd=True))
except ImportError:
    pass

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Jarvis")
WAKE_WORD = os.getenv("WAKE_WORD", "").strip().lower()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras").strip().lower()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "").strip()
HF_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Cerebras and Groq both host openai/gpt-oss-120b with a genuinely generous,
# standing free tier (no monthly credit pool to exhaust like HF's own
# $0.10/month allowance) -- see README for a comparison. Cerebras is the
# default since its 1M-tokens/day free limit comfortably covers sending the
# full action catalog on every request; Groq is faster but has a tighter
# 200K-tokens/day cap for this specific model.
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Local inference via Ollama (spec section 5). OLLAMA_MODEL empty (the
# default) means the local provider reports itself unavailable -- set it
# once you've pulled a model (e.g. `ollama pull qwen2.5:7b-instruct`) to
# start using it. Resource-aware local-vs-cloud choice is a later slice;
# for now this just makes "local" a real, selectable chain member.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()

# Order in which providers are tried if the preferred one (LLM_PROVIDER) is
# unavailable or exhausted. Any provider without credentials set is skipped
# automatically -- you don't need to remove unused ones from this list.
PROVIDER_FALLBACK_ORDER = ["local", "cerebras", "groq", "huggingface"]

# NOTE: os.getenv(name, default) only falls back when the var is entirely
# unset -- an empty-but-present "ASSISTANT_DATA_PATH=" in .env (which is
# exactly what .env.example ships) would otherwise resolve to "" instead of
# the intended default, so an explicit blank check is needed here.
DATA_PATH = os.getenv("ASSISTANT_DATA_PATH", "").strip() or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "windows_automation_10000.jsonl"
)

TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "10"))
MIN_TFIDF_SCORE = float(os.getenv("MIN_TFIDF_SCORE", "0.03"))
CONFIRM_RISK_LEVELS = {"medium", "high"}
MIN_NLU_CONFIDENCE = os.getenv("MIN_NLU_CONFIDENCE", "medium").lower()
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))  # response is short structured JSON; some providers reserve this against the same TPM budget as the prompt
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.15"))

# Local semantic embedding model for the Tier 2 layer (embeddings.py) --
# catches paraphrases TF-IDF Tier 2 can't ("make it quieter" vs "lower the
# volume"). BAAI/bge-small-en-v1.5 via fastembed: ~30MB, ONNX, no torch.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# Cosine-similarity thresholds -- starting points, not calibrated numbers.
# Unlike the TF-IDF Tier 2 thresholds (checked against all 10,000 dataset
# rows), the model download this depends on wasn't reachable from the
# sandbox this was built in, so these haven't been tuned against real
# embedding output. Tune against your own usage once the model downloads.
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.62"))
SEMANTIC_MIN_GAP = float(os.getenv("SEMANTIC_MIN_GAP", "0.08"))

# target -> fixed diagnostic identifier. These are read-only scripts in tools.py.
SAFE_DIAGNOSTIC_TARGETS = {
    "list_processes": "list_processes",
    "list_services": "list_services",
    "network_status": "network_status",
    "show_ip_config": "ip_config",
    "check_disk": "disk_usage",
    "show_os": "os_info",
    "show_battery": "battery_status",
    "show_cpu": "cpu_info",
    "show_memory": "memory_info",
    "list_startup": "startup_apps",
}

TTS_RATE = int(os.getenv("TTS_RATE", "185"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
MIC_TIMEOUT_SECONDS = int(os.getenv("MIC_TIMEOUT_SECONDS", "6"))
MIC_PHRASE_TIME_LIMIT = int(os.getenv("MIC_PHRASE_TIME_LIMIT", "8"))
