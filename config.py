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

# Prefer the local Ollama brain. The provider adapter falls through to cloud
# providers when Ollama is unavailable, so local operation never becomes a
# single point of failure.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "").strip()
HF_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Local first, then stronger cloud fallbacks. LLM_PROVIDER can still override
# the preferred starting point without changing this fallback safety net.
PROVIDER_FALLBACK_ORDER = ["ollama", "cerebras", "groq", "huggingface"]

DATA_PATH = os.getenv("ASSISTANT_DATA_PATH", "").strip() or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "windows_automation_10000.jsonl"
)

TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "10"))
MIN_TFIDF_SCORE = float(os.getenv("MIN_TFIDF_SCORE", "0.03"))
CONFIRM_RISK_LEVELS = {"medium", "high"}
MIN_NLU_CONFIDENCE = os.getenv("MIN_NLU_CONFIDENCE", "medium").lower()
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.15"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.62"))
SEMANTIC_MIN_GAP = float(os.getenv("SEMANTIC_MIN_GAP", "0.08"))

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
