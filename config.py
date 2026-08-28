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
DEBUG = False
WAKE_WORD = os.getenv("WAKE_WORD", "").strip().lower()

# Local-only test mode. Keep cloud providers configured for later, but do not
# instantiate or call them while this is enabled.
CLOUD_LLM_ENABLED = os.getenv("CLOUD_LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

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
PROVIDER_FALLBACK_ORDER = ["local", "cerebras", "groq", "huggingface"]

MIN_AVAILABLE_RAM_MB_FOR_LOCAL = int(os.getenv("MIN_AVAILABLE_RAM_MB_FOR_LOCAL", "2500"))
MAX_CPU_PERCENT_FOR_LOCAL = float(os.getenv("MAX_CPU_PERCENT_FOR_LOCAL", "90"))
SYSTEM_POLL_INTERVAL_SECONDS = float(os.getenv("SYSTEM_POLL_INTERVAL_SECONDS", "5"))
SYSTEM_NETWORK_PROBE_EVERY_N_POLLS = int(os.getenv("SYSTEM_NETWORK_PROBE_EVERY_N_POLLS", "6"))
ANOMALY_CPU_THRESHOLD_PERCENT = float(os.getenv("ANOMALY_CPU_THRESHOLD_PERCENT", "90"))
ANOMALY_CPU_MIN_DURATION_SECONDS = float(os.getenv("ANOMALY_CPU_MIN_DURATION_SECONDS", "600"))
ANOMALY_MEMORY_THRESHOLD_PERCENT = float(os.getenv("ANOMALY_MEMORY_THRESHOLD_PERCENT", "90"))
ANOMALY_MEMORY_MIN_DURATION_SECONDS = float(os.getenv("ANOMALY_MEMORY_MIN_DURATION_SECONDS", "900"))
ANOMALY_DISK_THRESHOLD_PERCENT = float(os.getenv("ANOMALY_DISK_THRESHOLD_PERCENT", "90"))
ANOMALY_DISK_MIN_DURATION_SECONDS = float(os.getenv("ANOMALY_DISK_MIN_DURATION_SECONDS", "0"))
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "1800"))

DATA_PATH = os.getenv("ASSISTANT_DATA_PATH", "").strip() or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "windows_automation_10000.jsonl"
)

TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "10"))
# Legacy lexical retrieval is intentionally disabled during the semantic-router
# migration. Set true only for regression comparison/debugging.
LEGACY_LEXICAL_FALLBACK = os.getenv("LEGACY_LEXICAL_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
MIN_TFIDF_SCORE = float(os.getenv("MIN_TFIDF_SCORE", "0.03"))
CONFIRM_RISK_LEVELS = {"medium", "high"}
MIN_NLU_CONFIDENCE = os.getenv("MIN_NLU_CONFIDENCE", "medium").lower()
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.15"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.52"))
SEMANTIC_MIN_GAP = float(os.getenv("SEMANTIC_MIN_GAP", "0.05"))

SAFE_DIAGNOSTIC_TARGETS = {
    "list_processes": "list_processes", "list_services": "list_services", "network_status": "network_status",
    "show_ip_config": "ip_config", "check_disk": "disk_usage", "show_os": "os_info",
    "show_battery": "battery_status", "show_cpu": "cpu_info", "show_memory": "memory_info", "list_startup": "startup_apps",
}

TTS_RATE = int(os.getenv("TTS_RATE", "185"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
MIC_TIMEOUT_SECONDS = int(os.getenv("MIC_TIMEOUT_SECONDS", "6"))
MIC_PHRASE_TIME_LIMIT = int(os.getenv("MIC_PHRASE_TIME_LIMIT", "8"))
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaNeural")
TTS_STREAM_RATE = os.getenv("TTS_STREAM_RATE", "+5%")
TTS_STREAM_PITCH = os.getenv("TTS_STREAM_PITCH", "+0Hz")
TTS_STREAM_SAMPLE_RATE = int(os.getenv("TTS_STREAM_SAMPLE_RATE", "24000"))
