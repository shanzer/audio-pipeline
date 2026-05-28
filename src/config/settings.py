import os
import re

def _strip_scheme(host: str) -> str:
    return re.sub(r"^https?://", "", host).rstrip("/")

def _base_url(url_var: str, host_var: str, port_var: str, default_host: str, default_port: str) -> str:
    """
    Resolve a service base URL from env vars.

    Priority:
      1. <url_var>  — full URL, used as-is          e.g. OLLAMA_BASE_URL=http://localhost:4000
      2. <host_var> + <port_var>                     e.g. OLLAMA_HOST=yoyodyne OLLAMA_PORT=11434
         - strips any accidental http:// from host
         - if host already contains a port, ignores port_var
    """
    if url := os.getenv(url_var, "").strip():
        return url.rstrip("/")
    raw_host = os.getenv(host_var, default_host).strip()
    port = os.getenv(port_var, default_port).strip()
    host = _strip_scheme(raw_host)
    if ":" in host:          # host already has a port (e.g. localhost:4000)
        return f"http://{host}"
    return f"http://{host}:{port}"


# ── Data directories ──────────────────────────────────────────────────────────
_DATA_DIR = os.path.expanduser(
    os.getenv("PIPELINE_DATA_DIR", "~/data/audio-pipeline")
)

RECORDINGS_INBOX   = os.path.join(_DATA_DIR, "inbox")
RECORDINGS_ARCHIVE = os.path.join(_DATA_DIR, "archive")
RECORDINGS_FAILED  = os.path.join(_DATA_DIR, "failed")
SQLITE_DB_PATH     = os.path.join(_DATA_DIR, "db", "jobs.db")
LOG_PATH           = os.path.join(_DATA_DIR, "logs", "pipeline.log")

# ── Remote services ───────────────────────────────────────────────────────────
# WhisperX service (Mac Mini) — set WHISPER_SERVICE_URL for SSH tunnels
WHISPER_SERVICE_URL = _base_url(
    "WHISPER_SERVICE_URL", "WHISPER_SERVICE_HOST", "WHISPER_SERVICE_PORT",
    "yoyodyne", "8765",
)

# Ollama (Mac Mini) — set OLLAMA_BASE_URL for SSH tunnels
OLLAMA_BASE_URL = _base_url(
    "OLLAMA_BASE_URL", "OLLAMA_HOST", "OLLAMA_PORT",
    "yoyodyne", "11434",
)

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL        = "mxbai-embed-large"
EMBED_DIMENSIONS   = 1024

LLM_MODEL          = "qwen2.5:14b"

CHUNK_TARGET_TOKENS  = 350
CHUNK_OVERLAP_TOKENS = 50

DEFAULT_MIN_SPEAKERS = 1
DEFAULT_MAX_SPEAKERS = 6
