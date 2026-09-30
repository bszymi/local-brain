import os
from pathlib import Path

ROOT = Path(os.environ.get("BRAIN_HOME", Path(__file__).resolve().parent.parent / "data"))
DB_PATH = ROOT / "brain.db"
TRANSCRIPTS_DIR = ROOT / "transcripts"
MODELS_DIR = ROOT / "models"

OLLAMA_URL = os.environ.get("BRAIN_OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("BRAIN_LLM", "qwen3:8b")
EMBED_MODEL = os.environ.get("BRAIN_EMBED", "embeddinggemma")
LLM_CTX = int(os.environ.get("BRAIN_LLM_CTX", "16384"))

WHISPER_MODEL = os.environ.get("BRAIN_WHISPER", "large-v3-turbo")
WHISPER_DEVICE = os.environ.get("BRAIN_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("BRAIN_WHISPER_COMPUTE", "int8")

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
              ".mp4", ".mov", ".mkv", ".webm", ".avi"}
TEXT_EXTS = {".txt", ".md", ".markdown"}


def ensure_dirs() -> None:
    for d in (ROOT, TRANSCRIPTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
