#!/usr/bin/env bash
# One command to install (first run) and start local-brain.
#   ./start.sh                 install what's missing, start Ollama + web UI
#   ./start.sh --port 9000     any extra args are passed to `brain ui`
# Safe to re-run: every step is skipped when already done.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
OLLAMA_URL="http://127.0.0.1:11434"

say()  { printf '\033[1;31m●\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

OS="$(uname -s)"

# ---------------------------------------------------------------- 1. Python
find_python() {
  for v in python3.12 python3.11 python3.13 python3.10; do
    have "$v" && { echo "$v"; return; }
  done
}
PY="$(find_python || true)"
if [ -z "$PY" ]; then
  if [ "$OS" = "Darwin" ] && have brew; then
    say "Installing Python 3.12 via Homebrew ..."
    brew install python@3.12
    PY=python3.12
  else
    die "Python 3.10–3.13 is required. Install it (macOS: 'brew install python@3.12') and re-run."
  fi
fi

# ---------------------------------------------------------------- 2. Ollama
if ! have ollama; then
  if [ "$OS" = "Darwin" ] && have brew; then
    say "Installing Ollama via Homebrew ..."
    brew install ollama
  elif [ "$OS" = "Linux" ]; then
    die "Ollama is required. Install it with: curl -fsSL https://ollama.com/install.sh | sh"
  else
    die "Ollama is required. Install Homebrew (https://brew.sh) or Ollama (https://ollama.com), then re-run."
  fi
fi

# ---------------------------------------------------------------- 3. Python env
STAMP=".venv/.installed-$(cksum pyproject.toml | cut -d' ' -f1)"
if [ ! -f "$STAMP" ]; then
  say "Setting up Python environment (one time) ..."
  [ -d .venv ] || "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e .
  rm -f .venv/.installed-*
  touch "$STAMP"
fi

# ---------------------------------------------------------------- 4. Start Ollama (loopback only)
OLLAMA_PID=""
cleanup() {
  if [ -n "$OLLAMA_PID" ]; then
    say "Stopping Ollama ..."
    kill "$OLLAMA_PID" 2>/dev/null || true
    OLLAMA_PID=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null; then
  say "Starting Ollama on 127.0.0.1 ..."
  OLLAMA_HOST=127.0.0.1:11434 ollama serve >"${TMPDIR:-/tmp}/local-brain-ollama.log" 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 40); do
    curl -sf "$OLLAMA_URL/api/version" >/dev/null && break
    sleep 0.5
  done
  curl -sf "$OLLAMA_URL/api/version" >/dev/null || die "Ollama did not start. See ${TMPDIR:-/tmp}/local-brain-ollama.log"
fi

# ---------------------------------------------------------------- 5. Models (first run only)
WHISPER="${BRAIN_WHISPER:-large-v3-turbo}"
DATA="${BRAIN_HOME:-$ROOT/data}"
if [ ! -f "$DATA/models/faster-whisper-$WHISPER/model.bin" ] \
   || ! .venv/bin/python - <<'EOF'
import sys, requests
from brain import config
names = [m["name"] for m in requests.get(config.OLLAMA_URL + "/api/tags", timeout=5).json()["models"]]
ok = all(any(n.split(":")[0] == m.split(":")[0] and (":" not in m or n == m) for n in names)
         for m in (config.LLM_MODEL, config.EMBED_MODEL))
sys.exit(0 if ok else 1)
EOF
then
  FREE_GB=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')
  say "First run: downloading models (~7.5 GB). Free disk: ${FREE_GB} GB."
  [ "$FREE_GB" -ge 9 ] || say "Warning: less than 9 GB free — the download may not fit."
  .venv/bin/brain setup
fi

# ---------------------------------------------------------------- 6. Go
say "Starting local brain — press Ctrl-C to stop."
.venv/bin/brain ui "$@"
