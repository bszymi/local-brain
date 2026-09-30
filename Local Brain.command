#!/bin/zsh
# Double-click to start your local brain (Ollama + web UI). Close this window to stop.
cd "$(dirname "$0")"
if ! curl -s http://127.0.0.1:11434/api/version >/dev/null; then
  echo "Starting Ollama (local only)..."
  OLLAMA_HOST=127.0.0.1:11434 ollama serve >/dev/null 2>&1 &
  OLLAMA_PID=$!
  trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
  for i in {1..20}; do curl -s http://127.0.0.1:11434/api/version >/dev/null && break; sleep 0.5; done
fi
exec .venv/bin/brain ui
