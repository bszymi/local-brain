"""Thin client for the local Ollama server (127.0.0.1 only)."""

import json
import sys
from typing import Iterator

import numpy as np
import requests

from . import config


def _post(path: str, payload: dict, stream: bool = False):
    try:
        r = requests.post(f"{config.OLLAMA_URL}{path}", json=payload, stream=stream, timeout=600)
    except requests.ConnectionError:
        sys.exit("Ollama is not running. Start it with: OLLAMA_HOST=127.0.0.1 ollama serve")
    r.raise_for_status()
    return r


def embed(texts: list[str], kind: str = "document") -> np.ndarray:
    # embeddinggemma / nomic expect task prefixes; harmless for other models.
    prefix = "task: search result | query: " if kind == "query" else "title: none | text: "
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        r = _post("/api/embed", {"model": config.EMBED_MODEL, "input": batch})
        out.extend(r.json()["embeddings"])
    vecs = np.asarray(out, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs


def chat(messages: list[dict], schema: dict | None = None, temperature: float = 0.2) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_ctx": config.LLM_CTX, "temperature": temperature},
    }
    if schema:
        payload["format"] = schema
    return _post("/api/chat", payload).json()["message"]["content"]


def chat_stream(messages: list[dict], temperature: float = 0.3) -> Iterator[str]:
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"num_ctx": config.LLM_CTX, "temperature": temperature},
    }
    for line in _post("/api/chat", payload, stream=True).iter_lines():
        if line:
            yield json.loads(line).get("message", {}).get("content", "")


def available_models() -> list[str]:
    r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
    return [m["name"] for m in r.json().get("models", [])]
