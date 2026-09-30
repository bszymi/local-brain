"""Turn raw text (e.g. a transcript) into structured knowledge with the local LLM."""

import json

from . import llm

KNOWLEDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "key_points", "topics", "people", "decisions",
                 "action_items", "facts"],
}

EXTRACT_PROMPT = """You are building a personal knowledge base. Read the material below and extract
what is worth remembering. Be faithful to the source; never invent anything.

- title: short descriptive title
- summary: 3-6 sentence summary
- key_points: the most important ideas
- topics: short topic tags (lowercase)
- people: people or organisations mentioned
- decisions: decisions that were made (empty if none)
- action_items: tasks / to-dos, with owner if stated (empty if none)
- facts: standalone, self-contained facts worth recalling later

Write in the same language as the material.

MATERIAL:
{text}"""

SECTION_PROMPT = """Summarise this part of a longer recording/document in detail, keeping all
names, numbers, decisions, tasks and facts. Write in the same language as the text.

{text}"""

# Roughly 4 chars/token; leave headroom inside the context window.
MAX_DIRECT_CHARS = 30000


def _condense(text: str) -> str:
    """Map step for long material: summarise sections so the extraction fits in context."""
    parts = [text[i:i + MAX_DIRECT_CHARS] for i in range(0, len(text), MAX_DIRECT_CHARS)]
    notes = []
    for n, part in enumerate(parts, 1):
        print(f"  condensing section {n}/{len(parts)} ...", flush=True)
        notes.append(llm.chat([{"role": "user", "content": SECTION_PROMPT.format(text=part)}]))
    return "\n\n".join(notes)


def extract(text: str) -> dict:
    material = text if len(text) <= MAX_DIRECT_CHARS else _condense(text)
    raw = llm.chat([{"role": "user", "content": EXTRACT_PROMPT.format(text=material)}],
                   schema=KNOWLEDGE_SCHEMA, temperature=0.1)
    return json.loads(raw)


def as_text(k: dict) -> str:
    """Render extracted knowledge as text so it is searchable alongside the raw chunks."""
    lines = [f"# {k.get('title', '')}", k.get("summary", "")]
    for field in ("key_points", "decisions", "action_items", "facts"):
        items = k.get(field) or []
        if items:
            lines.append(f"\n{field.replace('_', ' ').title()}:")
            lines += [f"- {i}" for i in items]
    if k.get("topics"):
        lines.append("\nTopics: " + ", ".join(k["topics"]))
    if k.get("people"):
        lines.append("People: " + ", ".join(k["people"]))
    return "\n".join(lines)
