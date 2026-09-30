"""Core operations shared by the CLI and the web UI."""

from pathlib import Path
from typing import Callable, Iterator

from . import config

ANSWER_SYSTEM = """You are the user's private second brain. Answer the question using ONLY the
context snippets from their knowledge base below. Cite sources inline as [#doc_id].
If the context does not contain the answer, say so plainly. Answer in the language of the question."""

Log = Callable[[str], None]


def ingest_text(db, *, source, kind, content, title=None, language=None, duration=None,
                extract=True, log: Log = print) -> int:
    from . import knowledge, llm, store

    k = None
    if extract:
        log("extracting knowledge with local LLM ...")
        k = knowledge.extract(content)
        title = title or k.get("title")
    chunks = store.chunk_text(content)
    if k:
        chunks.insert(0, knowledge.as_text(k))
    log(f"embedding {len(chunks)} chunks ...")
    vecs = llm.embed(chunks)
    doc_id = store.add_document(db, source=source, kind=kind, content=content, title=title,
                                language=language, duration=duration,
                                summary=k.get("summary") if k else None, knowledge=k,
                                chunks=chunks, embeddings=vecs)
    log(f"stored as #{doc_id}: {title}")
    return doc_id


def ingest_file(db, f: Path, *, language=None, extract=True, title=None, log: Log = print) -> int:
    source = str(f.resolve())
    if f.suffix.lower() in config.AUDIO_EXTS:
        from . import transcribe

        log("transcribing with faster-whisper ...")
        text, meta = transcribe.transcribe(f, language=language)
        if not text.strip():
            raise ValueError("no speech detected in the recording")
        out = config.TRANSCRIPTS_DIR / f"{f.stem}.txt"
        out.write_text(text, encoding="utf-8")
        log(f"{meta['duration']}s of audio ({meta['language']}) in {meta['elapsed']}s")
        return ingest_text(db, source=source, kind="audio", content=text, title=title,
                           language=meta["language"], duration=meta["duration"],
                           extract=extract, log=log)
    return ingest_text(db, source=source, kind="text", content=f.read_text(encoding="utf-8"),
                       title=title, extract=extract, log=log)


def ingest_note(db, text: str, log: Log = print) -> int:
    text = text.strip()
    return ingest_text(db, source="note", kind="note", content=text,
                       title=text.splitlines()[0][:70], extract=len(text) > 400, log=log)


def retrieve(question: str, k: int = 8) -> list[dict]:
    from . import llm, store

    return store.search(store.connect(), question, llm.embed([question], kind="query")[0], k=k)


def answer_stream(question: str, history: list[dict], k: int = 8) -> tuple[list[dict], Iterator[str]]:
    """Returns (retrieved hits, stream of answer text pieces)."""
    from . import llm

    hits = retrieve(question, k)
    context = "\n\n".join(f"[#{h['doc_id']}] {h['title'] or ''}\n{h['text']}" for h in hits) or "(empty)"
    messages = [{"role": "system", "content": ANSWER_SYSTEM}, *history,
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]
    return hits, llm.chat_stream(messages)


def cited_sources(hits: list[dict], answer: str) -> list[dict]:
    cited = [h for h in hits if f"#{h['doc_id']}]" in answer] or hits
    seen, out = set(), []
    for h in sorted(cited, key=lambda h: h["doc_id"]):
        if h["doc_id"] not in seen:
            seen.add(h["doc_id"])
            out.append({"id": h["doc_id"], "title": h["title"] or Path(h["source"]).name})
    return out
