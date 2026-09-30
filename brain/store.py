"""SQLite knowledge store: documents, extracted notes, and embedded chunks."""

import json
import re
import sqlite3
from datetime import datetime

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- audio | text | note
    title       TEXT,
    created_at  TEXT NOT NULL,
    language    TEXT,
    duration    REAL,
    content     TEXT NOT NULL,
    summary     TEXT,
    knowledge   TEXT                    -- JSON from the LLM extraction
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    db = sqlite3.connect(config.DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(SCHEMA)
    return db


def chunk_text(text: str, size: int = config.CHUNK_CHARS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split on line/sentence boundaries into ~size-char chunks with overlap."""
    pieces = [p for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    chunks, current = [], ""
    for p in pieces:
        if current and len(current) + len(p) + 1 > size:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += (" " if current else "") + p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def add_document(db, *, source, kind, content, title=None, language=None, duration=None,
                 summary=None, knowledge=None, chunks=None, embeddings=None) -> int:
    cur = db.execute(
        "INSERT INTO documents (source, kind, title, created_at, language, duration, content, summary, knowledge)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source, kind, title, datetime.now().isoformat(timespec="seconds"), language, duration,
         content, summary, json.dumps(knowledge, ensure_ascii=False) if knowledge else None),
    )
    doc_id = cur.lastrowid
    for i, (text, vec) in enumerate(zip(chunks or [], embeddings if embeddings is not None else [])):
        db.execute("INSERT INTO chunks (doc_id, idx, text, embedding) VALUES (?, ?, ?, ?)",
                   (doc_id, i, text, np.asarray(vec, dtype=np.float32).tobytes()))
    db.commit()
    return doc_id


def find_by_source(db, source: str):
    return db.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()


def delete_document(db, doc_id: int) -> None:
    db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db.commit()


def _fts_query(q: str) -> str:
    words = re.findall(r"\w+", q, flags=re.UNICODE)
    return " OR ".join(f'"{w}"' for w in words if len(w) > 2)


def search(db, query: str, query_vec: np.ndarray, k: int = 8) -> list[dict]:
    """Hybrid search: vector similarity + FTS5 keyword match, fused with RRF."""
    rows = db.execute(
        "SELECT c.id, c.doc_id, c.text, c.embedding, d.title, d.source FROM chunks c"
        " JOIN documents d ON d.id = c.doc_id"
    ).fetchall()
    if not rows:
        return []
    by_id = {r["id"]: r for r in rows}
    matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = matrix @ query_vec
    vec_rank = [rows[i]["id"] for i in np.argsort(-sims)[: k * 4]]
    sim_by_id = {rows[i]["id"]: float(sims[i]) for i in range(len(rows))}

    fts_rank = []
    fq = _fts_query(query)
    if fq:
        fts_rank = [r[0] for r in db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?", (fq, k * 4))]

    scores: dict[int, float] = {}
    for ranking in (vec_rank, fts_rank):
        for pos, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + pos)

    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{"chunk_id": cid, "doc_id": by_id[cid]["doc_id"], "title": by_id[cid]["title"],
             "source": by_id[cid]["source"], "text": by_id[cid]["text"],
             "similarity": sim_by_id[cid]} for cid in top]
