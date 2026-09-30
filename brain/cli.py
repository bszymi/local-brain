"""local-brain command line interface."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config, offline, service


def _log(msg):
    print(f"  {msg}", flush=True)

# --------------------------------------------------------------------------- setup

def cmd_setup(args):
    """Download models once. This is the ONLY command that touches the internet."""
    from . import llm, transcribe

    config.ensure_dirs()
    transcribe.download_model()
    try:
        have = llm.available_models()
    except Exception:
        sys.exit("Ollama is not running. Start it with: OLLAMA_HOST=127.0.0.1 ollama serve")
    for model in (config.LLM_MODEL, config.EMBED_MODEL):
        if not any(m == model or m.split(":")[0] == model for m in have):
            print(f"Pulling {model} via Ollama ...")
            subprocess.run(["ollama", "pull", model], check=True)
    print("\nSetup complete. Everything from here on runs fully offline.")


# --------------------------------------------------------------------------- ingest

def _iter_inputs(paths):
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in config.AUDIO_EXTS | config.TEXT_EXTS:
                    yield f
        elif p.exists():
            yield p
        else:
            print(f"skip: {p} not found", file=sys.stderr)


def cmd_add(args):
    from . import store

    db = store.connect()
    for f in _iter_inputs(args.paths):
        source = str(f.resolve())
        existing = store.find_by_source(db, source)
        if existing and not args.force:
            print(f"skip: {f.name} already in brain as #{existing['id']} (use --force to redo)")
            continue
        if existing:
            store.delete_document(db, existing["id"])
        print(f"\n==> {f.name}")
        service.ingest_file(db, f, language=args.language, extract=not args.no_extract, log=_log)


def cmd_note(args):
    from . import store

    text = " ".join(args.text) if args.text else sys.stdin.read()
    service.ingest_note(store.connect(), text, log=_log)


def cmd_transcribe(args):
    from . import transcribe

    text, meta = transcribe.transcribe(Path(args.file), language=args.language)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"\n{meta}", file=sys.stderr)


# --------------------------------------------------------------------------- query

def cmd_search(args):
    q = " ".join(args.query)
    for hit in service.retrieve(q, args.k):
        snippet = hit["text"].replace("\n", " ")[:220]
        print(f"#{hit['doc_id']}  {hit['title'] or ''}  (sim {hit['similarity']:.2f})\n    {snippet}\n")


def _answer(question, k, history):
    hits, stream = service.answer_stream(question, history, k)
    answer = ""
    for piece in stream:
        answer += piece
        print(piece, end="", flush=True)
    print()
    if hits:
        print("\nsources: " + "; ".join(f"#{s['id']} {s['title']}" for s in service.cited_sources(hits, answer)))
    return answer


def cmd_ask(args):
    if args.question:
        _answer(" ".join(args.question), args.k, [])
        return
    print("local-brain chat (Ctrl-D to quit)")
    history = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        a = _answer(q, args.k, history[-6:])
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


# --------------------------------------------------------------------------- browse

def cmd_list(args):
    from . import store

    for r in store.connect().execute(
            "SELECT id, kind, title, created_at, duration FROM documents ORDER BY id DESC"):
        dur = f" {int(r['duration'] // 60)}m{int(r['duration'] % 60):02d}s" if r["duration"] else ""
        print(f"#{r['id']:<4} {r['created_at'][:16]}  {r['kind']:<5}{dur:>8}  {r['title'] or ''}")


def cmd_show(args):
    from . import knowledge, store

    r = store.connect().execute("SELECT * FROM documents WHERE id = ?", (args.id,)).fetchone()
    if not r:
        sys.exit(f"no document #{args.id}")
    print(f"#{r['id']}  {r['source']}\n")
    if r["knowledge"]:
        print(knowledge.as_text(json.loads(r["knowledge"])))
    if args.full or not r["knowledge"]:
        print("\n--- content ---\n" + r["content"])


def cmd_ui(args):
    from . import web

    web.serve(port=args.port, open_browser=not args.no_browser)


def cmd_delete(args):
    from . import store

    store.delete_document(store.connect(), args.id)
    print(f"deleted #{args.id}")


# --------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(prog="brain", description="A private, fully offline second brain.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="download models (only step that uses the network)").set_defaults(func=cmd_setup)

    s = sub.add_parser("add", help="add audio/video/text files or folders to the brain")
    s.add_argument("paths", nargs="+")
    s.add_argument("-l", "--language", help="force audio language, e.g. en, pl")
    s.add_argument("--force", action="store_true", help="re-process files already in the brain")
    s.add_argument("--no-extract", action="store_true", help="skip LLM knowledge extraction")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("note", help="add a quick text note (argument or stdin)")
    s.add_argument("text", nargs="*")
    s.set_defaults(func=cmd_note)

    s = sub.add_parser("transcribe", help="just transcribe a file, don't store it")
    s.add_argument("file")
    s.add_argument("-l", "--language")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("ask", help="ask a question (no question = interactive chat)")
    s.add_argument("question", nargs="*")
    s.add_argument("-k", type=int, default=8, help="number of snippets to retrieve")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("search", help="semantic + keyword search")
    s.add_argument("query", nargs="+")
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(func=cmd_search)

    sub.add_parser("list", help="list documents").set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="show extracted knowledge for a document")
    s.add_argument("id", type=int)
    s.add_argument("--full", action="store_true", help="also print the full transcript/text")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("delete", help="delete a document")
    s.add_argument("id", type=int)
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser("ui", help="open the local web interface (record, upload, ask)")
    s.add_argument("--port", type=int, default=8777)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_ui)

    args = p.parse_args(argv)
    if args.cmd != "setup":
        offline.enforce()
    args.func(args)


if __name__ == "__main__":
    main()
