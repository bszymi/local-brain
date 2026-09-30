"""Local web UI: http://127.0.0.1:8777 — record voice, drop files, ask questions.

Standard library only. Binds to loopback, rejects foreign Host/Origin headers
(so other websites open in your browser can't talk to it), and all heavy work
runs on a single background worker so models are loaded once.
"""

import itertools
import json
import queue
import re
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, service, store

STATIC = Path(__file__).parent / "static"
UPLOADS = config.ROOT / "uploads"

_jobs: dict[int, dict] = {}
_job_ids = itertools.count(1)
_queue: "queue.Queue[tuple[int, callable]]" = queue.Queue()
_whisper_lock = threading.Lock()


# --------------------------------------------------------------------------- jobs

def _submit(label: str, fn) -> dict:
    job_id = next(_job_ids)
    job = {"id": job_id, "label": label, "status": "queued", "log": [], "doc_id": None,
           "error": None, "started": time.time()}
    _jobs[job_id] = job
    _queue.put((job_id, fn))
    return job


def _worker():
    while True:
        job_id, fn = _queue.get()
        job = _jobs[job_id]
        job["status"] = "running"
        try:
            db = store.connect()
            job["doc_id"] = fn(db, lambda m: job["log"].append(m))
            job["status"] = "done"
        except Exception as e:  # surface the failure in the UI
            traceback.print_exc()
            job["status"], job["error"] = "error", str(e)
        job["elapsed"] = round(time.time() - job["started"], 1)


def _ingest_upload(path: Path, language, title):
    def run(db, log):
        with _whisper_lock:
            return service.ingest_file(db, path, language=language, title=title, log=log)
    return run


# --------------------------------------------------------------------------- http

def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w.\- ]+", "_", Path(name).name).strip() or "upload"
    return f"{datetime.now():%Y%m%d-%H%M%S}_{name}"


class Handler(BaseHTTPRequestHandler):
    server_version = "local-brain"

    def log_message(self, fmt, *args):  # quieter console
        if not self.path.startswith("/api/jobs"):
            super().log_message(fmt, *args)

    # --- helpers
    def _allowed(self) -> bool:
        port = self.server.server_address[1]
        ok_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in ok_hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin.split("//", 1)[-1] in ok_hosts

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _deny(self):
        self._json({"error": "forbidden"}, 403)

    # --- routes
    def do_GET(self):
        if not self._allowed():
            return self._deny()
        url = urlparse(self.path)
        if url.path == "/":
            body = (STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; media-src 'self' blob:; "
                             "img-src 'self' data:; connect-src 'self'")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/status":
            from . import llm
            try:
                models = llm.available_models()
                ollama = True
            except Exception:
                models, ollama = [], False
            db = store.connect()
            self._json({
                "ollama": ollama,
                "llm": config.LLM_MODEL, "embed": config.EMBED_MODEL,
                "whisper": config.WHISPER_MODEL,
                "models_ready": ollama and all(any(m.split(":")[0] == x.split(":")[0] for m in models)
                                               for x in (config.LLM_MODEL, config.EMBED_MODEL)),
                "docs": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            })
        elif url.path == "/api/docs":
            rows = store.connect().execute(
                "SELECT id, kind, title, created_at, duration, language, summary FROM documents ORDER BY id DESC")
            self._json([dict(r) for r in rows])
        elif m := re.fullmatch(r"/api/docs/(\d+)", url.path):
            r = store.connect().execute("SELECT * FROM documents WHERE id = ?", (int(m[1]),)).fetchone()
            if not r:
                return self._json({"error": "not found"}, 404)
            d = dict(r)
            d["knowledge"] = json.loads(d["knowledge"]) if d["knowledge"] else None
            self._json(d)
        elif url.path == "/api/jobs":
            self._json(sorted(_jobs.values(), key=lambda j: -j["id"])[:20])
        else:
            self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        if not self._allowed():
            return self._deny()
        if m := re.fullmatch(r"/api/docs/(\d+)", urlparse(self.path).path):
            store.delete_document(store.connect(), int(m[1]))
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._allowed():
            return self._deny()
        url = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(url.query).items()}

        if url.path == "/api/upload":
            # Raw body upload: ?name=file.m4a — no multipart parsing needed.
            name = qs.get("name", "recording.webm")
            ext = Path(name).suffix.lower()
            if ext not in config.AUDIO_EXTS | config.TEXT_EXTS:
                return self._json({"error": f"unsupported file type {ext}"}, 400)
            UPLOADS.mkdir(parents=True, exist_ok=True)
            path = UPLOADS / _safe_name(name)
            path.write_bytes(self._body())
            job = _submit(name, _ingest_upload(path, qs.get("lang") or None, qs.get("title") or None))
            return self._json(job)

        if url.path == "/api/note":
            text = json.loads(self._body()).get("text", "").strip()
            if not text:
                return self._json({"error": "empty note"}, 400)
            job = _submit("note: " + text[:40], lambda db, log: service.ingest_note(db, text, log=log))
            return self._json(job)

        if url.path == "/api/transcribe":
            # Voice question: transcribe only, nothing is stored.
            from . import transcribe
            tmp = UPLOADS / _safe_name("question" + (Path(qs.get("name", "q.webm")).suffix or ".webm"))
            UPLOADS.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(self._body())
            try:
                with _whisper_lock:
                    text, meta = transcribe.transcribe(tmp, language=qs.get("lang") or None)
            finally:
                tmp.unlink(missing_ok=True)
            plain = re.sub(r"^\[\d\d:\d\d:\d\d\] ", "", text, flags=re.M).replace("\n", " ").strip()
            return self._json({"text": plain, "language": meta["language"]})

        if url.path == "/api/ask":
            req = json.loads(self._body())
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(obj):
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

            try:
                hits, stream = service.answer_stream(req["question"], req.get("history", [])[-6:])
                answer = ""
                for piece in stream:
                    answer += piece
                    emit({"t": piece})
                emit({"sources": service.cited_sources(hits, answer) if hits else []})
            except Exception as e:
                emit({"error": str(e)})
            self.close_connection = True
            return

        self._json({"error": "not found"}, 404)


def serve(port: int = 8777, open_browser: bool = True):
    config.ensure_dirs()
    threading.Thread(target=_worker, daemon=True).start()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        raise SystemExit(f"Port {port} is already in use. Try: brain ui --port {port + 1}")
    url = f"http://127.0.0.1:{port}"
    print(f"local-brain UI running at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
