# local-brain

A private second brain that runs entirely on this Mac. Nothing is sent over the network.

```
audio/video ──► faster-whisper (large-v3-turbo) ──► transcript ─┐
text / notes ───────────────────────────────────────────────────┤
                                                                ▼
             qwen3:8b via Ollama ──► title, summary, key points, decisions,
                                     action items, facts, people, topics
                                                                ▼
             embeddinggemma ──► vectors ──► SQLite (data/brain.db) + FTS5
                                                                ▼
             brain ask ──► hybrid search (vector + keyword) ──► qwen3 answers with [#doc] citations
```

## Privacy guarantees

- **Network is blocked in-process.** Every command except `brain setup` installs a socket guard
  (`brain/offline.py`) that refuses any connection or DNS lookup that isn't loopback. If some
  library ever tried to phone home, it would crash instead of leaking.
- Hugging Face offline mode is forced (`HF_HUB_OFFLINE=1`), so the Whisper model is only loaded from `data/models/`.
- The LLM and the embeddings run in Ollama, which listens on `127.0.0.1:11434` only.
- All data lives in `data/`: the SQLite DB, the transcripts and the Whisper weights.

## Quick start: one command

```bash
git clone https://github.com/bszymi/local-brain && ./local-brain/start.sh
```

On the first run, `start.sh`:
1. installs Python 3.12 and Ollama with Homebrew, but only if they're missing (macOS)
2. creates the Python environment in `.venv/`
3. starts Ollama, listening on `127.0.0.1` only
4. downloads the models, about 7.5 GB (the only step that uses the internet)
5. opens the app at http://127.0.0.1:8777

After that, `./start.sh` from the `local-brain` folder (or double-clicking **`Local Brain.command`** in Finder)
starts it in a couple of seconds. **Ctrl-C** stops everything, including Ollama if the script started it.
Extra arguments go to the web UI, e.g. `./start.sh --port 9000`.

Requirements: macOS with [Homebrew](https://brew.sh) (Apple Silicon recommended), about 10 GB of free disk and 16 GB of RAM.
On Linux, install Ollama first (`curl -fsSL https://ollama.com/install.sh | sh`) and Python 3.10–3.13; the rest works the same.

To update later: `git pull && ./start.sh`. Dependencies are reinstalled automatically when they change.

## The interface

Double-click **`Local Brain.command`** in Finder, or run `brain ui`. It opens http://127.0.0.1:8777 where you can:

- **Record**: press the big red button, speak, and press it again. The recording is transcribed, summarised and saved.
- **Drop files**: drag audio, video or text files anywhere on the page.
- **Quick note**: type something and save it.
- **Ask**: type a question or press the small mic to ask by voice. Answers cite the memories they came from. 🔊 reads the answer aloud with on-device voices only.
- **Memories**: browse everything, with summary, decisions, action items and the full transcript for each.

The first time you record, the browser will ask for microphone permission. Allow it.

## Command line

The CLI lives at `.venv/bin/brain`. Tip: `alias brain=~/local-brain/.venv/bin/brain`. Ollama must be running (`./start.sh` takes care of that).

```bash
brain add ~/Recordings/meeting.m4a        # transcribe + extract knowledge + index
brain add ~/Voice\ Memos/ notes/*.md      # folders and text files work too
brain add call.mp3 -l pl                  # force the language (it is auto-detected otherwise)
brain note "Passport expires March 2028"  # quick note (or pipe text into `brain note`)

brain ask "What did we decide about the database?"
brain ask                                 # interactive chat with your brain
brain search mortgage interest            # raw hybrid search

brain list | brain show 3 [--full] | brain delete 3
brain transcribe file.m4a -o out.txt      # transcription only, not stored
```

## Configuration (env vars)

| var | default | notes |
|---|---|---|
| `BRAIN_HOME` | `./data` | where the DB, transcripts and models live |
| `BRAIN_LLM` | `qwen3:8b` | any Ollama chat model |
| `BRAIN_EMBED` | `embeddinggemma` | multilingual embeddings. If you change it, re-add your docs |
| `BRAIN_WHISPER` | `large-v3-turbo` | e.g. `small`, `medium`, `large-v3`. Re-run `brain setup` after changing |
| `BRAIN_LLM_CTX` | `16384` | context window |

Speed on an M3 Pro: roughly 1.5× real time for English audio on CPU (int8). The first run is slower while the model loads.
