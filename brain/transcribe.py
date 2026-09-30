import sys
import time
from pathlib import Path

from . import config


def _model_path() -> Path:
    return config.MODELS_DIR / f"faster-whisper-{config.WHISPER_MODEL}"


def download_model() -> Path:
    """One-time download of the Whisper weights (the only step that uses the network)."""
    from faster_whisper.utils import download_model as fw_download

    target = _model_path()
    if (target / "model.bin").exists():
        return target
    print(f"Downloading Whisper model '{config.WHISPER_MODEL}' to {target} ...")
    fw_download(config.WHISPER_MODEL, output_dir=str(target))
    return target


_model = None


def _load():
    global _model
    if _model is None:
        path = _model_path()
        if not (path / "model.bin").exists():
            sys.exit(f"Whisper model not found at {path}. Run `brain setup` first.")
        from faster_whisper import WhisperModel

        _model = WhisperModel(str(path), device=config.WHISPER_DEVICE,
                              compute_type=config.WHISPER_COMPUTE)
    return _model


def _ts(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(audio: Path, language: str | None = None) -> tuple[str, dict]:
    """Transcribe an audio/video file. Returns (timestamped text, info)."""
    model = _load()
    started = time.time()
    segments, info = model.transcribe(str(audio), language=language, vad_filter=True,
                                      beam_size=5)
    lines = []
    for seg in segments:
        line = f"[{_ts(seg.start)}] {seg.text.strip()}"
        lines.append(line)
        print(f"  {line}", file=sys.stderr, flush=True)
    meta = {
        "language": info.language,
        "duration": round(info.duration, 1),
        "elapsed": round(time.time() - started, 1),
    }
    return "\n".join(lines), meta
