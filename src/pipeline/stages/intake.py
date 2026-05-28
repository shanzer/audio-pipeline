import re
from pathlib import Path

from pipeline.util.audio import get_audio_info, normalize_audio


def validate_audio(file_path: str) -> dict:
    """
    Validate that file_path is readable audio using ffprobe.
    Returns {duration_sec, channels, sample_rate, format}.
    Raises ValueError if not valid audio.
    """
    info = get_audio_info(file_path)
    if info["duration_sec"] == 0:
        raise ValueError(f"Audio duration is 0 for {file_path}")
    return info


def normalize_audio_file(file_path: str, output_path: str) -> str:
    """
    Normalize audio to 16kHz WAV, preserving channel count.
    Returns path to normalized file.
    """
    return normalize_audio(file_path, output_path)


_ZOOM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"zoom", r"GMT\d{4}", r"meeting",
]]
_IPHONE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"voice.?memo", r"iphone", r"recording\s*\d+",
]]


def detect_source(file_path: str, metadata: dict | None = None) -> str:
    """Infer recording source from filename and metadata."""
    name = Path(file_path).name
    meta = metadata or {}

    if meta.get("source"):
        return meta["source"]

    if name.startswith("iphone_"):
        return "iphone"

    for pat in _ZOOM_PATTERNS:
        if pat.search(name):
            return "zoom"
    for pat in _IPHONE_PATTERNS:
        if pat.search(name):
            return "iphone"

    return "ambient"
