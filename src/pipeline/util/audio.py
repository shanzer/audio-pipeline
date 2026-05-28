import shutil
import subprocess
import json
from pathlib import Path


def _ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe not found on PATH")
    return path


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH")
    return path


def probe_audio(file_path: str) -> dict:
    """Run ffprobe and return stream/format metadata dict."""
    cmd = [
        _ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def get_audio_info(file_path: str) -> dict:
    """Return {duration_sec, channels, sample_rate, format} for an audio file."""
    data = probe_audio(file_path)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise ValueError(f"No audio streams found in {file_path}")
    stream = streams[0]
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or stream.get("duration", 0))
    return {
        "duration_sec": int(duration),
        "channels": int(stream.get("channels", 1)),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "format": fmt.get("format_name", "unknown"),
    }


def normalize_audio(file_path: str, output_path: str) -> str:
    """Convert audio to 16kHz WAV, preserving channel count."""
    info = get_audio_info(file_path)
    channels = info["channels"]
    cmd = [
        _ffmpeg_path(),
        "-y",
        "-i", str(file_path),
        "-ar", "16000",
        "-ac", str(channels),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalize failed: {result.stderr[-500:]}")
    return str(output_path)
