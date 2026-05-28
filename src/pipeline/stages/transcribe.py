import logging
import os
import time

import requests

from config.settings import (
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MAX_SPEAKERS,
    WHISPER_SERVICE_URL,
)

log = logging.getLogger(__name__)


def run_whisperx(
    audio_path: str,
    min_speakers: int = DEFAULT_MIN_SPEAKERS,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    **_kwargs,
) -> dict:
    """
    POST audio file to WhisperX service on Mac Mini.
    Returns {segments, language, word_segments}.
    Raises requests.HTTPError on non-2xx.
    **_kwargs absorbs legacy device=/compute_type= args from callers.
    """
    url = f"{WHISPER_SERVICE_URL}/transcribe"
    t0 = time.perf_counter()

    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"min_speakers": min_speakers, "max_speakers": max_speakers},
            timeout=3600,
        )

    resp.raise_for_status()
    result = resp.json()

    log.info(
        "Transcription complete: %.2fs, %d segments, lang=%s",
        time.perf_counter() - t0,
        len(result.get("segments", [])),
        result.get("language", "?"),
    )

    return result
