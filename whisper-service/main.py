import logging
import os
import pathlib
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

# ── Compat patches — must run before whisperx import ─────────────────────────
import soundfile as sf
import torch as _torch
import torchaudio
import huggingface_hub as _hfhub

if not hasattr(torchaudio, "AudioMetaData"):
    @dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str
    torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "info"):
    def _torchaudio_info(path, backend=None):
        i = sf.info(path)
        return torchaudio.AudioMetaData(i.samplerate, i.frames, i.channels, 16, "PCM_S")
    torchaudio.info = _torchaudio_info

# whisperx ships checkpoints that require weights_only=False
_orig_torch_load = _torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)
_torch.load = _patched_torch_load

_orig_hf_hub_download = _hfhub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_hf_hub_download(*args, **kwargs)
_hfhub.hf_hub_download = _patched_hf_hub_download
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

import mlx_whisper
import whisperx
from whisperx.diarize import DiarizationPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# mlx-community/whisper-large-v3-turbo: near large-v3 accuracy, ~8x faster on Apple Silicon
WHISPER_MODEL  = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "mps")   # alignment + diarization models
HF_TOKEN       = os.getenv("HF_TOKEN", "")

_align_models: dict = {}
_diarize_model = None
_model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_diarize_model()   # pre-warm; mlx model loads on first transcribe call
    yield


app = FastAPI(lifespan=lifespan)


def get_align_model(language: str):
    with _model_lock:
        if language not in _align_models:
            log.info("Loading alignment model lang=%s", language)
            model, meta = whisperx.load_align_model(language_code=language, device=WHISPER_DEVICE)
            _align_models[language] = (model, meta)
        return _align_models[language]


def get_diarize_model():
    global _diarize_model
    with _model_lock:
        if _diarize_model is None:
            log.info("Loading diarization model")
            _diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN or None, device=WHISPER_DEVICE)
    return _diarize_model


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    min_speakers: int = Form(1),
    max_speakers: int = Form(6),
):
    if not (1 <= min_speakers <= max_speakers <= 20):
        raise HTTPException(
            status_code=422,
            detail=f"invalid speaker range: min_speakers={min_speakers}, max_speakers={max_speakers} (must satisfy 1 ≤ min ≤ max ≤ 20)",
        )

    tmp_path = None
    upload_suffix = pathlib.Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=upload_suffix, delete=False) as tmp:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Load audio for alignment/diarization (numpy float32, 16 kHz)
        audio = whisperx.load_audio(tmp_path)

        # Transcribe with mlx-whisper (uses M-series GPU + Neural Engine)
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=WHISPER_MODEL,
            word_timestamps=True,
        )
        language = result.get("language") or "en"
        if not result.get("language"):
            log.warning("mlx_whisper returned no language; defaulting to 'en'")

        # Word-level alignment
        align_model, align_meta = get_align_model(language)
        result = whisperx.align(
            result["segments"], align_model, align_meta, audio, WHISPER_DEVICE,
            return_char_alignments=False,
        )

        # Speaker diarization
        diarize_model = get_diarize_model()
        diarize_df = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_df, result)

        return {
            "segments": result["segments"],
            "language": language,
            "word_segments": result.get("word_segments", []),
        }
    except Exception as exc:
        log.exception("Transcription failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        if tmp_path:
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WHISPER_SERVICE_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
