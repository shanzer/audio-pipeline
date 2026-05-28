# Audio Pipeline Rearchitecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the audio pipeline so the iMac runs all pipeline/storage/webui code while the Mac Mini serves models only via a WhisperX FastAPI wrapper and Ollama.

**Architecture:** The iMac's `transcribe.py` POSTs audio to `http://banzai.local:8765/transcribe` and gets segments JSON back. `embed.py` and `summarize.py` point their Ollama client at `banzai.local:11434`. Audio lives on the iMac. iPhone uploads via a new `/upload` route on the iMac's Flask webui. The Mac Mini runs nothing but the WhisperX service and Ollama.

**Tech Stack:** Python 3.11, FastAPI + uvicorn (whisper-service on Mac Mini), Flask (webui on iMac), whisperx 3.3.4, ollama, psycopg2-binary, pgvector, watchdog, launchd

---

## File Map

### New files
| File | Machine | Purpose |
|---|---|---|
| `whisper-service/main.py` | Mac Mini | FastAPI app — WhisperX singleton + `/transcribe` endpoint |
| `whisper-service/requirements.txt` | Mac Mini | ML + FastAPI deps |
| `whisper-service/com.shanzer.whisper-service.plist` | Mac Mini | launchd service definition |
| `whisper-service/tests/conftest.py` | Mac Mini | Mocks ML deps so tests run without GPU |
| `whisper-service/tests/test_main.py` | Mac Mini | Health + transcribe endpoint tests |
| `pipeline/webui/app.py` | iMac | Merged webui (from audio-pipeline/) + `/upload` route |
| `pipeline/webui/templates/base.html` | iMac | Copied from `audio-pipeline/webui/templates/` |
| `pipeline/webui/templates/index.html` | iMac | Copied from `audio-pipeline/webui/templates/` |
| `pipeline/webui/templates/recording.html` | iMac | Copied from `audio-pipeline/webui/templates/` |
| `pipeline/webui/static/styles.css` | iMac | Copied from `audio-pipeline/webui/static/` |
| `pipeline/webui/requirements.txt` | iMac | Flask + psycopg2 |
| `pipeline/tests/test_intake_source.py` | iMac | Tests for `detect_source()` including `iphone_` prefix |
| `pipeline/tests/test_transcribe_client.py` | iMac | Tests for HTTP transcription client |
| `pipeline/tests/test_upload.py` | iMac | Tests for `/upload` endpoint |

### Modified files
| File | Change |
|---|---|
| `pipeline/config/settings.py` | Add `WHISPER_SERVICE_HOST/PORT`, `OLLAMA_HOST/PORT`; remove `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_BATCH_SIZE` |
| `pipeline/pipeline/stages/transcribe.py` | Replace local WhisperX with HTTP POST to whisper-service |
| `pipeline/pipeline/stages/embed.py` | Pass `host=` to `ollama.Client()` |
| `pipeline/pipeline/stages/summarize.py` | Same Ollama host change |
| `pipeline/pipeline/stages/intake.py` | Check `iphone_` prefix before regex patterns in `detect_source()` |
| `pipeline/run.py` | Remove webhook stage; add `osascript` notification helper |
| `pipeline/requirements.txt` | Remove all ML/whisperx deps; keep watchdog, psycopg2, ollama, requests, flask |

### Deleted files
- `pipeline/pipeline/stages/notify.py`
- `pipeline/pipeline/util/hmac_sign.py`
- `pipeline/tests/test_hmac.py`

---

## Pre-flight: Initialize git in pipeline/

- [ ] **Initialize git**

```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
git init
git add .
git commit -m "chore: initial commit — existing pipeline before rearchitecture"
```

---

## [MAC MINI] Task 1: WhisperX Service — FastAPI app

**Files:**
- Create: `whisper-service/main.py`
- Create: `whisper-service/requirements.txt`

- [ ] **Step 1: Create the whisper-service directory**

```bash
mkdir -p /Users/shanzer/whisper-service/tests
```

- [ ] **Step 2: Write requirements.txt**

```
fastapi>=0.110.0
huggingface_hub>=1.0.0
pyannote.audio>=3.0.0
python-dotenv>=1.0.0
soundfile>=0.12.0
torch>=2.11.0
torchaudio>=2.11.0
uvicorn[standard]>=0.29.0
whisperx==3.3.4
```

Write to `/Users/shanzer/whisper-service/requirements.txt`.

- [ ] **Step 3: Write main.py**

Write to `/Users/shanzer/whisper-service/main.py`:

```python
import logging
import os
import tempfile
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

import whisperx
from whisperx.diarize import DiarizationPipeline
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "8"))
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI()

_whisper_model = None
_align_models: dict = {}
_diarize_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        log.info("Loading WhisperX model %s on %s/%s", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
        _whisper_model = whisperx.load_model(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _whisper_model


def get_align_model(language: str):
    if language not in _align_models:
        log.info("Loading alignment model lang=%s", language)
        model, meta = whisperx.load_align_model(language_code=language, device=WHISPER_DEVICE)
        _align_models[language] = (model, meta)
    return _align_models[language]


def get_diarize_model():
    global _diarize_model
    if _diarize_model is None:
        log.info("Loading diarization model")
        _diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=WHISPER_DEVICE)
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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio = whisperx.load_audio(tmp_path)

        model = get_whisper_model()
        result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)
        language = result.get("language", "en")

        align_model, align_meta = get_align_model(language)
        result = whisperx.align(
            result["segments"], align_model, align_meta, audio, WHISPER_DEVICE,
            return_char_alignments=False,
        )

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
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WHISPER_SERVICE_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
```

- [ ] **Step 4: Install deps into the Mac Mini venv**

```bash
source ~/venvs/whisper-pipeline/bin/activate
pip install -r /Users/shanzer/whisper-service/requirements.txt
```

---

## [MAC MINI] Task 2: WhisperX Service — Tests

**Files:**
- Create: `whisper-service/tests/conftest.py`
- Create: `whisper-service/tests/test_main.py`

- [ ] **Step 1: Write conftest.py** (mocks ML deps so tests run without the models loaded)

Write to `/Users/shanzer/whisper-service/tests/conftest.py`:

```python
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Mock all heavy ML deps before any import of main
for _mod in [
    "torch", "torchaudio", "soundfile", "huggingface_hub",
    "whisperx", "whisperx.diarize",
]:
    sys.modules.setdefault(_mod, MagicMock())

# Ensure parent dir is on path so `from main import app` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 2: Write the tests**

Write to `/Users/shanzer/whisper-service/tests/test_main.py`:

```python
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app  # conftest has pre-mocked all ML deps


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_transcribe_returns_segments(client):
    fake_segments = [
        {"start": 0.0, "end": 2.0, "text": "hello world", "speaker": "SPEAKER_00", "words": []}
    ]
    fake_audio = MagicMock()

    with (
        patch("main.whisperx.load_audio", return_value=fake_audio),
        patch("main.get_whisper_model") as mock_model,
        patch("main.get_align_model") as mock_align,
        patch("main.get_diarize_model") as mock_diarize,
        patch("main.whisperx.align", return_value={"segments": fake_segments, "word_segments": []}),
        patch("main.whisperx.assign_word_speakers", return_value={"segments": fake_segments, "word_segments": []}),
    ):
        mock_model.return_value.transcribe.return_value = {
            "segments": fake_segments,
            "language": "en",
        }
        mock_align.return_value = (MagicMock(), MagicMock())
        mock_diarize.return_value.return_value = MagicMock()

        resp = client.post(
            "/transcribe",
            files={"file": ("test.wav", io.BytesIO(b"RIFF" + b"\x00" * 40), "audio/wav")},
            data={"min_speakers": "1", "max_speakers": "2"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "segments" in body
    assert "language" in body
    assert body["language"] == "en"
    assert len(body["segments"]) == 1


def test_transcribe_missing_file_returns_422(client):
    resp = client.post("/transcribe", data={"min_speakers": "1", "max_speakers": "2"})
    assert resp.status_code == 422


def test_transcribe_model_error_returns_500(client):
    with (
        patch("main.whisperx.load_audio", side_effect=RuntimeError("model exploded")),
    ):
        resp = client.post(
            "/transcribe",
            files={"file": ("test.wav", io.BytesIO(b"RIFF" + b"\x00" * 40), "audio/wav")},
            data={"min_speakers": "1", "max_speakers": "2"},
        )
    assert resp.status_code == 500
    assert "error" in resp.json()
```

- [ ] **Step 3: Run tests and verify they pass**

```bash
cd /Users/shanzer/whisper-service
source ~/venvs/whisper-pipeline/bin/activate
pip install pytest
pytest tests/ -v
```

Expected output:
```
tests/test_main.py::test_health_returns_ok PASSED
tests/test_main.py::test_transcribe_returns_segments PASSED
tests/test_main.py::test_transcribe_missing_file_returns_422 PASSED
tests/test_main.py::test_transcribe_model_error_returns_500 PASSED
4 passed in ...
```

- [ ] **Step 4: Commit**

```bash
cd /Users/shanzer/whisper-service
git init
git add .
git commit -m "feat: add WhisperX FastAPI service with health + transcribe endpoints"
```

---

## [MAC MINI] Task 3: WhisperX Service — launchd Setup

**Files:**
- Create: `whisper-service/com.shanzer.whisper-service.plist`

- [ ] **Step 1: Write the launchd plist**

Write to `/Users/shanzer/whisper-service/com.shanzer.whisper-service.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.shanzer.whisper-service</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/shanzer/venvs/whisper-pipeline/bin/python</string>
        <string>/Users/shanzer/whisper-service/main.py</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HF_TOKEN</key>
        <string>REPLACE_WITH_YOUR_HF_TOKEN</string>
        <key>WHISPER_DEVICE</key>
        <string>cpu</string>
        <key>WHISPER_COMPUTE_TYPE</key>
        <string>int8</string>
        <key>WHISPER_BATCH_SIZE</key>
        <string>8</string>
        <key>WHISPER_SERVICE_PORT</key>
        <string>8765</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/shanzer/whisper-service/service.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/shanzer/whisper-service/service.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/shanzer/whisper-service</string>
    <key>ProcessType</key>
    <string>Background</string>
    <key>Nice</key>
    <integer>10</integer>
</dict>
</plist>
```

- [ ] **Step 2: Fill in HF_TOKEN and install the service**

Replace `REPLACE_WITH_YOUR_HF_TOKEN` in the plist with the actual token from `pipeline/config/.env`.

```bash
cp /Users/shanzer/whisper-service/com.shanzer.whisper-service.plist \
   ~/Library/LaunchAgents/com.shanzer.whisper-service.plist

launchctl load ~/Library/LaunchAgents/com.shanzer.whisper-service.plist
```

- [ ] **Step 3: Verify the service started**

```bash
# Check it's running (takes 60+ seconds for model to load on first start)
curl http://localhost:8765/health
```

Expected: `{"ok":true}`

```bash
# Check logs if it fails to start
tail -50 /Users/shanzer/whisper-service/service.log
```

- [ ] **Step 4: Verify from iMac (run on iMac)**

```bash
curl http://banzai.local:8765/health
```

Expected: `{"ok":true}`

- [ ] **Step 5: Commit plist**

```bash
cd /Users/shanzer/whisper-service
git add com.shanzer.whisper-service.plist
git commit -m "chore: add launchd plist for whisper-service autostart"
```

---

## [IMAC] Task 4: Update settings.py

**Files:**
- Modify: `pipeline/config/settings.py`

- [ ] **Step 1: Rewrite settings.py**

Write to `pipeline/config/settings.py`:

```python
import os

RECORDINGS_INBOX = os.path.expanduser("~/Recordings/inbox")
RECORDINGS_ARCHIVE = os.path.expanduser("~/Recordings/archive")
RECORDINGS_FAILED = os.path.expanduser("~/Recordings/failed")
SQLITE_DB_PATH = os.path.expanduser("~/pipeline/db/jobs.db")

# WhisperX service (Mac Mini)
WHISPER_SERVICE_HOST = os.getenv("WHISPER_SERVICE_HOST", "banzai.local")
WHISPER_SERVICE_PORT = int(os.getenv("WHISPER_SERVICE_PORT", "8765"))

# Ollama (Mac Mini)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "banzai.local")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

EMBED_MODEL = "mxbai-embed-large"
EMBED_DIMENSIONS = 1024

LLM_MODEL = "qwen2.5:14b"

CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50

DEFAULT_MIN_SPEAKERS = 1
DEFAULT_MAX_SPEAKERS = 6
```

- [ ] **Step 2: Update .env.template to match**

Write to `pipeline/config/.env.template`:

```
POSTGRES_PASSWORD=
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=audio_pipeline
POSTGRES_USER=pipeline

WHISPER_SERVICE_HOST=banzai.local
WHISPER_SERVICE_PORT=8765

OLLAMA_HOST=banzai.local
OLLAMA_PORT=11434

WEBUI_SECRET_KEY=
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
source ~/venvs/pipeline/bin/activate  # or whichever venv is used on iMac
pytest tests/ -v
```

Expected: existing chunker tests pass, test_hmac.py is still present and passes (we delete it in Task 9).

- [ ] **Step 4: Commit**

```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
git add config/settings.py config/.env.template
git commit -m "feat: add remote service settings (whisper + ollama host/port)"
```

---

## [IMAC] Task 5: Rewrite transcribe.py as HTTP client

**Files:**
- Modify: `pipeline/pipeline/stages/transcribe.py`
- Create: `pipeline/tests/test_transcribe_client.py`

- [ ] **Step 1: Write the failing test**

Write to `pipeline/tests/test_transcribe_client.py`:

```python
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def wav_file(tmp_path):
    p = tmp_path / "test.wav"
    p.write_bytes(b"RIFF" + b"\x00" * 44)
    return str(p)


def test_run_whisperx_posts_to_service(wav_file):
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello", "speaker": "SPEAKER_00", "words": []}],
        "language": "en",
        "word_segments": [],
    }
    fake_response.raise_for_status = MagicMock()

    with patch("pipeline.stages.transcribe.requests.post", return_value=fake_response) as mock_post:
        from pipeline.stages.transcribe import run_whisperx
        result = run_whisperx(wav_file, min_speakers=1, max_speakers=3)

    assert result["language"] == "en"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "hello"

    call_args = mock_post.call_args
    url = call_args[0][0]
    assert "8765" in url
    assert "transcribe" in url
    assert call_args[1]["data"]["min_speakers"] == 1
    assert call_args[1]["data"]["max_speakers"] == 3
    assert call_args[1]["timeout"] == 3600


def test_run_whisperx_raises_on_http_error(wav_file):
    import requests as req
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = req.HTTPError("500 Server Error")

    with patch("pipeline.stages.transcribe.requests.post", return_value=fake_response):
        from pipeline.stages.transcribe import run_whisperx
        with pytest.raises(req.HTTPError):
            run_whisperx(wav_file)


def test_run_whisperx_uses_settings_host(wav_file):
    fake_response = MagicMock()
    fake_response.json.return_value = {"segments": [], "language": "en", "word_segments": []}
    fake_response.raise_for_status = MagicMock()

    with (
        patch("pipeline.stages.transcribe.requests.post", return_value=fake_response) as mock_post,
        patch("pipeline.stages.transcribe.WHISPER_SERVICE_HOST", "myhost"),
        patch("pipeline.stages.transcribe.WHISPER_SERVICE_PORT", 9999),
    ):
        from pipeline.stages.transcribe import run_whisperx
        run_whisperx(wav_file)

    url = mock_post.call_args[0][0]
    assert "myhost" in url
    assert "9999" in url
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
pytest tests/test_transcribe_client.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — transcribe.py still imports whisperx.

- [ ] **Step 3: Rewrite transcribe.py**

Write to `pipeline/pipeline/stages/transcribe.py`:

```python
import logging
import os
import time

import requests

from config.settings import (
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MAX_SPEAKERS,
    WHISPER_SERVICE_HOST,
    WHISPER_SERVICE_PORT,
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
    Raises requests.HTTPError on non-2xx. Raises requests.Timeout if service
    takes longer than 1 hour (shouldn't happen for any realistic recording).
    """
    url = f"http://{WHISPER_SERVICE_HOST}:{WHISPER_SERVICE_PORT}/transcribe"
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
```

Note: `**_kwargs` absorbs legacy `device=` and `compute_type=` args that callers may still pass. They are silently ignored since those concerns now live in the service.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transcribe_client.py -v
```

Expected:
```
tests/test_transcribe_client.py::test_run_whisperx_posts_to_service PASSED
tests/test_transcribe_client.py::test_run_whisperx_raises_on_http_error PASSED
tests/test_transcribe_client.py::test_run_whisperx_uses_settings_host PASSED
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages/transcribe.py tests/test_transcribe_client.py
git commit -m "feat: transcribe.py — replace local WhisperX with HTTP client to banzai.local:8765"
```

---

## [IMAC] Task 6: Update embed.py and summarize.py for remote Ollama

**Files:**
- Modify: `pipeline/pipeline/stages/embed.py`
- Modify: `pipeline/pipeline/stages/summarize.py`

- [ ] **Step 1: Update embed.py**

In `pipeline/pipeline/stages/embed.py`, change the import block and the client instantiation:

Old import line:
```python
from config.settings import EMBED_MODEL, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS
```

New:
```python
from config.settings import EMBED_MODEL, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS, OLLAMA_HOST, OLLAMA_PORT
```

Old client line (inside `chunk_and_embed`):
```python
    client = ollama.Client()
```

New:
```python
    client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
```

- [ ] **Step 2: Update summarize.py**

In `pipeline/pipeline/stages/summarize.py`, change the import block and client instantiation:

Old import line:
```python
from config.settings import LLM_MODEL
```

New:
```python
from config.settings import LLM_MODEL, OLLAMA_HOST, OLLAMA_PORT
```

Old `_call_llm` function:
```python
def _call_llm(prompt: str, model: str) -> str:
    client = ollama.Client()
    resp = client.generate(model=model, prompt=prompt, stream=False)
    return resp["response"].strip()
```

New:
```python
def _call_llm(prompt: str, model: str) -> str:
    client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
    resp = client.generate(model=model, prompt=prompt, stream=False)
    return resp["response"].strip()
```

- [ ] **Step 3: Run existing tests to verify nothing broke**

```bash
pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add pipeline/stages/embed.py pipeline/stages/summarize.py
git commit -m "feat: point ollama client at banzai.local:11434 for embed + summarize"
```

---

## [IMAC] Task 7: Update intake.py — iphone_ prefix detection

**Files:**
- Modify: `pipeline/pipeline/stages/intake.py`
- Create: `pipeline/tests/test_intake_source.py`

- [ ] **Step 1: Write the failing tests**

Write to `pipeline/tests/test_intake_source.py`:

```python
import pytest
from pipeline.stages.intake import detect_source


def test_iphone_prefix_wins_over_regex():
    assert detect_source("iphone_1748000000_memo.m4a") == "iphone"


def test_iphone_prefix_case_sensitive():
    # prefix check is exact; mixed case filenames without prefix use regex
    assert detect_source("iPhone_recording.m4a") != "iphone"  # no iphone_ prefix → regex/ambient


def test_metadata_source_overrides_prefix():
    assert detect_source("iphone_1748000000_memo.m4a", {"source": "zoom"}) == "zoom"


def test_zoom_pattern_detected():
    assert detect_source("zoom_meeting_GMT0800.mp4") == "zoom"


def test_meeting_pattern_detected():
    assert detect_source("quarterly_meeting_2026.wav") == "zoom"


def test_iphone_regex_pattern():
    assert detect_source("Voice Memo 42.m4a") == "iphone"


def test_ambient_fallback():
    assert detect_source("random_audio_file.wav") == "ambient"


def test_metadata_source_wins_over_all():
    assert detect_source("zoom_meeting.mp4", {"source": "ambient"}) == "ambient"
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_intake_source.py -v
```

Expected: `test_iphone_prefix_wins_over_regex` FAILS (prefix not yet checked).

- [ ] **Step 3: Update detect_source() in intake.py**

In `pipeline/pipeline/stages/intake.py`, update `detect_source`:

```python
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_intake_source.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages/intake.py tests/test_intake_source.py
git commit -m "feat: detect_source() recognizes iphone_ filename prefix from upload endpoint"
```

---

## [IMAC] Task 8: Update run.py — macOS notification, remove webhook

**Files:**
- Modify: `pipeline/run.py`

- [ ] **Step 1: Update the settings imports at the top of run.py**

`WHISPER_DEVICE` and `WHISPER_COMPUTE_TYPE` were removed from `settings.py` in Task 4. Remove them from the import block at the top of `run.py`.

Old import block (lines 23-31):
```python
from config.settings import (
    RECORDINGS_ARCHIVE,
    RECORDINGS_FAILED,
    RECORDINGS_INBOX,
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MAX_SPEAKERS,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)
```

New:
```python
from config.settings import (
    RECORDINGS_ARCHIVE,
    RECORDINGS_FAILED,
    RECORDINGS_INBOX,
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MAX_SPEAKERS,
)
```

- [ ] **Step 2: Add the notification helper and remove the webhook stage**

Open `pipeline/run.py` and make these changes:

**Add** `import subprocess` to the imports at the top (it's stdlib, no install needed).

**Add** this helper function after the `_move` function:

```python
def _notify(title: str, duration_sec: int) -> None:
    """Send macOS Notification Center alert. Non-fatal if osascript unavailable."""
    mins, secs = divmod(int(duration_sec or 0), 60)
    duration_str = f"{mins}m {secs:02d}s"
    safe_title = (title or "Recording processed").replace('"', "'").replace("\\", "")
    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{duration_str}" with title "Audio Pipeline" subtitle "{safe_title}"',
            ],
            timeout=5,
            check=False,
        )
    except Exception:
        pass
```

**Replace** the entire Stage 6 block in `process_job()`:

Old (lines ~124-138):
```python
        # Stage 6: Webhook
        log.info("[job=%d] Stage: notify", job_id)
        speaker_labels = list({s.get("speaker") for s in segments if s.get("speaker")})
        payload = build_payload(
            recording_id=recording_id,
            job=job,
            summary=summary,
            speakers=speaker_labels,
            recorded_at=str(Path(file_path).stat().st_mtime),
            duration_sec=audio_meta.get("duration_sec", 0),
            source=job["source"] if hasattr(job, "__getitem__") else job.source,
        )
        try:
            send_webhook(payload, os.environ.get("N8N_WEBHOOK_URL", ""), os.environ.get("HMAC_SECRET", ""))
        except Exception as e:
            log.warning("[job=%d] Webhook failed (non-fatal): %s", job_id, e)
```

New:
```python
        # Stage 6: Notify
        _notify(
            title=summary.get("title", ""),
            duration_sec=audio_meta.get("duration_sec", 0),
        )
```

**Remove** the notify/hmac imports from the lazy import block inside `process_job()`:

Remove this line:
```python
    from pipeline.stages.notify import build_payload, send_webhook
```

- [ ] **Step 3: Verify the pipeline runs without errors**

```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
python -c "from run import process_job, worker_loop, main; print('imports ok')"
```

Expected: `imports ok`

- [ ] **Step 4: Commit**

```bash
git add run.py
git commit -m "feat: replace n8n webhook with macOS Notification Center alert on job complete"
```

---

## [IMAC] Task 9: Webui — merge audio-pipeline/webui + add /upload

**Files:**
- Create: `pipeline/webui/app.py`
- Create: `pipeline/webui/requirements.txt`
- Create: `pipeline/webui/templates/` (copied)
- Create: `pipeline/webui/static/` (copied)
- Create: `pipeline/tests/test_upload.py`

- [ ] **Step 1: Copy templates and static assets**

```bash
cp -r /Users/shanzer/src/audio-pipeline/audio-pipeline/webui/templates \
      /Users/shanzer/src/audio-pipeline/pipeline/webui/templates

cp -r /Users/shanzer/src/audio-pipeline/audio-pipeline/webui/static \
      /Users/shanzer/src/audio-pipeline/pipeline/webui/static
```

- [ ] **Step 2: Write the webui requirements.txt**

Write to `pipeline/webui/requirements.txt`:

```
flask>=3.0.0
gunicorn>=22.0.0
psycopg2-binary>=2.9.0
werkzeug>=3.0.0
```

- [ ] **Step 3: Write the failing upload tests**

Write to `pipeline/tests/test_upload.py`:

```python
import io
import os
import pytest
from pathlib import Path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDINGS_INBOX", str(tmp_path))
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    # Import app after env is set
    import importlib
    import pipeline.webui.app as webui_module
    importlib.reload(webui_module)
    webui_module.app.config["TESTING"] = True
    with webui_module.app.test_client() as c:
        yield c, tmp_path


def test_upload_no_file_field(client):
    c, _ = client
    resp = c.post("/upload")
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False
    assert "no file" in resp.get_json()["error"]


def test_upload_empty_filename(client):
    c, _ = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"data"), "")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_upload_bad_extension(client):
    c, _ = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"data"), "audio.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.get_json()["error"]


def test_upload_success_wav(client):
    c, inbox = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"RIFF" + b"\x00" * 40), "memo.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    filename = body["filename"]
    assert filename.startswith("iphone_")
    assert filename.endswith(".wav")
    assert (inbox / filename).exists()


def test_upload_success_m4a(client):
    c, inbox = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"\x00" * 100), "voice memo.m4a")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["filename"].startswith("iphone_")
    assert body["filename"].endswith(".m4a")
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/test_upload.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.webui'`

- [ ] **Step 5: Write webui/app.py**

Write to `pipeline/webui/app.py` — this is the existing `audio-pipeline/webui/app.py` with the `/upload` route added and Postgres host defaulting to `127.0.0.1`:

```python
#!/usr/bin/env python3
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras
from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.getenv("WEBUI_SECRET_KEY", "local-audio-pipeline-webui")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

ALLOWED_EXTENSIONS = {".m4a", ".mp4", ".wav", ".mp3", ".aac"}


def db_connect():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "audio_pipeline"),
        user=os.getenv("POSTGRES_USER", "pipeline"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    psycopg2.extras.register_uuid(conn_or_curs=conn)
    return conn


def query_all(sql, params=None):
    with db_connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            return list(cur.fetchall())


def query_one(sql, params=None):
    rows = query_all(sql, params)
    return rows[0] if rows else None


def normalize_json(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def fmt_duration(seconds):
    if seconds is None:
        return "Unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def fmt_datetime(value):
    if not value:
        return "Unknown"
    if isinstance(value, str):
        return value
    local_value = value.astimezone() if value.tzinfo else value.replace(tzinfo=timezone.utc).astimezone()
    return local_value.strftime("%b %-d, %Y %-I:%M %p")


def timestamp(seconds):
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def safe_filename_stem(filename):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", filename or "recording").strip("-")
    return cleaned or "recording"


def summary_sections(summary):
    if not summary:
        return []
    return [
        ("Topics", normalize_json(summary.get("topics"), [])),
        ("Decisions", normalize_json(summary.get("decisions"), [])),
        ("Action items", normalize_json(summary.get("action_items"), [])),
        ("Risks", normalize_json(summary.get("risks"), [])),
    ]


def format_summary_markdown(recording, summary):
    title = summary.get("title") if summary else recording["filename"]
    lines = [
        f"# {title or recording['filename']}",
        "",
        f"- Recording: {recording['filename']}",
        f"- Source: {recording.get('source') or 'unknown'}",
        f"- Recorded: {fmt_datetime(recording.get('recorded_at'))}",
        f"- Duration: {fmt_duration(recording.get('duration_sec'))}",
        "",
    ]
    for label, values in summary_sections(summary):
        lines.append(f"## {label}")
        lines.extend(f"- {item}" for item in values) if values else lines.append("- None recorded")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def format_transcript(recording, segments):
    lines = [
        f"Transcript: {recording['filename']}",
        f"Recorded: {fmt_datetime(recording.get('recorded_at'))}",
        f"Duration: {fmt_duration(recording.get('duration_sec'))}",
        "",
    ]
    for segment in segments:
        speaker = segment.get("speaker_label") or "Unknown"
        start = timestamp(segment.get("start_time"))
        end = timestamp(segment.get("end_time"))
        text = (segment.get("text") or "").strip()
        lines.append(f"[{start} - {end}] {speaker}: {text}")
    return "\n".join(lines).strip() + "\n"


@app.template_filter("duration")
def duration_filter(value):
    return fmt_duration(value)


@app.template_filter("datetime")
def datetime_filter(value):
    return fmt_datetime(value)


@app.template_filter("timestamp")
def timestamp_filter(value):
    return timestamp(value)


# ── Upload endpoint (iPhone intake) ──────────────────────────────────────────

@app.post("/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no file field in request"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": f"unsupported file type: {ext}"}), 400

    safe_name = secure_filename(f.filename)
    ts = int(time.time())
    filename = f"iphone_{ts}_{safe_name}"

    inbox = Path(os.getenv("RECORDINGS_INBOX", os.path.expanduser("~/Recordings/inbox")))
    inbox.mkdir(parents=True, exist_ok=True)
    f.save(str(inbox / filename))

    return jsonify({"ok": True, "filename": filename})


# ── Recording views ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    recordings = query_all(
        """
        SELECT
            r.id, r.filename, r.source, r.duration_sec, r.recorded_at,
            r.processed_at, r.status, r.error, r.speaker_count,
            latest_summary.title AS summary_title,
            COALESCE(segment_stats.segment_count, 0) AS segment_count,
            COALESCE(chunk_stats.chunk_count, 0) AS chunk_count
        FROM recordings r
        LEFT JOIN LATERAL (
            SELECT title FROM summaries WHERE recording_id = r.id
            ORDER BY created_at DESC LIMIT 1
        ) latest_summary ON true
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS segment_count FROM segments WHERE recording_id = r.id
        ) segment_stats ON true
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS chunk_count FROM chunks WHERE recording_id = r.id
        ) chunk_stats ON true
        ORDER BY r.recorded_at DESC, r.processed_at DESC NULLS LAST
        """
    )
    totals = {
        "recordings": len(recordings),
        "hours": sum((row.get("duration_sec") or 0) for row in recordings) / 3600,
        "segments": sum(row.get("segment_count") or 0 for row in recordings),
    }
    return render_template("index.html", recordings=recordings, totals=totals)


def load_recording_bundle(recording_id):
    recording = query_one("SELECT * FROM recordings WHERE id = %s", (recording_id,))
    if not recording:
        abort(404)

    summary = query_one(
        "SELECT * FROM summaries WHERE recording_id = %s ORDER BY created_at DESC LIMIT 1",
        (recording_id,),
    )
    if summary:
        for key in ("topics", "decisions", "action_items", "risks"):
            summary[key] = normalize_json(summary.get(key), [])
        summary["raw_json"] = normalize_json(summary.get("raw_json"), {})

    speakers = query_all(
        "SELECT diarization_label, resolved_name, channel FROM speakers "
        "WHERE recording_id = %s ORDER BY diarization_label",
        (recording_id,),
    )
    segments = query_all(
        "SELECT segment_index, speaker_label, start_time, end_time, text "
        "FROM segments WHERE recording_id = %s ORDER BY segment_index",
        (recording_id,),
    )
    return recording, summary, speakers, segments


@app.route("/recordings/<uuid:recording_id>")
def recording_detail(recording_id):
    recording, summary, speakers, segments = load_recording_bundle(recording_id)
    transcript_text = format_transcript(recording, segments)
    return render_template(
        "recording.html",
        recording=recording,
        summary=summary,
        speakers=speakers,
        segments=segments,
        transcript_text=transcript_text,
    )


@app.post("/recordings/<uuid:recording_id>/delete")
def delete_recording(recording_id):
    recording = query_one("SELECT filename FROM recordings WHERE id = %s", (recording_id,))
    if not recording:
        abort(404)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM recordings WHERE id = %s", (recording_id,))
    flash(f"Deleted {recording['filename']}.")
    return redirect(url_for("index"))


@app.route("/recordings/<uuid:recording_id>/download/<kind>")
def download(recording_id, kind):
    recording, summary, _speakers, segments = load_recording_bundle(recording_id)
    basename = safe_filename_stem(recording["filename"])

    if kind == "summary":
        body = format_summary_markdown(recording, summary)
        content_type = "text/markdown; charset=utf-8"
        filename = f"{basename}-summary.md"
    elif kind == "summary-json":
        body = json.dumps(summary.get("raw_json") if summary else {}, indent=2, default=str) + "\n"
        content_type = "application/json"
        filename = f"{basename}-summary.json"
    elif kind == "transcript":
        body = format_transcript(recording, segments)
        content_type = "text/plain; charset=utf-8"
        filename = f"{basename}-transcript.txt"
    else:
        abort(404)

    return Response(
        body,
        content_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.context_processor
def template_helpers():
    return {"summary_sections": summary_sections, "now": datetime.now}
```

- [ ] **Step 6: Run upload tests**

```bash
pytest tests/test_upload.py -v
```

Expected:
```
tests/test_upload.py::test_upload_no_file_field PASSED
tests/test_upload.py::test_upload_empty_filename PASSED
tests/test_upload.py::test_upload_bad_extension PASSED
tests/test_upload.py::test_upload_success_wav PASSED
tests/test_upload.py::test_upload_success_m4a PASSED
5 passed
```

- [ ] **Step 7: Commit**

```bash
git add pipeline/webui/ tests/test_upload.py
git commit -m "feat: add webui with /upload endpoint for iPhone audio intake"
```

---

## [IMAC] Task 10: Cleanup — delete retired files, update requirements

**Files:**
- Delete: `pipeline/pipeline/stages/notify.py`
- Delete: `pipeline/pipeline/util/hmac_sign.py`
- Delete: `pipeline/tests/test_hmac.py`
- Modify: `pipeline/requirements.txt`

- [ ] **Step 1: Delete retired files**

```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
rm pipeline/stages/notify.py
rm pipeline/util/hmac_sign.py
rm tests/test_hmac.py
```

- [ ] **Step 2: Verify no remaining imports of deleted modules**

```bash
grep -r "from pipeline.stages.notify\|from pipeline.util.hmac_sign\|import hmac_sign\|import notify" pipeline/ tests/
```

Expected: no output. If any hits appear, remove those imports.

- [ ] **Step 3: Write new pipeline/requirements.txt**

The old requirements.txt has the full ML stack pinned. The iMac pipeline no longer needs any of that. Write a clean requirements file:

Write to `pipeline/requirements.txt`:

```
flask>=3.0.0
gunicorn>=22.0.0
ollama>=0.6.0
pgvector>=0.4.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
requests>=2.33.0
watchdog>=6.0.0
werkzeug>=3.0.0
```

- [ ] **Step 4: Create a fresh venv on the iMac and install**

```bash
python3.11 -m venv ~/venvs/pipeline
source ~/venvs/pipeline/bin/activate
pip install -r /Users/shanzer/src/audio-pipeline/pipeline/requirements.txt
```

- [ ] **Step 5: Run the full test suite with the new venv**

```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
source ~/venvs/pipeline/bin/activate
pytest tests/ -v
```

Expected: all tests pass. Verify specifically:
- `tests/test_chunker.py` — still passes
- `tests/test_transcribe_client.py` — passes
- `tests/test_intake_source.py` — passes
- `tests/test_upload.py` — passes
- `tests/test_hmac.py` — gone, not in output

- [ ] **Step 6: Commit**

```bash
git add -u  # stages deletions
git add requirements.txt
git commit -m "chore: remove ML deps from iMac pipeline; delete notify/hmac modules"
```

- [ ] **Step 7: Note audio-pipeline/ retirement**

The `audio-pipeline/` directory is superseded. Do not delete it yet — confirm the new pipeline is running cleanly in production first. Once stable, delete with:

```bash
rm -rf /Users/shanzer/src/audio-pipeline/audio-pipeline/
```

---

## Task 11: Smoke Test — end-to-end verification

- [ ] **Step 1: Verify the whisper service is reachable from the iMac**

On the iMac:
```bash
curl http://banzai.local:8765/health
```
Expected: `{"ok":true}`

- [ ] **Step 2: Verify Ollama is reachable from the iMac**

```bash
curl http://banzai.local:11434/api/tags
```
Expected: JSON listing available models including `mxbai-embed-large` and `qwen2.5:14b`.

- [ ] **Step 3: Drop a test audio file in the inbox and watch the pipeline**

```bash
# In one terminal — start the pipeline
cd /Users/shanzer/src/audio-pipeline/pipeline
source ~/venvs/pipeline/bin/activate
python run.py

# In another terminal — copy a short WAV into the inbox
cp /path/to/any/short/audio.wav ~/Recordings/inbox/test_smoke.wav
```

Watch the logs for: `Enqueued → Stage: intake → Stage: transcribe → Stage: embed → Stage: summarize → Stage: store → Complete`.

- [ ] **Step 4: Verify the Notification Center alert fires**

After the job completes, a macOS notification should appear in the top-right corner of the iMac with title "Audio Pipeline" and the recording title as subtitle.

- [ ] **Step 5: Verify the webui shows the recording**

Start the webui:
```bash
cd /Users/shanzer/src/audio-pipeline/pipeline
source ~/venvs/pipeline/bin/activate
POSTGRES_PASSWORD=yourpassword python -m flask --app webui/app.py run --port 5000
```

Open `http://localhost:5000` — the test recording should appear with title, duration, and segments count.

- [ ] **Step 6: Verify iPhone upload path**

From the iMac terminal, simulate an iPhone upload:
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@/path/to/test.m4a"
```

Expected: `{"ok":true,"filename":"iphone_<timestamp>_test.m4a"}`

Verify the file appears in `~/Recordings/inbox/` and gets picked up by the pipeline watcher.
