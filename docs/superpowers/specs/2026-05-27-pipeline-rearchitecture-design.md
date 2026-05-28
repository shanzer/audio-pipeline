# Audio Pipeline Rearchitecture Design

**Date:** 2026-05-27
**Status:** Approved

---

## 1. Problem Statement

The pipeline was designed to run entirely on the Mac Mini (banzai.local). The new architecture splits responsibilities: the iMac runs the pipeline code and owns all storage; the Mac Mini runs models only. A second input path (iPhone) must be added alongside the existing Mac audio capture path.

---

## 2. Architecture

### Machines

| Machine | Role |
|---|---|
| **Intel iMac** | Pipeline orchestration, storage, webui, Postgres. Better raw CPU — handles ffmpeg, queue, all non-ML work. |
| **Mac Mini (banzai.local)** | Models only: WhisperX service + Ollama. Nothing else runs here. |
| **iPhone** | iOS Shortcut records audio and POSTs to iMac upload endpoint. |

### Data flow

```
[iPhone]
  iOS Shortcut → POST http://imac.local/upload
                      ↓
[iMac] ~/Recordings/inbox/  ←  Audio Hijack (Loopback, Zoom/Teams)
         ↓ watchdog watcher
         ↓ SQLite job queue
         ↓ ffmpeg: validate + normalize (WAV 16kHz)
         ↓ POST audio → banzai.local:8765/transcribe → segments JSON
         ↓ POST chunks → banzai.local:11434 (Ollama embed)
         ↓ POST transcript → banzai.local:11434 (Ollama summarize)
         ↓ write to Postgres + pgvector (Docker, localhost)
         ↓ macOS Notification Center (osascript)
         ↓ move file to ~/Recordings/archive/
```

---

## 3. Input Paths

### Path 1 — Mac audio capture (Loopback + Audio Hijack)
- Loopback creates a virtual device combining mic (ch0) and system audio (ch1)
- Audio Hijack captures from the virtual device and saves to `~/Recordings/inbox/`
- No code changes required for this path — files land in inbox as before

### Path 2 — iPhone
- iOS Shortcut: Record Audio action → POST multipart file to `http://imac.local/upload`
- Upload endpoint saves file to `~/Recordings/inbox/` with `iphone_` filename prefix
- The prefix ensures `detect_source()` classifies it as `iphone` without relying on fragile filename pattern matching
- Shortcut receives `{ok: true, filename: "..."}` JSON response to confirm delivery

---

## 4. Components

### 4a. WhisperX Service (new — lives on Mac Mini)

Small FastAPI app. Loads WhisperX model once as a process-level singleton. Exposes one endpoint.

**Endpoint:** `POST /transcribe`
- Body: multipart form — `file` (audio), `min_speakers` (int), `max_speakers` (int)
- Response: `{segments: [...], language: "en", word_segments: [...]}`
- Same segment schema as current WhisperX output — no changes needed in pipeline stages

**Operation:**
- Runs as a launchd service on Mac Mini
- Logs to `~/whisper-service/service.log`
- No authentication (LAN-only, not exposed externally)

### 4b. Upload Endpoint (new route on iMac Flask app)

`POST /upload`
- Accepts `multipart/form-data` with field `file`
- Validates extension (`.m4a`, `.mp4`, `.wav`, `.mp3`, `.aac`)
- Saves to `~/Recordings/inbox/iphone_<timestamp>_<original_name>`
- Returns `{"ok": true, "filename": "..."}` on success, `{"ok": false, "error": "..."}` on failure
- No authentication (LAN-only)

### 4c. macOS Notification (replaces notify.py)

Called at end of `process_job()` on success:

```python
subprocess.run([
    "osascript", "-e",
    f'display notification "{title}" with title "Pipeline" subtitle "{duration}"'
])
```

Non-fatal — failure to notify does not fail the job.

### 4d. Webui (merged from audio-pipeline/)

The `audio-pipeline/webui/app.py` is more complete than anything currently in `pipeline/`. Merge it into `pipeline/webui/app.py` and add the `/upload` route. The webui runs on the iMac alongside the pipeline.

---

## 5. Settings Changes

`config/settings.py` additions:

```python
WHISPER_SERVICE_HOST = os.getenv("WHISPER_SERVICE_HOST", "banzai.local")
WHISPER_SERVICE_PORT = int(os.getenv("WHISPER_SERVICE_PORT", 8765))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "banzai.local")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
```

`config/settings.py` removals:
- `WHISPER_DEVICE` — belongs to the Mac Mini service, not the pipeline
- `WHISPER_COMPUTE_TYPE` — same
- `WHISPER_BATCH_SIZE` — same

---

## 6. Code Changes

### pipeline/pipeline/stages/transcribe.py
- Remove all WhisperX imports, model loading, compat patches
- Replace `run_whisperx()` with an HTTP POST to `WHISPER_SERVICE_HOST:WHISPER_SERVICE_PORT/transcribe`
- Send normalized WAV file + speaker hint params
- Parse and return segments JSON
- Timeout: 3600s (long recordings)

### pipeline/pipeline/stages/embed.py
- `ollama.Client()` → `ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")`

### pipeline/pipeline/stages/summarize.py
- Same Ollama host change

### pipeline/pipeline/stages/notify.py
- Remove entirely — replaced by inline `osascript` call in `run.py`

### pipeline/pipeline/util/hmac_sign.py
- Remove — no longer needed (n8n/webhook dropped)

### pipeline/run.py
- Remove webhook stage from `process_job()`
- Add `osascript` notification on job success

### pipeline/pipeline/stages/intake.py
- Update `detect_source()`: if filename starts with `iphone_` → return `"iphone"` immediately (before regex patterns)

---

## 7. Retirements

| Item | Action |
|---|---|
| `audio-pipeline/` directory | Retired after webui is merged into `pipeline/`. Scripts (process_inbox.py, process_recording.py) are superseded by the watcher+queue approach. |
| `pipeline/pipeline/stages/notify.py` | Deleted |
| `pipeline/pipeline/util/hmac_sign.py` | Deleted |
| `pipeline/pipeline/util/tests/test_hmac.py` | Deleted |
| n8n VPS webhook | Dropped — no replacement |
| Notion integration | Dropped — no replacement |

---

## 8. New Repository Structure

```
pipeline/                          ← canonical codebase (lives on iMac)
  config/
    .env                           # POSTGRES_*, WHISPER_SERVICE_*, OLLAMA_*
    settings.py
  db/
    jobs.db
    schema.sql
  pipeline/
    watcher.py
    queue.py
    stages/
      intake.py
      transcribe.py                # now HTTP client, not local WhisperX
      embed.py
      summarize.py
      store.py
    util/
      audio.py
      chunker.py
  webui/
    app.py                         # merged from audio-pipeline/webui/ + /upload route
    templates/
    static/
  prompts/
    summarize.txt
  tests/
  run.py
  requirements.txt

whisper-service/                   ← lives on Mac Mini only
  main.py                          # FastAPI app
  requirements.txt
  com.shanzer.whisper-service.plist  # launchd plist
```

---

## 9. Out of Scope

- Authentication on upload endpoint or whisper service (LAN-only, not needed now)
- iPhone off-network access (Tailscale can be added later if needed)
- WhisperX MPS tuning — handled inside the Mac Mini service independently
- Speaker voice fingerprinting
- Search UI (pgvector queries)
