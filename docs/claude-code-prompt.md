# Claude Code Implementation Prompt — Audio Pipeline

## Context

You are building a local audio capture and transcription pipeline on a Mac Mini (banzai.local, Apple Silicon M4, 32GB). The full architecture and all technical decisions are documented in `audio-pipeline-design.md`. Read that document first and treat every decision in it as locked — do not re-evaluate tool choices, language choices, or schema decisions.

You are operating as the developer and support team for this system. Handle errors autonomously. If a command fails, diagnose and fix it. Do not ask for permission to proceed between steps unless you hit a genuine blocker that requires a human decision.

---

## What Has Been Done

- Homebrew Python 3.11 is installed at `/opt/homebrew/opt/python@3.11/`
- A venv exists at `~/venvs/whisper-pipeline` created with Python 3.11
- Nothing else has been done. Dependencies are not yet installed.

---

## What You Are Building

Work through the following phases in order. Complete and validate each phase before starting the next. Do not skip ahead.

---

## Phase 1: Environment Setup

### 1.1 Activate venv and verify Python version

```bash
source ~/venvs/whisper-pipeline/bin/activate
python --version  # must show 3.11.x
which pip         # must be inside ~/venvs/whisper-pipeline
```

If Python version is wrong, stop and report. Do not proceed with a wrong Python version.

### 1.2 Install dependencies in strict order

PyTorch must be installed before WhisperX. Do not install them in the same pip command.

```bash
# Step 1: PyTorch for Apple Silicon
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: WhisperX pinned version
pip install "whisperx==3.3.4"

# Step 3: Remaining dependencies
pip install psycopg2-binary pgvector ollama python-dotenv huggingface_hub watchdog
```

If any step fails with a version conflict, resolve it before proceeding. Do not use `--ignore-requires-python` or `--no-deps` without flagging the issue.

### 1.3 Freeze requirements immediately

```bash
pip freeze > ~/pipeline/requirements.txt
```

Create `~/pipeline/` directory first if it does not exist.

### 1.4 Validate PyTorch MPS availability

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

Record the output in a comment in `config/settings.py` when you create it. MPS availability affects runtime device selection.

### 1.5 HuggingFace login

```bash
huggingface-cli login
```

This will prompt for a token interactively. After login, verify:

```bash
python3 -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

**STOP here and tell the user:** They must manually accept terms for two pyannote models at these URLs before Phase 2 can proceed:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Wait for user confirmation before proceeding to Phase 2.

---

## Phase 2: Docker + Postgres

### 2.1 Create directory structure

```
~/docker/audio-pipeline/
  docker-compose.yml
  .env
  init/
    001_schema.sql
```

### 2.2 docker-compose.yml

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: audio-pipeline-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: audio_pipeline
      POSTGRES_USER: pipeline
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init:/docker-entrypoint-initdb.d
    ports:
      - "127.0.0.1:5432:5432"

volumes:
  pgdata:
```

**Critical:** The port binding `127.0.0.1:5432:5432` is non-negotiable. Do not change it to `5432:5432`.

### 2.3 .env for Docker

Generate a random password:

```bash
openssl rand -base64 32
```

Write `~/docker/audio-pipeline/.env`:
```
POSTGRES_PASSWORD=<generated>
```

### 2.4 001_schema.sql

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE recordings (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename      TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    source        TEXT NOT NULL,
    duration_sec  INTEGER,
    recorded_at   TIMESTAMPTZ NOT NULL,
    processed_at  TIMESTAMPTZ,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    speaker_count INTEGER,
    metadata      JSONB DEFAULT '{}'
);

CREATE TABLE speakers (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id        UUID REFERENCES recordings(id) ON DELETE CASCADE,
    diarization_label   TEXT NOT NULL,
    resolved_name       TEXT,
    channel             INTEGER
);

CREATE TABLE segments (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id    UUID REFERENCES recordings(id) ON DELETE CASCADE,
    segment_index   INTEGER NOT NULL,
    speaker_label   TEXT,
    start_time      FLOAT NOT NULL,
    end_time        FLOAT NOT NULL,
    text            TEXT NOT NULL,
    words           JSONB
);

CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id    UUID REFERENCES recordings(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    speaker_label   TEXT,
    start_time      FLOAT,
    end_time        FLOAT,
    embedding       vector(1024)
);

CREATE TABLE summaries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id    UUID REFERENCES recordings(id) ON DELETE CASCADE,
    title           TEXT,
    topics          JSONB DEFAULT '[]',
    decisions       JSONB DEFAULT '[]',
    action_items    JSONB DEFAULT '[]',
    risks           JSONB DEFAULT '[]',
    raw_json        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON recordings(status);
CREATE INDEX ON recordings(recorded_at);
CREATE INDEX ON segments(recording_id);
CREATE INDEX ON chunks(recording_id);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 2.5 Start and validate

```bash
cd ~/docker/audio-pipeline
docker compose up -d
sleep 5
docker compose ps  # postgres should show healthy or running
```

Validate schema applied:
```bash
docker exec audio-pipeline-db psql -U pipeline -d audio_pipeline -c "\dt"
```

Expected: recordings, speakers, segments, chunks, summaries tables visible.

---

## Phase 3: Project Scaffold

Create the full directory structure from the design document. Create all `__init__.py` files. Create `config/settings.py` with non-secret constants:

```python
# config/settings.py
import os

VENV_PATH = os.path.expanduser("~/venvs/whisper-pipeline")
RECORDINGS_INBOX = os.path.expanduser("~/Recordings/inbox")
RECORDINGS_ARCHIVE = os.path.expanduser("~/Recordings/archive")
RECORDINGS_FAILED = os.path.expanduser("~/Recordings/failed")
SQLITE_DB_PATH = os.path.expanduser("~/pipeline/db/jobs.db")

WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cpu"       # benchmark mps vs cpu, update if mps is faster
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BATCH_SIZE = 8

EMBED_MODEL = "mxbai-embed-large"
EMBED_DIMENSIONS = 1024

LLM_MODEL = "qwen2.5:14b"

CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50

DEFAULT_MIN_SPEAKERS = 1
DEFAULT_MAX_SPEAKERS = 6
```

Create `config/.env.template` (not .env itself — user fills this in):

```
HF_TOKEN=
POSTGRES_PASSWORD=
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=audio_pipeline
POSTGRES_USER=pipeline
N8N_WEBHOOK_URL=
HMAC_SECRET=
```

Create the `~/Recordings/inbox/`, `~/Recordings/archive/`, and `~/Recordings/failed/` directories.

---

## Phase 4: Validate WhisperX Pipeline

Before writing any pipeline code, validate the WhisperX output directly. This is not optional. The purpose is to confirm output schema and diarization quality on real audio before building plumbing around it.

Create `tests/validate.py`. The script must:

1. Accept a path to an audio file as a command-line argument
2. Run all three WhisperX stages: transcribe → align → diarize
3. Benchmark and print elapsed time for each stage separately
4. Accept `--min-speakers` and `--max-speakers` arguments (default 1, 6)
5. Accept `--device` argument (default: cpu)
6. Print the first 5 segments as formatted JSON
7. Print a summary: total segments, unique speaker labels found, total duration
8. Write full output to `tests/validate_output.json`

Load `HF_TOKEN` from `.env` file using `python-dotenv`. Do not hardcode any token.

Run it:

```bash
cd ~/pipeline
source ~/venvs/whisper-pipeline/bin/activate
python3 tests/validate.py /path/to/test.m4a --min-speakers 2 --max-speakers 2
```

**Do not proceed to Phase 5 until:**
- Script runs without error
- Speaker labels appear on segments (not null)
- Timestamps look sane (monotonically increasing, within audio duration)
- Segment text is coherent English

If speaker labels are null on all segments, diagnose before continuing. Common causes: HF token not valid, pyannote terms not accepted, audio too short for diarization.

---

## Phase 5: SQLite Job Queue

Implement `db/schema.sql` for SQLite:

```sql
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'unknown',
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    started_at  TEXT,
    completed_at TEXT,
    error       TEXT,
    metadata    TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
```

Implement `pipeline/queue.py` with these functions:
- `init_db()` — create tables if not exist
- `enqueue(file_path, source='unknown', metadata=None)` — insert job, ignore if already exists (UNIQUE constraint)
- `dequeue()` — select one pending job, set status=processing, return row or None. Use a transaction to prevent double-processing.
- `mark_done(job_id, recording_id)` — set status=done, completed_at
- `mark_failed(job_id, error)` — set status=failed, error=message
- `get_pending_count()` — count of pending jobs

All database operations must use context managers and explicit transactions. No implicit commits.

---

## Phase 6: Pipeline Stages

Implement each stage as a module in `pipeline/stages/`. Each stage takes the job record and any outputs from prior stages as arguments. Each stage returns its output or raises an exception (caller handles marking job failed).

### intake.py
Functions:
- `validate_audio(file_path)` → `{duration_sec, channels, sample_rate, format}` — uses ffprobe via subprocess. Raise `ValueError` if file is not valid audio.
- `normalize_audio(file_path, output_path)` → normalized WAV path — ffmpeg to 16kHz, preserve channel count (mono or stereo). Return output path.
- `detect_source(file_path, metadata)` → `'zoom'|'iphone'|'ambient'` — infer from metadata or filename patterns.

### transcribe.py
Functions:
- `run_whisperx(audio_path, min_speakers, max_speakers, device, compute_type)` → `{segments, language, word_segments}`
- Each segment must have: `start`, `end`, `text`, `speaker` (may be null), `words`
- Load model once per process (module-level singleton, not per-call)
- Log elapsed time for each of the three stages: transcribe, align, diarize

### embed.py
Functions:
- `chunk_segments(segments, target_tokens, overlap_tokens)` → `[{text, speaker_label, start_time, end_time, chunk_index}]`
  - Group segments to target token count
  - Respect segment boundaries (never split mid-segment)
  - Overlap by including last N tokens of previous chunk at start of next
  - Dominant speaker = speaker_label of longest segment in chunk by duration
- `embed_chunks(chunks, model_name)` → chunks with `embedding` field added
  - Call Ollama embedding API for each chunk
  - Batch if Ollama client supports it; otherwise sequential
  - Retry once on timeout

### summarize.py
Functions:
- `build_transcript(segments)` → formatted string with speaker labels and timestamps
- `run_summary(transcript, model_name)` → parsed dict `{title, topics, decisions, action_items, risks}`
  - Load prompt template from `prompts/summarize.txt`
  - Require JSON-only output in prompt
  - Parse response, validate required keys present
  - Retry once with explicit "return only JSON, no other text" instruction if parse fails
  - Raise `ValueError` if second attempt also fails to parse

Prompt template `prompts/summarize.txt`:
```
You are extracting structured information from a meeting transcript.
Return ONLY a valid JSON object with no preamble, no explanation, no markdown code fences.

Required format:
{
  "title": "Brief descriptive title for this meeting/recording",
  "topics": ["topic1", "topic2"],
  "decisions": ["decision1", "decision2"],
  "action_items": [
    {"owner": "name or unknown", "task": "description", "due": "date or null"}
  ],
  "risks": ["risk1", "risk2"]
}

Transcript:
{transcript}
```

### store.py
Functions:
- `write_recording(conn, job, audio_metadata)` → `recording_id (UUID)`
- `write_segments(conn, recording_id, segments)` → count written
- `write_chunks(conn, recording_id, chunks)` → count written
- `write_summary(conn, recording_id, summary_dict)` → summary_id
- `mark_recording_done(conn, recording_id)`
- All writes in a single transaction per recording. If any write fails, roll back and re-raise.

### notify.py
Functions:
- `build_payload(recording_id, job, summary, speakers)` → dict (lightweight JSON, no transcript, no embeddings)
- `sign_payload(payload_bytes, secret)` → HMAC-SHA256 hex digest
- `send_webhook(payload, webhook_url, secret)` → True on 2xx, raise on failure
  - Header: `X-Pipeline-Signature: sha256=<hex_digest>`
  - Timeout: 10 seconds
  - Do not retry — n8n webhook is best-effort, pipeline success does not depend on it

---

## Phase 7: Watcher + Orchestrator

### watcher.py
Use `watchdog` library to monitor `~/Recordings/inbox/`. On `FileCreatedEvent` or `FileMovedEvent` (destination in inbox):
- Wait 2 seconds after event before enqueuing (allow file write to complete)
- Call `queue.enqueue(file_path)`
- Log the enqueue action

Only watch for audio file extensions: `.m4a`, `.mp4`, `.wav`, `.mp3`, `.aac`

### run.py
Entry point. Start two threads:
1. Watchdog observer on inbox directory
2. Worker loop: poll SQLite for pending jobs every 10 seconds, dequeue and process one at a time

Worker loop for each job:
```
dequeue()
→ intake: validate + normalize
→ transcribe: whisperx
→ embed: chunk + embed
→ summarize: LLM
→ store: postgres writes
→ notify: n8n webhook
→ mark_done() + move file to archive/
on any exception:
→ mark_failed(error) + move file to failed/
→ log full traceback
→ continue to next job (do not crash worker)
```

Log all stage transitions with timestamp and job_id. Use Python's `logging` module, not print statements. Log to both stdout and `~/pipeline/pipeline.log`.

---

## Phase 8: Integration Test

With Postgres running and Ollama running with `mxbai-embed-large` and `qwen2.5:14b` pulled:

1. Drop a real audio file into `~/Recordings/inbox/`
2. Run `python3 run.py` with logging at DEBUG level
3. Verify:
   - Job moves through pending → processing → done in SQLite
   - All five Postgres tables have rows for the recording
   - embedding column in chunks is not null
   - summaries table has valid JSON in action_items
   - Audio file moved to archive/
4. Run a manual pgvector similarity query to confirm embeddings are valid:

```sql
SELECT chunk_index, text, start_time
FROM chunks
WHERE recording_id = '<uuid>'
ORDER BY embedding <=> (
    SELECT embedding FROM chunks WHERE chunk_index = 0 AND recording_id = '<uuid>'
) LIMIT 5;
```

---

## Implementation Rules

1. **No hardcoded secrets.** All tokens, passwords, URLs loaded from `.env` via `python-dotenv`.
2. **No cloud calls.** Everything runs locally. No OpenAI API, no external embedding services.
3. **Fail loudly, recover gracefully.** Exceptions crash the stage, not the worker. The worker catches, marks failed, and continues.
4. **Pin external calls.** All subprocess calls use explicit paths or validated executables. Do not assume ffmpeg or ffprobe are on PATH without checking.
5. **Log everything.** Every stage transition, every timing measurement, every error with full traceback.
6. **No silent data loss.** If a Postgres write fails, roll back the entire recording transaction. Never write partial data.
7. **Requirements.txt is the source of truth.** After any pip install, re-freeze.
8. **Test the chunker independently.** `tests/test_chunker.py` must have at least 5 test cases covering: empty input, single segment, segments that require overlap, segments longer than target, mixed-speaker segments.

---

## Stopping Conditions

Stop and report to the user if you encounter:
- Python version is not 3.11.x after activating the venv
- MPS shows as unavailable (unexpected, report but continue with cpu)
- WhisperX validate.py produces segments with null speaker on all segments after HF terms are confirmed accepted
- Docker fails to start with the localhost-only port binding
- Ollama is not running or mxbai-embed-large is not pulled

For everything else: diagnose and fix autonomously.
