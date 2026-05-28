# Audio Capture & Transcription Pipeline — Design Document

**Version:** 1.0  
**Status:** Architecture Locked, Implementation Starting  
**Platform:** Mac (primary capture), Mac Mini banzai.local (compute), VPS (integration)

---

## 1. Problem Statement

Build a personal audio note-taking and meeting capture system that:
- Records from iPhone (ambient, voice memos) and Mac (Zoom/Meet system audio + mic)
- Transcribes locally with speaker diarization
- Produces structured output (action items, decisions, topics, risks)
- Stores transcripts with semantic embeddings for future search
- Integrates with existing Notion workspace
- Keeps sensitive audio and transcript data on local hardware

---

## 2. Architecture Overview

```
[Capture Layer]
  iPhone: iOS Shortcut → Voice Memo → iCloud Drive /Recordings/inbox/
  Mac:    Loopback (virtual device) → Audio Hijack → ~/Recordings/inbox/

[Pipeline Layer — Mac Mini banzai.local]
  Folder watcher (watchdog)
  → SQLite job queue (pending/processing/done/failed)
  → ffmpeg: validate, normalize, silence-detect chunk boundary
  → WhisperX: transcribe + word-align + diarize (large-v3, CPU int8)
  → Ollama (qwen2.5:14b): structured JSON summary
  → Ollama (mxbai-embed-large): chunk embeddings
  → PostgreSQL + pgvector (Docker, localhost-bound)
  → POST lightweight JSON → n8n webhook (HMAC-SHA256 signed)

[Integration Layer — VPS]
  n8n webhook receiver
  → Notion page creation (via Notion MCP)
  → Notification
  → (future) additional integrations
```

---

## 3. Locked Decisions

All of the following are decided and must not be revisited during implementation.

| Decision | Choice | Rationale |
|---|---|---|
| Pipeline language | Python 3.11 | WhisperX dependency tree, ML ecosystem compatibility |
| Transcription | WhisperX 3.3.4 + large-v3 | Integrated diarization + alignment, single output schema |
| Compute backend | CPU int8 (benchmark MPS) | CTranslate2 MPS support inconsistent on Apple Silicon |
| Diarization | pyannote via WhisperX | Integrated pipeline, no custom alignment glue code |
| Embedding model | mxbai-embed-large (1024d) | Better retrieval benchmarks vs nomic-embed-text (768d) |
| Embedding inference | Ollama on Mac Mini | Privacy, no cloud exposure, already running |
| LLM summarization | qwen2.5:14b via Ollama | Privacy, already running, sufficient for structured extraction |
| Vector store | pgvector in Postgres (Docker) | Co-located with transcript store, pgvector extension |
| Postgres binding | 127.0.0.1:5432 only | Never expose to network interfaces |
| Job queue | SQLite (same pattern as Gmail triage tool) | Persistent, auditable, no additional dependencies |
| n8n integration | Lightweight JSON webhook (HMAC-SHA256) | Compute stays local, n8n handles downstream fanout only |
| Audio routing (Mac) | Loopback: Ch0=local mic, Ch1=Zoom system audio | Clean two-channel separation simplifies diarization |
| Recording trigger (Mac) | Audio Hijack → webhook on stop | Direct automation, no polling |
| Chunk strategy | WhisperX segments grouped 300-400 tokens with 50-token overlap | Natural boundaries, semantic context preserved |
| ivfflat index | Create on schema init, tune lists when >10k chunks | Avoid backfill, tune later |

---

## 4. Data Schema

PostgreSQL with pgvector extension. Applied via Docker init scripts on first container start.

### recordings
Primary record for each audio file processed.

| Column | Type | Notes |
|---|---|---|
| id | UUID PK | uuid_generate_v4() |
| filename | TEXT | Original filename |
| file_path | TEXT | Full path at time of processing |
| source | TEXT | 'zoom', 'iphone', 'ambient' |
| duration_sec | INTEGER | ffprobe output |
| recorded_at | TIMESTAMPTZ | File mtime or metadata |
| processed_at | TIMESTAMPTZ | Pipeline completion time |
| status | TEXT | pending/processing/done/failed |
| error | TEXT | Last error message if failed |
| speaker_count | INTEGER | min/max hint passed to diarizer |
| metadata | JSONB | Extensible bag: channel count, loopback flag, etc. |

### speakers
Maps diarization labels to resolved identities per recording.

| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| recording_id | UUID FK | ON DELETE CASCADE |
| diarization_label | TEXT | SPEAKER_00, SPEAKER_01, etc. |
| resolved_name | TEXT | 'Mike', 'John', null if unknown |
| channel | INTEGER | 0=local mic, 1=remote, null if mono |

### segments
WhisperX segment output. One row per segment.

| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| recording_id | UUID FK | ON DELETE CASCADE |
| segment_index | INTEGER | Ordering |
| speaker_label | TEXT | Diarization label, null if unresolved |
| start_time | FLOAT | Seconds |
| end_time | FLOAT | Seconds |
| text | TEXT | Segment transcript |
| words | JSONB | Word-level timestamps from WhisperX |

### chunks
Grouped segments for embedding. 300-400 token target, 50-token overlap.

| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| recording_id | UUID FK | ON DELETE CASCADE |
| chunk_index | INTEGER | Ordering |
| text | TEXT | Concatenated segment text |
| speaker_label | TEXT | Dominant speaker in chunk |
| start_time | FLOAT | Start of first segment in chunk |
| end_time | FLOAT | End of last segment in chunk |
| embedding | vector(1024) | mxbai-embed-large output |

### summaries
LLM structured output per recording.

| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| recording_id | UUID FK | ON DELETE CASCADE |
| title | TEXT | Extracted title |
| topics | JSONB | [] |
| decisions | JSONB | [] |
| action_items | JSONB | [{owner, task, due}] |
| risks | JSONB | [] |
| raw_json | JSONB | Full LLM response for debugging |
| created_at | TIMESTAMPTZ | |

### Indexes
```sql
CREATE INDEX ON recordings(status);
CREATE INDEX ON recordings(recorded_at);
CREATE INDEX ON segments(recording_id);
CREATE INDEX ON chunks(recording_id);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## 5. Pipeline Stages

### Stage 0: Intake
- Folder watcher monitors `~/Recordings/inbox/`
- New file detected → insert row into SQLite job queue with status=pending
- Dequeue worker picks up pending jobs one at a time

### Stage 1: Validation & Normalization
- ffprobe: validate file is readable audio, extract duration
- ffmpeg: normalize to 16kHz mono WAV (WhisperX input format)
- Detect if file exceeds 25MB (API limit reference) → chunk on silence boundaries if needed
- For two-channel Loopback recordings: preserve stereo, pass channel metadata

### Stage 2: Transcription + Diarization (WhisperX)
- Load large-v3 model
- Transcribe → align → diarize
- Pass `min_speakers`, `max_speakers` from job metadata (default 1, 6)
- For two-channel recordings: SPEAKER_00 = channel 0 (local mic), diarize channel 1 separately
- Output: segments[] with speaker, start, end, text, words

### Stage 3: Chunking + Embedding
- Group segments into 300-400 token chunks with 50-token overlap
- Respect segment boundaries (no mid-segment splits)
- For each chunk: POST to Ollama mxbai-embed-large
- Store chunks + embeddings to Postgres

### Stage 4: LLM Summarization
- Assemble full transcript with speaker labels
- POST to Ollama qwen2.5:14b with structured prompt
- Prompt requires JSON output: {title, topics[], decisions[], action_items[{owner,task,due}], risks[]}
- Validate JSON parse before writing to DB
- Retry once on parse failure with explicit JSON-only instruction

### Stage 5: Write to Postgres
- Insert recording (status=processing already set)
- Insert segments[]
- Insert chunks[] with embeddings
- Insert summary
- Update recording status=done, processed_at=now()

### Stage 6: Webhook to n8n
- Assemble lightweight payload (no transcript, no embeddings):
```json
{
  "recording_id": "uuid",
  "title": "string",
  "source": "zoom|iphone|ambient",
  "recorded_at": "iso8601",
  "duration_sec": 0,
  "speakers": ["Mike", "SPEAKER_01"],
  "topics": [],
  "decisions": [],
  "action_items": [],
  "risks": [],
  "postgres_row_id": "uuid"
}
```
- Sign with HMAC-SHA256, include in X-Pipeline-Signature header
- POST to n8n webhook URL (from .env)
- n8n creates Notion page, sends notification

---

## 6. File & Directory Structure

```
~/pipeline/
  config/
    .env                    # secrets: HF_TOKEN, POSTGRES_*, N8N_WEBHOOK_*, HMAC_SECRET
    settings.py             # non-secret config constants
  db/
    jobs.db                 # SQLite job queue
    schema.sql              # SQLite schema
  pipeline/
    __init__.py
    watcher.py              # watchdog folder monitor
    queue.py                # SQLite job queue operations
    stages/
      __init__.py
      intake.py             # ffprobe/ffmpeg validation + normalization
      transcribe.py         # WhisperX transcription + diarization
      embed.py              # chunking + Ollama embedding
      summarize.py          # Ollama LLM structured extraction
      store.py              # Postgres write operations
      notify.py             # n8n webhook dispatch
    util/
      audio.py              # ffmpeg/ffprobe helpers
      chunker.py            # segment grouping with overlap
      hmac_sign.py          # HMAC-SHA256 webhook signing
  prompts/
    summarize.txt           # LLM summarization prompt template
  tests/
    test_chunker.py
    test_hmac.py
    validate.py             # standalone validation script (run first)
  requirements.txt          # pinned, from pip freeze
  run.py                    # entry point: starts watcher + dequeue worker

~/Recordings/
  inbox/                    # watched directory
  archive/                  # processed files moved here
  failed/                   # files that failed processing

~/docker/audio-pipeline/
  docker-compose.yml
  .env                      # POSTGRES_PASSWORD
  init/
    001_schema.sql          # pgvector schema, applied on first start
```

---

## 7. Environment State at Handoff

The following has been completed before Claude Code takes over:

- [x] Homebrew Python 3.11 installed
- [x] venv created at `~/venvs/whisper-pipeline` with Python 3.11
- [ ] Dependencies installed (PyTorch, WhisperX, psycopg2, pgvector, ollama, python-dotenv, huggingface_hub)
- [ ] requirements.txt frozen
- [ ] HuggingFace token configured + terms accepted for pyannote models
- [ ] Docker Compose for Postgres + pgvector created and started
- [ ] Database schema applied
- [ ] validate.py run against a test audio file

Claude Code picks up from dependency installation and completes everything through a passing validate.py run before building the pipeline proper.

---

## 8. Operational Considerations

**Privacy:** Audio files and full transcripts never leave the Mac Mini. The VPS receives only the structured summary JSON. Postgres is bound to localhost only.

**Failure modes:** SQLite job queue persists state across crashes. Failed jobs are moved to `failed/` directory and marked status=failed with error message. Reprocessing is manual: move file back to inbox/, reset job status.

**Performance:** large-v3 + diarization on a 1-hour recording will take 20-45 minutes on CPU int8. This is acceptable (post-meeting, not real-time). Pipeline process should run at low priority (nice 10) and yield if Ollama is under load.

**MPS:** Benchmark both `device="cpu" compute_type="int8"` and `device="mps"` on the actual Mac Mini. Use whichever is faster. CTranslate2 MPS support has known issues — CPU int8 may win.

**Speaker resolution:** Diarization labels (SPEAKER_00) are best-effort. The speakers table allows manual label resolution. For two-channel Loopback recordings, SPEAKER_00 = local mic = Mike is deterministic — only remote participants need diarization.

**ivfflat index:** Requires data to be useful. With fewer than ~1000 chunks it's slower than a sequential scan. Monitor and consider `SET enable_indexscan = off` until data volume justifies it.

---

## 9. Out of Scope (Future)

- Speaker voice fingerprinting (automatic resolved_name from audio)
- Real-time transcription
- Search UI (pgvector search queries to be implemented as separate tool)
- iPhone Shortcut automation (implement after Mac pipeline is stable)
- Tailscale setup for remote Postgres access from n8n (n8n writes to Notion only for now)
