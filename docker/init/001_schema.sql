CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE recordings (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename      TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'unknown',
    duration_sec  INTEGER,
    recorded_at   TIMESTAMPTZ NOT NULL,
    processed_at  TIMESTAMPTZ,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    speaker_count INTEGER,
    metadata      JSONB DEFAULT '{}'
);

CREATE TABLE speakers (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recording_id      UUID REFERENCES recordings(id) ON DELETE CASCADE,
    diarization_label TEXT NOT NULL,
    resolved_name     TEXT,
    channel           INTEGER
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
