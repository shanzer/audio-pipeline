import json
import logging
from datetime import datetime, timezone

import psycopg2
from pgvector.psycopg2 import register_vector

log = logging.getLogger(__name__)


def _register(conn) -> None:
    register_vector(conn)


def write_recording(conn, job: dict, audio_metadata: dict) -> str:
    """Insert recording row. Returns recording UUID."""
    _register(conn)
    filename = job["filename"] if hasattr(job, "__getitem__") else job.filename
    file_path = job["file_path"] if hasattr(job, "__getitem__") else job.file_path
    source = job["source"] if hasattr(job, "__getitem__") else job.source

    meta = {
        "channels": audio_metadata.get("channels", 1),
        "sample_rate": audio_metadata.get("sample_rate"),
        "format": audio_metadata.get("format"),
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recordings
                (filename, file_path, source, duration_sec, recorded_at, status, metadata)
            VALUES (%s, %s, %s, %s, %s, 'processing', %s)
            RETURNING id
            """,
            (
                filename,
                file_path,
                source,
                audio_metadata.get("duration_sec"),
                datetime.now(timezone.utc),
                json.dumps(meta),
            ),
        )
        return str(cur.fetchone()[0])


def write_segments(conn, recording_id: str, segments: list[dict]) -> int:
    """Bulk insert segment rows. Returns count written."""
    with conn.cursor() as cur:
        rows = [
            (
                recording_id,
                i,
                seg.get("speaker"),
                seg.get("start", 0.0),
                seg.get("end", 0.0),
                seg.get("text", "").strip(),
                json.dumps(seg.get("words", [])),
            )
            for i, seg in enumerate(segments)
        ]
        cur.executemany(
            """
            INSERT INTO segments
                (recording_id, segment_index, speaker_label, start_time, end_time, text, words)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        return len(rows)


def write_chunks(conn, recording_id: str, chunks: list[dict]) -> int:
    """Bulk insert chunk rows with embeddings. Returns count written."""
    _register(conn)
    with conn.cursor() as cur:
        rows = [
            (
                recording_id,
                chunk["chunk_index"],
                chunk["text"],
                chunk.get("speaker_label"),
                chunk.get("start_time"),
                chunk.get("end_time"),
                chunk.get("embedding"),
            )
            for chunk in chunks
        ]
        cur.executemany(
            """
            INSERT INTO chunks
                (recording_id, chunk_index, text, speaker_label, start_time, end_time, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        return len(rows)


def write_summary(conn, recording_id: str, summary_dict: dict) -> str:
    """Insert summary row. Returns summary UUID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO summaries
                (recording_id, title, topics, decisions, action_items, risks, raw_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                recording_id,
                summary_dict.get("title"),
                json.dumps(summary_dict.get("topics", [])),
                json.dumps(summary_dict.get("decisions", [])),
                json.dumps(summary_dict.get("action_items", [])),
                json.dumps(summary_dict.get("risks", [])),
                json.dumps(summary_dict),
            ),
        )
        return str(cur.fetchone()[0])


def mark_recording_done(conn, recording_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE recordings
            SET status = 'done', processed_at = NOW()
            WHERE id = %s
            """,
            (recording_id,),
        )


def get_pg_connection(env: dict):
    """Create and return a psycopg2 connection using env vars."""
    log.info("POSTGRES - Connect: %s -- %s [%s]", env.get("POSTGRES_DB"), env.get("POSTGRES_USER"), env.get("POSTGRES_PASSWORD"))
    conn = psycopg2.connect(
        host=env.get("POSTGRES_HOST", "127.0.0.1"),
        port=int(env.get("POSTGRES_PORT", 5432)),
        dbname=env.get("POSTGRES_DB", "audio_pipeline"),
        user=env.get("POSTGRES_USER", "pipeline"),
        password=env["POSTGRES_PASSWORD"],
    )
    conn.autocommit = False
    return conn
