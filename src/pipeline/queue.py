import json
import sqlite3
from pathlib import Path

from config.settings import SQLITE_DB_PATH

_SCHEMA = Path(__file__).parent.parent / "db" / "schema.sql"


def _conn() -> sqlite3.Connection:
    db_path = Path(SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA.read_text())


def enqueue(file_path: str, source: str = "unknown", metadata: dict = None) -> bool:
    """Insert a new job. Returns True if inserted, False if already exists."""
    filename = Path(file_path).name
    meta_json = json.dumps(metadata or {})
    with _conn() as conn:
        with conn:
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (file_path, filename, source, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (str(file_path), filename, source, meta_json),
                )
                return True
            except sqlite3.IntegrityError:
                return False


def dequeue() -> sqlite3.Row | None:
    """Atomically claim one pending job. Returns the job row or None."""
    with _conn() as conn:
        with conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    started_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (row["id"],),
            )
            return row


def mark_done(job_id: int, recording_id: str) -> None:
    with _conn() as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'done',
                    completed_at = datetime('now'),
                    updated_at = datetime('now'),
                    metadata = json_set(metadata, '$.recording_id', ?)
                WHERE id = ?
                """,
                (recording_id, job_id),
            )


def mark_failed(job_id: int, error: str) -> None:
    with _conn() as conn:
        with conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    updated_at = datetime('now'),
                    error = ?
                WHERE id = ?
                """,
                (error[:4000], job_id),
            )


def get_pending_count() -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'pending'"
        ).fetchone()[0]
