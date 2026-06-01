#!/usr/bin/env python3
"""
Entry point: starts folder watcher + worker loop.
Run: python3 run.py
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Load secrets before any pipeline imports
_env_path = Path(__file__).parent / "config" / ".env"
load_dotenv(_env_path, override=True)

from config.settings import (
    RECORDINGS_ARCHIVE,
    RECORDINGS_FAILED,
    RECORDINGS_INBOXES,
    LOG_PATH,
    DEFAULT_MIN_SPEAKERS,
    DEFAULT_MAX_SPEAKERS,
)
from pipeline.queue import init_db, dequeue, mark_done, mark_failed, get_pending_count, reset_stalled
from pipeline.watcher import start_watcher


def _setup_logging() -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH),
    ]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


log = logging.getLogger("run")


def _move(src: str, dest_dir: str) -> None:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(dest_dir) / Path(src).name
    shutil.move(src, str(dest))


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


def process_job(job) -> None:
    job_id = job["id"]
    file_path = job["file_path"]
    log.info("[job=%d] Starting: %s", job_id, file_path)

    # Lazy imports — transcribe module applies compat patches on first import
    from pipeline.stages.intake import validate_audio, normalize_audio_file, detect_source
    from pipeline.stages.transcribe import run_whisperx
    from pipeline.stages.embed import chunk_and_embed
    from pipeline.stages.summarize import build_transcript, run_summary
    from pipeline.stages.store import write_recording, write_segments, write_chunks, write_summary, mark_recording_done, get_pg_connection

    # Stage 1: Validate + normalize
    log.info("[job=%d] Stage: intake", job_id)
    audio_meta = validate_audio(file_path)
    tmp_wav = tempfile.mktemp(suffix=".wav", prefix="pipeline_")
    try:
        normalized = normalize_audio_file(file_path, tmp_wav)

        # Stage 2: Transcribe
        log.info("[job=%d] Stage: transcribe", job_id)
        meta_json = job["metadata"] if hasattr(job, "__getitem__") else job.metadata
        import json as _json
        meta = _json.loads(meta_json) if isinstance(meta_json, str) else (meta_json or {})
        min_spk = meta.get("min_speakers", DEFAULT_MIN_SPEAKERS)
        max_spk = meta.get("max_speakers", DEFAULT_MAX_SPEAKERS)
        transcription = run_whisperx(
            normalized,
            min_speakers=min_spk,
            max_speakers=max_spk,
        )
        segments = transcription["segments"]
        log.info("[job=%d] Transcribed: %d segments", job_id, len(segments))

        # Stage 3: Chunk + embed
        log.info("[job=%d] Stage: embed", job_id)
        chunks = chunk_and_embed(segments)
        log.info("[job=%d] Embedded: %d chunks", job_id, len(chunks))

        # Stage 4: Summarize
        log.info("[job=%d] Stage: summarize", job_id)
        transcript_text = build_transcript(segments)
        summary = run_summary(transcript_text)
        log.info("[job=%d] Summary title: %s", job_id, summary.get("title", ""))

        # Stage 5: Write to Postgres
        log.info("[job=%d] Stage: store", job_id)
        pg_conn = get_pg_connection(os.environ)
        try:
            recording_id = write_recording(pg_conn, job, audio_meta)
            write_segments(pg_conn, recording_id, segments)
            write_chunks(pg_conn, recording_id, chunks)
            write_summary(pg_conn, recording_id, summary)
            mark_recording_done(pg_conn, recording_id)
            pg_conn.commit()
        except Exception:
            pg_conn.rollback()
            raise
        finally:
            pg_conn.close()
        log.info("[job=%d] Stored: recording_id=%s", job_id, recording_id)

        # Stage 6: Notify
        _notify(
            title=summary.get("title", ""),
            duration_sec=audio_meta.get("duration_sec", 0),
        )

        # Done
        mark_done(job_id, recording_id)
        _move(file_path, RECORDINGS_ARCHIVE)
        log.info("[job=%d] Complete: moved to archive", job_id)

    finally:
        Path(tmp_wav).unlink(missing_ok=True)


def worker_loop() -> None:
    log.info("Worker loop started")
    while True:
        try:
            job = dequeue()
            if job is None:
                pending = get_pending_count()
                log.debug("No pending jobs (total pending=%d), sleeping 10s", pending)
                time.sleep(10)
                continue

            try:
                process_job(job)
            except Exception:
                job_id = job["id"]
                file_path = job["file_path"]
                tb = traceback.format_exc()
                log.error("[job=%d] Failed:\n%s", job_id, tb)
                mark_failed(job_id, tb[-3000:])
                try:
                    if Path(file_path).exists():
                        _move(file_path, RECORDINGS_FAILED)
                except Exception as move_err:
                    log.error("[job=%d] Could not move to failed/: %s", job_id, move_err)

        except Exception:
            log.error("Worker loop error:\n%s", traceback.format_exc())
            time.sleep(10)


def main() -> None:
    _setup_logging()
    log.info("Pipeline starting")

    init_db()
    n = reset_stalled()
    if n:
        log.warning("Reset %d stalled processing job(s) to pending", n)

    observer = start_watcher(RECORDINGS_INBOXES)

    worker_thread = threading.Thread(target=worker_loop, daemon=True, name="worker")
    worker_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down")
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
