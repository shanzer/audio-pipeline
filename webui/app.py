#!/usr/bin/env python3
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras
from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = os.getenv("WEBUI_SECRET_KEY", "local-audio-pipeline-webui")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

ALLOWED_EXTENSIONS = {".m4a", ".mp4", ".wav", ".mp3", ".aac"}


def db_connect():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
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


def safe_filename(filename):
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
        if values:
            for item in values:
                lines.append(f"- {item}")
        else:
            lines.append("- None recorded")
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


@app.route("/")
def index():
    recordings = query_all(
        """
        SELECT
            r.id,
            r.filename,
            r.source,
            r.duration_sec,
            r.recorded_at,
            r.processed_at,
            r.status,
            r.error,
            r.speaker_count,
            latest_summary.title AS summary_title,
            COALESCE(segment_stats.segment_count, 0) AS segment_count,
            COALESCE(chunk_stats.chunk_count, 0) AS chunk_count
        FROM recordings r
        LEFT JOIN LATERAL (
            SELECT title
            FROM summaries
            WHERE recording_id = r.id
            ORDER BY created_at DESC
            LIMIT 1
        ) latest_summary ON true
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS segment_count
            FROM segments
            WHERE recording_id = r.id
        ) segment_stats ON true
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS chunk_count
            FROM chunks
            WHERE recording_id = r.id
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
        """
        SELECT *
        FROM summaries
        WHERE recording_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (recording_id,),
    )
    if summary:
        summary["topics"] = normalize_json(summary.get("topics"), [])
        summary["decisions"] = normalize_json(summary.get("decisions"), [])
        summary["action_items"] = normalize_json(summary.get("action_items"), [])
        summary["risks"] = normalize_json(summary.get("risks"), [])
        summary["raw_json"] = normalize_json(summary.get("raw_json"), {})

    speakers = query_all(
        """
        SELECT diarization_label, resolved_name, channel
        FROM speakers
        WHERE recording_id = %s
        ORDER BY diarization_label
        """,
        (recording_id,),
    )
    segments = query_all(
        """
        SELECT segment_index, speaker_label, start_time, end_time, text
        FROM segments
        WHERE recording_id = %s
        ORDER BY segment_index
        """,
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
    basename = safe_filename(recording["filename"])

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


@app.context_processor
def template_helpers():
    return {
        "summary_sections": summary_sections,
        "now": datetime.now,
    }
