import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def q(tmp_path, monkeypatch):
    import pipeline.queue as queue_mod
    monkeypatch.setattr(queue_mod, "SQLITE_DB_PATH", str(tmp_path / "test.db"))
    queue_mod.init_db()
    return queue_mod


class TestEnqueue:
    def test_new_job_is_enqueued(self, q):
        assert q.enqueue("/inbox/audio.m4a", source="iphone") is True

    def test_duplicate_pending_job_not_re_enqueued(self, q):
        q.enqueue("/inbox/audio.m4a")
        assert q.enqueue("/inbox/audio.m4a") is False

    def test_failed_job_is_re_queued(self, q):
        q.enqueue("/inbox/audio.m4a")
        job = q.dequeue()
        q.mark_failed(job["id"], "ffprobe failed: partial download")
        assert q.enqueue("/inbox/audio.m4a") is True

    def test_re_queued_failed_job_becomes_pending(self, q):
        q.enqueue("/inbox/audio.m4a")
        job = q.dequeue()
        q.mark_failed(job["id"], "error")
        q.enqueue("/inbox/audio.m4a")
        assert q.get_pending_count() == 1

    def test_done_job_not_re_queued(self, q):
        q.enqueue("/inbox/audio.m4a")
        job = q.dequeue()
        q.mark_done(job["id"], "recording-uuid-here")
        assert q.enqueue("/inbox/audio.m4a") is False

    def test_processing_job_not_re_queued(self, q):
        q.enqueue("/inbox/audio.m4a")
        q.dequeue()  # moves to 'processing'
        assert q.enqueue("/inbox/audio.m4a") is False


class TestResetStalled:
    def test_processing_job_reset_to_pending(self, q):
        q.enqueue("/inbox/audio.m4a")
        q.dequeue()
        count = q.reset_stalled()
        assert count == 1
        assert q.get_pending_count() == 1

    def test_multiple_processing_jobs_all_reset(self, q):
        q.enqueue("/inbox/a.m4a")
        q.enqueue("/inbox/b.m4a")
        q.dequeue()
        q.dequeue()
        assert q.reset_stalled() == 2
        assert q.get_pending_count() == 2

    def test_done_and_failed_jobs_unaffected(self, q):
        q.enqueue("/inbox/a.m4a")
        q.enqueue("/inbox/b.m4a")
        job = q.dequeue()
        q.mark_done(job["id"], "rec-id")
        q.enqueue("/inbox/b.m4a")
        job2 = q.dequeue()
        q.mark_failed(job2["id"], "error")
        assert q.reset_stalled() == 0

    def test_no_stalled_jobs_returns_zero(self, q):
        q.enqueue("/inbox/audio.m4a")
        assert q.reset_stalled() == 0
