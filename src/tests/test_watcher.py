import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── _is_evicted ───────────────────────────────────────────────────────────────

class TestIsEvicted:
    def _handler(self, watched):
        from pipeline.watcher import InboxHandler
        return InboxHandler(watched_dirs={watched})

    def test_zero_byte_file_is_evicted(self, tmp_path):
        f = tmp_path / "audio.m4a"
        f.write_bytes(b"")
        assert self._handler(tmp_path)._is_evicted(f) is True

    def test_nonzero_file_is_not_evicted(self, tmp_path):
        f = tmp_path / "audio.m4a"
        f.write_bytes(b"real audio data")
        assert self._handler(tmp_path)._is_evicted(f) is False

    def test_missing_file_is_evicted(self, tmp_path):
        assert self._handler(tmp_path)._is_evicted(tmp_path / "ghost.m4a") is True


# ── _handle eviction guard ────────────────────────────────────────────────────

class TestHandleEvictionGuard:
    def test_zero_byte_file_not_enqueued(self, tmp_path):
        f = tmp_path / "audio.m4a"
        f.write_bytes(b"")
        from pipeline.watcher import InboxHandler
        handler = InboxHandler(watched_dirs={tmp_path})
        with patch("pipeline.watcher.enqueue") as mock_eq, \
             patch("time.sleep"):
            handler._handle(str(f))
            mock_eq.assert_not_called()

    def test_valid_file_enqueued(self, tmp_path):
        f = tmp_path / "audio.m4a"
        f.write_bytes(b"real audio data")
        from pipeline.watcher import InboxHandler
        handler = InboxHandler(watched_dirs={tmp_path})
        with patch("pipeline.watcher.enqueue", return_value=True) as mock_eq, \
             patch("pipeline.watcher.detect_source", return_value="ambient"), \
             patch("time.sleep"):
            handler._handle(str(f))
            mock_eq.assert_called_once_with(str(f), source="ambient")

    def test_non_audio_extension_not_enqueued(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_bytes(b"some text")
        from pipeline.watcher import InboxHandler
        handler = InboxHandler(watched_dirs={tmp_path})
        with patch("pipeline.watcher.enqueue") as mock_eq, \
             patch("time.sleep"):
            handler._handle(str(f))
            mock_eq.assert_not_called()


# ── on_moved multi-inbox ──────────────────────────────────────────────────────

class TestOnMoved:
    def test_move_into_watched_dir_is_handled(self, tmp_path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        f = inbox / "audio.m4a"
        f.write_bytes(b"data")

        from pipeline.watcher import InboxHandler
        from watchdog.events import FileMovedEvent
        handler = InboxHandler(watched_dirs={inbox.resolve()})
        with patch.object(handler, "_handle") as mock_handle:
            handler.on_moved(FileMovedEvent("/tmp/other.m4a", str(f)))
            mock_handle.assert_called_once_with(str(f))

    def test_move_into_second_watched_dir_is_handled(self, tmp_path):
        inbox1 = tmp_path / "local"
        inbox2 = tmp_path / "icloud"
        inbox1.mkdir()
        inbox2.mkdir()
        f = inbox2 / "audio.m4a"
        f.write_bytes(b"data")

        from pipeline.watcher import InboxHandler
        from watchdog.events import FileMovedEvent
        handler = InboxHandler(watched_dirs={inbox1.resolve(), inbox2.resolve()})
        with patch.object(handler, "_handle") as mock_handle:
            handler.on_moved(FileMovedEvent("/tmp/other.m4a", str(f)))
            mock_handle.assert_called_once_with(str(f))

    def test_move_outside_watched_dirs_is_ignored(self, tmp_path):
        inbox = tmp_path / "inbox"
        other = tmp_path / "other"
        inbox.mkdir()
        other.mkdir()
        f = other / "audio.m4a"

        from pipeline.watcher import InboxHandler
        from watchdog.events import FileMovedEvent
        handler = InboxHandler(watched_dirs={inbox.resolve()})
        with patch.object(handler, "_handle") as mock_handle:
            handler.on_moved(FileMovedEvent("/tmp/source.m4a", str(f)))
            mock_handle.assert_not_called()


# ── start_watcher ─────────────────────────────────────────────────────────────

class TestStartWatcher:
    def test_creates_all_inbox_dirs(self, tmp_path):
        inbox1 = tmp_path / "local"
        inbox2 = tmp_path / "icloud"

        from pipeline.watcher import start_watcher
        observer = start_watcher([str(inbox1), str(inbox2)])
        try:
            assert inbox1.is_dir()
            assert inbox2.is_dir()
        finally:
            observer.stop()
            observer.join()

    def test_single_path_list_works(self, tmp_path):
        inbox = tmp_path / "only"
        from pipeline.watcher import start_watcher
        observer = start_watcher([str(inbox)])
        try:
            assert inbox.is_dir()
        finally:
            observer.stop()
            observer.join()


# ── settings: RECORDINGS_INBOXES ─────────────────────────────────────────────

class TestBuildInboxes:
    def test_local_only_when_icloud_empty(self):
        from config.settings import _build_inboxes
        result = _build_inboxes("/data/inbox", "")
        assert result == ["/data/inbox"]

    def test_icloud_appended_when_set(self):
        from config.settings import _build_inboxes
        result = _build_inboxes("/data/inbox", "/mnt/icloud")
        assert result == ["/data/inbox", "/mnt/icloud"]

    def test_whitespace_icloud_env_treated_as_empty(self):
        from config.settings import _build_inboxes
        result = _build_inboxes("/data/inbox", "   ")
        assert result == ["/data/inbox"]
