import logging
import time
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config.settings import RECORDINGS_INBOX
from pipeline.queue import enqueue
from pipeline.stages.intake import detect_source

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".m4a", ".mp4", ".wav", ".mp3", ".aac"}


class InboxHandler(FileSystemEventHandler):
    def __init__(self, watched_dirs: set[Path]):
        self._watched = watched_dirs
        super().__init__()

    def _is_evicted(self, p: Path) -> bool:
        """Return True if the file is an iCloud stub (0-byte placeholder)."""
        try:
            return p.stat().st_size == 0
        except OSError:
            return True

    def _is_stable(self, p: Path) -> bool:
        """Return True when file has non-zero, stable size over 3 seconds.
        Guards against iCloud partial downloads (file arrives as stub then
        gets replaced with real content, triggering a second IN_CREATE)."""
        try:
            size1 = p.stat().st_size
            if size1 == 0:
                return False
            time.sleep(3)
            return p.stat().st_size == size1
        except OSError:
            return False

    def _handle(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        if not self._is_stable(p):
            log.warning("Skipping not-yet-ready file: %s", p.name)
            return
        source = detect_source(str(p))
        queued = enqueue(str(p), source=source)
        if queued:
            log.info("Enqueued: %s (source=%s)", p.name, source)
        else:
            log.debug("Already queued: %s", p.name)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            dest = Path(event.dest_path)
            if dest.parent.resolve() in self._watched:
                self._handle(event.dest_path)


def start_watcher(inbox_paths: list[str] | None = None) -> Observer:
    """Start watchdog observer on one or more inbox directories."""
    paths = inbox_paths or [RECORDINGS_INBOX]
    watched = {Path(p).resolve() for p in paths}
    handler = InboxHandler(watched_dirs=watched)
    observer = Observer()
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        observer.schedule(handler, str(p), recursive=False)
        log.info("Watcher started on %s", p)
    observer.start()
    return observer
