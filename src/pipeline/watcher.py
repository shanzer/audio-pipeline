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
    def _handle(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        # Wait briefly to allow file write to complete
        time.sleep(2)
        if not p.exists():
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
            if dest.parent == Path(RECORDINGS_INBOX):
                self._handle(event.dest_path)


def start_watcher() -> Observer:
    """Start watchdog observer on inbox directory. Returns the Observer."""
    Path(RECORDINGS_INBOX).mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(InboxHandler(), RECORDINGS_INBOX, recursive=False)
    observer.start()
    log.info("Watcher started on %s", RECORDINGS_INBOX)
    return observer
