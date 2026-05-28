import io
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDINGS_INBOX", str(tmp_path))
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    import importlib
    import webui.app as webui_module
    importlib.reload(webui_module)
    webui_module.app.config["TESTING"] = True
    with webui_module.app.test_client() as c:
        yield c, tmp_path


def test_upload_no_file_field(client):
    c, _ = client
    resp = c.post("/upload")
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False
    assert "no file" in resp.get_json()["error"]


def test_upload_empty_filename(client):
    c, _ = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"data"), "")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_upload_bad_extension(client):
    c, _ = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"data"), "audio.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.get_json()["error"]


def test_upload_success_wav(client):
    c, inbox = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"RIFF" + b"\x00" * 40), "memo.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    filename = body["filename"]
    assert filename.startswith("iphone_")
    assert filename.endswith(".wav")
    assert (inbox / filename).exists()


def test_upload_success_m4a(client):
    c, inbox = client
    resp = c.post(
        "/upload",
        data={"file": (io.BytesIO(b"\x00" * 100), "voice memo.m4a")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["filename"].startswith("iphone_")
    assert body["filename"].endswith(".m4a")
