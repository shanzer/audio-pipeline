import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app  # conftest has pre-mocked all ML deps


@pytest.fixture
def client():
    return TestClient(app)


def _fake_segments():
    return [{"start": 0.0, "end": 2.0, "text": "hello world", "speaker": "SPEAKER_00", "words": []}]


def _audio_upload(filename="test.wav"):
    return ("file", (filename, io.BytesIO(b"RIFF" + b"\x00" * 40), "audio/wav"))


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_transcribe_happy_path(client):
    fake_segments = _fake_segments()

    with (
        patch("main.whisperx.load_audio", return_value=MagicMock()),
        patch("main.mlx_whisper.transcribe", return_value={"segments": fake_segments, "language": "en"}),
        patch("main.get_align_model") as mock_align,
        patch("main.get_diarize_model") as mock_diarize,
        patch("main.whisperx.align", return_value={"segments": fake_segments, "word_segments": []}),
        patch("main.whisperx.assign_word_speakers", return_value={"segments": fake_segments, "word_segments": []}),
    ):
        mock_align.return_value = (MagicMock(), MagicMock())
        mock_diarize.return_value.return_value = MagicMock()

        resp = client.post(
            "/transcribe",
            files=[_audio_upload()],
            data={"min_speakers": "1", "max_speakers": "2"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "segments" in body
    assert "language" in body
    assert "word_segments" in body
    assert body["language"] == "en"
    assert len(body["segments"]) == 1


def test_transcribe_missing_file_returns_422(client):
    resp = client.post("/transcribe", data={"min_speakers": "1", "max_speakers": "2"})
    assert resp.status_code == 422


def test_transcribe_speaker_range_min_zero_returns_422(client):
    resp = client.post(
        "/transcribe",
        files=[_audio_upload()],
        data={"min_speakers": "0", "max_speakers": "2"},
    )
    assert resp.status_code == 422


def test_transcribe_speaker_range_min_greater_than_max_returns_422(client):
    resp = client.post(
        "/transcribe",
        files=[_audio_upload()],
        data={"min_speakers": "5", "max_speakers": "2"},
    )
    assert resp.status_code == 422


def test_transcribe_speaker_range_max_exceeds_limit_returns_422(client):
    resp = client.post(
        "/transcribe",
        files=[_audio_upload()],
        data={"min_speakers": "1", "max_speakers": "21"},
    )
    assert resp.status_code == 422


def test_transcribe_model_error_returns_500(client):
    with patch("main.whisperx.load_audio", side_effect=RuntimeError("model exploded")):
        resp = client.post(
            "/transcribe",
            files=[_audio_upload()],
            data={"min_speakers": "1", "max_speakers": "2"},
        )
    assert resp.status_code == 500
    body = resp.json()
    assert "error" in body
    assert "model exploded" in body["error"]


def test_transcribe_passes_speaker_hints_to_diarizer(client):
    fake_segments = _fake_segments()

    with (
        patch("main.whisperx.load_audio", return_value=MagicMock()),
        patch("main.mlx_whisper.transcribe", return_value={"segments": fake_segments, "language": "en"}),
        patch("main.get_align_model") as mock_align,
        patch("main.get_diarize_model") as mock_diarize,
        patch("main.whisperx.align", return_value={"segments": fake_segments, "word_segments": []}),
        patch("main.whisperx.assign_word_speakers", return_value={"segments": fake_segments, "word_segments": []}),
    ):
        mock_align.return_value = (MagicMock(), MagicMock())
        diarize_instance = MagicMock()
        mock_diarize.return_value = diarize_instance

        client.post(
            "/transcribe",
            files=[_audio_upload()],
            data={"min_speakers": "2", "max_speakers": "4"},
        )

    diarize_instance.assert_called_once()
    call_kwargs = diarize_instance.call_args
    assert call_kwargs.kwargs.get("min_speakers") == 2 or (len(call_kwargs.args) > 1 and call_kwargs.args[1] == 2)
    assert call_kwargs.kwargs.get("max_speakers") == 4 or (len(call_kwargs.args) > 2 and call_kwargs.args[2] == 4)
