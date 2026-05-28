import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def wav_file(tmp_path):
    p = tmp_path / "test.wav"
    p.write_bytes(b"RIFF" + b"\x00" * 44)
    return str(p)


def test_run_whisperx_posts_to_service(wav_file):
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello", "speaker": "SPEAKER_00", "words": []}],
        "language": "en",
        "word_segments": [],
    }
    fake_response.raise_for_status = MagicMock()

    with patch("pipeline.stages.transcribe.requests.post", return_value=fake_response) as mock_post:
        from pipeline.stages.transcribe import run_whisperx
        result = run_whisperx(wav_file, min_speakers=1, max_speakers=3)

    assert result["language"] == "en"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "hello"

    call_args = mock_post.call_args
    url = call_args[0][0]
    assert "8765" in url
    assert "transcribe" in url
    assert call_args[1]["data"]["min_speakers"] == 1
    assert call_args[1]["data"]["max_speakers"] == 3
    assert call_args[1]["timeout"] == 3600


def test_run_whisperx_raises_on_http_error(wav_file):
    import requests as req
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = req.HTTPError("500 Server Error")

    with patch("pipeline.stages.transcribe.requests.post", return_value=fake_response):
        from pipeline.stages.transcribe import run_whisperx
        with pytest.raises(req.HTTPError):
            run_whisperx(wav_file)


def test_run_whisperx_uses_settings_url(wav_file):
    fake_response = MagicMock()
    fake_response.json.return_value = {"segments": [], "language": "en", "word_segments": []}
    fake_response.raise_for_status = MagicMock()

    with (
        patch("pipeline.stages.transcribe.requests.post", return_value=fake_response) as mock_post,
        patch("pipeline.stages.transcribe.WHISPER_SERVICE_URL", "http://myhost:9999"),
    ):
        from pipeline.stages.transcribe import run_whisperx
        run_whisperx(wav_file)

    url = mock_post.call_args[0][0]
    assert "myhost" in url
    assert "9999" in url
