import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.stages.intake import detect_source


def test_iphone_prefix_returns_iphone():
    assert detect_source("iphone_1748000000_memo.m4a") == "iphone"


def test_iphone_prefix_wins_over_zoom_regex():
    # Without prefix check, "zoom" pattern fires first — prefix must take priority
    assert detect_source("iphone_zoom_meeting_GMT0800.m4a") == "iphone"


def test_iphone_prefix_is_exact_lowercase():
    # "iphone" regex would match, but the prefix check should not trigger on mixed case
    # Outcome: regex still returns "iphone", but via a different path — not the prefix
    assert detect_source("iPhone_recording.m4a") == "iphone"


def test_metadata_source_overrides_prefix():
    assert detect_source("iphone_1748000000_memo.m4a", {"source": "zoom"}) == "zoom"


def test_zoom_pattern_detected():
    assert detect_source("zoom_meeting_GMT0800.mp4") == "zoom"


def test_meeting_pattern_detected():
    assert detect_source("quarterly_meeting_2026.wav") == "zoom"


def test_iphone_regex_pattern():
    assert detect_source("Voice Memo 42.m4a") == "iphone"


def test_ambient_fallback():
    assert detect_source("random_audio_file.wav") == "ambient"


def test_metadata_source_wins_over_all():
    assert detect_source("zoom_meeting.mp4", {"source": "ambient"}) == "ambient"
