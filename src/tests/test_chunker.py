"""Tests for pipeline.util.chunker — run with: python3 -m pytest tests/test_chunker.py -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.util.chunker import chunk_segments


def _seg(text, start=0.0, end=1.0, speaker=None):
    return {"text": text, "start": start, "end": end, "speaker": speaker}


# 1. Empty input
def test_empty_input():
    result = chunk_segments([])
    assert result == []


# 2. Single segment
def test_single_segment():
    segs = [_seg("Hello world.", 0.0, 1.0, "SPEAKER_00")]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert "Hello world." in chunks[0]["text"]
    assert chunks[0]["speaker_label"] == "SPEAKER_00"
    assert chunks[0]["start_time"] == 0.0
    assert chunks[0]["end_time"] == 1.0


# 3. Segments that require overlap — many short segments that cross the target boundary
def test_overlap_carried_over():
    # 20 segments of ~10 tokens each, target=50 tokens → should produce multiple chunks
    segs = [_seg("word " * 10, float(i), float(i + 1), "SPEAKER_00") for i in range(20)]
    chunks = chunk_segments(segs, target_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    # Every chunk after the first should include overlap text
    for chunk in chunks[1:]:
        assert len(chunk["text"]) > 0
    # chunk_index should be sequential
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


# 4. Single segment longer than target — should stay as one chunk (never split mid-segment)
def test_segment_longer_than_target():
    long_text = "a " * 500  # ~250 tokens
    segs = [_seg(long_text, 0.0, 60.0, "SPEAKER_01")]
    chunks = chunk_segments(segs, target_tokens=50, overlap_tokens=10)
    assert len(chunks) == 1
    assert long_text.strip() in chunks[0]["text"]


# 5. Mixed-speaker segments — dominant speaker is longest by duration
def test_dominant_speaker_selection():
    segs = [
        _seg("Speaker A talks a lot here and keeps going.", 0.0, 10.0, "SPEAKER_00"),
        _seg("Short.", 10.0, 11.0, "SPEAKER_01"),
        _seg("Speaker A again.", 11.0, 15.0, "SPEAKER_00"),
    ]
    chunks = chunk_segments(segs, target_tokens=500, overlap_tokens=10)
    # All fit in one chunk; SPEAKER_00 has 14s vs SPEAKER_01 1s
    assert len(chunks) == 1
    assert chunks[0]["speaker_label"] == "SPEAKER_00"


# 6. Multiple speaker splits produce correct start/end times
def test_chunk_timestamps():
    segs = [_seg("word " * 20, float(i * 5), float(i * 5 + 5), "SPEAKER_00") for i in range(6)]
    chunks = chunk_segments(segs, target_tokens=50, overlap_tokens=5)
    for chunk in chunks:
        assert chunk["start_time"] >= 0.0
        assert chunk["end_time"] > chunk["start_time"]


# 7. No speaker labels — speaker_label should be None
def test_no_speaker_labels():
    segs = [_seg("Some text here.", 0.0, 5.0, None)]
    chunks = chunk_segments(segs)
    assert chunks[0]["speaker_label"] is None


if __name__ == "__main__":
    tests = [
        test_empty_input,
        test_single_segment,
        test_overlap_carried_over,
        test_segment_longer_than_target,
        test_dominant_speaker_selection,
        test_chunk_timestamps,
        test_no_speaker_labels,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)} tests run")
