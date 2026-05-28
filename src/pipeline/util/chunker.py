from __future__ import annotations


def _approx_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token (sufficient for chunk sizing)."""
    return max(1, len(text) // 4)


def _dominant_speaker(segments: list[dict]) -> str | None:
    """Return the speaker label with the most cumulative duration in segments."""
    totals: dict[str, float] = {}
    for seg in segments:
        label = seg.get("speaker")
        if label:
            duration = seg.get("end", 0) - seg.get("start", 0)
            totals[label] = totals.get(label, 0.0) + duration
    return max(totals, key=totals.__getitem__) if totals else None


def chunk_segments(
    segments: list[dict],
    target_tokens: int = 350,
    overlap_tokens: int = 50,
) -> list[dict]:
    """
    Group WhisperX segments into chunks suitable for embedding.

    Rules:
    - Never split mid-segment (respect segment boundaries)
    - Target ~target_tokens tokens per chunk
    - Prepend last overlap_tokens of previous chunk text to the next chunk
    - Dominant speaker = speaker with most cumulative duration in chunk
    """
    if not segments:
        return []

    chunks: list[dict] = []
    current_segs: list[dict] = []
    current_tokens = 0
    overlap_tail = ""  # text carried over from previous chunk

    def flush(segs: list[dict], tail: str) -> dict:
        text = (tail + " " + " ".join(s.get("text", "").strip() for s in segs)).strip()
        return {
            "chunk_index": len(chunks),
            "text": text,
            "speaker_label": _dominant_speaker(segs),
            "start_time": segs[0].get("start", 0.0),
            "end_time": segs[-1].get("end", 0.0),
        }

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        seg_tokens = _approx_tokens(seg_text)

        if current_segs and (current_tokens + seg_tokens) > target_tokens:
            chunks.append(flush(current_segs, overlap_tail))

            # Build overlap tail from end of the chunk we just flushed
            all_text = " ".join(s.get("text", "").strip() for s in current_segs)
            words = all_text.split()
            # Keep last N words approximating overlap_tokens
            overlap_word_count = max(1, overlap_tokens * 4 // 5)  # ~5 chars/word
            overlap_tail = " ".join(words[-overlap_word_count:]) if words else ""

            current_segs = [seg]
            current_tokens = _approx_tokens(overlap_tail) + seg_tokens
        else:
            current_segs.append(seg)
            current_tokens += seg_tokens

    if current_segs:
        chunks.append(flush(current_segs, overlap_tail))

    return chunks
