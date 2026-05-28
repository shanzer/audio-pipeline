import logging
import time

import ollama

from config.settings import EMBED_MODEL, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS, OLLAMA_BASE_URL
from pipeline.util.chunker import chunk_segments

log = logging.getLogger(__name__)


def chunk_and_embed(
    segments: list[dict],
    model_name: str = EMBED_MODEL,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Chunk segments and embed each chunk using Ollama.
    Returns list of chunk dicts with 'embedding' field added.
    """
    chunks = chunk_segments(segments, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        log.warning("No chunks produced from %d segments", len(segments))
        return []

    log.info("Embedding %d chunks with model=%s", len(chunks), model_name)
    client = ollama.Client(host=OLLAMA_BASE_URL)

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        for attempt in (1, 2):
            try:
                t0 = time.perf_counter()
                resp = client.embeddings(model=model_name, prompt=text)
                chunk["embedding"] = resp["embedding"]
                log.debug("Chunk %d embedded in %.2fs", i, time.perf_counter() - t0)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Embedding failed for chunk {i} after retry: {e}") from e
                log.warning("Embedding chunk %d failed (attempt %d): %s — retrying", i, attempt, e)
                time.sleep(1)

    return chunks
