import json
import logging
from pathlib import Path

import ollama

from config.settings import LLM_MODEL, OLLAMA_BASE_URL

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "summarize.txt"
_REQUIRED_KEYS = {"title", "topics", "decisions", "action_items", "risks"}


def build_transcript(segments: list[dict]) -> str:
    """Format segments as a readable transcript with speaker labels and timestamps."""
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or "UNKNOWN"
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        lines.append(f"[{start:.1f}s] {speaker}: {text}")
    return "\n".join(lines)


def _call_llm(prompt: str, model: str) -> str:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    resp = client.generate(model=model, prompt=prompt, stream=False)
    return resp["response"].strip()


def _parse_response(text: str) -> dict:
    # Strip markdown fences if present
    cleaned = text
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    return json.loads(cleaned)


def run_summary(transcript: str, model_name: str = LLM_MODEL) -> dict:
    """
    Send transcript to Ollama LLM and return parsed summary dict.
    Retries once with explicit JSON-only instruction if first parse fails.
    Raises ValueError if both attempts fail to parse.
    """
    template = _PROMPT_PATH.read_text()
    prompt = template.replace("{transcript}", transcript)

    for attempt in (1, 2):
        if attempt == 2:
            prompt = (
                "Return ONLY a valid JSON object with no other text, "
                "no explanation, no markdown:\n\n" + prompt
            )
        raw = _call_llm(prompt, model_name)
        log.debug("LLM raw response (attempt %d): %s", attempt, raw[:200])

        try:
            parsed = _parse_response(raw)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                raise ValueError(f"LLM failed to return valid JSON after 2 attempts: {e}") from e
            log.warning("JSON parse failed on attempt 1, retrying: %s", e)
            continue

        missing = _REQUIRED_KEYS - set(parsed.keys())
        if missing:
            if attempt == 2:
                raise ValueError(f"LLM response missing required keys: {missing}")
            log.warning("Response missing keys %s, retrying", missing)
            continue

        return parsed

    raise ValueError("run_summary: unreachable")
