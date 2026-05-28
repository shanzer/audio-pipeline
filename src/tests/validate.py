#!/usr/bin/env python3
"""
Standalone WhisperX validation script.
Run before building pipeline code to confirm output schema and diarization quality.

Usage:
    python3 tests/validate.py /path/to/audio.m4a [--min-speakers 1] [--max-speakers 6] [--device cpu]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import os

# Load .env from project root (one level up from tests/)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print(f"ERROR: HF_TOKEN not set in {_env_path}", file=sys.stderr)
    sys.exit(1)

# huggingface_hub 1.0+ replaced use_auth_token with token. Patch hf_hub_download
# so old pyannote/whisperx code that still passes use_auth_token keeps working.
import huggingface_hub as _hfhub
_orig_hf_hub_download = _hfhub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _orig_hf_hub_download(*args, **kwargs)
_hfhub.hf_hub_download = _patched_hf_hub_download

# torchaudio 2.6+ removed AudioMetaData, info(), and list_audio_backends().
# Patch them back using soundfile so pyannote.audio 3.x keeps working.
import torchaudio
import soundfile as sf

if not hasattr(torchaudio, "AudioMetaData"):
    from dataclasses import dataclass

    @dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "info"):
    def _torchaudio_info(path, backend=None):
        i = sf.info(path)
        return torchaudio.AudioMetaData(
            sample_rate=i.samplerate,
            num_frames=i.frames,
            num_channels=i.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )
    torchaudio.info = _torchaudio_info

# PyTorch 2.6+ defaults weights_only=True, which blocks omegaconf globals in
# pyannote/speechbrain checkpoints. Patch torch.load to use weights_only=False
# for trusted local model files loaded by pyannote and lightning.
import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)
_torch.load = _patched_torch_load


def parse_args():
    p = argparse.ArgumentParser(description="Validate WhisperX pipeline output")
    p.add_argument("audio_file", help="Path to audio file")
    p.add_argument("--min-speakers", type=int, default=1)
    p.add_argument("--max-speakers", type=int, default=6)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    return p.parse_args()


def fmt(elapsed: float) -> str:
    return f"{elapsed:.2f}s"


def main():
    args = parse_args()
    audio_path = Path(args.audio_file).expanduser().resolve()

    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Audio file : {audio_path}")
    print(f"Device     : {args.device}")
    print(f"Speakers   : {args.min_speakers}–{args.max_speakers}")
    print(f"HF token   : {HF_TOKEN[:8]}…")
    print()

    import whisperx

    compute_type = "int8" if args.device == "cpu" else "float16"

    # Stage 1: Transcribe
    print("Loading WhisperX model (large-v3)…")
    t0 = time.perf_counter()
    model = whisperx.load_model("large-v3", device=args.device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=8)
    t_transcribe = time.perf_counter() - t0
    print(f"  transcribe : {fmt(t_transcribe)}  ({len(result['segments'])} segments, lang={result.get('language')})")

    # Stage 2: Align
    t1 = time.perf_counter()
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=args.device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, args.device, return_char_alignments=False
    )
    t_align = time.perf_counter() - t1
    print(f"  align      : {fmt(t_align)}")

    # Stage 3: Diarize
    t2 = time.perf_counter()
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=args.device)
    diarize_segments = diarize_model(
        audio,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)
    t_diarize = time.perf_counter() - t2
    print(f"  diarize    : {fmt(t_diarize)}")

    segments = result["segments"]
    total_duration = segments[-1]["end"] if segments else 0.0
    speaker_labels = {s.get("speaker") for s in segments if s.get("speaker")}

    print()
    print("=== First 5 segments ===")
    for seg in segments[:5]:
        print(json.dumps({
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "speaker": seg.get("speaker"),
            "text": seg.get("text", "").strip(),
        }, indent=2))

    print()
    print("=== Summary ===")
    print(f"  Total segments   : {len(segments)}")
    print(f"  Unique speakers  : {sorted(speaker_labels) if speaker_labels else 'NONE (diarization may have failed)'}")
    print(f"  Total duration   : {total_duration:.1f}s")
    print(f"  Transcribe time  : {fmt(t_transcribe)}")
    print(f"  Align time       : {fmt(t_align)}")
    print(f"  Diarize time     : {fmt(t_diarize)}")

    output_path = Path(__file__).parent / "validate_output.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull output written to: {output_path}")

    if not speaker_labels:
        print("\nWARNING: No speaker labels assigned. Check HF token and pyannote terms acceptance.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
