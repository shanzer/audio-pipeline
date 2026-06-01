import sys
from unittest.mock import MagicMock
from pathlib import Path

# Mock all heavy ML deps before any import of main
for _mod in [
    "torch", "torchaudio", "soundfile", "huggingface_hub",
    "whisperx", "whisperx.diarize",
    "pyannote", "pyannote.audio",
    "mlx_whisper",
]:
    sys.modules.setdefault(_mod, MagicMock())

# Also mock dotenv so load_dotenv() is a no-op
sys.modules.setdefault("dotenv", MagicMock())

# Ensure whisper-service root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
