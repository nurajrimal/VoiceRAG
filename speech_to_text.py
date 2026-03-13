"""
speech_to_text.py
Converts audio input to text using OpenAI Whisper (runs 100% locally).
"""

import os
import tempfile
import numpy as np


def transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw audio bytes using Whisper.

    Args:
        audio_bytes: Raw PCM audio bytes
        sample_rate: Sample rate of the audio

    Returns:
        Transcribed text string
    """
    try:
        import whisper
        import soundfile as sf

        # Write audio bytes to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sf.write(tmp_path, audio_array, sample_rate)

        # Load Whisper model (tiny is fast; use "base" or "small" for more accuracy)
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path, language="en", fp16=False)
        text = result["text"].strip()

        os.unlink(tmp_path)
        print(f"[STT] Transcribed: '{text}'")
        return text

    except ImportError:
        return "[Error] Whisper not installed. Run: pip install openai-whisper"
    except Exception as e:
        return f"[Error] Transcription failed: {str(e)}"


def transcribe_audio_file(audio_path: str) -> str:
    """
    Transcribe an audio file (WAV, MP3, M4A, etc.) using Whisper.

    Args:
        audio_path: Path to the audio file

    Returns:
        Transcribed text
    """
    try:
        import whisper

        if not os.path.exists(audio_path):
            return f"[Error] Audio file not found: {audio_path}"

        print(f"[STT] Transcribing file: {audio_path}")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language="en", fp16=False)
        text = result["text"].strip()
        print(f"[STT] Transcribed: '{text}'")
        return text

    except ImportError:
        return "[Error] Whisper not installed. Run: pip install openai-whisper"
    except Exception as e:
        return f"[Error] Transcription failed: {str(e)}"


def is_whisper_available() -> bool:
    """Check if Whisper is installed and importable."""
    try:
        import whisper
        return True
    except ImportError:
        return False