"""
text_to_speech.py
Converts text answers to speech using gTTS (Google Text-to-Speech).
Fully free, no API key required.
"""

import os
import tempfile
import io


def text_to_speech_bytes(text: str, lang: str = "en") -> bytes | None:
    """
    Convert text to MP3 audio bytes using gTTS.

    Args:
        text: Text to speak
        lang: Language code (default: 'en')

    Returns:
        MP3 audio as bytes, or None on failure
    """
    try:
        from gtts import gTTS

        # Truncate very long answers for TTS (keep first 1000 chars)
        tts_text = text[:1000] + ("..." if len(text) > 1000 else "")

        tts = gTTS(text=tts_text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()

    except ImportError:
        print("[TTS] gTTS not installed. Run: pip install gTTS")
        return None
    except Exception as e:
        print(f"[TTS] Error generating speech: {e}")
        return None


def text_to_speech_file(text: str, output_path: str = "response.mp3", lang: str = "en") -> str | None:
    """
    Convert text to an MP3 file.

    Args:
        text: Text to convert
        output_path: Where to save the MP3
        lang: Language code

    Returns:
        Path to the saved MP3, or None on failure
    """
    try:
        from gtts import gTTS

        tts_text = text[:1000] + ("..." if len(text) > 1000 else "")
        tts = gTTS(text=tts_text, lang=lang, slow=False)
        tts.save(output_path)
        print(f"[TTS] Audio saved to: {output_path}")
        return output_path

    except ImportError:
        print("[TTS] gTTS not installed. Run: pip install gTTS")
        return None
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None


def is_tts_available() -> bool:
    """Check if gTTS is installed."""
    try:
        from gtts import gTTS
        return True
    except ImportError:
        return False