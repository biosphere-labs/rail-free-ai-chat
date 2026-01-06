import asyncio
from pathlib import Path
from typing import AsyncGenerator
import edge_tts

DEFAULT_VOICE = "en-US-AriaNeural"

# Popular voices for quick selection
POPULAR_VOICES = [
    ("en-US-AriaNeural", "Aria (US Female)"),
    ("en-US-GuyNeural", "Guy (US Male)"),
    ("en-US-JennyNeural", "Jenny (US Female)"),
    ("en-GB-SoniaNeural", "Sonia (UK Female)"),
    ("en-GB-RyanNeural", "Ryan (UK Male)"),
    ("en-AU-NatashaNeural", "Natasha (AU Female)"),
    ("en-IN-NeerjaNeural", "Neerja (IN Female)"),
]


async def speak(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Convert text to speech using Edge TTS.

    Args:
        text: The text to convert to speech.
        voice: The voice ID to use (e.g., "en-US-AriaNeural").

    Returns:
        Audio data as bytes (MP3 format).
    """
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    return audio_data


async def speak_stream(text: str, voice: str = DEFAULT_VOICE) -> AsyncGenerator[bytes, None]:
    """Stream text to speech chunks using Edge TTS.

    Args:
        text: The text to convert to speech.
        voice: The voice ID to use.

    Yields:
        Audio data chunks as bytes (MP3 format).
    """
    communicate = edge_tts.Communicate(text, voice)

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def speak_to_file(text: str, output_path: Path, voice: str = DEFAULT_VOICE) -> None:
    """Stream text to speech directly to a file.

    Args:
        text: The text to convert to speech.
        output_path: Path to write the MP3 file.
        voice: The voice ID to use.
    """
    communicate = edge_tts.Communicate(text, voice)

    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
                f.flush()  # Flush each chunk so it's readable immediately


async def list_voices(language_filter: str = "en") -> list[dict]:
    """List available voices, optionally filtered by language.

    Args:
        language_filter: Language code prefix to filter by (e.g., "en", "es").

    Returns:
        List of voice dictionaries with Name, ShortName, Gender, Locale.
    """
    voices = await edge_tts.list_voices()

    if language_filter:
        voices = [
            v for v in voices if v["Locale"].lower().startswith(language_filter.lower())
        ]

    return voices


def speak_sync(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Synchronous wrapper for speak().

    Args:
        text: The text to convert to speech.
        voice: The voice ID to use.

    Returns:
        Audio data as bytes (MP3 format).
    """
    return asyncio.run(speak(text, voice))
