"""Transcriber -- converts audio turns to text using OpenAI Whisper API.

Writes the PCM turn to a temporary WAV file, sends it to whisper-1,
then cleans the raw transcript for downstream reasoning.
"""

import io
import os
import re
import wave

from openai import AsyncOpenAI

from src.models import TranscriptResult

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _clean_transcript(raw: str) -> str:
    """Clean raw Whisper output for downstream reasoning.

    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple spaces
    - Remove common disfluencies (um, uh, hmm)
    """
    text = raw.lower().strip()
    # Remove disfluencies
    text = re.sub(r"\b(um|uh|uhh|hmm|hm|erm|ah)\b", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def transcribe(
    audio_bytes: bytes, sample_rate: int = 16000
) -> TranscriptResult:
    """Transcribe an audio turn via OpenAI Whisper API."""
    if len(audio_bytes) < 1000:
        return TranscriptResult(
            raw_text="",
            cleaned_text="",
            is_empty=True,
            word_count=0,
        )

    wav_data = _pcm_to_wav_bytes(audio_bytes, sample_rate)
    client = _get_client()

    try:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=("turn.wav", wav_data, "audio/wav"),
            response_format="verbose_json",
            language="en",
        )

        raw_text = response.text if hasattr(response, "text") else str(response)
        cleaned = _clean_transcript(raw_text)

        return TranscriptResult(
            raw_text=raw_text,
            cleaned_text=cleaned,
            language=getattr(response, "language", "en"),
            word_count=len(cleaned.split()) if cleaned else 0,
            is_empty=len(cleaned.strip()) == 0,
        )
    except Exception as e:
        print(f"[TRANSCRIBE] ERROR: {e}")
        return TranscriptResult(
            raw_text=f"[error: {e}]",
            cleaned_text="",
            is_empty=True,
            word_count=0,
        )
