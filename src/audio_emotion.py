"""GPT-4o audio emotion analysis -- bonus path A.

Sends raw PCM audio directly to GPT-4o-audio-preview to extract prosodic
signals that text transcription loses: anger intensity, urgency, stress level.

This supplements the existing text-based LLM detector. It does NOT replace
the Whisper + gpt-4o-mini pipeline -- it runs alongside it as an additional
signal layer.

Usage
-----
Set ENABLE_AUDIO_EMOTION=true in .env to activate. The function returns None
when disabled or on API error, so the pipeline degrades gracefully.

Output schema
-------------
    anger_level:      0.0-1.0  (0=calm, 1=shouting/furious)
    urgency:          0.0-1.0  (0=relaxed, 1=panicked/highly urgent)
    stress_level:     0.0-1.0  (0=relaxed speech, 1=extreme stress)
    is_clearly_speech: bool    (False for noise/music/silence)
    emotion_label:    str      one of: calm, neutral, tense, angry, distressed
    reasoning:        str      one sentence describing dominant tone
"""

import base64
import io
import json
import os
import wave

from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container for the API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


EMOTION_PROMPT = """Listen to this audio carefully and analyze the emotional tone and intensity.

Return a JSON object with exactly these fields:
{
  "anger_level": <float 0.0-1.0, where 0=completely calm, 1=shouting/furious>,
  "urgency": <float 0.0-1.0, where 0=relaxed pace, 1=panicked/highly urgent>,
  "stress_level": <float 0.0-1.0, where 0=relaxed natural speech, 1=extreme vocal stress>,
  "is_clearly_speech": <boolean, true if clearly human speech, false if music/noise/silence>,
  "emotion_label": <string, exactly one of: "calm", "neutral", "tense", "angry", "distressed">,
  "reasoning": <string, one sentence describing the dominant emotional tone you hear>
}

Focus on vocal qualities: pitch variation, speaking pace, volume, vocal tension.
A raised voice or fast agitated speech should score high on urgency/stress.
Shouted threats or extreme anger should score high on anger_level.
TV audio or background music should have is_clearly_speech=false.

Respond with only valid JSON, no markdown."""


async def analyze_audio_emotion(
    audio_bytes: bytes,
    sample_rate: int = 16000,
) -> dict | None:
    """Analyze audio with GPT-4o for prosodic emotion signals.

    Returns a dict with anger_level, urgency, stress_level, is_clearly_speech,
    emotion_label, and reasoning. Returns None if disabled or on error.

    Activation: set ENABLE_AUDIO_EMOTION=true in environment.
    """
    if os.getenv("ENABLE_AUDIO_EMOTION", "").lower() != "true":
        return None

    if len(audio_bytes) < 1000:
        return None

    try:
        wav_bytes = _pcm_to_wav_bytes(audio_bytes, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        client = _get_client()
        response = await client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": EMOTION_PROMPT,
                        },
                    ],
                }
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None

        # Strip optional markdown code fences (```json ... ```)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        raw = json.loads(content)

        # Normalise and clamp all numeric fields
        return {
            "anger_level": float(min(max(raw.get("anger_level", 0.0), 0.0), 1.0)),
            "urgency": float(min(max(raw.get("urgency", 0.0), 0.0), 1.0)),
            "stress_level": float(min(max(raw.get("stress_level", 0.0), 0.0), 1.0)),
            "is_clearly_speech": bool(raw.get("is_clearly_speech", True)),
            "emotion_label": str(raw.get("emotion_label", "neutral")),
            "reasoning": str(raw.get("reasoning", "")),
        }

    except json.JSONDecodeError:
        # Model returned non-JSON (rare) — degrade silently
        return None
    except Exception as e:
        print(f"[AUDIO EMOTION] GPT-4o audio error: {e}")
        return None
