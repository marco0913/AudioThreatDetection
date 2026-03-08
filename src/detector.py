"""Threat detector -- deterministic heuristics + LLM structured reasoning.

Two-stage detection:
1. Deterministic keyword/pattern matching (fast, predictable)
2. LLM structured output for semantic reasoning (slower, nuanced)

The LLM receives the transcript + audio features and returns a
schema-validated DetectorOutput via OpenAI Structured Outputs.
"""

import os
import re
from typing import Any

from openai import AsyncOpenAI

from src.models import DetectorOutput, TranscriptResult, TurnAudioFeatures

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


# --- Deterministic keyword heuristics ---

# Threat keywords grouped by category.
# These are chosen to cover the corpus scenarios:
#   - Direct threats ("hurt you", "kill you")
#   - Distress signals ("help me", "stop it")
#   - Harassment patterns ("give me your", profanity + threat combos)
THREAT_KEYWORDS: dict[str, list[str]] = {
    "physical_threat": [
        r"\b(i('ll| will))\s+(hurt|kill|stab|hit|punch|shoot|cut)\s+(you|him|her)\b",
        r"\bhurt you\b",
        r"\bkill you\b",
        r"\bbeat you\b",
    ],
    "verbal_threat": [
        r"\bi('m| am)\s+(going to|gonna)\s+(hurt|kill|destroy|end)\b",
        r"\byou('re| are)\s+(dead|done|finished)\b",
        r"\bi('ll| will)\s+make you\s+(pay|suffer|regret)\b",
    ],
    "distress_signal": [
        r"\bhelp\s*me\b",
        r"\bsomebody help\b",
        r"\bstop\s*(it)?\s*please\b",
        r"\blet\s*me\s*go\b",
        r"\bdon'?t\s+touch\s+me\b",
    ],
    "harassment": [
        r"\bgive\s+me\s+(your|that|the)\b.*\bor\b",
        r"\bi('ll| will)\s+hurt\b",
    ],
}


def _run_keyword_heuristics(text: str) -> tuple[list[str], str | None]:
    """Run deterministic keyword matching against cleaned transcript.

    Returns (matched_keywords, best_category_or_None).
    """
    text_lower = text.lower()
    matches: list[str] = []
    categories_hit: list[str] = []

    for category, patterns in THREAT_KEYWORDS.items():
        for pattern in patterns:
            found = re.findall(pattern, text_lower)
            if found:
                matches.append(pattern.replace(r"\b", "").replace("\\", ""))
                if category not in categories_hit:
                    categories_hit.append(category)

    best_category = categories_hit[0] if categories_hit else None
    return matches, best_category


# --- LLM structured output ---

SYSTEM_PROMPT = """You are a threat detection system for a personal safety wearable device (badge).

The badge is worn by a person. Audio is captured from their immediate environment.
Analyze the transcript of a short audio turn and assess whether it represents a safety concern.

CRITICAL distinction -- is_directed:
- is_directed=true: The speech involves the wearer directly. This includes:
  - Someone speaking TO the wearer using "you" (threats, intimidation, ultimatums)
  - Someone arguing WITH the wearer face-to-face
  - The wearer being confronted, cornered, or coerced
- is_directed=false: The speech is AMBIENT -- it does NOT involve the wearer. This includes:
  - TV, radio, or media audio (sports commentary, movie dialogue, news)
  - Overheard arguments between OTHER people nearby
  - Background noise without interpersonal engagement with the wearer

KEY RULE: If the transcript uses second-person language ("you promised", "give me your", "I'll hurt you") in a conversational context (not clearly media/TV), treat it as DIRECTED. The badge captures nearby speech -- if someone is saying "you" aggressively, they are likely talking to the wearer.

Threat categories:
- verbal_threat: Aggressive speech with intent to harm or intimidate
- physical_threat: Explicit threat of physical violence
- distress_signal: The wearer or someone nearby calling for help
- harassment: Coercion, demands, or persistent unwanted aggression
- ambient_anger: Background anger not directed at the wearer (TV, sports, etc.)
- none: No safety concern

Audio context:
- rms_db: loudness (speech is typically -20 to -30dB; shouting is > -18dB)
- abrupt_change_score: sudden loudness spikes (0-1; shouting/slamming > 0.4)
- zero_crossing_rate: noise (higher = noisier)

Be appropriately sensitive. Heated face-to-face confrontations with ultimatums or intimidation ARE safety concerns even without explicit threat words. TV/media anger is NOT."""

USER_PROMPT_TEMPLATE = """Transcript: "{transcript}"

Audio features:
- Loudness (RMS): {rms_db:.1f} dB
- Abrupt change score: {abrupt_change:.3f}
- Zero-crossing rate: {zcr:.4f}
- Duration: {duration:.1f}s

Keyword matches from deterministic scan: {keyword_matches}

Assess this turn for threat signals."""


# JSON schema for OpenAI Structured Outputs
DETECTOR_JSON_SCHEMA: dict[str, Any] = {
    "name": "detector_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "threat_score": {
                "type": "number",
                "description": "Threat level from 0.0 (safe) to 1.0 (extreme danger)",
            },
            "threat_category": {
                "type": "string",
                "enum": [
                    "none",
                    "verbal_threat",
                    "physical_threat",
                    "distress_signal",
                    "harassment",
                    "ambient_anger",
                    "unknown",
                ],
                "description": "Category of detected threat, or 'none' if safe",
            },
            "is_directed": {
                "type": "boolean",
                "description": "True if the threat is directed at the badge wearer, False if ambient (TV, background)",
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentence explanation of the assessment for a human operator",
            },
            "llm_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Semantic signals identified (e.g. 'direct_threat', 'profanity', 'ultimatum')",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the assessment from 0.0 to 1.0",
            },
        },
        "required": [
            "threat_score",
            "threat_category",
            "is_directed",
            "reasoning",
            "llm_flags",
            "confidence",
        ],
        "additionalProperties": False,
    },
}


async def _llm_detect(
    transcript: TranscriptResult,
    audio_features: TurnAudioFeatures,
    keyword_matches: list[str],
) -> dict:
    """Call OpenAI with structured output to assess threat."""
    client = _get_client()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        transcript=transcript.cleaned_text,
        rms_db=audio_features.rms_db,
        abrupt_change=audio_features.abrupt_change_score,
        zcr=audio_features.zero_crossing_rate,
        duration=audio_features.duration_seconds,
        keyword_matches=keyword_matches if keyword_matches else "none",
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": DETECTOR_JSON_SCHEMA,
        },
        temperature=0.1,  # Low temperature for determinism
        max_tokens=300,
    )

    import json

    return json.loads(response.choices[0].message.content)


# --- Combined detection ---


async def detect_threat(
    transcript: TranscriptResult,
    audio_features: TurnAudioFeatures,
) -> DetectorOutput:
    """Run two-stage threat detection: heuristics then LLM.

    If the transcript is empty, returns a default no-threat result
    (graceful degradation for muffled/unintelligible audio).
    """
    # Graceful degradation: empty transcript
    if transcript.is_empty:
        return DetectorOutput(
            threat_score=0.1 if audio_features.rms_db > -15 else 0.0,
            threat_category="unknown" if audio_features.rms_db > -15 else "none",
            is_directed=False,
            reasoning="Transcript empty or unintelligible. Audio-only assessment.",
            confidence=0.2,
        )

    # Stage 1: Deterministic keyword matching
    keyword_matches, keyword_category = _run_keyword_heuristics(
        transcript.cleaned_text
    )

    # Stage 2: LLM structured reasoning
    try:
        llm_result = await _llm_detect(transcript, audio_features, keyword_matches)
    except Exception as e:
        print(f"[DETECT] LLM error: {e}")
        # Fallback to keyword-only result
        if keyword_matches:
            return DetectorOutput(
                threat_score=0.7,
                threat_category=keyword_category or "unknown",
                is_directed=True,
                reasoning=f"LLM unavailable. Keyword match: {keyword_matches}",
                keywords_matched=keyword_matches,
                confidence=0.5,
            )
        return DetectorOutput(
            reasoning=f"LLM unavailable, no keyword matches. Error: {e}",
        )

    return DetectorOutput(
        threat_score=llm_result["threat_score"],
        threat_category=llm_result["threat_category"],
        is_directed=llm_result["is_directed"],
        reasoning=llm_result["reasoning"],
        keywords_matched=keyword_matches,
        llm_flags=llm_result["llm_flags"],
        confidence=llm_result["confidence"],
    )
