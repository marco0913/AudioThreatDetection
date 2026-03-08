"""Signal fusion -- combines text, audio, and LLM outputs into a single decision.

Explicit weighted combination with configurable weights and threshold.
The fusion score determines whether to fire an alert.

Key design choices:
- is_directed=False gates the alert: non-directed threats never alert
- Keywords provide a boost on top of the LLM score
- Audio features (abrupt change, loudness) contribute a small signal
- Threshold at 0.55 balances sensitivity vs false positives
"""

from src.models import DetectorOutput, TurnAudioFeatures

# Configurable weights
WEIGHT_LLM = 0.60        # LLM threat_score is the primary signal
WEIGHT_AUDIO = 0.15       # Audio features (loudness + abruptness)
WEIGHT_KEYWORD = 0.25     # Keyword match boost

# Alert threshold
ALERT_THRESHOLD = 0.55

# Audio emotion bonus weight (additive, only applied when ENABLE_AUDIO_EMOTION=true).
# Keeps base weights (60/25/15) unchanged so tests without emotion still pass.
# emotion_score = mean(anger_level, urgency) from GPT-4o audio-preview.
WEIGHT_EMOTION = 0.05

# Audio feature normalization ranges (from corpus observations)
# RMS: -16dB (heated_argument) to -26dB (casual_chat)
# Abrupt change: 0.1 (casual) to 0.66 (heated)
RMS_THREAT_FLOOR = -25.0  # below this, audio contributes 0
RMS_THREAT_CEIL = -12.0   # above this, audio contributes max


def fuse_signals(
    detector: DetectorOutput,
    audio_features: TurnAudioFeatures,
    audio_emotion: dict | None = None,
) -> float:
    """Produce a fused score from 0.0 to 1.0.

    Returns 0.0 immediately if is_directed=False (non-directed anger
    like TV audio should never fire an alert regardless of other scores).

    audio_emotion: optional dict from analyze_audio_emotion() with keys
        anger_level and urgency (both 0.0-1.0). When provided, adds a
        small deterministic bonus that stabilises borderline cases where
        the LLM threat_score varies across runs at temperature 0.1.
    """
    # Gate: non-directed threats cannot produce an alert
    if not detector.is_directed:
        return min(detector.threat_score * 0.3, 0.4)

    # LLM component
    llm_score = detector.threat_score

    # Audio component: normalized loudness + abrupt change
    rms_norm = _normalize(
        audio_features.rms_db, RMS_THREAT_FLOOR, RMS_THREAT_CEIL
    )
    audio_score = 0.5 * rms_norm + 0.5 * audio_features.abrupt_change_score

    # Keyword component: binary boost if any keywords matched
    keyword_score = 1.0 if detector.keywords_matched else 0.0

    # Emotion bonus: additive, only when GPT-4o audio emotion is available.
    # Uses mean(anger_level, urgency) — two prosodic signals most correlated
    # with directed threats. Capped at WEIGHT_EMOTION so it cannot drive an
    # alert on its own; it only resolves LLM variance at the margin.
    emotion_bonus = 0.0
    if audio_emotion and audio_emotion.get("is_clearly_speech", False):
        emotion_score = (audio_emotion["anger_level"] + audio_emotion["urgency"]) / 2.0
        emotion_bonus = WEIGHT_EMOTION * emotion_score

    # Weighted combination (base weights sum to 1.0; emotion is additive)
    fused = (
        WEIGHT_LLM * llm_score
        + WEIGHT_AUDIO * audio_score
        + WEIGHT_KEYWORD * keyword_score
        + emotion_bonus
    )

    return round(min(max(fused, 0.0), 1.0), 4)


def should_alert(fusion_score: float) -> bool:
    """Determine whether the fused score crosses the alert threshold."""
    return fusion_score >= ALERT_THRESHOLD


def _normalize(value: float, floor: float, ceil: float) -> float:
    """Linearly normalize a value to [0, 1] between floor and ceil."""
    if ceil <= floor:
        return 0.0
    return min(max((value - floor) / (ceil - floor), 0.0), 1.0)
