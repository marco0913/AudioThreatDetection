"""Tests for src/fusion.py -- signal fusion logic.

All tests are deterministic (no API calls, no I/O).
"""

import pytest

from src.fusion import ALERT_THRESHOLD, fuse_signals, should_alert
from src.models import DetectorOutput, TurnAudioFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(
    threat_score: float = 0.0,
    threat_category: str = "none",
    is_directed: bool = False,
    keywords_matched: list | None = None,
) -> DetectorOutput:
    return DetectorOutput(
        threat_score=threat_score,
        threat_category=threat_category,
        is_directed=is_directed,
        keywords_matched=keywords_matched or [],
        reasoning="test",
        confidence=0.9,
    )


def make_audio(rms_db: float = -25.0, abrupt: float = 0.0) -> TurnAudioFeatures:
    return TurnAudioFeatures(
        rms_db=rms_db,
        peak_db=rms_db + 6,
        zero_crossing_rate=0.05,
        abrupt_change_score=abrupt,
        duration_seconds=2.0,
    )


# ---------------------------------------------------------------------------
# is_directed gate
# ---------------------------------------------------------------------------

class TestIsDirectedGate:
    def test_non_directed_high_score_stays_below_threshold(self):
        """Ambient anger (TV, background) must never cross the alert threshold."""
        detector = make_detector(
            threat_score=0.9, threat_category="ambient_anger", is_directed=False
        )
        audio = make_audio(rms_db=-16.0, abrupt=0.7)
        score = fuse_signals(detector, audio)
        assert score < ALERT_THRESHOLD, (
            f"Non-directed score {score} should be below threshold {ALERT_THRESHOLD}"
        )

    def test_non_directed_score_is_capped_at_04(self):
        """Non-directed fusion score must be capped at 0.4."""
        detector = make_detector(threat_score=1.0, is_directed=False)
        audio = make_audio()
        score = fuse_signals(detector, audio)
        assert score <= 0.4

    def test_non_directed_zero_threat_score_is_zero(self):
        detector = make_detector(threat_score=0.0, is_directed=False)
        score = fuse_signals(detector, make_audio())
        assert score == 0.0

    def test_false_positive_tv_profile_does_not_alert(self):
        """Profile matching false_positive_tv.wav: ambient_anger, not directed."""
        detector = make_detector(
            threat_score=0.1,
            threat_category="ambient_anger",
            is_directed=False,
        )
        audio = make_audio(rms_db=-21.0, abrupt=0.34)
        score = fuse_signals(detector, audio)
        assert not should_alert(score)


# ---------------------------------------------------------------------------
# Directed threats
# ---------------------------------------------------------------------------

class TestDirectedThreats:
    def test_directed_high_threat_with_keywords_crosses_threshold(self):
        """Profile matching keyword_only.wav: explicit threat, keywords matched."""
        detector = make_detector(
            threat_score=0.85,
            threat_category="physical_threat",
            is_directed=True,
            keywords_matched=["hurt you", "i will hurt"],
        )
        audio = make_audio(rms_db=-23.0, abrupt=0.20)
        score = fuse_signals(detector, audio)
        assert should_alert(score), f"Expected alert, got score={score}"
        assert score > 0.7, f"Expected HIGH severity, got {score}"

    def test_directed_threat_no_keywords_can_still_alert(self):
        """Profile matching heated_argument.wav: no keywords, LLM-detected."""
        detector = make_detector(
            threat_score=0.80,
            threat_category="harassment",
            is_directed=True,
            keywords_matched=[],
        )
        # heated_argument audio profile
        audio = make_audio(rms_db=-16.2, abrupt=0.658)
        score = fuse_signals(detector, audio)
        assert should_alert(score), f"Expected alert for harassment, got score={score}"

    def test_directed_low_threat_does_not_alert(self):
        """Low LLM score + no keywords + quiet audio should stay below threshold."""
        detector = make_detector(
            threat_score=0.2, threat_category="none", is_directed=True
        )
        audio = make_audio(rms_db=-26.0, abrupt=0.05)
        score = fuse_signals(detector, audio)
        assert not should_alert(score), f"Expected no alert, got score={score}"

    def test_keyword_boost_pushes_score_higher(self):
        """Keyword match should produce a higher score than no keywords at same LLM score."""
        detector_base = make_detector(
            threat_score=0.5, is_directed=True, keywords_matched=[]
        )
        detector_keywords = make_detector(
            threat_score=0.5, is_directed=True, keywords_matched=["hurt you"]
        )
        audio = make_audio(rms_db=-22.0, abrupt=0.1)
        score_base = fuse_signals(detector_base, audio)
        score_keywords = fuse_signals(detector_keywords, audio)
        assert score_keywords > score_base

    def test_emotion_bonus_stabilises_borderline_heated_argument(self):
        """A threat_score of 0.75 (low LLM run) should alert when emotion confirms anger."""
        # This models the case where LLM returns 0.75 (borderline — seen in live runs).
        # Without emotion: 0.60*0.75 + 0.15*audio ≈ 0.55 (right at threshold, unreliable).
        # With anger=0.80, urgency=0.90: emotion_bonus = 0.05 * 0.85 ≈ 0.042 → clears threshold.
        detector = make_detector(
            threat_score=0.75, threat_category="harassment", is_directed=True
        )
        audio = make_audio(rms_db=-16.2, abrupt=0.658)
        emotion = {"anger_level": 0.80, "urgency": 0.90, "is_clearly_speech": True}
        score = fuse_signals(detector, audio, audio_emotion=emotion)
        assert should_alert(score), f"Expected alert with emotion boost, got score={score}"

    def test_emotion_absent_leaves_base_formula_unchanged(self):
        """Passing audio_emotion=None must produce the same score as not passing it."""
        detector = make_detector(threat_score=0.85, is_directed=True)
        audio = make_audio(rms_db=-16.0, abrupt=0.5)
        assert fuse_signals(detector, audio) == fuse_signals(detector, audio, audio_emotion=None)

    def test_emotion_bonus_cannot_alert_by_itself(self):
        """Even maximum emotion should not push a zero-threat score over the threshold."""
        detector = make_detector(threat_score=0.0, is_directed=True)
        audio = make_audio(rms_db=-30.0, abrupt=0.0)
        emotion = {"anger_level": 1.0, "urgency": 1.0, "is_clearly_speech": True}
        score = fuse_signals(detector, audio, audio_emotion=emotion)
        assert not should_alert(score), f"Emotion alone should not alert, got score={score}"

    def test_emotion_not_applied_when_not_speech(self):
        """Emotion bonus is skipped when is_clearly_speech=False (noise/music)."""
        detector = make_detector(threat_score=0.75, is_directed=True)
        audio = make_audio(rms_db=-16.2, abrupt=0.658)
        emotion_noise = {"anger_level": 1.0, "urgency": 1.0, "is_clearly_speech": False}
        score_with_noise_emotion = fuse_signals(detector, audio, audio_emotion=emotion_noise)
        score_no_emotion = fuse_signals(detector, audio, audio_emotion=None)
        assert score_with_noise_emotion == score_no_emotion


# ---------------------------------------------------------------------------
# should_alert threshold
# ---------------------------------------------------------------------------

class TestShouldAlert:
    def test_exactly_at_threshold_alerts(self):
        assert should_alert(ALERT_THRESHOLD) is True

    def test_just_below_threshold_does_not_alert(self):
        assert should_alert(ALERT_THRESHOLD - 0.001) is False

    def test_zero_does_not_alert(self):
        assert should_alert(0.0) is False

    def test_one_alerts(self):
        assert should_alert(1.0) is True


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

class TestScoreBounds:
    def test_score_never_exceeds_one(self):
        detector = make_detector(
            threat_score=1.0, is_directed=True, keywords_matched=["x"]
        )
        audio = make_audio(rms_db=-12.0, abrupt=1.0)
        score = fuse_signals(detector, audio)
        assert 0.0 <= score <= 1.0

    def test_score_never_below_zero(self):
        detector = make_detector(threat_score=0.0, is_directed=True)
        audio = make_audio(rms_db=-50.0, abrupt=0.0)
        score = fuse_signals(detector, audio)
        assert score >= 0.0
