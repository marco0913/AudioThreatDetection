"""End-to-end pipeline tests -- real audio, mocked API stages.

Strategy
--------
- Read real WAV files from audio/ to exercise ingestion + segmentation +
  audio_features deterministically.
- Mock `transcribe` and `detect_threat` with realistic pre-defined outputs
  so no OpenAI API calls are made.
- Mock `publish` to prevent HTTP calls; capture what would have been sent.
- Run the full pipeline via cli.run_pipeline and assert the correct
  alert / no-alert decision for each corpus file.

Corpus expected outcomes (from tasks/decisions.md and README):
  casual_chat.wav       -> no alert (quiet, non-threatening)
  heated_argument.wav   -> alert    (is_directed=True, harassment)
  keyword_only.wav      -> alert    (is_directed=True, physical_threat, keywords)
  false_positive_tv.wav -> no alert (is_directed=False, ambient_anger)
  muffled_noise.wav     -> no alert (low energy, indistinct, no threat)
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.models import DetectorOutput, TranscriptResult

# Corpus directory (relative to project root where pytest is run)
AUDIO_DIR = Path(__file__).parent.parent / "audio"


# ---------------------------------------------------------------------------
# Pre-defined mock outputs for each corpus file
# ---------------------------------------------------------------------------

TRANSCRIPT_CASUAL = TranscriptResult(
    raw_text="yeah so i was just thinking about the weekend plans maybe we could go hiking",
    cleaned_text="yeah so i was just thinking about the weekend plans maybe we could go hiking",
    language="en",
    word_count=15,
    is_empty=False,
)

DETECTOR_CASUAL = DetectorOutput(
    threat_score=0.05,
    threat_category="none",
    is_directed=False,
    reasoning="Casual conversation with no threat indicators.",
    keywords_matched=[],
    llm_flags=[],
    confidence=0.95,
)

TRANSCRIPT_HEATED = TranscriptResult(
    raw_text="you promised me you would stop this give me what you owe me right now",
    cleaned_text="you promised me you would stop this give me what you owe me right now",
    language="en",
    word_count=16,
    is_empty=False,
)

DETECTOR_HEATED = DetectorOutput(
    threat_score=0.80,
    threat_category="harassment",
    is_directed=True,
    reasoning="Second-person confrontational language directed at another person. Aggressive demands.",
    keywords_matched=[],
    llm_flags=["aggressive_demand", "confrontational"],
    confidence=0.85,
)

TRANSCRIPT_KEYWORD = TranscriptResult(
    raw_text="i will hurt you if you don't do what i say i'm serious",
    cleaned_text="i will hurt you if you don't do what i say i'm serious",
    language="en",
    word_count=14,
    is_empty=False,
)

DETECTOR_KEYWORD = DetectorOutput(
    threat_score=0.90,
    threat_category="physical_threat",
    is_directed=True,
    reasoning="Explicit physical threat with clear intent and target.",
    keywords_matched=["hurt you", "i will hurt"],
    llm_flags=["explicit_threat", "physical_violence"],
    confidence=0.95,
)

TRANSCRIPT_TV = TranscriptResult(
    raw_text="and the crowd goes wild as the player makes the move incredible scenes here tonight",
    cleaned_text="and the crowd goes wild as the player makes the move incredible scenes here tonight",
    language="en",
    word_count=16,
    is_empty=False,
)

DETECTOR_TV = DetectorOutput(
    threat_score=0.10,
    threat_category="ambient_anger",
    is_directed=False,
    reasoning="Sports commentary or TV audio. Excited crowd noise, not directed at any individual.",
    keywords_matched=[],
    llm_flags=[],
    confidence=0.90,
)

TRANSCRIPT_MUFFLED = TranscriptResult(
    raw_text="",
    cleaned_text="",
    language=None,
    word_count=0,
    is_empty=True,
)

DETECTOR_MUFFLED = DetectorOutput(
    threat_score=0.05,
    threat_category="none",
    is_directed=False,
    reasoning="No intelligible speech detected. Likely background noise or silence.",
    keywords_matched=[],
    llm_flags=[],
    confidence=0.50,
)


# ---------------------------------------------------------------------------
# Parameterized corpus scenarios
# ---------------------------------------------------------------------------

CORPUS_SCENARIOS = [
    pytest.param(
        "casual_chat.wav",
        TRANSCRIPT_CASUAL,
        DETECTOR_CASUAL,
        False,  # expected: no alert
        id="casual_chat",
    ),
    pytest.param(
        "heated_argument.wav",
        TRANSCRIPT_HEATED,
        DETECTOR_HEATED,
        True,   # expected: alert
        id="heated_argument",
    ),
    pytest.param(
        "keyword_only.wav",
        TRANSCRIPT_KEYWORD,
        DETECTOR_KEYWORD,
        True,   # expected: alert
        id="keyword_only",
    ),
    pytest.param(
        "false_positive_tv.wav",
        TRANSCRIPT_TV,
        DETECTOR_TV,
        False,  # expected: no alert
        id="false_positive_tv",
    ),
    pytest.param(
        "muffled_noise.wav",
        TRANSCRIPT_MUFFLED,
        DETECTOR_MUFFLED,
        False,  # expected: no alert
        id="muffled_noise",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("wav_file,mock_transcript,mock_detector,expect_alert", CORPUS_SCENARIOS)
async def test_corpus_alert_decision(wav_file, mock_transcript, mock_detector, expect_alert):
    """Validate alert/no-alert outcome for each corpus file with mocked API calls."""
    wav_path = AUDIO_DIR / wav_file
    if not wav_path.exists():
        pytest.skip(f"Corpus file not found: {wav_path}")

    published_events = []

    async def fake_publish(event, url=None):
        published_events.append(event)
        return True

    with (
        patch("cli.transcribe", new=AsyncMock(return_value=mock_transcript)),
        patch("cli.detect_threat", new=AsyncMock(return_value=mock_detector)),
        patch("cli.publish", new=fake_publish),
        patch("cli.incident_manager") as mock_mgr,
    ):
        # Set up incident_manager mock so should_publish=True whenever called
        from src.incident_manager import AlertDecision, Incident
        mock_incident = Incident(badge_id="badge_001", turn_ids=["t0"], peak_score=0.9)
        mock_mgr.process_alert.return_value = AlertDecision(
            incident=mock_incident,
            should_publish=True,
            alert_type="THREAT_DETECTED",
            reason="test",
        )

        import cli
        results = await cli.run_pipeline(str(wav_path), badge_id="badge_001")

    assert len(results) > 0, f"Pipeline produced no turn results for {wav_file}"

    any_alert = any(r.alert_fired for r in results)
    assert any_alert == expect_alert, (
        f"{wav_file}: expected alert={expect_alert} but got alert={any_alert}. "
        f"Fusion scores: {[r.fusion_score for r in results]}"
    )


# ---------------------------------------------------------------------------
# Publish is called only when an alert fires
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_called_on_alert():
    """When an alert fires, publish() must be called with the AlertEvent."""
    wav_path = AUDIO_DIR / "keyword_only.wav"
    if not wav_path.exists():
        pytest.skip("keyword_only.wav not found")

    published_events = []

    async def fake_publish(event, url=None):
        published_events.append(event)
        return True

    with (
        patch("cli.transcribe", new=AsyncMock(return_value=TRANSCRIPT_KEYWORD)),
        patch("cli.detect_threat", new=AsyncMock(return_value=DETECTOR_KEYWORD)),
        patch("cli.publish", new=fake_publish),
        patch("cli.incident_manager") as mock_mgr,
    ):
        from src.incident_manager import AlertDecision, Incident
        mock_incident = Incident(badge_id="badge_001", turn_ids=["t0"], peak_score=0.9)
        mock_mgr.process_alert.return_value = AlertDecision(
            incident=mock_incident,
            should_publish=True,
            alert_type="THREAT_DETECTED",
            reason="test",
        )

        import cli
        results = await cli.run_pipeline(str(wav_path), badge_id="badge_001")

    assert any(r.alert_fired for r in results), "Expected at least one alert for keyword_only.wav"
    assert len(published_events) > 0, "publish() should have been called"
    event = published_events[0]
    assert event.alert_type == "THREAT_DETECTED"
    assert event.badge_id == "badge_001"
    assert event.fusion_score > 0.55


@pytest.mark.asyncio
async def test_no_publish_when_no_alert():
    """When no alert fires, publish() must NOT be called."""
    wav_path = AUDIO_DIR / "casual_chat.wav"
    if not wav_path.exists():
        pytest.skip("casual_chat.wav not found")

    published_events = []

    async def fake_publish(event, url=None):
        published_events.append(event)
        return True

    with (
        patch("cli.transcribe", new=AsyncMock(return_value=TRANSCRIPT_CASUAL)),
        patch("cli.detect_threat", new=AsyncMock(return_value=DETECTOR_CASUAL)),
        patch("cli.publish", new=fake_publish),
    ):
        import cli
        results = await cli.run_pipeline(str(wav_path), badge_id="badge_001")

    assert not any(r.alert_fired for r in results)
    assert len(published_events) == 0, "publish() should not be called when no alert fires"


# ---------------------------------------------------------------------------
# Fusion score contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fusion_scores_are_in_range():
    """All fusion scores must be in [0.0, 1.0]."""
    wav_path = AUDIO_DIR / "heated_argument.wav"
    if not wav_path.exists():
        pytest.skip("heated_argument.wav not found")

    with (
        patch("cli.transcribe", new=AsyncMock(return_value=TRANSCRIPT_HEATED)),
        patch("cli.detect_threat", new=AsyncMock(return_value=DETECTOR_HEATED)),
        patch("cli.publish", new=AsyncMock(return_value=True)),
        patch("cli.incident_manager") as mock_mgr,
    ):
        from src.incident_manager import AlertDecision, Incident
        mock_incident = Incident(badge_id="badge_001", turn_ids=["t0"], peak_score=0.8)
        mock_mgr.process_alert.return_value = AlertDecision(
            incident=mock_incident,
            should_publish=True,
            alert_type="THREAT_DETECTED",
            reason="test",
        )

        import cli
        results = await cli.run_pipeline(str(wav_path))

    for r in results:
        assert 0.0 <= r.fusion_score <= 1.0, f"Out-of-range fusion_score: {r.fusion_score}"


@pytest.mark.asyncio
async def test_turn_results_have_audio_features():
    """Each TurnResult must have non-default audio features (features were extracted)."""
    wav_path = AUDIO_DIR / "heated_argument.wav"
    if not wav_path.exists():
        pytest.skip("heated_argument.wav not found")

    with (
        patch("cli.transcribe", new=AsyncMock(return_value=TRANSCRIPT_HEATED)),
        patch("cli.detect_threat", new=AsyncMock(return_value=DETECTOR_HEATED)),
        patch("cli.publish", new=AsyncMock(return_value=True)),
        patch("cli.incident_manager") as mock_mgr,
    ):
        from src.incident_manager import AlertDecision, Incident
        mock_incident = Incident(badge_id="badge_001", turn_ids=["t0"], peak_score=0.8)
        mock_mgr.process_alert.return_value = AlertDecision(
            incident=mock_incident,
            should_publish=True,
            alert_type="THREAT_DETECTED",
            reason="test",
        )

        import cli
        results = await cli.run_pipeline(str(wav_path))

    for r in results:
        # Real audio must have some energy -- duration and rms should be non-default
        assert r.audio_features.duration_seconds > 0.0
        # rms_db of real speech is above the -200 dB floor
        assert r.audio_features.rms_db > -100.0
