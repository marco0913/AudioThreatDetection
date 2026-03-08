"""Core data models — the contracts that every pipeline stage depends on."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class TurnAudioFeatures(BaseModel):
    """Audio-level signals extracted directly from the raw waveform."""

    rms_db: float = 0.0
    peak_db: float = 0.0
    zero_crossing_rate: float = 0.0
    abrupt_change_score: float = 0.0
    duration_seconds: float = 0.0


class TranscriptResult(BaseModel):
    """Output of the Whisper transcription + cleanup step."""

    raw_text: str = ""
    cleaned_text: str = ""
    language: str | None = None
    word_count: int = 0
    is_empty: bool = True


class DetectorOutput(BaseModel):
    """Structured output from the threat detector (heuristics + LLM)."""

    threat_score: float = Field(default=0.0, ge=0.0, le=1.0)
    threat_category: Literal[
        "none",
        "verbal_threat",
        "physical_threat",
        "distress_signal",
        "harassment",
        "ambient_anger",
        "unknown",
    ] = "none"
    is_directed: bool = False
    reasoning: str = "No threat detected."
    keywords_matched: list[str] = Field(default_factory=list)
    llm_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TurnResult(BaseModel):
    """Per-turn intermediate result carrying all signals through the pipeline."""

    turn_id: str
    badge_id: str = "badge_001"
    source_file: str = ""
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    audio_features: TurnAudioFeatures = Field(default_factory=TurnAudioFeatures)
    transcript: TranscriptResult = Field(default_factory=TranscriptResult)
    detector: DetectorOutput = Field(default_factory=DetectorOutput)
    fusion_score: float = 0.0
    alert_fired: bool = False
    latency_ms: dict[str, float] = Field(default_factory=dict)


class AlertEvent(BaseModel):
    """Published to the subscriber service when the pipeline decides to alert."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: Literal[
        "THREAT_DETECTED", "INCIDENT_UPDATE", "INCIDENT_RESOLVED"
    ] = "THREAT_DETECTED"
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    badge_id: str = "badge_001"
    incident_id: str = Field(default_factory=lambda: f"inc_{uuid.uuid4().hex[:8]}")
    triggered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turn_id: str = ""
    source_file: str = ""
    turns_in_incident: list[str] = Field(default_factory=list)
    transcript_excerpt: str = ""
    threat_category: str = "none"
    fusion_score: float = 0.0
    audio_signals: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    total_pipeline_latency_ms: float = 0.0
