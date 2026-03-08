"""Incident manager -- cooldown and grouping logic to prevent alert spam.

Per-badge state machine:
  IDLE -> ACTIVE_INCIDENT -> COOLDOWN -> IDLE

When a turn triggers an alert:
- If no active incident for this badge: create one, fire THREAT_DETECTED
- If active incident exists and within cooldown window: add turn to incident,
  fire INCIDENT_UPDATE only if severity escalates
- After cooldown_seconds with no new triggers: mark resolved

This prevents the system from spamming alerts for the same ongoing situation.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class IncidentState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass
class Incident:
    incident_id: str = field(default_factory=lambda: f"inc_{uuid.uuid4().hex[:8]}")
    badge_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_trigger_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_ids: list[str] = field(default_factory=list)
    peak_score: float = 0.0
    state: IncidentState = IncidentState.ACTIVE
    alert_count: int = 0


@dataclass
class AlertDecision:
    """What the incident manager decides about an incoming alert."""
    incident: Incident
    should_publish: bool  # True if this alert should be sent to subscriber
    alert_type: str       # THREAT_DETECTED, INCIDENT_UPDATE, or suppressed
    reason: str           # Why this decision was made


class IncidentManager:
    """Manages per-badge incident state and cooldown windows."""

    def __init__(self, cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        self._incidents: dict[str, Incident] = {}

    def process_alert(
        self, badge_id: str, turn_id: str, fusion_score: float
    ) -> AlertDecision:
        """Decide how to handle an incoming alert for a badge.

        Returns an AlertDecision indicating whether to publish and what type.
        """
        now = datetime.now(timezone.utc)
        existing = self._incidents.get(badge_id)

        # Check if existing incident has expired (cooldown elapsed)
        if existing and existing.state == IncidentState.ACTIVE:
            elapsed = (now - existing.last_trigger_at).total_seconds()
            if elapsed > self.cooldown_seconds:
                existing.state = IncidentState.COOLDOWN
                existing = None  # treat as new

        if existing and existing.state == IncidentState.ACTIVE:
            # Add turn to existing incident
            existing.turn_ids.append(turn_id)
            existing.last_trigger_at = now

            if fusion_score > existing.peak_score:
                # Severity escalation -- worth publishing an update
                old_peak = existing.peak_score
                existing.peak_score = fusion_score
                existing.alert_count += 1
                return AlertDecision(
                    incident=existing,
                    should_publish=True,
                    alert_type="INCIDENT_UPDATE",
                    reason=f"Severity escalated from {old_peak:.2f} to {fusion_score:.2f}",
                )
            else:
                # Same or lower severity -- suppress to avoid spam
                return AlertDecision(
                    incident=existing,
                    should_publish=False,
                    alert_type="suppressed",
                    reason=f"Score {fusion_score:.2f} <= peak {existing.peak_score:.2f}, suppressed within cooldown",
                )
        else:
            # New incident
            incident = Incident(
                badge_id=badge_id,
                turn_ids=[turn_id],
                peak_score=fusion_score,
                alert_count=1,
            )
            self._incidents[badge_id] = incident
            return AlertDecision(
                incident=incident,
                should_publish=True,
                alert_type="THREAT_DETECTED",
                reason="New incident opened",
            )

    def get_active_incident(self, badge_id: str) -> Incident | None:
        """Return the active incident for a badge, if any."""
        inc = self._incidents.get(badge_id)
        if inc and inc.state == IncidentState.ACTIVE:
            return inc
        return None
