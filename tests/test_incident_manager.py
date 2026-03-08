"""Tests for src/incident_manager.py -- cooldown state machine.

All tests are deterministic (no API calls, no I/O).
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.incident_manager import IncidentManager, IncidentState


class TestFirstAlert:
    def test_first_alert_publishes(self):
        mgr = IncidentManager()
        decision = mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        assert decision.should_publish is True

    def test_first_alert_type_is_threat_detected(self):
        mgr = IncidentManager()
        decision = mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        assert decision.alert_type == "THREAT_DETECTED"

    def test_first_alert_creates_incident(self):
        mgr = IncidentManager()
        decision = mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        assert decision.incident is not None
        assert decision.incident.badge_id == "badge_1"
        assert "turn_001" in decision.incident.turn_ids

    def test_first_alert_sets_peak_score(self):
        mgr = IncidentManager()
        decision = mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        assert decision.incident.peak_score == 0.7


class TestDuplicateSuppression:
    def test_same_score_is_suppressed(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        decision = mgr.process_alert("badge_1", "turn_002", fusion_score=0.7)
        assert decision.should_publish is False

    def test_lower_score_is_suppressed(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.8)
        decision = mgr.process_alert("badge_1", "turn_002", fusion_score=0.6)
        assert decision.should_publish is False
        assert decision.alert_type == "suppressed"

    def test_suppressed_turn_still_added_to_incident(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        decision = mgr.process_alert("badge_1", "turn_002", fusion_score=0.7)
        assert "turn_002" in decision.incident.turn_ids


class TestSeverityEscalation:
    def test_higher_score_publishes_update(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.6)
        decision = mgr.process_alert("badge_1", "turn_002", fusion_score=0.8)
        assert decision.should_publish is True

    def test_escalation_alert_type_is_incident_update(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.6)
        decision = mgr.process_alert("badge_1", "turn_002", fusion_score=0.8)
        assert decision.alert_type == "INCIDENT_UPDATE"

    def test_peak_score_updated_on_escalation(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.6)
        decision = mgr.process_alert("badge_1", "turn_002", fusion_score=0.85)
        assert decision.incident.peak_score == 0.85

    def test_escalation_followed_by_same_score_suppressed(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.6)
        mgr.process_alert("badge_1", "turn_002", fusion_score=0.85)
        decision = mgr.process_alert("badge_1", "turn_003", fusion_score=0.85)
        assert decision.should_publish is False


class TestCooldown:
    def test_new_incident_after_cooldown_expires(self):
        mgr = IncidentManager(cooldown_seconds=60.0)
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)

        # Manually backdate the last_trigger_at past cooldown
        incident = mgr._incidents["badge_1"]
        incident.last_trigger_at = datetime.now(timezone.utc) - timedelta(seconds=61)

        # Next alert should open a new incident
        decision = mgr.process_alert("badge_1", "turn_002", fusion_score=0.7)
        assert decision.alert_type == "THREAT_DETECTED"
        assert decision.should_publish is True

    def test_expired_incident_state_set_to_cooldown(self):
        mgr = IncidentManager(cooldown_seconds=60.0)
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        old_incident = mgr._incidents["badge_1"]
        old_incident.last_trigger_at = datetime.now(timezone.utc) - timedelta(seconds=61)

        mgr.process_alert("badge_1", "turn_002", fusion_score=0.7)
        # Old incident should be in COOLDOWN state now
        assert old_incident.state == IncidentState.COOLDOWN


class TestMultipleBadges:
    def test_different_badges_have_independent_incidents(self):
        mgr = IncidentManager()
        d1 = mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        d2 = mgr.process_alert("badge_2", "turn_001", fusion_score=0.7)
        # Both should be THREAT_DETECTED (separate incidents)
        assert d1.alert_type == "THREAT_DETECTED"
        assert d2.alert_type == "THREAT_DETECTED"
        assert d1.incident.incident_id != d2.incident.incident_id

    def test_get_active_incident_returns_none_when_idle(self):
        mgr = IncidentManager()
        assert mgr.get_active_incident("badge_99") is None

    def test_get_active_incident_returns_incident_after_alert(self):
        mgr = IncidentManager()
        mgr.process_alert("badge_1", "turn_001", fusion_score=0.7)
        inc = mgr.get_active_incident("badge_1")
        assert inc is not None
        assert inc.badge_id == "badge_1"
