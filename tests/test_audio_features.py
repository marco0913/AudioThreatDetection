"""Tests for src/audio_features.py -- deterministic feature extraction.

All tests use synthetic PCM data so no WAV files or API calls are needed.
"""

import struct

import numpy as np
import pytest

from src.audio_features import extract_features, _abrupt_change_score
from src.models import TurnAudioFeatures


SAMPLE_RATE = 16000


def _make_pcm(samples: np.ndarray) -> bytes:
    """Convert float32 samples in [-1.0, 1.0] to 16-bit mono PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    ints = (clipped * 32767).astype(np.int16)
    return ints.tobytes()


def _sine_pcm(freq_hz: float, duration_s: float, amplitude: float = 0.5) -> bytes:
    """Generate a mono 16-bit PCM sine wave."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return _make_pcm(samples)


def _silence_pcm(duration_s: float) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    return _make_pcm(np.zeros(n))


# ---------------------------------------------------------------------------
# Guard: too-short input returns default TurnAudioFeatures
# ---------------------------------------------------------------------------

class TestShortInput:
    def test_empty_bytes_returns_default(self):
        result = extract_features(b"")
        assert result == TurnAudioFeatures()

    def test_two_bytes_returns_default(self):
        """Less than 4 bytes (< 2 int16 samples) -> default."""
        result = extract_features(b"\x00\x01")
        assert result == TurnAudioFeatures()

    def test_three_bytes_returns_default(self):
        result = extract_features(b"\x00\x01\x02")
        assert result == TurnAudioFeatures()


# ---------------------------------------------------------------------------
# RMS loudness: louder input -> higher rms_db
# ---------------------------------------------------------------------------

class TestRmsLoudness:
    def test_louder_sine_higher_rms(self):
        """Amplitude 0.8 sine must be louder than amplitude 0.2 sine."""
        loud = extract_features(_sine_pcm(440, 1.0, amplitude=0.8))
        quiet = extract_features(_sine_pcm(440, 1.0, amplitude=0.2))
        assert loud.rms_db > quiet.rms_db

    def test_silence_very_low_rms(self):
        """Pure silence should be far below -40 dB."""
        result = extract_features(_silence_pcm(1.0))
        assert result.rms_db < -40.0

    def test_full_scale_near_zero_db(self):
        """A ±1.0 square wave should be near 0 dBFS."""
        n = SAMPLE_RATE  # 1 second
        samples = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        result = extract_features(_make_pcm(samples))
        # RMS of a square wave at amplitude 1.0 is exactly 1.0 -> 0 dBFS
        assert result.rms_db >= -1.0

    def test_rms_db_is_negative_for_sub_unity(self):
        """Any amplitude below 1.0 must yield negative dBFS."""
        result = extract_features(_sine_pcm(440, 1.0, amplitude=0.5))
        assert result.rms_db < 0.0


# ---------------------------------------------------------------------------
# Peak amplitude
# ---------------------------------------------------------------------------

class TestPeakAmplitude:
    def test_peak_db_gte_rms_db(self):
        """Peak is always >= RMS by definition."""
        result = extract_features(_sine_pcm(440, 1.0, amplitude=0.5))
        assert result.peak_db >= result.rms_db

    def test_louder_signal_higher_peak(self):
        loud = extract_features(_sine_pcm(440, 1.0, amplitude=0.9))
        quiet = extract_features(_sine_pcm(440, 1.0, amplitude=0.3))
        assert loud.peak_db > quiet.peak_db


# ---------------------------------------------------------------------------
# Zero-crossing rate
# ---------------------------------------------------------------------------

class TestZeroCrossingRate:
    def test_high_frequency_has_higher_zcr(self):
        """1000 Hz sine crosses zero more often than 100 Hz sine."""
        high = extract_features(_sine_pcm(1000, 1.0, amplitude=0.5))
        low = extract_features(_sine_pcm(100, 1.0, amplitude=0.5))
        assert high.zero_crossing_rate > low.zero_crossing_rate

    def test_silence_has_zero_zcr(self):
        result = extract_features(_silence_pcm(1.0))
        assert result.zero_crossing_rate == 0.0

    def test_zcr_between_zero_and_one(self):
        result = extract_features(_sine_pcm(440, 1.0, amplitude=0.5))
        assert 0.0 <= result.zero_crossing_rate <= 1.0


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

class TestDuration:
    def test_duration_matches_input_length(self):
        pcm = _sine_pcm(440, 2.0, amplitude=0.5)
        result = extract_features(pcm, sample_rate=SAMPLE_RATE)
        assert abs(result.duration_seconds - 2.0) < 0.01

    def test_half_second_duration(self):
        pcm = _sine_pcm(440, 0.5, amplitude=0.5)
        result = extract_features(pcm, sample_rate=SAMPLE_RATE)
        assert abs(result.duration_seconds - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Abrupt change score
# ---------------------------------------------------------------------------

class TestAbruptChangeScore:
    def test_sudden_amplitude_jump_scores_high(self):
        """Silence followed by loud signal -> high abrupt change score."""
        n = SAMPLE_RATE  # 1 second total
        samples = np.zeros(n)
        # Second half: loud sine
        t2 = np.linspace(0, 0.5, n // 2, endpoint=False)
        samples[n // 2:] = 0.9 * np.sin(2 * np.pi * 440 * t2)
        result = extract_features(_make_pcm(samples))
        assert result.abrupt_change_score > 0.5

    def test_constant_signal_scores_low(self):
        """Steady sine wave has no abrupt change."""
        result = extract_features(_sine_pcm(440, 2.0, amplitude=0.5))
        assert result.abrupt_change_score < 0.1

    def test_score_bounded_zero_to_one(self):
        n = SAMPLE_RATE * 2
        samples = np.zeros(n)
        samples[n // 2:] = 1.0  # maximum step change
        result = extract_features(_make_pcm(samples))
        assert 0.0 <= result.abrupt_change_score <= 1.0

    def test_too_short_for_windows_returns_zero(self):
        """Audio shorter than two 200ms windows -> 0.0."""
        pcm = _sine_pcm(440, 0.3, amplitude=0.5)  # only 300ms
        normalized = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        score = _abrupt_change_score(normalized, SAMPLE_RATE)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_turn_audio_features(self):
        result = extract_features(_sine_pcm(440, 1.0))
        assert isinstance(result, TurnAudioFeatures)

    def test_fields_are_rounded(self):
        """rms_db and peak_db rounded to 2dp; zcr and abrupt to 4dp."""
        result = extract_features(_sine_pcm(440, 1.0, amplitude=0.5))
        assert result.rms_db == round(result.rms_db, 2)
        assert result.zero_crossing_rate == round(result.zero_crossing_rate, 4)
        assert result.abrupt_change_score == round(result.abrupt_change_score, 4)
