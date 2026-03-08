"""Audio feature extraction -- deterministic signals from raw waveform.

Extracts four features from 16-bit mono PCM audio using pure numpy.
All features operate on the signal normalised to [-1.0, 1.0].

Features and formulas
---------------------
RMS loudness (dB)
    Measures the average power of the waveform:

        RMS  = sqrt( mean( x_i^2 ) )
        dB   = 20 * log10( max(RMS, epsilon) )

    The factor of 20 (not 10) is used because dB on *amplitude* ratios
    is defined as 20*log10(A); power dB uses 10*log10(P), and since
    P ∝ A^2, the two are equivalent. epsilon = 1e-10 avoids log(0).
    Reference: Oppenheim & Schafer, "Discrete-Time Signal Processing",
    3rd ed., 2010, ch. 2.

Peak amplitude (dB)
    Maximum absolute sample value expressed in dB:

        peak    = max( |x_i| )
        peak_dB = 20 * log10( max(peak, epsilon) )

    0 dBFS (full scale) corresponds to peak = 1.0 after normalisation.
    A typical speech peak is around -3 to -12 dBFS.

Zero-crossing rate (ZCR)
    Fraction of adjacent sample pairs where the sign changes:

        ZCR = mean( |sign(x_{i+1}) - sign(x_i)| > 0 )

    High ZCR indicates noisy or fricative content; low ZCR indicates
    voiced speech or silence. Reference: Rabiner & Schafer,
    "Digital Processing of Speech Signals", 1978, ch. 4.
    Corpus observation: muffled_noise.wav has ZCR ≈ 0.144 (high),
    speech files are typically 0.05-0.09.

Abrupt change score (0.0  1.0)
    Maximum RMS delta between consecutive 200 ms windows, normalised
    so that a raw amplitude delta of 0.3 maps to 1.0:

        rms_k       = sqrt( mean( x_i^2 ) )  for each 200 ms window k
        max_delta   = max( |rms_{k+1} - rms_k| )
        score       = min( max_delta / 0.3, 1.0 )

    The normalisation constant 0.3 was calibrated empirically:
    it approximates the RMS difference between silence and full-volume
    speech (decision D12 in tasks/decisions.md). Corpus: heated_argument
    scores 0.658 (loud shouting with variation); casual_chat scores
    0.10-0.25 (normal conversation).

Fusion role
-----------
The audio component contributes 15 % of the fusion score:

    audio_score = 0.5 * normalised_rms + 0.5 * abrupt_change_score

where normalised_rms = clamp((rms_db - (-50)) / (0 - (-50)), 0, 1),
mapping the practical speech range of -50 to 0 dBFS onto [0, 1].
(See src/fusion.py for the full fusion formula.)
"""

import numpy as np

from src.models import TurnAudioFeatures


def extract_features(audio_bytes: bytes, sample_rate: int = 16000) -> TurnAudioFeatures:
    """Extract audio-level features from raw 16-bit mono PCM bytes."""
    if len(audio_bytes) < 4:
        return TurnAudioFeatures()

    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    duration = len(samples) / sample_rate

    # Normalise to [-1.0, 1.0]: 16-bit signed integers span -32768..+32767.
    normalized = samples / 32768.0

    # RMS loudness in dB -- see module docstring for formula derivation.
    # epsilon = 1e-10 keeps log10 defined for silent frames (~-200 dB).
    rms = np.sqrt(np.mean(normalized**2))
    rms_db = 20 * np.log10(max(rms, 1e-10))

    # Peak amplitude in dB (0 dBFS = full scale = 1.0 amplitude).
    peak = np.max(np.abs(normalized))
    peak_db = 20 * np.log10(max(peak, 1e-10))

    # Zero-crossing rate: fraction of consecutive sample pairs where the
    # sign flips. np.diff(signs) is ±2 at crossings, 0 otherwise.
    signs = np.sign(normalized)
    zcr = np.mean(np.abs(np.diff(signs)) > 0)

    # Abrupt change score: normalised max RMS delta across 200 ms windows.
    abrupt = _abrupt_change_score(normalized, sample_rate)

    return TurnAudioFeatures(
        rms_db=round(float(rms_db), 2),
        peak_db=round(float(peak_db), 2),
        zero_crossing_rate=round(float(zcr), 4),
        abrupt_change_score=round(float(abrupt), 4),
        duration_seconds=round(duration, 3),
    )


def _abrupt_change_score(
    normalized: np.ndarray, sample_rate: int, window_ms: int = 200
) -> float:
    """Compute the maximum RMS delta between consecutive 200 ms windows.

    A high score indicates a sudden loudness spike (e.g. shout, slam).
    Returns a normalised score in [0.0, 1.0].

    Formula:
        rms_k     = sqrt( mean( x_i^2 ) )  for each non-overlapping window k
        max_delta = max( |rms_{k+1} - rms_k| )  over all consecutive pairs
        score     = min( max_delta / 0.3, 1.0 )

    The normalisation constant 0.3 was set empirically: it approximates the
    RMS amplitude difference between sustained silence (~0.0) and loud
    conversational speech (~0.3 peak RMS). See decision D12 in
    tasks/decisions.md and the module docstring for corpus validation data.
    """
    window_size = int(sample_rate * window_ms / 1000)
    if len(normalized) < window_size * 2:
        return 0.0

    # Compute RMS for each non-overlapping window.
    n_windows = len(normalized) // window_size
    rms_values = []
    for i in range(n_windows):
        chunk = normalized[i * window_size : (i + 1) * window_size]
        rms_values.append(np.sqrt(np.mean(chunk**2)))

    if len(rms_values) < 2:
        return 0.0

    rms_arr = np.array(rms_values)
    # np.diff gives rms_{k+1} - rms_k; abs makes direction irrelevant.
    deltas = np.abs(np.diff(rms_arr))
    max_delta = float(np.max(deltas))

    # Clamp to [0, 1]: deltas >= 0.3 are treated as maximally abrupt.
    return min(max_delta / 0.3, 1.0)
