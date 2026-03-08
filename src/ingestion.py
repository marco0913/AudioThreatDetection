"""Audio ingestion -- streams WAV files as chunked PCM buffers.

Simulates receiving audio from a live wearable device by reading
the WAV header and yielding time-aligned PCM frames.
"""

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class AudioStream:
    """Metadata about the audio source."""

    sample_rate: int
    sample_width: int  # bytes per sample
    channels: int
    total_frames: int
    duration_seconds: float


def read_wav(wav_path: str) -> tuple[AudioStream, bytes]:
    """Read a WAV file and return metadata + raw PCM bytes."""
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    with wave.open(str(path), "rb") as wf:
        meta = AudioStream(
            sample_rate=wf.getframerate(),
            sample_width=wf.getsampwidth(),
            channels=wf.getnchannels(),
            total_frames=wf.getnframes(),
            duration_seconds=wf.getnframes() / wf.getframerate(),
        )
        pcm = wf.readframes(wf.getnframes())

    return meta, pcm


def stream_chunks(
    pcm: bytes, meta: AudioStream, chunk_ms: int = 500
) -> Iterator[bytes]:
    """Yield PCM chunks of `chunk_ms` duration, simulating a live stream.

    The last chunk may be shorter than chunk_ms.
    """
    bytes_per_frame = meta.sample_width * meta.channels
    frames_per_chunk = int(meta.sample_rate * chunk_ms / 1000)
    chunk_size = frames_per_chunk * bytes_per_frame

    offset = 0
    while offset < len(pcm):
        yield pcm[offset : offset + chunk_size]
        offset += chunk_size
