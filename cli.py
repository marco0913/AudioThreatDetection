"""Pipeline entrypoint -- wires all stages and runs against a WAV file."""

import argparse
import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.audio_emotion import analyze_audio_emotion
from src.audio_features import extract_features
from src.detector import detect_threat
from src.fusion import fuse_signals, should_alert
from src.incident_manager import IncidentManager
from src.ingestion import read_wav
from src.models import AlertEvent, TurnResult
from src.publisher import publish
from src.segmenter import segment_turns
from src.transcriber import transcribe


incident_manager = IncidentManager()


async def run_pipeline(wav_path: str, badge_id: str = "badge_001") -> list[TurnResult]:
    """Run the full pipeline on a WAV file, returning all turn results."""
    source_file = Path(wav_path).name
    results: list[TurnResult] = []

    print(f"[PIPELINE] Processing: {source_file}")

    # --- Ingestion ---
    t0 = time.perf_counter()
    meta, pcm = read_wav(wav_path)
    ingestion_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[INGEST] {meta.sample_rate}Hz {meta.sample_width*8}bit "
        f"{meta.channels}ch {meta.duration_seconds:.2f}s"
    )

    # --- Segmentation ---
    t0 = time.perf_counter()
    turns = segment_turns(pcm, sample_rate=meta.sample_rate)
    segmentation_ms = (time.perf_counter() - t0) * 1000
    print(f"[SEGMENT] {len(turns)} turn(s) detected in {segmentation_ms:.0f}ms")

    for turn in turns:
        turn_latency = {
            "ingestion": ingestion_ms,
            "segmentation": segmentation_ms,
        }
        turn_id = f"{badge_id}_{int(turn.start_ms)}"

        # --- Audio Features ---
        t0 = time.perf_counter()
        audio_features = extract_features(turn.audio_bytes, sample_rate=meta.sample_rate)
        turn_latency["audio_features"] = (time.perf_counter() - t0) * 1000

        # --- GPT-4o Audio Emotion (bonus path A, opt-in via ENABLE_AUDIO_EMOTION=true) ---
        t0 = time.perf_counter()
        audio_emotion = await analyze_audio_emotion(turn.audio_bytes, sample_rate=meta.sample_rate)
        if audio_emotion:
            turn_latency["audio_emotion"] = (time.perf_counter() - t0) * 1000
            print(
                f"[AUDIO EMOTION] anger={audio_emotion['anger_level']:.2f} "
                f"urgency={audio_emotion['urgency']:.2f} "
                f"stress={audio_emotion['stress_level']:.2f} "
                f"label={audio_emotion['emotion_label']} "
                f"speech={audio_emotion['is_clearly_speech']} "
                f"-- {audio_emotion['reasoning']}"
            )

        # --- Transcription ---
        t0 = time.perf_counter()
        transcript = await transcribe(turn.audio_bytes, sample_rate=meta.sample_rate)
        turn_latency["transcription"] = (time.perf_counter() - t0) * 1000

        # --- Detection ---
        t0 = time.perf_counter()
        detector = await detect_threat(transcript, audio_features)
        turn_latency["detection"] = (time.perf_counter() - t0) * 1000

        # --- Fusion ---
        t0 = time.perf_counter()
        fusion_score = fuse_signals(detector, audio_features, audio_emotion)
        alert_fired = should_alert(fusion_score)
        turn_latency["fusion"] = (time.perf_counter() - t0) * 1000

        total_ms = sum(turn_latency.values())

        turn_result = TurnResult(
            turn_id=turn_id,
            badge_id=badge_id,
            source_file=source_file,
            audio_features=audio_features,
            transcript=transcript,
            detector=detector,
            fusion_score=fusion_score,
            alert_fired=alert_fired,
            latency_ms=turn_latency,
        )
        results.append(turn_result)

        # Log per-turn summary
        print(
            f"[TURN {turn.turn_index}] "
            f"{turn.start_ms:.0f}-{turn.end_ms:.0f}ms "
            f"({audio_features.duration_seconds:.2f}s) "
            f"rms={audio_features.rms_db:.1f}dB "
            f"zcr={audio_features.zero_crossing_rate:.3f} "
            f"abrupt={audio_features.abrupt_change_score:.3f}"
        )
        print(
            f"[TURN {turn.turn_index}] "
            f"transcript: \"{transcript.cleaned_text[:80]}{'...' if len(transcript.cleaned_text) > 80 else ''}\""
        )
        print(
            f"[TURN {turn.turn_index}] "
            f"category={detector.threat_category} "
            f"directed={detector.is_directed} "
            f"keywords={detector.keywords_matched} "
            f"llm_flags={detector.llm_flags}"
        )
        print(
            f"[TURN {turn.turn_index}] fusion_score={fusion_score:.4f} "
            f"-> {'ALERT FIRED' if alert_fired else 'no alert'} "
            f"(total={total_ms:.0f}ms)"
        )

        # --- Publish if alert ---
        if alert_fired:
            decision = incident_manager.process_alert(badge_id, turn_id, fusion_score)
            print(
                f"[INCIDENT] {decision.alert_type} "
                f"incident={decision.incident.incident_id} "
                f"publish={decision.should_publish} "
                f"({decision.reason})"
            )
            if decision.should_publish:
                event = AlertEvent(
                    alert_type=decision.alert_type,
                    badge_id=badge_id,
                    turn_id=turn_id,
                    source_file=source_file,
                    incident_id=decision.incident.incident_id,
                    turns_in_incident=decision.incident.turn_ids,
                    transcript_excerpt=transcript.cleaned_text,
                    threat_category=detector.threat_category,
                    fusion_score=fusion_score,
                    audio_signals={
                        "rms_db": audio_features.rms_db,
                        "peak_db": audio_features.peak_db,
                        "abrupt_change": audio_features.abrupt_change_score,
                    },
                    reasoning=detector.reasoning,
                    severity=_score_to_severity(fusion_score),
                    total_pipeline_latency_ms=total_ms,
                )
                await publish(event)

    print(f"[PIPELINE] Done. {len(results)} turn(s), {sum(1 for r in results if r.alert_fired)} alert(s).")
    return results


def _score_to_severity(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.7:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"


def main():
    parser = argparse.ArgumentParser(description="Audio Threat Detection Pipeline")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to WAV file to process",
    )
    parser.add_argument(
        "--badge-id",
        type=str,
        default="badge_001",
        help="Badge identifier (default: badge_001)",
    )
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"[ERROR] File not found: {args.audio}")
        return

    asyncio.run(run_pipeline(args.audio, args.badge_id))


if __name__ == "__main__":
    main()
