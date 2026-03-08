# Audio Threat Detection Service

Real-time audio threat detection prototype for an AI wearable badge. Listens to audio like a live wearable, segments into spoken turns, transcribes with Whisper, reasons about threats using deterministic heuristics + LLM structured output, and publishes structured alerts to a subscriber service.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in your OpenAI API key
cp .env.example .env

# 4. Terminal 1 — start the alert subscriber service
python subscriber/main.py
# Verify: curl http://localhost:8765/health → {"status":"ok"}

# 5. Terminal 2 — run the pipeline on a WAV file
python cli.py --audio audio/heated_argument.wav
```

Alerts appear in Terminal 1 in real time and are persisted to `alerts.jsonl`.

### Bonus path A — GPT-4o audio emotion (opt-in)

Add `ENABLE_AUDIO_EMOTION=true` to `.env`. The pipeline will send each audio turn directly to `gpt-4o-audio-preview` and log prosodic emotion signals (anger level, urgency, stress) that text transcription alone cannot capture.

```bash
# .env
OPENAI_API_KEY=sk-...
ENABLE_AUDIO_EMOTION=true
```

### Run tests (no API key needed)

```bash
python -m pytest tests/ -v
```

All 63 tests run offline using synthetic audio and mocked API calls.

### Corpus Results

| File | Alert | Score | Category | Directed | Correct |
|------|-------|-------|----------|----------|---------|
| casual_chat.wav | None | 0.00 | none | No | Yes |
| heated_argument.wav | MEDIUM | 0.58 | harassment | Yes | Yes |
| keyword_only.wav | HIGH | 0.79 | physical_threat | Yes | Yes |
| false_positive_tv.wav | None | 0.03 | ambient_anger | No | Yes |
| muffled_noise.wav | None | 0.00 | none | No | Yes |

Tests: `python -m pytest tests/ -v` -- 63 tests, 0 failures, < 8s runtime (no API calls).

## Architecture

```
WAV file → Ingestion (read + metadata)
         → Segmenter (energy VAD → turn boundaries)
         → Audio Features (RMS, ZCR, abrupt change)
         → Transcriber (Whisper API + cleanup)
         → Detector (keyword heuristics + LLM structured output)
         → Fusion (weighted score + threshold)
         → Incident Manager (cooldown / grouping)
         → Publisher (HTTP POST → subscriber service)
```

The pipeline service and alert subscriber are fully decoupled. The pipeline publishes `AlertEvent` JSON to a subscriber endpoint over HTTP. The publisher module (`src/publisher.py`) targets a configurable URL — swapping in Google Cloud Pub/Sub, SNS, or a webhook relay requires changing one config value, not the pipeline logic.

## WAV Test Corpus

| File | Expected behavior | Tests |
|------|-------------------|-------|
| `casual_chat.wav` | No alert | Low fusion score, `threat_category: none` |
| `heated_argument.wav` | MEDIUM/HIGH alert | `is_directed: True`, verbal threat detected |
| `keyword_only.wav` | HIGH alert | Keyword fast path, low latency |
| `false_positive_tv.wav` | No alert | `is_directed: False`, ambient anger classified correctly |
| `muffled_noise.wav` | No alert / LOW | Graceful degradation on low-confidence transcript |

## Design Notes

### VAD / End-of-Turn Strategy

Energy-threshold silence detection. Chosen over webrtcvad (C dependency) and Silero VAD (PyTorch dependency) for simplicity and zero external dependencies beyond numpy.

**Parameters:**
- Frame size: 30ms
- Energy threshold: -35dB (frames above this are "speech")
- Minimum silence gap: 600ms (splits turns)
- Minimum turn length: 300ms (discards noise bursts)

**Corpus results:**
| File | Turns detected | Notes |
|------|---------------|-------|
| casual_chat.wav | 3 | Clean separation at natural pauses |
| heated_argument.wav | 1 | Continuous shouting, no silence gaps |
| keyword_only.wav | 1 | Single short utterance |
| false_positive_tv.wav | 2 | TV audio + brief second segment |
| muffled_noise.wav | 1 | Continuous noise, fallback to single turn |

**Tradeoff:** May miss turns in very quiet speech or split mid-word on pauses. Acceptable for prototype -- real production would use Silero VAD or a tuned WebRTC model.

### Audio Feature Signals

Four deterministic features extracted per turn, computed from 16-bit PCM via numpy:

- **RMS dB**: Overall loudness. `heated_argument.wav` at -16.2dB vs `casual_chat.wav` at -22 to -26dB shows good separation.
- **Peak dB**: Maximum amplitude. Detects clipping or shouts.
- **Zero-crossing rate (ZCR)**: Fraction of adjacent samples crossing zero. Higher for noise/unvoiced sounds vs clean speech.
- **Abrupt change score**: Maximum RMS delta between consecutive 200ms windows, normalized to 0-1. Detects sudden shouts or slams. `heated_argument.wav` scores 0.658 vs `casual_chat.wav` at 0.10-0.25.

For detailed derivations, interactive plots, and per-file analysis, see `audio_features_explained.ipynb` and `plots/REPORT.md`.

### Threat Taxonomy & Keywords

Four threat categories, each with regex patterns designed around the corpus:

| Category | Example patterns | Corpus hit |
|----------|-----------------|------------|
| physical_threat | "I will hurt/kill you", "hurt you" | keyword_only.wav |
| verbal_threat | "I'm going to hurt/kill", "you're dead" | -- |
| distress_signal | "help me", "let me go", "don't touch me" | -- |
| harassment | "give me your X or", "I'll hurt" | keyword_only.wav |

Keywords run first (< 1ms) as a fast deterministic scan. Matched keywords are passed to the LLM as additional context. They also provide a 25% boost in the fusion score.

**Design choice:** Keywords are high-precision, low-recall. They catch obvious threats instantly. The LLM handles nuance (e.g. `heated_argument.wav` has no keyword hits but is correctly flagged via semantic reasoning).

### Fusion Weights & Threshold

```
fusion_score = 0.60 * llm_threat_score
             + 0.25 * keyword_match (1.0 if any match, else 0.0)
             + 0.15 * audio_score (50% normalized RMS + 50% abrupt change)
             + 0.05 * mean(anger_level, urgency)   # additive; only when ENABLE_AUDIO_EMOTION=true
```

The emotion term is additive (base weights still sum to 1.0) and only applied when `ENABLE_AUDIO_EMOTION=true`. It uses the GPT-4o audio-preview prosodic signals — vocal anger and urgency — to stabilise borderline LLM variance. It is capped such that emotion alone cannot trigger an alert; it only resolves marginal cases.

**is_directed gate:** If `is_directed=False`, fusion score is capped at 0.4 (below threshold). This prevents false positives from ambient anger (TV, background arguments).

**Alert threshold:** 0.55. Calibrated against the corpus:
- keyword_only.wav: 0.79 (well above threshold)
- heated_argument.wav: 0.58–0.62 with emotion enabled (stabilised from 0.55–0.58 variance)
- false_positive_tv.wav: 0.03 (well below, gated by is_directed=False)
- casual_chat.wav: 0.00 (no threat signal)

### Cooldown / Incident Grouping

Per-badge state machine:

```
IDLE -- (alert fires) --> ACTIVE_INCIDENT -- (60s no trigger) --> COOLDOWN --> IDLE
```

During ACTIVE_INCIDENT:
- Subsequent turns are added to the existing incident
- Only publish INCIDENT_UPDATE if severity escalates (fusion_score > peak_score)
- Otherwise suppress to prevent alert spam

This ensures a 10-minute shouting match generates 1 initial alert + updates only on escalation, not 20 separate alerts.

### LLM Structured Output

Using `gpt-4o-mini` with OpenAI Structured Outputs (JSON schema `response_format`). The schema is:

```json
{
  "threat_score": 0.0-1.0,
  "threat_category": "none|verbal_threat|physical_threat|distress_signal|harassment|ambient_anger|unknown",
  "is_directed": true/false,
  "reasoning": "1-2 sentence explanation for ARC operator",
  "llm_flags": ["direct_threat", "profanity", "ultimatum", ...],
  "confidence": 0.0-1.0
}
```

Temperature set to 0.1 for determinism. The LLM receives the transcript, audio features, and keyword scan results as context.

**Key prompt design:** The system prompt explicitly instructs the LLM to treat second-person aggressive speech ("you promised", "give me your") as directed at the wearer, while TV/sports reactions are classified as ambient. This distinguishes `heated_argument.wav` (directed, fires alert) from `false_positive_tv.wav` (ambient, no alert).

## Time Breakdown

Work ran approximately 10:30 AM -- 11:00 PM (~12.5 hours wall clock, ~10 hours active).

| Phase | Time | What |
|-------|------|------|
| Planning & design | ~1 hr | Read task PDF, schema design, architecture decisions, documentation scaffold |
| Phase 1: Scaffold | ~1 hr | Stub pipeline, real publisher/subscriber, E2E validation, Windows encoding fixes |
| Phase 2: Audio | ~3 hr | Ingestion, audio feature notebook + plots, audio features (numpy), VAD segmenter, corpus validation |
| Phase 3: AI pipeline | ~2 hr | Whisper transcriber, keyword heuristics, LLM structured output detector, fusion weights, incident manager. Prompt iteration for is_directed classification. |
| Phase 4: Tests & polish | ~2 hr | 63 automated tests (audio_features, fusion, incident_manager, e2e), README finalization, retrospective |
| Bonus A: Audio emotion | ~45 min | GPT-4o audio-preview integration, fusion bonus weight, env var opt-in |
| Final review & cleanup | ~1 hr | Known limitations analysis, submission review, doc cleanup |
| Breaks / context switches | ~2 hr | Lunch, short breaks |

## Known Limitations

**Keyword scoring is naive.** `keyword_score = 1.0 if keywords_matched else 0.0` — binary, 25% weight, no negation handling, no phrase-level context. "I want to shoot hoops" gets the same boost as "I'll shoot you." A modest LLM score (0.30) paired with a keyword hit already approaches the 0.55 threshold (`0.60×0.30 + 0.25×1.0 = 0.43`), and any audio signal closes the gap.

**Audio normalization constants are corpus-overfit.** `RMS_THREAT_FLOOR = -25.0` and `RMS_THREAT_CEIL = -12.0` were anchored to 5 files. A whispered threat at −30 dB → `rms_norm = 0.0` (audio contributes nothing). A loud TV at −14 dB → `rms_norm = 0.85` (falsely elevated). These constants need calibration on a much larger, more diverse corpus.

**`heated_argument` alert margin is thin — LLM variance causes inconsistent results.** `gpt-4o-mini` at temperature 0.1 is not fully deterministic. On the same `heated_argument.wav` file it returns `threat_score=0.75` on some runs and `0.80` on others, producing scores of `0.5499` (no alert) and `0.5799` (alert fired) respectively. This was observed repeatedly in live testing.

Two mitigations exist, each with tradeoffs:

| Option | How | Pros | Cons |
|--------|-----|------|------|
| **Enable emotion bonus** (`ENABLE_AUDIO_EMOTION=true`) | GPT-4o audio-preview adds `+0.05 × mean(anger, urgency)` to the fusion score | Stabilises both LLM runs above threshold (0.59 and 0.62). Emotion signal is deterministic for the same audio. | ~2s extra latency per turn. Additional API cost. Occasional empty-response from audio model (handled gracefully). |
| **Lower the threshold** (`ALERT_THRESHOLD = 0.50` in `fusion.py`) | Moves the bar so even the low LLM run clears it | No extra API calls, no latency cost | Increases false positive risk — weaker signals may now fire. Requires re-validation against full corpus. |
| **Accept the variance** | Keep current defaults (`ENABLE_AUDIO_EMOTION=false`, threshold 0.55) | Simplest, fastest | `heated_argument` may silently miss on ~50% of runs. Unacceptable in production. |

**Current default:** `ENABLE_AUDIO_EMOTION=false`. For a reliable demo, set `ENABLE_AUDIO_EMOTION=true` in `.env`.

**`is_directed=False` is an absolute hard gate with no fallback.** If the LLM mistakenly classifies a genuine directed threat as `ambient_anger`, the alert is permanently suppressed — the formula returns at most 0.4, below threshold, with no warning. A logged soft-gate would be safer than a hard zero.

**No temporal context across turns.** Each turn is scored in isolation. An escalation pattern — calm → agitated → threatening across 3 turns — produces no special signal. The incident manager tracks the peak score but cannot detect a rising trend within a cooldown window.

**Tiny corpus.** The fusion weights (60/25/15) and threshold (0.55) were calibrated on 5 WAV files. No single audio feature separates threats from non-threats on its own — the multi-modal approach is necessary but the weight balance is essentially hand-tuned. Real production needs hundreds of labeled examples.

**VAD is energy-only.** The −35 dB silence threshold works for clean recordings. In a noisy environment, the VAD generates many false turns or misses quiet speech. A learned VAD model (Silero, WebRTC) generalizes significantly better.

**API latency.** Per-turn end-to-end latency is ~3-6 seconds, dominated by Whisper and LLM inference. This is too slow for immediate real-time response from a wearable.

**No streaming input.** The current implementation processes complete WAV files. A real wearable needs a streaming pipeline reading from a microphone buffer with real-time VAD frame-by-frame.

## What I'd Improve With More Time

**1. Replace binary keyword score with scored phrase matching.** Add negation detection ("I don't want to hurt you"), phrase-level context windows, and a graded score (0.0–1.0) rather than binary. This is the highest-risk gap in the current fusion formula.

**2. Pre-alert on audio features.** Fire a provisional alert in < 1ms based on RMS + abrupt change score; upgrade/downgrade once the LLM confirms. Reduces time-to-first-alert from ~4s to ~50ms — critical for a live safety device.

**3. Real VAD model (Silero VAD).** Drop-in replacement for the energy-threshold approach. Handles real-world noise, quiet speech, and overlapping speakers significantly better.

**4. Broader labeled corpus + evaluation harness.** Systematic precision/recall evaluation across 100+ labeled examples to anchor normalization constants and validate weight balance. Use Whisper's multi-language support for non-English scenarios.

**5. Dockerisation + Pub/Sub transport.** The publisher is already isolated behind `src/publisher.py`. Swapping to Google Cloud Pub/Sub is a one-file change. A docker-compose with Pub/Sub emulator + subscriber would make the demo fully self-contained.

**6. Streaming microphone input.** Wire the Turn segmentation logic to a live audio source (PyAudio or sounddevice), enabling true wearable simulation from a laptop microphone.

## Additional Artifacts

- **`audio_features_explained.ipynb`** -- Jupyter notebook walking through the audio feature extraction pipeline with interactive plots. Shows per-file waveforms, RMS frame-by-frame energy, ZCR patterns, abrupt change scores, and the fusion breakdown. Useful for understanding the intuition behind each deterministic signal.
- **`plots/REPORT.md`** -- Generated visual analysis report with 8 plots covering waveform comparison, RMS/ZCR distributions, abrupt change visualization, and a combined dashboard. Key finding: no single audio feature separates threats from non-threats, validating the multi-modal fusion approach.

---