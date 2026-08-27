"""Tunable settings for the live captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Audio ---------------------------------------------------------------
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # Silero VAD v6 works on 512-sample frames @16k == 32 ms
FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000.0


@dataclass
class Settings:
    # --- Model ---
    # large-v3 (not turbo): turbo's pruned 4-layer decoder degrades 2-4x more on
    # accented/meeting speech than on clean audio, and accents are the point here.
    model_size: str = "large-v3"
    device: str = "auto"
    # Resolved at startup by hardware.resolve_device / compute_type_for: float16 on a GPU,
    # int8 on CPU where it is the difference between usable and not. (On Turing GPUs int8
    # measured ~15% slower than float16, so it is not a win there.)
    compute_type: str = "float16"
    language: str | None = "en"

    # Greedy decoding for provisional text (fast), beam search for final (accurate).
    partial_beam_size: int = 1
    final_beam_size: int = 5

    # --- Decoding robustness ---
    # Retry a segment at higher temperature when the greedy result looks degenerate.
    # Costs an occasional extra pass; meaningfully more robust on noisy far-field audio.
    temperature_fallback: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    # Above this gzip compression ratio the output is repetitive - a hallucination loop.
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # --- Context ---
    # Recent finalised text is fed back as an initial_prompt so Whisper has conversational
    # context. This helps proper nouns and continuity, but unbounded feedback can trigger
    # hallucination loops, so it is capped and expires after a silence.
    use_context_prompt: bool = True
    context_chars: int = 220
    context_expiry_s: float = 30.0
    # Names, places and jargon the model would otherwise mangle. Prepended to the prompt.
    vocabulary: tuple[str, ...] = ()

    # --- Audio preprocessing ---
    # Deliberately NOT noise suppression: neural denoising measurably lowers ASR accuracy
    # and disproportionately clips accented speakers. These are conservative fixes only.
    highpass_hz: float = 80.0     # remove HVAC rumble / handling noise below speech
    target_rms: float = 0.06      # gentle level normalisation for quiet far-field audio
    max_gain: float = 8.0         # cap so near-silence isn't amplified into noise

    # --- Speaker labelling ---
    enable_speakers: bool = True
    # WeSpeaker CAM++ trained on VoxCeleb. Renamed from the upstream filename, which contains
    # "++" — MakeAppx percent-encodes those into the MSIX part name, and relying on the
    # installer to decode them back is a needless hazard for a path resolved by literal name.
    speaker_model: str = "speaker-embedding-campplus-en.onnx"
    speaker_threshold: float = 0.50
    max_speakers: int = 8
    # Embeddings need ~2-3 s of speech to be dependable; guard against short turns.
    speaker_min_identify_s: float = 1.0
    speaker_min_new_s: float = 2.0
    # Speakers the user has muted - typically their own voice. Muted speakers are
    # identified but never sent to Whisper, which saves the whole ~350 ms decode.
    muted_speakers: tuple[int, ...] = ()

    # --- Endpointing ---
    vad_start_threshold: float = 0.6
    vad_end_threshold: float = 0.35
    # Consecutive speech frames required to open an utterance (debounces clicks/knocks).
    start_frames: int = 3
    # Silence before an utterance is considered finished. This dominates perceived
    # latency far more than GPU speed does.
    end_silence_ms: int = 520
    # Audio kept before speech onset so word onsets aren't clipped.
    pre_roll_ms: int = 320
    # Utterances shorter than this are discarded as noise.
    min_utterance_ms: int = 300
    # Force a commit on long monologues so partial latency stays bounded.
    max_utterance_s: float = 20.0

    # --- Cadence ---
    # Minimum gap between provisional re-transcriptions.
    partial_interval_ms: int = 450
    # Don't attempt a partial until there's enough audio to be meaningful.
    min_partial_ms: int = 700

    # --- Network ---
    host: str = "127.0.0.1"
    ws_port: int = 8766
    http_port: int = 8765

    # --- Input device ---
    input_device: int | str | None = None
    # A WASAPI loopback endpoint index. When set it replaces the microphone, so what is
    # played through that output gets captioned instead of what is spoken.
    loopback_device: int | None = None

    # Where a recording is written when the user presses record. None means the default in
    # recorder.default_root(). Nothing is created until a recording actually starts, so an
    # install that never records leaves nothing behind.
    recordings_path: str | None = None

    # Phrases Whisper commonly hallucinates over near-silence.
    hallucinations: tuple[str, ...] = field(
        default_factory=lambda: (
            "thank you.",
            "thanks for watching!",
            "thank you for watching!",
            "you",
            ".",
            "bye.",
            "please subscribe.",
        )
    )

    # Segments the model itself scores as more likely silence than speech. Real speech in
    # this app measures around 0.002, so 0.6 only catches material the model is already
    # unsure is speech at all. faster-whisper's own no_speech_threshold is not enough on its
    # own: it suppresses only when the log-prob check ALSO fails, and hallucinated caption
    # credits are decoded confidently.
    drop_no_speech_above: float = 0.6

    # Below this, a word is shown as uncertain. Chosen from measurement: on a clean decode
    # words sit at 0.97-1.00, while genuinely ambiguous ones drop sharply (a misheard leading
    # article measured 0.19), so the gap is wide and 0.55 sits well inside it.
    low_confidence_below: float = 0.55
