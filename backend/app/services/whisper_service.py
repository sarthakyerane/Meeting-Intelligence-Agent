"""
Audio transcription using faster-whisper (CTranslate2 backend) with
speaker diarization via pyannote.audio (Week 2).

Pipeline:
  1. faster-whisper → word-level timestamps + transcript segments
  2. pyannote speaker-diarization-3.1 → speaker turn timeline
  3. Merge: assign speaker label to each whisper word using time overlap
  4. Return diarized transcript: "[SPEAKER_01 00:01-00:08] John said we should..."

Graceful degradation: if HF_TOKEN is missing or pyannote fails,
falls back to plain text transcript automatically.
"""

import logging
import tempfile
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_whisper_model = None
_diarize_pipeline = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("[WHISPER] Loading faster-whisper base model (int8 CPU)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("[WHISPER] Model loaded.")
    return _whisper_model


def _get_diarize_pipeline():
    """Load pyannote diarization pipeline. Returns None if HF_TOKEN not set."""
    global _diarize_pipeline
    if _diarize_pipeline is not None:
        return _diarize_pipeline

    from app.config import get_settings
    settings = get_settings()

    if not settings.hf_token:
        logger.warning("[DIARIZE] HF_TOKEN not set — speaker diarization disabled.")
        return None

    try:
        from pyannote.audio import Pipeline
        logger.info("[DIARIZE] Loading pyannote speaker-diarization-3.1 (first run downloads ~1GB)...")
        _diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )
        logger.info("[DIARIZE] Pipeline loaded.")
        return _diarize_pipeline
    except Exception as e:
        logger.error(f"[DIARIZE] Failed to load pyannote pipeline: {e}")
        return None


def _format_ts(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _merge_diarization(segments, diarization) -> str:
    """
    Merge faster-whisper segments with pyannote speaker turns.

    Strategy: for each whisper segment, find which pyannote speaker 
    has the most overlap with that segment's time window and assign 
    that speaker label. Groups consecutive segments by the same speaker.
    """
    annotated = []

    for seg in segments:
        seg_start = seg.start
        seg_end = seg.end
        seg_text = seg.text.strip()

        # Find best overlapping speaker from pyannote
        best_speaker = "SPEAKER_?"
        best_overlap = 0.0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        annotated.append((seg_start, seg_end, best_speaker, seg_text))

    # Group consecutive segments with same speaker
    lines = []
    current_speaker = None
    current_start = 0.0
    current_end = 0.0
    current_texts = []

    for start, end, speaker, text in annotated:
        if speaker != current_speaker:
            if current_texts:
                lines.append(
                    f"[{current_speaker} {_format_ts(current_start)}-{_format_ts(current_end)}] "
                    f"{' '.join(current_texts)}"
                )
            current_speaker = speaker
            current_start = start
            current_end = end
            current_texts = [text]
        else:
            current_end = end
            current_texts.append(text)

    # Flush last group
    if current_texts:
        lines.append(
            f"[{current_speaker} {_format_ts(current_start)}-{_format_ts(current_end)}] "
            f"{' '.join(current_texts)}"
        )

    return "\n".join(lines)


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.mp3") -> str:
    """
    Transcribe audio bytes → transcript string.

    If pyannote is available and HF_TOKEN is set:
      Returns diarized transcript with speaker labels.
    Otherwise:
      Returns plain text transcript.
    """
    suffix = os.path.splitext(filename)[1] or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # ── Step 1: faster-whisper transcription ────────────────────────────
        model = _get_whisper_model()
        segments_gen, info = model.transcribe(
            tmp_path,
            beam_size=5,
            word_timestamps=True,   # needed for accurate speaker assignment
        )
        logger.info(f"[WHISPER] Language: {info.language} ({info.language_probability:.0%})")

        # Materialise the generator — pyannote also needs the audio file open
        segments = list(segments_gen)
        logger.info(f"[WHISPER] {len(segments)} segments transcribed")

        # ── Step 2: pyannote diarization ────────────────────────────────────
        pipeline = _get_diarize_pipeline()

        if pipeline is not None:
            try:
                import torch
                logger.info("[DIARIZE] Running speaker diarization...")
                diarization = pipeline(tmp_path)
                transcript = _merge_diarization(segments, diarization)
                logger.info(f"[DIARIZE] Diarized transcript: {len(transcript)} chars")
                return transcript
            except Exception as e:
                logger.warning(f"[DIARIZE] Diarization failed, falling back to plain text: {e}")

        # ── Fallback: plain text ─────────────────────────────────────────────
        transcript = " ".join(seg.text.strip() for seg in segments)
        logger.info(f"[WHISPER] Plain transcript: {len(transcript)} chars")
        return transcript

    finally:
        os.unlink(tmp_path)
