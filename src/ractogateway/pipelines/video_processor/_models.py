"""Models for VideoProcessorPipeline."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeduplicationMethod(str, Enum):
    """Frame similarity algorithm used for deduplication."""

    PHASH = "phash"  # perceptual hash — fast, no ML (default)
    SSIM = "ssim"  # structural similarity — more accurate, needs scikit-image


class FrameAnalysisMode(str, Enum):
    """How frames are sent to the vision LLM."""

    INDIVIDUAL = "individual"  # one API call per frame
    GRID = "grid"  # stitch N frames into a collage → one API call


class VideoProcessingMode(str, Enum):
    """How much of the video should be processed."""

    ACTIVE = "active"  # process the full video
    PASSIVE = "passive"  # process only a focused time window


class TranscriberBackend(str, Enum):
    """Audio transcription backend."""

    # ── Local / open-source ──────────────────────────────────────────────
    FASTER_WHISPER = "faster-whisper"  # faster-whisper lib (default)
    OPENAI_WHISPER = "openai-whisper"  # openai-whisper lib
    HUGGINGFACE_LOCAL = "huggingface-local"  # HF transformers ASR pipeline

    # ── Cloud APIs ───────────────────────────────────────────────────────
    OPENAI_API = "openai-api"  # OpenAI Whisper API
    GOOGLE_API = "google-api"  # Google Cloud Speech-to-Text v2
    HUGGINGFACE_API = "huggingface-api"  # HuggingFace Inference API
    GROQ_API = "groq-api"  # Groq (ultra-fast Whisper cloud)
    DEEPGRAM_API = "deepgram-api"  # Deepgram Nova

    # ── Self-hosted ──────────────────────────────────────────────────────
    OLLAMA = "ollama"  # Ollama server (audio-capable models)


# ---------------------------------------------------------------------------
# VideoInput type alias
# ---------------------------------------------------------------------------

#: Accepted video sources:
#:  - str / Path  → local file path, or http/https URL, or YouTube URL
#:  - bytes       → raw video bytes (written to a temp file internally)
#:  - list        → pre-extracted frame image paths (skip extraction step)
VideoInput = str | Path | bytes | list


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class VideoConfig(BaseModel):
    """All tunable hyperparameters for VideoProcessorPipeline."""

    # ── Frame extraction ─────────────────────────────────────────────────
    fps: float = 1.0
    """Frames to sample per second of video."""

    similarity_threshold: float = 90.0
    """Discard a frame whose similarity to the previous kept frame is >= this %.
    Lower = keep more frames. Range 0-100."""

    max_frames: int | None = None
    """Hard cap on frames kept (None = no cap)."""

    dedup_method: DeduplicationMethod = DeduplicationMethod.PHASH
    """Algorithm used to compare frame similarity."""

    frame_format: str = "JPEG"
    """Image format for kept frames: 'JPEG' (smaller) or 'PNG' (lossless)."""

    # ── Vision analysis ──────────────────────────────────────────────────
    analyze_frames: bool = True
    """Pass kept frames to the vision LLM for content extraction."""

    frame_analysis_mode: FrameAnalysisMode = FrameAnalysisMode.INDIVIDUAL
    """Individual = one LLM call per frame; Grid = stitch frames into a collage."""

    grid_size: int = 4
    """Number of frames per grid collage (used when frame_analysis_mode='grid')."""

    batch_size: int = 10
    """How many frames to submit to the LLM concurrently per batch."""

    max_workers: int = 4
    """Thread-pool size for concurrent LLM frame analysis calls."""

    max_process_workers: int = 4
    """Process-pool size for CPU-bound frame extraction / hashing."""

    # ── Audio transcription ──────────────────────────────────────────────
    transcribe_audio: bool = True
    """Extract and transcribe the video's audio track."""

    transcriber_backend: TranscriberBackend = TranscriberBackend.FASTER_WHISPER
    """Which transcription engine to use."""

    transcriber_model: str = "base"
    """Model name / size — interpretation is backend-specific.

    Examples:
      faster-whisper / openai-whisper  : "tiny" "base" "small" "medium" "large-v3"
      huggingface-local / -api         : HF model ID e.g. "openai/whisper-large-v3"
      openai-api                       : "whisper-1"
      google-api                       : "long" "short" "latest_long"
      groq-api                         : "whisper-large-v3" "whisper-large-v3-turbo"
      deepgram-api                     : "nova-3" "nova-2" "enhanced" "base"
      ollama                           : model name on server e.g. "whisper"
    """

    transcriber_api_key: str | None = None
    """API key for cloud transcription backends (falls back to env vars)."""

    transcriber_base_url: str | None = None
    """Base URL for self-hosted / Ollama transcription endpoints."""

    language: str | None = None
    """BCP-47 language code (e.g. 'en', 'fr'). None = auto-detect."""

    # ── Output ───────────────────────────────────────────────────────────
    generate_summary: bool = True
    """Generate a comprehensive textual summary at the end."""

    store_in_rag: bool = False
    """Push all extracted content into the supplied rag_pipeline for Q&A."""

    processing_mode: VideoProcessingMode = VideoProcessingMode.ACTIVE
    """`active` processes full video; `passive` processes only a time window."""

    focus_time_seconds: float | None = None
    """Center timestamp in seconds for passive mode (e.g. 130 for 02:10)."""

    window_seconds: float = 5.0
    """Passive-mode half-window size in seconds (focus ± window_seconds)."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class FrameEntry(BaseModel):
    """One video frame, after extraction and optional analysis."""

    frame_id: int
    """Zero-based sequential frame identifier."""

    timestamp: float
    """Position in the video in seconds."""

    similarity_to_prev: float | None = None
    """Similarity percentage to the previous kept frame (None for first frame)."""

    kept: bool = True
    """False if discarded by the deduplication step."""

    analysis: str | None = None
    """LLM-generated description of visual content (whiteboard, screen, etc.)."""

    image_data: bytes | None = None
    """Raw image bytes for kept + analyzed frames."""

    image_format: str = "JPEG"


class TranscriptSegment(BaseModel):
    """A time-bounded transcription segment aligned to frame IDs."""

    start: float
    """Segment start time in seconds."""

    end: float
    """Segment end time in seconds."""

    text: str
    """Transcribed text for this segment."""

    frame_ids: list[int] = Field(default_factory=list)
    """IDs of kept frames whose timestamps fall within [start, end]."""


class VideoSection(BaseModel):
    """A merged time section combining visual analysis + audio transcript."""

    timestamp_start: float
    timestamp_end: float
    frame_ids: list[int] = Field(default_factory=list)
    visual_content: str = ""
    """Combined LLM analyses for all frames in this section."""

    audio_content: str = ""
    """Concatenated transcript text for this section's time range."""


class VideoProcessorUsage(BaseModel):
    """Accounting of tokens and frame counts across the full pipeline."""

    frames_extracted: int = 0
    frames_kept: int = 0
    frames_discarded: int = 0
    analysis_input_tokens: int = 0
    analysis_output_tokens: int = 0
    summary_input_tokens: int = 0
    summary_output_tokens: int = 0
    audio_duration_seconds: float = 0.0

    @property
    def total_analysis_tokens(self) -> int:
        return self.analysis_input_tokens + self.analysis_output_tokens

    @property
    def total_summary_tokens(self) -> int:
        return self.summary_input_tokens + self.summary_output_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_analysis_tokens + self.total_summary_tokens


# ---------------------------------------------------------------------------
# Stage error
# ---------------------------------------------------------------------------


class StageError(BaseModel):
    """Structured record of a failure in one pipeline stage."""

    stage: str
    """Name of the pipeline stage that failed (e.g. 'extract', 'transcribe')."""

    error_type: str
    """Exception class name (e.g. 'ImportError', 'RuntimeError')."""

    message: str
    """str(exc) — the error message."""

    traceback: str | None = None
    """Full Python traceback as a string (available in safe_mode)."""

    def __str__(self) -> str:
        return f"[{self.stage}] {self.error_type}: {self.message}"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class VideoProcessorResult(BaseModel):
    """Full output of a VideoProcessorPipeline run."""

    video_path: str
    """Original source identifier (path, URL, or '<bytes>' for buffer input)."""

    frames: list[FrameEntry] = Field(default_factory=list)
    """All extracted frames (kept and discarded)."""

    transcript: list[TranscriptSegment] = Field(default_factory=list)
    """Audio transcript segmented by timestamp."""

    sections: list[VideoSection] = Field(default_factory=list)
    """Merged visual + audio sections ordered by time."""

    summary: str | None = None
    """Comprehensive LLM-generated summary of the entire video."""

    rag_stored: bool = False
    rag_chunk_count: int = 0

    usage: VideoProcessorUsage = Field(default_factory=VideoProcessorUsage)

    error: str | None = None
    """Short description of the first fatal error (backward-compatible)."""

    failed_stage: str | None = None
    """Name of the stage that caused a fatal pipeline abort, if any."""

    stage_errors: list[StageError] = Field(default_factory=list)
    """All per-stage errors collected during the run (fatal + non-fatal)."""

    processing_mode: VideoProcessingMode = VideoProcessingMode.ACTIVE
    """Whether this run processed full video (`active`) or a window (`passive`)."""

    window_start_seconds: float | None = None
    """Passive-mode window start timestamp in source-video seconds."""

    window_end_seconds: float | None = None
    """Passive-mode window end timestamp in source-video seconds."""

    question: str | None = None
    """Optional user question answered from this run."""

    answer: str | None = None
    """Answer generated for `question`, when question-answer mode is used."""

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def has_errors(self) -> bool:
        """True if any stage encountered an error."""
        return bool(self.stage_errors)

    @property
    def is_failed(self) -> bool:
        """True if the pipeline aborted early due to a fatal stage error."""
        return self.failed_stage is not None

    # ── Convenience accessors ────────────────────────────────────────────

    def get_transcript_text(self) -> str:
        """Full transcript as a single string."""
        return " ".join(s.text for s in self.transcript)

    def get_all_visual_content(self) -> str:
        """All frame analyses concatenated in timestamp order."""
        parts = []
        for f in self.frames:
            if f.kept and f.analysis:
                parts.append(f"[{f.timestamp:.1f}s] {f.analysis}")
        return "\n\n".join(parts)

    # ── Export helpers ───────────────────────────────────────────────────

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str | None:
        """Serialise result to JSON. Returns JSON string if path is None."""
        data = self.model_dump(exclude={"frames": {"__all__": {"image_data"}}})
        text = json.dumps(data, indent=indent, default=str)
        if path:
            Path(path).write_text(text, encoding="utf-8")
            return None
        return text

    def to_markdown(self, path: str | None = None) -> str | None:
        """Build a structured Markdown report. Returns string if path is None."""
        lines: list[str] = []

        lines.append(f"# Video Processing Report\n\n**Source:** `{self.video_path}`\n")
        lines.append(
            f"**Frames extracted:** {self.usage.frames_extracted}  "
            f"**Kept:** {self.usage.frames_kept}  "
            f"**Discarded:** {self.usage.frames_discarded}\n"
        )

        if self.failed_stage:
            lines.append(f"> **Pipeline aborted at stage: `{self.failed_stage}`**\n")

        if self.stage_errors:
            lines.append("## Errors\n")
            for se in self.stage_errors:
                lines.append(f"### `{se.stage}` — {se.error_type}\n\n```\n{se.message}\n```\n")
                if se.traceback:
                    lines.append(f"<details><summary>Traceback</summary>\n\n```\n{se.traceback}```\n</details>\n")

        if self.summary:
            lines.append(f"## Summary\n\n{self.summary}\n")

        if self.transcript:
            lines.append("## Full Transcript\n")
            for seg in self.transcript:
                lines.append(f"**[{seg.start:.1f}s - {seg.end:.1f}s]** {seg.text}\n")

        if self.sections:
            lines.append("## Sections\n")
            for sec in self.sections:
                lines.append(
                    f"### {sec.timestamp_start:.1f}s - {sec.timestamp_end:.1f}s\n"
                )
                if sec.visual_content:
                    lines.append(f"**Visual:** {sec.visual_content}\n")
                if sec.audio_content:
                    lines.append(f"**Audio:** {sec.audio_content}\n")

        text = "\n".join(lines)
        if path:
            Path(path).write_text(text, encoding="utf-8")
            return None
        return text


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VideoRateLimitExceededError(RuntimeError):
    """Raised when a rate_limiter denies a VideoProcessorPipeline request."""
