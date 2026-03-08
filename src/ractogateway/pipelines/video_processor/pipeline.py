"""VideoProcessorPipeline — main entry point.

Processes a video through five stages:
  1. Load         — resolve source (path / URL / YouTube / bytes / frame list)
  2. Extract      — sample frames with OpenCV, deduplicate by similarity
  3. Transcribe   — extract audio, transcribe with chosen backend
  4. Analyse      — pass frames to vision LLM (individual or grid mode)
  5. Summarise    — produce comprehensive Markdown summary
  6. RAG store    — optionally index everything via RactoRAG

Usage::

    from ractogateway.openai_developer_kit import Chat
    from ractogateway.pipelines.video_processor import (
        VideoProcessorPipeline,
        DeduplicationMethod,
        TranscriberBackend,
    )

    kit = Chat(api_key="...", model="gpt-4o")

    pipeline = VideoProcessorPipeline(
        kit=kit,
        fps=1.0,
        similarity_threshold=85.0,
        transcriber=TranscriberBackend.FASTER_WHISPER,
        transcriber_model="base",
        analyze_frames=True,
        generate_summary=True,
        safe_mode=True,
    )

    result = pipeline.run("lecture.mp4")
    print(result.summary)
    print(result.get_transcript_text())

    # YouTube / URL / bytes also accepted:
    result = pipeline.run("https://www.youtube.com/watch?v=...")
    result = pipeline.run("https://example.com/video.mp4")
    result = pipeline.run(video_bytes)           # bytes
    result = pipeline.run(["frame1.jpg", ...])   # pre-extracted frames
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import traceback as _tb
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ._models import (
    DeduplicationMethod,
    FrameAnalysisMode,
    StageError,
    TranscriberBackend,
    VideoProcessingMode,
    VideoProcessorResult,
    VideoProcessorUsage,
    VideoRateLimitExceededError,
)

# Sentinel for "not supplied by caller"
_UNSET = object()


def _make_stage_error(stage: str, exc: BaseException) -> StageError:
    """Capture a stage failure with full traceback into a StageError."""
    return StageError(
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
        traceback=_tb.format_exc(),
    )


class VideoProcessorPipeline:
    """Synchronous + asynchronous video processing pipeline.

    Parameters
    ----------
    kit:
        A RactoGateway developer kit (``Chat``) used for both frame analysis and
        summary generation unless *analysis_kit* or *summary_kit* are provided.
    analysis_kit:
        Optional separate kit for vision/frame analysis (e.g. a vision-specific
        model).  Falls back to *kit* when not supplied.
    summary_kit:
        Optional separate kit for summary generation (e.g. a larger model).
        Falls back to *kit* when not supplied.
    transcriber:
        Which audio transcription backend to use.
    transcriber_model:
        Model name / size for the chosen backend.
    transcriber_api_key:
        API key for cloud transcription backends (or read from env vars).
    transcriber_base_url:
        Base URL for self-hosted endpoints (Ollama etc.).
    fps:
        Video frames to sample per second.
    similarity_threshold:
        Frames with similarity >= this % to the previous kept frame are discarded.
        E.g. ``90.0`` keeps frames that differ by more than 10 %.
    dedup_method:
        :attr:`DeduplicationMethod.PHASH` (fast, default) or
        :attr:`DeduplicationMethod.SSIM` (more accurate).
    max_frames:
        Hard cap on the number of kept frames (``None`` = no cap).
    frame_format:
        ``"JPEG"`` (smaller, lossy) or ``"PNG"`` (lossless).
    frame_analysis_mode:
        :attr:`FrameAnalysisMode.INDIVIDUAL` (one LLM call per frame, default) or
        :attr:`FrameAnalysisMode.GRID` (stitch into a collage).
    grid_size:
        Frames per grid collage (only used in GRID mode).
    batch_size:
        Concurrent LLM calls per batch during frame analysis.
    max_workers:
        Thread-pool size for concurrent LLM calls.
    max_process_workers:
        Process-pool size for CPU-bound frame extraction / hashing.
    language:
        BCP-47 language code for transcription (``None`` = auto-detect).
    transcribe_audio:
        Whether to extract and transcribe the audio track.
    analyze_frames:
        Whether to pass frames to the vision LLM.
    generate_summary:
        Whether to generate a comprehensive summary at the end.
    rag_pipeline:
        An optional :class:`ractogateway.rag.pipeline.RactoRAG` instance.
        When supplied and *store_in_rag* is ``True`` (or per-call), all
        extracted content is indexed for retrieval.
    safe_mode:
        Catch all exceptions and return them in ``result.error`` instead of
        raising.
    tracer:
        Optional :class:`ractogateway.telemetry.RactoTracer` for OTEL tracing.
    metrics:
        Optional :class:`ractogateway.telemetry.GatewayMetricsMiddleware`.
    rate_limiter:
        Duck-typed rate limiter with ``check_and_consume(user_id, tokens)`` and
        ``get_remaining(user_id)`` methods.
    user_id:
        Default user identifier passed to the rate limiter.
    """

    def __init__(
        self,
        kit: Any,
        *,
        analysis_kit: Any = None,
        summary_kit: Any = None,
        # Transcription
        transcriber: TranscriberBackend = TranscriberBackend.FASTER_WHISPER,
        transcriber_model: str = "base",
        transcriber_api_key: str | None = None,
        transcriber_base_url: str | None = None,
        # Frame extraction
        fps: float = 1.0,
        similarity_threshold: float = 90.0,
        dedup_method: DeduplicationMethod = DeduplicationMethod.PHASH,
        max_frames: int | None = None,
        frame_format: str = "JPEG",
        # Analysis
        frame_analysis_mode: FrameAnalysisMode = FrameAnalysisMode.INDIVIDUAL,
        grid_size: int = 4,
        batch_size: int = 10,
        max_workers: int = 4,
        max_process_workers: int = 4,
        language: str | None = None,
        # Feature flags
        transcribe_audio: bool = True,
        analyze_frames: bool = True,
        generate_summary: bool = True,
        # Processing scope
        processing_mode: VideoProcessingMode = VideoProcessingMode.ACTIVE,
        focus_time_seconds: float | None = None,
        window_seconds: float = 5.0,
        # Integrations
        rag_pipeline: Any = None,
        # Safety & observability
        safe_mode: bool = False,
        tracer: Any = None,
        metrics: Any = None,
        rate_limiter: Any = None,
        user_id: str = "default",
    ) -> None:
        self._kit = kit
        self._analysis_kit = analysis_kit or kit
        self._summary_kit = summary_kit or kit

        self._transcriber_backend = TranscriberBackend(transcriber)
        self._transcriber_model = transcriber_model
        self._transcriber_api_key = transcriber_api_key
        self._transcriber_base_url = transcriber_base_url

        self._fps = fps
        self._similarity_threshold = similarity_threshold
        self._dedup_method = DeduplicationMethod(dedup_method)
        self._max_frames = max_frames
        self._frame_format = frame_format

        self._frame_analysis_mode = FrameAnalysisMode(frame_analysis_mode)
        self._grid_size = grid_size
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._max_process_workers = max_process_workers
        self._language = language

        self._transcribe_audio = transcribe_audio
        self._analyze_frames = analyze_frames
        self._generate_summary = generate_summary
        self._processing_mode = VideoProcessingMode(processing_mode)
        self._focus_time_seconds = focus_time_seconds
        self._window_seconds = window_seconds

        self._rag_pipeline = rag_pipeline
        self._safe_mode = safe_mode
        self._tracer = tracer
        self._metrics = metrics
        self._rate_limiter = rate_limiter
        self._user_id = user_id

        # Build transcriber once (loads model lazily)
        from ._transcriber import get_transcriber  # noqa: PLC0415

        self._transcriber_obj = get_transcriber(
            self._transcriber_backend,
            self._transcriber_model,
            self._transcriber_api_key,
            self._transcriber_base_url,
        )

    # ── Public sync API ──────────────────────────────────────────────────────

    def run(
        self,
        source: str | Path | bytes | list,
        *,
        # Per-call overrides
        fps: Any = _UNSET,
        similarity_threshold: Any = _UNSET,
        dedup_method: Any = _UNSET,
        max_frames: Any = _UNSET,
        analyze_frames: Any = _UNSET,
        frame_analysis_mode: Any = _UNSET,
        grid_size: Any = _UNSET,
        batch_size: Any = _UNSET,
        transcribe_audio: Any = _UNSET,
        language: Any = _UNSET,
        generate_summary: Any = _UNSET,
        processing_mode: Any = _UNSET,
        focus_time_seconds: Any = _UNSET,
        window_seconds: Any = _UNSET,
        store_in_rag: bool = False,
        user_id: Any = _UNSET,
    ) -> VideoProcessorResult:
        """Process *source* and return a :class:`VideoProcessorResult`.

        All keyword arguments override the constructor defaults for this call only.
        In ``safe_mode=True`` fatal stage errors are captured into
        ``result.failed_stage`` / ``result.stage_errors`` and the pipeline
        returns a partial result instead of raising.  Non-fatal stage errors
        (transcription, analysis, summary) are always captured into
        ``result.stage_errors`` so the pipeline continues with whatever data
        is available.
        """
        return self._run_pipeline(source, locals())

    async def arun(
        self,
        source: str | Path | bytes | list,
        *,
        fps: Any = _UNSET,
        similarity_threshold: Any = _UNSET,
        dedup_method: Any = _UNSET,
        max_frames: Any = _UNSET,
        analyze_frames: Any = _UNSET,
        frame_analysis_mode: Any = _UNSET,
        grid_size: Any = _UNSET,
        batch_size: Any = _UNSET,
        transcribe_audio: Any = _UNSET,
        language: Any = _UNSET,
        generate_summary: Any = _UNSET,
        processing_mode: Any = _UNSET,
        focus_time_seconds: Any = _UNSET,
        window_seconds: Any = _UNSET,
        store_in_rag: bool = False,
        user_id: Any = _UNSET,
    ) -> VideoProcessorResult:
        """Async variant of :meth:`run`."""
        return await self._arun_pipeline(source, locals())

    # ── Config resolution ────────────────────────────────────────────────────

    def _resolve(self, kwargs: dict[str, Any], name: str, default: Any) -> Any:
        val = kwargs.get(name, _UNSET)
        return default if val is _UNSET else val

    def _resolved_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "fps": self._resolve(kwargs, "fps", self._fps),
            "similarity_threshold": self._resolve(
                kwargs, "similarity_threshold", self._similarity_threshold
            ),
            "dedup_method": DeduplicationMethod(
                self._resolve(kwargs, "dedup_method", self._dedup_method)
            ),
            "max_frames": self._resolve(kwargs, "max_frames", self._max_frames),
            "analyze_frames": self._resolve(
                kwargs, "analyze_frames", self._analyze_frames
            ),
            "frame_analysis_mode": FrameAnalysisMode(
                self._resolve(kwargs, "frame_analysis_mode", self._frame_analysis_mode)
            ),
            "grid_size": self._resolve(kwargs, "grid_size", self._grid_size),
            "batch_size": self._resolve(kwargs, "batch_size", self._batch_size),
            "transcribe_audio": self._resolve(
                kwargs, "transcribe_audio", self._transcribe_audio
            ),
            "language": self._resolve(kwargs, "language", self._language),
            "generate_summary": self._resolve(
                kwargs, "generate_summary", self._generate_summary
            ),
            "processing_mode": VideoProcessingMode(
                self._resolve(kwargs, "processing_mode", self._processing_mode)
            ),
            "focus_time_seconds": self._resolve(
                kwargs, "focus_time_seconds", self._focus_time_seconds
            ),
            "window_seconds": self._resolve(
                kwargs, "window_seconds", self._window_seconds
            ),
            "store_in_rag": kwargs.get("store_in_rag", False),
            "user_id": self._resolve(kwargs, "user_id", self._user_id),
        }

    # ── Rate limiter ─────────────────────────────────────────────────────────

    def _check_rate_limit(self, user_id: str, tokens: int = 1) -> None:
        if self._rate_limiter is None:
            return
        allowed = self._rate_limiter.check_and_consume(user_id, tokens)
        if not allowed:
            raise VideoRateLimitExceededError(
                f"Rate limit exceeded for user '{user_id}'. "
                f"Remaining: {self._rate_limiter.get_remaining(user_id)}"
            )

    # ── Source label helper ──────────────────────────────────────────────────

    @staticmethod
    def _source_label(source: str | Path | bytes | list) -> str:
        if isinstance(source, (bytes, bytearray)):
            return "<bytes>"
        if isinstance(source, list):
            return f"<{len(source)} pre-extracted frames>"
        return str(source)

    @staticmethod
    def parse_timestamp(value: float | int | str) -> float:
        """Parse timestamp values like ``130``, ``"02:10"``, ``"2 mins 10 sec"``."""
        if isinstance(value, (int, float)):
            if float(value) < 0:
                raise ValueError("timestamp must be >= 0.")
            return float(value)

        text = value.strip().lower()
        if not text:
            raise ValueError("timestamp string cannot be empty.")

        # HH:MM:SS or MM:SS
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                minutes, seconds = parts
                return float(minutes) * 60.0 + float(seconds)
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
            raise ValueError(f"Unsupported timestamp format: {value!r}")

        # Human-readable units: "2 mins 10 sec", "1h 3m", "90s"
        total = 0.0
        for num, unit in re.findall(
            r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)",
            text,
        ):
            val = float(num)
            if unit.startswith("h"):
                total += val * 3600.0
            elif unit.startswith("m"):
                total += val * 60.0
            else:
                total += val
        if total > 0:
            return total

        # Plain numeric string -> seconds
        return float(text)

    def _resolve_window(self, cfg: dict[str, Any]) -> tuple[float | None, float | None]:
        mode = cfg["processing_mode"]
        if mode == VideoProcessingMode.ACTIVE:
            return None, None

        focus = cfg["focus_time_seconds"]
        if focus is None:
            raise ValueError(
                "focus_time_seconds is required when processing_mode='passive'."
            )
        focus_s = self.parse_timestamp(focus)
        if focus_s < 0:
            raise ValueError("focus_time_seconds must be >= 0.")

        window = float(cfg["window_seconds"])
        if window <= 0:
            raise ValueError("window_seconds must be > 0.")

        start = max(0.0, focus_s - window)
        end = focus_s + window
        return start, end

    @staticmethod
    def _shift_result_timestamps(
        result: VideoProcessorResult,
        *,
        offset_seconds: float,
    ) -> VideoProcessorResult:
        """Shift frame/transcript/section timestamps by an absolute offset."""
        if offset_seconds == 0:
            return result

        frames = [
            f.model_copy(update={"timestamp": f.timestamp + offset_seconds})
            for f in result.frames
        ]
        transcript = [
            seg.model_copy(
                update={
                    "start": seg.start + offset_seconds,
                    "end": seg.end + offset_seconds,
                }
            )
            for seg in result.transcript
        ]
        sections = [
            sec.model_copy(
                update={
                    "timestamp_start": sec.timestamp_start + offset_seconds,
                    "timestamp_end": sec.timestamp_end + offset_seconds,
                }
            )
            for sec in result.sections
        ]
        return result.model_copy(
            update={
                "frames": frames,
                "transcript": transcript,
                "sections": sections,
            }
        )

    # ── Sync pipeline execution ──────────────────────────────────────────────

    def _fatal_stage_result(
        self,
        label: str,
        usage: VideoProcessorUsage,
        stage_errors: list[StageError],
        err: StageError,
    ) -> VideoProcessorResult:
        """Return a partial result when a fatal stage fails in safe_mode."""
        all_errors = [*stage_errors, err]
        return VideoProcessorResult(
            video_path=label,
            usage=usage,
            failed_stage=err.stage,
            stage_errors=all_errors,
            error=str(err),
        )

    def _run_pipeline(
        self,
        source: str | Path | bytes | list,
        call_kwargs: dict[str, Any],
    ) -> VideoProcessorResult:
        from ._analyzer import analyze_frames_sync  # noqa: PLC0415
        from ._extractor import (  # noqa: PLC0415
            deduplicate_frames,
            extract_frames,
            extract_frames_window,
            load_frames_from_paths,
        )
        from ._loader import resolve_video_source  # noqa: PLC0415
        from ._rag import store_result_in_rag  # noqa: PLC0415
        from ._summarizer import build_sections, generate_summary_sync  # noqa: PLC0415
        from ._transcriber import align_frames_to_transcript  # noqa: PLC0415

        cfg = self._resolved_config(call_kwargs)
        label = self._source_label(source)
        usage = VideoProcessorUsage()
        stage_errors: list[StageError] = []
        processing_mode: VideoProcessingMode = cfg["processing_mode"]
        window_start_seconds: float | None = None
        window_end_seconds: float | None = None

        try:
            window_start_seconds, window_end_seconds = self._resolve_window(cfg)
        except Exception as exc:
            err = _make_stage_error("config", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise

        try:
            self._check_rate_limit(cfg["user_id"])
        except Exception as exc:
            err = _make_stage_error("rate_limit", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise

        # ── 1. Resolve source ── FATAL: nothing to process without a source ──
        try:
            video_path, frame_paths = resolve_video_source(source)
        except Exception as exc:
            err = _make_stage_error("load", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise

        if (
            processing_mode == VideoProcessingMode.PASSIVE
            and frame_paths is not None
        ):
            exc = ValueError(
                "processing_mode='passive' does not support pre-extracted frame lists. "
                "Provide a video file/URL/bytes source."
            )
            err = _make_stage_error("load", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise exc

        # ── 2. Extract frames ── FATAL: no frames = pipeline is meaningless ──
        try:
            if frame_paths is not None:
                raw_frames = load_frames_from_paths(
                    frame_paths, frame_format=self._frame_format
                )
            elif window_start_seconds is not None:
                raw_frames = extract_frames_window(
                    video_path,  # type: ignore[arg-type]
                    fps=cfg["fps"],
                    max_frames=cfg["max_frames"],
                    frame_format=self._frame_format,
                    start_time_seconds=window_start_seconds,
                    end_time_seconds=window_end_seconds,
                )
            else:
                raw_frames = extract_frames(
                    video_path,  # type: ignore[arg-type]
                    fps=cfg["fps"],
                    max_frames=cfg["max_frames"],
                    frame_format=self._frame_format,
                    max_process_workers=self._max_process_workers,
                )
        except Exception as exc:
            err = _make_stage_error("extract", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise RuntimeError(f"[Stage: extract] {err.error_type}: {err.message}") from exc

        usage.frames_extracted = len(raw_frames)

        # ── 3. Deduplicate ── FATAL: corrupted frame list is unrecoverable ───
        try:
            frames = deduplicate_frames(
                raw_frames,
                similarity_threshold=cfg["similarity_threshold"],
                method=cfg["dedup_method"],
            )
        except Exception as exc:
            err = _make_stage_error("deduplicate", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise RuntimeError(
                f"[Stage: deduplicate] {err.error_type}: {err.message}"
            ) from exc

        usage.frames_kept = sum(1 for f in frames if f.kept)
        usage.frames_discarded = sum(1 for f in frames if not f.kept)

        # ── 4. Transcription ── NON-FATAL: video still useful without audio ──
        transcript: list = []
        if cfg["transcribe_audio"] and video_path is not None:
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        self._run_transcription,
                        video_path,
                        cfg["language"],
                        usage,
                        window_start_seconds,
                        window_end_seconds,
                        window_start_seconds or 0.0,
                    )
                    transcript = fut.result()
                transcript = align_frames_to_transcript(frames, transcript)
            except Exception as exc:
                err = _make_stage_error("transcribe", exc)
                stage_errors.append(err)
                if not self._safe_mode:
                    raise RuntimeError(
                        f"[Stage: transcribe] {err.error_type}: {err.message}"
                    ) from exc

        # ── 5. Frame analysis ── NON-FATAL: frames still kept, just no LLM text
        if cfg["analyze_frames"]:
            try:
                frames = analyze_frames_sync(
                    frames,
                    self._analysis_kit,
                    mode=cfg["frame_analysis_mode"],
                    batch_size=cfg["batch_size"],
                    max_workers=self._max_workers,
                    grid_size=cfg["grid_size"],
                    usage=usage,
                )
            except Exception as exc:
                err = _make_stage_error("analyze", exc)
                stage_errors.append(err)
                if not self._safe_mode:
                    raise RuntimeError(
                        f"[Stage: analyze] {err.error_type}: {err.message}"
                    ) from exc

        # ── 6. Build sections ── NON-FATAL: result still has frames + transcript
        sections: list = []
        try:
            sections = build_sections(frames, transcript)
        except Exception as exc:
            err = _make_stage_error("build_sections", exc)
            stage_errors.append(err)
            if not self._safe_mode:
                raise RuntimeError(
                    f"[Stage: build_sections] {err.error_type}: {err.message}"
                ) from exc

        # ── 7. Summary ── NON-FATAL: rest of result is still valid ───────────
        summary: str | None = None
        if cfg["generate_summary"]:
            try:
                summary = generate_summary_sync(
                    sections, transcript, self._summary_kit, usage
                )
            except Exception as exc:
                err = _make_stage_error("summarize", exc)
                stage_errors.append(err)
                if not self._safe_mode:
                    raise RuntimeError(
                        f"[Stage: summarize] {err.error_type}: {err.message}"
                    ) from exc

        # ── 8. Build result ──────────────────────────────────────────────────
        top_error = str(stage_errors[0]) if stage_errors else None
        result = VideoProcessorResult(
            video_path=label,
            frames=frames,
            transcript=transcript,
            sections=sections,
            summary=summary,
            usage=usage,
            stage_errors=stage_errors,
            error=top_error,
            processing_mode=processing_mode,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
        )

        # ── 9. RAG storage ── NON-FATAL: indexing failure must not lose result
        if cfg["store_in_rag"] and self._rag_pipeline is not None:
            try:
                count = store_result_in_rag(result, self._rag_pipeline)
                result = result.model_copy(
                    update={"rag_stored": True, "rag_chunk_count": count}
                )
            except Exception as exc:
                err = _make_stage_error("rag_store", exc)
                if self._safe_mode:
                    result = result.model_copy(
                        update={
                            "stage_errors": [*result.stage_errors, err],
                            "error": result.error or str(err),
                        }
                    )
                else:
                    raise RuntimeError(
                        f"[Stage: rag_store] {err.error_type}: {err.message}"
                    ) from exc

        return result

    def _run_transcription(
        self,
        video_path: Path,
        language: str | None,
        usage: VideoProcessorUsage,
        start_time_seconds: float | None = None,
        end_time_seconds: float | None = None,
        time_offset_seconds: float = 0.0,
    ) -> list:
        """Run audio extraction + transcription in a thread."""
        from ._transcriber import extract_audio, get_audio_duration  # noqa: PLC0415

        audio_path = extract_audio(
            video_path,
            start_time_seconds=start_time_seconds,
            end_time_seconds=end_time_seconds,
        )
        with contextlib.suppress(Exception):
            usage.audio_duration_seconds = get_audio_duration(audio_path)
        segments = self._transcriber_obj.transcribe(audio_path, language)
        if time_offset_seconds:
            segments = [
                seg.model_copy(
                    update={
                        "start": seg.start + time_offset_seconds,
                        "end": seg.end + time_offset_seconds,
                    }
                )
                for seg in segments
            ]
        return segments

    # ── Async pipeline execution ─────────────────────────────────────────────

    @staticmethod
    def _build_qa_context(
        result: VideoProcessorResult,
        *,
        max_context_chars: int,
    ) -> str:
        lines: list[str] = []
        if result.sections:
            for sec in result.sections:
                lines.append(
                    f"[{sec.timestamp_start:.1f}s - {sec.timestamp_end:.1f}s]"
                )
                if sec.visual_content:
                    lines.append(f"VISUAL: {sec.visual_content}")
                if sec.audio_content:
                    lines.append(f"AUDIO: {sec.audio_content}")
                lines.append("")
        elif result.transcript:
            for seg in result.transcript:
                lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        else:
            for frame in result.frames:
                if frame.kept and frame.analysis:
                    lines.append(f"[{frame.timestamp:.1f}s] {frame.analysis}")

        context = "\n".join(lines).strip()
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n\n[context truncated]"
        return context

    def _qa_answer_sync(
        self,
        *,
        question: str,
        context: str,
    ) -> tuple[str, dict[str, int]]:
        from ractogateway._models.chat import ChatConfig  # noqa: PLC0415
        from ractogateway.prompts.engine import RactoPrompt  # noqa: PLC0415

        prompt = RactoPrompt(
            role="video question-answering specialist",
            aim=(
                "Answer the user question using only the provided video timeline context. "
                "Always anchor claims to timestamp evidence."
            ),
            constraints=[
                "Do not fabricate details not present in context.",
                "Cite timestamp evidence when available.",
                "If context is insufficient, explicitly say so.",
            ],
            tone="Professional and precise.",
            output_format="Markdown with sections: Answer, Evidence, Confidence",
            context=context,
        )

        try:
            resp = self._summary_kit.chat(
                ChatConfig(
                    user_message=question,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=900,
                )
            )
        except TypeError:
            resp = self._summary_kit.chat(prompt=prompt)

        return (resp.content or "").strip(), (resp.usage or {})

    async def _qa_answer_async(
        self,
        *,
        question: str,
        context: str,
    ) -> tuple[str, dict[str, int]]:
        from ractogateway._models.chat import ChatConfig  # noqa: PLC0415
        from ractogateway.prompts.engine import RactoPrompt  # noqa: PLC0415

        prompt = RactoPrompt(
            role="video question-answering specialist",
            aim=(
                "Answer the user question using only the provided video timeline context. "
                "Always anchor claims to timestamp evidence."
            ),
            constraints=[
                "Do not fabricate details not present in context.",
                "Cite timestamp evidence when available.",
                "If context is insufficient, explicitly say so.",
            ],
            tone="Professional and precise.",
            output_format="Markdown with sections: Answer, Evidence, Confidence",
            context=context,
        )

        try:
            resp = await self._summary_kit.achat(
                ChatConfig(
                    user_message=question,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=900,
                )
            )
        except TypeError:
            resp = await self._summary_kit.achat(prompt=prompt)

        return (resp.content or "").strip(), (resp.usage or {})

    def answer_question(
        self,
        source: str | Path | bytes | list,
        *,
        question: str,
        processing_mode: VideoProcessingMode | str = VideoProcessingMode.ACTIVE,
        focus_time: float | int | str | None = None,
        window_seconds: float = 5.0,
        max_context_chars: int = 40_000,
        **run_kwargs: Any,
    ) -> VideoProcessorResult:
        """Process video then answer a user question from extracted timeline context."""
        if not question.strip():
            raise ValueError("question cannot be empty.")

        mode = VideoProcessingMode(processing_mode)
        run_kwargs.setdefault("analyze_frames", True)
        run_kwargs.setdefault("transcribe_audio", True)
        run_kwargs.setdefault("generate_summary", False)
        run_kwargs["processing_mode"] = mode
        run_kwargs["window_seconds"] = window_seconds
        if focus_time is not None:
            run_kwargs["focus_time_seconds"] = self.parse_timestamp(focus_time)

        result = self.run(source, **run_kwargs)
        if result.failed_stage is not None:
            return result.model_copy(update={"question": question})

        context = self._build_qa_context(result, max_context_chars=max_context_chars)
        if not context:
            return result.model_copy(
                update={
                    "question": question,
                    "answer": "No analyzable context was extracted from this video run.",
                }
            )

        try:
            answer, usage = self._qa_answer_sync(question=question, context=context)
            updated_usage = result.usage.model_copy(
                update={
                    "summary_input_tokens": (
                        result.usage.summary_input_tokens
                        + int(usage.get("prompt_tokens", 0))
                    ),
                    "summary_output_tokens": (
                        result.usage.summary_output_tokens
                        + int(usage.get("completion_tokens", 0))
                    ),
                }
            )
            return result.model_copy(
                update={
                    "question": question,
                    "answer": answer,
                    "usage": updated_usage,
                }
            )
        except Exception as exc:
            err = _make_stage_error("answer", exc)
            if self._safe_mode:
                return result.model_copy(
                    update={
                        "question": question,
                        "stage_errors": [*result.stage_errors, err],
                        "error": result.error or str(err),
                    }
                )
            raise RuntimeError(
                f"[Stage: answer] {err.error_type}: {err.message}"
            ) from exc

    async def aanswer_question(
        self,
        source: str | Path | bytes | list,
        *,
        question: str,
        processing_mode: VideoProcessingMode | str = VideoProcessingMode.ACTIVE,
        focus_time: float | int | str | None = None,
        window_seconds: float = 5.0,
        max_context_chars: int = 40_000,
        **run_kwargs: Any,
    ) -> VideoProcessorResult:
        """Async variant of :meth:`answer_question`."""
        if not question.strip():
            raise ValueError("question cannot be empty.")

        mode = VideoProcessingMode(processing_mode)
        run_kwargs.setdefault("analyze_frames", True)
        run_kwargs.setdefault("transcribe_audio", True)
        run_kwargs.setdefault("generate_summary", False)
        run_kwargs["processing_mode"] = mode
        run_kwargs["window_seconds"] = window_seconds
        if focus_time is not None:
            run_kwargs["focus_time_seconds"] = self.parse_timestamp(focus_time)

        result = await self.arun(source, **run_kwargs)
        if result.failed_stage is not None:
            return result.model_copy(update={"question": question})

        context = self._build_qa_context(result, max_context_chars=max_context_chars)
        if not context:
            return result.model_copy(
                update={
                    "question": question,
                    "answer": "No analyzable context was extracted from this video run.",
                }
            )

        try:
            answer, usage = await self._qa_answer_async(question=question, context=context)
            updated_usage = result.usage.model_copy(
                update={
                    "summary_input_tokens": (
                        result.usage.summary_input_tokens
                        + int(usage.get("prompt_tokens", 0))
                    ),
                    "summary_output_tokens": (
                        result.usage.summary_output_tokens
                        + int(usage.get("completion_tokens", 0))
                    ),
                }
            )
            return result.model_copy(
                update={
                    "question": question,
                    "answer": answer,
                    "usage": updated_usage,
                }
            )
        except Exception as exc:
            err = _make_stage_error("answer", exc)
            if self._safe_mode:
                return result.model_copy(
                    update={
                        "question": question,
                        "stage_errors": [*result.stage_errors, err],
                        "error": result.error or str(err),
                    }
                )
            raise RuntimeError(
                f"[Stage: answer] {err.error_type}: {err.message}"
            ) from exc

    async def _arun_pipeline(
        self,
        source: str | Path | bytes | list,
        call_kwargs: dict[str, Any],
    ) -> VideoProcessorResult:
        from ._analyzer import analyze_frames_async  # noqa: PLC0415
        from ._extractor import (  # noqa: PLC0415
            deduplicate_frames_fast,
            extract_frames,
            extract_frames_window,
            load_frames_from_paths,
        )
        from ._loader import resolve_video_source  # noqa: PLC0415
        from ._rag import store_result_in_rag_async  # noqa: PLC0415
        from ._summarizer import build_sections, generate_summary_async  # noqa: PLC0415
        from ._transcriber import align_frames_to_transcript  # noqa: PLC0415

        cfg = self._resolved_config(call_kwargs)
        label = self._source_label(source)
        usage = VideoProcessorUsage()
        stage_errors: list[StageError] = []
        # Python 3.10+: get_running_loop() is safe inside a coroutine; no deprecation.
        loop = asyncio.get_running_loop()
        processing_mode: VideoProcessingMode = cfg["processing_mode"]
        window_start_seconds: float | None = None
        window_end_seconds: float | None = None

        try:
            window_start_seconds, window_end_seconds = self._resolve_window(cfg)
        except Exception as exc:
            err = _make_stage_error("config", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise

        try:
            self._check_rate_limit(cfg["user_id"])
        except Exception as exc:
            err = _make_stage_error("rate_limit", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise

        # ── 1. Resolve source — non-blocking via to_thread ── FATAL ──────────
        try:
            video_path, frame_paths = await asyncio.to_thread(
                resolve_video_source, source
            )
        except Exception as exc:
            err = _make_stage_error("load", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise

        # ── 2. Extract frames — CPU-bound in thread pool ── FATAL ─────────────
        def _extract() -> list:
            if frame_paths is not None:
                if processing_mode == VideoProcessingMode.PASSIVE:
                    raise ValueError(
                        "processing_mode='passive' does not support pre-extracted frame lists. "
                        "Provide a video file/URL/bytes source."
                    )
                return load_frames_from_paths(
                    frame_paths, frame_format=self._frame_format
                )
            if window_start_seconds is not None:
                return extract_frames_window(
                    video_path,  # type: ignore[arg-type]
                    fps=cfg["fps"],
                    max_frames=cfg["max_frames"],
                    frame_format=self._frame_format,
                    start_time_seconds=window_start_seconds,
                    end_time_seconds=window_end_seconds,
                )
            return extract_frames(
                video_path,  # type: ignore[arg-type]
                fps=cfg["fps"],
                max_frames=cfg["max_frames"],
                frame_format=self._frame_format,
                max_process_workers=self._max_process_workers,
            )

        try:
            with ThreadPoolExecutor(max_workers=self._max_process_workers) as pool:
                raw_frames = await loop.run_in_executor(pool, _extract)
        except Exception as exc:
            err = _make_stage_error("extract", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise RuntimeError(f"[Stage: extract] {err.error_type}: {err.message}") from exc

        usage.frames_extracted = len(raw_frames)

        # ── 3. Deduplicate — parallel pHash pre-computation ── FATAL ──────────
        # deduplicate_frames_fast computes all hashes in parallel via
        # ThreadPoolExecutor then does O(1) XOR+popcount comparisons.
        # Falls back to sequential SSIM path when method=SSIM.
        try:
            frames = await asyncio.to_thread(
                deduplicate_frames_fast,
                raw_frames,
                similarity_threshold=cfg["similarity_threshold"],
                method=cfg["dedup_method"],
                max_hash_workers=self._max_process_workers,
            )
        except Exception as exc:
            err = _make_stage_error("deduplicate", exc)
            if self._safe_mode:
                return self._fatal_stage_result(label, usage, stage_errors, err)
            raise RuntimeError(
                f"[Stage: deduplicate] {err.error_type}: {err.message}"
            ) from exc

        usage.frames_kept = sum(1 for f in frames if f.kept)
        usage.frames_discarded = sum(1 for f in frames if not f.kept)

        # ── 4+5. Transcription || Frame analysis — RUN CONCURRENTLY ───────────
        # Transcription (blocking I/O in a thread) and LLM analysis (async
        # network I/O via Semaphore-gated asyncio.gather) are independent after
        # deduplication completes.  Running them in parallel halves wall-clock
        # time for typical lecture videos where each takes 1-10+ minutes.

        async def _transcribe_stage() -> list:
            if not cfg["transcribe_audio"] or video_path is None:
                return []
            return await asyncio.to_thread(
                self._run_transcription,
                video_path,
                cfg["language"],
                usage,
                window_start_seconds,
                window_end_seconds,
                window_start_seconds or 0.0,
            )

        async def _analyze_stage(current_frames: list) -> list:
            if not cfg["analyze_frames"]:
                return current_frames
            return await analyze_frames_async(
                current_frames,
                self._analysis_kit,
                mode=cfg["frame_analysis_mode"],
                batch_size=cfg["batch_size"],
                max_workers=self._max_workers,
                grid_size=cfg["grid_size"],
                usage=usage,
            )

        transcript_result, frames_result = await asyncio.gather(
            _transcribe_stage(),
            _analyze_stage(frames),
            return_exceptions=True,
        )

        # Handle transcription result (NON-FATAL)
        transcript: list = []
        if isinstance(transcript_result, BaseException):
            err = _make_stage_error("transcribe", transcript_result)
            stage_errors.append(err)
            if not self._safe_mode:
                raise RuntimeError(
                    f"[Stage: transcribe] {err.error_type}: {err.message}"
                ) from transcript_result
        else:
            transcript = transcript_result
            transcript = align_frames_to_transcript(frames, transcript)

        # Handle analysis result (NON-FATAL)
        if isinstance(frames_result, BaseException):
            err = _make_stage_error("analyze", frames_result)
            stage_errors.append(err)
            if not self._safe_mode:
                raise RuntimeError(
                    f"[Stage: analyze] {err.error_type}: {err.message}"
                ) from frames_result
        else:
            frames = frames_result

        # ── 6. Build sections — NON-FATAL ─────────────────────────────────────
        sections: list = []
        try:
            sections = build_sections(frames, transcript)
        except Exception as exc:
            err = _make_stage_error("build_sections", exc)
            stage_errors.append(err)
            if not self._safe_mode:
                raise RuntimeError(
                    f"[Stage: build_sections] {err.error_type}: {err.message}"
                ) from exc

        # ── 7. Summary — NON-FATAL ────────────────────────────────────────────
        summary: str | None = None
        if cfg["generate_summary"]:
            try:
                summary = await generate_summary_async(
                    sections, transcript, self._summary_kit, usage
                )
            except Exception as exc:
                err = _make_stage_error("summarize", exc)
                stage_errors.append(err)
                if not self._safe_mode:
                    raise RuntimeError(
                        f"[Stage: summarize] {err.error_type}: {err.message}"
                    ) from exc

        # ── 8. Build result ───────────────────────────────────────────────────
        top_error = str(stage_errors[0]) if stage_errors else None
        result = VideoProcessorResult(
            video_path=label,
            frames=frames,
            transcript=transcript,
            sections=sections,
            summary=summary,
            usage=usage,
            stage_errors=stage_errors,
            error=top_error,
            processing_mode=processing_mode,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
        )

        # ── 9. RAG storage — NON-FATAL ────────────────────────────────────────
        if cfg["store_in_rag"] and self._rag_pipeline is not None:
            try:
                count = await store_result_in_rag_async(result, self._rag_pipeline)
                result = result.model_copy(
                    update={"rag_stored": True, "rag_chunk_count": count}
                )
            except Exception as exc:
                err = _make_stage_error("rag_store", exc)
                if self._safe_mode:
                    result = result.model_copy(
                        update={
                            "stage_errors": [*result.stage_errors, err],
                            "error": result.error or str(err),
                        }
                    )
                else:
                    raise RuntimeError(
                        f"[Stage: rag_store] {err.error_type}: {err.message}"
                    ) from exc

        return result


# ---------------------------------------------------------------------------
# Async-only variant (for FastAPI / async servers)
# ---------------------------------------------------------------------------


class AsyncVideoProcessorPipeline:
    """Async-only variant of :class:`VideoProcessorPipeline`.

    Exposes a single ``async run()`` method — suitable for FastAPI endpoints
    where you do not want a sync ``run()`` in the public API.

    All constructor parameters are identical to :class:`VideoProcessorPipeline`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._inner = VideoProcessorPipeline(*args, **kwargs)

    async def run(
        self,
        source: str | Path | bytes | list,
        **kwargs: Any,
    ) -> VideoProcessorResult:
        """Async-only process entrypoint."""
        return await self._inner.arun(source, **kwargs)

    async def answer_question(
        self,
        source: str | Path | bytes | list,
        *,
        question: str,
        processing_mode: VideoProcessingMode | str = VideoProcessingMode.ACTIVE,
        focus_time: float | int | str | None = None,
        window_seconds: float = 5.0,
        max_context_chars: int = 40_000,
        **run_kwargs: Any,
    ) -> VideoProcessorResult:
        """Async-only variant of :meth:`VideoProcessorPipeline.aanswer_question`."""
        return await self._inner.aanswer_question(
            source,
            question=question,
            processing_mode=processing_mode,
            focus_time=focus_time,
            window_seconds=window_seconds,
            max_context_chars=max_context_chars,
            **run_kwargs,
        )

    @staticmethod
    def parse_timestamp(value: float | int | str) -> float:
        """Delegate to :meth:`VideoProcessorPipeline.parse_timestamp`."""
        return VideoProcessorPipeline.parse_timestamp(value)
