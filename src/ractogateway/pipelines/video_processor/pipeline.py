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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ._models import (
    DeduplicationMethod,
    FrameAnalysisMode,
    TranscriberBackend,
    VideoProcessorResult,
    VideoProcessorUsage,
    VideoRateLimitExceededError,
)

# Sentinel for "not supplied by caller"
_UNSET = object()


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
        store_in_rag: bool = False,
        user_id: Any = _UNSET,
    ) -> VideoProcessorResult:
        """Process *source* and return a :class:`VideoProcessorResult`.

        All keyword arguments override the constructor defaults for this call only.
        """
        if self._safe_mode:
            try:
                return self._run_pipeline(source, locals())
            except Exception as exc:  # noqa: BLE001
                source_label = (
                    str(source)
                    if not isinstance(source, (bytes, bytearray))
                    else "<bytes>"
                )
                return VideoProcessorResult(
                    video_path=source_label,
                    error=f"{type(exc).__name__}: {exc}",
                )
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
        store_in_rag: bool = False,
        user_id: Any = _UNSET,
    ) -> VideoProcessorResult:
        """Async variant of :meth:`run`."""
        if self._safe_mode:
            try:
                return await self._arun_pipeline(source, locals())
            except Exception as exc:  # noqa: BLE001
                source_label = (
                    str(source)
                    if not isinstance(source, (bytes, bytearray))
                    else "<bytes>"
                )
                return VideoProcessorResult(
                    video_path=source_label,
                    error=f"{type(exc).__name__}: {exc}",
                )
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

    # ── Sync pipeline execution ──────────────────────────────────────────────

    def _run_pipeline(
        self,
        source: str | Path | bytes | list,
        call_kwargs: dict[str, Any],
    ) -> VideoProcessorResult:
        from ._analyzer import analyze_frames_sync  # noqa: PLC0415
        from ._extractor import (  # noqa: PLC0415
            deduplicate_frames,
            extract_frames,
            load_frames_from_paths,
        )
        from ._loader import resolve_video_source  # noqa: PLC0415
        from ._rag import store_result_in_rag  # noqa: PLC0415
        from ._summarizer import build_sections, generate_summary_sync  # noqa: PLC0415
        from ._transcriber import align_frames_to_transcript  # noqa: PLC0415

        cfg = self._resolved_config(call_kwargs)
        label = self._source_label(source)
        usage = VideoProcessorUsage()

        self._check_rate_limit(cfg["user_id"])

        # ── 1. Resolve source ────────────────────────────────────────────────
        video_path, frame_paths = resolve_video_source(source)

        # ── 2. Extract frames ────────────────────────────────────────────────
        if frame_paths is not None:
            # Pre-extracted frames — load from disk
            raw_frames = load_frames_from_paths(
                frame_paths, frame_format=self._frame_format
            )
        else:
            raw_frames = extract_frames(
                video_path,  # type: ignore[arg-type]
                fps=cfg["fps"],
                max_frames=cfg["max_frames"],
                frame_format=self._frame_format,
                max_process_workers=self._max_process_workers,
            )

        usage.frames_extracted = len(raw_frames)

        # ── 3. Deduplicate ───────────────────────────────────────────────────
        frames = deduplicate_frames(
            raw_frames,
            similarity_threshold=cfg["similarity_threshold"],
            method=cfg["dedup_method"],
        )
        usage.frames_kept = sum(1 for f in frames if f.kept)
        usage.frames_discarded = sum(1 for f in frames if not f.kept)

        # ── 4. Transcription ─────────────────────────────────────────────────
        transcript = []
        if cfg["transcribe_audio"] and video_path is not None:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    self._run_transcription,
                    video_path,
                    cfg["language"],
                    usage,
                )
                transcript = fut.result()
            transcript = align_frames_to_transcript(frames, transcript)

        # ── 5. Frame analysis ────────────────────────────────────────────────
        if cfg["analyze_frames"]:
            frames = analyze_frames_sync(
                frames,
                self._analysis_kit,
                mode=cfg["frame_analysis_mode"],
                batch_size=cfg["batch_size"],
                max_workers=self._max_workers,
                grid_size=cfg["grid_size"],
                usage=usage,
            )

        # ── 6. Build sections ────────────────────────────────────────────────
        sections = build_sections(frames, transcript)

        # ── 7. Summary ───────────────────────────────────────────────────────
        summary: str | None = None
        if cfg["generate_summary"]:
            summary = generate_summary_sync(
                sections, transcript, self._summary_kit, usage
            )

        # ── 8. Build result ──────────────────────────────────────────────────
        result = VideoProcessorResult(
            video_path=label,
            frames=frames,
            transcript=transcript,
            sections=sections,
            summary=summary,
            usage=usage,
        )

        # ── 9. RAG storage ────────────────────────────────────────────────────
        if cfg["store_in_rag"] and self._rag_pipeline is not None:
            count = store_result_in_rag(result, self._rag_pipeline)
            result = result.model_copy(
                update={"rag_stored": True, "rag_chunk_count": count}
            )

        return result

    def _run_transcription(
        self,
        video_path: Path,
        language: str | None,
        usage: VideoProcessorUsage,
    ) -> list:
        """Run audio extraction + transcription in a thread."""
        from ._transcriber import extract_audio, get_audio_duration  # noqa: PLC0415

        audio_path = extract_audio(video_path)
        with contextlib.suppress(Exception):
            usage.audio_duration_seconds = get_audio_duration(audio_path)
        return self._transcriber_obj.transcribe(audio_path, language)

    # ── Async pipeline execution ─────────────────────────────────────────────

    async def _arun_pipeline(
        self,
        source: str | Path | bytes | list,
        call_kwargs: dict[str, Any],
    ) -> VideoProcessorResult:
        from ._analyzer import analyze_frames_async  # noqa: PLC0415
        from ._extractor import (  # noqa: PLC0415
            deduplicate_frames,
            extract_frames,
            load_frames_from_paths,
        )
        from ._loader import resolve_video_source  # noqa: PLC0415
        from ._rag import store_result_in_rag_async  # noqa: PLC0415
        from ._summarizer import build_sections, generate_summary_async  # noqa: PLC0415
        from ._transcriber import align_frames_to_transcript  # noqa: PLC0415

        cfg = self._resolved_config(call_kwargs)
        label = self._source_label(source)
        usage = VideoProcessorUsage()
        loop = asyncio.get_event_loop()

        self._check_rate_limit(cfg["user_id"])

        # ── 1. Resolve source (potentially downloads) — in thread ────────────
        video_path, frame_paths = await loop.run_in_executor(
            None, resolve_video_source, source
        )

        # ── 2 & 3. Extract + deduplicate — CPU-bound, run in process pool ─────
        def _extract_and_dedup() -> list:
            if frame_paths is not None:
                raw = load_frames_from_paths(
                    frame_paths, frame_format=self._frame_format
                )
            else:
                raw = extract_frames(
                    video_path,  # type: ignore[arg-type]
                    fps=cfg["fps"],
                    max_frames=cfg["max_frames"],
                    frame_format=self._frame_format,
                    max_process_workers=self._max_process_workers,
                )
            return raw

        with ThreadPoolExecutor(max_workers=self._max_process_workers) as pool:
            raw_frames = await loop.run_in_executor(pool, _extract_and_dedup)

        usage.frames_extracted = len(raw_frames)

        frames = await loop.run_in_executor(
            None,
            lambda: deduplicate_frames(
                raw_frames,
                similarity_threshold=cfg["similarity_threshold"],
                method=cfg["dedup_method"],
            ),
        )
        usage.frames_kept = sum(1 for f in frames if f.kept)
        usage.frames_discarded = sum(1 for f in frames if not f.kept)

        # ── 4. Transcription — blocking, run in thread ────────────────────────
        transcript = []
        if cfg["transcribe_audio"] and video_path is not None:
            transcript = await loop.run_in_executor(
                None,
                self._run_transcription,
                video_path,
                cfg["language"],
                usage,
            )
            transcript = align_frames_to_transcript(frames, transcript)

        # ── 5. Frame analysis — async gather ─────────────────────────────────
        if cfg["analyze_frames"]:
            frames = await analyze_frames_async(
                frames,
                self._analysis_kit,
                mode=cfg["frame_analysis_mode"],
                batch_size=cfg["batch_size"],
                max_workers=self._max_workers,
                grid_size=cfg["grid_size"],
                usage=usage,
            )

        # ── 6. Build sections ─────────────────────────────────────────────────
        sections = build_sections(frames, transcript)

        # ── 7. Summary ────────────────────────────────────────────────────────
        summary: str | None = None
        if cfg["generate_summary"]:
            summary = await generate_summary_async(
                sections, transcript, self._summary_kit, usage
            )

        # ── 8. Build result ───────────────────────────────────────────────────
        result = VideoProcessorResult(
            video_path=label,
            frames=frames,
            transcript=transcript,
            sections=sections,
            summary=summary,
            usage=usage,
        )

        # ── 9. RAG storage ────────────────────────────────────────────────────
        if cfg["store_in_rag"] and self._rag_pipeline is not None:
            count = await store_result_in_rag_async(result, self._rag_pipeline)
            result = result.model_copy(
                update={"rag_stored": True, "rag_chunk_count": count}
            )

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
