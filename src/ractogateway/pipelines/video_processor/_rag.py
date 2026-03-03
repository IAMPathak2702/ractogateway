"""RactoRAG integration for VideoProcessorPipeline.

Stores extracted video content (visual analyses + transcript segments) into
the existing RactoRAG pipeline for subsequent Q&A retrieval.
"""

from __future__ import annotations

from typing import Any

from ._models import TranscriptSegment, VideoProcessorResult, VideoSection

# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------


def _require_ractorag() -> Any:
    try:
        from ractogateway.rag.pipeline import RactoRAG
        return RactoRAG
    except ImportError as exc:
        raise ImportError(
            "RactoRAG is required for RAG storage. "
            "Install with: pip install ractogateway[rag]"
        ) from exc


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def _section_to_text(section: VideoSection, video_path: str) -> str:
    """Convert a VideoSection into a plain-text document for RAG ingestion."""
    parts: list[str] = [
        f"[Video: {video_path}]",
        f"[Time: {section.timestamp_start:.1f}s - {section.timestamp_end:.1f}s]",
    ]
    if section.visual_content:
        parts.append(f"Visual content:\n{section.visual_content}")
    if section.audio_content:
        parts.append(f"Audio / transcript:\n{section.audio_content}")
    return "\n\n".join(parts)


def _transcript_to_text(seg: TranscriptSegment, video_path: str) -> str:
    """Convert a single TranscriptSegment to a plain-text RAG document."""
    return (
        f"[Video: {video_path}]\n"
        f"[Time: {seg.start:.1f}s - {seg.end:.1f}s]\n"
        f"Transcript: {seg.text}"
    )


# ---------------------------------------------------------------------------
# Storage functions
# ---------------------------------------------------------------------------


def store_result_in_rag(
    result: VideoProcessorResult,
    rag_pipeline: Any,
) -> int:
    """Ingest all video sections and transcript segments into *rag_pipeline*.

    Parameters
    ----------
    result:
        The completed :class:`VideoProcessorResult` to store.
    rag_pipeline:
        A :class:`ractogateway.rag.pipeline.RactoRAG` instance (or any object
        with a compatible ``add_text(text, metadata)`` interface).

    Returns
    -------
    int
        Number of chunks successfully stored.
    """
    _require_ractorag()  # import guard / helpful error

    chunks_stored = 0
    video_path = result.video_path

    # Store merged sections (visual + audio) — best for Q&A
    for section in result.sections:
        text = _section_to_text(section, video_path)
        metadata = {
            "source": video_path,
            "type": "video_section",
            "timestamp_start": section.timestamp_start,
            "timestamp_end": section.timestamp_end,
            "frame_ids": section.frame_ids,
        }
        try:
            rag_pipeline.add_text(text, metadata=metadata)
            chunks_stored += 1
        except AttributeError:
            # Fallback: try add_document interface
            rag_pipeline.add_document(text, metadata=metadata)
            chunks_stored += 1

    # Also store raw transcript segments for audio-only retrieval
    for seg in result.transcript:
        # Skip if already covered by a section
        if any(
            sec.timestamp_start <= seg.start and sec.timestamp_end >= seg.end
            for sec in result.sections
        ):
            continue
        text = _transcript_to_text(seg, video_path)
        metadata = {
            "source": video_path,
            "type": "transcript_segment",
            "timestamp_start": seg.start,
            "timestamp_end": seg.end,
        }
        try:
            rag_pipeline.add_text(text, metadata=metadata)
            chunks_stored += 1
        except AttributeError:
            rag_pipeline.add_document(text, metadata=metadata)
            chunks_stored += 1

    # Store summary as a single high-level document
    if result.summary:
        summary_text = (
            f"[Video: {video_path}]\n"
            f"[Type: summary]\n\n"
            f"{result.summary}"
        )
        metadata = {
            "source": video_path,
            "type": "video_summary",
        }
        try:
            rag_pipeline.add_text(summary_text, metadata=metadata)
            chunks_stored += 1
        except AttributeError:
            rag_pipeline.add_document(summary_text, metadata=metadata)
            chunks_stored += 1

    return chunks_stored


async def store_result_in_rag_async(
    result: VideoProcessorResult,
    rag_pipeline: Any,
) -> int:
    """Async variant of :func:`store_result_in_rag`.

    Falls back to the sync implementation if the rag_pipeline does not
    expose async add methods.
    """
    # Check for async interface
    if hasattr(rag_pipeline, "aadd_text") or hasattr(rag_pipeline, "aadd_document"):
        chunks_stored = 0
        video_path = result.video_path

        for section in result.sections:
            text = _section_to_text(section, video_path)
            metadata = {
                "source": video_path,
                "type": "video_section",
                "timestamp_start": section.timestamp_start,
                "timestamp_end": section.timestamp_end,
                "frame_ids": section.frame_ids,
            }
            add_fn = getattr(rag_pipeline, "aadd_text", None) or rag_pipeline.aadd_document
            await add_fn(text, metadata=metadata)
            chunks_stored += 1

        for seg in result.transcript:
            if any(
                sec.timestamp_start <= seg.start and sec.timestamp_end >= seg.end
                for sec in result.sections
            ):
                continue
            text = _transcript_to_text(seg, video_path)
            metadata = {
                "source": video_path,
                "type": "transcript_segment",
                "timestamp_start": seg.start,
                "timestamp_end": seg.end,
            }
            add_fn = getattr(rag_pipeline, "aadd_text", None) or rag_pipeline.aadd_document
            await add_fn(text, metadata=metadata)
            chunks_stored += 1

        if result.summary:
            summary_text = (
                f"[Video: {video_path}]\n[Type: summary]\n\n{result.summary}"
            )
            add_fn = getattr(rag_pipeline, "aadd_text", None) or rag_pipeline.aadd_document
            await add_fn(summary_text, metadata={"source": video_path, "type": "video_summary"})
            chunks_stored += 1

        return chunks_stored

    # Sync fallback
    return store_result_in_rag(result, rag_pipeline)
