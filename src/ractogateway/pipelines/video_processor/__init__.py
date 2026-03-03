"""VideoProcessorPipeline — process tutorial/lecture videos for RAG & Q&A.

Quick start::

    from ractogateway.openai_developer_kit import Chat
    from ractogateway.pipelines.video_processor import (
        VideoProcessorPipeline,
        TranscriberBackend,
        DeduplicationMethod,
    )

    kit = Chat(api_key="sk-...", model="gpt-4o")

    pipeline = VideoProcessorPipeline(
        kit=kit,
        fps=1.0,
        similarity_threshold=85.0,
        transcriber=TranscriberBackend.FASTER_WHISPER,
        transcriber_model="base",
        analyze_frames=True,
        generate_summary=True,
    )

    # Accepts: local path, URL, YouTube link, bytes buffer, or pre-extracted frames
    result = pipeline.run("lecture.mp4")
    print(result.summary)
    result.to_markdown("report.md")

Install::

    pip install ractogateway[pipelines-video]           # core (OpenCV, pHash, ffmpeg)
    pip install ractogateway[pipelines-video-whisper]   # + faster-whisper
    pip install ractogateway[pipelines-video-full]      # all of the above
    pip install ractogateway[pipelines-video-yt]        # + yt-dlp (YouTube support)
"""

from ._models import (
    DeduplicationMethod,
    FrameAnalysisMode,
    FrameEntry,
    TranscriberBackend,
    TranscriptSegment,
    VideoConfig,
    VideoInput,
    VideoProcessorResult,
    VideoProcessorUsage,
    VideoRateLimitExceededError,
    VideoSection,
)
from .pipeline import AsyncVideoProcessorPipeline, VideoProcessorPipeline

__all__ = [
    # Pipeline classes
    "AsyncVideoProcessorPipeline",
    "VideoProcessorPipeline",
    # Config & enums
    "DeduplicationMethod",
    "FrameAnalysisMode",
    "TranscriberBackend",
    "VideoConfig",
    "VideoInput",
    # Data models
    "FrameEntry",
    "TranscriptSegment",
    "VideoSection",
    # Result & usage
    "VideoProcessorResult",
    "VideoProcessorUsage",
    # Exceptions
    "VideoRateLimitExceededError",
]
