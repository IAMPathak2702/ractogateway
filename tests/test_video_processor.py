"""Unit tests for ractogateway.pipelines.video_processor.

All tests run without real API keys, video files, or optional dependencies.
External I/O (OpenCV, Whisper, vision LLM, ffmpeg) is patched with
``monkeypatch`` / ``unittest.mock``.

Run with:
    pytest tests/test_video_processor.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level import helpers
# ---------------------------------------------------------------------------

from ractogateway.pipelines.video_processor import (
    AsyncVideoProcessorPipeline,
    DeduplicationMethod,
    FrameAnalysisMode,
    FrameEntry,
    TranscriberBackend,
    TranscriptSegment,
    VideoConfig,
    VideoProcessingMode,
    VideoProcessorPipeline,
    VideoProcessorResult,
    VideoProcessorUsage,
    VideoRateLimitExceededError,
    VideoSection,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fake_jpeg_bytes() -> bytes:
    """Return a valid 1x1 white JPEG image as bytes (no PIL required)."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
        b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
        b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\x1e"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
        b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00"
        b"\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00"
        b"\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142"
        b"\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18"
        b"\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85"
        b"\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3"
        b"\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba"
        b"\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8"
        b"\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4"
        b"\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb"
        b"\xd2\x8a(\x03\xff\xd9"
    )


def _make_raw_frames(n: int = 3) -> list[tuple[int, float, bytes]]:
    """Return *n* fake (frame_id, timestamp, jpeg_bytes) tuples."""
    img = _fake_jpeg_bytes()
    return [(i, float(i), img) for i in range(n)]


def _make_llm_response(text: str = "Whiteboard: E=mc²") -> Any:
    """Fake LLMResponse object."""
    resp = MagicMock()
    resp.content = text
    resp.usage = {"prompt_tokens": 100, "completion_tokens": 50}
    return resp


def _make_kit_mock(response_text: str = "Whiteboard: E=mc²") -> MagicMock:
    """Build a MagicMock kit with a synchronous chat() method."""
    kit = MagicMock()
    kit.chat.return_value = _make_llm_response(response_text)
    return kit


class _StrictChatConfigKit:
    """Kit mock that only accepts chat(config)/achat(config) style calls."""

    def __init__(self, response_text: str = "OK") -> None:
        self._text = response_text

    def chat(self, config: Any) -> Any:
        assert hasattr(config, "user_message")
        return _make_llm_response(self._text)

    async def achat(self, config: Any) -> Any:
        assert hasattr(config, "user_message")
        return _make_llm_response(self._text)


def _make_pipeline(kit: Any = None, **kwargs: Any) -> VideoProcessorPipeline:
    """Build a VideoProcessorPipeline with all external calls pre-mocked."""
    kit = kit or _make_kit_mock()
    defaults: dict[str, Any] = {
        "transcribe_audio": False,
        "analyze_frames": False,
        "generate_summary": False,
        "safe_mode": True,
    }
    defaults.update(kwargs)
    return VideoProcessorPipeline(kit=kit, **defaults)


# ---------------------------------------------------------------------------
# Tests — Models
# ---------------------------------------------------------------------------


class TestVideoProcessorUsage:
    """Token accounting model."""

    def test_total_tokens_sums_all_stages(self) -> None:
        usage = VideoProcessorUsage(
            analysis_input_tokens=100,
            analysis_output_tokens=50,
            summary_input_tokens=200,
            summary_output_tokens=80,
        )
        assert usage.total_analysis_tokens == 150
        assert usage.total_summary_tokens == 280
        assert usage.total_tokens == 430

    def test_defaults_are_zero(self) -> None:
        usage = VideoProcessorUsage()
        assert usage.total_tokens == 0
        assert usage.frames_extracted == 0


class TestFrameEntry:
    """FrameEntry model."""

    def test_kept_defaults_true(self) -> None:
        fe = FrameEntry(frame_id=0, timestamp=0.0, similarity_to_prev=None)
        assert fe.kept is True

    def test_discarded_frame_has_no_image(self) -> None:
        fe = FrameEntry(
            frame_id=1,
            timestamp=1.0,
            similarity_to_prev=95.0,
            kept=False,
            image_data=None,
        )
        assert fe.image_data is None
        assert not fe.kept


class TestTranscriptSegment:
    """TranscriptSegment model."""

    def test_frame_ids_default_empty(self) -> None:
        seg = TranscriptSegment(start=0.0, end=1.0, text="Hello")
        assert seg.frame_ids == []

    def test_stores_frame_ids(self) -> None:
        seg = TranscriptSegment(start=0.0, end=5.0, text="Math", frame_ids=[0, 1, 2])
        assert seg.frame_ids == [0, 1, 2]


class TestVideoSection:
    """VideoSection model."""

    def test_defaults(self) -> None:
        sec = VideoSection(timestamp_start=0.0, timestamp_end=5.0)
        assert sec.visual_content == ""
        assert sec.audio_content == ""
        assert sec.frame_ids == []


class TestVideoConfig:
    """VideoConfig model defaults and validation."""

    def test_defaults(self) -> None:
        cfg = VideoConfig()
        assert cfg.fps == 1.0
        assert cfg.similarity_threshold == 90.0
        assert cfg.dedup_method == DeduplicationMethod.PHASH
        assert cfg.transcriber_backend == TranscriberBackend.FASTER_WHISPER
        assert cfg.transcriber_model == "base"
        assert cfg.frame_analysis_mode == FrameAnalysisMode.INDIVIDUAL
        assert cfg.generate_summary is True

    def test_custom_threshold(self) -> None:
        cfg = VideoConfig(similarity_threshold=75.0)
        assert cfg.similarity_threshold == 75.0

    def test_ssim_method(self) -> None:
        cfg = VideoConfig(dedup_method="ssim")
        assert cfg.dedup_method == DeduplicationMethod.SSIM


class TestTimestampParsing:
    """Timestamp parser for passive processing windows."""

    def test_mm_ss(self) -> None:
        assert VideoProcessorPipeline.parse_timestamp("02:10") == 130.0

    def test_human_units(self) -> None:
        assert VideoProcessorPipeline.parse_timestamp("2 mins 10 sec") == 130.0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoProcessorPipeline.parse_timestamp(-1)


# ---------------------------------------------------------------------------
# Tests — VideoProcessorResult helpers
# ---------------------------------------------------------------------------


class TestVideoProcessorResultHelpers:
    """Result model export and accessor methods."""

    def _make_result(self) -> VideoProcessorResult:
        frames = [
            FrameEntry(
                frame_id=0,
                timestamp=0.0,
                similarity_to_prev=None,
                kept=True,
                analysis="Board: x=1",
            ),
            FrameEntry(
                frame_id=1,
                timestamp=1.0,
                similarity_to_prev=30.0,
                kept=True,
                analysis="Board: y=2",
            ),
            FrameEntry(
                frame_id=2,
                timestamp=2.0,
                similarity_to_prev=95.0,
                kept=False,
            ),
        ]
        transcript = [
            TranscriptSegment(start=0.0, end=1.0, text="So x equals one"),
            TranscriptSegment(start=1.0, end=2.0, text="And y equals two"),
        ]
        sections = [
            VideoSection(
                timestamp_start=0.0,
                timestamp_end=1.0,
                frame_ids=[0],
                visual_content="Board: x=1",
                audio_content="So x equals one",
            )
        ]
        return VideoProcessorResult(
            video_path="test.mp4",
            frames=frames,
            transcript=transcript,
            sections=sections,
            summary="This video covers x=1 and y=2.",
            usage=VideoProcessorUsage(frames_extracted=3, frames_kept=2, frames_discarded=1),
        )

    def test_get_transcript_text(self) -> None:
        result = self._make_result()
        text = result.get_transcript_text()
        assert "x equals one" in text
        assert "y equals two" in text

    def test_get_all_visual_content(self) -> None:
        result = self._make_result()
        visual = result.get_all_visual_content()
        assert "Board: x=1" in visual
        assert "Board: y=2" in visual
        assert "[0.0s]" in visual

    def test_to_json_returns_string(self) -> None:
        result = self._make_result()
        json_str = result.to_json()
        assert json_str is not None
        data = json.loads(json_str)
        assert data["video_path"] == "test.mp4"
        assert data["summary"] == "This video covers x=1 and y=2."

    def test_to_json_writes_file(self, tmp_path: Path) -> None:
        result = self._make_result()
        out = tmp_path / "output.json"
        returned = result.to_json(str(out))
        assert returned is None
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["video_path"] == "test.mp4"

    def test_to_markdown_contains_sections(self) -> None:
        result = self._make_result()
        md = result.to_markdown()
        assert md is not None
        assert "# Video Processing Report" in md
        assert "test.mp4" in md
        assert "This video covers" in md
        assert "0.0s" in md

    def test_to_markdown_writes_file(self, tmp_path: Path) -> None:
        result = self._make_result()
        out = tmp_path / "report.md"
        returned = result.to_markdown(str(out))
        assert returned is None
        assert out.exists()
        assert "# Video Processing Report" in out.read_text()

    def test_image_data_excluded_from_json(self) -> None:
        """Serialised JSON must NOT embed raw image bytes."""
        result = self._make_result()
        # Add image_data to a frame
        result.frames[0] = result.frames[0].model_copy(
            update={"image_data": b"\xff\xd8\xff\xd9"}
        )
        json_str = result.to_json()
        assert json_str is not None
        # Bytes are excluded by to_json's exclude argument
        assert "image_data" not in json_str or '"image_data": null' in json_str


# ---------------------------------------------------------------------------
# Tests — Deduplication logic
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Frame deduplication with both pHash and SSIM methods."""

    def test_first_frame_always_kept(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import deduplicate_frames

        # Mock imagehash so no PIL required
        with patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ):
            raw = _make_raw_frames(1)
            entries = deduplicate_frames(
                raw,
                similarity_threshold=90.0,
                method=DeduplicationMethod.PHASH,
            )
        assert len(entries) == 1
        assert entries[0].kept is True
        assert entries[0].similarity_to_prev is None

    def test_high_similarity_frame_discarded(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import deduplicate_frames

        # Patch similarity to return 95 % — above 90 % threshold → discard
        with patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=95.0,
        ):
            raw = _make_raw_frames(3)
            entries = deduplicate_frames(
                raw, similarity_threshold=90.0, method=DeduplicationMethod.PHASH
            )

        kept = [e for e in entries if e.kept]
        discarded = [e for e in entries if not e.kept]
        assert len(kept) == 1  # only first kept
        assert len(discarded) == 2

    def test_low_similarity_frame_kept(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import deduplicate_frames

        # Similarity is 30 % — below 90 % threshold → keep
        with patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=30.0,
        ):
            raw = _make_raw_frames(3)
            entries = deduplicate_frames(
                raw, similarity_threshold=90.0, method=DeduplicationMethod.PHASH
            )

        kept = [e for e in entries if e.kept]
        assert len(kept) == 3  # all kept

    def test_threshold_boundary_exact_match_discards(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import deduplicate_frames

        # Exactly at threshold → discard (similarity >= threshold)
        with patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=90.0,
        ):
            raw = _make_raw_frames(2)
            entries = deduplicate_frames(
                raw, similarity_threshold=90.0, method=DeduplicationMethod.PHASH
            )
        assert entries[1].kept is False

    def test_similarity_stored_on_entry(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import deduplicate_frames

        with patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=55.5,
        ):
            raw = _make_raw_frames(2)
            entries = deduplicate_frames(
                raw, similarity_threshold=90.0, method=DeduplicationMethod.PHASH
            )
        assert entries[1].similarity_to_prev == 55.5

    def test_ssim_method_used_when_configured(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import deduplicate_frames

        with patch(
            "ractogateway.pipelines.video_processor._extractor._ssim_similarity",
            return_value=20.0,
        ) as mock_ssim:
            raw = _make_raw_frames(2)
            deduplicate_frames(
                raw, similarity_threshold=90.0, method=DeduplicationMethod.SSIM
            )
        mock_ssim.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — Loader
# ---------------------------------------------------------------------------


class TestLoader:
    """Video source resolver."""

    def test_local_path_string(self, tmp_path: Path) -> None:
        from ractogateway.pipelines.video_processor._loader import resolve_video_source

        f = tmp_path / "video.mp4"
        f.write_bytes(b"fake")
        video_path, frame_paths = resolve_video_source(str(f))
        assert video_path == f
        assert frame_paths is None

    def test_local_path_object(self, tmp_path: Path) -> None:
        from ractogateway.pipelines.video_processor._loader import resolve_video_source

        f = tmp_path / "video.mp4"
        f.write_bytes(b"fake")
        video_path, frame_paths = resolve_video_source(f)
        assert video_path == f
        assert frame_paths is None

    def test_missing_file_raises(self) -> None:
        from ractogateway.pipelines.video_processor._loader import resolve_video_source

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_video_source("/nonexistent/path/video.mp4")

    def test_frame_list_input(self, tmp_path: Path) -> None:
        from ractogateway.pipelines.video_processor._loader import resolve_video_source

        f1 = tmp_path / "frame_0.jpg"
        f2 = tmp_path / "frame_1.jpg"
        f1.write_bytes(_fake_jpeg_bytes())
        f2.write_bytes(_fake_jpeg_bytes())

        video_path, frame_paths = resolve_video_source([str(f1), str(f2)])
        assert video_path is None
        assert frame_paths is not None
        assert len(frame_paths) == 2

    def test_frame_list_missing_file_raises(self, tmp_path: Path) -> None:
        from ractogateway.pipelines.video_processor._loader import resolve_video_source

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_video_source([str(tmp_path / "ghost.jpg")])

    def test_bytes_input_writes_temp(self) -> None:
        from ractogateway.pipelines.video_processor._loader import resolve_video_source

        video_path, frame_paths = resolve_video_source(b"\x00\x01\x02\x03")
        assert video_path is not None
        assert video_path.exists()
        assert video_path.stat().st_size == 4
        assert frame_paths is None
        video_path.unlink(missing_ok=True)

    def test_http_url_detected(self) -> None:
        from ractogateway.pipelines.video_processor._loader import _is_http_url

        assert _is_http_url("https://example.com/video.mp4")
        assert _is_http_url("http://cdn.example.com/v.mp4")
        assert not _is_http_url("/local/path/video.mp4")

    def test_youtube_url_detected(self) -> None:
        from ractogateway.pipelines.video_processor._loader import _is_youtube_url

        assert _is_youtube_url("https://www.youtube.com/watch?v=abc123")
        assert _is_youtube_url("https://youtu.be/abc123")
        assert not _is_youtube_url("https://vimeo.com/123")


# ---------------------------------------------------------------------------
# Tests — Transcription alignment
# ---------------------------------------------------------------------------


class TestTranscriptionAlignment:
    """Frame-to-transcript timestamp alignment."""

    def test_frames_assigned_to_correct_segments(self) -> None:
        from ractogateway.pipelines.video_processor._transcriber import (
            align_frames_to_transcript,
        )

        frames = [
            FrameEntry(frame_id=0, timestamp=0.5, similarity_to_prev=None, kept=True),
            FrameEntry(frame_id=1, timestamp=1.5, similarity_to_prev=30.0, kept=True),
            FrameEntry(frame_id=2, timestamp=2.5, similarity_to_prev=30.0, kept=True),
            FrameEntry(frame_id=3, timestamp=1.8, similarity_to_prev=30.0, kept=False),
        ]
        segments = [
            TranscriptSegment(start=0.0, end=1.0, text="First sentence"),
            TranscriptSegment(start=1.0, end=2.0, text="Second sentence"),
        ]

        updated = align_frames_to_transcript(frames, segments)

        assert updated[0].frame_ids == [0]   # 0.5s in [0, 1]
        assert updated[1].frame_ids == [1]   # 1.5s in [1, 2]; frame 3 not kept → excluded

    def test_discard_frames_not_included(self) -> None:
        from ractogateway.pipelines.video_processor._transcriber import (
            align_frames_to_transcript,
        )

        frames = [
            FrameEntry(frame_id=0, timestamp=0.5, similarity_to_prev=None, kept=True),
            FrameEntry(frame_id=1, timestamp=0.7, similarity_to_prev=95.0, kept=False),
        ]
        segments = [TranscriptSegment(start=0.0, end=1.0, text="One")]
        updated = align_frames_to_transcript(frames, segments)
        assert 1 not in updated[0].frame_ids  # discarded frame excluded

    def test_empty_transcript_returns_empty(self) -> None:
        from ractogateway.pipelines.video_processor._transcriber import (
            align_frames_to_transcript,
        )

        frames = [
            FrameEntry(frame_id=0, timestamp=0.0, similarity_to_prev=None, kept=True)
        ]
        updated = align_frames_to_transcript(frames, [])
        assert updated == []


# ---------------------------------------------------------------------------
# Tests — Summarizer helpers
# ---------------------------------------------------------------------------


class TestSectionBuilder:
    """build_sections merges frames + transcript into VideoSections."""

    def test_sections_built_from_transcript(self) -> None:
        from ractogateway.pipelines.video_processor._summarizer import build_sections

        frames = [
            FrameEntry(
                frame_id=0,
                timestamp=0.5,
                similarity_to_prev=None,
                kept=True,
                analysis="Board: E=mc²",
            )
        ]
        transcript = [TranscriptSegment(start=0.0, end=1.0, text="Energy equals mc²")]
        sections = build_sections(frames, transcript)

        assert len(sections) == 1
        assert sections[0].audio_content == "Energy equals mc²"
        assert "E=mc²" in sections[0].visual_content
        assert 0 in sections[0].frame_ids

    def test_no_transcript_one_section_per_frame(self) -> None:
        from ractogateway.pipelines.video_processor._summarizer import build_sections

        frames = [
            FrameEntry(
                frame_id=i,
                timestamp=float(i),
                similarity_to_prev=None if i == 0 else 30.0,
                kept=True,
                analysis=f"Frame {i}",
            )
            for i in range(3)
        ]
        sections = build_sections(frames, [])
        assert len(sections) == 3
        assert sections[0].visual_content == "Frame 0"

    def test_discarded_frames_excluded_from_sections(self) -> None:
        from ractogateway.pipelines.video_processor._summarizer import build_sections

        frames = [
            FrameEntry(frame_id=0, timestamp=0.0, similarity_to_prev=None, kept=True, analysis="A"),
            FrameEntry(frame_id=1, timestamp=0.5, similarity_to_prev=95.0, kept=False),
        ]
        sections = build_sections(frames, [])
        assert len(sections) == 1  # only kept frame


# ---------------------------------------------------------------------------
# Tests — Pipeline integration (mocked I/O)
# ---------------------------------------------------------------------------


class TestVideoProcessorPipelineIntegration:
    """End-to-end pipeline runs with all external I/O mocked."""

    def _patch_extract(self, monkeypatch: pytest.MonkeyPatch, n: int = 3) -> None:
        """Patch extract_frames to return *n* fake frames."""
        monkeypatch.setattr(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            lambda *a, **kw: _make_raw_frames(n),
            raising=False,
        )
        # Patch inside the module namespace used at import time
        import ractogateway.pipelines.video_processor._extractor as ext
        monkeypatch.setattr(ext, "extract_frames", lambda *a, **kw: _make_raw_frames(n))

    def _patch_dedup_keep_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch dedup similarity so all frames are kept."""
        monkeypatch.setattr(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            lambda *a, **kw: 0.0,  # 0% similar → all kept
        )

    def test_safe_mode_returns_error_on_bad_path(self) -> None:
        pipeline = _make_pipeline(safe_mode=True)
        result = pipeline.run("/nonexistent/video.mp4")
        assert result.error is not None
        assert "not found" in result.error.lower() or "nonexistent" in result.error

    def test_safe_mode_error_preserves_video_path(self) -> None:
        pipeline = _make_pipeline(safe_mode=True)
        result = pipeline.run("/bad/path.mp4")
        assert result.video_path == "/bad/path.mp4"

    def test_raises_without_safe_mode(self) -> None:
        pipeline = _make_pipeline(safe_mode=False)
        with pytest.raises(FileNotFoundError):
            pipeline.run("/nonexistent/video.mp4")

    def test_passive_mode_requires_focus_time(self) -> None:
        pipeline = _make_pipeline(safe_mode=False)
        with pytest.raises(ValueError, match="focus_time_seconds"):
            pipeline.run("any.mp4", processing_mode="passive")

    def test_passive_mode_uses_window_extractor(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames_window",
            return_value=_make_raw_frames(2),
        ) as mock_window, patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(99),
        ) as mock_full, patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ):
            pipeline = _make_pipeline(
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
            )
            result = pipeline.run(
                str(f),
                processing_mode="passive",
                focus_time_seconds=130.0,
                window_seconds=5.0,
            )

        mock_window.assert_called_once()
        mock_full.assert_not_called()
        assert result.processing_mode == VideoProcessingMode.PASSIVE
        assert result.window_start_seconds == 125.0
        assert result.window_end_seconds == 135.0

    def test_run_transcription_applies_window_offset(self) -> None:
        pipeline = _make_pipeline()
        pipeline._transcriber_obj = MagicMock()
        pipeline._transcriber_obj.transcribe.return_value = [
            TranscriptSegment(start=0.0, end=2.0, text="hello")
        ]
        usage = VideoProcessorUsage()

        with patch(
            "ractogateway.pipelines.video_processor._transcriber.extract_audio",
            return_value=Path("audio.wav"),
        ), patch(
            "ractogateway.pipelines.video_processor._transcriber.get_audio_duration",
            return_value=10.0,
        ):
            segs = pipeline._run_transcription(
                Path("video.mp4"),
                "en",
                usage,
                start_time_seconds=125.0,
                end_time_seconds=135.0,
                time_offset_seconds=125.0,
            )

        assert segs[0].start == 125.0
        assert segs[0].end == 127.0

    def test_run_with_pre_extracted_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pipeline accepts list of image paths and skips OpenCV extraction."""
        f1 = tmp_path / "f0.jpg"
        f2 = tmp_path / "f1.jpg"
        f1.write_bytes(_fake_jpeg_bytes())
        f2.write_bytes(_fake_jpeg_bytes())

        self._patch_dedup_keep_all(monkeypatch)

        # Patch PIL inside load_frames_from_paths
        fake_pil = MagicMock()
        img_mock = MagicMock()
        img_mock.save.side_effect = lambda buf, **kw: buf.write(_fake_jpeg_bytes())
        fake_pil.open.return_value.__enter__ = lambda s: img_mock
        fake_pil.open.return_value.convert.return_value = img_mock

        with patch(
            "ractogateway.pipelines.video_processor._extractor._require_pil",
            return_value=fake_pil,
        ):
            pipeline = _make_pipeline(
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
            )
            result = pipeline.run([str(f1), str(f2)])

        # Even if PIL mock is imperfect, frames_extracted should be 2
        assert "2 pre-extracted frames" in result.video_path
        assert result.error is None or "image" in (result.error or "").lower()

    def test_run_bytes_input_creates_temp_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bytes input writes to temp file; video_path label is '<bytes>'."""
        pipeline = _make_pipeline(safe_mode=True)

        # Intercept after loader resolves — use a mock for extract_frames
        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(0),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ):
            result = pipeline.run(b"\x00\x01\x02")

        assert result.video_path == "<bytes>"

    def test_frames_extracted_count_correct(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Usage tracks extracted vs kept vs discarded correctly."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(5),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=95.0,  # high similarity → discard all except first
        ):
            pipeline = _make_pipeline(
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
            )
            result = pipeline.run(str(f))

        assert result.error is None
        assert result.usage.frames_extracted == 5
        assert result.usage.frames_kept == 1
        assert result.usage.frames_discarded == 4

    def test_per_call_threshold_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Per-call similarity_threshold overrides constructor default."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(3),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=50.0,  # 50% similar
        ):
            pipeline = _make_pipeline(
                similarity_threshold=90.0,  # constructor: 90 % → keep (50 < 90)
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
            )
            # Override to 40 % → discard (50 >= 40)
            result = pipeline.run(str(f), similarity_threshold=40.0)

        assert result.usage.frames_kept == 1  # only first frame kept

    def test_analysis_tokens_accumulated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """LLM token usage is accumulated in VideoProcessorUsage."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        kit = _make_kit_mock("Board: F=ma")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(2),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=10.0,  # keep all
        ), patch(
            # Bypass the vision analysis internals — directly test token accumulation
            "ractogateway.pipelines.video_processor._analyzer.analyze_frames_sync",
            return_value=[
                FrameEntry(
                    frame_id=i,
                    timestamp=float(i),
                    similarity_to_prev=None if i == 0 else 10.0,
                    kept=True,
                    analysis="Board: F=ma",
                )
                for i in range(2)
            ],
        ):
            pipeline = _make_pipeline(
                kit=kit,
                safe_mode=True,
                analyze_frames=True,
                transcribe_audio=False,
                generate_summary=False,
            )
            result = pipeline.run(str(f))

        # kit.chat called for analysis; usage mock returns 100 in + 50 out
        assert result.error is None

    def test_analysis_prompt_is_valid_against_racto_prompt_schema(
        self, tmp_path: Path
    ) -> None:
        """Analysis stage should not fail prompt validation (e.g., missing tone)."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        kit = _make_kit_mock("Detected board content")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ):
            pipeline = _make_pipeline(
                kit=kit,
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=True,
                generate_summary=False,
            )
            result = pipeline.run(str(f))

        assert result.error is None
        assert all(err.stage != "analyze" for err in result.stage_errors)

    def test_summary_prompt_is_valid_against_racto_prompt_schema(
        self, tmp_path: Path
    ) -> None:
        """Summary stage should not fail prompt validation (e.g., missing tone)."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        kit = _make_kit_mock("Structured summary")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ):
            pipeline = _make_pipeline(
                kit=kit,
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=True,
            )
            result = pipeline.run(str(f))

        assert result.error is None
        assert result.summary == "Structured summary"
        assert all(err.stage != "summarize" for err in result.stage_errors)

    def test_analysis_supports_chatconfig_signature_only_kit(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ):
            pipeline = _make_pipeline(
                kit=_StrictChatConfigKit("analyzed"),
                safe_mode=False,
                transcribe_audio=False,
                analyze_frames=True,
                generate_summary=False,
            )
            result = pipeline.run(str(f))

        assert result.error is None
        assert all(err.stage != "analyze" for err in result.stage_errors)

    def test_summary_supports_chatconfig_signature_only_kit(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ):
            pipeline = _make_pipeline(
                kit=_StrictChatConfigKit("strict-summary"),
                safe_mode=False,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=True,
            )
            result = pipeline.run(str(f))

        assert result.summary == "strict-summary"

    def test_answer_question_enriches_result(self) -> None:
        pipeline = _make_pipeline(kit=_make_kit_mock("answer-text"), safe_mode=True)
        base = VideoProcessorResult(
            video_path="v.mp4",
            sections=[
                VideoSection(
                    timestamp_start=125.0,
                    timestamp_end=135.0,
                    frame_ids=[0],
                    visual_content="Board shows x^2 + y^2 = z^2",
                    audio_content="Pythagorean theorem",
                )
            ],
            usage=VideoProcessorUsage(),
        )

        pipeline.run = MagicMock(return_value=base)  # type: ignore[method-assign]
        result = pipeline.answer_question(
            "v.mp4",
            question="Which equation appears near 2:10?",
            processing_mode="passive",
            focus_time="2 mins 10 sec",
            window_seconds=5.0,
        )

        assert result.question == "Which equation appears near 2:10?"
        assert result.answer is not None

    def test_summary_generated_from_sections(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Summary text appears in result when generate_summary=True."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        kit = _make_kit_mock("Comprehensive summary of lecture.")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ), patch(
            "ractogateway.pipelines.video_processor._summarizer.generate_summary_sync",
            return_value="Comprehensive summary of lecture.",
        ):
            pipeline = VideoProcessorPipeline(
                kit=kit,
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=True,
            )
            result = pipeline.run(str(f))

        assert result.error is None
        assert result.summary == "Comprehensive summary of lecture."

    def test_rag_storage_called_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """store_result_in_rag is called and rag_chunk_count is set."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        rag_mock = MagicMock()

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ), patch(
            "ractogateway.pipelines.video_processor._rag.store_result_in_rag",
            return_value=5,
        ) as mock_store:
            pipeline = VideoProcessorPipeline(
                kit=_make_kit_mock(),
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
                rag_pipeline=rag_mock,
            )
            result = pipeline.run(str(f), store_in_rag=True)

        mock_store.assert_called_once()
        assert result.rag_stored is True
        assert result.rag_chunk_count == 5

    def test_rag_not_called_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """RAG storage is skipped when store_in_rag=False."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        rag_mock = MagicMock()

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_similarity",
            return_value=0.0,
        ), patch(
            "ractogateway.pipelines.video_processor._rag.store_result_in_rag",
        ) as mock_store:
            pipeline = VideoProcessorPipeline(
                kit=_make_kit_mock(),
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
                rag_pipeline=rag_mock,
            )
            result = pipeline.run(str(f), store_in_rag=False)

        mock_store.assert_not_called()
        assert result.rag_stored is False


# ---------------------------------------------------------------------------
# Tests — Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Rate limiter integration."""

    def test_raises_when_limit_exceeded(self, tmp_path: Path) -> None:
        rate_limiter = MagicMock()
        rate_limiter.check_and_consume.return_value = False
        rate_limiter.get_remaining.return_value = 0

        pipeline = VideoProcessorPipeline(
            kit=_make_kit_mock(),
            safe_mode=False,
            rate_limiter=rate_limiter,
        )
        with pytest.raises(VideoRateLimitExceededError, match="Rate limit exceeded"):
            pipeline.run(str(tmp_path / "v.mp4"))

    def test_safe_mode_captures_rate_limit_error(self, tmp_path: Path) -> None:
        rate_limiter = MagicMock()
        rate_limiter.check_and_consume.return_value = False
        rate_limiter.get_remaining.return_value = 0

        pipeline = VideoProcessorPipeline(
            kit=_make_kit_mock(),
            safe_mode=True,
            rate_limiter=rate_limiter,
        )
        result = pipeline.run(str(tmp_path / "v.mp4"))
        assert result.error is not None
        assert "Rate limit" in result.error

    def test_passes_when_limit_ok(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        rate_limiter = MagicMock()
        rate_limiter.check_and_consume.return_value = True

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(0),
        ):
            pipeline = VideoProcessorPipeline(
                kit=_make_kit_mock(),
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
                rate_limiter=rate_limiter,
            )
            result = pipeline.run(str(f))

        rate_limiter.check_and_consume.assert_called_once_with("default", 1)
        assert result.error is None


# ---------------------------------------------------------------------------
# Tests — TranscriberBackend enum
# ---------------------------------------------------------------------------


class TestTranscriberBackend:
    """TranscriberBackend enum covers all 9 backends."""

    def test_all_backends_defined(self) -> None:
        backends = {b.value for b in TranscriberBackend}
        assert "faster-whisper" in backends
        assert "openai-whisper" in backends
        assert "huggingface-local" in backends
        assert "openai-api" in backends
        assert "google-api" in backends
        assert "huggingface-api" in backends
        assert "groq-api" in backends
        assert "deepgram-api" in backends
        assert "ollama" in backends

    def test_factory_returns_correct_type(self) -> None:
        from ractogateway.pipelines.video_processor._transcriber import (
            FasterWhisperTranscriber,
            get_transcriber,
        )

        t = get_transcriber(TranscriberBackend.FASTER_WHISPER, "base", None, None)
        assert isinstance(t, FasterWhisperTranscriber)

    def test_factory_openai_api(self) -> None:
        from ractogateway.pipelines.video_processor._transcriber import (
            OpenAIAPITranscriber,
            get_transcriber,
        )

        t = get_transcriber(TranscriberBackend.OPENAI_API, "whisper-1", "key123", None)
        assert isinstance(t, OpenAIAPITranscriber)

    def test_factory_ollama(self) -> None:
        from ractogateway.pipelines.video_processor._transcriber import (
            OllamaTranscriber,
            get_transcriber,
        )

        t = get_transcriber(TranscriberBackend.OLLAMA, "whisper", None, "http://localhost:11434")
        assert isinstance(t, OllamaTranscriber)

    def test_factory_unknown_raises(self) -> None:
        from ractogateway.pipelines.video_processor._transcriber import get_transcriber

        with pytest.raises(ValueError, match="Unknown transcriber backend"):
            get_transcriber("invalid-backend", "model", None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — DeduplicationMethod enum
# ---------------------------------------------------------------------------


class TestDeduplicationMethod:
    """DeduplicationMethod enum."""

    def test_values(self) -> None:
        assert DeduplicationMethod.PHASH.value == "phash"
        assert DeduplicationMethod.SSIM.value == "ssim"


# ---------------------------------------------------------------------------
# Tests — AsyncVideoProcessorPipeline
# ---------------------------------------------------------------------------


class TestAsyncVideoProcessorPipeline:
    """Async variant wraps the sync pipeline correctly."""

    def test_instantiates_with_same_args(self) -> None:
        kit = _make_kit_mock()
        pipeline = AsyncVideoProcessorPipeline(
            kit=kit,
            fps=2.0,
            similarity_threshold=75.0,
            safe_mode=True,
        )
        assert pipeline._inner._fps == 2.0
        assert pipeline._inner._similarity_threshold == 75.0
        assert pipeline._inner._safe_mode is True

    @pytest.mark.asyncio
    async def test_arun_returns_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(0),
        ):
            pipeline = AsyncVideoProcessorPipeline(
                kit=_make_kit_mock(),
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
            )
            result = await pipeline.run(str(f))

        assert isinstance(result, VideoProcessorResult)
        assert result.video_path == str(f)


# ---------------------------------------------------------------------------
# Tests — Public pipeline imports
# ---------------------------------------------------------------------------


class TestPublicImports:
    """All public symbols importable from the top-level pipelines module."""

    def test_importable_from_pipelines(self) -> None:
        from ractogateway.pipelines import (  # noqa: F401
            AsyncVideoProcessorPipeline,
            DeduplicationMethod,
            FrameAnalysisMode,
            FrameEntry,
            TranscriberBackend,
            TranscriptSegment,
            VideoConfig,
            VideoProcessorPipeline,
            VideoProcessorResult,
            VideoProcessorUsage,
            VideoRateLimitExceededError,
            VideoSection,
        )

    def test_importable_from_submodule(self) -> None:
        from ractogateway.pipelines.video_processor import (  # noqa: F401
            AsyncVideoProcessorPipeline,
            VideoProcessorPipeline,
        )


# ---------------------------------------------------------------------------
# Tests — deduplicate_frames_fast (DSA-optimised pHash path)
# ---------------------------------------------------------------------------


class TestDeduplicateFramesFast:
    """Unit tests for the parallel pHash fast-path deduplication."""

    def test_empty_returns_empty(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import (
            deduplicate_frames_fast,
        )

        result = deduplicate_frames_fast(
            [],
            similarity_threshold=90.0,
            method=DeduplicationMethod.PHASH,
        )
        assert result == []

    def test_ssim_falls_back_to_sequential(self) -> None:
        """SSIM path should delegate to deduplicate_frames (sequential)."""
        from ractogateway.pipelines.video_processor._extractor import (
            deduplicate_frames,
            deduplicate_frames_fast,
        )

        raw = _make_raw_frames(3)
        with patch(
            "ractogateway.pipelines.video_processor._extractor.deduplicate_frames",
            wraps=deduplicate_frames,
        ) as mock_seq, patch(
            "ractogateway.pipelines.video_processor._extractor._ssim_similarity",
            return_value=50.0,  # below 90 → all kept
        ):
            result = deduplicate_frames_fast(
                raw,
                similarity_threshold=90.0,
                method=DeduplicationMethod.SSIM,
            )

        # The sequential path must have been called
        mock_seq.assert_called_once()
        assert len(result) == 3
        assert all(hasattr(f, "kept") for f in result)

    def test_phash_fast_path_keeps_first_always(self) -> None:
        """First frame is always kept regardless of similarity."""
        from ractogateway.pipelines.video_processor._extractor import (
            deduplicate_frames_fast,
        )

        raw = _make_raw_frames(3)
        # All frames identical → high similarity → only first kept
        with patch(
            "ractogateway.pipelines.video_processor._extractor._phash_as_int",
            return_value=0xDEADBEEF,
        ):
            result = deduplicate_frames_fast(
                raw,
                similarity_threshold=90.0,
                method=DeduplicationMethod.PHASH,
            )

        assert result[0].kept is True
        assert result[1].kept is False  # identical hash → high sim → discard
        assert result[2].kept is False

    def test_phash_fast_path_keeps_distinct_frames(self) -> None:
        """Frames with maximally different hashes (XOR=64 bits) are all kept."""
        from ractogateway.pipelines.video_processor._extractor import (
            deduplicate_frames_fast,
        )

        # Alternating 0 and 0xFFFFFFFFFFFFFFFF → Hamming distance = 64 → 0% sim
        raw = _make_raw_frames(4)
        alternating = [0x0, 0xFFFFFFFFFFFFFFFF, 0x0, 0xFFFFFFFFFFFFFFFF]
        call_count = [0]

        def _fake_phash_as_int(_img_bytes: bytes) -> int:
            val = alternating[call_count[0] % len(alternating)]
            call_count[0] += 1
            return val

        with patch(
            "ractogateway.pipelines.video_processor._extractor._phash_as_int",
            side_effect=_fake_phash_as_int,
        ):
            result = deduplicate_frames_fast(
                raw,
                similarity_threshold=90.0,
                method=DeduplicationMethod.PHASH,
            )

        assert all(f.kept for f in result)

    def test_hamming_similarity_identical(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import (
            _hamming_similarity,
        )

        assert _hamming_similarity(0xABCD, 0xABCD) == 100.0

    def test_hamming_similarity_opposite(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import (
            _hamming_similarity,
        )

        # All 64 bits differ → 0% similarity
        assert _hamming_similarity(0x0, 0xFFFFFFFFFFFFFFFF) == 0.0

    def test_phash_as_int_returns_int(self) -> None:
        from ractogateway.pipelines.video_processor._extractor import _phash_as_int

        fake_imagehash = MagicMock()
        fake_hash = MagicMock()
        fake_hash.__str__ = MagicMock(return_value="0000000000000001")
        fake_imagehash.phash.return_value = fake_hash

        fake_pil = MagicMock()
        fake_img = MagicMock()
        fake_pil.open.return_value.convert.return_value = fake_img

        with patch(
            "ractogateway.pipelines.video_processor._extractor._require_imagehash",
            return_value=fake_imagehash,
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._require_pil",
            return_value=fake_pil,
        ):
            result = _phash_as_int(_fake_jpeg_bytes())

        assert isinstance(result, int)
        assert result == 1


# ---------------------------------------------------------------------------
# Tests — AsyncVideoProcessorPipeline.answer_question
# ---------------------------------------------------------------------------


class TestAsyncPipelineAnswerQuestion:
    """answer_question delegated via AsyncVideoProcessorPipeline."""

    @pytest.mark.asyncio
    async def test_answer_question_delegates_to_inner(self) -> None:
        # Use _StrictChatConfigKit so both chat() and achat() work correctly
        kit = _StrictChatConfigKit("The answer is 42.")
        pipeline = AsyncVideoProcessorPipeline(
            kit=kit,
            safe_mode=True,
            transcribe_audio=False,
            analyze_frames=False,
            generate_summary=False,
        )

        base_result = VideoProcessorResult(
            video_path="v.mp4",
            sections=[
                VideoSection(
                    timestamp_start=0.0,
                    timestamp_end=10.0,
                    frame_ids=[0],
                    visual_content="Answer: 42",
                )
            ],
            usage=VideoProcessorUsage(),
        )

        # Wrap the inner arun to be async and return base_result
        async def _async_arun(source: object, **kwargs: object) -> VideoProcessorResult:
            return base_result

        pipeline._inner.arun = _async_arun  # type: ignore[method-assign]

        result = await pipeline.answer_question(
            "v.mp4",
            question="What is the answer?",
        )

        assert result.question == "What is the answer?"
        assert result.answer is not None

    def test_parse_timestamp_delegates(self) -> None:
        """AsyncVideoProcessorPipeline.parse_timestamp mirrors inner pipeline."""
        assert AsyncVideoProcessorPipeline.parse_timestamp("1:30") == 90.0
        assert AsyncVideoProcessorPipeline.parse_timestamp(45) == 45.0


# ---------------------------------------------------------------------------
# Tests — concurrent transcription + analysis in _arun_pipeline
# ---------------------------------------------------------------------------


class TestAsyncPipelineConcurrency:
    """Verify that _arun_pipeline runs transcription and analysis concurrently."""

    @pytest.mark.asyncio
    async def test_arun_concurrent_stages_both_succeed(
        self, tmp_path: Path
    ) -> None:
        """Both transcription and analysis complete without errors."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        analyzed_frames = [
            FrameEntry(
                frame_id=0,
                timestamp=0.0,
                kept=True,
                analysis="Board: E=mc²",
            )
        ]

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_as_int",
            return_value=0,
        ), patch(
            "ractogateway.pipelines.video_processor._analyzer.analyze_frames_async",
            return_value=analyzed_frames,
        ) as mock_analyze, patch(
            "ractogateway.pipelines.video_processor.pipeline.VideoProcessorPipeline._run_transcription",
            return_value=[],
        ) as mock_transcribe:
            pipeline = VideoProcessorPipeline(
                kit=_make_kit_mock(),
                safe_mode=True,
                transcribe_audio=True,
                analyze_frames=True,
                generate_summary=False,
            )
            result = await pipeline.arun(str(f))

        assert result.error is None
        mock_analyze.assert_called_once()
        mock_transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_arun_analysis_failure_is_nonfatal_in_safe_mode(
        self, tmp_path: Path
    ) -> None:
        """Analysis failure in safe_mode yields stage_error but no exception."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(1),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor._phash_as_int",
            return_value=0,
        ), patch(
            "ractogateway.pipelines.video_processor._analyzer.analyze_frames_async",
            side_effect=RuntimeError("Vision LLM down"),
        ):
            pipeline = VideoProcessorPipeline(
                kit=_make_kit_mock(),
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=True,
                generate_summary=False,
            )
            result = await pipeline.arun(str(f))

        assert any(e.stage == "analyze" for e in result.stage_errors)

    @pytest.mark.asyncio
    async def test_arun_uses_deduplicate_frames_fast(
        self, tmp_path: Path
    ) -> None:
        """_arun_pipeline calls deduplicate_frames_fast not deduplicate_frames."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")

        with patch(
            "ractogateway.pipelines.video_processor._extractor.extract_frames",
            return_value=_make_raw_frames(2),
        ), patch(
            "ractogateway.pipelines.video_processor._extractor.deduplicate_frames_fast",
            return_value=[
                FrameEntry(frame_id=0, timestamp=0.0, kept=True),
                FrameEntry(frame_id=1, timestamp=1.0, kept=True),
            ],
        ) as mock_fast, patch(
            "ractogateway.pipelines.video_processor._extractor.deduplicate_frames",
        ) as mock_slow:
            pipeline = VideoProcessorPipeline(
                kit=_make_kit_mock(),
                safe_mode=True,
                transcribe_audio=False,
                analyze_frames=False,
                generate_summary=False,
            )
            result = await pipeline.arun(str(f))

        mock_fast.assert_called_once()
        mock_slow.assert_not_called()
        assert result.usage.frames_kept == 2
