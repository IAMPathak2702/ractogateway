"""Comprehensive summary generation for VideoProcessorPipeline.

Combines visual frame analyses + audio transcripts → one structured summary
covering whiteboard equations, screen content, and spoken explanations.
"""

from __future__ import annotations

from typing import Any

from ._models import TranscriptSegment, VideoProcessorUsage, VideoSection

# ---------------------------------------------------------------------------
# Summary prompt
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """\
You are an expert at analysing recorded tutorial and lecture videos.
You will receive a chronological log of:
  • Visual content (what was written on the whiteboard/board, shown on screen)
  • Audio transcript (what the presenter said)

Your task is to generate a comprehensive, structured summary. The summary MUST include:

1. **Overview** - What is this video about? (2-3 sentences)
2. **Key Topics Covered** — Bulleted list of main subjects
3. **Whiteboard / Board Content** — ALL equations, formulas, proofs, diagrams described \
verbatim. Group related items together. Use LaTeX-style notation where helpful.
4. **Screen / Slide Content** — ALL text, code, charts or diagrams shown on screen
5. **Detailed Explanation** — A section-by-section walkthrough aligned with timestamps, \
combining what was said and what was shown
6. **Key Concepts & Definitions** — Important terms and their meanings as explained
7. **Conclusions / Takeaways** — What should the viewer remember?

Be thorough. Do not omit any equation or formula from the board content section."""


def _build_context(
    sections: list[VideoSection],
    transcript: list[TranscriptSegment],
    max_chars: int = 80_000,
) -> str:
    """Build the LLM context string from sections + transcript."""
    lines: list[str] = []

    # Timeline sections (visual + audio merged)
    if sections:
        lines.append("=== TIMELINE (visual + audio by timestamp) ===\n")
        for sec in sections:
            lines.append(
                f"[{sec.timestamp_start:.1f}s -{sec.timestamp_end:.1f}s]"
            )
            if sec.visual_content:
                lines.append(f"  VISUAL: {sec.visual_content}")
            if sec.audio_content:
                lines.append(f"  AUDIO:  {sec.audio_content}")
            lines.append("")

    # Full transcript (for context if no sections)
    if transcript and not sections:
        lines.append("=== FULL TRANSCRIPT ===\n")
        for seg in transcript:
            lines.append(f"[{seg.start:.1f}s] {seg.text}")

    context = "\n".join(lines)
    # Trim if too long
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[… context truncated for length …]"
    return context


def generate_summary_sync(
    sections: list[VideoSection],
    transcript: list[TranscriptSegment],
    kit: Any,
    usage: VideoProcessorUsage,
) -> str:
    """Generate summary synchronously using *kit*.

    Updates *usage* with summary token counts. Returns the summary string.
    """
    from ractogateway.prompts.engine import RactoPrompt  # noqa: PLC0415

    context = _build_context(sections, transcript)
    if not context.strip():
        return "(No content extracted from this video.)"

    prompt = RactoPrompt(
        role="expert lecture and tutorial summariser",
        aim=_SUMMARY_SYSTEM_PROMPT,
        constraints=[
            "Include every equation and formula exactly as it appeared",
            "Preserve mathematical notation faithfully",
            "Organise by the 7 numbered sections listed above",
            "Use Markdown formatting with headers and bullet points",
        ],
        tone="Professional and comprehensive.",
        output_format="A structured Markdown document with all 7 required sections",
        context=context,
    )

    response = kit.chat(prompt=prompt)
    usage.summary_input_tokens += (response.usage or {}).get("prompt_tokens", 0)
    usage.summary_output_tokens += (response.usage or {}).get("completion_tokens", 0)
    return response.content or "(Summary generation returned no content.)"


async def generate_summary_async(
    sections: list[VideoSection],
    transcript: list[TranscriptSegment],
    kit: Any,
    usage: VideoProcessorUsage,
) -> str:
    """Async variant of :func:`generate_summary_sync`."""
    from ractogateway.prompts.engine import RactoPrompt  # noqa: PLC0415

    context = _build_context(sections, transcript)
    if not context.strip():
        return "(No content extracted from this video.)"

    prompt = RactoPrompt(
        role="expert lecture and tutorial summariser",
        aim=_SUMMARY_SYSTEM_PROMPT,
        constraints=[
            "Include every equation and formula exactly as it appeared",
            "Preserve mathematical notation faithfully",
            "Organise by the 7 numbered sections listed above",
            "Use Markdown formatting with headers and bullet points",
        ],
        tone="Professional and comprehensive.",
        output_format="A structured Markdown document with all 7 required sections",
        context=context,
    )

    response = await kit.achat(prompt=prompt)
    usage.summary_input_tokens += (response.usage or {}).get("prompt_tokens", 0)
    usage.summary_output_tokens += (response.usage or {}).get("completion_tokens", 0)
    return response.content or "(Summary generation returned no content.)"


# ---------------------------------------------------------------------------
# Section builder (merges frames + transcript into VideoSections)
# ---------------------------------------------------------------------------


def build_sections(
    frames: list,  # list[FrameEntry]
    transcript: list[TranscriptSegment],
) -> list[VideoSection]:
    """Merge visual frame analyses and transcript segments into VideoSections.

    Each transcript segment becomes one section; frames are matched by
    timestamp overlap.  If there are no transcript segments, each kept
    frame becomes its own section.
    """
    from ._models import VideoSection  # noqa: PLC0415

    kept_frames = [f for f in frames if f.kept]

    if transcript:
        sections: list[VideoSection] = []
        for seg in transcript:
            matching_ids = [
                f.frame_id
                for f in kept_frames
                if seg.start <= f.timestamp <= seg.end
            ]
            visual = "\n\n".join(
                f.analysis
                for f in kept_frames
                if f.frame_id in matching_ids and f.analysis
            )
            sections.append(
                VideoSection(
                    timestamp_start=seg.start,
                    timestamp_end=seg.end,
                    frame_ids=matching_ids,
                    visual_content=visual,
                    audio_content=seg.text,
                )
            )
        return sections

    # No transcript — one section per frame
    return [
        VideoSection(
            timestamp_start=f.timestamp,
            timestamp_end=f.timestamp,
            frame_ids=[f.frame_id],
            visual_content=f.analysis or "",
            audio_content="",
        )
        for f in kept_frames
    ]
