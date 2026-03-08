"""Vision LLM frame analysis for VideoProcessorPipeline.

Supports two modes:
  INDIVIDUAL -- one LLM call per frame (highest accuracy)
  GRID       -- stitch N frames into a collage grid -> one LLM call per batch
               (fewer API calls, lower cost)

Uses RactoGateway's existing RactoFile + ChatConfig.attachments interface
so any vision-capable kit (OpenAI gpt-4o, Anthropic claude-3, Gemini 1.5, ...)
works transparently.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ._models import FrameAnalysisMode, FrameEntry, VideoProcessorUsage

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _require_pil():  # type: ignore[return]
    try:
        from PIL import Image  # noqa: PLC0415
        return Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image analysis. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FRAME_ANALYSIS_PROMPT = """\
You are analysing a single frame from a tutorial or lecture video.
Extract ALL of the following that are present:

1. **Whiteboard / Blackboard content** -- Copy every equation, formula, symbol, \
word, or diagram *exactly* as written. Preserve mathematical notation.
2. **Screen / slide content** -- Any text, code, diagrams, charts shown on a monitor \
or projected slide. Copy verbatim.
3. **Visual description** -- A brief (1-2 sentence) description of what is happening \
in the frame (person writing, demo running, etc.).

If a category has no content, omit it. Be precise and complete."""

_GRID_ANALYSIS_PROMPT = """\
You are analysing a grid of {n} consecutive video frames from a tutorial or lecture.
The frames are arranged left-to-right, top-to-bottom in chronological order.

For EACH frame (label them Frame 1, Frame 2 ... Frame {n}), extract:
1. **Whiteboard / Blackboard content** -- every equation, formula, symbol, or word, \
copied exactly.
2. **Screen / slide content** -- any visible text, code, charts.
3. **Visual description** -- 1 sentence describing the frame.

Be precise. Do not hallucinate content not visible in the image."""


# ---------------------------------------------------------------------------
# Config helper — kit-agnostic ChatConfig with attachments
# ---------------------------------------------------------------------------


def _build_attachment_config(kit: Any, attachments: list) -> Any:  # noqa: ANN401
    """Return a ``ChatConfig(attachments=...)`` instance for *kit*'s provider.

    Resolves the correct ``ChatConfig`` class from the kit's own package by
    traversing the module hierarchy, so this works with OpenAI, Anthropic,
    Google, Ollama, and HuggingFace kits without hard-coding provider names.
    Returns ``None`` if the config class cannot be found (attachments silently
    dropped — the kit will still receive the text prompt).
    """
    kit_mod = getattr(type(kit), "__module__", "") or ""
    parts = kit_mod.split(".")
    for end in range(len(parts), 0, -1):  # noqa: B905 — intentional range
        pkg = ".".join(parts[:end])
        try:
            mod = importlib.import_module(pkg)
            config_cls = getattr(mod, "ChatConfig", None)
            if config_cls is not None:
                return config_cls(attachments=attachments)
        except ImportError:  # noqa: S110
            continue
    return None


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _build_grid(frames_bytes: list[bytes], grid_cols: int = 2) -> bytes:
    """Stitch a list of JPEG/PNG images into a grid collage.

    Returns raw JPEG bytes of the collage.
    """
    pil = _require_pil()

    images = [pil.open(io.BytesIO(b)).convert("RGB") for b in frames_bytes]
    if not images:
        raise ValueError("No images to build grid from")

    thumb_w, thumb_h = 320, 240
    thumbs = [img.resize((thumb_w, thumb_h)) for img in images]

    cols = min(grid_cols, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)

    canvas = pil.new("RGB", (cols * thumb_w, rows * thumb_h), color=(30, 30, 30))
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        canvas.paste(thumb, (col * thumb_w, row * thumb_h))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Single-frame analysis (sync)
# ---------------------------------------------------------------------------


def _analyze_single_frame_sync(frame: FrameEntry, kit: Any) -> tuple[str, dict]:  # noqa: ANN401
    """Analyse one frame synchronously. Returns (analysis_text, usage_dict)."""
    from ractogateway.prompts.engine import RactoFile, RactoPrompt  # noqa: PLC0415

    if not frame.image_data:
        return "", {}

    mime = "image/jpeg" if frame.image_format.upper() == "JPEG" else "image/png"
    rf = RactoFile.from_bytes(frame.image_data, mime)

    prompt = RactoPrompt(
        role="vision analysis assistant",
        aim=_FRAME_ANALYSIS_PROMPT,
        constraints=["Be concise but complete", "Copy text verbatim -- do not paraphrase"],
        tone="Professional and factual.",
        output_format="Structured plain text with labelled sections",
    )

    config = _build_attachment_config(kit, [rf])
    response = kit.chat(prompt=prompt, config=config)
    text = response.content or ""
    usage = response.usage or {}
    return text, usage


# ---------------------------------------------------------------------------
# Grid-frame analysis (sync)
# ---------------------------------------------------------------------------


def _analyze_grid_sync(
    frames: list[FrameEntry],
    kit: Any,
    grid_cols: int,
) -> tuple[str, dict]:  # noqa: ANN401
    """Analyse a batch of frames as a grid collage. Returns (analysis_text, usage_dict)."""
    from ractogateway.prompts.engine import RactoFile, RactoPrompt  # noqa: PLC0415

    valid = [f for f in frames if f.image_data]
    if not valid:
        return "", {}

    grid_bytes = _build_grid([f.image_data for f in valid], grid_cols=grid_cols)  # type: ignore[arg-type]
    rf = RactoFile.from_bytes(grid_bytes, "image/jpeg")

    n = len(valid)
    prompt = RactoPrompt(
        role="vision analysis assistant",
        aim=_GRID_ANALYSIS_PROMPT.format(n=n),
        constraints=[
            "Label each frame as 'Frame 1', 'Frame 2', etc.",
            "Copy all text/equations verbatim",
        ],
        tone="Professional and factual.",
        output_format="Structured plain text, one section per frame",
    )

    config = _build_attachment_config(kit, [rf])
    response = kit.chat(prompt=prompt, config=config)
    return response.content or "", response.usage or {}


# ---------------------------------------------------------------------------
# Async single-frame analysis
# ---------------------------------------------------------------------------


async def _analyze_single_frame_async(frame: FrameEntry, kit: Any) -> tuple[str, dict]:  # noqa: ANN401
    """Async variant of single-frame analysis."""
    from ractogateway.prompts.engine import RactoFile, RactoPrompt  # noqa: PLC0415

    if not frame.image_data:
        return "", {}

    mime = "image/jpeg" if frame.image_format.upper() == "JPEG" else "image/png"
    rf = RactoFile.from_bytes(frame.image_data, mime)

    prompt = RactoPrompt(
        role="vision analysis assistant",
        aim=_FRAME_ANALYSIS_PROMPT,
        constraints=["Be concise but complete", "Copy text verbatim -- do not paraphrase"],
        tone="Professional and factual.",
        output_format="Structured plain text with labelled sections",
    )

    config = _build_attachment_config(kit, [rf])
    response = await kit.achat(prompt=prompt, config=config)
    return response.content or "", response.usage or {}


# ---------------------------------------------------------------------------
# Public batch-analysis functions
# ---------------------------------------------------------------------------


def analyze_frames_sync(
    frames: list[FrameEntry],
    kit: Any,
    *,
    mode: FrameAnalysisMode,
    batch_size: int,
    max_workers: int,
    grid_size: int,
    usage: VideoProcessorUsage,
) -> list[FrameEntry]:
    """Analyse all kept frames synchronously via a ThreadPoolExecutor.

    Updates *usage* in place with token counts.
    Returns a new list of :class:`FrameEntry` with ``analysis`` populated.
    """
    kept = [f for f in frames if f.kept and f.image_data]
    if not kept:
        return frames

    results: dict[int, str] = {}

    if mode == FrameAnalysisMode.INDIVIDUAL:
        # Split into batches and process concurrently
        batches = [kept[i : i + batch_size] for i in range(0, len(kept), batch_size)]
        for batch in batches:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
                future_to_frame = {
                    pool.submit(_analyze_single_frame_sync, f, kit): f for f in batch
                }
                for fut in as_completed(future_to_frame):
                    frm = future_to_frame[fut]
                    text, usg = fut.result()
                    results[frm.frame_id] = text
                    usage.analysis_input_tokens += usg.get("prompt_tokens", 0)
                    usage.analysis_output_tokens += usg.get("completion_tokens", 0)
    else:
        # GRID mode -- stitch into grid batches
        grid_batches = [kept[i : i + grid_size] for i in range(0, len(kept), grid_size)]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(grid_batches))) as pool:
            future_to_batch = {
                pool.submit(_analyze_grid_sync, gb, kit, 2): gb for gb in grid_batches
            }
            for fut in as_completed(future_to_batch):
                gb = future_to_batch[fut]
                text, usg = fut.result()
                usage.analysis_input_tokens += usg.get("prompt_tokens", 0)
                usage.analysis_output_tokens += usg.get("completion_tokens", 0)
                # Distribute analysis text to all frames in the grid batch
                for frm in gb:
                    results[frm.frame_id] = text

    # Rebuild frame list with analysis populated
    updated: list[FrameEntry] = []
    for f in frames:
        if f.frame_id in results:
            updated.append(f.model_copy(update={"analysis": results[f.frame_id]}))
        else:
            updated.append(f)
    return updated


async def analyze_frames_async(
    frames: list[FrameEntry],
    kit: Any,
    *,
    mode: FrameAnalysisMode,
    batch_size: int,
    max_workers: int,
    grid_size: int,
    usage: VideoProcessorUsage,
) -> list[FrameEntry]:
    """Async variant -- uses asyncio.gather for concurrent LLM calls."""
    kept = [f for f in frames if f.kept and f.image_data]
    if not kept:
        return frames

    results: dict[int, str] = {}

    if mode == FrameAnalysisMode.INDIVIDUAL:
        batches = [kept[i : i + batch_size] for i in range(0, len(kept), batch_size)]
        for batch in batches:
            tasks = [_analyze_single_frame_async(f, kit) for f in batch]
            outputs = await asyncio.gather(*tasks, return_exceptions=False)
            for frm, (text, usg) in zip(batch, outputs, strict=False):
                results[frm.frame_id] = text
                usage.analysis_input_tokens += usg.get("prompt_tokens", 0)
                usage.analysis_output_tokens += usg.get("completion_tokens", 0)
    else:
        # GRID: run in thread pool (sync PIL stitching + sync kit call)
        loop = asyncio.get_event_loop()
        grid_batches = [kept[i : i + grid_size] for i in range(0, len(kept), grid_size)]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(grid_batches))) as pool:
            futures = [
                loop.run_in_executor(pool, _analyze_grid_sync, gb, kit, 2)
                for gb in grid_batches
            ]
            outputs = await asyncio.gather(*futures)
        for gb, (text, usg) in zip(grid_batches, outputs, strict=False):
            usage.analysis_input_tokens += usg.get("prompt_tokens", 0)
            usage.analysis_output_tokens += usg.get("completion_tokens", 0)
            for frm in gb:
                results[frm.frame_id] = text

    updated: list[FrameEntry] = []
    for f in frames:
        if f.frame_id in results:
            updated.append(f.model_copy(update={"analysis": results[f.frame_id]}))
        else:
            updated.append(f)
    return updated
