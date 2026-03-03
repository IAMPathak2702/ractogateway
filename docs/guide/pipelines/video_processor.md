# Video Processor Pipeline

`VideoProcessorPipeline` turns any video file — local, remote, or YouTube — into
structured knowledge: unique frames extracted at a configurable rate, audio transcribed
by your chosen model, every whiteboard equation and screen text captured by a vision LLM,
and a 7-section comprehensive summary produced automatically. Results can optionally
be indexed in a vector store for downstream Q&A via RactoRAG.

Use `AsyncVideoProcessorPipeline` in async app stacks (FastAPI, etc.).

## Best Use Cases

- Indexing lecture recordings and university course videos for RAG
- Extracting all equations and proofs from math / physics tutorial videos
- Building searchable archives of technical demo recordings
- Generating structured notes from conference talks or webinars
- Processing training videos for corporate knowledge bases

## Installation

```bash
# Core — frame extraction, dedup, audio extraction, HTTP download
pip install "ractogateway[pipelines-video]"

# + local transcription with faster-whisper (recommended)
pip install "ractogateway[pipelines-video-whisper]"

# + YouTube video download via yt-dlp
pip install "ractogateway[pipelines-video-yt]"

# Everything at once
pip install "ractogateway[pipelines-video-full]"
```

## Minimal Example

```python
from ractogateway.openai_developer_kit import Chat
from ractogateway.pipelines.video_processor import (
    VideoProcessorPipeline,
    TranscriberBackend,
)

kit = Chat(api_key="sk-...", model="gpt-4o")

pipeline = VideoProcessorPipeline(
    kit=kit,
    fps=1.0,                       # sample 1 frame per second
    similarity_threshold=85.0,     # keep frames that differ by more than 15 %
    transcriber=TranscriberBackend.FASTER_WHISPER,
    transcriber_model="base",
    analyze_frames=True,
    generate_summary=True,
)

result = pipeline.run("lecture.mp4")

print(f"Kept {result.usage.frames_kept} / {result.usage.frames_extracted} frames")
print(result.summary)
```

**Example output**

```
Kept 18 / 120 frames

# Video Summary

## Overview
This lecture introduces Newton's Laws of Motion, covering all three laws with
mathematical formulations and worked examples on the whiteboard.

## Whiteboard / Board Content
- F = ma  (Newton's Second Law)
- ΣF = 0  (First Law equilibrium condition)
- F₁₂ = −F₂₁  (Third Law)
- Worked example: m = 5 kg, a = 2 m/s² → F = 10 N

## Screen / Slide Content
- Slide 3: "Newton's Laws — Historical Context (1687)"
- Code snippet (Python simulation): `force = mass * acceleration`

## Key Concepts & Definitions
...
```

---

## Accepted Video Sources

The pipeline accepts **five different input types** — no need to pre-download:

```python
# 1. Local file path (str or Path)
result = pipeline.run("lecture.mp4")
result = pipeline.run(Path("/recordings/session1.mov"))

# 2. HTTP / HTTPS URL (requires httpx — included in pipelines-video)
result = pipeline.run("https://cdn.university.edu/physics101.mp4")

# 3. YouTube URL (requires yt-dlp — install pipelines-video-yt)
result = pipeline.run("https://www.youtube.com/watch?v=abc123xyz")
result = pipeline.run("https://youtu.be/abc123xyz")

# 4. Raw bytes buffer (in-memory)
with open("lecture.mp4", "rb") as f:
    video_bytes = f.read()
result = pipeline.run(video_bytes)

# 5. Pre-extracted frame images (skip OpenCV extraction entirely)
frame_paths = ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]
result = pipeline.run(frame_paths)
```

---

## Frame Deduplication

Identical or near-identical frames (e.g., static slides) are automatically
filtered to reduce LLM cost and noise.

### Configuring the threshold

```python
pipeline = VideoProcessorPipeline(
    kit=kit,
    similarity_threshold=85.0,   # keep frames that differ by > 15 %
    # similarity_threshold=95.0  # aggressive — only keep dramatically different frames
    # similarity_threshold=60.0  # conservative — keep more frames
)
```

- **>= threshold** → frame is **discarded** (too similar to previous)
- **< threshold** → frame is **kept**

### Choosing the algorithm

```python
from ractogateway.pipelines.video_processor import DeduplicationMethod

# pHash — fast, good for most content (default)
pipeline = VideoProcessorPipeline(kit=kit, dedup_method=DeduplicationMethod.PHASH)

# SSIM — structural similarity, more accurate for subtle changes
# Requires: scikit-image (included in pipelines-video)
pipeline = VideoProcessorPipeline(kit=kit, dedup_method=DeduplicationMethod.SSIM)
```

### Inspecting deduplication results

```python
for frame in result.frames:
    status = "KEPT" if frame.kept else "SKIP"
    sim = f"{frame.similarity_to_prev:.1f}%" if frame.similarity_to_prev else "first"
    print(f"Frame {frame.frame_id:3d}  [{frame.timestamp:6.1f}s]  {status}  sim={sim}")
```

**Example output**

```
Frame   0  [  0.0s]  KEPT  sim=first
Frame   1  [  1.0s]  SKIP  sim=97.2%
Frame   2  [  2.0s]  SKIP  sim=98.1%
Frame   3  [  3.0s]  KEPT  sim=41.3%   ← new content on board
Frame   4  [  4.0s]  KEPT  sim=22.7%   ← more writing
```

---

## Audio Transcription

### Choosing a backend

```python
from ractogateway.pipelines.video_processor import TranscriberBackend

# Local — no API key needed
pipeline = VideoProcessorPipeline(
    kit=kit,
    transcriber=TranscriberBackend.FASTER_WHISPER,
    transcriber_model="base",      # tiny / base / small / medium / large-v3
)

# Cloud — OpenAI Whisper API
pipeline = VideoProcessorPipeline(
    kit=kit,
    transcriber=TranscriberBackend.OPENAI_API,
    transcriber_model="whisper-1",
    transcriber_api_key="sk-...",  # or set OPENAI_API_KEY env var
)

# Cloud — Groq (ultra-fast, cheap)
pipeline = VideoProcessorPipeline(
    kit=kit,
    transcriber=TranscriberBackend.GROQ_API,
    transcriber_model="whisper-large-v3-turbo",
    transcriber_api_key="gsk_...",  # or GROQ_API_KEY
)

# Cloud — Deepgram Nova 3
pipeline = VideoProcessorPipeline(
    kit=kit,
    transcriber=TranscriberBackend.DEEPGRAM_API,
    transcriber_model="nova-3",
    transcriber_api_key="...",      # or DEEPGRAM_API_KEY
)

# Local — HuggingFace (any ASR model)
pipeline = VideoProcessorPipeline(
    kit=kit,
    transcriber=TranscriberBackend.HUGGINGFACE_LOCAL,
    transcriber_model="openai/whisper-large-v3",
)

# Self-hosted — Ollama
pipeline = VideoProcessorPipeline(
    kit=kit,
    transcriber=TranscriberBackend.OLLAMA,
    transcriber_model="whisper",
    transcriber_base_url="http://localhost:11434",
)
```

### Language detection

```python
# Auto-detect (default)
pipeline = VideoProcessorPipeline(kit=kit, language=None)

# Force a specific language
pipeline = VideoProcessorPipeline(kit=kit, language="en")
pipeline = VideoProcessorPipeline(kit=kit, language="fr")
pipeline = VideoProcessorPipeline(kit=kit, language="de")
```

### Disabling transcription

```python
# Vision analysis only — no audio
pipeline = VideoProcessorPipeline(kit=kit, transcribe_audio=False)
```

---

## Vision LLM Analysis

Every kept frame is passed to a vision-capable model to extract:
1. All text/equations written on the whiteboard or blackboard (copied verbatim)
2. All text, code, or diagrams visible on screen
3. A brief description of the scene

### Choosing the analysis provider

```python
from ractogateway.openai_developer_kit import Chat as GPTChat
from ractogateway.anthropic_developer_kit import Chat as ClaudeChat
from ractogateway.google_developer_kit import Chat as GeminiChat

# GPT-4o (OpenAI)
pipeline = VideoProcessorPipeline(
    kit=GPTChat(model="gpt-4o"),
    analyze_frames=True,
)

# Claude 3.5 Sonnet (Anthropic)
pipeline = VideoProcessorPipeline(
    kit=ClaudeChat(model="claude-sonnet-4-6"),
    analyze_frames=True,
)

# Gemini 1.5 Flash (Google) — cheapest for high frame counts
pipeline = VideoProcessorPipeline(
    kit=GeminiChat(model="gemini-1.5-flash"),
    analyze_frames=True,
)
```

### Individual vs Grid mode

```python
from ractogateway.pipelines.video_processor import FrameAnalysisMode

# INDIVIDUAL — one LLM call per frame (highest accuracy)
pipeline = VideoProcessorPipeline(
    kit=kit,
    frame_analysis_mode=FrameAnalysisMode.INDIVIDUAL,
    batch_size=10,          # 10 concurrent calls per batch
    max_workers=4,          # thread-pool size
)

# GRID — stitch 4 frames into a 2×2 collage → one LLM call (4× cheaper)
pipeline = VideoProcessorPipeline(
    kit=kit,
    frame_analysis_mode=FrameAnalysisMode.GRID,
    grid_size=4,            # frames per collage
)
```

### Separate kits per step

Use a powerful model for analysis and a faster/cheaper one for summary:

```python
pipeline = VideoProcessorPipeline(
    kit=GPTChat(model="gpt-4o-mini"),           # fallback / summary
    analysis_kit=GPTChat(model="gpt-4o"),       # vision analysis
    summary_kit=GPTChat(model="gpt-4o"),        # summary
)
```

---

## Summary Generation

The summary LLM receives a chronological log of every visual analysis + transcript
segment and produces a structured 7-section Markdown document:

1. **Overview** — what the video is about
2. **Key Topics Covered** — bulleted list
3. **Whiteboard / Board Content** — ALL equations and formulas verbatim
4. **Screen / Slide Content** — all visible text and code
5. **Detailed Explanation** — timeline walkthrough
6. **Key Concepts & Definitions** — important terms
7. **Conclusions / Takeaways** — key points to remember

```python
result = pipeline.run("lecture.mp4")

# Full Markdown summary
print(result.summary)

# Save to file
result.to_markdown("lecture_notes.md")
```

---

## RAG Storage

Index the full video content for Q&A retrieval using any RactoRAG-compatible store:

```python
from ractogateway.rag.pipeline import RactoRAG
from ractogateway.rag.embedders import OpenAIEmbedder
from ractogateway.rag.stores import PineconeStore  # or ChromaStore, FAISSStore, etc.

rag = RactoRAG(
    embedder=OpenAIEmbedder(api_key="sk-..."),
    store=PineconeStore(api_key="...", index_name="lectures"),
)

pipeline = VideoProcessorPipeline(
    kit=kit,
    rag_pipeline=rag,
)

result = pipeline.run("lecture.mp4", store_in_rag=True)
print(f"Stored {result.rag_chunk_count} chunks")

# Now query the video content
answer = rag.query("What is Newton's Second Law?")
print(answer)
```

Stored chunks include:
- One document per `VideoSection` (visual + audio merged)
- One document per transcript segment (audio-only retrieval)
- One document for the full summary

---

## Async Usage

```python
import asyncio
from ractogateway.pipelines.video_processor import AsyncVideoProcessorPipeline

pipeline = AsyncVideoProcessorPipeline(
    kit=kit,
    fps=1.0,
    similarity_threshold=85.0,
    transcriber=TranscriberBackend.FASTER_WHISPER,
    generate_summary=True,
)

async def process():
    result = await pipeline.run("lecture.mp4")
    print(result.summary)

asyncio.run(process())
```

### FastAPI integration

```python
from fastapi import FastAPI, UploadFile
from ractogateway.pipelines.video_processor import AsyncVideoProcessorPipeline

app = FastAPI()
pipeline = AsyncVideoProcessorPipeline(kit=kit, safe_mode=True)

@app.post("/process-video")
async def process_video(file: UploadFile):
    video_bytes = await file.read()
    result = await pipeline.run(video_bytes)
    return {
        "summary": result.summary,
        "frames_kept": result.usage.frames_kept,
        "transcript": result.get_transcript_text(),
        "error": result.error,
    }
```

---

## Per-Call Overrides

Any constructor parameter can be overridden per call:

```python
# Constructor defaults
pipeline = VideoProcessorPipeline(
    kit=kit,
    fps=1.0,
    similarity_threshold=90.0,
    generate_summary=True,
)

# Override for a specific run
result = pipeline.run(
    "dense_lecture.mp4",
    fps=2.0,                     # sample more frames
    similarity_threshold=70.0,   # keep more variation
    language="fr",               # French audio
    generate_summary=False,      # skip summary for speed
)
```

---

## Production Controls

### Safe mode

```python
# Never raises — errors go to result.error
pipeline = VideoProcessorPipeline(kit=kit, safe_mode=True)
result = pipeline.run("video.mp4")
if result.error:
    print(f"Processing failed: {result.error}")
```

### Rate limiting

```python
from ractogateway.redis import RedisRateLimiter, RateLimitConfig

limiter = RedisRateLimiter(
    url="redis://localhost:6379",
    config=RateLimitConfig(max_tokens_per_minute=10),
)

pipeline = VideoProcessorPipeline(
    kit=kit,
    rate_limiter=limiter,
    user_id="user_42",
)
```

### Telemetry

```python
from ractogateway.telemetry import RactoTracer

tracer = RactoTracer(otlp_endpoint="http://localhost:4317")

pipeline = VideoProcessorPipeline(
    kit=kit,
    tracer=tracer,
)
```

### Controlling concurrency

```python
pipeline = VideoProcessorPipeline(
    kit=kit,
    max_workers=8,          # more concurrent LLM analysis calls
    max_process_workers=4,  # more CPU processes for frame extraction
    batch_size=20,          # larger analysis batches
)
```

---

## Result Object

```python
result = pipeline.run("lecture.mp4")

# Source metadata
result.video_path           # "lecture.mp4"
result.error                # None or error string (safe_mode only)

# Frames
result.frames               # list[FrameEntry]
result.usage.frames_extracted   # 120
result.usage.frames_kept        # 18
result.usage.frames_discarded   # 102

# Transcript
result.transcript           # list[TranscriptSegment]
result.get_transcript_text()  # full text as one string

# Visual content
result.get_all_visual_content()  # all analyses in timestamp order

# Sections (merged)
result.sections             # list[VideoSection] — visual + audio by time

# Summary
result.summary              # Markdown string

# RAG
result.rag_stored           # True if stored
result.rag_chunk_count      # 42

# Token usage
result.usage.total_tokens           # 12500
result.usage.total_analysis_tokens  # 10000
result.usage.total_summary_tokens   # 2500
result.usage.audio_duration_seconds # 3600.0
```

### Export helpers

```python
# JSON (image bytes excluded automatically)
result.to_json("lecture_result.json")
json_str = result.to_json()        # no path → returns string

# Markdown report (summary + transcript + sections)
result.to_markdown("lecture_notes.md")
md_str = result.to_markdown()      # no path → returns string
```

---

## Complete End-to-End Example

```python
from ractogateway.openai_developer_kit import Chat
from ractogateway.pipelines.video_processor import (
    VideoProcessorPipeline,
    DeduplicationMethod,
    FrameAnalysisMode,
    TranscriberBackend,
)

kit = Chat(api_key="sk-...", model="gpt-4o")

pipeline = VideoProcessorPipeline(
    kit=kit,
    # Frame extraction
    fps=1.0,
    similarity_threshold=85.0,
    dedup_method=DeduplicationMethod.PHASH,
    frame_format="JPEG",
    # Vision analysis
    analyze_frames=True,
    frame_analysis_mode=FrameAnalysisMode.INDIVIDUAL,
    batch_size=10,
    max_workers=4,
    # Transcription
    transcribe_audio=True,
    transcriber=TranscriberBackend.FASTER_WHISPER,
    transcriber_model="small",
    language="en",
    # Output
    generate_summary=True,
    # Safety
    safe_mode=True,
)

# Works with any source
for source in [
    "lecture.mp4",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://cdn.example.com/video.mp4",
]:
    result = pipeline.run(source)
    if result.error:
        print(f"Error: {result.error}")
        continue

    print(f"\n{'='*60}")
    print(f"Source: {result.video_path}")
    print(f"Frames: {result.usage.frames_kept} kept / {result.usage.frames_extracted} extracted")
    print(f"Tokens: {result.usage.total_tokens:,}")
    print(f"\n{result.summary}")

    result.to_markdown(f"notes_{result.video_path.replace('/', '_')}.md")
```

---

## See also

- [API Reference — Video Processor](../../api/video_processor.md)
- [Pipelines overview](../pipelines.md)
- [RAG guide](../rag.md)
- [Telemetry guide](../telemetry.md)
- [Redis rate limiter](../redis.md)
