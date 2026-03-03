# API Reference — Video Processor Pipeline

Module: `ractogateway.pipelines.video_processor`

```bash
pip install "ractogateway[pipelines-video]"           # core
pip install "ractogateway[pipelines-video-whisper]"   # + faster-whisper
pip install "ractogateway[pipelines-video-yt]"        # + YouTube via yt-dlp
pip install "ractogateway[pipelines-video-full]"      # all of the above
```

---

## VideoProcessorPipeline

```python
class VideoProcessorPipeline
```

Five-stage pipeline that turns a raw video into structured knowledge:
frame extraction → deduplication → audio transcription → vision LLM analysis → summary.
Optionally stores everything in a RactoRAG vector store for Q&A.

### Constructor

```python
VideoProcessorPipeline(
    kit,
    *,
    analysis_kit=None,
    summary_kit=None,
    # Transcription
    transcriber=TranscriberBackend.FASTER_WHISPER,
    transcriber_model="base",
    transcriber_api_key=None,
    transcriber_base_url=None,
    # Frame extraction
    fps=1.0,
    similarity_threshold=90.0,
    dedup_method=DeduplicationMethod.PHASH,
    max_frames=None,
    frame_format="JPEG",
    # Vision analysis
    frame_analysis_mode=FrameAnalysisMode.INDIVIDUAL,
    grid_size=4,
    batch_size=10,
    max_workers=4,
    max_process_workers=4,
    language=None,
    # Feature flags
    transcribe_audio=True,
    analyze_frames=True,
    generate_summary=True,
    # Integrations
    rag_pipeline=None,
    # Safety & observability
    safe_mode=False,
    tracer=None,
    metrics=None,
    rate_limiter=None,
    user_id="default",
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kit` | `Any` | required | Main developer kit — used for summary + fallback frame analysis |
| `analysis_kit` | `Any \| None` | `None` | Separate kit for frame-by-frame vision analysis (falls back to `kit`) |
| `summary_kit` | `Any \| None` | `None` | Separate kit for summary generation (falls back to `kit`) |
| `transcriber` | `TranscriberBackend` | `FASTER_WHISPER` | Audio transcription backend |
| `transcriber_model` | `str` | `"base"` | Model name / size — interpretation is backend-specific |
| `transcriber_api_key` | `str \| None` | `None` | API key for cloud backends (or read from env var) |
| `transcriber_base_url` | `str \| None` | `None` | Base URL for self-hosted endpoints (Ollama etc.) |
| `fps` | `float` | `1.0` | Frames to sample per second of video |
| `similarity_threshold` | `float` | `90.0` | Discard a frame when its similarity to the previous kept frame is ≥ this % |
| `dedup_method` | `DeduplicationMethod` | `PHASH` | `PHASH` (fast) or `SSIM` (more accurate) |
| `max_frames` | `int \| None` | `None` | Hard cap on kept frames |
| `frame_format` | `str` | `"JPEG"` | `"JPEG"` or `"PNG"` |
| `frame_analysis_mode` | `FrameAnalysisMode` | `INDIVIDUAL` | `INDIVIDUAL` or `GRID` (collage) |
| `grid_size` | `int` | `4` | Frames per grid collage (GRID mode only) |
| `batch_size` | `int` | `10` | Concurrent LLM calls per analysis batch |
| `max_workers` | `int` | `4` | Thread-pool size for concurrent LLM calls |
| `max_process_workers` | `int` | `4` | Process-pool size for CPU-bound extraction |
| `language` | `str \| None` | `None` | BCP-47 code (e.g. `"en"`), `None` = auto-detect |
| `transcribe_audio` | `bool` | `True` | Extract and transcribe the audio track |
| `analyze_frames` | `bool` | `True` | Pass kept frames to the vision LLM |
| `generate_summary` | `bool` | `True` | Generate a comprehensive Markdown summary |
| `rag_pipeline` | `Any \| None` | `None` | `RactoRAG` instance for optional storage |
| `safe_mode` | `bool` | `False` | Catch all exceptions; return them in `result.error` |
| `tracer` | `Any \| None` | `None` | `RactoTracer` for OTEL tracing |
| `metrics` | `Any \| None` | `None` | `GatewayMetricsMiddleware` for Prometheus |
| `rate_limiter` | `Any \| None` | `None` | Duck-typed rate limiter |
| `user_id` | `str` | `"default"` | Default user identifier for rate limiter |

---

### Methods

#### run

```python
def run(
    source: str | Path | bytes | list,
    *,
    fps=None,
    similarity_threshold=None,
    dedup_method=None,
    max_frames=None,
    analyze_frames=None,
    frame_analysis_mode=None,
    grid_size=None,
    batch_size=None,
    transcribe_audio=None,
    language=None,
    generate_summary=None,
    store_in_rag=False,
    user_id=None,
) -> VideoProcessorResult
```

Process a video synchronously. All keyword arguments override the constructor
defaults for this call only.

**Accepted `source` types**

| Type | Example | Behaviour |
|------|---------|-----------|
| `str` (local path) | `"lecture.mp4"` | Opens with OpenCV directly |
| `Path` (local path) | `Path("lecture.mp4")` | Opens with OpenCV directly |
| `str` (HTTP/HTTPS URL) | `"https://cdn.example.com/v.mp4"` | Downloaded via `httpx` to temp file |
| `str` (YouTube URL) | `"https://youtu.be/abc"` | Downloaded via `yt-dlp` to temp file |
| `bytes` | `video_bytes` | Written to a temp file, then processed |
| `list[str \| Path]` | `["f0.jpg", "f1.jpg"]` | Pre-extracted frames — extraction step skipped |

#### arun

```python
async def arun(source, **kwargs) -> VideoProcessorResult
```

Async variant of `run()`. CPU-bound steps run in a `ThreadPoolExecutor`;
LLM calls use `asyncio.gather` for concurrency.

---

## AsyncVideoProcessorPipeline

```python
class AsyncVideoProcessorPipeline
```

Async-only variant of `VideoProcessorPipeline`. Exposes a single
`async run()` method — suitable for FastAPI endpoints.

Constructor parameters are identical to `VideoProcessorPipeline`.

### Methods

#### run

```python
async def run(source, **kwargs) -> VideoProcessorResult
```

---

## Models

### VideoProcessorResult

```python
class VideoProcessorResult(BaseModel)
```

Full output of a pipeline run.

| Field | Type | Description |
|-------|------|-------------|
| `video_path` | `str` | Source identifier (path, URL, or `"<bytes>"`) |
| `frames` | `list[FrameEntry]` | All extracted frames (kept and discarded) |
| `transcript` | `list[TranscriptSegment]` | Audio transcript with timestamps |
| `sections` | `list[VideoSection]` | Merged visual + audio sections |
| `summary` | `str \| None` | Comprehensive Markdown summary |
| `rag_stored` | `bool` | `True` if content was pushed to RactoRAG |
| `rag_chunk_count` | `int` | Number of chunks stored |
| `usage` | `VideoProcessorUsage` | Token and frame accounting |
| `error` | `str \| None` | Error message when `safe_mode=True` |

**Methods**

- `get_transcript_text() -> str` — Full transcript as a single string
- `get_all_visual_content() -> str` — All frame analyses in timestamp order
- `to_json(path=None, *, indent=2) -> str | None` — JSON export (image bytes excluded)
- `to_markdown(path=None) -> str | None` — Structured Markdown report

---

### FrameEntry

```python
class FrameEntry(BaseModel)
```

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | `int` | Zero-based sequential frame ID |
| `timestamp` | `float` | Position in the video in seconds |
| `similarity_to_prev` | `float \| None` | Similarity % to previous kept frame |
| `kept` | `bool` | `False` if discarded by deduplication |
| `analysis` | `str \| None` | LLM-generated content description |
| `image_data` | `bytes \| None` | Raw image bytes (kept frames only) |
| `image_format` | `str` | `"JPEG"` or `"PNG"` |

---

### TranscriptSegment

```python
class TranscriptSegment(BaseModel)
```

| Field | Type | Description |
|-------|------|-------------|
| `start` | `float` | Segment start time (seconds) |
| `end` | `float` | Segment end time (seconds) |
| `text` | `str` | Transcribed text |
| `frame_ids` | `list[int]` | Kept frame IDs within this time window |

---

### VideoSection

```python
class VideoSection(BaseModel)
```

Merged time section combining visual analyses + audio transcript.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_start` | `float` | Section start (seconds) |
| `timestamp_end` | `float` | Section end (seconds) |
| `frame_ids` | `list[int]` | Frame IDs in this section |
| `visual_content` | `str` | Combined LLM frame analyses |
| `audio_content` | `str` | Transcript text for this window |

---

### VideoProcessorUsage

```python
class VideoProcessorUsage(BaseModel)
```

| Field | Type | Description |
|-------|------|-------------|
| `frames_extracted` | `int` | Total frames sampled from video |
| `frames_kept` | `int` | Frames that passed deduplication |
| `frames_discarded` | `int` | Frames removed by deduplication |
| `analysis_input_tokens` | `int` | Prompt tokens for frame analysis |
| `analysis_output_tokens` | `int` | Completion tokens for frame analysis |
| `summary_input_tokens` | `int` | Prompt tokens for summary |
| `summary_output_tokens` | `int` | Completion tokens for summary |
| `audio_duration_seconds` | `float` | Duration of extracted audio |

**Properties**

- `total_analysis_tokens` — `analysis_input + analysis_output`
- `total_summary_tokens` — `summary_input + summary_output`
- `total_tokens` — Sum of all tokens

---

### VideoConfig

```python
class VideoConfig(BaseModel)
```

Convenience model bundling all pipeline hyperparameters (mirrors constructor params).

---

## Enums

### TranscriberBackend

```python
class TranscriberBackend(str, Enum)
```

| Value | Backend | Extra required |
|-------|---------|---------------|
| `"faster-whisper"` | faster-whisper lib (default) | `pipelines-video-whisper` |
| `"openai-whisper"` | openai-whisper lib | `pipelines-video-openai-whisper` |
| `"huggingface-local"` | HF transformers ASR | `transformers torch` |
| `"openai-api"` | OpenAI Whisper API | `openai` (core) |
| `"google-api"` | Google Cloud Speech-to-Text v2 | `google-cloud-speech` |
| `"huggingface-api"` | HuggingFace Inference API | `huggingface_hub` |
| `"groq-api"` | Groq Whisper (ultra-fast) | `groq` |
| `"deepgram-api"` | Deepgram Nova | `deepgram-sdk` |
| `"ollama"` | Self-hosted Ollama | `ollama` (core) |

**`transcriber_model` values by backend**

| Backend | Valid model values |
|---------|------------------|
| `faster-whisper` / `openai-whisper` | `"tiny"` `"base"` `"small"` `"medium"` `"large"` `"large-v2"` `"large-v3"` |
| `huggingface-local` / `huggingface-api` | Any HF model ID, e.g. `"openai/whisper-large-v3"` |
| `openai-api` | `"whisper-1"` |
| `google-api` | `"long"` `"short"` `"latest_long"` |
| `groq-api` | `"whisper-large-v3"` `"whisper-large-v3-turbo"` `"distil-whisper-large-v3-en"` |
| `deepgram-api` | `"nova-3"` `"nova-2"` `"enhanced"` `"base"` |
| `ollama` | Model name on your Ollama server, e.g. `"whisper"` |

---

### DeduplicationMethod

```python
class DeduplicationMethod(str, Enum)
```

| Value | Algorithm | Speed | Accuracy |
|-------|-----------|-------|---------|
| `"phash"` | Perceptual hash | Fast | Good for most videos |
| `"ssim"` | Structural Similarity | Slower | Higher accuracy |

---

### FrameAnalysisMode

```python
class FrameAnalysisMode(str, Enum)
```

| Value | Behaviour | Best for |
|-------|-----------|---------|
| `"individual"` | One LLM call per frame | High accuracy, whiteboard extraction |
| `"grid"` | Stitch N frames into a collage → one call | Cost reduction, dense videos |

---

## Exceptions

### VideoRateLimitExceededError

```python
class VideoRateLimitExceededError(RuntimeError)
```

Raised (or captured in `result.error` when `safe_mode=True`) when the
`rate_limiter` denies a pipeline request.

---

## See also

- [Video Processor Guide](../guide/pipelines/video_processor.md)
- [Pipelines overview](../guide/pipelines.md)
- [RAG guide](../guide/rag.md)
- [Telemetry guide](../guide/telemetry.md)
