"""Audio extraction and transcription for VideoProcessorPipeline.

Supports 9 transcription backends via a pluggable BaseTranscriber:

  Local / open-source:
    faster-whisper     — faster-whisper library (default)
    openai-whisper     — openai-whisper library
    huggingface-local  — HuggingFace transformers ASR pipeline

  Cloud APIs:
    openai-api         — OpenAI Whisper API
    google-api         — Google Cloud Speech-to-Text v2
    huggingface-api    — HuggingFace Inference API
    groq-api           — Groq Whisper (ultra-fast cloud)
    deepgram-api       — Deepgram Nova

  Self-hosted:
    ollama             — Ollama server (audio-capable models)
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from ._models import TranscriberBackend, TranscriptSegment

# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------


def _require_ffmpeg():  # type: ignore[return]
    try:
        import ffmpeg  # noqa: PLC0415
        return ffmpeg
    except ImportError as exc:
        raise ImportError(
            "ffmpeg-python is required for audio extraction. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


def _require_faster_whisper():  # type: ignore[return]
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
        return WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for this backend. "
            "Install with: pip install ractogateway[pipelines-video-whisper]"
        ) from exc


def _require_openai_whisper():  # type: ignore[return]
    try:
        import whisper  # noqa: PLC0415
        return whisper
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is required for this backend. "
            "Install with: pip install ractogateway[pipelines-video-openai-whisper]"
        ) from exc


def _require_transformers():  # type: ignore[return]
    try:
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        return hf_pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the huggingface-local backend. "
            "Install with: pip install transformers torch"
        ) from exc


def _require_openai_client():  # type: ignore[return]
    try:
        from openai import OpenAI  # noqa: PLC0415
        return OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required for the openai-api transcription backend. "
            "Install with: pip install ractogateway[openai]"
        ) from exc


def _require_google_speech():  # type: ignore[return]
    try:
        from google.cloud import speech  # noqa: PLC0415
        return speech
    except ImportError as exc:
        raise ImportError(
            "google-cloud-speech is required for the google-api backend. "
            "Install with: pip install google-cloud-speech"
        ) from exc


def _require_huggingface_hub():  # type: ignore[return]
    try:
        from huggingface_hub import InferenceClient  # noqa: PLC0415
        return InferenceClient
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for the huggingface-api backend. "
            "Install with: pip install ractogateway[huggingface]"
        ) from exc


def _require_groq():  # type: ignore[return]
    try:
        from groq import Groq  # noqa: PLC0415
        return Groq
    except ImportError as exc:
        raise ImportError(
            "groq is required for the groq-api transcription backend. "
            "Install with: pip install groq"
        ) from exc


def _require_deepgram():  # type: ignore[return]
    try:
        from deepgram import DeepgramClient  # noqa: PLC0415
        return DeepgramClient
    except ImportError as exc:
        raise ImportError(
            "deepgram-sdk is required for the deepgram-api backend. "
            "Install with: pip install deepgram-sdk"
        ) from exc


def _require_ollama():  # type: ignore[return]
    try:
        import ollama  # noqa: PLC0415
        return ollama
    except ImportError as exc:
        raise ImportError(
            "ollama is required for the ollama transcription backend. "
            "Install with: pip install ractogateway[ollama]"
        ) from exc


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio(video_path: Path) -> Path:
    """Extract audio from *video_path* to a WAV temp file via ffmpeg-python."""
    ffmpeg = _require_ffmpeg()

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="ractoaudio_")
    os.close(fd)
    audio_path = Path(tmp_path)

    (
        ffmpeg
        .input(str(video_path))
        .output(
            str(audio_path),
            format="wav",
            acodec="pcm_s16le",
            ac=1,       # mono
            ar="16000",  # 16 kHz — optimal for Whisper
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path


def get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ffmpeg probe."""
    ffmpeg = _require_ffmpeg()
    probe = ffmpeg.probe(str(audio_path))
    return float(probe["format"]["duration"])


# ---------------------------------------------------------------------------
# Frame ↔ transcript alignment
# ---------------------------------------------------------------------------


def align_frames_to_transcript(
    frames: list,  # list[FrameEntry]
    segments: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Assign frame IDs to transcript segments by timestamp overlap."""
    updated: list[TranscriptSegment] = []
    for seg in segments:
        ids = [
            f.frame_id
            for f in frames
            if f.kept and seg.start <= f.timestamp <= seg.end
        ]
        updated.append(seg.model_copy(update={"frame_ids": ids}))
    return updated


# ---------------------------------------------------------------------------
# Abstract base transcriber
# ---------------------------------------------------------------------------


class BaseTranscriber(ABC):
    """Abstract interface for all transcription backends."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
    ) -> list[TranscriptSegment]:
        """Transcribe audio file and return time-stamped segments."""
        ...


# ---------------------------------------------------------------------------
# Local backends
# ---------------------------------------------------------------------------


class FasterWhisperTranscriber(BaseTranscriber):
    """Local transcription using the faster-whisper library."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is None:
            whisper_cls = _require_faster_whisper()
            self._model = whisper_cls(self._model_size, device="auto", compute_type="auto")

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        self._load()
        kwargs: dict = {}
        if language:
            kwargs["language"] = language
        segments_iter, _ = self._model.transcribe(str(audio_path), **kwargs)
        return [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in segments_iter
        ]


class OpenAIWhisperTranscriber(BaseTranscriber):
    """Local transcription using the openai-whisper library."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            whisper = _require_openai_whisper()
            self._model = whisper.load_model(self._model_size)

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        self._load()
        kwargs: dict = {"verbose": False}
        if language:
            kwargs["language"] = language
        result = self._model.transcribe(str(audio_path), **kwargs)
        return [
            TranscriptSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
            )
            for seg in result.get("segments", [])
        ]


class HuggingFaceLocalTranscriber(BaseTranscriber):
    """Local ASR transcription via HuggingFace transformers pipeline."""

    def __init__(self, model_id: str = "openai/whisper-base") -> None:
        self._model_id = model_id
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is None:
            hf_pipeline = _require_transformers()
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=self._model_id,
                return_timestamps=True,
            )

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        self._load()
        gen_kwargs: dict = {}
        if language:
            gen_kwargs["language"] = language
        result = self._pipe(str(audio_path), generate_kwargs=gen_kwargs)
        segments: list[TranscriptSegment] = []
        chunks = result.get("chunks", [])  # type: ignore[union-attr]
        if chunks:
            for chunk in chunks:
                ts = chunk.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
                segments.append(
                    TranscriptSegment(
                        start=float(ts[0] or 0.0),
                        end=float(ts[1] or 0.0),
                        text=chunk["text"].strip(),
                    )
                )
        else:
            text = (result.get("text") or "").strip()  # type: ignore[union-attr]
            segments.append(TranscriptSegment(start=0.0, end=0.0, text=text))
        return segments


# ---------------------------------------------------------------------------
# Cloud API backends
# ---------------------------------------------------------------------------


class OpenAIAPITranscriber(BaseTranscriber):
    """Cloud transcription via OpenAI Whisper API."""

    def __init__(self, model: str = "whisper-1", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        openai_cls = _require_openai_client()
        client = openai_cls(api_key=self._api_key)
        kwargs: dict = {"model": self._model, "response_format": "verbose_json"}
        if language:
            kwargs["language"] = language
        with audio_path.open("rb") as f:
            response = client.audio.transcriptions.create(file=f, **kwargs)
        return [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=seg.get("text", "").strip(),
            )
            for seg in (response.segments or [])  # type: ignore[union-attr]
        ]


class GoogleAPITranscriber(BaseTranscriber):
    """Cloud transcription via Google Cloud Speech-to-Text v2."""

    def __init__(
        self,
        model: str = "long",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        # Google uses Application Default Credentials; api_key unused here
        _ = api_key

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        speech = _require_google_speech()
        client = speech.SpeechClient()

        audio_bytes = audio_path.read_bytes()
        audio = speech.RecognitionAudio(content=audio_bytes)

        lang_code = language or "en-US"
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=lang_code,
            model=self._model,
            enable_word_time_offsets=True,
        )
        response = client.recognize(config=config, audio=audio)

        segments: list[TranscriptSegment] = []
        for result in response.results:
            alt = result.alternatives[0]
            words = alt.words
            if words:
                start = words[0].start_time.total_seconds()
                end = words[-1].end_time.total_seconds()
            else:
                start = end = 0.0
            segments.append(
                TranscriptSegment(start=start, end=end, text=alt.transcript.strip())
            )
        return segments


class HuggingFaceAPITranscriber(BaseTranscriber):
    """Cloud transcription via HuggingFace Inference API."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        api_key: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY", "")

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        inference_client_cls = _require_huggingface_hub()
        client = inference_client_cls(token=self._api_key)
        result = client.automatic_speech_recognition(
            audio=audio_path.read_bytes(),
            model=self._model_id,
        )
        text = (result.text if hasattr(result, "text") else str(result)).strip()
        return [TranscriptSegment(start=0.0, end=0.0, text=text)]


class GroqTranscriber(BaseTranscriber):
    """Cloud transcription via Groq Whisper API (ultra-fast)."""

    def __init__(
        self,
        model: str = "whisper-large-v3",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        groq_cls = _require_groq()
        client = groq_cls(api_key=self._api_key)
        kwargs: dict = {"model": self._model, "response_format": "verbose_json"}
        if language:
            kwargs["language"] = language
        with audio_path.open("rb") as f:
            response = client.audio.transcriptions.create(file=f, **kwargs)
        segs = getattr(response, "segments", None) or []
        if segs:
            return [
                TranscriptSegment(
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    text=s.get("text", "").strip(),
                )
                for s in segs
            ]
        text = getattr(response, "text", "") or ""
        return [TranscriptSegment(start=0.0, end=0.0, text=text.strip())]


class DeepgramTranscriber(BaseTranscriber):
    """Cloud transcription via Deepgram Nova."""

    def __init__(
        self,
        model: str = "nova-3",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        deepgram_cls = _require_deepgram()
        from deepgram import PrerecordedOptions  # noqa: PLC0415

        client = deepgram_cls(api_key=self._api_key)
        options = PrerecordedOptions(
            model=self._model,
            smart_format=True,
            utterances=True,
            language=language or "en",
        )
        with audio_path.open("rb") as f:
            buffer_data = f.read()
        source = {"buffer": buffer_data}
        response = client.listen.prerecorded.v("1").transcribe_file(source, options)

        utterances = (
            response.results.utterances  # type: ignore[union-attr]
            if response.results  # type: ignore[union-attr]
            else []
        )
        if utterances:
            return [
                TranscriptSegment(
                    start=float(u.start),
                    end=float(u.end),
                    text=u.transcript.strip(),
                )
                for u in utterances
            ]
        # Fallback to channel transcript
        text = ""
        with contextlib.suppress(Exception):
            text = (
                response.results.channels[0].alternatives[0].transcript  # type: ignore[index]
            )
        return [TranscriptSegment(start=0.0, end=0.0, text=text.strip())]


class OllamaTranscriber(BaseTranscriber):
    """Self-hosted transcription via Ollama server (audio-capable models)."""

    def __init__(
        self,
        model: str = "whisper",
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def transcribe(self, audio_path: Path, language: str | None) -> list[TranscriptSegment]:
        ollama = _require_ollama()
        import base64  # noqa: PLC0415

        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
        prompt = "Transcribe the audio."
        if language:
            prompt += f" Language: {language}."

        client = ollama.Client(host=self._base_url)
        response = client.generate(
            model=self._model,
            prompt=prompt,
            images=[audio_b64],
        )
        text = (response.get("response") or "").strip()
        return [TranscriptSegment(start=0.0, end=0.0, text=text)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_transcriber(
    backend: TranscriberBackend,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> BaseTranscriber:
    """Return the concrete :class:`BaseTranscriber` for the given *backend*."""
    if backend == TranscriberBackend.FASTER_WHISPER:
        return FasterWhisperTranscriber(model_size=model)
    if backend == TranscriberBackend.OPENAI_WHISPER:
        return OpenAIWhisperTranscriber(model_size=model)
    if backend == TranscriberBackend.HUGGINGFACE_LOCAL:
        return HuggingFaceLocalTranscriber(model_id=model)
    if backend == TranscriberBackend.OPENAI_API:
        return OpenAIAPITranscriber(model=model, api_key=api_key)
    if backend == TranscriberBackend.GOOGLE_API:
        return GoogleAPITranscriber(model=model, api_key=api_key)
    if backend == TranscriberBackend.HUGGINGFACE_API:
        return HuggingFaceAPITranscriber(model_id=model, api_key=api_key)
    if backend == TranscriberBackend.GROQ_API:
        return GroqTranscriber(model=model, api_key=api_key)
    if backend == TranscriberBackend.DEEPGRAM_API:
        return DeepgramTranscriber(model=model, api_key=api_key)
    if backend == TranscriberBackend.OLLAMA:
        return OllamaTranscriber(model=model, base_url=base_url)
    raise ValueError(f"Unknown transcriber backend: {backend!r}")
