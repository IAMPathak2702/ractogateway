"""OCR backends for PageIndexRAG.

Each backend converts raw image bytes (PNG/JPEG) into extracted text.
All backends follow the same interface so they are interchangeable.

Available backends
------------------
- :class:`TesseractOcrBackend`   — free, offline, ``pytesseract`` wrapper
- :class:`EasyOcrBackend`        — deep-learning, 80+ languages, offline
- :class:`GoogleVisionBackend`   — Google Cloud Vision API
- :class:`GoogleDocumentAIBackend` — Google Document AI (tables, forms)
- :class:`AWSTextractBackend`    — AWS Textract (forms, tables, key-value)
- :class:`AzureDocumentIntelligenceBackend` — Azure Form Recognizer v4

Quick start::

    from ractogateway.rag.page_index import PageIndexRAG
    from ractogateway.rag.page_index._ocr import TesseractOcrBackend

    rag = PageIndexRAG(llm_kit=kit, ocr_backend=TesseractOcrBackend())
    rag.ingest("scanned_report.pdf")   # OCR fallback auto-triggered
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseOcrBackend(ABC):
    """Abstract base class for OCR backends.

    Implementors must provide :meth:`extract_text`.
    An async default is provided via :meth:`aextract_text` that offloads
    the synchronous call to a thread-pool executor.
    """

    @abstractmethod
    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Convert *image_bytes* to plain text.

        Parameters
        ----------
        image_bytes:
            Raw image data (PNG, JPEG, TIFF, …).
        mime_type:
            MIME type hint used by cloud APIs (default ``"image/png"``).

        Returns
        -------
        str
            Extracted text, or an empty string if nothing was recognised.
        """

    async def aextract_text(
        self, image_bytes: bytes, mime_type: str = "image/png"
    ) -> str:
        """Async variant of :meth:`extract_text` (thread-pool offload)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_text, image_bytes, mime_type)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "pillow is required for image-based OCR. "
            "Install it with:  pip install ractogateway[rag-image]"
        ) from exc
    return Image


def _bytes_to_pil(image_bytes: bytes) -> Any:
    import io

    image_module = _require_pillow()
    return image_module.open(io.BytesIO(image_bytes))


def _load_service_account_credentials(service_account: Any, path: str) -> Any:
    credentials_factory = cast(
        "Callable[[str], Any]",
        service_account.Credentials.from_service_account_file,
    )
    return credentials_factory(path)


# ---------------------------------------------------------------------------
# Tesseract (offline, free)
# ---------------------------------------------------------------------------


class TesseractOcrBackend(BaseOcrBackend):
    """OCR via `Tesseract <https://github.com/tesseract-ocr/tesseract>`_.

    Requires ``pytesseract`` and a working Tesseract installation.
    Install with::

        pip install ractogateway[rag-ocr-tesseract]
        # Also install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki

    Parameters
    ----------
    lang:
        Tesseract language string, e.g. ``"eng"`` (default), ``"eng+deu"``.
    config:
        Extra Tesseract config flags, e.g. ``"--psm 6"``.
    confidence_threshold:
        Pages where the mean word confidence is below this value (0-100)
        are flagged in the returned metadata; text is still returned.
        Set to ``0`` to disable filtering.
    """

    def __init__(
        self,
        lang: str = "eng",
        config: str = "",
        confidence_threshold: float = 40.0,
    ) -> None:
        self._lang = lang
        self._config = config
        self._confidence_threshold = confidence_threshold

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "pytesseract is required for TesseractOcrBackend. "
                "Install it with:  pip install ractogateway[rag-ocr-tesseract]"
            ) from exc
        img = _bytes_to_pil(image_bytes)
        return str(
            pytesseract.image_to_string(img, lang=self._lang, config=self._config)
        )

    def extract_with_confidence(self, image_bytes: bytes) -> tuple[str, float]:
        """Return ``(text, mean_confidence)`` for confidence-aware ingestion."""
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "pytesseract is required for TesseractOcrBackend. "
                "Install it with:  pip install ractogateway[rag-ocr-tesseract]"
            ) from exc

        img = _bytes_to_pil(image_bytes)
        data = pytesseract.image_to_data(
            img, lang=self._lang, config=self._config, output_type=pytesseract.Output.DATAFRAME
        )
        words = data[data["conf"] >= 0]
        mean_conf: float = float(words["conf"].mean()) if not words.empty else 0.0
        text = " ".join(words["text"].dropna().astype(str).str.strip().loc[words["conf"] >= 0])
        return text, mean_conf


# ---------------------------------------------------------------------------
# EasyOCR (offline, deep-learning)
# ---------------------------------------------------------------------------


class EasyOcrBackend(BaseOcrBackend):
    """OCR via `EasyOCR <https://github.com/JaidedAI/EasyOCR>`_.

    Deep-learning model; no cloud API required.  Supports 80+ languages.
    Install with::

        pip install ractogateway[rag-ocr-easy]

    Parameters
    ----------
    languages:
        List of language codes, e.g. ``["en"]`` (default) or ``["en", "de"]``.
    gpu:
        Use CUDA GPU if available (default ``False`` for broad compatibility).
    """

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._reader: Any = None  # lazy init

    def _get_reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError(
                    "easyocr is required for EasyOcrBackend. "
                    "Install it with:  pip install ractogateway[rag-ocr-easy]"
                ) from exc
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        import io

        reader = self._get_reader()
        results = reader.readtext(io.BytesIO(image_bytes), detail=0)
        return "\n".join(str(r) for r in results)


# ---------------------------------------------------------------------------
# Google Cloud Vision
# ---------------------------------------------------------------------------


class GoogleVisionBackend(BaseOcrBackend):
    """OCR via Google Cloud Vision API (``DOCUMENT_TEXT_DETECTION``).

    Install with::

        pip install ractogateway[rag-ocr-google]

    Parameters
    ----------
    credentials_path:
        Path to a service-account JSON key file.  If ``None`` the SDK uses
        Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``).
    """

    def __init__(self, credentials_path: str | None = None) -> None:
        self._credentials_path = credentials_path
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import vision
                from google.oauth2 import service_account
            except ImportError as exc:
                raise ImportError(
                    "google-cloud-vision is required for GoogleVisionBackend. "
                    "Install it with:  pip install ractogateway[rag-ocr-google]"
                ) from exc
            if self._credentials_path:
                creds = _load_service_account_credentials(
                    service_account,
                    self._credentials_path,
                )
                self._client = vision.ImageAnnotatorClient(credentials=creds)
            else:
                self._client = vision.ImageAnnotatorClient()
        return self._client

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        try:
            from google.cloud import vision
        except ImportError as exc:
            raise ImportError(
                "google-cloud-vision is required for GoogleVisionBackend. "
                "Install it with:  pip install ractogateway[rag-ocr-google]"
            ) from exc
        client = self._get_client()
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")
        return response.full_text_annotation.text or ""


# ---------------------------------------------------------------------------
# Google Document AI
# ---------------------------------------------------------------------------


class GoogleDocumentAIBackend(BaseOcrBackend):
    """OCR via `Google Document AI <https://cloud.google.com/document-ai>`_.

    Best for structured documents: tables, forms, invoices, contracts.
    Install with::

        pip install ractogateway[rag-ocr-google]

    Parameters
    ----------
    project_id:
        GCP project ID.
    processor_id:
        Document AI processor ID (e.g. an OCR or Form Parser processor).
    location:
        Processor region, usually ``"us"`` or ``"eu"`` (default ``"us"``).
    credentials_path:
        Optional path to a service-account JSON key.
    """

    def __init__(
        self,
        project_id: str,
        processor_id: str,
        location: str = "us",
        credentials_path: str | None = None,
    ) -> None:
        self._project_id = project_id
        self._processor_id = processor_id
        self._location = location
        self._credentials_path = credentials_path
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google.api_core.client_options import ClientOptions
                from google.cloud import documentai
                from google.oauth2 import service_account
            except ImportError as exc:
                raise ImportError(
                    "google-cloud-documentai is required for GoogleDocumentAIBackend. "
                    "Install it with:  pip install ractogateway[rag-ocr-google]"
                ) from exc
            opts = ClientOptions(
                api_endpoint=f"{self._location}-documentai.googleapis.com"
            )
            if self._credentials_path:
                creds = _load_service_account_credentials(
                    service_account,
                    self._credentials_path,
                )
                self._client = documentai.DocumentProcessorServiceClient(
                    client_options=opts, credentials=creds
                )
            else:
                self._client = documentai.DocumentProcessorServiceClient(
                    client_options=opts
                )
        return self._client

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        try:
            from google.cloud import documentai
        except ImportError as exc:
            raise ImportError(
                "google-cloud-documentai is required for GoogleDocumentAIBackend. "
                "Install it with:  pip install ractogateway[rag-ocr-google]"
            ) from exc
        client = self._get_client()
        name = (
            f"projects/{self._project_id}/locations/{self._location}"
            f"/processors/{self._processor_id}"
        )
        raw_doc = documentai.RawDocument(content=image_bytes, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
        result = client.process_document(request=request)
        return result.document.text or ""


# ---------------------------------------------------------------------------
# AWS Textract
# ---------------------------------------------------------------------------


class AWSTextractBackend(BaseOcrBackend):
    """OCR via `AWS Textract <https://aws.amazon.com/textract/>`_.

    Best for forms and tables; uses ``DetectDocumentText`` for plain text.
    Install with::

        pip install ractogateway[rag-ocr-aws]

    Parameters
    ----------
    region_name:
        AWS region (default ``"us-east-1"``).
    aws_access_key_id / aws_secret_access_key:
        Optional explicit credentials; if omitted, boto3 uses the standard
        credential chain (env vars, ``~/.aws/credentials``, IAM role, etc.).
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._region = region_name
        self._key_id = aws_access_key_id
        self._secret = aws_secret_access_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for AWSTextractBackend. "
                    "Install it with:  pip install ractogateway[rag-ocr-aws]"
                ) from exc
            kwargs: dict[str, Any] = {"region_name": self._region}
            if self._key_id:
                kwargs["aws_access_key_id"] = self._key_id
            if self._secret:
                kwargs["aws_secret_access_key"] = self._secret
            self._client = boto3.client("textract", **kwargs)
        return self._client

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        client = self._get_client()
        response = client.detect_document_text(Document={"Bytes": image_bytes})
        lines = [
            block["Text"]
            for block in response.get("Blocks", [])
            if block["BlockType"] == "LINE"
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Azure Document Intelligence (Form Recognizer v4)
# ---------------------------------------------------------------------------


class AzureDocumentIntelligenceBackend(BaseOcrBackend):
    """OCR via `Azure Document Intelligence
    <https://learn.microsoft.com/azure/ai-services/document-intelligence/>`_.

    Previously called Azure Form Recognizer.  The ``prebuilt-read`` model
    is used by default; swap in ``prebuilt-document`` for richer extraction.
    Install with::

        pip install ractogateway[rag-ocr-azure]

    Parameters
    ----------
    endpoint:
        Azure resource endpoint URL.
    api_key:
        Azure resource API key.
    model_id:
        Document Intelligence model to use (default ``"prebuilt-read"``).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_id: str = "prebuilt-read",
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model_id = model_id
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from azure.ai.documentintelligence import DocumentIntelligenceClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError as exc:
                raise ImportError(
                    "azure-ai-documentintelligence is required for "
                    "AzureDocumentIntelligenceBackend. "
                    "Install it with:  pip install ractogateway[rag-ocr-azure]"
                ) from exc
            from azure.core.credentials import AzureKeyCredential

            self._client = DocumentIntelligenceClient(
                endpoint=self._endpoint,
                credential=AzureKeyCredential(self._api_key),
            )
        return self._client

    def extract_text(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        try:
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        except ImportError as exc:
            raise ImportError(
                "azure-ai-documentintelligence is required for "
                "AzureDocumentIntelligenceBackend. "
                "Install it with:  pip install ractogateway[rag-ocr-azure]"
            ) from exc
        client = self._get_client()
        poller = client.begin_analyze_document(
            self._model_id,
            AnalyzeDocumentRequest(bytes_source=image_bytes),
        )
        result = poller.result()
        paragraphs = [p.content for p in (result.paragraphs or []) if p.content]
        return "\n".join(paragraphs)
