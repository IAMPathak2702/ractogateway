"""Unified video source resolver for VideoProcessorPipeline.

Accepts five input types:
  - str / Path   : local file path  -OR-  http/https URL  -OR-  YouTube URL
  - bytes        : raw video buffer  →  written to a temp file
  - list[str|Path]: pre-extracted frame image paths (skips extraction step)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------


def _require_httpx() -> None:
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "httpx is required to download video URLs. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


def _require_ytdlp() -> None:
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "yt-dlp is required to download YouTube videos. "
            "Install with: pip install ractogateway[pipelines-video-yt]"
        ) from exc


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------


def _is_youtube_url(text: str) -> bool:
    return "youtube.com" in text or "youtu.be" in text


def _is_http_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


def _is_frame_list(source: object) -> bool:
    """True if source is a non-empty list (of frame paths)."""
    return isinstance(source, list)


def resolve_video_source(
    source: str | Path | bytes | list,
) -> tuple[Path | None, list[Path] | None]:
    """Resolve any supported video source into a concrete path or frame list.

    Returns
    -------
    (video_path, None)
        For file-path / URL / bytes inputs — caller should use video_path with OpenCV.
    (None, frame_paths)
        For pre-extracted frame lists — caller skips OpenCV extraction entirely.
    """
    # ── Pre-extracted frame list ─────────────────────────────────────────
    if _is_frame_list(source):
        frame_paths = [Path(p) for p in source]  # type: ignore[union-attr]
        missing = [p for p in frame_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Pre-extracted frame files not found: {[str(p) for p in missing]}"
            )
        return None, frame_paths

    # ── Raw bytes buffer ─────────────────────────────────────────────────
    if isinstance(source, (bytes, bytearray)):
        return _bytes_to_tempfile(bytes(source)), None

    # ── String or Path ───────────────────────────────────────────────────
    text = str(source)

    if _is_youtube_url(text):
        return _download_youtube(text), None

    if _is_http_url(text):
        return _download_http(text), None

    # Local file path
    local = Path(text)
    if not local.exists():
        raise FileNotFoundError(f"Video file not found: {local}")
    return local, None


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------


def _bytes_to_tempfile(data: bytes) -> Path:
    """Write raw video bytes to a named temp file and return its path."""
    suffix = ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ractovideo_")
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return Path(tmp_path)


def _download_http(url: str) -> Path:
    """Download a video from an HTTP/HTTPS URL to a temp file."""
    _require_httpx()
    import httpx

    suffix = Path(url.split("?", maxsplit=1)[0]).suffix or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ractovideo_http_")
    os.close(fd)
    dest = Path(tmp_path)

    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                fh.write(chunk)

    return dest


def _download_youtube(url: str) -> Path:
    """Download a YouTube video (best mp4) to a temp directory via yt-dlp."""
    _require_ytdlp()
    import yt_dlp

    tmp_dir = tempfile.mkdtemp(prefix="ractovideo_yt_")
    output_template = str(Path(tmp_dir) / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "video")  # type: ignore[union-attr]

    # Find the downloaded file
    for ext in ("mp4", "mkv", "webm"):
        candidate = Path(tmp_dir) / f"{video_id}.{ext}"
        if candidate.exists():
            return candidate

    # Fallback: first file in tmp_dir
    files = list(Path(tmp_dir).iterdir())
    if files:
        return files[0]

    raise RuntimeError(f"yt-dlp downloaded nothing for URL: {url}")
