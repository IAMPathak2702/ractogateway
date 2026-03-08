"""Frame extraction and deduplication for VideoProcessorPipeline.

Extraction uses OpenCV (CPU-bound → ProcessPoolExecutor).
Deduplication supports two algorithms:
  - pHash  : perceptual hash via imagehash  (fast, default)
  - SSIM   : structural similarity via scikit-image  (more accurate)
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ._models import DeduplicationMethod, FrameEntry

# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------


def _require_cv2():  # type: ignore[return]
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for frame extraction. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


def _require_imagehash():  # type: ignore[return]
    try:
        import imagehash
        return imagehash
    except ImportError as exc:
        raise ImportError(
            "imagehash is required for pHash deduplication. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


def _require_pil():  # type: ignore[return]
    try:
        from PIL import Image
        return Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image processing. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


def _require_skimage():  # type: ignore[return]
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for SSIM deduplication. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


def _require_numpy():  # type: ignore[return]
    try:
        import numpy as np
        return np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for SSIM deduplication. "
            "Install with: pip install ractogateway[pipelines-video]"
        ) from exc


# ---------------------------------------------------------------------------
# Internal raw-frame dataclass (not part of public API)
# ---------------------------------------------------------------------------


@dataclass
class _RawFrame:
    frame_id: int
    timestamp: float  # seconds
    bgr_bytes: bytes  # raw OpenCV BGR array serialised via numpy tobytes()
    height: int
    width: int


# ---------------------------------------------------------------------------
# Frame extraction (runs in worker processes)
# ---------------------------------------------------------------------------


def _extract_worker(
    video_path: str,
    worker_idx: int,
    n_workers: int,
    target_fps: float,
    max_frames: int | None,
    frame_format: str,
) -> list[tuple[int, float, bytes]]:
    """Extract frames from a video file in a subprocess.

    Returns list of (frame_id, timestamp_seconds, image_bytes).
    Each worker handles every n_workers-th frame (stripe partition).
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native_fps / target_fps))
    encode_ext = ".jpg" if frame_format.upper() == "JPEG" else ".png"
    encode_params: list[int] = (
        [cv2.IMWRITE_JPEG_QUALITY, 85] if encode_ext == ".jpg" else []
    )

    results: list[tuple[int, float, bytes]] = []
    frame_idx = 0
    global_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0 and frame_idx // step % n_workers == worker_idx:
            timestamp = frame_idx / native_fps
            ok, buf = cv2.imencode(encode_ext, frame, encode_params)
            if ok:
                results.append((global_id, timestamp, buf.tobytes()))
            global_id += 1

        frame_idx += 1
        if max_frames and global_id >= max_frames:
            break

    cap.release()
    del np  # suppress unused import warning
    return results


def extract_frames(
    video_path: Path,
    *,
    fps: float = 1.0,
    max_frames: int | None = None,
    frame_format: str = "JPEG",
    max_process_workers: int = 4,
) -> list[tuple[int, float, bytes]]:
    """Extract frames from *video_path* at *fps* frames-per-second.

    Returns a list of ``(frame_id, timestamp_seconds, image_bytes)`` tuples
    sorted by frame_id.  Uses a ProcessPoolExecutor for speed.
    """
    _require_cv2()

    n_workers = min(max_process_workers, os.cpu_count() or 1)
    path_str = str(video_path)

    futures_map: dict = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for idx in range(n_workers):
            fut = pool.submit(
                _extract_worker,
                path_str,
                idx,
                n_workers,
                fps,
                max_frames,
                frame_format,
            )
            futures_map[fut] = idx

    all_frames: list[tuple[int, float, bytes]] = []
    for fut in as_completed(futures_map):
        all_frames.extend(fut.result())

    # Re-sort by frame_id (workers stripe, so ordering is interleaved)
    all_frames.sort(key=lambda t: t[0])

    # Enforce max_frames after merge
    if max_frames:
        all_frames = all_frames[:max_frames]

    return all_frames


def extract_frames_window(
    video_path: Path,
    *,
    fps: float = 1.0,
    max_frames: int | None = None,
    frame_format: str = "JPEG",
    start_time_seconds: float = 0.0,
    end_time_seconds: float | None = None,
) -> list[tuple[int, float, bytes]]:
    """Extract frames from a bounded time window of *video_path*.

    Timestamps in returned tuples are absolute to the original source video.
    This path is intentionally single-process because passive windows are small
    and this avoids process-pool overhead for short clips.
    """
    cv2 = _require_cv2()

    if start_time_seconds < 0:
        raise ValueError("start_time_seconds must be >= 0.")
    if end_time_seconds is not None and end_time_seconds < start_time_seconds:
        raise ValueError("end_time_seconds must be >= start_time_seconds.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native_fps / fps))
    encode_ext = ".jpg" if frame_format.upper() == "JPEG" else ".png"
    encode_params: list[int] = (
        [cv2.IMWRITE_JPEG_QUALITY, 85] if encode_ext == ".jpg" else []
    )

    start_frame = max(0, int(start_time_seconds * native_fps))
    end_frame = (
        int(end_time_seconds * native_fps)
        if end_time_seconds is not None
        else None
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    results: list[tuple[int, float, bytes]] = []
    frame_idx = start_frame
    global_id = 0

    while True:
        if end_frame is not None and frame_idx > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            timestamp = frame_idx / native_fps
            ok, buf = cv2.imencode(encode_ext, frame, encode_params)
            if ok:
                results.append((global_id, timestamp, buf.tobytes()))
                global_id += 1
                if max_frames and global_id >= max_frames:
                    break

        frame_idx += 1

    cap.release()
    return results


def load_frames_from_paths(
    frame_paths: list[Path],
    *,
    frame_format: str = "JPEG",
) -> list[tuple[int, float, bytes]]:
    """Load pre-extracted frame images from disk.

    Timestamps are inferred as sequential integers (0, 1, 2 …) since
    the user skipped the extraction step.
    """
    pil = _require_pil()
    results: list[tuple[int, float, bytes]] = []
    for idx, p in enumerate(frame_paths):
        img = pil.open(p).convert("RGB")
        buf = io.BytesIO()
        fmt = "JPEG" if frame_format.upper() == "JPEG" else "PNG"
        img.save(buf, format=fmt)
        results.append((idx, float(idx), buf.getvalue()))
    return results


# ---------------------------------------------------------------------------
# Similarity comparison
# ---------------------------------------------------------------------------


def _phash_similarity(img_bytes_a: bytes, img_bytes_b: bytes) -> float:
    """Return perceptual hash similarity (0-100 %)."""
    imagehash = _require_imagehash()
    pil = _require_pil()

    img_a = pil.open(io.BytesIO(img_bytes_a)).convert("RGB")
    img_b = pil.open(io.BytesIO(img_bytes_b)).convert("RGB")

    hash_a = imagehash.phash(img_a)
    hash_b = imagehash.phash(img_b)

    # pHash produces a 64-bit hash; max hamming distance = 64
    distance = hash_a - hash_b
    return (1.0 - distance / 64.0) * 100.0


def _phash_as_int(img_bytes: bytes) -> int:
    """Compute pHash and return as a plain int (picklable, O(1) comparison).

    The hash is a 64-bit perceptual hash encoded as a hex string by imagehash;
    we convert to int so Hamming-distance comparisons cost O(1) via XOR + popcount.
    """
    imagehash = _require_imagehash()
    pil = _require_pil()
    h = imagehash.phash(pil.open(io.BytesIO(img_bytes)).convert("RGB"))
    return int(str(h), 16)


def _hamming_similarity(hash_int_a: int, hash_int_b: int) -> float:
    """Hamming-based pHash similarity in % between two pre-computed hash ints.

    O(1) — XOR + popcount, no PIL or imagehash required after pre-computation.
    64-bit hash → max hamming distance = 64.
    """
    distance = bin(hash_int_a ^ hash_int_b).count("1")
    return (1.0 - distance / 64.0) * 100.0


def _hash_frame_tuple(frame_tuple: tuple[int, float, bytes]) -> int:
    """Top-level helper for :func:`deduplicate_frames_fast` (ThreadPoolExecutor map)."""
    return _phash_as_int(frame_tuple[2])


def _ssim_similarity(img_bytes_a: bytes, img_bytes_b: bytes) -> float:
    """Return SSIM structural similarity (0-100 %)."""
    structural_similarity = _require_skimage()
    np = _require_numpy()
    pil = _require_pil()

    img_a = np.array(pil.open(io.BytesIO(img_bytes_a)).convert("L"))
    img_b = np.array(pil.open(io.BytesIO(img_bytes_b)).convert("L"))

    # Resize to same shape if needed
    if img_a.shape != img_b.shape:
        pil_b = pil.open(io.BytesIO(img_bytes_b)).convert("L").resize(
            (img_a.shape[1], img_a.shape[0])
        )
        img_b = np.array(pil_b)

    score: float = structural_similarity(img_a, img_b, data_range=255)
    return score * 100.0


def compute_similarity(
    img_bytes_a: bytes,
    img_bytes_b: bytes,
    method: DeduplicationMethod,
) -> float:
    """Compute similarity % between two images using *method*."""
    if method == DeduplicationMethod.PHASH:
        return _phash_similarity(img_bytes_a, img_bytes_b)
    return _ssim_similarity(img_bytes_a, img_bytes_b)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_frames(
    raw_frames: list[tuple[int, float, bytes]],
    *,
    similarity_threshold: float,
    method: DeduplicationMethod,
) -> list[FrameEntry]:
    """Deduplicate *raw_frames* using the chosen similarity method.

    A frame is **discarded** when its similarity to the last *kept* frame
    is >= *similarity_threshold* (e.g. 90 %).

    Parameters
    ----------
    raw_frames:
        List of ``(frame_id, timestamp, image_bytes)`` as returned by
        :func:`extract_frames`.
    similarity_threshold:
        Percentage threshold (0-100).  Frames >= this are dropped.
    method:
        :class:`DeduplicationMethod.PHASH` or :class:`DeduplicationMethod.SSIM`.

    Returns
    -------
    list[FrameEntry]
        All frames with ``kept`` flag set appropriately.
    """
    entries: list[FrameEntry] = []
    last_kept_bytes: bytes | None = None

    for frame_id, timestamp, img_bytes in raw_frames:
        if last_kept_bytes is None:
            # Always keep the first frame
            entries.append(
                FrameEntry(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    similarity_to_prev=None,
                    kept=True,
                    image_data=img_bytes,
                )
            )
            last_kept_bytes = img_bytes
            continue

        sim = compute_similarity(img_bytes, last_kept_bytes, method)
        keep = sim < similarity_threshold

        entries.append(
            FrameEntry(
                frame_id=frame_id,
                timestamp=timestamp,
                similarity_to_prev=round(sim, 2),
                kept=keep,
                image_data=img_bytes if keep else None,
            )
        )
        if keep:
            last_kept_bytes = img_bytes

    return entries


def deduplicate_frames_fast(
    raw_frames: list[tuple[int, float, bytes]],
    *,
    similarity_threshold: float,
    method: DeduplicationMethod,
    max_hash_workers: int = 4,
) -> list[FrameEntry]:
    """DSA-optimised deduplication for the async pipeline.

    **pHash fast path** — two-stage algorithm:

    1. Pre-compute all perceptual hashes in **parallel** via
       :class:`~concurrent.futures.ThreadPoolExecutor` (O(n/k) wall-time
       with k workers instead of O(n) sequential).
    2. Sequential dedup using O(1) integer XOR + popcount (Hamming distance),
       bypassing PIL/imagehash entirely after the pre-computation stage.

    This converts what was O(n) sequential CPU work into O(n/k) parallel work
    plus O(n) trivial bit-arithmetic — a significant speedup for large frame
    sets (20-200+ frames).

    **SSIM path** — falls back to :func:`deduplicate_frames` because SSIM
    comparison is inherently sequential (each frame compared to the last *kept*
    frame, which is only known after the previous step).
    """
    if not raw_frames:
        return []

    if method == DeduplicationMethod.SSIM:
        # SSIM requires full image pixels at each step; sequential fallback.
        return deduplicate_frames(
            raw_frames,
            similarity_threshold=similarity_threshold,
            method=method,
        )

    # ── pHash fast path ─────────────────────────────────────────────────────
    _require_imagehash()  # ensure dep available before spawning threads
    n_workers = min(max_hash_workers, len(raw_frames))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        hashes: list[int] = list(pool.map(_hash_frame_tuple, raw_frames))

    entries: list[FrameEntry] = []
    last_kept_hash: int | None = None

    for (frame_id, timestamp, img_bytes), h in zip(raw_frames, hashes, strict=False):
        if last_kept_hash is None:
            entries.append(
                FrameEntry(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    similarity_to_prev=None,
                    kept=True,
                    image_data=img_bytes,
                )
            )
            last_kept_hash = h
            continue

        sim = _hamming_similarity(h, last_kept_hash)
        keep = sim < similarity_threshold
        entries.append(
            FrameEntry(
                frame_id=frame_id,
                timestamp=timestamp,
                similarity_to_prev=round(sim, 2),
                kept=keep,
                image_data=img_bytes if keep else None,
            )
        )
        if keep:
            last_kept_hash = h

    return entries
