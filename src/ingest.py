"""Video ingestion module for Active-Speaker Detection and Smart Video Reframing.

Handles video validation, metadata extraction, and generating downscaled
analysis frames. The original video is never modified here; it is only
opened for reading to extract information.
"""

import logging
from dataclasses import dataclass
from typing import Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Metadata extracted from a video file.

    Attributes:
        path: Absolute path to the video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames in the video.
        codec: FourCC codec string.
    """

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    codec: str


def validate_video(path: str) -> bool:
    """Check that the video file exists and can be opened by OpenCV.

    Args:
        path: File path to the video.

    Returns:
        True if the video can be opened, False otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return False

    cap.release()
    return True


def get_video_info(path: str) -> VideoInfo:
    """Extract metadata from the video file.

    Args:
        path: File path to the video.

    Returns:
        A VideoInfo instance with the video metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the video cannot be opened or has invalid metadata.
    """
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((codec_int >> 8 * i) & 0xFF) for i in range(4))

    cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions: {width}x{height}")
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")
    if total_frames <= 0:
        raise ValueError(f"Invalid frame count: {total_frames}")

    logger.info(
        "Video info: %dx%d @ %.2f fps, %d frames, codec=%s",
        width, height, fps, total_frames, codec,
    )

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        codec=codec,
    )


def iter_analysis_frames(
    path: str,
    target_width: int = 640,
    target_fps: float = 15.0,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield downscaled frames for face detection analysis.

    Reads the video and yields frames scaled down to the target width
    while preserving aspect ratio. Frames are yielded at approximately
    the target FPS by skipping frames from the original video.

    Args:
        path: File path to the video.
        target_width: Desired width for analysis frames.
        target_fps: Desired frame rate for analysis.

    Yields:
        Tuples of (frame_index, timestamp_seconds, downscaled_frame).
        The frame_index is the original frame number in the source video.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error("Cannot open video for analysis: %s", path)
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0

    # Calculate how many original frames to skip to reach target_fps
    skip_interval = max(1, round(original_fps / target_fps))

    scale_factor = None
    frame_idx = 0
    yielded_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_interval == 0:
            h, w = frame.shape[:2]
            if scale_factor is None:
                scale_factor = target_width / w

            new_width = target_width
            new_height = max(1, int(h * scale_factor))
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            timestamp = frame_idx / original_fps
            yield frame_idx, timestamp, resized
            yielded_count += 1

        frame_idx += 1

    cap.release()
    logger.info(
        "Analysis complete: %d frames yielded from %d original frames "
        "(skip_interval=%d, analysis_fps≈%.1f)",
        yielded_count, frame_idx, skip_interval,
        yielded_count / max(1, frame_idx / original_fps),
    )
