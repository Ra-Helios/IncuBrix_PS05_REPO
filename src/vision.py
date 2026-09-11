"""Computer vision module using MediaPipe Face Landmarker.

Provides face detection, facial landmark extraction, face bounding box
calculation, and simple mouth/lip metrics. All processing is CPU-only.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("MediaPipe not installed. Face detection unavailable.")


@dataclass
class FaceDetection:
    """A single face detection with bounding box, landmarks, and lip metric.

    Attributes:
        bbox_x1: Left edge of the bounding box (in analysis frame coords).
        bbox_y1: Top edge of the bounding box.
        bbox_x2: Right edge of the bounding box.
        bbox_y2: Bottom edge of the bounding box.
        center_x: Horizontal center of the face.
        center_y: Vertical center of the face.
        face_width: Width of the bounding box.
        face_height: Height of the bounding box.
        confidence: Detection confidence score (0-1).
        landmarks: List of (x, y) tuples for selected face landmarks.
        mouth_opening_ratio: Normalized ratio indicating mouth openness.
        mouth_aperture: Alias for mouth_opening_ratio (height / width).
        timestamp_ms: Video timestamp (ms) of the analyzed frame.
        face_visibility: Fraction of face landmarks strictly inside the
            frame (0-1). A fully visible face is 1.0.
        landmark_quality: Overall landmark usability in [0,1] (a face that is
            tiny, close to the edge, or turned away scores lower).
        mouth_visibility: Fraction of lip landmarks strictly inside (0-1).
        size_quality: How large the face is relative to the frame (0-1).
        head_pose_quality: Frontalness heuristic in [0,1] derived from the
            nose/cheek asymmetry (no pose model is used).
        visual_quality: Combined usability = min of visibility, size, pose
            and mouth components (explainable "weakest link").
    """

    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    center_x: float
    center_y: float
    face_width: float
    face_height: float
    confidence: float
    landmarks: List[Tuple[float, float]] = field(default_factory=list)
    mouth_opening_ratio: float = 0.0
    mouth_aperture: float = 0.0
    timestamp_ms: Optional[float] = None
    face_visibility: float = 1.0
    landmark_quality: float = 1.0
    mouth_visibility: float = 1.0
    size_quality: float = 1.0
    head_pose_quality: float = 1.0
    visual_quality: float = 1.0


class FaceDetector:
    """MediaPipe-based face detector running on CPU.

    Uses MediaPipe Face Landmarker to detect faces and extract 478 facial
    landmarks. From these landmarks, it computes a simple mouth opening ratio.

    Attributes:
        confidence_threshold: Minimum detection confidence to accept a face.
    """

    # MediaPipe lip landmarks (upper and lower lip inner points)
    UPPER_LIP_TOP = 13
    LOWER_LIP_BOTTOM = 14
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # Landmarks used for the explainable head-pose (frontalness) heuristic.
    NOSE_TIP = 1
    EYE_OUTER = (33, 263)

    # Lip landmarks (outer + inner) used for mouth visibility.
    LIP_LANDMARKS = (
        0, 11, 12, 13, 14,
        61, 62, 63, 64, 65, 66, 67,
        146, 152, 155, 157, 158, 159, 160, 161,
        173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
        291, 292, 293, 294, 295, 296, 297,
        308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320,
        321, 322, 375, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386,
        387, 388, 402, 405, 407,
    )

    def __init__(self, confidence_threshold: float = 0.5, max_faces: int = 1) -> None:
        """Initialize the face detector.

        Args:
            confidence_threshold: Minimum confidence to accept a detection.
            max_faces: Maximum number of simultaneous faces to track.

        Raises:
            ImportError: If MediaPipe is not installed.
        """
        if not HAS_MEDIAPIPE:
            raise ImportError(
                "MediaPipe is required for face detection. "
                "Install with: pip install mediapipe"
            )

        self.confidence_threshold = confidence_threshold
        self._default_confidence = confidence_threshold

        model_path = _ensure_model_downloaded()

        base_options = BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=confidence_threshold,
            min_face_presence_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )

        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        logger.info(
            "MediaPipe Face Landmarker initialized (CPU mode, max_faces=%d)",
            max_faces,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a single frame and extract landmarks.

        Args:
            frame: BGR image as a numpy array (OpenCV format).

        Returns:
            A list of FaceDetection objects, one per detected face.
            Returns an empty list if no faces are found.
        """
        # MediaPipe expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._detector.detect(mp_image)

        detections: List[FaceDetection] = []

        if not result.face_landmarks:
            return detections

        h, w = frame.shape[:2]

        for face_landmarks_list in result.face_landmarks:
            if not face_landmarks_list:
                continue

            landmarks = face_landmarks_list

            # Compute bounding box from all 478 landmarks
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]

            # Convert normalized coordinates to pixel coordinates
            x1 = min(xs) * w
            y1 = min(ys) * h
            x2 = max(xs) * w
            y2 = max(ys) * h

            face_w = x2 - x1
            face_h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Extract selected landmark pixel positions
            landmark_pixels = [(lm.x * w, lm.y * h) for lm in landmarks]
            landmark_norm = [(lm.x, lm.y) for lm in landmarks]

            # Calculate mouth opening ratio using upper/lower lip landmarks
            mouth_ratio = self._compute_mouth_ratio(landmarks, w, h)

            # Explainable visual-quality evidence (see helper).
            quality = compute_face_quality(
                landmark_norm, face_width=face_w, frame_width=w
            )

            # Use face presence confidence as detection confidence.
            # Different MediaPipe versions expose this differently; fall
            # back to the configured threshold value when the field is
            # not available in this build (any returned detection already
            # passed the internal confidence gate).
            confidence = self._extract_confidence(result)

            det = FaceDetection(
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                center_x=cx,
                center_y=cy,
                face_width=face_w,
                face_height=face_h,
                confidence=confidence,
                landmarks=landmark_pixels,
                mouth_opening_ratio=mouth_ratio,
                mouth_aperture=mouth_ratio,
                **quality,
            )
            detections.append(det)

        return detections

    def _extract_confidence(self, result) -> float:
        """Extract a detection confidence from the MediaPipe result.

        MediaPipe Face Landmarker versions expose the detection confidence
        differently. This method tries known attribute names and falls back
        to the configured threshold value, which is honest because any
        returned detection already passed MediaPipe's internal confidence
        gate.

        Args:
            result: The FaceLandmarkerResult from MediaPipe.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        for attr in (
            "face_detection_confidences",
            "face_detection_confidence",
            "face_landmarks_confidences",
        ):
            value = getattr(result, attr, None)
            if value:
                return float(value[0])
        return self._default_confidence

    def _compute_mouth_ratio(
        self, landmarks: list, frame_w: int, frame_h: int
    ) -> float:
        """Compute a normalized mouth opening ratio from lip landmarks.

        Uses the vertical distance between the upper lip top and lower lip
        bottom, normalized by the face height to make it scale-invariant.

        Args:
            landmarks: List of MediaPipe face landmarks.
            frame_w: Frame width for coordinate scaling.
            frame_h: Frame height for coordinate scaling.

        Returns:
            A float between 0.0 (closed) and 1.0 (fully open), approximately.
        """
        try:
            upper = landmarks[self.UPPER_LIP_TOP]
            lower = landmarks[self.LOWER_LIP_BOTTOM]
            left = landmarks[self.LEFT_MOUTH]
            right = landmarks[self.RIGHT_MOUTH]

            mouth_height = abs(lower.y - upper.y) * frame_h
            face_width = abs(right.x - left.x) * frame_w

            if face_width <= 0:
                return 0.0

            ratio = mouth_height / face_width
            return round(min(1.0, max(0.0, ratio)), 4)

        except (IndexError, AttributeError):
            return 0.0

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_detector") and self._detector:
            self._detector.close()


def compute_face_quality(
    landmarks_norm: List[Tuple[float, float]],
    face_width: float,
    frame_width: float,
) -> dict:
    """Compute explainable visual-quality evidence for a detected face.

    Heuristics are deliberately simple and model-free so they stay
    interviewable:

    - face_visibility: fraction of all landmarks strictly inside the image.
      MediaPipe clamps out-of-image landmarks to the 0/1 edges, so a face
      partially off-screen scores below 1.0.
    - size_quality: how large the face is relative to the frame width
      (>=20% of the width counts as full quality).
    - head_pose_quality: frontalness estimate from nose/cheek asymmetry
      (|right-left| / (right+left)). A frontal face is ~1.0; a strongly
      turned face drops toward 0.
    - mouth_visibility: fraction of the lip landmarks strictly inside the
      image.

    Args:
        landmarks_norm: Normalized (x, y) landmark coordinates in [0, 1].
        face_width: Face bbox width in pixels.
        frame_width: Analysis frame width in pixels.

    Returns:
        Dict with the FaceDetection quality fields.
    """
    n = len(landmarks_norm)
    if n == 0:
        return {
            "face_visibility": 0.0,
            "size_quality": 0.0,
            "head_pose_quality": 0.0,
            "mouth_visibility": 0.0,
            "landmark_quality": 0.0,
            "visual_quality": 0.0,
        }

    inside = sum(
        1 for (x, y) in landmarks_norm if 0.0 < x < 1.0 and 0.0 < y < 1.0
    )
    face_visibility = inside / n

    size_quality = min(1.0, face_width / max(1e-6, 0.20 * frame_width))

    head_pose_quality = _pose_quality(landmarks_norm)

    lip = [
        (x, y)
        for idx, (x, y) in enumerate(landmarks_norm)
        if idx in FaceDetector.LIP_LANDMARKS
    ]
    mouth_visibility = (
        sum(1 for (x, y) in lip if 0.0 < x < 1.0 and 0.0 < y < 1.0) / len(lip)
        if lip
        else 0.0
    )

    landmark_quality = min(face_visibility, size_quality, head_pose_quality)
    visual_quality = min(
        landmark_quality,
        mouth_visibility,
    )

    return {
        "face_visibility": round(face_visibility, 4),
        "size_quality": round(size_quality, 4),
        "head_pose_quality": round(head_pose_quality, 4),
        "mouth_visibility": round(mouth_visibility, 4),
        "landmark_quality": round(landmark_quality, 4),
        "visual_quality": round(visual_quality, 4),
    }


def combined_visual_quality(face: FaceDetection) -> float:
    """Return the combined visual usability of a face.

    The "weakest link" min() combination is the conservative choice: any
    single weak component (tiny, edge-clipped, or side-facing) drags the
    combined quality down, which is the honest representation of
    uncertainty required for weak-evidence faces.

    Args:
        face: A face detection.

    Returns:
        Combined quality in [0, 1].
    """
    return float(getattr(face, "visual_quality", 1.0))


def _pose_quality(landmarks_norm: List[Tuple[float, float]]) -> float:
    """Estimate frontalness from nose-to-eye-line symmetry.

    Uses the normalized landmark indices from the MediaPipe canonical face
    mesh (nose tip 1, outer eye corners 33/263). The nose is projected onto
    the line joining the two outer eye corners:

        ratio = |dot(nose - eye_mid, eye_dir)| / (eye_span / 2)

    A frontal face projects the nose onto the eye-line midpoint (ratio ~ 0);
    a face turned toward a shoulder projects it close to the near eye corner
    (ratio ~ 1). The perpendicular (vertical) nose-to-eye distance is ignored
    so looking up/down does not count as turning away.

    Args:
        landmarks_norm: Normalized landmark list (may be shorter than 478).

    Returns:
        Frontalness in [0, 1].
    """
    n = len(landmarks_norm)
    eye_l, eye_r = FaceDetector.EYE_OUTER
    if FaceDetector.NOSE_TIP >= n or eye_l >= n or eye_r >= n:
        return 0.5

    nose = landmarks_norm[FaceDetector.NOSE_TIP]
    p_left = landmarks_norm[eye_l]
    p_right = landmarks_norm[eye_r]

    span_vec = (p_right[0] - p_left[0], p_right[1] - p_left[1])
    span = (span_vec[0] ** 2 + span_vec[1] ** 2) ** 0.5
    if span <= 1e-6:
        return 0.5

    mid = ((p_left[0] + p_right[0]) / 2, (p_left[1] + p_right[1]) / 2)
    v = (nose[0] - mid[0], nose[1] - mid[1])
    along = (v[0] * span_vec[0] + v[1] * span_vec[1]) / span
    ratio = abs(along) / (span / 2)
    return max(0.0, min(1.0, 1.0 - ratio))


def _ensure_model_downloaded() -> str:
    """Ensure the MediaPipe face landmark model file is available.

    Downloads the face_landmarker.task model on first use and caches it
    in ~/.mediapipe/models. MediaPipe 1.x also bundles the task model in
    its wheel; this function prefers the bundled copy and falls back to
    downloading if it is not present.

    Returns:
        Path to the face landmark model file.

    Raises:
        RuntimeError: If the model cannot be located or downloaded.
    """
    import os
    import urllib.request

    # Prefer the bundled copy delivered with the MediaPipe wheel.
    bundled_candidates = [
        os.path.join(
            os.path.dirname(mp.__file__),
            "modules",
            "face_landmarker",
            "face_landmarker.task",
        ),
        os.path.join(
            os.path.dirname(mp_vision.__file__),
            "face_landmarker",
            "face_landmarker.task",
        ),
    ]
    for candidate in bundled_candidates:
        if candidate and os.path.isfile(candidate):
            logger.info("Using bundled MediaPipe face landmark model: %s", candidate)
            return candidate

    # Fall back to the cached/downloaded copy under the user home.
    model_name = "face_landmarker.task"
    model_dir = os.path.join(os.path.expanduser("~"), ".mediapipe", "models")
    model_path = os.path.join(model_dir, model_name)

    if os.path.isfile(model_path):
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    logger.info("Downloading MediaPipe face landmarker model to %s", model_path)
    try:
        urllib.request.urlretrieve(url, model_path)
    except OSError as exc:
        raise RuntimeError(
            "Unable to download the MediaPipe face landmarker model "
            f"({exc}). Check network access."
        ) from exc
    logger.info("Model downloaded successfully")

    return model_path
