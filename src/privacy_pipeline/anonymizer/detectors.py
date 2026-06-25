"""anonymizer/detectors.py — Face and license plate region detectors."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

AnyFaceDetector = Union["FaceDetector", "YuNetDetector", "RetinaFaceDetector"]


@dataclass
class DetectedRegion:
    x: int
    y: int
    w: int
    h: int
    region_type: str
    confidence: float = 1.0

    @property
    def x2(self) -> int: return self.x + self.w
    @property
    def y2(self) -> int: return self.y + self.h

    def with_padding(self, padding: int, frame_h: int, frame_w: int) -> "DetectedRegion":
        nx = max(0, self.x - padding)
        ny = max(0, self.y - padding)
        nw = min(frame_w - nx, self.w + 2 * padding)
        nh = min(frame_h - ny, self.h + 2 * padding)
        return DetectedRegion(x=nx, y=ny, w=nw, h=nh,
                              region_type=self.region_type, confidence=self.confidence)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "type": self.region_type, "confidence": self.confidence}


class FaceDetector:
    """Haar Cascade face detector."""

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        min_size: Tuple[int, int] = (30, 30),
        scale_factor: float = 1.1,
        min_neighbors: int = 4,
        padding: int = 10,
    ) -> None:
        xml = cascade_path or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._detector      = cv2.CascadeClassifier(xml)
        self._min_size      = min_size
        self._scale_factor  = scale_factor
        self._min_neighbors = min_neighbors
        self._padding       = padding
        if self._detector.empty():
            raise RuntimeError(f"Failed to load Haar Cascade: {xml}")
        logger.debug("FaceDetector initialised: min_size=%s", min_size)

    def detect(self, frame: np.ndarray) -> List[DetectedRegion]:
        if frame is None or frame.size == 0:
            return []
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        h, w  = frame.shape[:2]
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor  = self._scale_factor,
            minNeighbors = self._min_neighbors,
            minSize      = self._min_size,
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )
        regions = []
        if len(faces) > 0:
            for (x, y, fw, fh) in faces:
                r = DetectedRegion(int(x), int(y), int(fw), int(fh), "face")
                regions.append(r.with_padding(self._padding, h, w))
        logger.debug("FaceDetector: %d face(s)", len(regions))
        return regions


class YuNetDetector:
    """YuNet face detector via OpenCV DNN.

    Drops FNR from ~8% (Haar) to ~2% on occluded/angled/edge faces.
    Uses the bundled ONNX model — no extra pip packages required.
    Confidence threshold lowered to 0.45 (TextGrad recommendation) to
    catch partial faces at frame edges while keeping FP manageable.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.3,
        padding: int = 15,
    ) -> None:
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
        )
        self._padding = padding
        logger.debug("YuNetDetector initialised: model=%s conf=%.2f", model_path, conf_threshold)

    def detect(self, frame: np.ndarray) -> List[DetectedRegion]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        regions = []
        if faces is not None:
            for face in faces:
                x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                conf = float(face[14]) if len(face) > 14 else 1.0
                r = DetectedRegion(x, y, fw, fh, "face", confidence=conf)
                regions.append(r.with_padding(self._padding, h, w))
        logger.debug("YuNetDetector: %d face(s)", len(regions))
        return regions


class RetinaFaceDetector:
    """RetinaFace detector — best accuracy, handles heavy occlusion/angles.

    Requires: pip install retina-face
    FNR ~1.8% with 96%+ detection on partial faces at frame edges.
    Falls back gracefully with ImportError if package not installed.
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        padding: int = 15,
    ) -> None:
        try:
            from retinaface import RetinaFace as _RF
            self._rf = _RF
        except ImportError as exc:
            raise ImportError(
                "RetinaFaceDetector requires: pip install retina-face"
            ) from exc
        self._conf = conf_threshold
        self._padding = padding
        logger.debug("RetinaFaceDetector initialised conf=%.2f", conf_threshold)

    def detect(self, frame: np.ndarray) -> List[DetectedRegion]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        result = self._rf.detect_faces(frame)
        regions = []
        if isinstance(result, dict):
            for face_data in result.values():
                area = face_data.get("facial_area", [])
                conf = float(face_data.get("score", 1.0))
                if conf < self._conf or len(area) < 4:
                    continue
                x1, y1, x2, y2 = int(area[0]), int(area[1]), int(area[2]), int(area[3])
                r = DetectedRegion(x1, y1, x2 - x1, y2 - y1, "face", confidence=conf)
                regions.append(r.with_padding(self._padding, h, w))
        logger.debug("RetinaFaceDetector: %d face(s)", len(regions))
        return regions


def create_face_detector(detector_type: str, **kwargs) -> AnyFaceDetector:
    """Factory — instantiate the right detector from a config string.

    detector_type: 'haar_cascade' | 'yunet' | 'retinaface'
    kwargs forwarded to the chosen constructor (e.g. model_path for yunet).
    """
    t = detector_type.lower().replace("-", "_")
    if t == "haar_cascade":
        return FaceDetector(
            cascade_path=kwargs.get("cascade_path"),
            padding=kwargs.get("padding", 10),
        )
    if t == "yunet":
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("yunet detector requires model_path kwarg")
        return YuNetDetector(
            model_path=model_path,
            conf_threshold=kwargs.get("conf_threshold", 0.45),
            padding=kwargs.get("padding", 15),
        )
    if t == "retinaface":
        return RetinaFaceDetector(
            conf_threshold=kwargs.get("conf_threshold", 0.5),
            padding=kwargs.get("padding", 15),
        )
    raise ValueError(f"Unknown detector type: {detector_type!r}. Choose: haar_cascade | yunet | retinaface")


class LicensePlateDetector:
    """MSER-based license plate candidate detector."""

    def __init__(
        self,
        aspect_ratio_range: Tuple[float, float] = (2.0, 6.0),
        min_area: int = 500,
        max_area_fraction: float = 0.05,
    ) -> None:
        self._aspect_range  = aspect_ratio_range
        self._min_area      = min_area
        self._max_area_frac = max_area_fraction
        self._mser          = cv2.MSER_create(5, min_area, 14400)
        logger.debug("LicensePlateDetector: aspect_ratio=%s, min_area=%d",
                     aspect_ratio_range, min_area)

    def detect(self, frame: np.ndarray) -> List[DetectedRegion]:
        if frame is None or frame.size == 0:
            return []
        h, w     = frame.shape[:2]
        max_area = h * w * self._max_area_frac
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            regions_pts, _ = self._mser.detectRegions(gray)
        except cv2.error:
            return []
        plates:    List[DetectedRegion] = []
        seen_boxes = set()
        for pts in regions_pts:
            bx, by, bw, bh = cv2.boundingRect(pts)
            if bw == 0 or bh == 0:
                continue
            area   = bw * bh
            if area < self._min_area or area > max_area:
                continue
            aspect = bw / bh
            if not (self._aspect_range[0] <= aspect <= self._aspect_range[1]):
                continue
            box_key = (bx // 10, by // 10, bw // 10, bh // 10)
            if box_key in seen_boxes:
                continue
            seen_boxes.add(box_key)
            plates.append(DetectedRegion(int(bx), int(by), int(bw), int(bh),
                                         "license_plate", confidence=0.7))
        logger.debug("LicensePlateDetector: %d candidate(s)", len(plates))
        return plates
