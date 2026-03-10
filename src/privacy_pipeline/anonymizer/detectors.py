"""anonymizer/detectors.py — Face and license plate region detectors."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


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
