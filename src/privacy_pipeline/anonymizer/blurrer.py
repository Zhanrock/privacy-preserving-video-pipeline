"""
anonymizer/blurrer.py
----------------------
Applies anonymization effects to detected regions in video frames.

Blur methods
------------
gaussian    — Gaussian blur (smooth, natural-looking anonymization)
pixelate    — Pixel mosaic (obvious but informative — shows something is hidden)
solid       — Solid fill (complete occlusion — strongest privacy)

Design: ``RegionBlurrer`` is a stateless transformer — takes a frame and
regions, returns an anonymized copy without modifying the original.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import List

from privacy_pipeline.anonymizer.detectors import DetectedRegion
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class RegionBlurrer:
    """
    Applies anonymization effects to detected sensitive regions.

    Parameters
    ----------
    method:    "gaussian" | "pixelate" | "solid"
    strength:  Blur kernel size (gaussian) or block size (pixelate).
               Ignored for "solid".
    fill_color: BGR fill colour for "solid" mode (default: black).
    """

    SUPPORTED_METHODS = ("gaussian", "pixelate", "solid")

    def __init__(
        self,
        method: str = "pixelate",
        strength: int = 25,
        fill_color: tuple = (0, 0, 0),
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown blur method: '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        if strength < 1:
            raise ValueError(f"strength must be ≥ 1, got {strength}")

        self._method     = method
        self._strength   = strength
        self._fill_color = fill_color

    def apply(
        self,
        frame: np.ndarray,
        regions: List[DetectedRegion],
    ) -> np.ndarray:
        """
        Apply blur/mask to all regions in the frame.

        Parameters
        ----------
        frame:   BGR image (H, W, 3) — not modified in-place.
        regions: List of DetectedRegion to anonymize.

        Returns
        -------
        New frame with sensitive regions anonymized.
        """
        if not regions:
            return frame.copy()

        result = frame.copy()
        for region in regions:
            result = self._apply_to_region(result, region)

        logger.debug(
            "RegionBlurrer: applied %s to %d region(s)",
            self._method, len(regions),
        )
        return result

    def _apply_to_region(
        self, frame: np.ndarray, region: DetectedRegion
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x1   = max(0, region.x)
        y1   = max(0, region.y)
        x2   = min(w, region.x2)
        y2   = min(h, region.y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        if self._method == "gaussian":
            return self._gaussian(frame, x1, y1, x2, y2)
        elif self._method == "pixelate":
            return self._pixelate(frame, x1, y1, x2, y2)
        else:  # solid
            return self._solid(frame, x1, y1, x2, y2)

    def _gaussian(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        roi    = frame[y1:y2, x1:x2]
        ksize  = self._strength | 1   # Must be odd
        ksize  = max(3, ksize)
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        result  = frame.copy()
        result[y1:y2, x1:x2] = blurred
        return result

    def _pixelate(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        roi    = frame[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        if rh == 0 or rw == 0:
            return frame

        block  = max(1, self._strength)
        # Shrink to block-pixel thumbnail, then scale back up
        small_w = max(1, rw // block)
        small_h = max(1, rh // block)
        small   = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

        result = frame.copy()
        result[y1:y2, x1:x2] = pixelated
        return result

    def _solid(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        result = frame.copy()
        result[y1:y2, x1:x2] = self._fill_color
        return result

    def __repr__(self) -> str:
        return (
            f"RegionBlurrer(method={self._method}, "
            f"strength={self._strength})"
        )
