"""
pipeline/processor.py
----------------------
PrivacyPipeline — the main orchestrator.

Connects detection → anonymization → crypto → MySQL logging into a single
coherent, sub-30ms per-frame processing loop.

Architecture
------------
    FrameResult = process_frame(raw_frame)
         │
         ├── 1. Hash raw frame (SHA-256, for deduplication)
         ├── 2. Detect sensitive regions (faces, plates)
         ├── 3. Apply anonymization (blur/pixelate/mask)
         ├── 4. Pseudonymise tracking IDs (HMAC-SHA256)
         ├── 5. Log detection metadata → MySQL / SQLite
         └── 6. Return anonymized frame + FrameResult

All PII-related data is either hashed or never stored.
The anonymized frame may be forwarded to cloud analytics safely.

Latency budget (sub-30ms target at 1080p):
  Face detection    ~5–15ms   (Haar Cascade, downscaled input)
  Plate detection   ~3–8ms    (MSER)
  Blurring          ~1–3ms    (OpenCV SIMD)
  DB write          ~0.5–2ms  (buffered, non-blocking option)
  Total             ~10–28ms  ✓
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
from privacy_pipeline.anonymizer.detectors import (
    DetectedRegion, FaceDetector, LicensePlateDetector, create_face_detector,
)
from privacy_pipeline.crypto.hasher import TrackingIDHasher
from privacy_pipeline.database.connection import DatabaseManager
from privacy_pipeline.database.repositories import (
    AuditEvent, AuditRepository, DetectionEvent, DetectionRepository,
    SessionRepository, TrackingIDRepository,
)
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Output of processing one video frame."""

    frame_sequence:        int
    anonymized_frame:      np.ndarray
    regions_detected:      List[DetectedRegion]
    face_count:            int
    plate_count:           int
    frame_hash_sha256:     str
    processing_latency_ms: float
    captured_at:           str
    session_id:            str
    pseudonym_hash:        Optional[str] = None
    metadata:              Dict[str, Any] = field(default_factory=dict)

    @property
    def has_detections(self) -> bool:
        return self.face_count > 0 or self.plate_count > 0

    def to_dict(self) -> Dict:
        return {
            "frame_sequence":        self.frame_sequence,
            "face_count":            self.face_count,
            "plate_count":           self.plate_count,
            "frame_hash_sha256":     self.frame_hash_sha256,
            "processing_latency_ms": round(self.processing_latency_ms, 3),
            "captured_at":           self.captured_at,
            "session_id":            self.session_id,
            "has_detections":        self.has_detections,
        }


# ---------------------------------------------------------------------------
# PrivacyPipeline
# ---------------------------------------------------------------------------

class PrivacyPipeline:
    """
    End-to-end privacy-preserving video processing pipeline.

    Parameters
    ----------
    db:                 DatabaseManager instance (MySQL or SQLite).
    hasher:             TrackingIDHasher for pseudonymising tracking IDs.
    face_detector:      FaceDetector (or None to disable face detection).
    plate_detector:     LicensePlateDetector (or None to disable).
    face_blurrer:       RegionBlurrer for faces.
    plate_blurrer:      RegionBlurrer for plates.
    session_id:         UUID string for this processing session.
    source_identifier:  Human-readable label for the video source.
    max_latency_ms:     Warn if a frame takes longer than this.

    Example
    -------
    >>> pipeline = PrivacyPipeline.from_config(cfg, db)
    >>> for frame in video_source:
    ...     result = pipeline.process_frame(frame)
    ...     send_to_cloud(result.anonymized_frame)
    >>> summary = pipeline.close()
    """

    def __init__(
        self,
        db: DatabaseManager,
        hasher: TrackingIDHasher,
        face_detector: Optional[FaceDetector],
        plate_detector: Optional[LicensePlateDetector],
        face_blurrer: RegionBlurrer,
        plate_blurrer: RegionBlurrer,
        session_id: Optional[str] = None,
        source_identifier: str = "unknown",
        max_latency_ms: float = 30.0,
        log_to_db: bool = True,
    ) -> None:
        self._db              = db
        self._hasher          = hasher
        self._face_detector   = face_detector
        self._plate_detector  = plate_detector
        self._face_blurrer    = face_blurrer
        self._plate_blurrer   = plate_blurrer
        self._max_latency_ms  = max_latency_ms
        self._log_to_db       = log_to_db
        self._source          = source_identifier

        self.session_id = session_id or str(uuid.uuid4())

        # Repositories
        self._sessions    = SessionRepository(db)
        self._detections  = DetectionRepository(db)
        self._tracking    = TrackingIDRepository(db)
        self._audit       = AuditRepository(db)

        # Runtime statistics
        self._frame_count       = 0
        self._faces_total       = 0
        self._plates_total      = 0
        self._latencies_ms: List[float] = []

        # Register session in DB
        if log_to_db:
            self._sessions.create(
                session_id        = self.session_id,
                source_identifier = source_identifier,
            )
            self._audit.log(AuditEvent(
                event_type    = "session",
                action        = "start_session",
                session_id    = self.session_id,
                resource_type = "video_source",
                resource_id   = source_identifier,
                compliance_tag = "GDPR-A30",
            ))

        logger.info(
            "PrivacyPipeline started: session=%s, source=%s, "
            "face_detection=%s, plate_detection=%s",
            self.session_id, source_identifier,
            face_detector is not None,
            plate_detector is not None,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg,
        db: DatabaseManager,
        source_identifier: str = "video_source",
        session_id: Optional[str] = None,
    ) -> "PrivacyPipeline":
        """Build a fully configured pipeline from a Config object."""
        import os
        anon = cfg.anonymization

        # Hasher — load key from env or config
        hmac_key = cfg.crypto.hmac.get("secret_key", "") or os.environ.get("PPVP_HMAC_KEY", "")
        if hmac_key:
            key_bytes = (
                bytes.fromhex(hmac_key)
                if len(hmac_key) >= 64 and all(c in "0123456789abcdefABCDEF" for c in hmac_key)
                else hmac_key.encode("utf-8")
            )
        else:
            key_bytes = None  # ephemeral

        hasher = TrackingIDHasher(secret_key=key_bytes)

        # Detectors — detector type driven by config (haar_cascade | yunet | retinaface)
        face_det = (
            create_face_detector(
                getattr(anon.face, "detector", "haar_cascade"),
                model_path     = getattr(anon.face, "detector_model_path", None),
                conf_threshold = getattr(anon.face, "conf_threshold", 0.45),
                padding        = anon.face.padding_px,
            )
            if anon.face.enabled else None
        )
        plate_det = (
            LicensePlateDetector(
                aspect_ratio_range=tuple(anon.license_plate.aspect_ratio_range),
            )
            if anon.license_plate.enabled else None
        )

        # Blurrers
        face_blurrer  = RegionBlurrer(
            method   = anon.face.blur_method,
            strength = anon.face.blur_strength,
        )
        plate_blurrer = RegionBlurrer(
            method   = anon.license_plate.blur_method,
            strength = 0,  # solid fill
        )

        return cls(
            db                = db,
            hasher            = hasher,
            face_detector     = face_det,
            plate_detector    = plate_det,
            face_blurrer      = face_blurrer,
            plate_blurrer     = plate_blurrer,
            session_id        = session_id,
            source_identifier = source_identifier,
            max_latency_ms    = cfg.pipeline.max_latency_ms,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        tracking_id: Optional[str] = None,
        captured_at: Optional[str] = None,
    ) -> FrameResult:
        """
        Process a single video frame through the full anonymization pipeline.

        Parameters
        ----------
        frame:       BGR frame (H, W, 3).
        tracking_id: Optional camera/person ID to pseudonymise.
        captured_at: ISO timestamp when the frame was captured.

        Returns
        -------
        FrameResult with anonymized frame and detection metadata.
        """
        t0 = time.monotonic()
        self._frame_count += 1

        if captured_at is None:
            captured_at = datetime.now(timezone.utc).isoformat()

        # ── 1. Hash raw frame ──────────────────────────────────────────
        frame_hash = TrackingIDHasher.hash_numpy_frame(frame)

        # ── 2. Detection ───────────────────────────────────────────────
        face_regions:  List[DetectedRegion] = []
        plate_regions: List[DetectedRegion] = []

        if self._face_detector:
            face_regions  = self._face_detector.detect(frame)
        if self._plate_detector:
            plate_regions = self._plate_detector.detect(frame)

        all_regions = face_regions + plate_regions

        # ── 3. Anonymization ───────────────────────────────────────────
        result_frame = frame.copy()
        if face_regions:
            result_frame = self._face_blurrer.apply(result_frame, face_regions)
        if plate_regions:
            result_frame = self._plate_blurrer.apply(result_frame, plate_regions)

        # ── 4. Pseudonymise tracking ID ────────────────────────────────
        pseudonym = None
        if tracking_id:
            pseudonym = self._hasher.pseudonymise(tracking_id)
            if self._log_to_db:
                self._tracking.upsert(pseudonym)

        # ── 5. Log to DB ───────────────────────────────────────────────
        latency_ms = (time.monotonic() - t0) * 1000
        self._latencies_ms.append(latency_ms)
        self._faces_total  += len(face_regions)
        self._plates_total += len(plate_regions)

        if self._log_to_db:
            event = DetectionEvent(
                session_id            = self.session_id,
                frame_sequence        = self._frame_count,
                captured_at           = captured_at,
                frame_hash_sha256     = frame_hash,
                face_count            = len(face_regions),
                plate_count           = len(plate_regions),
                processing_latency_ms = latency_ms,
                anonymization_applied = True,
                detection_meta        = {
                    "regions": [r.to_dict() for r in all_regions],
                } if all_regions else None,
            )
            self._detections.insert(event)

        if latency_ms > self._max_latency_ms:
            logger.warning(
                "Frame %d exceeded latency target: %.2fms (target: %.1fms)",
                self._frame_count, latency_ms, self._max_latency_ms,
            )

        return FrameResult(
            frame_sequence        = self._frame_count,
            anonymized_frame      = result_frame,
            regions_detected      = all_regions,
            face_count            = len(face_regions),
            plate_count           = len(plate_regions),
            frame_hash_sha256     = frame_hash,
            processing_latency_ms = latency_ms,
            captured_at           = captured_at,
            session_id            = self.session_id,
            pseudonym_hash        = pseudonym,
        )

    def process_batch(
        self,
        frames: List[np.ndarray],
        tracking_id: Optional[str] = None,
    ) -> List[FrameResult]:
        """Process a list of frames and return results."""
        return [self.process_frame(f, tracking_id=tracking_id) for f in frames]

    # ------------------------------------------------------------------
    # Session summary / close
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current session statistics."""
        lats = self._latencies_ms
        return {
            "session_id":        self.session_id,
            "total_frames":      self._frame_count,
            "total_faces":       self._faces_total,
            "total_plates":      self._plates_total,
            "mean_latency_ms":   round(sum(lats) / len(lats), 3) if lats else 0.0,
            "p99_latency_ms":    round(
                sorted(lats)[int(len(lats) * 0.99)] if lats else 0.0, 3
            ),
            "max_latency_ms":    round(max(lats), 3) if lats else 0.0,
            "latency_target_ms": self._max_latency_ms,
            "under_target_pct":  (
                round(sum(1 for l in lats if l <= self._max_latency_ms)
                      / len(lats) * 100, 1)
                if lats else 100.0
            ),
        }

    def close(self) -> Dict[str, Any]:
        """Finalise session: write summary to DB, log audit event, return stats."""
        stats = self.get_stats()

        if self._log_to_db and self._frame_count > 0:
            self._sessions.update_stats(
                session_id         = self.session_id,
                total_frames       = self._frame_count,
                frames_with_faces  = self._faces_total,
                frames_with_plates = self._plates_total,
                mean_latency_ms    = stats["mean_latency_ms"],
                p99_latency_ms     = stats["p99_latency_ms"],
            )
            self._sessions.close_session(self.session_id, status="completed")

            self._audit.log_anonymization(
                session_id  = self.session_id,
                frame_count = self._frame_count,
                face_count  = self._faces_total,
                plate_count = self._plates_total,
            )

        logger.info(
            "PrivacyPipeline closed: session=%s | frames=%d | "
            "faces=%d | plates=%d | mean_latency=%.2fms | under_target=%.1f%%",
            self.session_id,
            stats["total_frames"],
            stats["total_faces"],
            stats["total_plates"],
            stats["mean_latency_ms"],
            stats["under_target_pct"],
        )
        return stats

    def __enter__(self) -> "PrivacyPipeline":
        return self

    def __exit__(self, *args) -> None:
        self.close()
