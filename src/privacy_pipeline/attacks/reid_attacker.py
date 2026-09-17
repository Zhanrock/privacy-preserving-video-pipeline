"""attacks/reid_attacker.py — face re-identification attacker.

This is the adversary PrivGrad's anonymizer must defeat: given a (possibly
blurred) face region, try to match it to the right identity in a gallery of
clear reference photos. Re-identification accuracy on the anonymizer's
output — anchored against chance-level accuracy and against accuracy on the
*unblurred* frame — is the attack-based privacy metric that replaces the
hand-tuned ``recall * (0.6 + 0.4 * mean_confidence)`` formula in
``benchmark_detectors.py`` (that formula has no adversary in it at all, which
is disqualifying for a privacy venue — see PrivGrad's research-hub decisions).

Embedding model: insightface's ``w600k_r50`` (ArcFace, ResNet-50), the
recognition component of the buffalo_l pack already used for
``RetinaFaceDetector`` in ``detectors.py`` — same model file, no new download.

Known limitation (by design, for now): proper ArcFace preprocessing aligns
the face crop to a canonical pose using 5 facial landmarks before resizing to
112x112. This module skips alignment and does a plain resize, because
``DetectedRegion`` doesn't carry landmarks. That makes this a *weaker* than
state-of-the-art attacker — which means a privacy result of "this attacker
fails" is a floor, not a ceiling: a real adversary using landmark alignment
could only do better. Do not report this as an upper bound on attacker
capability without addressing that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from privacy_pipeline.anonymizer.detectors import DetectedRegion
from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_PATH = os.path.expanduser(
    os.path.join("~", ".insightface", "models", "buffalo_l", "w600k_r50.onnx")
)


class ArcFaceEmbedder:
    """Wraps insightface's ``w600k_r50`` ONNX model for 512-d face embeddings.

    Uses the raw onnxruntime session directly (not insightface's
    ``model_zoo``/``FaceAnalysis`` wrappers) because those assume landmark-
    aligned input; see module docstring for why we skip alignment here.
    """

    INPUT_SIZE = 112

    def __init__(self, model_path: Optional[str] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ArcFaceEmbedder requires: pip install onnxruntime"
            ) from exc

        path = model_path or _DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ArcFace recognition weights not found at {path}. "
                "Trigger the one-time buffalo_l download with: "
                "python -c \"from insightface.app import FaceAnalysis; "
                "FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)\""
            )
        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        logger.debug("ArcFaceEmbedder initialised: model=%s", path)

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised 512-d embedding for a BGR face crop."""
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("empty face crop")
        resized = cv2.resize(face_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb - 127.5) / 127.5
        chw = np.transpose(normalized, (2, 0, 1))[None, ...]  # (1, 3, 112, 112)
        (feat,) = self._session.run(None, {self._input_name: chw})
        vec = feat[0]
        return vec / (np.linalg.norm(vec) + 1e-10)


@dataclass
class GalleryEntry:
    identity: str
    embedding: np.ndarray


class ReIDAttacker:
    """Matches a probe embedding against a gallery of known identities."""

    def __init__(self, embedder: ArcFaceEmbedder) -> None:
        self._embedder = embedder
        self._gallery: List[GalleryEntry] = []

    def enroll(self, identity: str, face_bgr: np.ndarray) -> None:
        """Add one clear reference photo's embedding to the gallery."""
        self._gallery.append(GalleryEntry(identity, self._embedder.embed(face_bgr)))

    @property
    def gallery_size(self) -> int:
        return len({g.identity for g in self._gallery})

    def identify(self, probe_bgr: np.ndarray) -> tuple[str, float]:
        """Return (best_matching_identity, cosine_similarity) for a probe crop.

        Probe may be a clear or anonymized region — the attacker doesn't know
        which; it just tries its best on whatever pixels it's given.
        """
        if not self._gallery:
            raise RuntimeError("gallery is empty — call enroll() first")
        probe_vec = self._embedder.embed(probe_bgr)
        sims = [
            (entry.identity, float(np.dot(probe_vec, entry.embedding)))
            for entry in self._gallery
        ]
        return max(sims, key=lambda pair: pair[1])

    def crop_region(self, frame: np.ndarray, region: DetectedRegion) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, region.x), max(0, region.y)
        x2, y2 = min(w, region.x2), min(h, region.y2)
        return frame[y1:y2, x1:x2]
