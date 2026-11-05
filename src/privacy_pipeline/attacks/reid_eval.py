"""
attacks/reid_eval.py
--------------------
Reusable privacy / utility evaluator for one anonymization setting.

Used by scripts/privacy_utility_sweep.py (the privacy-vs-utility curve) and by
scripts/textgrad_pipeline_v2.py (--reid-gallery), so TextGrad is scored on the
same attack-based metric the paper reports instead of detector recall.

Privacy:  top-1 accuracy of an ArcFace re-identification attacker that aligns
          on landmarks YuNet re-finds in the anonymized frame (falls back to an
          unaligned crop when it finds nothing). Gallery = one clear photo per
          identity; R random gallery splits give mean +- std.
Utility:  fraction of anonymized faces a standard detector (YuNet) still
          localizes at IoU >= 0.5 -- i.e. counting / tracking analytics keep
          working on the anonymized stream. A proxy, not a downstream task.

Strength can be absolute (pixels, as RegionBlurrer takes it) or relative to the
detected face width, so one setting means the same thing for a 100 px LFW face
and a 30 px CCTV face:
    pixelate, relative: strength = blocks across the face width
    gaussian, relative: strength = kernel size as % of the face width
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
from privacy_pipeline.anonymizer.detectors import DetectedRegion

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
DEFAULT_ARCFACE = os.path.expanduser(
    os.path.join("~", ".insightface", "models", "buffalo_l", "w600k_r50.onnx")
)


@dataclass(frozen=True)
class AnonSpec:
    method: str  # gaussian | pixelate | solid | clear
    strength: float = 0
    mode: str = "absolute"  # absolute | relative
    padding: int = 15

    @property
    def key(self) -> str:
        return f"{self.method}_{self.mode[0]}{self.strength:g}_p{self.padding}"

    def pixel_strength(self, face_w: int) -> int:
        """Strength in pixels for RegionBlurrer, for a face `face_w` px wide."""
        if self.mode == "absolute":
            return max(1, int(round(self.strength)))
        if self.method == "pixelate":  # blocks across the face
            return max(1, int(round(face_w / max(self.strength, 1e-6))))
        return max(3, int(round(face_w * self.strength / 100.0)))  # gaussian, % width


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class ReIDEvaluator:
    """Loads an identity-labelled gallery (one sub-folder of .jpg per identity)
    once, then scores any AnonSpec. Results are cached per spec and scale."""

    def __init__(
        self,
        gallery_dir: Path,
        yunet_path: Path,
        arcface_path: str = DEFAULT_ARCFACE,
        splits: int = 5,
        seed: int = 0,
        scale: float = 1.0,
        max_identities: Optional[int] = None,
    ) -> None:
        import onnxruntime as ort

        self.scale = scale
        self._det = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 320), score_threshold=0.45, nms_threshold=0.3
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 2
        opts.log_severity_level = 3
        self._emb = ort.InferenceSession(arcface_path, opts, providers=["CPUExecutionProvider"])
        self._emb_in = self._emb.get_inputs()[0].name
        self._cache: Dict[str, dict] = {}

        people = sorted(p for p in Path(gallery_dir).iterdir() if p.is_dir())
        if max_identities:
            people = people[:max_identities]

        # Gallery = clear full-resolution photo (the attacker's reference shot);
        # probes are optionally downscaled to simulate smaller faces in video.
        self.items: List[tuple] = []  # (identity, probe_frame, box, landmarks)
        gal_crops, self.missed = [], 0
        for person in people:
            for img in sorted(person.glob("*.jpg")):
                full = cv2.imread(str(img))
                if full is None:
                    continue
                d_full = self._detect(full)
                frame = full
                if scale != 1.0:
                    frame = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                d = self._detect(frame)
                if d_full is None or d is None:
                    self.missed += 1
                    continue
                gal_crops.append(self._align(full, d_full[1]))
                self.items.append((person.name, frame, d[0], d[1]))
        if not self.items:
            raise RuntimeError(f"no detectable faces under {gallery_dir}")

        self.ids = np.array([i[0] for i in self.items])
        self.uniq = sorted(set(self.ids))
        self.chance = 1.0 / len(self.uniq)
        self.face_px = float(np.median([i[2][2] for i in self.items]))
        self._gallery_emb = self._embed(gal_crops)

        rng = np.random.default_rng(seed)
        self._splits = []
        for _ in range(splits):
            g = np.array([rng.choice(np.where(self.ids == u)[0]) for u in self.uniq])
            self._splits.append((g, np.setdiff1d(np.arange(len(self.items)), g)))

        self.upper_bound = self.evaluate(AnonSpec("clear"))["attack_acc"]

    # -- primitives ---------------------------------------------------------
    def _detect(self, frame):
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda f: f[2] * f[3])
        box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        return box, f[4:14].reshape(5, 2).astype(np.float32)

    @staticmethod
    def _align(frame, lmk):
        M, _ = cv2.estimateAffinePartial2D(lmk, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        return cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)

    @staticmethod
    def _crop(frame, box):
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        y1, y2, x1, x2 = max(0, y), min(h, y + bh), max(0, x), min(w, x + bw)
        c = frame[y1:y2, x1:x2]
        return cv2.resize(c, (112, 112))

    def _embed(self, crops: List[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            end = i + batch
            chunk = crops[i:end]
            x = np.stack(
                [
                    np.transpose(
                        (cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5,
                        (2, 0, 1),
                    )
                    for c in chunk
                ]
            )
            (f,) = self._emb.run(None, {self._emb_in: x})
            out.append(f)
        v = np.concatenate(out)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)

    def anonymize(self, frame, box, spec: AnonSpec):
        if spec.method == "clear":
            return frame
        h, w = frame.shape[:2]
        region = DetectedRegion(*box, "face").with_padding(spec.padding, h, w)
        blurrer = RegionBlurrer(spec.method, strength=spec.pixel_strength(box[2]))
        return blurrer.apply(frame, [region])

    # -- main entry point ---------------------------------------------------
    def evaluate(self, spec: AnonSpec) -> dict:
        if spec.key in self._cache:
            return self._cache[spec.key]
        crops, found, oracle_crops = [], 0, []
        for _, frame, box, lmk in self.items:
            anon = self.anonymize(frame, box, spec)
            d = self._detect(anon)
            if d is not None and _iou(d[0], box) >= 0.5:
                found += 1
            crops.append(self._align(anon, d[1]) if d is not None else self._crop(anon, box))
            oracle_crops.append(self._align(anon, lmk))
        emb, emb_oracle = self._embed(crops), self._embed(oracle_crops)

        def acc(e):
            a = []
            for g, p in self._splits:
                pred = np.array(self.uniq)[(e[p] @ self._gallery_emb[g].T).argmax(1)]
                a.append(float((pred == self.ids[p]).mean()))
            return float(np.mean(a)), float(np.std(a))

        m, s = acc(emb)
        mo, _ = acc(emb_oracle)
        ub = getattr(self, "upper_bound", m)
        res = {
            "spec": asdict(spec),
            "scale": self.scale,
            "attack_acc": m,
            "attack_std": s,
            "attack_acc_oracle": mo,
            "chance": self.chance,
            "privacy": float(min(1, max(0, 1 - (m - self.chance) / max(ub - self.chance, 1e-6)))),
            "utility_detect": found / len(self.items),
            "median_face_px": self.face_px,
            "n_faces": len(self.items),
            "n_identities": len(self.uniq),
        }
        self._cache[spec.key] = res
        return res
