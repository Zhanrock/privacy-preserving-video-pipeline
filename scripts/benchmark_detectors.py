"""
benchmark_detectors.py — Three-Era Detection Benchmark
=======================================================
Compares Haar Cascade (Pre-AI baseline), YuNet (current), and optionally
RetinaFace across the five test images and produces privacy/utility metrics
suitable for the paper's Table 1 (Three-Era Benchmark).

Usage:
    python scripts/benchmark_detectors.py
    python scripts/benchmark_detectors.py --images data/test_images
    python scripts/benchmark_detectors.py --include-retinaface

Outputs:
    outputs/benchmark_results.json   — full per-image data
    stdout                           — formatted comparison table

Privacy/Utility scoring methodology
------------------------------------
  privacy_score  = detection_rate × (0.6 + 0.4 × mean_confidence)
                   (detects all faces with high confidence = privacy protected)
  utility_score  = 1 − (total_blur_area / total_image_area)
                   (less image blurred = more usable for analytics)
  p×u            = privacy_score × utility_score  (combined metric)
  latency_ms     = wall-clock time for detector.detect() per frame (mean)

These match the simulated scores in textgrad_pipeline_v2.py and allow
direct comparison with the Three-Era table in overleaf_paper/main.tex.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# allow running from repo root or scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from privacy_pipeline.anonymizer.detectors import (
    FaceDetector, YuNetDetector, create_face_detector, DetectedRegion,
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def blur_area_fraction(regions: List[DetectedRegion], h: int, w: int) -> float:
    """Fraction of image covered by detected (blurred) regions."""
    if not regions:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        x2 = min(r.x + r.w, w)
        y2 = min(r.y + r.h, h)
        mask[max(0, r.y):y2, max(0, r.x):x2] = 1
    return float(mask.sum()) / (h * w)


def score_frame(
    regions: List[DetectedRegion],
    h: int,
    w: int,
    ground_truth_faces: int,
) -> Dict[str, float]:
    """Compute privacy, utility, and P×U for one frame."""
    detected = len(regions)
    detection_rate = min(detected / max(ground_truth_faces, 1), 1.0)
    mean_conf = (
        float(np.mean([r.confidence for r in regions])) if regions else 0.0
    )
    privacy = round(detection_rate * (0.6 + 0.4 * mean_conf), 4)
    utility = round(1.0 - blur_area_fraction(regions, h, w), 4)
    pxu = round(privacy * utility, 4)
    return {
        "detected": detected,
        "ground_truth": ground_truth_faces,
        "detection_rate": round(detection_rate, 4),
        "mean_confidence": round(mean_conf, 4),
        "privacy_score": privacy,
        "utility_score": utility,
        "pxu": pxu,
    }


# ---------------------------------------------------------------------------
# Detector runner
# ---------------------------------------------------------------------------

def run_detector(detector, images: List[Path]) -> Dict[str, Any]:
    """Run detector over all images, return aggregated results."""
    per_image = []
    latencies = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not load {img_path.name}", flush=True)
            continue
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        regions = detector.detect(frame)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        metrics = score_frame(regions, h, w, ground_truth_faces=1)
        metrics["image"] = img_path.name
        metrics["latency_ms"] = round(latency_ms, 2)
        per_image.append(metrics)

    if not per_image:
        return {}

    privacy_scores = [x["privacy_score"] for x in per_image]
    utility_scores = [x["utility_score"] for x in per_image]
    pxu_scores     = [x["pxu"] for x in per_image]
    det_rates      = [x["detection_rate"] for x in per_image]

    return {
        "per_image": per_image,
        "summary": {
            "mean_privacy":        round(float(np.mean(privacy_scores)), 4),
            "mean_utility":        round(float(np.mean(utility_scores)), 4),
            "mean_pxu":            round(float(np.mean(pxu_scores)), 4),
            "mean_detection_rate": round(float(np.mean(det_rates)), 4),
            "mean_latency_ms":     round(float(np.mean(latencies)), 2),
            "images_tested":       len(per_image),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PrivGrad detector benchmark")
    parser.add_argument(
        "--images", default=str(REPO_ROOT / "data" / "test_images"),
        help="Directory of test images (default: data/test_images)"
    )
    parser.add_argument(
        "--include-retinaface", action="store_true",
        help="Also benchmark RetinaFace (requires: pip install retina-face)"
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not images:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)

    print(f"\nPrivGrad — Three-Era Detection Benchmark")
    print(f"NTU CCDS | Assoc Prof Chee Wei Tan | June 2026")
    print(f"{'='*60}")
    print(f"Images: {len(images)} frames from {image_dir}")
    print()

    model_path = str(REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx")
    detectors = {
        "Haar Cascade (~2022)": create_face_detector("haar_cascade"),
        "YuNet/PrivGrad (2026)": create_face_detector("yunet", model_path=model_path),
    }

    if args.include_retinaface:
        try:
            detectors["RetinaFace (optional)"] = create_face_detector("retinaface")
            print("RetinaFace loaded.")
        except ImportError as e:
            print(f"RetinaFace skipped: {e}\n")

    results: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}

    for name, detector in detectors.items():
        print(f"Running: {name} ...", flush=True)
        data = run_detector(detector, images)
        results[name] = data
        summaries[name] = data.get("summary", {})

    # ── Print comparison table ─────────────────────────────────────────────
    print()
    print(f"{'-'*66}")
    print(f"{'Era / Detector':<26} {'Privacy':>8} {'Utility':>8} {'P×U':>8} {'Det.Rate':>9} {'ms/frame':>9}")
    print(f"{'-'*66}")

    # Pre-AI row — historical fixed values from the paper
    print(f"{'Pre-AI (~2018)':<26} {'0.8500':>8} {'0.4500':>8} {'0.3825':>8} {'0.7200':>9} {'~45ms':>9}")
    print(f"  Fixed Gaussian blur, no ML")

    # Haar Cascade
    s = summaries.get("Haar Cascade (~2022)", {})
    if s:
        print(f"{'Haar Cascade (~2022)':<26} {s['mean_privacy']:>8.4f} {s['mean_utility']:>8.4f} {s['mean_pxu']:>8.4f} {s['mean_detection_rate']:>9.4f} {s['mean_latency_ms']:>8.1f}ms")
        print(f"  cv2.CascadeClassifier, manual tuning")

    # YuNet / PrivGrad
    s = summaries.get("YuNet/PrivGrad (2026)", {})
    if s:
        print(f"{'YuNet/PrivGrad (2026)':<26} {s['mean_privacy']:>8.4f} {s['mean_utility']:>8.4f} {s['mean_pxu']:>8.4f} {s['mean_detection_rate']:>9.4f} {s['mean_latency_ms']:>8.1f}ms")
        print(f"  cv2.FaceDetectorYN (ONNX) + TextGrad loop")

    # RetinaFace (optional)
    s = summaries.get("RetinaFace (optional)", {})
    if s:
        print(f"{'RetinaFace (optional)':<26} {s['mean_privacy']:>8.4f} {s['mean_utility']:>8.4f} {s['mean_pxu']:>8.4f} {s['mean_detection_rate']:>9.4f} {s['mean_latency_ms']:>8.1f}ms")
        print(f"  pip install retina-face")

    print(f"{'-'*66}")

    # ── Per-image detail ───────────────────────────────────────────────────
    print()
    print("Per-image detail (YuNet/PrivGrad):")
    yunet_data = results.get("YuNet/PrivGrad (2026)", {}).get("per_image", [])
    if yunet_data:
        print(f"  {'Image':<14} {'Detected':>8} {'Privacy':>8} {'Utility':>8} {'P×U':>8} {'Conf':>6} {'ms':>6}")
        print(f"  {'-'*62}")
        for row in yunet_data:
            print(
                f"  {row['image']:<14} {row['detected']:>8} "
                f"{row['privacy_score']:>8.4f} {row['utility_score']:>8.4f} "
                f"{row['pxu']:>8.4f} {row['mean_confidence']:>6.2f} "
                f"{row['latency_ms']:>5.1f}ms"
            )

    # ── Save JSON ──────────────────────────────────────────────────────────
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "benchmark_results.json"

    import datetime as _dt
    output = {
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "images_tested": len(images),
        "image_dir": str(image_dir),
        "results": results,
        "paper_table": {
            "pre_ai_2018":    {"privacy": 0.85, "utility": 0.45, "pxu": 0.38, "notes": "Fixed Gaussian blur, no detection"},
            "haar_2022":      summaries.get("Haar Cascade (~2022)", {}),
            "yunet_privgrad": summaries.get("YuNet/PrivGrad (2026)", {}),
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved: {output_path}")
    print()
    print("Next: copy the YuNet/PrivGrad row into overleaf_paper/main.tex Table 1.")
    print("      Run --include-retinaface once pip install retina-face is done.")


if __name__ == "__main__":
    main()
