"""
reid_attack_benchmark.py -- Attack-Based Privacy Metric (Re-Identification)
============================================================================
Replaces benchmark_detectors.py's hand-tuned
    privacy_score = recall x (0.6 + 0.4 x mean_confidence)
with an actual adversary: a face-recognition attacker (ArcFace/w600k_r50)
that tries to re-identify a person from the anonymizer's OUTPUT (blurred
region), scored against a gallery of clear reference photos.

Metric:
  attack_accuracy(strength) = attacker's top-1 identification accuracy on
                               frames blurred at that strength
  chance_accuracy           = 1 / gallery_size  (random-guess floor)
  upper_bound_accuracy      = attack_accuracy on the UNBLURRED frame
  privacy_score(strength)   = 1 - (attack_accuracy - chance) / (upper_bound - chance)
                               clamped to [0, 1]; 1.0 = attacker at chance
                               (best privacy), 0.0 = attacker as good as on
                               the clear frame (no privacy benefit at all)

A missed detection (face never blurred) is not given a separate penalty term
here -- it doesn't need one. The frame reaching the attacker unblurred IS the
attack succeeding at (or near) upper_bound_accuracy on exactly that face, so
FNs surface directly in attack_accuracy instead of through a hand-picked
recall weight.

REQUIRES A LABELED DATASET this repo does not yet ship: a directory of
per-identity face photos (at least 2 images each -- one enrolled in the
gallery, the rest used as probes). WiderFace has no identity labels, so it
cannot be reused here. Point --gallery-dir at something like an LFW subset:
    data/reid_gallery/<person_name>/*.jpg

Without a real dataset, --smoke-test runs an end-to-end plumbing check using
single WiderFace crops as one-image "identities" (gallery photo == probe
photo, just re-cropped) -- this proves the code path works but is NOT a
privacy result: an attacker matching a photo to itself is a much easier task
than matching two different photos of the same person, so smoke-test numbers
will look artificially strong. Do not report them as findings.

Usage:
    python scripts/reid_attack_benchmark.py --smoke-test
    python scripts/reid_attack_benchmark.py --gallery-dir data/reid_gallery

Outputs:
    outputs/reid_attack_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from privacy_pipeline.anonymizer.blurrer import RegionBlurrer
from privacy_pipeline.anonymizer.detectors import DetectedRegion, create_face_detector
from privacy_pipeline.attacks.reid_attacker import ArcFaceEmbedder, ReIDAttacker

WIDER_VAL_IMAGES = (
    REPO_ROOT / "data" / "test_images" / "WIDER_val" / "WIDER_val" / "images"
)

# Blur strengths to sweep -- mirrors the range TextGrad's config already
# tunes for the "gaussian" blur method in blurrer.py.
DEFAULT_STRENGTHS = [0, 5, 15, 25, 45, 75]


def load_gallery_dataset(gallery_dir: Path) -> Dict[str, List[Path]]:
    """<gallery_dir>/<identity>/*.jpg|png -> {identity: [image paths]}."""
    identities: Dict[str, List[Path]] = {}
    if not gallery_dir.exists():
        return identities
    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if len(images) >= 2:
            identities[person_dir.name] = images
    return identities


def build_smoke_test_dataset(n_identities: int = 15) -> Dict[str, List[Path]]:
    """Pick N WiderFace images and treat each as its own single-photo identity.

    Gallery photo and probe photo are literally the same file -- see the
    module docstring's warning about why this over-states attacker accuracy.
    """
    if not WIDER_VAL_IMAGES.exists():
        raise FileNotFoundError(f"WiderFace images not found at {WIDER_VAL_IMAGES}")
    all_images = sorted(WIDER_VAL_IMAGES.rglob("*.jpg"))[: n_identities * 3]
    chosen = all_images[:n_identities]
    return {f"smoke_{i:03d}": [path, path] for i, path in enumerate(chosen)}


def largest_face(frame: np.ndarray, regions: List[DetectedRegion]) -> DetectedRegion | None:
    if not regions:
        return None
    return max(regions, key=lambda r: r.w * r.h)


def run_attack(
    dataset: Dict[str, List[Path]],
    strengths: List[int],
    blur_method: str = "gaussian",
) -> Dict[str, Any]:
    detector = create_face_detector(
        "yunet",
        model_path=str(REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx"),
        padding=0,  # evaluation, not blur-time margin -- see benchmark_detectors.py
    )
    embedder = ArcFaceEmbedder()

    gallery_paths = {identity: paths[0] for identity, paths in dataset.items()}
    probe_paths = {identity: paths[1:] for identity, paths in dataset.items()}
    chance_accuracy = 1.0 / len(dataset) if dataset else 0.0

    results: Dict[str, Any] = {"chance_accuracy": round(chance_accuracy, 4), "strengths": {}}

    for strength in strengths:
        blurrer = None if strength == 0 else RegionBlurrer(method=blur_method, strength=strength)
        attacker = ReIDAttacker(embedder)

        # Enroll gallery from CLEAR reference photos -- the attacker owns
        # these independently (e.g. a social-media profile photo); they are
        # never run through the anonymizer.
        skipped_enroll = 0
        for identity, path in gallery_paths.items():
            frame = cv2.imread(str(path))
            region = largest_face(frame, detector.detect(frame))
            if region is None:
                skipped_enroll += 1
                continue
            attacker.enroll(identity, attacker.crop_region(frame, region))

        correct = 0
        total = 0
        for identity, paths in probe_paths.items():
            for probe_path in paths:
                frame = cv2.imread(str(probe_path))
                region = largest_face(frame, detector.detect(frame))
                if region is None:
                    continue
                if blurrer is not None:
                    frame = blurrer.apply(frame, [region])
                crop = attacker.crop_region(frame, region)
                try:
                    predicted_identity, _sim = attacker.identify(crop)
                except (RuntimeError, ValueError):
                    continue
                total += 1
                correct += int(predicted_identity == identity)

        accuracy = correct / total if total else 0.0
        results["strengths"][str(strength)] = {
            "attack_accuracy": round(accuracy, 4),
            "n_probes_scored": total,
            "n_identities_enrolled": len(gallery_paths) - skipped_enroll,
        }

    upper_bound = results["strengths"].get("0", {}).get("attack_accuracy", chance_accuracy)
    denom = max(upper_bound - chance_accuracy, 1e-6)
    for entry in results["strengths"].values():
        raw = 1.0 - (entry["attack_accuracy"] - chance_accuracy) / denom
        entry["privacy_score"] = round(min(1.0, max(0.0, raw)), 4)

    results["upper_bound_accuracy"] = round(upper_bound, 4)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery-dir", type=Path, default=REPO_ROOT / "data" / "reid_gallery")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Use single-photo WiderFace pseudo-identities. NOT a privacy result.")
    parser.add_argument("--strengths", type=int, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument("--blur-method", choices=RegionBlurrer.SUPPORTED_METHODS, default="gaussian")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "reid_attack_results.json")
    args = parser.parse_args()

    if args.smoke_test:
        print("SMOKE TEST MODE -- gallery photo == probe photo. Plumbing check only, not a finding.")
        dataset = build_smoke_test_dataset()
    else:
        dataset = load_gallery_dataset(args.gallery_dir)
        if not dataset:
            print(f"No labeled identities found under {args.gallery_dir}.")
            print("Populate it as <gallery-dir>/<person_name>/*.jpg (>=2 photos each, e.g. an LFW subset),")
            print("or pass --smoke-test to verify the pipeline runs without real data.")
            sys.exit(1)

    print(f"Running re-identification attack across {len(dataset)} identities, "
          f"strengths={args.strengths}, blur={args.blur_method}")
    results = run_attack(dataset, args.strengths, args.blur_method)

    print(f"\nchance accuracy:       {results['chance_accuracy']:.4f}")
    print(f"upper-bound accuracy:  {results['upper_bound_accuracy']:.4f}  (strength=0, unblurred)")
    print(f"\n{'strength':>8} | {'attack_acc':>10} | {'privacy_score':>13} | n_probes")
    for strength, entry in results["strengths"].items():
        print(f"{strength:>8} | {entry['attack_accuracy']:>10.4f} | "
              f"{entry['privacy_score']:>13.4f} | {entry['n_probes_scored']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
