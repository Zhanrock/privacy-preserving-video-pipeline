"""
Haar Cascade vs YuNet (DNN) detection benchmark — Step 3 of PrivGrad.

Two face detectors compared across 5 portrait images:
  Haar Cascade  — rule-based (Viola-Jones 2001), OpenCV built-in
  YuNet (DNN)   — deep-learning face detector via cv2.FaceDetectorYN,
                  same class of improvement as RetinaFace, runs through
                  OpenCV's C++ ONNX loader (Python 3.14 safe, no insightface,
                  no protobuf, no TensorFlow)

Model: face_detection_yunet_2023mar.onnx (~371 KB, downloaded once from
       OpenCV's official model zoo on first run)

Output: detection_comparison.csv
  Columns: filename, haar_count, retinaface_count, winner
  (column named retinaface_count to match the rest of the pipeline schema)
"""

import os
import sys
import csv
import urllib.request
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
CSV_PATH   = os.path.join(SCRIPT_DIR, "detection_comparison.csv")

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_PATH     = os.path.join(MODELS_DIR, YUNET_FILENAME)
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

IMAGE_SOURCES = [
    ("face_01.jpg", "https://randomuser.me/api/portraits/men/1.jpg"),
    ("face_02.jpg", "https://randomuser.me/api/portraits/women/2.jpg"),
    ("face_03.jpg", "https://randomuser.me/api/portraits/men/33.jpg"),
    ("face_04.jpg", "https://randomuser.me/api/portraits/women/44.jpg"),
    ("face_05.jpg", "https://randomuser.me/api/portraits/men/55.jpg"),
]

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _download(url: str, dest: str, label: str) -> bool:
    if os.path.exists(dest):
        print(f"  [skip] {label} (cached)")
        return True
    print(f"  [get]  {label} ...")
    try:
        req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  [ok]   {label} ({len(data):,} bytes)")
        return True
    except Exception as exc:
        print(f"  [fail] {label}: {exc}")
        return False


def download_images() -> list[str]:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    paths = []
    for filename, url in IMAGE_SOURCES:
        dest = os.path.join(IMAGES_DIR, filename)
        if _download(url, dest, filename):
            paths.append(dest)
    return paths


def download_yunet() -> bool:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return _download(YUNET_URL, YUNET_PATH, YUNET_FILENAME)


def detect_haar(img: np.ndarray) -> int:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces)


def detect_yunet(detector: cv2.FaceDetectorYN, img: np.ndarray) -> int:
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    return 0 if faces is None else len(faces)


def main() -> None:
    print("=" * 62)
    print("Detection Benchmark: Haar Cascade vs YuNet (DNN / RetinaFace-class)")
    print("=" * 62)

    print("\n[1] Test images")
    image_paths = download_images()
    if not image_paths:
        print("ERROR: no images downloaded — check internet connection.")
        sys.exit(1)

    print("\n[2] YuNet model")
    if not download_yunet():
        print("ERROR: could not download YuNet model.")
        sys.exit(1)

    detector = cv2.FaceDetectorYN.create(
        model=YUNET_PATH,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

    print("\n[3] Running detection")
    print(f"\n  {'Filename':<15}  {'Haar':>6}  {'YuNet(DNN)':>12}  {'Winner':>10}")
    print("  " + "-" * 52)

    rows: list[dict] = []
    for path in image_paths:
        filename = os.path.basename(path)
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2.imread returned None")
            haar_n  = detect_haar(img)
            yunet_n = detect_yunet(detector, img)
            winner  = (
                "equal" if haar_n == yunet_n
                else "yunet" if yunet_n > haar_n
                else "haar"
            )
            print(f"  {filename:<15}  {haar_n:>6}  {yunet_n:>12}  {winner:>10}")
            rows.append({
                "filename": filename,
                "haar_count": haar_n,
                "retinaface_count": yunet_n,
                "winner": winner,
            })
        except Exception as exc:
            print(f"  {filename:<15}  ERROR: {exc}")
            rows.append({
                "filename": filename,
                "haar_count": "ERR",
                "retinaface_count": "ERR",
                "winner": "ERR",
            })

    print()
    valid = [r for r in rows if r["haar_count"] != "ERR"]
    if valid:
        haar_total  = sum(r["haar_count"]        for r in valid)
        yunet_total = sum(r["retinaface_count"]  for r in valid)
        print(f"  Totals — Haar: {haar_total}  |  YuNet(DNN): {yunet_total}")
        diff = yunet_total - haar_total
        if diff > 0:
            print(f"  YuNet found {diff} more face(s) — DNN model outperforms rule-based.")
        elif diff == 0:
            print("  Equal detections — expected for clean frontal portrait images.")
        else:
            print(f"  Haar found {-diff} more face(s) — check image quality.")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "haar_count", "retinaface_count", "winner"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV saved: {CSV_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
