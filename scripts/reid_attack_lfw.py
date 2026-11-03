"""
reid_attack_lfw.py -- re-identification attack on an identity-labelled LFW subset
================================================================================
Extends reid_attack_benchmark.py in three ways, without changing it:

1. Three attackers, weakest to strongest:
     unaligned        plain resize of the face box (the original scaffold)
     aligned_redetect 5-point ArcFace alignment using landmarks YuNet finds on
                      the ANONYMIZED frame; falls back to unaligned if YuNet
                      finds nothing there (a realistic attacker)
     aligned_oracle   5-point alignment using landmarks from the CLEAR frame
                      (the attacker somehow knows exact landmark positions;
                      an upper bound on this attacker family)
   Closes the "no landmark alignment" gap noted in the research decisions.
2. Three anonymizers from blurrer.py: gaussian, pixelate, solid.
3. Error bars: R random gallery splits (which photo per identity is enrolled
   as the clear reference); report mean +- std of top-1 accuracy.

Embeddings for every image under every condition are computed once and the
splits re-index them, so R costs nothing extra.

Detection runs once per image on the clear frame (the anonymizer's own
detect -> blur step). Faces YuNet misses are excluded and counted; on LFW
that is a handful, so detector FNs are not what this run measures.

Called by hand to produce results/reid_lfw_<date>.json; rerun when the anonymizer
or the TextGrad-tuned configs change.

Usage:
    python scripts/reid_attack_lfw.py --gallery-dir data/reid_gallery
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from privacy_pipeline.anonymizer.blurrer import RegionBlurrer  # noqa: E402
from privacy_pipeline.anonymizer.detectors import DetectedRegion  # noqa: E402
from privacy_pipeline.attacks.reid_attacker import _DEFAULT_MODEL_PATH  # noqa: E402

ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

CONDITIONS = (
    [("clear", 0)]
    + [("gaussian", s) for s in (15, 45, 75)]
    + [("pixelate", s) for s in (8, 16)]
)
# "solid" (black box) is left out: every crop is identical, so any attacker
# is at chance by construction and embedding it only burns CPU.
ATTACKERS = ("unaligned", "aligned_redetect", "aligned_oracle")


class Embedder:
    def __init__(self, path: str = _DEFAULT_MODEL_PATH) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name

    def embed(self, crops_bgr: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(crops_bgr), batch):
            x = np.stack([
                np.transpose((cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5, (2, 0, 1))
                for c in crops_bgr[i:i + batch]])
            (f,) = self.s.run(None, {self.name: x})
            out.append(f)
        v = np.concatenate(out)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)


def make_detector(pad: int):
    det = cv2.FaceDetectorYN.create(
        str(REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "", (320, 320),
        score_threshold=0.45, nms_threshold=0.3)

    def detect(frame):
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda f: f[2] * f[3])
        region = DetectedRegion(int(f[0]), int(f[1]), int(f[2]), int(f[3]), "face",
                                confidence=float(f[14])).with_padding(pad, h, w)
        return region, f[4:14].reshape(5, 2).astype(np.float32)
    return detect


def crop_unaligned(frame, r: DetectedRegion):
    h, w = frame.shape[:2]
    c = frame[max(0, r.y):min(h, r.y2), max(0, r.x):min(w, r.x2)]
    return cv2.resize(c, (112, 112))


def crop_aligned(frame, lmk):
    M, _ = cv2.estimateAffinePartial2D(lmk, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-dir", type=Path, default=REPO_ROOT / "data" / "reid_gallery")
    ap.add_argument("--padding", type=int, default=15, help="blur-time box padding (pipeline default 15)")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "reid_lfw_results.json")
    args = ap.parse_args()

    t0 = time.time()
    detect = make_detector(args.padding)
    emb = Embedder()

    items, missed = [], 0
    for person in sorted(p for p in args.gallery_dir.iterdir() if p.is_dir()):
        for img in sorted(person.glob("*.jpg")):
            frame = cv2.imread(str(img))
            d = detect(frame)
            if d is None:
                missed += 1
                continue
            items.append((person.name, frame, d[0], d[1]))
    ids = np.array([i[0] for i in items])
    uniq = sorted(set(ids))
    print(f"{len(items)} faces, {len(uniq)} identities, {missed} missed by YuNet on clear frames")

    # embeddings[attacker][condition] -> (N, 512)
    E = {a: {} for a in ATTACKERS}
    redetect_rate = {}
    for method, strength in CONDITIONS:
        key = f"{method}_{strength}"
        blurrer = None if method == "clear" else RegionBlurrer(method=method, strength=strength)
        un, orc, re_idx, re_crops = [], [], [], []
        for n, (_, frame, region, lmk) in enumerate(items):
            anon = frame if blurrer is None else blurrer.apply(frame, [region])
            un.append(crop_unaligned(anon, region))
            orc.append(crop_aligned(anon, lmk))
            d = detect(anon)
            if d is not None:
                re_idx.append(n)
                re_crops.append(crop_aligned(anon, d[1]))
        E["unaligned"][key] = emb.embed(un)
        E["aligned_oracle"][key] = emb.embed(orc)
        # redetect attacker = aligned on re-found landmarks, else falls back to unaligned
        r = E["unaligned"][key].copy()
        if re_crops:
            r[re_idx] = emb.embed(re_crops)
        E["aligned_redetect"][key] = r
        redetect_rate[key] = len(re_idx) / len(items)
        print(f"  {key:<12} embedded  (YuNet re-finds face on anonymized frame: {redetect_rate[key]:.0%})"
              f"  t={time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    splits = []
    for _ in range(args.splits):
        g = np.array([rng.choice(np.where(ids == u)[0]) for u in uniq])
        p = np.setdiff1d(np.arange(len(items)), g)
        splits.append((g, p))
    chance = 1 / len(uniq)

    results = {"n_faces": len(items), "n_identities": len(uniq), "yunet_missed_clear": missed,
               "padding": args.padding, "splits": args.splits, "chance": chance,
               "redetect_rate": redetect_rate, "attack_accuracy": {}}
    for a in ATTACKERS:
        results["attack_accuracy"][a] = {}
        for method, strength in CONDITIONS:
            key = f"{method}_{strength}"
            accs = []
            for g, p in splits:
                gal = E[a]["clear_0"][g]          # gallery is always the clear photo
                sims = E[a][key][p] @ gal.T
                pred = np.array(uniq)[sims.argmax(1)]
                accs.append(float((pred == ids[p]).mean()))
            results["attack_accuracy"][a][key] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}

    for a in ATTACKERS:
        ub = results["attack_accuracy"][a]["clear_0"]["mean"]
        for key, v in results["attack_accuracy"][a].items():
            v["privacy_score"] = float(min(1, max(0, 1 - (v["mean"] - chance) / max(ub - chance, 1e-6))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nchance = {chance:.3f}   ({len(uniq)} identities, {args.splits} gallery splits, padding={args.padding})")
    print(f"{'condition':<12}" + "".join(f"{a:>22}" for a in ATTACKERS))
    for method, strength in CONDITIONS:
        key = f"{method}_{strength}"
        row = "".join(f"{results['attack_accuracy'][a][key]['mean']:>15.3f} ±{results['attack_accuracy'][a][key]['std']:.3f}"
                      for a in ATTACKERS)
        print(f"{key:<12}{row}")
    print(f"\nwrote {args.output}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
