"""
privacy_utility_sweep.py -- privacy vs utility curve over anonymization settings
================================================================================
Called by hand to produce results/privacy_utility_<date>.json (+ .csv, .png);
rerun when the anonymizer, attacker or gallery changes.

For every (method, strength, mode, padding) in the grid and every face scale,
scores the aligned re-ID attacker (privacy) and detector survival (utility)
with privacy_pipeline.attacks.reid_eval, then marks the Pareto front.

--scales simulates smaller faces: probes are downscaled (the attacker keeps a
full-resolution reference photo), which is what CCTV-sized faces look like.
Absolute vs relative strength at several scales shows whether a fixed pixel
setting stays safe as faces shrink.

Usage:
    python scripts/privacy_utility_sweep.py --gallery-dir data/reid_gallery
    python scripts/privacy_utility_sweep.py --quick          # small grid
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from privacy_pipeline.attacks.reid_eval import (  # noqa: E402
    DEFAULT_ARCFACE,
    AnonSpec,
    ReIDEvaluator,
)


def grid(quick: bool) -> list:
    if quick:
        return (
            [AnonSpec("clear")]
            + [AnonSpec("gaussian", k) for k in (45, 75)]
            + [AnonSpec("pixelate", b) for b in (8, 16)]
            + [AnonSpec("pixelate", n, "relative") for n in (4, 8)]
            + [AnonSpec("solid")]
        )
    g = [AnonSpec("clear"), AnonSpec("solid")]
    g += [AnonSpec("gaussian", k) for k in (15, 31, 45, 61, 75, 101, 151)]
    g += [AnonSpec("pixelate", b) for b in (4, 6, 8, 10, 12, 14, 16, 20, 25)]
    g += [AnonSpec("pixelate", n, "relative") for n in (3, 4, 5, 6, 8, 10, 12)]
    g += [AnonSpec("gaussian", p, "relative") for p in (15, 30, 50, 75, 100)]
    # padding matters for alignment: does a bigger margin hide the landmarks?
    g += [AnonSpec("pixelate", 6, "relative", pad) for pad in (0, 30)]
    g += [AnonSpec("gaussian", 75, padding=pad) for pad in (0, 30)]
    return g


def pareto(rows: list) -> None:
    """Mark rows not dominated on (privacy up, utility up) within each scale."""
    for r in rows:
        r["pareto"] = not any(
            o is not r
            and o["scale"] == r["scale"]
            and o["privacy"] >= r["privacy"]
            and o["utility_detect"] >= r["utility_detect"]
            and (o["privacy"] > r["privacy"] or o["utility_detect"] > r["utility_detect"])
            for o in rows
        )


def plot(rows: list, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    scales = sorted({r["scale"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(scales), figsize=(5.2 * len(scales), 4.4), squeeze=False)
    styles = {
        ("gaussian", "absolute"): ("o", "#2a6fdb"),
        ("gaussian", "relative"): ("o", "#9cc0f5"),
        ("pixelate", "absolute"): ("s", "#d9480f"),
        ("pixelate", "relative"): ("s", "#f5a97f"),
        ("solid", "absolute"): ("X", "#333333"),
        ("clear", "absolute"): ("*", "#2b8a3e"),
    }
    for ax, sc in zip(axes[0], scales):
        sub = [r for r in rows if r["scale"] == sc]
        for (m, mode), (mk, col) in styles.items():
            pts = [r for r in sub if r["spec"]["method"] == m and r["spec"]["mode"] == mode]
            if pts:
                ax.scatter(
                    [r["attack_acc"] * 100 for r in pts],
                    [r["utility_detect"] * 100 for r in pts],
                    marker=mk, c=col, s=42, label=f"{m} ({mode})" if m not in ("clear", "solid") else m,
                    edgecolors="white", linewidths=0.6,
                )
        ax.axvline(sub[0]["chance"] * 100, ls="--", c="grey", lw=0.8)
        ax.set_xlabel("re-ID attack top-1 accuracy (%)  <- more private")
        ax.set_ylabel("face still detected after anonymization (%)")
        ax.set_title(f"probe scale {sc:g} (median face {sub[0]['median_face_px']:.0f} px)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-dir", type=Path, default=REPO_ROOT / "data" / "reid_gallery")
    ap.add_argument("--yunet", type=Path,
                    default=REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx")
    ap.add_argument("--arcface", default=DEFAULT_ARCFACE)
    ap.add_argument("--scales", default="1,0.5,0.3")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-prefix", type=Path,
                    default=REPO_ROOT / "outputs" / f"privacy_utility_{datetime.date.today()}")
    args = ap.parse_args()

    t0, rows = time.time(), []
    for sc in (float(s) for s in args.scales.split(",")):
        ev = ReIDEvaluator(args.gallery_dir, args.yunet, args.arcface, splits=args.splits, scale=sc)
        print(f"scale {sc:g}: {len(ev.items)} faces, "
              f"{len(ev.uniq)} ids, {ev.missed} missed, median face {ev.face_px:.0f}px, "
              f"chance {ev.chance:.3f}, clear acc {ev.upper_bound:.3f}", flush=True)
        for spec in grid(args.quick):
            r = ev.evaluate(spec)
            rows.append(r)
            print(f"  {spec.key:<26} attack {r['attack_acc']:.3f}±{r['attack_std']:.3f}"
                  f"  oracle {r['attack_acc_oracle']:.3f}  detect {r['utility_detect']:.3f}"
                  f"  t={time.time() - t0:.0f}s", flush=True)
    pareto(rows)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    meta = {"gallery": str(args.gallery_dir), "splits": args.splits, "scales": args.scales,
            "attacker": "ArcFace w600k_r50, 5-point aligned on YuNet re-detected landmarks",
            "utility": "YuNet re-detects the face at IoU>=0.5 on the anonymized frame",
            "date": str(datetime.date.today()), "seconds": round(time.time() - t0)}
    Path(f"{args.out_prefix}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale", "method", "mode", "strength", "padding", "attack_acc", "attack_std",
                    "attack_acc_oracle", "privacy", "utility_detect", "pareto"])
        for r in rows:
            s = r["spec"]
            w.writerow([r["scale"], s["method"], s["mode"], s["strength"], s["padding"],
                        f"{r['attack_acc']:.4f}", f"{r['attack_std']:.4f}",
                        f"{r['attack_acc_oracle']:.4f}", f"{r['privacy']:.4f}",
                        f"{r['utility_detect']:.4f}", r["pareto"]])
    plot(rows, Path(f"{args.out_prefix}.png"))
    print(f"\nPareto front (privacy >= 0.95 shown first):")
    for r in sorted((r for r in rows if r["pareto"]), key=lambda r: (r["scale"], -r["privacy"])):
        print(f"  scale {r['scale']:<4g} {AnonSpec(**r['spec']).key:<26} attack {r['attack_acc']:.3f}"
              f"  detect {r['utility_detect']:.3f}")
    print(f"wrote {args.out_prefix}.json/.csv/.png  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
