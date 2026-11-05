"""
prepare_lfw_gallery.py -- build data/reid_gallery/ from an LFW download
=======================================================================
Called by hand once per machine, before reid_attack_lfw.py or
privacy_utility_sweep.py. LFW is not redistributed in this repo.

Takes lfw.tgz / lfw-deepfunneled.tgz / a .zip, or an already-extracted folder
(one sub-folder of .jpg per person, e.g. Kaggle's "lfw-dataset"). Picks the
first N identities (alphabetical) with at least K photos and copies their
first K photos, so the subset is deterministic and documented.

Usage:
    python scripts/prepare_lfw_gallery.py data/lfw.tgz
    python scripts/prepare_lfw_gallery.py ~/Downloads/lfw-deepfunneled --ids 100 --per-id 5
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_people_root(root: Path) -> Path:
    """The folder whose children are per-person folders of .jpg."""
    best, best_n = root, 0
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        n = sum(1 for c in d.iterdir() if c.is_dir() and any(c.glob("*.jpg")))
        if n > best_n:
            best, best_n = d, n
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--ids", type=int, default=100)
    ap.add_argument("--per-id", type=int, default=5)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "reid_gallery")
    args = ap.parse_args()

    tmp = None
    src = args.source.expanduser()
    if src.is_file():
        tmp = Path(tempfile.mkdtemp())
        if src.suffix == ".zip":
            zipfile.ZipFile(src).extractall(tmp)
        else:
            with tarfile.open(src) as t:
                t.extractall(tmp)
        src = tmp
    root = find_people_root(src)

    people = [p for p in sorted(root.iterdir()) if p.is_dir()
              and len(list(p.glob("*.jpg"))) >= args.per_id][: args.ids]
    if len(people) < args.ids:
        raise SystemExit(f"only {len(people)} identities with >= {args.per_id} photos under {root}")
    shutil.rmtree(args.out, ignore_errors=True)
    for p in people:
        (args.out / p.name).mkdir(parents=True)
        for img in sorted(p.glob("*.jpg"))[: args.per_id]:
            shutil.copy(img, args.out / p.name / img.name)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {len(people)} identities x {args.per_id} photos to {args.out} (from {root})")


if __name__ == "__main__":
    main()
