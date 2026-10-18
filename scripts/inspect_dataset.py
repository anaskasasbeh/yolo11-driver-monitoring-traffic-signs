"""Describe a YOLO-format dataset and check it for common problems.

Reports, for each split:
  * number of images, label files, background images and objects per class
  * Roboflow augmentation: several exported copies of one source image in a split
  * leakage: the same source image (or an identical file) in more than one split
  * GTSRB track overlap: frames of the same physical sign in different splits.
    GTSRB file names look like "00000_00001_00020" (class_track_frame); frames of
    one track are near-identical, so a track split across train and test inflates
    test scores. This check is a heuristic based on file names.

Example:
    python scripts/inspect_dataset.py --data configs/tsd.yaml --json results/tsd_dataset_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
GTSRB_PATTERN = re.compile(r"^(\d{5})_(\d{5})_\d{5}")


def resolve_split_dirs(yaml_path: Path) -> tuple[dict[str, Path], dict[int, str]]:
    """Read a dataset yaml and return its split folders and class names.

    Paths are resolved the way Ultralytics does: relative to ``path`` if given,
    otherwise to the yaml's folder, with a fallback for Roboflow's "../train" style.
    """
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = yaml_path.parent
    if config.get("path"):
        root = (yaml_path.parent / config["path"]).resolve()  # an absolute path stays absolute

    names = config["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))

    dirs = {}
    for split in SPLITS:
        value = config.get(split)
        if not value:
            continue
        folder = (root / value).resolve()
        if not folder.exists() and value.startswith("../"):
            folder = (root / value[3:]).resolve()
        dirs[split] = folder
    return dirs, {int(k): v for k, v in names.items()}


def label_path(image: Path) -> Path:
    """Map .../images/x.jpg to .../labels/x.txt (Ultralytics convention)."""
    parts = list(image.parts)
    if "images" not in parts:
        return image.with_suffix(".txt")
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_key(image: Path) -> str:
    """Name of the original image before Roboflow added ".rf.<hash>"."""
    return image.stem.split(".rf.")[0]


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def inspect_split(folder: Path, names: dict[int, str]) -> dict:
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    objects: Counter[str] = Counter()
    images_with: Counter[str] = Counter()
    missing_labels = backgrounds = 0
    unknown_ids: Counter[int] = Counter()

    for image in images:
        label = label_path(image)
        if not label.is_file():
            missing_labels += 1
            continue
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            backgrounds += 1
            continue
        seen_here = set()
        for row in rows:
            class_id = int(float(row[0]))
            if class_id not in names:
                unknown_ids[class_id] += 1
                continue
            objects[names[class_id]] += 1
            seen_here.add(names[class_id])
        images_with.update(seen_here)

    keys = Counter(source_key(p) for p in images)
    return {
        "images": len(images),
        "missing_labels": missing_labels,
        "background_images": backgrounds,
        "objects_per_class": {n: objects.get(n, 0) for n in names.values()},
        "images_per_class": {n: images_with.get(n, 0) for n in names.values()},
        "unknown_class_ids": dict(unknown_ids),
        "source_images": len(keys),
        "extra_copies_of_same_source": sum(c - 1 for c in keys.values() if c > 1),
        "_keys": set(keys),
        "_files": images,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dataset yaml, e.g. configs/dms.yaml")
    parser.add_argument("--json", type=Path, help="also write the report to this JSON file")
    parser.add_argument("--no-hash", action="store_true", help="skip the identical-file check (faster)")
    args = parser.parse_args(argv)

    dirs, names = resolve_split_dirs(args.data.resolve())
    missing = [f"{s}: {d}" for s, d in dirs.items() if not d.is_dir()]
    if missing:
        print("Split folders not found:\n  " + "\n  ".join(missing) + "\nSee data/README.md.", file=sys.stderr)
        return 1

    stats = {split: inspect_split(folder, names) for split, folder in dirs.items()}
    total = sum(s["images"] for s in stats.values())

    print(f"\nDataset: {args.data}  ({len(names)} classes, {total} images)\n")
    print(f"{'split':<6}{'images':>8}{'share':>8}{'backgr.':>9}{'no label':>10}{'objects':>9}")
    for split, s in stats.items():
        share = 100 * s["images"] / total if total else 0
        print(
            f"{split:<6}{s['images']:>8}{share:>7.1f}%{s['background_images']:>9}"
            f"{s['missing_labels']:>10}{sum(s['objects_per_class'].values()):>9}"
        )

    print(f"\n{'class':<18}" + "".join(f"{s:>16}" for s in stats) + "   (objects / images)")
    for name in names.values():
        cells = "".join(f"{s['objects_per_class'][name]:>8} /{s['images_per_class'][name]:>6}" for s in stats.values())
        flag = (
            "   <- no objects in some split" if any(s["objects_per_class"][name] == 0 for s in stats.values()) else ""
        )
        print(f"{name:<18}{cells}{flag}")

    print("\nRoboflow augmentation check (several copies of one source image in a split):")
    for split, s in stats.items():
        print(f"  {split:<6} {s['source_images']} source images, {s['extra_copies_of_same_source']} extra copies")

    report: dict = {"dataset": str(args.data), "classes": names, "splits": {}, "leakage": {}}
    print("\nLeakage check (same source image name in two splits):")
    for a, b in combinations(stats, 2):
        shared = stats[a]["_keys"] & stats[b]["_keys"]
        report["leakage"][f"{a}&{b}_same_source_name"] = len(shared)
        example = f"  e.g. {sorted(shared)[0]}" if shared else ""
        print(f"  {a} & {b}: {len(shared)}{example}")

    tracks = {
        split: {m.group(1) + "_" + m.group(2) for k in s["_keys"] if (m := GTSRB_PATTERN.match(k))}
        for split, s in stats.items()
    }
    if any(tracks.values()):
        print("\nGTSRB track overlap (frames of the same physical sign in two splits):")
        for a, b in combinations(tracks, 2):
            shared = tracks[a] & tracks[b]
            report["leakage"][f"{a}&{b}_gtsrb_tracks"] = len(shared)
            print(f"  {a} & {b}: {len(shared)} shared tracks (of {len(tracks[a])} / {len(tracks[b])})")

    if not args.no_hash:
        print("\nIdentical-file check (byte-for-byte duplicates across splits):")
        hashes = {split: defaultdict(list) for split in stats}
        for split, s in stats.items():
            for path in s["_files"]:
                hashes[split][file_hash(path)].append(path.name)
        for a, b in combinations(hashes, 2):
            shared = set(hashes[a]) & set(hashes[b])
            report["leakage"][f"{a}&{b}_identical_files"] = len(shared)
            print(f"  {a} & {b}: {len(shared)}")

    for split, s in stats.items():
        report["splits"][split] = {k: v for k, v in s.items() if not k.startswith("_")}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
