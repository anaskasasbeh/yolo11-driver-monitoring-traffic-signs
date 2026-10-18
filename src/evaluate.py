"""Measure a trained model on one dataset split (default: the held-out test split).

The test split is never used during training or checkpoint selection, so its
scores show how the model performs on unseen images.

Outputs:
    results/<task>_<split>_metrics.json   overall scores, speed, environment
    results/<task>_<split>_per_class.csv  precision, recall, mAP per class
    runs/eval/<task>_<split>/             plots (PR curve, confusion matrix)

Examples:
    python src/evaluate.py --task dms
    python src/evaluate.py --task tsd --split val --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.utils.torch_utils import get_cpu_info

from config import RESULTS_DIR, RUNS_DIR, get_task

logger = logging.getLogger("evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["dms", "tsd"], required=True)
    parser.add_argument("--model", type=Path, help="weights file. Default: models/<task>_*.pt from config")
    parser.add_argument("--data", type=Path, help="dataset yaml. Default: configs/<task>.yaml")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001, help="low threshold is standard for mAP")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (Ultralytics default)")
    parser.add_argument("--device", default=None, help="'cpu', '0' for the first GPU. Default: auto")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="where to write JSON/CSV results")
    return parser.parse_args(argv)


def count_images(folder: str | Path) -> int:
    """Count image files in a split folder."""
    return sum(1 for p in Path(folder).rglob("*") if p.suffix.lower().lstrip(".") in IMG_FORMATS)


def hardware_name(device: str | None) -> str:
    """Describe the hardware the speed numbers were measured on, e.g. "CPU: Intel Core i5-10210U"."""
    if device != "cpu" and torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    try:
        return f"CPU: {get_cpu_info()}"
    except Exception:  # noqa: BLE001 - fall back to the generic platform string
        return f"CPU: {platform.processor() or platform.machine()}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    task = get_task(args.task)
    weights = args.model or task.weights
    data = (args.data or task.data_yaml).resolve()

    if not weights.is_file():
        logger.error("Model file not found: %s (see models/README.md)", weights)
        return 1
    try:
        dataset = check_det_dataset(str(data))
    except FileNotFoundError as exc:
        logger.error("%s\nSee data/README.md for the expected dataset layout.", exc)
        return 1
    if not dataset.get(args.split):
        logger.error("Dataset config %s has no '%s' split.", data, args.split)
        return 1

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        project=str(RUNS_DIR / "eval"),
        name=f"{task.name}_{args.split}",
        exist_ok=True,
        verbose=True,
    )
    box = metrics.box

    per_class = []
    for i, class_id in enumerate(box.ap_class_index):
        precision, recall, map50, map50_95 = box.class_result(i)
        per_class.append(
            {
                "class": metrics.names[int(class_id)],
                "images": int(metrics.nt_per_image[class_id]),
                "instances": int(metrics.nt_per_class[class_id]),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "mAP50": round(float(map50), 4),
                "mAP50-95": round(float(map50_95), 4),
            }
        )
    missing = [n for i, n in metrics.names.items() if metrics.nt_per_class[i] == 0]

    summary = {
        "task": task.name,
        "split": args.split,
        "model_file": weights.name,
        "dataset_config": data.name,
        "images": count_images(dataset[args.split]),
        "instances": int(metrics.nt_per_class.sum()),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "classes_without_instances": missing,
        "speed_ms_per_image": {k: round(v, 2) for k, v in metrics.speed.items()},
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou},
        "hardware": hardware_name(args.device),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{task.name}_{args.split}_metrics.json"
    csv_path = args.out / f"{task.name}_{args.split}_per_class.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0]) if per_class else ["class"])
        writer.writeheader()
        writer.writerows(per_class)

    logger.info(
        "%s %s split: P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f (%d images)",
        task.name.upper(),
        args.split,
        summary["precision"],
        summary["recall"],
        summary["mAP50"],
        summary["mAP50-95"],
        summary["images"],
    )
    if missing:
        logger.warning("No instances in this split for: %s", ", ".join(missing))
    logger.info("Wrote %s and %s", json_path, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
