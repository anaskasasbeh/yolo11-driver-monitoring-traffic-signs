"""Fine-tune a pretrained YOLO11 model on the DMS or TSD dataset.

Defaults are the settings used for the published models, including the data
augmentation (see AUGMENTATION in config.py).

Training needs a GPU to finish in reasonable time. On a Tesla T4, one epoch
takes about 5.2 min (DMS, YOLO11l) and 2.3 min (TSD, YOLO11m).

Examples:
    python src/train.py --task dms
    python src/train.py --task tsd --epochs 50 --batch 8
    python src/train.py --task dms --resume runs/dms/train/weights/last.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

from config import AUGMENTATION, RUNS_DIR, get_task

logger = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["dms", "tsd"], required=True)
    parser.add_argument("--data", type=Path, help="dataset yaml. Default: configs/<task>.yaml")
    parser.add_argument("--model", help="starting checkpoint. Default: yolo11l.pt (dms) / yolo11m.pt (tsd)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="images per batch; -1 = auto-fit GPU memory")
    parser.add_argument("--patience", type=int, help="early-stopping patience. Default: per task (config.py)")
    parser.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu'. Default: auto")
    parser.add_argument("--workers", type=int, default=8, help="data-loading worker processes")
    parser.add_argument("--name", default="train", help="run folder name under runs/<task>/")
    parser.add_argument("--resume", type=Path, help="continue an interrupted run from its last.pt")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    task = get_task(args.task)

    if args.resume:
        if not args.resume.is_file():
            logger.error("Checkpoint not found: %s", args.resume)
            return 1
        logger.info("Resuming %s training from %s", task.title, args.resume)
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        data = (args.data or task.data_yaml).resolve()
        if not data.is_file():
            logger.error("Dataset config not found: %s", data)
            return 1
        model = YOLO(args.model or task.base_model)
        logger.info("Training %s: %s on %s", task.title, args.model or task.base_model, data)
        try:
            model.train(
                data=str(data),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                patience=args.patience if args.patience is not None else task.patience,
                device=args.device,
                workers=args.workers,
                optimizer="auto",
                seed=0,
                deterministic=True,
                plots=True,
                project=str(RUNS_DIR / task.name),
                name=args.name,
                **AUGMENTATION,
            )
        except FileNotFoundError as exc:
            logger.error("%s\nSee data/README.md for the expected dataset layout.", exc)
            return 1

    best = Path(model.trainer.best)
    logger.info("Best weights: %s", best)
    logger.info("To use them for detection, copy that file to %s", task.weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
