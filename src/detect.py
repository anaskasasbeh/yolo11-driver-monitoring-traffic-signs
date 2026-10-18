"""Real-time detection with spoken safety alerts.

Runs the driver-monitoring model (in-cabin camera), the traffic-sign model
(road camera), or both side by side, on a webcam, a video file or an image.

Examples:
    python src/detect.py --task dms
    python src/detect.py --task tsd --source road.mp4 --save
    python src/detect.py --task both --dms-source 0 --tsd-source road.mp4
    python src/detect.py --task dms --source driver.jpg --no-show --save

Press "q" or Esc in the video window to quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from alerts import AlertManager, Speaker
from config import RUNS_DIR, TaskConfig, get_task

logger = logging.getLogger("detect")

WINDOW_NAME = "YOLO11 Driver Safety (press q to quit)"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BANNER_SECONDS = 3.0
MAX_CAMERA_FAILURES = 30  # consecutive failed reads before giving up on a webcam


class FrameSource:
    """Read frames from a webcam index, a video file or a single image."""

    def __init__(self, source: str, width: int = 640, height: int = 480) -> None:
        self.name = source
        self.is_camera = source.isdigit()
        self.is_image = Path(source).suffix.lower() in IMAGE_SUFFIXES
        self._image: np.ndarray | None = None
        self._capture: cv2.VideoCapture | None = None
        self._done = False

        if self.is_image:
            if not Path(source).is_file():
                raise FileNotFoundError(f"Image not found: {source}")
            self._image = cv2.imread(source)
            if self._image is None:
                raise ValueError(f"Could not read image: {source}")
            return

        if not self.is_camera and not Path(source).is_file():
            raise FileNotFoundError(f"Video not found: {source}")

        self._capture = cv2.VideoCapture(int(source) if self.is_camera else source)
        if not self._capture.isOpened():
            hint = " Try another index, e.g. --source 1." if self.is_camera else ""
            raise RuntimeError(f"Could not open source '{source}'.{hint}")
        if self.is_camera:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def fps(self) -> float:
        """Frame rate reported by the source (used when saving video)."""
        if self._capture is None:
            return 0.0
        return float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None when the source has ended."""
        if self._done:
            return None
        if self._image is not None:
            self._done = True
            return self._image.copy()

        assert self._capture is not None
        for _ in range(MAX_CAMERA_FAILURES if self.is_camera else 1):
            ok, frame = self._capture.read()
            if ok:
                return frame
        if self.is_camera:
            logger.error("Camera %s stopped sending frames.", self.name)
        self._done = True
        return None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()


class Detector:
    """One YOLO model plus the alert logic for its task."""

    def __init__(
        self,
        task: TaskConfig,
        weights: Path,
        conf: float,
        imgsz: int,
        device: str | None,
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(
                f"Model file not found: {weights}\n"
                f"Put the trained weights there (see models/README.md), or pass --model "
                f"(--{task.name}-model with --task both)."
            )
        self.task = task
        self.model = YOLO(str(weights))
        self.alerts = AlertManager(task.alert_rules)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.inference_ms: list[float] = []
        self._check_classes(weights)

    def _check_classes(self, weights: Path) -> None:
        """Warn if the model's classes are not the ones this task expects."""
        model_classes = tuple(self.model.names[i] for i in sorted(self.model.names))
        if model_classes != self.task.class_names:
            logger.warning(
                "%s has classes %s, but task '%s' expects %s. Alerts may not fire; "
                "check that this is the right weights file.",
                weights.name,
                list(model_classes)[:6],
                self.task.name,
                list(self.task.class_names)[:6],
            )

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Detect objects in one frame.

        Returns:
            The annotated frame and the alert messages to speak now.
        """
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        self.inference_ms.append(result.speed["inference"])

        detected = {result.names[int(c)] for c in result.boxes.cls.tolist()}
        events = self.task.derive_events(detected) if self.task.derive_events else detected
        messages = self.alerts.update(events)

        annotated = result.plot()
        _put_label(annotated, self.task.title, (10, 28))
        return annotated, messages


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7) -> None:
    """Draw white text on a dark box so it stays readable on any background."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    cv2.rectangle(image, (x - 4, y - h - 6), (x + w + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _side_by_side(panels: list[np.ndarray]) -> np.ndarray:
    """Resize panels to the same height and place them next to each other."""
    if len(panels) == 1:
        return panels[0]
    height = min(p.shape[0] for p in panels)
    resized = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in panels]
    return np.hstack(resized)


def _draw_banner(canvas: np.ndarray, text: str) -> None:
    """Show the latest alert as a red banner at the bottom of the frame."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 44), (w, h), (0, 0, 200), -1)
    cv2.putText(canvas, text, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def run(
    pairs: list[tuple[Detector, FrameSource]],
    speaker: Speaker,
    show: bool,
    save: bool,
    max_frames: int | None,
) -> None:
    """Main loop: read, detect, alert, draw, display and optionally save."""
    single_image = all(source.is_image for _, source in pairs)
    tag = "_".join(det.task.name for det, _ in pairs)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RUNS_DIR / "detect" / f"{tag}_{stamp}{'.jpg' if single_image else '.mp4'}"

    writer: cv2.VideoWriter | None = None
    banner, banner_until = "", 0.0
    frames, fps = 0, 0.0
    start = time.perf_counter()

    try:
        while max_frames is None or frames < max_frames:
            loop_start = time.perf_counter()
            panels = []
            for detector, source in pairs:
                frame = source.read()
                if frame is None:
                    break
                annotated, messages = detector.process(frame)
                for message in messages:
                    speaker.say(message)
                    banner, banner_until = message, time.monotonic() + BANNER_SECONDS
                panels.append(annotated)
            if len(panels) < len(pairs):
                break  # a source ended

            canvas = _side_by_side(panels)
            frames += 1
            loop_fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
            fps = loop_fps if frames == 1 else 0.9 * fps + 0.1 * loop_fps
            if not single_image:
                _put_label(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 120, 28))
            if banner and time.monotonic() < banner_until:
                _draw_banner(canvas, banner)

            if save:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if single_image:
                    cv2.imwrite(str(out_path), canvas)
                else:
                    if writer is None:
                        src_fps = pairs[0][1].fps
                        writer = cv2.VideoWriter(
                            str(out_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            src_fps if 1 <= src_fps <= 120 else 20.0,
                            (canvas.shape[1], canvas.shape[0]),
                        )
                    writer.write(canvas)

            if show:
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(0 if single_image else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if single_image:
                break
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        elapsed = time.perf_counter() - start
        for _, source in pairs:
            source.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        speaker.close()

    _log_summary(pairs, frames, elapsed)
    if save and frames:
        logger.info("Saved output to %s", out_path)


def _log_summary(pairs: list[tuple[Detector, FrameSource]], frames: int, elapsed: float) -> None:
    """Print run statistics measured during this run."""
    if frames == 0:
        logger.warning("No frames were processed.")
        return
    logger.info("Processed %d frame(s) in %.1f s (%.1f FPS end-to-end).", frames, elapsed, frames / elapsed)
    for detector, _ in pairs:
        times = detector.inference_ms[1:] or detector.inference_ms  # first call includes warm-up
        logger.info(
            "  %s model: mean inference %.1f ms/frame over %d frame(s)",
            detector.task.name.upper(),
            sum(times) / len(times),
            len(times),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["dms", "tsd", "both"], required=True, help="which model(s) to run")
    parser.add_argument("--source", default="0", help="webcam index, video or image (single task). Default: 0")
    parser.add_argument("--model", type=Path, help="weights file (single task). Default: from config")
    parser.add_argument("--dms-source", default="0", help="driver camera for --task both. Default: 0")
    parser.add_argument("--tsd-source", default="1", help="road camera or video for --task both. Default: 1")
    parser.add_argument("--dms-model", type=Path, help="DMS weights for --task both")
    parser.add_argument("--tsd-model", type=Path, help="TSD weights for --task both")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence. Default: 0.4")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size. Default: 640")
    parser.add_argument("--device", default=None, help="'cpu', '0' for the first GPU. Default: auto")
    parser.add_argument("--mute", action="store_true", help="log alerts without speaking them")
    parser.add_argument("--save", action="store_true", help="save the annotated output to runs/detect/")
    parser.add_argument("--no-show", action="store_true", help="do not open a window (headless)")
    parser.add_argument("--max-frames", type=int, help="stop after N frames")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    if args.task == "both":
        plan = [
            (get_task("dms"), args.dms_model, args.dms_source),
            (get_task("tsd"), args.tsd_model, args.tsd_source),
        ]
    else:
        plan = [(get_task(args.task), args.model, args.source)]

    pairs: list[tuple[Detector, FrameSource]] = []
    try:
        for task, weights, source in plan:
            detector = Detector(task, weights or task.weights, args.conf, args.imgsz, args.device)
            pairs.append((detector, FrameSource(source)))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        for _, source in pairs:
            source.release()
        return 1

    speaker = Speaker(enabled=not args.mute)
    run(pairs, speaker, show=not args.no_show, save=args.save, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
