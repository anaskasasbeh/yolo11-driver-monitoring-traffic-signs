"""Project settings: the two detection tasks, their classes, files and alert rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alerts import AlertRule

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"

# Online data augmentation used to train the published models, passed to model.train().
# Vertical flips, rotation and shear stay off: road signs and drivers are never upside down.
AUGMENTATION: dict[str, float] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
}


@dataclass(frozen=True)
class TaskConfig:
    """Everything that differs between the driver-monitoring and traffic-sign tasks.

    Attributes:
        name: Short id used on the command line ("dms" or "tsd").
        title: Human-readable name.
        data_yaml: Dataset config (paths + class names).
        weights: Default trained model file.
        base_model: Pretrained checkpoint used as the starting point for training.
        class_names: Classes in the order the model was trained with.
        alert_rules: Spoken alerts, keyed by class name or derived event.
        derive_events: Optional function that adds events computed from the
            detected classes (for example "No Seatbelt").
        patience: Early-stopping patience (epochs without improvement).
    """

    name: str
    title: str
    data_yaml: Path
    weights: Path
    base_model: str
    class_names: tuple[str, ...]
    alert_rules: dict[str, AlertRule]
    derive_events: Callable[[set[str]], set[str]] | None = None
    patience: int = 100


# --- Driver Monitoring System (in-cabin camera) ------------------------------

DMS_CLASSES = ("Open Eye", "Closed Eye", "Cigarette", "Phone", "Seatbelt")


def derive_dms_events(detected: set[str]) -> set[str]:
    """Add "No Seatbelt" when a driver's face is visible but no seatbelt is.

    The model can only detect a seatbelt that is there, not one that is missing,
    so a missing belt is inferred from its absence while the driver is in view.
    """
    driver_visible = bool(detected & {"Open Eye", "Closed Eye"})
    if driver_visible and "Seatbelt" not in detected:
        return detected | {"No Seatbelt"}
    return detected


DMS = TaskConfig(
    name="dms",
    title="Driver Monitoring System",
    data_yaml=CONFIGS_DIR / "dms.yaml",
    weights=MODELS_DIR / "dms_yolo11l.pt",
    base_model="yolo11l.pt",
    class_names=DMS_CLASSES,
    alert_rules={
        # Eyes closed for 1 s is far longer than a blink (~0.1-0.4 s).
        "Closed Eye": AlertRule("Wake up! Keep your eyes on the road.", min_duration=1.0, cooldown=5.0),
        "Phone": AlertRule("Put your phone down.", min_duration=1.0, cooldown=10.0),
        "Cigarette": AlertRule("Please put out the cigarette.", min_duration=1.0, cooldown=15.0),
        "No Seatbelt": AlertRule("Please fasten your seatbelt.", min_duration=5.0, cooldown=30.0),
    },
    derive_events=derive_dms_events,
    patience=100,
)

# --- Traffic Sign Detection (road-facing camera) -----------------------------

SPEED_LIMITS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)

# Same order as the training data.yaml (alphabetical, as exported by Roboflow).
TSD_CLASSES = (
    "Green Light",
    "Red Light",
    "Speed Limit 10",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
)

TSD = TaskConfig(
    name="tsd",
    title="Traffic Sign Detection",
    data_yaml=CONFIGS_DIR / "tsd.yaml",
    weights=MODELS_DIR / "tsd_yolo11m.pt",
    base_model="yolo11m.pt",
    class_names=TSD_CLASSES,
    alert_rules={
        "Stop": AlertRule("Stop sign ahead.", min_duration=0.3, cooldown=10.0),
        "Red Light": AlertRule("Red light ahead.", min_duration=0.3, cooldown=10.0),
        # Green lights need no warning, so they have no rule.
        **{
            f"Speed Limit {kmh}": AlertRule(f"Speed limit {kmh}.", min_duration=0.3, cooldown=20.0)
            for kmh in SPEED_LIMITS
        },
    },
    patience=15,
)

TASKS: dict[str, TaskConfig] = {DMS.name: DMS, TSD.name: TSD}


def get_task(name: str) -> TaskConfig:
    """Return the task config for "dms" or "tsd"."""
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"Unknown task '{name}'. Choose from: {', '.join(TASKS)}") from None
