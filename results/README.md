# Results

All numbers in this folder come from real training and evaluation runs.

## Test split (held-out images)

Produced by `python src/evaluate.py --task dms` and `--task tsd` on a laptop CPU (Intel Core i5-10210U).

| File | Content |
|---|---|
| `dms_test_metrics.json`, `tsd_test_metrics.json` | Overall precision, recall, mAP50, mAP50-95, inference speed, hardware and library versions |
| `dms_test_per_class.csv`, `tsd_test_per_class.csv` | Per-class precision, recall, mAP50 and mAP50-95 |

Confusion matrices for these runs are in `assets/dms_confusion_matrix.png` and `assets/tsd_confusion_matrix.png`.

## Training (validation split)

Measured on the validation split at the end of training, using the final `best.pt` weights.

| File | Content |
|---|---|
| `training_summary.json` | Overall scores for both models, and inference speed on a Tesla T4 GPU |
| `dms_val_per_class.csv`, `tsd_val_per_class.csv` | Per-class scores |
| `dms_training_history.csv` | Losses and validation scores per epoch (epochs 61-100) |
| `tsd_training_history.csv` | Losses and validation scores per epoch (epochs 50-100) |

## Dataset

| File | Content |
|---|---|
| `dataset_class_counts.csv` | Images and objects per class in each split, for both datasets |
