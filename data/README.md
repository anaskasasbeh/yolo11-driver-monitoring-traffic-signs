# Data

The datasets are not stored in this repository. Place them here with this layout;
`configs/dms.yaml` and `configs/tsd.yaml` point to these paths.

```
data/
├── dms/                     # Driver Monitoring System (in-cabin camera)
│   ├── train/images/  train/labels/
│   ├── valid/images/  valid/labels/
│   └── test/images/   test/labels/
└── tsd/                     # Traffic Sign Detection (road camera)
    └── (same layout)
```

Each image has a `.txt` label file with the same name, in YOLO format: one line per object,
`class_id x_center y_center width height`, with coordinates scaled to 0-1.

## Datasets

| | DMS | TSD |
|---|---|---|
| Images | 9,916 | 4,917 |
| Train / validation / test | 5,979 / 2,399 / 1,538 | 3,479 / 801 / 637 |
| Classes | 5: Open Eye, Closed Eye, Cigarette, Phone, Seatbelt | 15: Green Light, Red Light, Stop, Speed Limit 10-120 |
| Sources | Public datasets, exported in YOLO format with Roboflow | Public datasets including GTSRB (German Traffic Sign Recognition Benchmark), exported with Roboflow |

Objects per class and split: `results/dataset_class_counts.csv`.

## Check a dataset

```bash
python scripts/inspect_dataset.py --data configs/dms.yaml
```

This prints images and objects per class and split, and checks for duplicate images across splits.
