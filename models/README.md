# Models

The trained weights are not stored in git. Download them from the repository's
[Releases page](https://github.com/anaskasasbeh28/yolo11-driver-monitoring-traffic-signs/releases)
and save them here with these names:

| File | Task | Architecture | Size |
|---|---|---|---|
| `dms_yolo11l.pt` | Driver Monitoring System (5 classes) | YOLO11l, 25.3M parameters | ~49 MB |
| `tsd_yolo11m.pt` | Traffic Sign Detection (15 classes) | YOLO11m, 20.0M parameters | ~39 MB |

`src/detect.py` and `src/evaluate.py` load these names by default. Use `--model path/to/file.pt`
to load a file from somewhere else.

To check which model a file contains, print its class names:

```bash
python -c "from ultralytics import YOLO; print(YOLO('models/dms_yolo11l.pt').names)"
```
