# Faster R-CNN Training on RSNA Pneumonia Detection

This project trains a Faster R-CNN detector on the RSNA Pneumonia Detection Challenge dataset (`stage_2_train_images` and `stage_2_train_labels.csv`). It uses PyTorch and torchvision's `fasterrcnn_resnet50_fpn` backbone.

## Project layout

- `src/dataset.py` – RSNA dataset wrapper that reads DICOM files and formats detection targets.
- `src/transforms.py` – Minimal data augmentation utilities for detection.
- `src/engine.py` – Training and validation loops.
- `src/utils.py` – Helper utilities (seeding, stratified splits, config logging).
- `src/train.py` – CLI entrypoint for model training.
- `requirements.txt` – Python dependencies.

## Setup

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the RSNA data is laid out as follows (already matches the provided repository):

```
data/
├── stage_2_train_images/*.dcm
└── stage_2_train_labels.csv/stage_2_train_labels.csv
```

## Training

Run the training script from the repository root:

```bash
python -m src.train \
  --images-dir data/stage_2_train_images \
  --labels-csv data/stage_2_train_labels.csv/stage_2_train_labels.csv \
  --output-dir outputs/faster_rcnn \
  --epochs 10 \
  --batch-size 4 \
  --num-workers 4 \
  --use-amp
```

Key flags:

- `--val-split` – Fraction of patients used for validation (default 0.1).
- `--cache-images` – Keep decoded tensors in memory to save disk reads.
- `--limit-train` / `--limit-val` – Restrict patients for quick smoke tests.
- `--resume` – Path to a `*.pth` checkpoint to resume from.

Checkpoints (`best_model.pth`, `last_checkpoint.pth`), JSONL metrics, and the run config are stored in `--output-dir`.

## Notes

- Images are normalized to `[0, 1]` using the DICOM rescale slope/intercept and expanded to 3 channels to match torchvision's expectations.
- Validation losses are logged per epoch; full COCO-style metrics are not included but can be added later if needed.
- Mixed precision (`--use-amp`) is available when running on CUDA-capable hardware.

