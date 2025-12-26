# Version 1
This is the first version implementing Faster R-CNN on RSNA Pneumonia dataset, with all default hyper parameters and pre-trained parameters. 
  - No fine-tuning is done.
  - Visualations are done by (1) adding ground truth frames to training data, (2) adding ground truth v.s. predicted frames to training data
  - done testing and submission to leader board


# Faster R-CNN Training on RSNA Pneumonia Detection

This project trains a Faster R-CNN detector on the RSNA Pneumonia Detection Challenge dataset (`stage_2_train_images` and `stage_2_train_labels.csv`). It uses PyTorch and torchvision's `fasterrcnn_resnet50_fpn` backbone.

## Features

- **Faster R-CNN with ResNet50-FPN**: Pre-trained model fine-tuned for pneumonia detection
- **mAP Evaluation**: Mean Average Precision calculation at multiple IoU thresholds (0.4-0.75)
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support for faster training
- **Experiment Tracking**: Timestamped experiment directories with comprehensive metrics logging
- **Visualization Tools**: Plot training/validation losses and mAP curves
- **Checkpoint Management**: Save best models based on validation loss and mAP
- **Stratified Data Splitting**: Patient-level stratified train/validation split

## Project Structure

```
pna_project/
├── src/
│   ├── train.py              # Main training script
│   ├── test.py               # Inference on test set
│   ├── val.py                # Validation with mAP calculation
│   ├── dataset.py            # RSNA dataset wrapper for DICOM files
│   ├── transforms.py         # Data augmentation utilities
│   ├── engine.py             # Training and validation loops
│   ├── utils.py              # Helper utilities (seeding, splits, config)
│   ├── map_metric.py         # mAP calculation implementation
│   ├── plot_metrics.py       # Visualization tools for training metrics
│   ├── submit.py             # Submission file generation
│   ├── convert_format.py     # Format conversion utilities
│   ├── add_frame.py          # Image annotation utilities
│   └── add_frame_GT_vs_Pred.py  # Ground truth vs prediction comparison
├── data/
│   ├── stage_2_train_images/     # Training DICOM images
│   ├── stage_2_test_images/      # Test DICOM images
│   ├── stage_2_train_labels.csv  # Training annotations
│   └── stage_2_sample_submission.csv  # Submission template
├── outputs/                  # Training outputs (timestamped experiment folders)
├── train_fasterrcnn.sh      # Training script
├── run_test.sh              # Test inference script
├── run_val.sh               # Validation script
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment configuration
└── README.md
```

## Setup

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate pna_env
```

### Option 2: Using pip

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Data Setup

Ensure the RSNA data is laid out as follows:

```
data/
├── stage_2_train_images/*.dcm
├── stage_2_test_images/*.dcm
├── stage_2_train_labels.csv
└── stage_2_sample_submission.csv
```

## Training

### Basic Training

Run the training script from the repository root:

```bash
python -m src.train \
  --images-dir data/stage_2_train_images \
  --labels-csv data/stage_2_train_labels.csv \
  --output-dir outputs/faster_rcnn \
  --epochs 10 \
  --batch-size 2 \
  --num-workers 0 \
  --use-amp
```

### Training with mAP Evaluation

Enable mAP evaluation during training:

```bash
python -m src.train \
  --images-dir data/stage_2_train_images \
  --labels-csv data/stage_2_train_labels.csv \
  --output-dir outputs/faster_rcnn \
  --epochs 10 \
  --batch-size 2 \
  --eval-map \
  --score-threshold 0.05 \
  --eval-map-every 5000 \
  --use-amp
```

### Using Training Script

```bash
bash train_fasterrcnn.sh
```

### Key Training Parameters

- `--images-dir` – Directory with DICOM files (default: `data/stage_2_train_images`)
- `--labels-csv` – CSV file with bounding box annotations (default: `data/stage_2_train_labels.csv`)
- `--output-dir` – Output directory for checkpoints and logs (default: `outputs/faster_rcnn`)
- `--epochs` – Number of training epochs (default: 10)
- `--batch-size` – Batch size (default: 2)
- `--val-split` – Fraction of patients used for validation (default: 0.1)
- `--lr` – Learning rate (default: 0.005)
- `--weight-decay` – Weight decay (default: 1e-4)
- `--lr-step-size` – Learning rate scheduler step size (default: 3)
- `--lr-gamma` – Learning rate decay factor (default: 0.1)
- `--num-workers` – DataLoader workers (default: 0)
- `--use-amp` – Enable Automatic Mixed Precision training
- `--cache-images` – Cache decoded DICOM tensors in memory (may cause OOM)
- `--limit-train` / `--limit-val` – Restrict number of patients for quick testing
- `--resume` – Path to checkpoint to resume training from
- `--device` – Torch device (e.g., `cuda:0`, defaults to auto-detect)
- `--eval-map` – Enable mAP evaluation during validation
- `--score-threshold` – Score threshold for mAP evaluation (default: 0.05)
- `--eval-map-every` – Evaluate mAP every N iterations (default: 5000)
- `--seed` – Random seed for reproducibility (default: 42)

### Training Outputs

Each training run creates a timestamped directory (format: `YYYYMMDDHHMM`) containing:

- `best_model.pth` – Best model based on validation loss
- `best_model_map.pth` – Best model based on mAP (if `--eval-map` is enabled)
- `last_checkpoint.pth` – Latest checkpoint with optimizer/scheduler state
- `metrics.jsonl` – Epoch-level metrics (losses, mAP)
- `metrics_iters.jsonl` – Per-iteration training loss
- `metrics_map_iters.jsonl` – Per-iteration mAP (if `--eval-map` is enabled)
- `avg_iter_loss.json` – Average loss per iteration
- `run_config.json` – Training configuration

## Validation

Run validation on the training set with mAP calculation:

```bash
python src/val.py \
  --checkpoint outputs/faster_rcnn/YYYYMMDDHHMM/best_model_map.pth \
  --images-dir data/stage_2_train_images \
  --labels-csv data/stage_2_train_labels.csv \
  --output-csv outputs/faster_rcnn/YYYYMMDDHHMM/val_labels.csv \
  --batch-size 2 \
  --score-threshold 0.5 \
  --val-images-limit 5000
```

Or use the validation script:

```bash
bash run_val.sh [gpu_id]
```

### Validation Parameters

- `--checkpoint` – Path to trained checkpoint
- `--images-dir` – Directory with DICOM images
- `--labels-csv` – CSV file with ground truth annotations
- `--output-csv` – Output CSV file path (default: `val_labels.csv`)
- `--batch-size` – Validation batch size (default: 2)
- `--score-threshold` – Confidence threshold for detections (default: 0.5)
- `--val-images-limit` – Limit number of validation images (optional)
- `--device` – Torch device (optional)

## Testing

Run inference on the test set and generate submission file:

```bash
python src/test.py \
  --checkpoint outputs/faster_rcnn/YYYYMMDDHHMM/best_model_map.pth \
  --images-dir data/stage_2_test_images \
  --sample-submission data/stage_2_sample_submission.csv \
  --output-csv outputs/faster_rcnn/YYYYMMDDHHMM/submission.csv \
  --batch-size 2 \
  --score-threshold 0.5
```

Or use the test script:

```bash
bash run_test.sh [gpu_id]
```

### Test Parameters

- `--checkpoint` – Path to trained checkpoint
- `--images-dir` – Directory with test DICOM images
- `--sample-submission` – Sample submission CSV template
- `--output-csv` – Output submission CSV file path
- `--batch-size` – Inference batch size (default: 2)
- `--score-threshold` – Score threshold for detections (default: 0.5)
- `--test-sample-limit` – Limit number of test samples (optional)
- `--device` – Torch device (optional)

## Visualization

Plot training metrics from a completed experiment:

```bash
python src/plot_metrics.py \
  --metrics-file outputs/faster_rcnn/YYYYMMDDHHMM/metrics.jsonl \
  --iter-metrics-file outputs/faster_rcnn/YYYYMMDDHHMM/metrics_iters.jsonl \
  --avg-iter-loss-file outputs/faster_rcnn/YYYYMMDDHHMM/avg_iter_loss.json \
  --out-dir outputs/faster_rcnn/YYYYMMDDHHMM
```

### Generated Plots

- `losses_train.png` – Training loss components per epoch
- `losses_val.png` – Validation loss components per epoch
- `val_loss.png` – Total validation loss per epoch
- `val_map.png` – mAP per epoch (if mAP evaluation was enabled)
- `loss_iter.png` – Training loss per iteration
- `avg_loss_vs_iteration.png` – Average loss vs iteration
- `map_iter.png` – mAP per iteration (if `--eval-map-every` was used)

See `PLOT_USAGE.md` for detailed plotting instructions.

## Model Architecture

- **Backbone**: ResNet50 with Feature Pyramid Network (FPN)
- **Detector**: Faster R-CNN
- **Classes**: 2 (background + pneumonia)
- **Input**: DICOM images normalized to [0, 1] using DICOM rescale slope/intercept
- **Output**: Bounding boxes in (x, y, width, height) format

## Notes

- Images are normalized to `[0, 1]` using DICOM rescale slope/intercept and expanded to 3 channels to match torchvision's expectations.
- Patient-level stratified splitting ensures no data leakage between train/validation sets.
- Mixed precision (`--use-amp`) is available when running on CUDA-capable hardware.
- mAP is calculated at IoU thresholds from 0.4 to 0.75 with step 0.05.
- Checkpoints include model weights, optimizer state, scheduler state, and optional AMP scaler for seamless resuming.

## Dependencies

See `requirements.txt` for Python package dependencies:
- torch >= 2.1.0
- torchvision >= 0.16.0
- pandas >= 2.1.0
- numpy >= 1.24.0
- pydicom >= 2.4.0
- scikit-learn >= 1.3.0
- tqdm >= 4.66.0
- matplotlib (for plotting)

See `environment.yml` for conda environment specification.
