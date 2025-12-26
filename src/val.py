"""
验证脚本：使用训练好的 Faster R-CNN checkpoint 对训练集图像进行验证，计算 mAP。

使用示例（在项目根目录）：
    python src/val.py \
        --checkpoint outputs/faster_rcnn/202512151433/best_model_map.pth \
        --images-dir data/stage_2_train_images \
        --labels-csv data/stage_2_train_labels.csv \
        --val-images-limit 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import RSNADataset, collate_fn
from engine import evaluate_map
from train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Faster R-CNN model on RSNA training set.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/faster_rcnn/202512151433/best_model_map.pth"),
        help="Path to the trained checkpoint (.pth) containing model weights.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/stage_2_train_images"),
        help="Directory with stage_2_train_images DICOM files.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/stage_2_train_labels.csv"),
        help="Path to stage_2_train_labels.csv containing ground truth annotations.",
    )
    parser.add_argument(
        "--val-images-limit",
        type=int,
        default=None,
        help="Limit the number of validation images. If set, only the first N images will be used.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for reading DICOM files.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections. Only boxes with score >= threshold are considered.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g., cuda:0). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("val_labels.csv"),
        help="Output CSV file path in stage_2_train_labels.csv format.",
    )
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load model weights from checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # weights_only=False is required because checkpoints contain optimizer/scheduler state dicts,
    # not just model weights. Since we control the checkpoint files, this is safe.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {checkpoint_path}")


def get_patient_ids_from_csv(labels_csv: Path, limit: int | None = None) -> list[str]:
    """Extract unique patient IDs from labels CSV file."""
    df = pd.read_csv(labels_csv)
    patient_ids = df["patientId"].unique().tolist()

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"val-images-limit must be positive, got {limit}")
        patient_ids = patient_ids[:limit]
        print(f"Limited to {len(patient_ids)} patient IDs (val-images-limit={limit})")

    return patient_ids


def boxes_to_labels_format(boxes: Tensor, scores: Tensor, score_threshold: float) -> List[Dict[str, float]]:
    """
    Convert predicted boxes (xyxy format) to stage_2_train_labels.csv format.
    Returns a list of dictionaries with keys: x, y, width, height
    """
    if boxes.numel() == 0 or scores.numel() == 0:
        return []
    
    # Filter by score threshold
    keep = scores >= score_threshold
    if keep.sum() == 0:
        return []
    
    boxes = boxes[keep]
    
    labels = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        x = float(x1)
        y = float(y1)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        labels.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        })
    
    return labels


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Get patient IDs (with limit if specified)
    patient_ids = get_patient_ids_from_csv(args.labels_csv, args.val_images_limit)

    # Create dataset
    print(f"Loading dataset from {args.images_dir} with {len(patient_ids)} patient IDs...")
    dataset = RSNADataset(
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        patient_ids=patient_ids,
        transforms=None,  # No augmentation for validation
        cache_images=False,
    )
    print(f"Dataset created with {len(dataset)} samples")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Build and load model
    print("Building model...")
    model = build_model(num_classes=2, pretrained=False)
    model.to(device)
    model.eval()

    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.checkpoint, device)

    # Evaluate and collect predictions
    print(f"\nStarting validation with score_threshold={args.score_threshold}...")
    print("This may take a while...")
    
    # Collect predictions for CSV generation
    predictions_rows: List[Dict[str, str | float | int]] = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx} batches...")
            
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            
            # Process each sample in the batch
            for idx, (output, target) in enumerate(zip(outputs, targets)):
                # Get patient ID from the dataset
                dataset_idx = batch_idx * args.batch_size + idx
                record = dataset.records[dataset_idx]
                patient_id = record.patient_id
                
                # Get predicted boxes and scores
                pred_boxes = output["boxes"].cpu()
                pred_scores = output["scores"].cpu()
                
                # Convert to labels format
                boxes_labels = boxes_to_labels_format(pred_boxes, pred_scores, args.score_threshold)
                
                if len(boxes_labels) == 0:
                    # No detections, create one row with Target=0
                    predictions_rows.append({
                        "patientId": patient_id,
                        "x": "",
                        "y": "",
                        "width": "",
                        "height": "",
                        "Target": 0,
                    })
                else:
                    # Create one row per detection with Target=1
                    for box_label in boxes_labels:
                        predictions_rows.append({
                            "patientId": patient_id,
                            "x": box_label["x"],
                            "y": box_label["y"],
                            "width": box_label["width"],
                            "height": box_label["height"],
                            "Target": 1,
                        })
    
    # Calculate mAP using evaluate_map function
    val_map = evaluate_map(model, data_loader, device, score_threshold=args.score_threshold)
    
    # Save predictions to CSV
    print(f"\nSaving predictions to {args.output_csv}...")
    predictions_df = pd.DataFrame(predictions_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(predictions_rows)} rows to {args.output_csv}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"{'='*60}")
    print(f"Number of images evaluated: {len(dataset)}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"mAP: {val_map:.4f}")
    print(f"Predictions saved to: {args.output_csv}")
    print(f"Total rows in output CSV: {len(predictions_rows)}")
    print(f"Rows with detections (Target=1): {sum(1 for r in predictions_rows if r['Target'] == 1)}")
    print(f"Rows without detections (Target=0): {sum(1 for r in predictions_rows if r['Target'] == 0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

