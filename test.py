"""
运行阶段二测试集的推理脚本：加载训练好的 Faster R-CNN checkpoint，
对 `stage_2_test_images/` 中的所有 DICOM 图像推理，并按照
`stage_2_sample_submission.csv` 的格式生成提交文件。

使用示例（在项目根目录）：

    python test.py \
        --checkpoint outputs/faster_rcnn/best_model.pth \
        --images-dir data/stage_2_test_images \
        --sample-submission data/stage_2_sample_submission.csv \
        --output-csv outputs/faster_rcnn/submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import _load_dicom_image
from src.train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on RSNA test set and generate submission CSV.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/faster_rcnn/best_model.pth"),
        help="Path to the trained checkpoint (.pth) containing model weights.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/stage_2_test_images"),
        help="Directory with stage_2_test_images DICOM files.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/stage_2_sample_submission.csv"),
        help="CSV that lists patientId and acts as template for output ordering.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/faster_rcnn/submission.csv"),
        help="Where to save the generated submission CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for reading DICOM files.")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Only keep detections with score >= threshold.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string (e.g., cuda:0). Defaults to CUDA if available.")
    return parser.parse_args()


class RSNATestDataset(Dataset):
    """Dataset wrapper that only needs patient IDs and image directory."""

    def __init__(self, images_dir: Path, patient_ids: Sequence[str]) -> None:
        self.images_dir = Path(images_dir)
        self.patient_ids = list(patient_ids)
        if not self.images_dir.is_dir():
            raise NotADirectoryError(f"images dir not found: {self.images_dir}")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        patient_id = self.patient_ids[index]
        dicom_path = self.images_dir / f"{patient_id}.dcm"
        if not dicom_path.is_file():
            raise FileNotFoundError(f"missing DICOM: {dicom_path}")

        image = _load_dicom_image(dicom_path).to(torch.float32)
        return image, patient_id


def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[List[torch.Tensor], List[str]]:
    images, patient_ids = zip(*batch)
    return list(images), list(patient_ids)


def detections_to_prediction_string(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float,
) -> str:
    """
    将 Faster R-CNN 输出的 boxes/scores 过滤并转换成 Kaggle 期望的 PredictionString：
    "score x y width height ..."
    """

    if boxes.numel() == 0 or scores.numel() == 0:
        return ""

    keep = scores >= score_threshold
    if keep.sum() == 0:
        return ""

    boxes = boxes[keep]
    scores = scores[keep]

    parts: List[str] = []
    for score, box in zip(scores, boxes):
        x1, y1, x2, y2 = box.tolist()
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        parts.append(f"{score:.4f} {x1:.1f} {y1:.1f} {width:.1f} {height:.1f}")

    return " ".join(parts)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    sample_df = pd.read_csv(args.sample_submission)
    if "patientId" not in sample_df.columns:
        raise ValueError(f"`patientId` column missing in {args.sample_submission}")

    patient_ids: List[str] = sample_df["patientId"].tolist()
    dataset = RSNATestDataset(args.images_dir, patient_ids)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=2, pretrained=False)
    model.to(device)
    model.eval()
    load_checkpoint(model, args.checkpoint, device)

    predictions: Dict[str, str] = {}
    with torch.no_grad():
        for images, batch_patient_ids in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for patient_id, output in zip(batch_patient_ids, outputs):
                boxes = output["boxes"].detach().cpu()
                scores = output["scores"].detach().cpu()
                predictions[patient_id] = detections_to_prediction_string(boxes, scores, args.score_threshold)

    sample_df["PredictionString"] = sample_df["patientId"].map(lambda pid: predictions.get(pid, ""))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.output_csv, index=False)
    print(f"Saved submission to {args.output_csv}")


if __name__ == "__main__":
    main()


