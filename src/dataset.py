"""
面向 RSNA Pneumonia 数据集的 Dataset，读取 CSV 标注与 DICOM 图像，
为 Faster R-CNN 提供标准化后的图像与目标框。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    """存储单个样本的元数据，便于 __getitem__ 快速访问。"""

    patient_id: str
    dicom_path: Path
    has_target: bool
    boxes: np.ndarray  # shape: (N, 4) formatted as (x, y, width, height)


def _apply_windowing(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """根据 DICOM 的 slope/intercept 做线性变换并归一化到 [0, 1]。"""

    scaled = pixel_array.astype(np.float32) * slope + intercept
    scaled -= scaled.min()
    max_value = scaled.max()
    if max_value > 0:
        scaled /= max_value
    return scaled


def _load_dicom_image(path: Path) -> Tensor:
    """读取 DICOM 并返回 float32 张量，随后复制到 3 通道以匹配 torchvision 的输入。"""

    dataset = pydicom.dcmread(path)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixel_array = dataset.pixel_array
    normalized = _apply_windowing(pixel_array, slope, intercept)
    # Convert to 3-channel tensor because torchvision detection models expect that.
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1)
    return tensor


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """将 CSV 中的 [x, y, w, h] 转换为检测模型需要的 [x1, y1, x2, y2]。"""

    xyxy = xywh.copy()
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]
    return xyxy


class RSNADataset(Dataset):
    """
    Custom Dataset for the RSNA Pneumonia challenge.
    将相同 patientId 的行聚合为一个样本，返回 Faster R-CNN 期望的
    (image, target) 结构：
      * image: Float tensor, shape [3, H, W], normalized to [0, 1]
      * target: dict with keys boxes / labels / image_id / area / iscrowd
    """

    def __init__(
        self,
        labels_csv: str | Path,
        images_dir: str | Path,
        patient_ids: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Tensor, Dict[str, Tensor]], Tuple[Tensor, Dict[str, Tensor]]]] = None,
        cache_images: bool = False,
    ) -> None:
        """
        Args / 参数:
            labels_csv: Path to `stage_2_train_labels.csv`; 包含 patientId、box 坐标、Target。
            images_dir: Directory with DICOM files; 图像目录。
            patient_ids: Optional subset of patient IDs; 仅加载指定病人。
            transforms: Optional callable that augments `(image, target)`; 额外的数据增强。
            cache_images: If True, keep decoded tensors in memory to reduce I/O; 是否缓存图像。
        """
        self.images_dir = Path(images_dir)
        self.labels_csv = Path(labels_csv)
        self.transforms = transforms
        self.cache_images = cache_images

        df = pd.read_csv(self.labels_csv)
        # 如用户仅想使用部分病人，可通过 patient_ids 过滤。
        if patient_ids is not None:
            df = df[df["patientId"].isin(patient_ids)]

        grouped = df.groupby("patientId")
        self.records: List[SampleRecord] = []
        for patient_id, group in grouped:
            target_rows = group[group["Target"] == 1]
            boxes = target_rows[["x", "y", "width", "height"]].dropna().values
            record = SampleRecord(
                patient_id=patient_id,
                dicom_path=self.images_dir / f"{patient_id}.dcm",
                has_target=len(target_rows) > 0,
                boxes=boxes,
            )
            self.records.append(record)

        if not self.records:
            raise ValueError("No samples found. Check your patient IDs or CSV path.")

        # Simple in-memory cache so频繁访问同一病人时避免重复读取 DICOM。
        self._image_cache: Dict[str, Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _get_image(self, record: SampleRecord) -> Tensor:
        if self.cache_images and record.patient_id in self._image_cache:
            return self._image_cache[record.patient_id]

        image = _load_dicom_image(record.dicom_path)
        if self.cache_images:
            self._image_cache[record.patient_id] = image
        return image

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        record = self.records[index]
        image = self._get_image(record)

        # 默认构造空 box / label，兼容无病灶的阴性样本。
        boxes = torch.as_tensor([], dtype=torch.float32).reshape(0, 4)
        labels = torch.as_tensor([], dtype=torch.int64)

        if record.has_target and len(record.boxes) > 0:
            xyxy = _xywh_to_xyxy(record.boxes.astype(np.float32))
            boxes = torch.from_numpy(xyxy).to(dtype=torch.float32)
            # 纠正可能存在的 x1/x2、y1/y2 颠倒，并确保宽高为正。
            x1 = torch.min(boxes[:, 0], boxes[:, 2])
            y1 = torch.min(boxes[:, 1], boxes[:, 3])
            x2 = torch.max(boxes[:, 0], boxes[:, 2])
            y2 = torch.max(boxes[:, 1], boxes[:, 3])
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            valid = (x2 > x1) & (y2 > y1)
            boxes = boxes[valid]
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        # torchvision detection 期望的 target 键。
        target: Dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if boxes.numel()
            else torch.tensor([], dtype=torch.float32),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch: List[Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
    """检测模型需要的自定义 collate：保持 list 结构，不做堆叠。"""

    images, targets = zip(*batch)
    return list(images), list(targets)

