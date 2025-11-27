"""
Detection-friendly transform utilities that operate on (image, target) tuples.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
from torch import Tensor


class Compose:
    """顺序执行一组变换，模仿 torchvision 的接口风格。"""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    """以概率 p 对图像与 bbox 进行水平翻转，实现最基础的数据增强。"""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])
            width = image.shape[2]
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                x_min = boxes[:, 0]
                x_max = boxes[:, 2]
                boxes[:, 0] = width - x_max
                boxes[:, 2] = width - x_min
                target["boxes"] = boxes
        return image, target


class ToFloatTensor:
    """确保输入张量是 float32，避免 dtype 不一致导致的运行错误。"""

    def __call__(self, image: Tensor, target: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        return image.to(torch.float32), target


def build_train_transforms() -> Compose:
    """训练阶段的增强策略，目前包含 float 转换 + 随机翻转。"""
    return Compose(
        [
            ToFloatTensor(),
            RandomHorizontalFlip(p=0.5),
        ]
    )


def build_eval_transforms() -> Compose:
    """验证 / 推理阶段只做类型转换，保持结果稳定。"""
    return Compose(
        [
            ToFloatTensor(),
        ]
    )

