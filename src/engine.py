"""
封装 Faster R-CNN 的训练 / 验证循环，便于复用与扩展。
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List

import torch
from torch import Tensor


def reduce_dict(loss_dict: Dict[str, Tensor]) -> Dict[str, float]:
    """
    将每个 loss Tensor 转为 float，方便日志记录；若未来扩展到分布式可在此聚合。
    """

    return {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    print_freq: int = 25,
) -> Dict[str, float]:
    model.train()
    metric_logger = defaultdict[Any, list](list)
    # 记录一个开始时间用于打印吞吐信息。
    start = time.time()

    for iteration, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        loss_dict_reduced = reduce_dict(loss_dict)
        for key, value in loss_dict_reduced.items():
            metric_logger[key].append(value)

        if iteration % print_freq == 0:
            current = time.time()
            avg_loss = sum(metric_logger["loss_classifier"]) / len(metric_logger["loss_classifier"])
            print(f"[Epoch {epoch} | Iter {iteration}] loss={avg_loss:.4f} ({current - start:.1f}s)")

    return {k: sum(v) / len(v) for k, v in metric_logger.items() if v}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    model.train()  # detection 模型在 eval 模式下不会返回 loss，因此保持 train()
    metric_logger = defaultdict(list)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)

        loss_dict_reduced = reduce_dict(loss_dict)
        for key, value in loss_dict_reduced.items():
            metric_logger[key].append(value)

    return {k: sum(v) / len(v) for k, v in metric_logger.items() if v}

