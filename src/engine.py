"""
封装 Faster R-CNN 的训练 / 验证循环，便于复用与扩展。
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor


def _sanitize_targets(targets):
    """
    确保所有 bbox 满足 x2 > x1, y2 > y1，必要时修正或丢弃非法框，
    以避免 torchvision 的 AssertionError。
    """
    for t in targets:
        boxes = t.get("boxes")
        if boxes is None or boxes.numel() == 0:
            continue
        x1 = torch.min(boxes[:, 0], boxes[:, 2])
        y1 = torch.min(boxes[:, 1], boxes[:, 3])
        x2 = torch.max(boxes[:, 0], boxes[:, 2])
        y2 = torch.max(boxes[:, 1], boxes[:, 3])
        boxes_fixed = torch.stack([x1, y1, x2, y2], dim=1)
        valid = (x2 > x1) & (y2 > y1)
        t["boxes"] = boxes_fixed[valid]
        if "labels" in t:
            t["labels"] = t["labels"][valid]


def reduce_dict(loss_dict: Dict[str, Tensor]) -> Dict[str, float]:
    """
    将每个 loss Tensor 转为 float，方便日志记录；若未来扩展到分布式可在此聚合。
    """

    return {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}


def _save_avg_loss_to_json(avg_loss: float, iteration: int, epoch: int, output_dir: Path | str | None = None, json_file: str = "avg_iter_loss.json") -> None:
    """
    将avg_loss和iteration追加到JSON文件中。
    文件格式为JSON数组，每个元素包含epoch、iteration和avg_loss。
    
    Args:
        avg_loss: 平均损失值
        iteration: 迭代次数
        epoch: epoch编号
        output_dir: 输出目录，如果为None则保存到当前目录
        json_file: JSON文件名
    """
    # 确定文件路径
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / json_file
    else:
        json_path = Path(json_file)
    
    # 读取现有数据
    data = []
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []
    
    # 添加新数据
    data.append({
        "epoch": epoch,
        "iteration": iteration,
        "avg_loss": avg_loss
    })
    
    # 写回文件
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    print_freq: int = 25,
    output_dir: Path | str | None = None,
) -> Tuple[Dict[str, float], List[float]]:
    model.train()
    metric_logger = defaultdict[Any, list](list)
    iter_losses: List[float] = []
    # 记录一个开始时间用于打印吞吐信息。
    start = time.time()

    for iteration, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        _sanitize_targets(targets)

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
        iter_losses.append(float(losses.detach().cpu().item()))
        for key, value in loss_dict_reduced.items():
            metric_logger[key].append(value)

        if iteration % print_freq == 0:
            current = time.time()
            avg_loss = sum(metric_logger["loss_classifier"]) / len(metric_logger["loss_classifier"])
            print(f"[Epoch {epoch} | Iter {iteration}] loss={avg_loss:.4f} ({current - start:.1f}s)")
            _save_avg_loss_to_json(avg_loss, iteration, epoch, output_dir=output_dir)

    epoch_metrics = {k: sum(v) / len(v) for k, v in metric_logger.items() if v}
    return epoch_metrics, iter_losses


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
        _sanitize_targets(targets)

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)

        loss_dict_reduced = reduce_dict(loss_dict)
        for key, value in loss_dict_reduced.items():
            metric_logger[key].append(value)

    return {k: sum(v) / len(v) for k, v in metric_logger.items() if v}

