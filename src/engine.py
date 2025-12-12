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

from map_metric import calculate_mean_average_precision


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
    global_iteration: int = 0,
    eval_map_every: int | None = None,
    val_loader = None,
    score_threshold: float = 0.05,
) -> Tuple[Dict[str, float], List[float], List[Dict[str, float | int]]]:
    """
    训练一个epoch。
    
    Args:
        model: 模型
        optimizer: 优化器
        data_loader: 训练数据加载器
        device: 设备
        epoch: 当前epoch编号
        scaler: AMP scaler
        print_freq: 打印频率
        output_dir: 输出目录
        global_iteration: 全局iteration计数（跨epoch累计）
        eval_map_every: 每N个iteration计算一次mAP，None表示不计算
        val_loader: 验证数据加载器，用于计算mAP
        score_threshold: mAP计算的置信度阈值
    
    Returns:
        (epoch_metrics, iter_losses, map_records)
        map_records: 包含每N个iteration的mAP记录列表，每个记录包含epoch、iteration、global_iteration、val_map
    """
    model.train()
    metric_logger = defaultdict[Any, list](list)
    iter_losses: List[float] = []
    map_records: List[Dict[str, float | int]] = []  # 记录每N个iteration的mAP
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

        current_global_iter = global_iteration + iteration
        
        # 每N个iteration计算一次mAP
        if eval_map_every is not None and val_loader is not None and current_global_iter % eval_map_every == 0:
            val_map = evaluate_map(model, val_loader, device, score_threshold=score_threshold)
            print(f"[Epoch {epoch} | Global Iter {current_global_iter}] val_map={val_map:.4f}")
            # 记录mAP值
            map_records.append({
                "epoch": epoch,
                "iteration": iteration,
                "global_iteration": current_global_iter,
                "val_map": val_map,
            })
            # 确保计算完mAP后模型切换回训练模式
            model.train()

        if iteration % print_freq == 0:
            current = time.time()
            avg_loss = sum(metric_logger["loss_classifier"]) / len(metric_logger["loss_classifier"])
            print(f"[Epoch {epoch} | Iter {iteration}] loss={avg_loss:.4f} ({current - start:.1f}s)")
            _save_avg_loss_to_json(avg_loss, iteration, epoch, output_dir=output_dir)

    epoch_metrics = {k: sum(v) / len(v) for k, v in metric_logger.items() if v}
    return epoch_metrics, iter_losses, map_records


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


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    score_threshold: float = 0.05,
    iou_thresholds: List[float] | None = None,
) -> float:
    """
    评估模型在验证集上的加权平均precision (mAP)。
    
    Args:
        model: 检测模型
        data_loader: 验证数据加载器
        device: 计算设备
        score_threshold: 预测框的置信度阈值，低于此值的预测框将被过滤
        iou_thresholds: IoU阈值列表，默认为 [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    Returns:
        加权平均precision (mAP) 值
    """
    model.eval()  # 评估模式，用于获取预测结果
    
    all_pred_boxes: List[Tensor] = []
    all_pred_scores: List[Tensor] = []
    all_gt_boxes: List[Tensor] = []
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 获取模型预测
        outputs = model(images)
        
        # 处理每个样本
        for output, target in zip(outputs, targets):
            # 获取预测框和置信度
            pred_boxes = output["boxes"].cpu()  # [N, 4] xyxy格式
            pred_scores = output["scores"].cpu()  # [N]
            
            # 过滤低置信度的预测框
            if len(pred_boxes) > 0:
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
            
            # 获取真实标注框
            gt_boxes = target["boxes"].cpu()  # [M, 4] xyxy格式
            
            # 收集数据
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_gt_boxes.append(gt_boxes)
    
    # 计算mAP（所有框都是xyxy格式）
    map_score = calculate_mean_average_precision(
        all_pred_boxes,
        all_pred_scores,
        all_gt_boxes,
        iou_thresholds=iou_thresholds,
        pred_format="xyxy",
        gt_format="xyxy",
    )
    
    return map_score

