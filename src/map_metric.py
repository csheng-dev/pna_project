"""
计算基于IoU的平均Precision (mAP)指标。

该模块实现了在多个IoU阈值下计算平均精度的评估指标。
IoU阈值范围从0.4到0.75，步长为0.05。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


def convert_xywh_to_xyxy(boxes: Tensor | np.ndarray) -> Tensor:
    """
    将边界框从 (x, y, width, height) 格式转换为 (x1, y1, x2, y2) 格式。
    
    Args:
        boxes: 形状为 [N, 4] 的tensor或数组，格式为 [x, y, width, height]
    
    Returns:
        形状为 [N, 4] 的tensor，格式为 [x1, y1, x2, y2]
    """
    if not isinstance(boxes, Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    
    # 处理单个框的情况
    if len(boxes.shape) == 1:
        boxes = boxes.unsqueeze(0)
    
    x = boxes[:, 0]
    y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    
    return torch.stack([x1, y1, x2, y2], dim=1)


def convert_xyxy_to_xywh(boxes: Tensor | np.ndarray) -> Tensor:
    """
    将边界框从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height) 格式。
    
    Args:
        boxes: 形状为 [N, 4] 的tensor或数组，格式为 [x1, y1, x2, y2]
    
    Returns:
        形状为 [N, 4] 的tensor，格式为 [x, y, width, height]
    """
    if not isinstance(boxes, Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    
    # 处理单个框的情况
    if len(boxes.shape) == 1:
        boxes = boxes.unsqueeze(0)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    
    return torch.stack([x, y, width, height], dim=1)


def calculate_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两组边界框之间的IoU (Intersection over Union)。
    
    Args:
        boxes1: 形状为 [N, 4] 的tensor，格式为 [x1, y1, x2, y2]
        boxes2: 形状为 [M, 4] 的tensor，格式为 [x1, y1, x2, y2]
    
    Returns:
        形状为 [N, M] 的tensor，表示boxes1中每个框与boxes2中每个框的IoU
    """
    # 确保boxes是tensor
    if not isinstance(boxes1, Tensor):
        boxes1 = torch.tensor(boxes1, dtype=torch.float32)
    if not isinstance(boxes2, Tensor):
        boxes2 = torch.tensor(boxes2, dtype=torch.float32)
    
    # 计算交集区域
    # boxes1: [N, 4], boxes2: [M, 4]
    # 扩展维度以便广播: boxes1 -> [N, 1, 4], boxes2 -> [1, M, 4]
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
    
    # 计算交集的左上角和右下角坐标
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])  # [N, M]
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])  # [N, M]
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])  # [N, M]
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])  # [N, M]
    
    # 计算交集面积（如果交集存在）
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height  # [N, M]
    
    # 计算每个框的面积
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [N, 1]
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [1, M]
    
    # 计算并集面积
    union_area = boxes1_area + boxes2_area - inter_area  # [N, M]
    
    # 计算IoU，避免除零
    iou = inter_area / torch.clamp(union_area, min=1e-6)  # [N, M]
    
    return iou


def calculate_precision_at_iou_threshold(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    gt_boxes: Tensor,
    iou_threshold: float,
) -> float:
    """
    在给定的IoU阈值下计算单张图像的precision。
    
    Args:
        pred_boxes: 预测的边界框，形状为 [N, 4]，格式为 [x1, y1, x2, y2]
        pred_scores: 预测的置信度分数，形状为 [N]
        gt_boxes: 真实标注的边界框，形状为 [M, 4]，格式为 [x1, y1, x2, y2]
        iou_threshold: IoU阈值
    
    Returns:
        该IoU阈值下的precision值，计算公式为: precision = TP / (TP + FP + FN)
    """
    # 特殊情况：如果没有真实标注框，任何预测都会导致precision为0
    if len(gt_boxes) == 0:
        if len(pred_boxes) > 0:
            return 0.0
        else:
            # 如果没有预测也没有真实标注，通常返回1.0（完美匹配）
            return 1.0
    
    # 如果没有预测框，precision为0（因为TP=0, FP=0, 但根据定义应该是0）
    if len(pred_boxes) == 0:
        return 0.0
    
    # 按置信度降序排序预测框（置信度高的先评估）
    if len(pred_scores) > 0:
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
    
    # 计算所有预测框与所有真实框的IoU
    iou_matrix = calculate_iou(pred_boxes, gt_boxes)  # [N, M]
    
    # 跟踪哪些真实框已被匹配
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    # 统计TP和FP
    true_positives = 0
    false_positives = 0
    
    # 遍历每个预测框（按置信度从高到低）
    for i in range(len(pred_boxes)):
        # 找到与当前预测框IoU最高的真实框
        ious = iou_matrix[i]  # [M]
        
        # 只考虑IoU超过阈值且未被匹配的真实框
        valid_ious = ious.clone()
        valid_ious[gt_matched] = -1  # 已匹配的框设为-1，确保不会被选中
        
        max_iou, best_gt_idx = torch.max(valid_ious, dim=0)
        
        if max_iou >= iou_threshold:
            # 找到了匹配的真实框，这是一个TP
            true_positives += 1
            gt_matched[best_gt_idx] = True
        else:
            # 没有找到匹配的真实框，这是一个FP
            false_positives += 1
    
    # 计算FN：未被匹配的真实框数量
    false_negatives = len(gt_boxes) - true_positives
    
    # 计算precision: precision = TP / (TP + FP + FN)
    denominator = true_positives + false_positives + false_negatives
    if denominator == 0:
        precision = 0.0
    else:
        precision = true_positives / denominator
    
    return float(precision)


def calculate_average_precision_single_image(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    gt_boxes: Tensor,
    iou_thresholds: List[float] | None = None,
    pred_format: str = "xyxy",
    gt_format: str = "xyxy",
) -> float:
    """
    计算单张图像的平均precision（在多个IoU阈值下的precision的平均值）。
    
    Args:
        pred_boxes: 预测的边界框，形状为 [N, 4]
        pred_scores: 预测的置信度分数，形状为 [N]
        gt_boxes: 真实标注的边界框，形状为 [M, 4]
        iou_thresholds: IoU阈值列表，默认为 [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        pred_format: 预测框的格式，"xyxy" (x1, y1, x2, y2) 或 "xywh" (x, y, width, height)
        gt_format: 真实框的格式，"xyxy" (x1, y1, x2, y2) 或 "xywh" (x, y, width, height)
    
    Returns:
        单张图像的平均precision值
    """
    if iou_thresholds is None:
        iou_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    # 确保输入是tensor
    if not isinstance(pred_boxes, Tensor):
        pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
    if not isinstance(pred_scores, Tensor):
        pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
    if not isinstance(gt_boxes, Tensor):
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
    
    # 处理空输入
    if len(pred_boxes.shape) == 1 and pred_boxes.shape[0] == 4:
        pred_boxes = pred_boxes.unsqueeze(0)
    if len(gt_boxes.shape) == 1 and gt_boxes.shape[0] == 4:
        gt_boxes = gt_boxes.unsqueeze(0)
    
    # 转换格式为 xyxy（如果需要）
    if pred_format == "xywh":
        pred_boxes = convert_xywh_to_xyxy(pred_boxes)
    if gt_format == "xywh":
        gt_boxes = convert_xywh_to_xyxy(gt_boxes)
    
    # 在每个IoU阈值下计算precision
    precisions = []
    for threshold in iou_thresholds:
        precision = calculate_precision_at_iou_threshold(
            pred_boxes, pred_scores, gt_boxes, threshold
        ) / threshold
        precisions.append(precision)
    
    # 返回所有阈值的precision的平均值
    return float(np.mean(precisions))


def calculate_mean_average_precision(
    all_pred_boxes: List[Tensor],
    all_pred_scores: List[Tensor],
    all_gt_boxes: List[Tensor],
    iou_thresholds: List[float] | None = None,
    pred_format: str = "xyxy",
    gt_format: str = "xyxy",
) -> float:
    """
    计算多张图像的平均precision（mAP）。
    
    Args:
        all_pred_boxes: 每张图像的预测边界框列表，每个元素形状为 [N_i, 4]
        all_pred_scores: 每张图像的预测置信度列表，每个元素形状为 [N_i]
        all_gt_boxes: 每张图像的真实标注边界框列表，每个元素形状为 [M_i, 4]
        iou_thresholds: IoU阈值列表，默认为 [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        pred_format: 预测框的格式，"xyxy" (x1, y1, x2, y2) 或 "xywh" (x, y, width, height)
        gt_format: 真实框的格式，"xyxy" (x1, y1, x2, y2) 或 "xywh" (x, y, width, height)
    
    Returns:
        所有图像的平均precision（mAP）值
    """
    if iou_thresholds is None:
        iou_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    # 确保列表长度一致
    assert len(all_pred_boxes) == len(all_pred_scores) == len(all_gt_boxes), \
        "预测框、置信度和真实框的列表长度必须一致"
    
    # 计算每张图像的平均precision
    ap_values = []
    for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        ap = calculate_average_precision_single_image(
            pred_boxes, pred_scores, gt_boxes, iou_thresholds, pred_format, gt_format
        )
        ap_values.append(ap)
    
    # 返回所有图像AP的平均值
    return float(np.mean(ap_values))


# 示例使用
if __name__ == "__main__":
    # 示例1：使用 xyxy 格式（x1, y1, x2, y2）
    print("=" * 50)
    print("示例1: 使用 xyxy 格式")
    print("=" * 50)
    
    # 预测框：[x1, y1, x2, y2]
    pred_boxes_xyxy = torch.tensor([
        [10, 10, 50, 50],
        [60, 60, 100, 100],
        [15, 15, 45, 45],
    ], dtype=torch.float32)
    
    # 预测置信度（按置信度排序，高的先评估）
    pred_scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    
    # 真实标注框
    gt_boxes_xyxy = torch.tensor([
        [12, 12, 48, 48],
        [65, 65, 95, 95],
    ], dtype=torch.float32)
    
    # 计算单张图像的平均precision
    ap = calculate_average_precision_single_image(
        pred_boxes_xyxy, pred_scores, gt_boxes_xyxy, pred_format="xyxy", gt_format="xyxy"
    )
    print(f"单张图像的平均Precision: {ap:.4f}")
    
    # 示例2：使用 xywh 格式（x, y, width, height）- 从 CSV 文件读取的格式
    print("\n" + "=" * 50)
    print("示例2: 使用 xywh 格式（从 CSV 读取的格式）")
    print("=" * 50)
    
    # 预测框：[x, y, width, height] - 例如从 CSV: 264.0, 152.0, 213.0, 379.0
    pred_boxes_xywh = torch.tensor([
        [10, 10, 40, 40],  # x=10, y=10, width=40, height=40 -> xyxy: [10, 10, 50, 50]
        [60, 60, 40, 40],  # x=60, y=60, width=40, height=40 -> xyxy: [60, 60, 100, 100]
        [15, 15, 30, 30],  # x=15, y=15, width=30, height=30 -> xyxy: [15, 15, 45, 45]
    ], dtype=torch.float32)
    
    # 真实标注框：[x, y, width, height]
    gt_boxes_xywh = torch.tensor([
        [12, 12, 36, 36],  # x=12, y=12, width=36, height=36 -> xyxy: [12, 12, 48, 48]
        [65, 65, 30, 30],  # x=65, y=65, width=30, height=30 -> xyxy: [65, 65, 95, 95]
    ], dtype=torch.float32)
    
    # 计算单张图像的平均precision（指定格式为 xywh）
    ap_xywh = calculate_average_precision_single_image(
        pred_boxes_xywh, pred_scores, gt_boxes_xywh, pred_format="xywh", gt_format="xywh"
    )
    print(f"单张图像的平均Precision (xywh格式): {ap_xywh:.4f}")
    
    # 示例3：多张图像的评估
    print("\n" + "=" * 50)
    print("示例3: 多张图像的评估")
    print("=" * 50)
    
    all_pred_boxes = [pred_boxes_xyxy, pred_boxes_xyxy]
    all_pred_scores = [pred_scores, pred_scores]
    all_gt_boxes = [gt_boxes_xyxy, gt_boxes_xyxy]
    
    map_score = calculate_mean_average_precision(
        all_pred_boxes, all_pred_scores, all_gt_boxes, pred_format="xyxy", gt_format="xyxy"
    )
    print(f"多张图像的平均Precision (mAP): {map_score:.4f}")
    
    # 示例4：格式转换函数的使用
    print("\n" + "=" * 50)
    print("示例4: 格式转换")
    print("=" * 50)
    
    # 从 xywh 转换为 xyxy
    boxes_xywh = torch.tensor([[264.0, 152.0, 213.0, 379.0]], dtype=torch.float32)
    boxes_xyxy = convert_xywh_to_xyxy(boxes_xywh)
    print(f"xywh格式: {boxes_xywh[0].tolist()}")
    print(f"转换为xyxy: {boxes_xyxy[0].tolist()}")
    
    # 从 xyxy 转换回 xywh
    boxes_xywh_back = convert_xyxy_to_xywh(boxes_xyxy)
    print(f"转换回xywh: {boxes_xywh_back[0].tolist()}")

