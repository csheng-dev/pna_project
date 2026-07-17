"""
比较 Ground Truth (GT) 和 Prediction (Pred) 的可视化工具。
在同一张图像上绘制 GT 框和预测框，使用不同颜色区分。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw, ImageFont
from add_frame import load_dicom_image



def draw_boxes_on_image(
    image: np.ndarray,
    gt_boxes: np.ndarray | None = None,
    pred_boxes: np.ndarray | None = None,
    gt_color: str = "green",
    pred_color: str = "red",
    gt_label: str = "GT",
    pred_label: str = "Pred",
) -> Image.Image:
    """
    在图像上绘制 GT 框和预测框。
    
    Args:
        image: numpy 数组，形状为 (H, W)
        gt_boxes: GT 框数组，形状为 (N, 4)，格式为 (x, y, width, height)
        pred_boxes: 预测框数组，形状为 (M, 4)，格式为 (x, y, width, height)
        gt_color: GT 框的颜色（默认：绿色）
        pred_color: 预测框的颜色（默认：红色）
        gt_label: GT 框的标签文本
        pred_label: 预测框的标签文本
    
    Returns:
        PIL Image 对象
    """
    # 转换为 PIL Image（如果是灰度图，转换为RGB模式以支持彩色绘制）
    pil_image = Image.fromarray(image)
    # 如果图像是灰度模式，转换为RGB模式以支持彩色绘制
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体（如果失败则使用默认字体）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # 绘制 GT 框（绿色）
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x, y, width, height = box
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            
            # 绘制矩形框
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=3)
            # 绘制标签
            draw.text((x1, y1 - 20), gt_label, fill=gt_color, font=font)
    
    # 绘制预测框（红色）
    if pred_boxes is not None and len(pred_boxes) > 0:
        for box in pred_boxes:
            x, y, width, height = box
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            
            # 绘制矩形框（使用虚线样式，通过绘制多条短线实现）
            draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=3)
            # 绘制标签
            draw.text((x1, y1 - 40 if gt_boxes is not None and len(gt_boxes) > 0 else y1 - 20), 
                     pred_label, fill=pred_color, font=font)
    
    return pil_image


def compare_gt_vs_pred(
    gt_labels_csv: str | Path,
    pred_labels_csv: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    patient_ids: list[str] | None = None,
    num_images: int | None = None,
) -> None:
    """
    比较 GT 和预测结果，在同一张图像上绘制两种框。
    
    Args:
        gt_labels_csv: Ground Truth 标签 CSV 文件路径
        pred_labels_csv: 预测标签 CSV 文件路径
        images_dir: DICOM 图像目录
        output_dir: 输出目录
        patient_ids: 要处理的 patient ID 列表（如果为 None，则从 CSV 中获取）
        num_images: 要处理的图像数量（如果为 None，则处理所有图像）
    """
    # 读取 CSV 文件
    gt_df = pd.read_csv(gt_labels_csv)
    pred_df = pd.read_csv(pred_labels_csv)
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果没有指定 patient_ids，则获取所有有标注的 patient IDs
    if patient_ids is None:
        # 获取 GT 中有 Target=1 的 patient IDs
        gt_positive = gt_df[gt_df["Target"] == 1]["patientId"].unique().tolist()
        # 获取预测中有 Target=1 的 patient IDs
        pred_positive = pred_df[pred_df["Target"] == 1]["patientId"].unique().tolist()
        # 合并并去重
        patient_ids = list(set(gt_positive + pred_positive))
    
    # 限制处理数量
    if num_images is not None:
        patient_ids = patient_ids[:num_images]
    
    print(f"找到 {len(patient_ids)} 个 patient IDs 需要处理")
    
    for idx, patient_id in enumerate(patient_ids, 1):
        print(f"处理 {idx}/{len(patient_ids)}: {patient_id}")
        
        # 查找 DICOM 文件
        dicom_path = images_dir / f"{patient_id}.dcm"
        if not dicom_path.exists():
            print(f"  警告: 找不到文件 {dicom_path}")
            continue
        
        # 获取 GT 框
        gt_patient_df = gt_df[gt_df["patientId"] == patient_id]
        gt_target_rows = gt_patient_df[gt_patient_df["Target"] == 1]
        gt_boxes = None
        if len(gt_target_rows) > 0:
            gt_boxes = gt_target_rows[["x", "y", "width", "height"]].dropna().values
        
        # 获取预测框
        pred_patient_df = pred_df[pred_df["patientId"] == patient_id]
        pred_target_rows = pred_patient_df[pred_patient_df["Target"] == 1]
        pred_boxes = None
        if len(pred_target_rows) > 0:
            pred_boxes = pred_target_rows[["x", "y", "width", "height"]].dropna().values
        
        try:
            # 读取 DICOM 图像
            image_array = load_dicom_image(dicom_path)
            
            # 在图像上绘制框
            annotated_image = draw_boxes_on_image(
                image_array,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                gt_color="green",
                pred_color="red",
                gt_label="GT",
                pred_label="Pred",
            )
            
            # 保存图像
            output_path = output_dir / f"{patient_id}_GT_vs_Pred.png"
            annotated_image.save(output_path)
            print(f"  已保存: {output_path}")
            if gt_boxes is not None:
                print(f"    GT 框数量: {len(gt_boxes)}")
            if pred_boxes is not None:
                print(f"    预测框数量: {len(pred_boxes)}")
            
        except Exception as e:
            print(f"  错误: 处理 {patient_id} 时出错: {e}")
            continue
    
    print(f"\n处理完成！共处理 {len(patient_ids)} 个图像")


def main():
    """主程序"""
    # 设置路径
    gt_labels_csv = Path("data/stage_2_train_labels.csv")
    pred_labels_csv = Path("outputs/faster_rcnn/202512151433/val_labels_nms_formatted.csv")
    images_dir = Path("data/stage_2_train_images")
    output_dir = Path("outputs/gt_vs_pred_comparison_nms")
    
    # 比较 GT 和预测结果
    compare_gt_vs_pred(
        gt_labels_csv=gt_labels_csv,
        pred_labels_csv=pred_labels_csv,
        images_dir=images_dir,
        output_dir=output_dir,
        patient_ids=None,  # None 表示处理所有有标注的图像
        num_images=2000,    # 只处理前100个图像
    )


if __name__ == "__main__":
    main()

