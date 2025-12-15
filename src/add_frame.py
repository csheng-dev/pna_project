"""添加帧相关的功能模块。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw


def load_dicom_image(dicom_path: Path) -> np.ndarray:
    """读取 DICOM 图像并转换为 numpy 数组（归一化到 0-255）。"""
    dataset = pydicom.dcmread(dicom_path)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixel_array = dataset.pixel_array
    
    # 应用 windowing
    scaled = pixel_array.astype(np.float32) * slope + intercept
    scaled -= scaled.min()
    max_value = scaled.max()
    if max_value > 0:
        scaled /= max_value
    
    # 转换为 0-255 的 uint8
    scaled = (scaled * 255).astype(np.uint8)
    return scaled


def draw_boxes_on_image(image: np.ndarray, boxes: np.ndarray) -> Image.Image:
    """
    在图像上绘制标注框。
    
    Args:
        image: numpy 数组，形状为 (H, W)
        boxes: numpy 数组，形状为 (N, 4)，格式为 (x, y, width, height)
    
    Returns:
        PIL Image 对象
    """
    # 转换为 PIL Image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 绘制每个框
    for box in boxes:
        x, y, width, height = box
        # 转换为 x1, y1, x2, y2
        x1, y1 = x, y
        x2, y2 = x + width, y + height
        
        # 绘制矩形框（红色，线宽3）
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    return pil_image


def process_patient_images(
    patient_ids: list[str],
    labels_csv: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    num_images: int = 200,
) -> None:
    """
    处理前 num_images 个 patient_id 的图像，绘制标注框并保存。
    
    Args:
        patient_ids: patient_id 列表
        labels_csv: CSV 文件路径
        images_dir: DICOM 图像目录
        output_dir: 输出目录
        num_images: 要处理的图像数量
    """
    df = pd.read_csv(labels_csv)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理前 num_images 个 patient_id
    selected_ids = patient_ids[:num_images]
    
    for idx, patient_id in enumerate(selected_ids, 1):
        print(f"处理 {idx}/{len(selected_ids)}: {patient_id}")
        
        # 查找 DICOM 文件
        dicom_path = images_dir / f"{patient_id}.dcm"
        if not dicom_path.exists():
            print(f"  警告: 找不到文件 {dicom_path}")
            continue
        
        # 读取该 patient_id 的所有标注框
        patient_df = df[df["patientId"] == patient_id]
        target_rows = patient_df[patient_df["Target"] == 1]
        
        if len(target_rows) == 0:
            print(f"  警告: {patient_id} 没有 Target=1 的标注")
            continue
        
        # 提取框坐标 (x, y, width, height)
        boxes = target_rows[["x", "y", "width", "height"]].dropna().values
        
        if len(boxes) == 0:
            print(f"  警告: {patient_id} 没有有效的框坐标")
            continue
        
        try:
            # 读取 DICOM 图像
            image_array = load_dicom_image(dicom_path)
            
            # 在图像上绘制框
            annotated_image = draw_boxes_on_image(image_array, boxes)
            
            # 保存图像
            output_path = output_dir / f"{patient_id}.png"
            annotated_image.save(output_path)
            print(f"  已保存: {output_path}")
            
        except Exception as e:
            print(f"  错误: 处理 {patient_id} 时出错: {e}")
            continue


def convert_negative_images_to_png(
    labels_csv: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    将 Target=0 的 patient_id 对应的图像转换为 PNG 格式并保存。
    
    Args:
        labels_csv: CSV 文件路径
        images_dir: DICOM 图像目录
        output_dir: 输出目录
    """
    df = pd.read_csv(labels_csv)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 筛选 Target=0 的行
    negative_df = df[df["Target"] == 0]
    
    # 提取 patientId 并去重
    patient_ids = negative_df["patientId"].unique().tolist()[:400]
    
    print(f"找到 {len(patient_ids)} 个不同的 patient_id (Target=0)")
    
    for idx, patient_id in enumerate(patient_ids, 1):
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(patient_ids)}")
        
        # 查找 DICOM 文件
        dicom_path = images_dir / f"{patient_id}.dcm"
        if not dicom_path.exists():
            print(f"  警告: 找不到文件 {dicom_path}")
            continue
        
        try:
            # 读取 DICOM 图像
            image_array = load_dicom_image(dicom_path)
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(image_array)
            
            # 保存图像
            output_path = output_dir / f"{patient_id}.png"
            pil_image.save(output_path)
            
        except Exception as e:
            print(f"  错误: 处理 {patient_id} 时出错: {e}")
            continue
    
    print(f"处理完成！共转换 {len(patient_ids)} 个图像")


# 主程序
if __name__ == "__main__":
    # 读取 CSV
    labels_csv = Path("/Users/sheng/Documents/pna_project/data/stage_2_train_labels.csv/stage_2_train_labels.csv")
    df = pd.read_csv(labels_csv)
    
    # 筛选 Target=1 的行
    positive_df = df[df["Target"] == 1]
    
    # 提取 patientId 并去重
    patient_ids = positive_df["patientId"].unique().tolist()
    
    print(f"找到 {len(patient_ids)} 个不同的 patient_id (Target=1)")
    
    # 处理前200个图像
    images_dir = Path("/Users/sheng/Documents/pna_project/data/stage_2_train_images")
    output_dir = Path("/Users/sheng/Documents/pna_project/data/edited_image")
    
    process_patient_images(
        patient_ids=patient_ids,
        labels_csv=labels_csv,
        images_dir=images_dir,
        output_dir=output_dir,
        num_images=200,
    )
    
    print("处理完成！")
    
    # 处理 Target=0 的图像
    print("\n" + "="*50)
    print("开始处理 Target=0 的图像...")
    print("="*50)
    
    output_dir_no_frame = Path("/Users/sheng/Documents/pna_project/data/edited_image_no_frame")
    
    convert_negative_images_to_png(
        labels_csv=labels_csv,
        images_dir=images_dir,
        output_dir=output_dir_no_frame,
    )
