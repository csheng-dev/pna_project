"""
将 submission.csv 格式转换为 stage_2_train_labels.csv 格式

submission.csv 格式：
- patientId, PredictionString
- PredictionString: "score x y width height score x y width height ..." (每5个数字为一组)
- 空字符串表示没有检测框

stage_2_train_labels.csv 格式：
- patientId, x, y, width, height, Target
- 每个检测框占一行
- Target=0 表示没有检测框，Target=1 表示有检测框
"""

import pandas as pd
from pathlib import Path


def parse_prediction_string(pred_str: str) -> list[tuple[float, float, float, float]]:
    """
    解析 PredictionString，返回检测框列表 (x, y, width, height)
    忽略 score (每组的第一个数字)
    """
    if pd.isna(pred_str) or pred_str == "":
        return []
    
    parts = pred_str.strip().split()
    if len(parts) % 5 != 0:
        raise ValueError(f"PredictionString 长度必须是5的倍数: {pred_str}")
    
    boxes = []
    for i in range(0, len(parts), 5):
        # 格式: score x y width height
        score = float(parts[i])
        x = float(parts[i + 1])
        y = float(parts[i + 2])
        width = float(parts[i + 3])
        height = float(parts[i + 4])
        boxes.append((x, y, width, height))
    
    return boxes


def convert_submission_to_labels(submission_csv: Path, output_csv: Path) -> None:
    """
    将 submission.csv 转换为 stage_2_train_labels.csv 格式
    """
    print(f"读取文件: {submission_csv}")
    df = pd.read_csv(submission_csv)
    
    rows = []
    for _, row in df.iterrows():
        patient_id = row["patientId"]
        pred_str = row["PredictionString"]
        
        boxes = parse_prediction_string(pred_str)
        
        if len(boxes) == 0:
            # 没有检测框，创建一行 Target=0
            rows.append({
                "patientId": patient_id,
                "x": "",
                "y": "",
                "width": "",
                "height": "",
                "Target": 0
            })
        else:
            # 有检测框，为每个检测框创建一行 Target=1
            for x, y, width, height in boxes:
                rows.append({
                    "patientId": patient_id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "Target": 1
                })
    
    # 创建新的DataFrame
    output_df = pd.DataFrame(rows)
    
    # 保存文件
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"转换完成！输出文件: {output_csv}")
    print(f"总计 {len(df)} 个患者，生成 {len(output_df)} 行数据")
    print(f"其中有 {sum(output_df['Target'] == 1)} 行检测框（Target=1），{sum(output_df['Target'] == 0)} 行无检测框（Target=0）")


def main():
    # 输入文件
    submission_csv = Path("outputs/faster_rcnn/202512151433/val_labels.csv")
    
    # 输出文件
    output_csv = Path("outputs/faster_rcnn/202512151433/val_labels_add_frame.csv")
    
    if not submission_csv.exists():
        raise FileNotFoundError(f"文件不存在: {submission_csv}")
    
    convert_submission_to_labels(submission_csv, output_csv)


if __name__ == "__main__":
    main()

