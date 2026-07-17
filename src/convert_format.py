"""
将 val_labels.csv 转换为与 stage_2_train_labels.csv 相同格式的 CSV 文件。

主要功能：
- 保持格式一致：patientId, x, y, width, height, Target
- 将浮点数格式化为整数（与 stage_2_train_labels.csv 格式一致）
- 保持空值（Target=0 时 x, y, width, height 为空）
"""

import pandas as pd
from pathlib import Path


def convert_val_labels_format(input_csv: Path, output_csv: Path) -> None:
    """
    将 val_labels.csv 转换为与 stage_2_train_labels.csv 相同格式。
    
    Args:
        input_csv: 输入的 val_labels.csv 文件路径
        output_csv: 输出的 CSV 文件路径
    """
    print(f"读取文件: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if "patientId" not in df.columns:
        raise ValueError(f"CSV 文件缺少 'patientId' 列")
    
    # 确保列顺序正确
    required_columns = ["patientId", "x", "y", "width", "height", "Target"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 文件缺少必要的列。需要: {required_columns}")
    
    # 创建新的 DataFrame，保持原有结构
    output_df = df.copy()
    
    # 格式化数值列（x, y, width, height）
    # 对于非空值，四舍五入为整数（与 stage_2_train_labels.csv 格式一致）
    numeric_columns = ["x", "y", "width", "height"]
    for col in numeric_columns:
        # 将非空值转换为浮点数后四舍五入为整数
        output_df[col] = output_df[col].apply(
            lambda val: round(float(val)) if pd.notna(val) and val != "" else ""
        )
    
    # 确保 Target 列是整数类型
    output_df["Target"] = output_df["Target"].astype(int)
    
    # 按照 stage_2_train_labels.csv 的列顺序排列
    output_df = output_df[required_columns]
    
    # 保存文件
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    
    print(f"转换完成！输出文件: {output_csv}")
    print(f"总计 {len(output_df)} 行数据")
    print(f"其中有 {sum(output_df['Target'] == 1)} 行检测框（Target=1），{sum(output_df['Target'] == 0)} 行无检测框（Target=0）")
    
    # 显示一些示例数据
    print(f"\n示例数据（前5行）:")
    print(output_df.head().to_string(index=False))


def main():
    """主程序"""
    # 输入文件
    input_csv = Path("val_labels_nms.csv")
    
    # 输出文件（转换为与 stage_2_train_labels.csv 相同格式）
    output_csv = Path("outputs/faster_rcnn/202512151433/val_labels_nms_formatted.csv")
    
    if not input_csv.exists():
        raise FileNotFoundError(f"文件不存在: {input_csv}")
    
    convert_val_labels_format(input_csv, output_csv)


if __name__ == "__main__":
    main()

