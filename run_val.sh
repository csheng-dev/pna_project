#!/bin/bash

# 运行 val.py 的启动脚本
# 用法: ./run_val.sh [gpu_id]
# gpu_id: 0 或 1，指定使用的GPU卡号（可选，默认自动检测）

set -e  # 遇到错误立即退出

# 运行验证脚本
if [ -z "$1" ]; then
    # 未指定GPU，使用自动检测
    python src/val.py \
        --checkpoint outputs/faster_rcnn/202512151433/best_model_map.pth \
        --images-dir data/stage_2_train_images \
        --labels-csv data/stage_2_train_labels.csv \
        --output-csv outputs/faster_rcnn/202512151433/val_labels_nms.csv \
        --batch-size 2 \
        --num-workers 0 \
        --score-threshold 0.5 \
        --val-images-limit 5000
else
    # 指定GPU
    python src/val.py \
        --checkpoint outputs/faster_rcnn/202512151433/best_model_map.pth \
        --images-dir data/stage_2_train_images \
        --labels-csv data/stage_2_train_labels.csv \
        --output-csv val_labels_nms.csv \
        --batch-size 2 \
        --num-workers 0 \
        --score-threshold 0.5 \
        --val-images-limit 5000 \
        --device cuda:$1
fi

