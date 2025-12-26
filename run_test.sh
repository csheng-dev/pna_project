#!/bin/bash

# 运行 test.py 的启动脚本
# 用法: ./run_test_or_val.sh [gpu_id]
# gpu_id: 0 或 1，指定使用的GPU卡号（可选，默认自动检测）

set -e  # 遇到错误立即退出

# 获取GPU编号参数（如果提供）
GPU_ID="$1"

# 设置设备参数
if [ -z "$GPU_ID" ]; then
    # 未指定GPU，使用自动检测
    DEVICE_ARG=""
else
    # 指定GPU
    DEVICE_ARG="--device cuda:$GPU_ID"
fi

# 对测试集进行预测
python src/test.py \
    --checkpoint outputs/faster_rcnn/202512151433/best_model_map.pth \
    --images-dir data/stage_2_test_images \
    --sample-submission data/stage_2_sample_submission.csv \
    --output-csv outputs/faster_rcnn/202512151433/submission.csv \
    --batch-size 2 \
    --num-workers 0 \
    --score-threshold 0.5 \
    --test-sample-limit 1000 \
    $DEVICE_ARG


