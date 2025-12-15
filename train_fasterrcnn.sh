#!/bin/bash
# Faster R-CNN 训练启动脚本
# 使用方法: bash train_fasterrcnn.sh 或 ./train_fasterrcnn.sh

set -e  # 遇到错误立即退出

# 设置项目根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "Faster R-CNN 训练启动脚本"
echo "工作目录: $SCRIPT_DIR"
echo "=========================================="

# 增加文件描述符限制（避免 "Too many open files" 错误）
# 尝试设置更高的限制
ulimit -n 8192 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# 激活 conda 环境（如果需要）
# 先 source conda 的初始化脚本，然后激活环境
# 根据你的 conda 安装路径调整下面的路径
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# 激活环境
conda activate pna_env

# 执行训练
python src/train.py \
    --images-dir data/stage_2_train_images \
    --labels-csv data/stage_2_train_labels.csv \
    --output-dir outputs/faster_rcnn \
    --epochs 10 \
    --batch-size 2 \
    --val-split 0.1 \
    --num-workers 0 \
    --lr 0.005 \
    --weight-decay 1e-4 \
    --lr-step-size 3 \
    --lr-gamma 0.1 \
    --seed 42 \
    --eval-map \
    --score-threshold 0.05 \
    --eval-map-every 5000 \
    --use-amp
    # --cache-images  # 注释掉，避免内存不足（OOM）

# 可选参数（取消注释以使用）：
# --device cuda:0 \
# --limit-train 100 \
# --limit-val 20 \
# --resume outputs/faster_rcnn/202512122208/last_checkpoint.pth \

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="

# 测试数据
# python src/test.py \
#     --checkpoint outputs/faster_rcnn/best_model.pth \
#     --images-dir data/stage_2_test_images \
#     --sample-submission data/stage_2_sample_submission.csv \
#     --output-csv outputs/faster_rcnn/submission.csv