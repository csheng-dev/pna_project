# plot_metrics.py 使用说明

## 使用方法

### 方法 1：使用默认路径（最简单）

如果训练时使用默认的输出目录 `outputs/faster_rcnn/`，可以直接运行：

```bash
cd /home/sheng/project/pna_project
python plot_metrics.py
```

这会自动读取：
- `outputs/faster_rcnn/metrics.jsonl` - epoch级别的指标
- `outputs/faster_rcnn/metrics_iters.jsonl` - 每个iteration的训练loss
- `outputs/faster_rcnn/avg_iter_loss.json` - avg_loss和iteration数据

生成的图表会保存在 `outputs/faster_rcnn/` 目录下。

### 方法 2：自定义输出目录

如果训练时使用了自定义的输出目录，可以指定路径：

```bash
python plot_metrics.py --out-dir outputs/my_experiment
```

这会从 `outputs/my_experiment/` 目录读取文件，并将图表保存到同一目录。

### 方法 3：指定所有文件路径

如果需要完全自定义所有文件的路径：

```bash
python plot_metrics.py \
    --metrics-file outputs/my_exp/metrics.jsonl \
    --iter-metrics-file outputs/my_exp/metrics_iters.jsonl \
    --avg-iter-loss-file outputs/my_exp/avg_iter_loss.json \
    --out-dir outputs/my_exp
```

## 生成的图表

脚本会生成以下 PNG 图片（如果相应的数据文件存在）：

1. **losses_train.png** - 训练时各个loss分量随epoch的变化
2. **losses_val.png** - 验证时各个loss分量随epoch的变化
3. **val_loss.png** - 总验证损失随epoch的变化
4. **loss_iter.png** - 每个iteration的训练loss曲线
5. **avg_loss_vs_iteration.png** - avg_loss vs iteration 曲线（新添加的功能）

## 参数说明

- `--metrics-file`: metrics.jsonl文件路径（epoch级别指标）
- `--iter-metrics-file`: metrics_iters.jsonl文件路径（每个iteration的loss）
- `--avg-iter-loss-file`: avg_iter_loss.json文件路径（avg_loss和iteration数据）
- `--out-dir`: 输出目录，用于保存生成的图表。如果不指定，默认使用metrics-file的父目录

## 示例

假设你的训练输出在 `outputs/faster_rcnn/`：

```bash
# 最简单的方式
python plot_metrics.py

# 或者指定输出目录
python plot_metrics.py --out-dir outputs/faster_rcnn
```

生成的图表会保存在 `outputs/faster_rcnn/` 目录下，包括：
- `avg_loss_vs_iteration.png` - avg_loss随iteration变化的曲线图


