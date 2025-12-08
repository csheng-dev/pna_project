"""
简单的可视化脚本：从 metrics.jsonl 中读取每个 epoch 的指标并画出 loss 曲线。

用法（在项目根目录运行）：

    python plot_metrics.py --metrics-file outputs/faster_rcnn/metrics.jsonl --out-dir outputs/faster_rcnn

会在 out-dir 下生成若干 png 图片：
- losses_train.png
- losses_val.png
- val_loss.png  （按 epoch 的总验证损失）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/validation losses from metrics.jsonl.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("outputs/faster_rcnn/metrics.jsonl"),
        help="Path to metrics.jsonl produced by train.py.",
    )
    parser.add_argument(
        "--iter-metrics-file",
        type=Path,
        default=Path("outputs/faster_rcnn/metrics_iters.jsonl"),
        help="Path to metrics_iters.jsonl with per-iteration training loss.",
    )
    parser.add_argument(
        "--avg-iter-loss-file",
        type=Path,
        default=Path("outputs/faster_rcnn/avg_iter_loss.json"),
        help="Path to avg_iter_loss.json with avg_loss and iteration.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Defaults to metrics-file parent directory.",
    )
    return parser.parse_args()


def load_metrics(metrics_file: Path) -> List[Dict]:
    """加载 epoch 级别的指标，如果文件不存在则返回空列表。"""
    if not metrics_file.is_file():
        return []

    records: List[Dict] = []
    with metrics_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_iter_metrics(iter_metrics_file: Path) -> List[Dict]:
    """读取逐 iter 的 loss 日志，每行包含 epoch / iteration / loss。"""

    if not iter_metrics_file.is_file():
        return []

    records: List[Dict] = []
    with iter_metrics_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_avg_iter_loss(json_file: Path) -> List[Dict]:
    """加载avg_iter_loss.json文件，返回包含epoch、iteration和avg_loss的记录列表。"""
    if not json_file.is_file():
        return []
    
    try:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # 确保数据是列表格式
        if isinstance(data, list):
            return data
        else:
            return []
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {json_file}: {e}")
        return []


def plot_metric_dict_per_epoch(
    epochs: List[int],
    metrics_per_epoch: List[Dict[str, float]],
    title: str,
    outfile: Path,
) -> None:
    """
    对于每个 key（例如 loss_classifier、loss_box_reg），画一条随 epoch 变化的曲线。
    """
    # 收集所有 key
    all_keys = sorted({k for m in metrics_per_epoch for k in m.keys()})
    if not all_keys:
        return

    plt.figure(figsize=(8, 5))
    for key in all_keys:
        ys = [m.get(key, float("nan")) for m in metrics_per_epoch]
        plt.plot(epochs, ys, marker="o", label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_val_loss(epochs: List[int], val_losses: List[float], outfile: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Total val loss")
    plt.title("Validation loss per epoch")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_iter_loss(iter_records: List[Dict], outfile: Path) -> None:
    """
    根据 metrics_iters.jsonl 中的记录画出 loss-iter 曲线。
    x 轴使用全局 step（按 epoch、iteration 排序后从 1 递增）。
    """

    if not iter_records:
        return

    # 按 epoch、iteration 排序确保时间顺序
    iter_records = sorted(iter_records, key=lambda r: (int(r["epoch"]), int(r["iteration"])))
    losses = [float(r["loss"]) for r in iter_records]
    steps = list(range(1, len(losses) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("Iteration (global step)")
    plt.ylabel("Training loss")
    plt.title("Training loss per iteration")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def plot_avg_loss_vs_iteration(avg_loss_records: List[Dict], outfile: Path) -> None:
    """
    根据 avg_iter_loss.json 中的记录画出 avg_loss vs iteration 曲线。
    x 轴使用全局 step（按 epoch、iteration 排序后从 1 递增），或直接使用 iteration（如果只有一个 epoch）。
    """
    if not avg_loss_records:
        return
    
    # 按 epoch、iteration 排序确保时间顺序
    avg_loss_records = sorted(avg_loss_records, key=lambda r: (int(r.get("epoch", 0)), int(r["iteration"])))
    
    iterations = [int(r["iteration"]) for r in avg_loss_records]
    avg_losses = [float(r["avg_loss"]) for r in avg_loss_records]
    
    # 如果有多个epoch，使用全局step（从1递增）
    # 如果只有一个epoch，直接使用iteration作为x轴
    epochs = [int(r.get("epoch", 1)) for r in avg_loss_records]
    unique_epochs = sorted(set(epochs))
    
    plt.figure(figsize=(10, 5))
    
    if len(unique_epochs) > 1:
        # 多个epoch：使用全局step
        steps = list(range(1, len(avg_losses) + 1))
        plt.plot(steps, avg_losses, marker="o", markersize=3, linewidth=1)
        plt.xlabel("Iteration (global step)")
    else:
        # 单个epoch：直接使用iteration
        plt.plot(iterations, avg_losses, marker="o", markersize=3, linewidth=1)
        plt.xlabel("Iteration")
    
    plt.ylabel("Average Loss")
    plt.title("Average Loss vs Iteration")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


def main() -> None:
    args = parse_args()
    metrics_file: Path = args.metrics_file
    iter_metrics_file: Path = args.iter_metrics_file
    avg_iter_loss_file: Path = args.avg_iter_loss_file
    out_dir: Path = args.out_dir or metrics_file.parent

    records = load_metrics(metrics_file)
    iter_records = load_iter_metrics(iter_metrics_file)
    avg_loss_records = load_avg_iter_loss(avg_iter_loss_file)

    # 如果有 epoch 级别的数据，画 epoch 相关的图
    if records:
        epochs: List[int] = [int(r["epoch"]) for r in records]
        train_metrics_list: List[Dict[str, float]] = [r.get("train_metrics", {}) for r in records]
        val_metrics_list: List[Dict[str, float]] = [r.get("val_metrics", {}) for r in records]
        val_losses: List[float] = [float(r.get("val_loss", 0.0)) for r in records]

        # 训练/验证各个 loss 分量
        plot_metric_dict_per_epoch(
            epochs,
            train_metrics_list,
            title="Training losses per epoch",
            outfile=out_dir / "losses_train.png",
        )
        plot_metric_dict_per_epoch(
            epochs,
            val_metrics_list,
            title="Validation losses per epoch",
            outfile=out_dir / "losses_val.png",
        )

        # 总验证损失
        plot_val_loss(epochs, val_losses, outfile=out_dir / "val_loss.png")
    else:
        print(f"Warning: {metrics_file} not found, skipping epoch-level plots.")

    # 逐 iter loss 曲线（只要有 iter 数据就画）
    if iter_records:
        plot_iter_loss(iter_records, outfile=out_dir / "loss_iter.png")
    else:
        print(f"Warning: {iter_metrics_file} not found, skipping iteration-level plot.")

    # avg_loss vs iteration 曲线
    if avg_loss_records:
        plot_avg_loss_vs_iteration(avg_loss_records, outfile=out_dir / "avg_loss_vs_iteration.png")
    else:
        print(f"Warning: {avg_iter_loss_file} not found, skipping avg_loss vs iteration plot.")

    if not records and not iter_records and not avg_loss_records:
        raise RuntimeError(f"No metrics found. Check if {metrics_file}, {iter_metrics_file}, or {avg_iter_loss_file} exists.")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()


