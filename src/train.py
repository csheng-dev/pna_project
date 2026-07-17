"""
入口脚本：解析命令行参数、构建数据管线并驱动 Faster R-CNN 训练流程。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import RSNADataset, collate_fn
from engine import evaluate, evaluate_map, train_one_epoch
from transforms import build_eval_transforms, build_train_transforms
from utils import count_parameters, save_config, seed_everything, stratified_patient_split


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """加载 torchvision Faster R-CNN 并替换分类头，以适应 2 类（背景+肺炎）。"""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    ### NMS ###
    # 查看当前NMS参数
    print(f"score_thresh: {model.roi_heads.score_thresh}")
    print(f"nms_thresh: {model.roi_heads.nms_thresh}")
    print(f"detections_per_img: {model.roi_heads.detections_per_img}")

    # 修改NMS参数（如果需要）
    model.roi_heads.score_thresh = 0.3  # 提高分数阈值
    model.roi_heads.nms_thresh = 0.4    # 降低NMS IoU阈值（更严格的NMS）
    model.roi_heads.detections_per_img = 200  # 增加每张图片的检测框数量
    ### NMS ###

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    """集中管理命令行参数，方便脚本化训练。"""
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on RSNA Pneumonia data.")
    parser.add_argument("--images-dir", type=Path, default=Path("data/stage_2_train_images"), help="Directory with DICOM files.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/stage_2_train_labels.csv"),
        help="CSV with bounding boxes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/faster_rcnn"), help="Where to store checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=3)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--cache-images", action="store_true", help="Cache decoded DICOM tensors in memory.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional cap on number of training patients for debugging.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional cap on number of validation patients for debugging.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--device", type=str, default=None, help="Torch device identifier (e.g., cuda:0).")
    parser.add_argument("--eval-map", action="store_true", help="Evaluate mAP during validation.")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score threshold for mAP evaluation.")
    parser.add_argument("--eval-map-every", type=int, default=5000, help="Evaluate mAP every N iterations (default: 5000).")
    return parser.parse_args()


def save_checkpoint(state: Dict, output_dir: Path, filename: str) -> None:
    """保存 checkpoint，包含模型/优化器/调度器及可选 AMP scaler。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / filename)


def main() -> None:
    args = parse_args()
    # 创建基于当前时间的实验文件夹（格式：YYYYMMDDHHMM）
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    experiment_dir = args.output_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    output_dir = experiment_dir
    
    print(f"Experiment output directory: {output_dir}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    # 保持随机性一致，方便复现实验。
    seed_everything(args.seed)

    train_ids, val_ids = stratified_patient_split(args.labels_csv, args.val_split, args.seed)
    if args.limit_train:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val:
        val_ids = val_ids[: args.limit_val]

    train_dataset = RSNADataset(
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        patient_ids=train_ids,
        transforms=build_train_transforms(),
        cache_images=args.cache_images,
    )
    val_dataset = RSNADataset(
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        patient_ids=val_ids,
        transforms=build_eval_transforms(),
        cache_images=args.cache_images,
    )

    train_loader = DataLoader[Any](
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=False,  # 避免文件描述符泄漏
        pin_memory=False,  # 如果 num_workers=0，pin_memory 应该为 False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=False,  # 避免文件描述符泄漏
        pin_memory=False,  # 如果 num_workers=0，pin_memory 应该为 False
    )

    model = build_model(num_classes=2, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    start_epoch = 1
    # 根据 PyTorch 版本选择 GradScaler 初始化方式
    if args.use_amp and device.type == "cuda":
        try:
            # PyTorch >= 2.0.0 推荐使用新 API
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            # 旧版本 PyTorch 回退到旧 API
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    config = {
        "images_dir": str(args.images_dir),
        "labels_csv": str(args.labels_csv),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "num_workers": args.num_workers,
        "val_split": args.val_split,
        "seed": args.seed,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "parameters": count_parameters(model),
    }
    save_config(config, output_dir)

    metrics_path = output_dir / "metrics.jsonl"
    iter_metrics_path = output_dir / "metrics_iters.jsonl"
    avg_iter_loss_path = output_dir / "avg_iter_loss.json"
    iter_map_path = output_dir / "metrics_map_iters.jsonl"  # 记录每N个iteration的mAP
    # 每次新启动训练前清空 iter 级别日志。
    if iter_metrics_path.exists():
        iter_metrics_path.unlink()
    # 每次新启动训练前清空 avg_iter_loss.json
    if avg_iter_loss_path.exists():
        avg_iter_loss_path.unlink()
    # 每次新启动训练前清空 iter 级别的 mAP 日志
    if iter_map_path.exists():
        iter_map_path.unlink()
    best_val_loss = float("inf")

    best_map = 0.0  # 用于跟踪最佳mAP
    
    # 计算全局iteration计数（用于跨epoch的iteration计数）
    # 如果从checkpoint恢复，需要计算之前的iteration数
    global_iteration = 0
    if args.resume and "global_iteration" in checkpoint:
        global_iteration = checkpoint["global_iteration"]
    else:
        # 计算之前所有epoch的总iteration数
        for prev_epoch in range(1, start_epoch):
            global_iteration += len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        # 训练一个epoch，并传递全局iteration计数和mAP评估参数
        train_metrics, iter_losses, map_records = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            device, 
            epoch, 
            scaler=scaler, 
            output_dir=output_dir,
            global_iteration=global_iteration,
            eval_map_every=args.eval_map_every if args.eval_map else None,
            val_loader=val_loader if args.eval_map else None,
            score_threshold=args.score_threshold,
        )
        
        # 记录每N个iteration的mAP
        if args.eval_map and map_records:
            for map_record in map_records:
                with iter_map_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(map_record) + "\n")
        
        # 更新全局iteration计数
        global_iteration += len(train_loader)
        
        val_metrics = evaluate(model, val_loader, device, scaler=scaler)
        
        # 计算mAP（如果启用）
        val_map = None
        if args.eval_map:
            val_map = evaluate_map(model, val_loader, device, score_threshold=args.score_threshold)
        
        lr_scheduler.step()

        val_loss = sum(val_metrics.values())
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # 如果启用了mAP评估，也根据mAP保存最佳模型
        if args.eval_map and val_map is not None:
            if val_map > best_map:
                best_map = val_map
                # 保存基于mAP的最佳模型
                checkpoint_state_map = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "val_loss": val_loss,
                    "val_map": val_map,
                    "global_iteration": global_iteration,
                }
                save_checkpoint(checkpoint_state_map, output_dir, "best_model_map.pth")

        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "val_loss": val_loss,
            "global_iteration": global_iteration,
        }
        if args.eval_map and val_map is not None:
            checkpoint_state["val_map"] = val_map
        save_checkpoint(checkpoint_state, output_dir, "last_checkpoint.pth")
        if is_best:
            save_checkpoint(checkpoint_state, output_dir, "best_model.pth")

        log_entry = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        if args.eval_map and val_map is not None:
            log_entry["val_map"] = val_map
            log_entry["best_map"] = best_map
        with metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_entry) + "\n")

        # 逐 iter 记录训练 loss 曲线（全局 step = 从 1 累加）
        for iteration, loss_value in enumerate(iter_losses, start=1):
            iter_entry = {
                "epoch": epoch,
                "iteration": iteration,
                "loss": loss_value,
            }
            with iter_metrics_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(iter_entry) + "\n")

        if args.eval_map and val_map is not None:
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f} best={best_val_loss:.4f} val_map={val_map:.4f} best_map={best_map:.4f}")
        else:
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f} best={best_val_loss:.4f}")


if __name__ == "__main__":
    main()

