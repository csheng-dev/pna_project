"""Utility helpers for training the Faster R-CNN model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def seed_everything(seed: int) -> None:
    """
    统一设置 Python / NumPy / Torch 的随机种子，保证实验可复现。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_patient_split(
    labels_csv: str | Path,
    val_split: float,
    seed: int,
    patient_subset: Sequence[str] | None = None,
) -> Tuple[Sequence[str], Sequence[str]]:
    """
    基于病人级别做分层抽样，使训练 / 验证的阴阳性比例保持一致。
    """

    df = pd.read_csv(labels_csv)
    if patient_subset is not None:
        df = df[df["patientId"].isin(patient_subset)]

    per_patient_target = df.groupby("patientId")["Target"].max()
    patients = per_patient_target.index.to_numpy()
    labels = per_patient_target.to_numpy()

    train_ids, val_ids = train_test_split(
        patients,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    return train_ids.tolist(), val_ids.tolist()


def save_config(config: dict, output_dir: str | Path) -> None:
    """
    将当前实验配置写入 JSON，方便以后复盘参数。
    """

    path = Path(output_dir) / "run_config.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型中可训练参数数量，可用于 sanity check。
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

