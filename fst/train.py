"""
训练脚本: 两阶段训练 FST 多头分类器.

Stage 1: 冻结 backbone, 只训练 neck + heads (快速收敛)
Stage 2: 解冻 backbone 最后几层, 全模型微调 (提升精度)

用法:
    python -m fst.train --data ./dataset --epochs1 10 --epochs2 30 --batch 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from fst.network import FSTClassifier, count_parameters
from fst.dataset import FSTDataset, collate_fn
from fst.models import NUM_POSITIONS


class FSTLoss(nn.Module):
    """
    多任务加权损失函数.

    - slot_type, maneuver, obstacles, marking: CrossEntropy
    - special_scene: BCEWithLogitsLoss (多标签)
    """

    def __init__(
        self,
        w_slot: float = 1.0,
        w_maneuver: float = 1.0,
        w_scene: float = 0.8,
        w_obstacle: float = 0.5,
        w_marking: float = 0.6,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_slot = w_slot
        self.w_maneuver = w_maneuver
        self.w_scene = w_scene
        self.w_obstacle = w_obstacle
        self.w_marking = w_marking

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        # Head A: slot_type
        losses["slot_type"] = self.ce(preds["slot_type"], targets["slot_type"]) * self.w_slot

        # Head B: maneuver
        losses["maneuver"] = self.ce(preds["maneuver"], targets["maneuver"]) * self.w_maneuver

        # Head C: special_scene (multi-label)
        losses["special_scene"] = self.bce(preds["special_scene"], targets["special_scene"]) * self.w_scene

        # Head D: obstacles (9 positions)
        obs_loss = torch.tensor(0.0, device=preds["slot_type"].device)
        for i in range(NUM_POSITIONS):
            obs_loss = obs_loss + self.ce(preds[f"obs_{i}"], targets[f"obs_{i}"])
        losses["obstacles"] = (obs_loss / NUM_POSITIONS) * self.w_obstacle

        # Head E: marking
        mk_loss = (
            self.ce(preds["line_color"], targets["line_color"])
            + self.ce(preds["line_vis"], targets["line_vis"])
            + self.ce(preds["line_style"], targets["line_style"])
        ) / 3.0
        losses["marking"] = mk_loss * self.w_marking

        losses["total"] = sum(losses.values())  # type: ignore
        return losses


def compute_accuracy(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """计算各 head 的 accuracy."""
    accs: Dict[str, float] = {}

    for key in ["slot_type", "maneuver", "line_color", "line_vis", "line_style"]:
        pred_cls = preds[key].argmax(dim=-1)
        accs[key] = (pred_cls == targets[key]).float().mean().item()

    # obstacles 平均准确率
    obs_correct = 0.0
    for i in range(NUM_POSITIONS):
        pred_cls = preds[f"obs_{i}"].argmax(dim=-1)
        obs_correct += (pred_cls == targets[f"obs_{i}"]).float().mean().item()
    accs["obstacles_avg"] = obs_correct / NUM_POSITIONS

    # special_scene: 使用 threshold=0.5 计算 F1-like accuracy
    pred_ss = (torch.sigmoid(preds["special_scene"]) > 0.5).float()
    accs["special_scene"] = (pred_ss == targets["special_scene"]).float().mean().item()

    return accs


def train_one_epoch(
    model: FSTClassifier,
    loader: DataLoader,
    criterion: FSTLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        preds = model(imgs)
        losses = criterion(preds, targets)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: FSTClassifier,
    loader: DataLoader,
    criterion: FSTLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_losses: Dict[str, float] = {}
    total_accs: Dict[str, float] = {}
    n_batches = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        preds = model(imgs)
        losses = criterion(preds, targets)
        accs = compute_accuracy(preds, targets)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        for k, v in accs.items():
            total_accs[k] = total_accs.get(k, 0.0) + v
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    avg_accs = {k: v / max(n_batches, 1) for k, v in total_accs.items()}
    return {**{f"loss_{k}": v for k, v in avg_losses.items()},
            **{f"acc_{k}": v for k, v in avg_accs.items()}}


def train(
    data_root: str,
    output_dir: str = "checkpoints",
    backbone: str = "efficientnet_b0",
    img_size: int = 384,
    batch_size: int = 32,
    epochs_stage1: int = 10,
    epochs_stage2: int = 30,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    val_split: float = 0.15,
    patience: int = 10,
    device_str: str = "auto",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 设备 ──
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # ── 数据集 ──
    full_ds = FSTDataset(data_root, img_size=img_size, augment=True)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    # 验证集不做增强
    val_ds.dataset = FSTDataset(data_root, img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    print(f"Dataset: {n_train} train / {n_val} val")

    # ── 模型 ──
    model = FSTClassifier(backbone_name=backbone, pretrained=True, freeze_backbone=True)
    model.to(device)
    total_p, train_p = count_parameters(model)
    print(f"Model params: {total_p:,} total, {train_p:,} trainable (backbone frozen)")

    criterion = FSTLoss()
    history: list = []

    # ── Stage 1: Backbone Frozen ──
    print(f"\n{'='*60}")
    print(f"Stage 1: Backbone frozen, {epochs_stage1} epochs, lr={lr_stage1}")
    print(f"{'='*60}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr_stage1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_stage1, eta_min=1e-6)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs_stage1 + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        val_loss = val_metrics["loss_total"]

        record = {"epoch": epoch, "stage": 1, "elapsed": elapsed,
                  **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        print(f"[S1 E{epoch:02d}] train_loss={train_metrics['total']:.4f} "
              f"val_loss={val_loss:.4f} "
              f"val_acc_slot={val_metrics.get('acc_slot_type', 0):.3f} "
              f"val_acc_mv={val_metrics.get('acc_maneuver', 0):.3f} "
              f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), output_path / "best_stage1.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Stage 2: Fine-tune ──
    print(f"\n{'='*60}")
    print(f"Stage 2: Backbone unfrozen (last 2 blocks), {epochs_stage2} epochs, lr={lr_stage2}")
    print(f"{'='*60}")

    # 加载最优 Stage 1 权重
    model.load_state_dict(torch.load(output_path / "best_stage1.pth", map_location=device, weights_only=True))
    model.unfreeze_backbone(last_n_blocks=2)
    _, train_p = count_parameters(model)
    print(f"Trainable params after unfreeze: {train_p:,}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr_stage2, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_stage2, eta_min=1e-6)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs_stage2 + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        val_loss = val_metrics["loss_total"]

        record = {"epoch": epoch + epochs_stage1, "stage": 2, "elapsed": elapsed,
                  **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        print(f"[S2 E{epoch:02d}] train_loss={train_metrics['total']:.4f} "
              f"val_loss={val_loss:.4f} "
              f"val_acc_slot={val_metrics.get('acc_slot_type', 0):.3f} "
              f"val_acc_mv={val_metrics.get('acc_maneuver', 0):.3f} "
              f"val_acc_obs={val_metrics.get('acc_obstacles_avg', 0):.3f} "
              f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), output_path / "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 保存训练历史
    with open(output_path / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nTraining complete. Best model saved to {output_path / 'best_model.pth'}")
    print(f"Training history saved to {output_path / 'history.json'}")


def main():
    parser = argparse.ArgumentParser(description="Train FST multi-head classifier")
    parser.add_argument("--data", required=True, help="Dataset root (with images/ and labels/)")
    parser.add_argument("--output", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--backbone", default="efficientnet_b0", help="timm backbone name")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs1", type=int, default=10, help="Stage 1 epochs (backbone frozen)")
    parser.add_argument("--epochs2", type=int, default=30, help="Stage 2 epochs (fine-tune)")
    parser.add_argument("--lr1", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--lr2", type=float, default=1e-4, help="Stage 2 learning rate")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train(
        data_root=args.data,
        output_dir=args.output,
        backbone=args.backbone,
        img_size=args.img_size,
        batch_size=args.batch,
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        lr_stage1=args.lr1,
        lr_stage2=args.lr2,
        val_split=args.val_split,
        patience=args.patience,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
