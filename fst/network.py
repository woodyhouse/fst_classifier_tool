"""
多头 CNN 分类器网络定义.

Backbone: EfficientNet-B0 (timm)
Heads:
  A) slot_type       → 4 类 softmax
  B) maneuver        → 5 类 softmax
  C) special_scene   → 14 维 sigmoid（多标签）
  D) obstacles       → 9 个位置 × 12 类 softmax
  E) marking         → 3 个子头 (color/vis/style)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import timm

from fst.models import (
    NUM_SLOT_TYPE, NUM_MANEUVER, NUM_SPECIAL_SCENE,
    NUM_OBSTACLE, NUM_LINE_COLOR, NUM_LINE_VIS, NUM_LINE_STYLE,
    NUM_POSITIONS,
)


class ClassificationHead(nn.Module):
    """单分类头: FC → BN → ReLU → Dropout → FC → logits."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FSTClassifier(nn.Module):
    """
    FST 多头分类器.

    Args:
        backbone_name: timm 模型名 (默认 efficientnet_b0)
        pretrained: 是否加载 ImageNet 预训练权重
        neck_dim: 共享 neck 的隐藏维度
        head_hidden: 每个 head 的隐藏层维度
        dropout: dropout 比例
        freeze_backbone: 是否冻结 backbone（Stage 1 训练用）
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        neck_dim: int = 512,
        head_hidden: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ── Backbone ──
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,  # 去掉分类头
        )
        backbone_out_dim = self.backbone.num_features  # e.g. 1280 for efficientnet_b0

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Shared Neck ──
        self.neck = nn.Sequential(
            nn.Linear(backbone_out_dim, neck_dim),
            nn.BatchNorm1d(neck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Head A: slot_type ──
        self.head_slot_type = ClassificationHead(neck_dim, head_hidden, NUM_SLOT_TYPE, dropout)

        # ── Head B: maneuver ──
        self.head_maneuver = ClassificationHead(neck_dim, head_hidden, NUM_MANEUVER, dropout)

        # ── Head C: special_scene (multi-label sigmoid) ──
        self.head_special_scene = ClassificationHead(neck_dim, head_hidden, NUM_SPECIAL_SCENE, dropout)

        # ── Head D: obstacles — 9 个位置各一个 softmax 头 ──
        self.obstacle_heads = nn.ModuleList([
            ClassificationHead(neck_dim, head_hidden, NUM_OBSTACLE, dropout)
            for _ in range(NUM_POSITIONS)
        ])

        # ── Head E: marking — 3 个子头 ──
        self.head_line_color = ClassificationHead(neck_dim, head_hidden, NUM_LINE_COLOR, dropout)
        self.head_line_vis = ClassificationHead(neck_dim, head_hidden, NUM_LINE_VIS, dropout)
        self.head_line_style = ClassificationHead(neck_dim, head_hidden, NUM_LINE_STYLE, dropout)

    def freeze_backbone(self):
        """冻结 backbone 参数 (Stage 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 2):
        """
        解冻 backbone 最后 N 个 block (Stage 2).
        对于 EfficientNet, blocks 存在 self.backbone.blocks 中.
        """
        # 先全部冻结
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 解冻最后 N 个 block
        if hasattr(self.backbone, "blocks"):
            total = len(self.backbone.blocks)
            for block in self.backbone.blocks[max(0, total - last_n_blocks):]:
                for param in block.parameters():
                    param.requires_grad = True
        # 解冻 classifier head (bn/norm)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] 归一化图片张量

        Returns:
            dict with keys:
                "slot_type":      [B, 4]  logits
                "maneuver":       [B, 5]  logits
                "special_scene":  [B, 14] logits (apply sigmoid at inference)
                "obs_0" .. "obs_8": [B, 12] logits (9个位置)
                "line_color":     [B, 6]  logits
                "line_vis":       [B, 4]  logits
                "line_style":     [B, 3]  logits
        """
        feat = self.backbone(x)          # [B, backbone_out_dim]
        feat = self.neck(feat)           # [B, neck_dim]

        out: Dict[str, torch.Tensor] = {}
        out["slot_type"] = self.head_slot_type(feat)
        out["maneuver"] = self.head_maneuver(feat)
        out["special_scene"] = self.head_special_scene(feat)

        for i, head in enumerate(self.obstacle_heads):
            out[f"obs_{i}"] = head(feat)

        out["line_color"] = self.head_line_color(feat)
        out["line_vis"] = self.head_line_vis(feat)
        out["line_style"] = self.head_line_style(feat)

        return out


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回 (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # 快速测试
    model = FSTClassifier(pretrained=False)
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}  Trainable: {trainable:,}")

    dummy = torch.randn(2, 3, 384, 384)
    out = model(dummy)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
