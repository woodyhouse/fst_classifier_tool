"""
PyTorch Dataset: 加载图片 + JSON 标注 → 训练用张量.

支持 FST 感知的数据增强（水平翻转时自动交换 1↔3, 4↔6, P_LEFT↔P_RIGHT）。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from fst.models import (
    SLOT_TYPE_TO_IDX, MANEUVER_TO_IDX, OBSTACLE_TO_IDX,
    LINE_COLOR_TO_IDX, LINE_VIS_TO_IDX, LINE_STYLE_TO_IDX,
    SPECIAL_SCENE_TO_IDX, SPECIAL_SCENE_CLASSES,
    POSITION_KEYS, MIRROR_POS_MAP,
    NUM_SPECIAL_SCENE,
)


class FSTDataset(Dataset):
    """
    FST 车位分类数据集.

    目录结构:
        root/
          images/   xxx.jpg
          labels/   xxx.json

    JSON 标注格式 (精简):
    {
      "image_id": "xxx",
      "slot_type": "PERPENDICULAR",
      "maneuver": "PARK_IN",
      "marking": {"line_color": "WHITE", "line_visibility": "CLEAR", "line_style": "SOLID"},
      "special_scene": {"P0": [], "P1": ["BRICK_GRASS"]},
      "obstacles": {"1": "EMPTY", ..., "P_RIGHT": "VEHICLE"}
    }
    """

    def __init__(
        self,
        root: str | Path,
        img_size: int = 384,
        augment: bool = True,
        mirror_prob: float = 0.5,
    ):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.img_size = img_size
        self.augment = augment
        self.mirror_prob = mirror_prob

        # 收集所有有标注的图片
        self.samples: List[Tuple[Path, Path]] = []
        for lbl_path in sorted(self.lbl_dir.glob("*.json")):
            stem = lbl_path.stem
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                img_path = self.img_dir / (stem + ext)
                if img_path.exists():
                    self.samples.append((img_path, lbl_path))
                    break

        if not self.samples:
            raise FileNotFoundError(
                f"No image-label pairs found in {self.root}. "
                f"Ensure images/ and labels/ directories exist with matching filenames."
            )

        # 标准化 transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # 训练增强
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            self.normalize,
        ])

        # 验证 transform（无增强）
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_label(self, lbl_path: Path) -> Dict[str, Any]:
        with open(lbl_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _mirror_label(self, label: Dict[str, Any]) -> Dict[str, Any]:
        """水平翻转时同步交换标注中的方位信息."""
        obs = label.get("obstacles", {})
        new_obs = {}
        for pos in POSITION_KEYS:
            mirror_pos = MIRROR_POS_MAP.get(pos, pos)
            new_obs[pos] = obs.get(mirror_pos, "UNKNOWN")
        label["obstacles"] = new_obs

        # 寻库方向翻转
        sd = label.get("search_direction", "UNKNOWN")
        if sd == "LEFT":
            label["search_direction"] = "RIGHT"
        elif sd == "RIGHT":
            label["search_direction"] = "LEFT"

        return label

    def _encode_label(self, label: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """将 JSON 标注转为训练用整数 / 多热张量."""
        targets: Dict[str, torch.Tensor] = {}

        # slot_type → int
        targets["slot_type"] = torch.tensor(
            SLOT_TYPE_TO_IDX.get(label.get("slot_type", "UNKNOWN"), SLOT_TYPE_TO_IDX["UNKNOWN"]),
            dtype=torch.long,
        )

        # maneuver → int
        targets["maneuver"] = torch.tensor(
            MANEUVER_TO_IDX.get(label.get("maneuver", "UNKNOWN"), MANEUVER_TO_IDX["UNKNOWN"]),
            dtype=torch.long,
        )

        # special_scene → multi-hot [14]
        ss = label.get("special_scene", {})
        p0_list = ss.get("P0", [])
        p1_list = ss.get("P1", [])
        ss_vec = torch.zeros(NUM_SPECIAL_SCENE, dtype=torch.float)
        for tag in p0_list + p1_list:
            idx = SPECIAL_SCENE_TO_IDX.get(tag)
            if idx is not None:
                ss_vec[idx] = 1.0
        targets["special_scene"] = ss_vec

        # obstacles → 9 个 int
        obs = label.get("obstacles", {})
        for i, pos in enumerate(POSITION_KEYS):
            val = obs.get(pos, "UNKNOWN")
            if isinstance(val, list):
                val = val[0] if val else "UNKNOWN"
            targets[f"obs_{i}"] = torch.tensor(
                OBSTACLE_TO_IDX.get(val, OBSTACLE_TO_IDX["UNKNOWN"]),
                dtype=torch.long,
            )

        # marking
        mk = label.get("marking", {})
        targets["line_color"] = torch.tensor(
            LINE_COLOR_TO_IDX.get(mk.get("line_color", "UNKNOWN"), LINE_COLOR_TO_IDX["UNKNOWN"]),
            dtype=torch.long,
        )
        targets["line_vis"] = torch.tensor(
            LINE_VIS_TO_IDX.get(mk.get("line_visibility", "UNKNOWN"), LINE_VIS_TO_IDX["UNKNOWN"]),
            dtype=torch.long,
        )
        targets["line_style"] = torch.tensor(
            LINE_STYLE_TO_IDX.get(mk.get("line_style", "UNKNOWN"), LINE_STYLE_TO_IDX["UNKNOWN"]),
            dtype=torch.long,
        )

        return targets

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path, lbl_path = self.samples[idx]

        # 加载图片
        img = Image.open(img_path).convert("RGB")
        label = self._load_label(lbl_path)

        # 数据增强：水平翻转
        do_mirror = self.augment and random.random() < self.mirror_prob
        if do_mirror:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = self._mirror_label(label)

        # 图像 transform
        if self.augment:
            img_tensor = self.train_transform(img)
        else:
            img_tensor = self.val_transform(img)

        # 编码标签
        targets = self._encode_label(label)

        return img_tensor, targets


def collate_fn(batch):
    """自定义 collate: 把 list of (img, dict) 合并为 (batch_img, batch_dict)."""
    imgs = torch.stack([b[0] for b in batch])
    target_keys = batch[0][1].keys()
    targets = {}
    for k in target_keys:
        targets[k] = torch.stack([b[1][k] for b in batch])
    return imgs, targets


if __name__ == "__main__":
    # 快速测试（需要实际数据）
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    try:
        ds = FSTDataset(root, augment=True)
        print(f"Loaded {len(ds)} samples from {root}")
        img, tgt = ds[0]
        print(f"Image shape: {img.shape}")
        for k, v in tgt.items():
            print(f"  {k}: {v}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Create dataset/images/ and dataset/labels/ with matching files first.")
