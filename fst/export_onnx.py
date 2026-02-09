"""
导出训练好的模型为 ONNX 格式，用于分发部署.

用法:
    python -m fst.export_onnx --checkpoint checkpoints/best_model.pth --output fst_classifier.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnx

from fst.network import FSTClassifier


class FSTClassifierFlat(torch.nn.Module):
    """
    包装 FSTClassifier，将 dict 输出展平为 tuple，以兼容 ONNX 导出.
    """

    def __init__(self, model: FSTClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return (
            out["slot_type"],
            out["maneuver"],
            out["special_scene"],
            out["obs_0"], out["obs_1"], out["obs_2"],
            out["obs_3"], out["obs_4"], out["obs_5"],
            out["obs_6"], out["obs_7"], out["obs_8"],
            out["line_color"],
            out["line_vis"],
            out["line_style"],
        )


OUTPUT_NAMES = [
    "slot_type", "maneuver", "special_scene",
    "obs_0", "obs_1", "obs_2", "obs_3", "obs_4",
    "obs_5", "obs_6", "obs_7", "obs_8",
    "line_color", "line_vis", "line_style",
]


def export_onnx(
    checkpoint_path: str,
    output_path: str = "fst_classifier.onnx",
    backbone: str = "efficientnet_b0",
    img_size: int = 384,
    opset: int = 17,
):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = FSTClassifier(backbone_name=backbone, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    flat_model = FSTClassifierFlat(model)
    flat_model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    print(f"Exporting to ONNX (opset {opset})...")
    torch.onnx.export(
        flat_model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={"image": {0: "batch_size"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    # 验证导出的模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model saved: {output_path} ({file_size:.1f} MB)")
    print(f"Input: image [batch, 3, {img_size}, {img_size}]")
    print(f"Outputs: {', '.join(OUTPUT_NAMES)}")


def main():
    parser = argparse.ArgumentParser(description="Export FST model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default="fst_classifier.onnx", help="Output ONNX path")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        backbone=args.backbone,
        img_size=args.img_size,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
