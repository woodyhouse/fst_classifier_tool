"""
混合推理示例 - 演示如何使用新的 CNN + YOLO + 几何分析架构.

用法:
    python examples/hybrid_inference_demo.py --image test.jpg --model checkpoints/best_model.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import json

from fst.hybrid_inference import HybridInferenceEngine
from fst.models import build_fst_text


def main():
    parser = argparse.ArgumentParser(description="混合推理演示")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--model", required=True, help="CNN 模型路径 (.pth 或 .onnx)")
    parser.add_argument("--yolo", default="n", choices=["n", "s", "m", "l", "x"], help="YOLO 模型大小")
    parser.add_argument("--device", default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--visualize", action="store_true", help="保存可视化结果")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化推理引擎
    print("初始化推理引擎...")
    print(f"  CNN 模型: {args.model}")
    print(f"  YOLO 模型: yolov8{args.yolo}")
    print(f"  设备: {args.device}")

    engine = HybridInferenceEngine(
        cnn_model_path=args.model,
        yolo_model_size=args.yolo,
        device=args.device,
    )

    # 读取图片
    print(f"\n读取图片: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误: 无法读取图片 {args.image}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_id = Path(args.image).stem

    # 推理
    print("\n执行推理...")
    result = engine.infer(image_rgb, image_id=image_id)

    # 生成 FST 文本
    fst_text = build_fst_text(result)

    # 打印结果
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)
    print(f"\nFST 文本: {fst_text}")
    print(f"\n车位类型: {result.slot.slot_type.value}")
    print(f"泊车动作: {result.maneuver}")
    print(f"车位线颜色: {result.slot.marking.line_color.value}")
    print(f"车位线可见度: {result.slot.marking.line_visibility.value}")
    print(f"车位线样式: {result.slot.marking.line_style.value}")

    if result.special_scene.P0:
        print(f"P0 场景: {', '.join(result.special_scene.P0)}")
    if result.special_scene.P1:
        print(f"P1 场景: {', '.join(result.special_scene.P1)}")

    print("\n9 宫格障碍物:")
    for pos in ["1", "2", "3", "4", "5", "6", "7", "P_LEFT", "P_RIGHT"]:
        obstacles = result.obstacles.pos_map.get(pos, ["UNKNOWN"])
        print(f"  位置 {pos:7s}: {', '.join(obstacles)}")

    # 保存 JSON
    json_path = output_dir / f"{image_id}_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\nJSON 结果已保存: {json_path}")

    # 可视化
    if args.visualize:
        print("\n生成可视化...")
        detections = engine.yolo_detector.detect(image_rgb)
        print(f"  YOLO 检测到 {len(detections)} 个物体:")
        for det in detections:
            fst_type = engine.yolo_detector.map_to_fst_type(det.class_name)
            print(f"    - {det.class_name} ({det.confidence:.2f}) → {fst_type}")

        vis = engine.visualize(image_rgb, result, detections)
        vis_path = output_dir / f"{image_id}_visualization.jpg"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  可视化已保存: {vis_path}")

    print("\n完成!")


if __name__ == "__main__":
    main()
