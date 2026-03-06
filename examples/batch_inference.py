"""
批量推理脚本 - 对整个目录的图片进行推理.

用法:
    python examples/batch_inference.py \
        --images ./test_images \
        --model checkpoints/best_model.pth \
        --output ./results
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import json
from tqdm import tqdm

from fst.hybrid_inference import HybridInferenceEngine
from fst.models import build_fst_text


def main():
    parser = argparse.ArgumentParser(description="批量推理")
    parser.add_argument("--images", required=True, help="图片目录")
    parser.add_argument("--model", required=True, help="CNN 模型路径")
    parser.add_argument("--yolo", default="n", help="YOLO 模型大小")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--output", default="batch_results", help="输出目录")
    parser.add_argument("--visualize", action="store_true", help="保存可视化")
    parser.add_argument("--extensions", default="jpg,jpeg,png", help="图片扩展名")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

    # 初始化引擎
    print("初始化推理引擎...")
    engine = HybridInferenceEngine(
        cnn_model_path=args.model,
        yolo_model_size=args.yolo,
        device=args.device,
    )

    # 收集图片
    image_dir = Path(args.images)
    extensions = args.extensions.split(",")
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*.{ext}"))
        image_files.extend(image_dir.glob(f"*.{ext.upper()}"))

    print(f"找到 {len(image_files)} 张图片")

    # 批量推理
    results = []
    for img_path in tqdm(image_files, desc="推理中"):
        try:
            # 读取图片
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 无法读取 {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_id = img_path.stem

            # 推理
            result = engine.infer(image_rgb, image_id=image_id)
            fst_text = build_fst_text(result)

            # 保存 JSON
            json_path = output_dir / f"{image_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

            # 可视化
            if args.visualize:
                detections = engine.yolo_detector.detect(image_rgb)
                vis = engine.visualize(image_rgb, result, detections)
                vis_path = vis_dir / f"{image_id}_vis.jpg"
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            results.append({
                "image_id": image_id,
                "fst_text": fst_text,
                "slot_type": result.slot.slot_type.value,
                "maneuver": result.maneuver,
            })

        except Exception as e:
            print(f"错误: {img_path} - {e}")
            continue

    # 保存汇总
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n完成! 处理了 {len(results)} 张图片")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
