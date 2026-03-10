#!/usr/bin/env python3
"""
测试简化版推理引擎
"""
import _bootstrap  # noqa: F401
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from fst.paths import CHECKPOINTS_DIR, DATASET_IMAGES_DIR, TEST_RESULTS_SIMPLIFIED_DIR
from fst.simplified_inference import SimplifiedHybridInference


def test_single_image(engine, image_path, output_dir):
    """测试单张图片"""
    print(f"\n处理: {image_path.name}")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [ERROR] Cannot read image")
        return None

    try:
        # 推理
        fst_output = engine.predict(image)

        # 可视化
        vis_img = engine.visualize(image, fst_output)
        output_path = output_dir / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), vis_img)

        # 保存 JSON
        json_path = output_dir / f"{image_path.stem}_fst.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(fst_output.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"  [OK] Slot: {fst_output.slot.slot_type.value}")
        print(f"  [OK] Maneuver: {fst_output.maneuver if isinstance(fst_output.maneuver, str) else fst_output.maneuver.value}")
        print(f"  [OK] Obstacles: {sum(1 for v in fst_output.obstacles.pos_map.values() if v != ['EMPTY'])} detected")
        print(f"  [OK] Saved: {output_path.name}")

        return {
            'image': image_path.name,
            'success': True,
            'slot_type': fst_output.slot.slot_type.value,
            'maneuver': fst_output.maneuver if isinstance(fst_output.maneuver, str) else fst_output.maneuver.value,
            'num_obstacles': sum(1 for v in fst_output.obstacles.pos_map.values() if v != ['EMPTY'])
        }

    except Exception as e:
        print(f"  [ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'image': image_path.name,
            'success': False,
            'error': str(e)
        }


def test_batch(engine, images_dir, output_dir, max_images=20):
    """批量测试"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(images_dir.glob(ext))

    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return

    print(f"\nFound {len(image_files)} images, testing first {max_images}")
    print("=" * 60)

    # 测试
    results = []
    for img_path in image_files[:max_images]:
        result = test_single_image(engine, img_path, output_dir)
        if result:
            results.append(result)

    # 生成报告
    generate_report(results, output_dir)


def generate_report(results, output_dir):
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("Test Report")
    print("=" * 60)

    total = len(results)
    success = sum(1 for r in results if r['success'])
    success_rate = success / total if total > 0 else 0

    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"Failed: {total - success}")
    print(f"Success Rate: {success_rate:.1%}")

    # 统计车位类型分布
    slot_types = {}
    for r in results:
        if r['success']:
            slot_type = r['slot_type']
            slot_types[slot_type] = slot_types.get(slot_type, 0) + 1

    print("\nSlot Type Distribution:")
    for slot_type, count in slot_types.items():
        print(f"  {slot_type}: {count}")

    # 保存 JSON 报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'total': total,
        'success': success,
        'failed': total - success,
        'success_rate': success_rate,
        'slot_types': slot_types,
        'results': results
    }

    report_path = output_dir / 'test_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Report saved: {report_path}")

    # 生成 HTML 报告
    generate_html_report(report, output_dir)


def generate_html_report(report, output_dir):
    """生成 HTML 报告"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Simplified Inference Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #4CAF50; color: white; padding: 20px; border-radius: 8px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 36px; font-weight: bold; color: #4CAF50; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .image-card {{ background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .image-card img {{ width: 100%; border-radius: 4px; }}
        .success {{ border-left: 4px solid #4CAF50; }}
        .failed {{ border-left: 4px solid #f44336; }}
        .info {{ margin-top: 10px; font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 Simplified Inference Test Report</h1>
        <p>Generated: {report['timestamp']}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{report['total']}</div>
            <div>Total Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #4CAF50;">{report['success']}</div>
            <div>Success</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #f44336;">{report['failed']}</div>
            <div>Failed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #FF9800;">{report['success_rate']:.1%}</div>
            <div>Success Rate</div>
        </div>
    </div>

    <h2>Results</h2>
    <div class="gallery">
"""

    for result in report['results']:
        status_class = 'success' if result['success'] else 'failed'
        status_text = '✓ Success' if result['success'] else '✗ Failed'
        img_name = result['image'].replace('.jpg', '_result.jpg').replace('.jpeg', '_result.jpg').replace('.png', '_result.jpg')

        info = ""
        if result['success']:
            info = f"Slot: {result['slot_type']}<br>Maneuver: {result['maneuver']}<br>Obstacles: {result['num_obstacles']}"
        else:
            info = f"Error: {result.get('error', 'Unknown')}"

        html += f"""
        <div class="image-card {status_class}">
            <img src="{img_name}" alt="{result['image']}">
            <div class="info">
                <strong>{result['image']}</strong><br>
                {status_text}<br>
                {info}
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    html_path = output_dir / 'report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[OK] HTML report saved: {html_path}")


def main():
    print("=" * 60)
    print("Simplified Inference Engine Test")
    print("=" * 60)

    # 检查模型文件
    model_path = CHECKPOINTS_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Please train the model first: python -m fst.train")
        return

    print(f"\n[OK] Loading model: {model_path}")

    # 初始化引擎
    try:
        engine = SimplifiedHybridInference(
            cnn_model_path=str(model_path),
            yolo_model_size="n",
            device="cuda",  # 改为 "cpu" 如果没有 GPU
        )
        print("[OK] Engine initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试
    images_dir = DATASET_IMAGES_DIR
    output_dir = TEST_RESULTS_SIMPLIFIED_DIR

    if not images_dir.exists():
        print(f"\n[ERROR] Images directory not found: {images_dir}")
        return

    # 批量测试（前 20 张）
    test_batch(engine, images_dir, output_dir, max_images=20)

    print("\n" + "=" * 60)
    print("Test completed!")
    print(f"View results: {output_dir / 'report.html'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
