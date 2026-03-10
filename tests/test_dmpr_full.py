#!/usr/bin/env python3
"""
完整版 DMPR-PS 停车位检测测试
使用真实的预训练模型
"""
import _bootstrap  # noqa: F401
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from fst.paths import DATASET_IMAGES_DIR, EXTERNAL_DIR, TEST_RESULTS_DIR

# 添加 DMPR-PS 到路径
DMPR_DIR = EXTERNAL_DIR / "DMPR-PS"
sys.path.insert(0, str(DMPR_DIR))

try:
    import config
    from model.detector import DirectionalPointDetector
    from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
    DMPR_AVAILABLE = True
except ImportError as e:
    print(f"⚠ DMPR-PS 导入失败: {e}")
    DMPR_AVAILABLE = False


class DMPRDetector:
    """DMPR-PS 检测器封装"""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[OK] Using device: {self.device}")

        # 创建模型
        self.model = DirectionalPointDetector(
            input_channel_size=3,
            depth_factor=32,
            output_channel_size=6
        )

        # 加载权重
        if model_path.exists():
            print(f"[OK] Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 处理不同的 checkpoint 格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)

            print("[OK] Model loaded successfully")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        """预处理图像"""
        if image.shape[0] != 512 or image.shape[1] != 512:
            image = cv2.resize(image, (512, 512))

        # 转换为 tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

    def detect_marking_points(self, image, thresh=0.5):
        """检测车位标记点"""
        h, w = image.shape[:2]

        # 预处理
        input_tensor = self.preprocess_image(image).to(self.device)

        # 推理
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # 解析预测结果
        pred_points = get_predicted_points(prediction[0], thresh)

        if not pred_points:
            return None, None, None

        # 提取标记点
        marking_points = [point for _, point in pred_points]

        # 推理车位槽
        slots = self.inference_slots(marking_points)

        # 转换回原始尺寸
        keypoints = []
        for _, point in pred_points:
            x = point.x * w
            y = point.y * h
            keypoints.append([x, y])

        return np.array(keypoints), slots, pred_points

    def inference_slots(self, marking_points):
        """从标记点推理车位槽"""
        num_detected = len(marking_points)
        slots = []

        for i in range(num_detected - 1):
            for j in range(i + 1, num_detected):
                point_i = marking_points[i]
                point_j = marking_points[j]

                # 长度过滤
                distance = calc_point_squre_dist(point_i, point_j)
                if not (config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST
                        or config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST):
                    continue

                # 穿透过滤
                if pass_through_third_point(marking_points, i, j):
                    continue

                result = pair_marking_points(point_i, point_j)
                if result == 1:
                    slots.append((i, j))
                elif result == -1:
                    slots.append((j, i))

        return slots


def visualize_detection(image, keypoints, slots, pred_points, output_path):
    """可视化检测结果"""
    vis_img = image.copy()
    h, w = image.shape[:2]

    if keypoints is None or len(keypoints) == 0:
        cv2.putText(vis_img, "No parking slot detected", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(str(output_path), vis_img)
        print(f"  [OK] Saved result: {output_path.name}")
        return

    # 绘制标记点
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255)]

    for idx, (x, y) in enumerate(keypoints):
        color = colors[idx % len(colors)]
        cv2.circle(vis_img, (int(x), int(y)), 8, color, -1)
        cv2.putText(vis_img, f"P{idx}", (int(x)+10, int(y)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 绘制车位槽
    if slots and pred_points:
        marking_points = [point for _, point in pred_points]

        for slot_idx, (i, j) in enumerate(slots):
            point_a = marking_points[i]
            point_b = marking_points[j]

            p0_x = w * point_a.x
            p0_y = h * point_a.y
            p1_x = w * point_b.x
            p1_y = h * point_b.y

            # 计算车位框的四个角点
            vec = np.array([p1_x - p0_x, p1_y - p0_y])
            vec = vec / np.linalg.norm(vec)

            distance = calc_point_squre_dist(point_a, point_b)
            if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
                separating_length = config.LONG_SEPARATOR_LENGTH
            else:
                separating_length = config.SHORT_SEPARATOR_LENGTH

            p2_x = p0_x + h * separating_length * vec[1]
            p2_y = p0_y - w * separating_length * vec[0]
            p3_x = p1_x + h * separating_length * vec[1]
            p3_y = p1_y - w * separating_length * vec[0]

            # 绘制车位框
            pts = np.array([
                [int(p0_x), int(p0_y)],
                [int(p1_x), int(p1_y)],
                [int(p3_x), int(p3_y)],
                [int(p2_x), int(p2_y)]
            ], dtype=np.int32)

            cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3)

            # 标注车位编号
            center_x = int((p0_x + p1_x + p2_x + p3_x) / 4)
            center_y = int((p0_y + p1_y + p2_y + p3_y) / 4)
            cv2.putText(vis_img, f"Slot {slot_idx+1}", (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 添加统计信息
    info_text = f"Points: {len(keypoints)} | Slots: {len(slots) if slots else 0}"
    cv2.putText(vis_img, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), vis_img)
    print(f"  [OK] Saved result: {output_path.name}")


def test_single_image(detector, image_path, output_dir):
    """测试单张图片"""
    print(f"\n处理: {image_path.name}")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [ERROR] Cannot read image")
        return None

    # 检测
    try:
        keypoints, slots, pred_points = detector.detect_marking_points(image, thresh=0.5)

        if keypoints is not None:
            print(f"  [OK] Detected {len(keypoints)} points, {len(slots) if slots else 0} slots")
            detected = True
        else:
            print(f"  [FAIL] No parking slot detected")
            detected = False

        # 可视化
        output_path = output_dir / f"{image_path.stem}_result.jpg"
        visualize_detection(image, keypoints, slots, pred_points, output_path)

        return {
            'image': image_path.name,
            'detected': detected,
            'num_points': len(keypoints) if keypoints is not None else 0,
            'num_slots': len(slots) if slots else 0,
            'keypoints': keypoints.tolist() if keypoints is not None else []
        }

    except Exception as e:
        print(f"  [ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'image': image_path.name,
            'detected': False,
            'error': str(e)
        }


def test_batch(detector, images_dir, output_dir):
    """批量测试"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']:
        image_files.extend(images_dir.glob(ext))

    if not image_files:
        print(f"[ERROR] No images found: {images_dir}")
        return

    print(f"\nFound {len(image_files)} images")
    print("=" * 60)

    # 测试每张图片
    results = []
    for img_path in image_files:
        result = test_single_image(detector, img_path, output_dir)
        if result:
            results.append(result)

    # 生成报告
    generate_report(results, output_dir)


def generate_report(results, output_dir):
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("测试报告")
    print("=" * 60)

    total = len(results)
    success = sum(1 for r in results if r['detected'])
    success_rate = success / total if total > 0 else 0

    print(f"总图片数: {total}")
    print(f"检测成功: {success}")
    print(f"检测失败: {total - success}")
    print(f"成功率: {success_rate:.1%}")

    # 保存 JSON 报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'total': total,
        'success': success,
        'failed': total - success,
        'success_rate': success_rate,
        'results': results
    }

    report_path = output_dir / 'test_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Report saved: {report_path}")

    # 生成 HTML 报告
    generate_html_report(report, output_dir)


def generate_html_report(report, output_dir):
    """生成 HTML 可视化报告"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DMPR-PS 测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2196F3; color: white; padding: 20px; border-radius: 8px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 36px; font-weight: bold; color: #2196F3; }}
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
        <h1>🚗 DMPR-PS 停车位检测测试报告</h1>
        <p>生成时间: {report['timestamp']}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{report['total']}</div>
            <div>总图片数</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #4CAF50;">{report['success']}</div>
            <div>检测成功</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #f44336;">{report['failed']}</div>
            <div>检测失败</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #FF9800;">{report['success_rate']:.1%}</div>
            <div>成功率</div>
        </div>
    </div>

    <h2>检测结果</h2>
    <div class="gallery">
"""

    for result in report['results']:
        status_class = 'success' if result['detected'] else 'failed'
        status_text = '✓ 检测成功' if result['detected'] else '✗ 检测失败'
        img_name = result['image'].replace('.jpg', '_result.jpg').replace('.jpeg', '_result.jpg').replace('.png', '_result.jpg')

        html += f"""
        <div class="image-card {status_class}">
            <img src="{img_name}" alt="{result['image']}">
            <div class="info">
                <strong>{result['image']}</strong><br>
                {status_text}<br>
                标记点: {result.get('num_points', 0)} | 车位: {result.get('num_slots', 0)}
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
    print("DMPR-PS Full Test")
    print("=" * 60)

    # 检查 DMPR-PS 是否可用
    if not DMPR_AVAILABLE:
        print("\n[ERROR] DMPR-PS not available")
        return

    # 检查模型文件
    model_path = DMPR_DIR / "weights" / "dmpr_ps.pth"
    if not model_path.exists():
        print(f"\n[ERROR] Model file not found: {model_path}")
        print("Please put the model file in this location")
        return

    print(f"\n[OK] Found model file: {model_path}")
    print(f"[OK] Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 加载模型
    try:
        detector = DMPRDetector(model_path)
    except Exception as e:
        print(f"\n[ERROR] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试图片
    images_dir = DATASET_IMAGES_DIR
    output_dir = TEST_RESULTS_DIR

    if not images_dir.exists():
        print(f"\n[ERROR] Images directory not found: {images_dir}")
        return

    # 批量测试
    test_batch(detector, images_dir, output_dir)

    print("\n" + "=" * 60)
    print("Test completed!")
    print(f"View results: {output_dir / 'report.html'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
