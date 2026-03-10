#!/usr/bin/env python3
"""
DMPR-PS + 透视变换测试
先将地面视角转换为鸟瞰图，再检测
"""
import _bootstrap  # noqa: F401
import cv2
import numpy as np
from pathlib import Path
import sys
from fst.paths import DATASET_IMAGES_DIR, EXTERNAL_DIR, TEST_RESULTS_BEV_DIR

# 添加 DMPR-PS 到路径
DMPR_DIR = EXTERNAL_DIR / "DMPR-PS"
sys.path.insert(0, str(DMPR_DIR))

from test_dmpr_full import DMPRDetector, visualize_detection, generate_report


def estimate_perspective_transform(image):
    """估计透视变换矩阵（简化版）"""
    h, w = image.shape[:2]

    # 假设相机参数（需要根据实际情况调整）
    # 源点：地面视角的梯形区域
    src_points = np.float32([
        [w * 0.3, h * 0.5],   # 左上（远处）
        [w * 0.7, h * 0.5],   # 右上（远处）
        [0, h],               # 左下（近处）
        [w, h]                # 右下（近处）
    ])

    # 目标点：鸟瞰图的矩形
    dst_points = np.float32([
        [w * 0.2, 0],
        [w * 0.8, 0],
        [w * 0.2, h],
        [w * 0.8, h]
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M


def ground_to_bev(image):
    """地面视角 → 鸟瞰图"""
    h, w = image.shape[:2]

    # 估计透视变换
    M = estimate_perspective_transform(image)

    # 应用变换
    bev_image = cv2.warpPerspective(image, M, (w, h))

    return bev_image, M


def bev_to_ground(keypoints, M, image_shape):
    """将 BEV 坐标转换回地面视角坐标"""
    if keypoints is None or len(keypoints) == 0:
        return None

    # 逆变换
    M_inv = np.linalg.inv(M)

    # 转换关键点
    keypoints_homogeneous = np.hstack([keypoints, np.ones((len(keypoints), 1))])
    keypoints_ground = (M_inv @ keypoints_homogeneous.T).T
    keypoints_ground = keypoints_ground[:, :2] / keypoints_ground[:, 2:]

    return keypoints_ground


def test_with_bev_transform(detector, image_path, output_dir):
    """使用 BEV 变换测试"""
    print(f"\n处理: {image_path.name}")

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [ERROR] Cannot read image")
        return None

    # 转换到 BEV
    print("  [1/3] Converting to BEV...")
    bev_image, M = ground_to_bev(image)

    # 保存 BEV 图像（用于调试）
    bev_path = output_dir / f"{image_path.stem}_bev.jpg"
    cv2.imwrite(str(bev_path), bev_image)

    # 在 BEV 上检测
    print("  [2/3] Detecting on BEV...")
    try:
        keypoints_bev, slots, pred_points = detector.detect_marking_points(bev_image, thresh=0.3)

        if keypoints_bev is not None:
            # 转换回地面视角
            print("  [3/3] Converting back to ground view...")
            keypoints_ground = bev_to_ground(keypoints_bev, M, image.shape)

            print(f"  [OK] Detected {len(keypoints_ground)} points, {len(slots) if slots else 0} slots")
            detected = True
        else:
            print(f"  [FAIL] No parking slot detected")
            keypoints_ground = None
            detected = False

        # 可视化（在原始图像上）
        output_path = output_dir / f"{image_path.stem}_result.jpg"
        visualize_detection(image, keypoints_ground, slots, pred_points, output_path)

        return {
            'image': image_path.name,
            'detected': detected,
            'num_points': len(keypoints_ground) if keypoints_ground is not None else 0,
            'num_slots': len(slots) if slots else 0,
            'keypoints': keypoints_ground.tolist() if keypoints_ground is not None else []
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


def main():
    print("=" * 60)
    print("DMPR-PS + BEV Transform Test")
    print("=" * 60)

    # 加载模型
    model_path = DMPR_DIR / "weights" / "dmpr_ps.pth"
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        return

    print(f"\n[OK] Loading model: {model_path}")
    detector = DMPRDetector(model_path)

    # 测试图片
    images_dir = DATASET_IMAGES_DIR
    output_dir = TEST_RESULTS_BEV_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(images_dir.glob(ext))

    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return

    # 只测试前 20 张（快速验证）
    print(f"\nTesting first 20 images (out of {len(image_files)} total)")
    print("=" * 60)

    results = []
    for img_path in image_files[:20]:
        result = test_with_bev_transform(detector, img_path, output_dir)
        if result:
            results.append(result)

    # 生成报告
    generate_report(results, output_dir)

    print("\n" + "=" * 60)
    print("Test completed!")
    print(f"View results: {output_dir / 'report.html'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
