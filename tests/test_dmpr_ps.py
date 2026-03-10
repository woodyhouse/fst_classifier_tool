#!/usr/bin/env python3
"""
测试 DMPR-PS 停车位检测模型
独立运行，不依赖 fst 工具
"""
import _bootstrap  # noqa: F401
import os
import sys
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from fst.paths import EXTERNAL_DIR

def setup_dmpr_ps():
    """下载并设置 DMPR-PS 模型"""
    print("=" * 60)
    print("DMPR-PS 停车位检测模型测试")
    print("=" * 60)

    dmpr_dir = EXTERNAL_DIR / "DMPR-PS"

    if not dmpr_dir.exists():
        print("\n[1/3] 克隆 DMPR-PS 仓库...")
        os.system(f"git clone https://github.com/Teoge/DMPR-PS.git {dmpr_dir}")
    else:
        print("\n[1/3] DMPR-PS 仓库已存在")

    # 添加到 Python 路径
    sys.path.insert(0, str(dmpr_dir))

    print("\n[2/3] 检查依赖...")
    try:
        import torch
        import torchvision
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  ✗ 缺少依赖: {e}")
        print("\n请安装: pip install torch torchvision")
        sys.exit(1)

    print("\n[3/3] 下载预训练模型...")
    model_path = dmpr_dir / "weights" / "dmpr_ps.pth"
    if not model_path.exists():
        print("  模型文件不存在，需要手动下载")
        print("  下载地址: https://github.com/Teoge/DMPR-PS/releases")
        print(f"  保存到: {model_path}")
        return None
    else:
        print(f"  ✓ 模型已存在: {model_path}")
        return model_path

def load_dmpr_model(model_path):
    """加载 DMPR-PS 模型"""
    try:
        from DMPR_PS.config import Config
        from DMPR_PS.model import DirectionalPointDetector

        # 加载配置
        config = Config()

        # 创建模型
        model = DirectionalPointDetector(config)

        # 加载权重
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 模型加载成功")
        else:
            print("⚠ 使用未训练的模型（仅测试架构）")

        model.eval()

        # 移到 GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✓ 模型运行在: {device}")

        return model, device

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None, None

def detect_parking_slots(model, device, image_path):
    """检测停车位"""
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"✗ 无法读取图像: {image_path}")
        return None

    h, w = img.shape[:2]
    print(f"\n图像尺寸: {w}x{h}")

    # 预处理
    img_resized = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 推理
    print("正在检测...")
    with torch.no_grad():
        outputs = model(img_tensor)

    # 解析结果
    # DMPR-PS 输出: 4个角点 + 置信度 + 方向
    if isinstance(outputs, dict):
        keypoints = outputs.get('keypoints', None)
        confidence = outputs.get('confidence', None)
        direction = outputs.get('direction', None)
    else:
        keypoints = outputs
        confidence = None
        direction = None

    return {
        'image': img,
        'keypoints': keypoints,
        'confidence': confidence,
        'direction': direction,
        'size': (w, h)
    }

def visualize_results(result, output_path):
    """可视化检测结果"""
    img = result['image'].copy()
    keypoints = result['keypoints']

    if keypoints is None:
        print("⚠ 未检测到停车位")
        return

    # 转换回原始尺寸
    h, w = result['size']
    keypoints = keypoints.cpu().numpy()
    keypoints[:, 0] *= w / 512
    keypoints[:, 1] *= h / 512

    # 绘制角点
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(img, (int(x), int(y)), 10, colors[i], -1)
        cv2.putText(img, f"P{i+1}", (int(x)+15, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

    # 绘制车位框
    pts = keypoints.astype(np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 3)

    # 绘制入口方向
    if result['direction'] is not None:
        direction = result['direction'].cpu().numpy()
        center = keypoints.mean(axis=0)
        arrow_end = center + direction * 100
        cv2.arrowedLine(img, tuple(center.astype(int)),
                       tuple(arrow_end.astype(int)), (255, 0, 255), 3)

    # 保存结果
    cv2.imwrite(str(output_path), img)
    print(f"✓ 结果已保存: {output_path}")

    # 显示结果
    cv2.imshow("Parking Slot Detection", cv2.resize(img, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='测试 DMPR-PS 停车位检测')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--output', default='dmpr_result.jpg', help='输出图片路径')
    args = parser.parse_args()

    # 设置模型
    model_path = setup_dmpr_ps()

    # 加载模型
    model, device = load_dmpr_model(model_path)
    if model is None:
        print("\n✗ 模型加载失败，无法继续")
        return

    # 检测
    result = detect_parking_slots(model, device, args.image)
    if result is None:
        return

    # 可视化
    visualize_results(result, args.output)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
