#!/usr/bin/env python3
"""
测试 PS2.0 停车位检测模型（基于分割）
独立运行，不依赖 fst 工具
"""
import os
import sys
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path

def setup_ps2():
    """下载并设置 PS2.0 模型"""
    print("=" * 60)
    print("PS2.0 停车位检测模型测试（分割方法）")
    print("=" * 60)

    ps2_dir = Path("./external/ps2.0")

    if not ps2_dir.exists():
        print("\n[1/3] 克隆 PS2.0 仓库...")
        os.system("git clone https://github.com/weili1457355863/ps2.0.git ./external/ps2.0")
    else:
        print("\n[1/3] PS2.0 仓库已存在")

    sys.path.insert(0, str(ps2_dir))

    print("\n[2/3] 检查依赖...")
    try:
        import torch
        import torchvision
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ 缺少依赖: {e}")
        sys.exit(1)

    print("\n[3/3] 检查模型...")
    model_path = ps2_dir / "weights" / "ps2.pth"
    if not model_path.exists():
        print("  ⚠ 模型文件不存在")
        print("  下载地址: https://github.com/weili1457355863/ps2.0/releases")
        return None
    else:
        print(f"  ✓ 模型已存在: {model_path}")
        return model_path

def load_ps2_model(model_path):
    """加载 PS2.0 模型"""
    try:
        # PS2.0 使用分割网络
        from ps2.model import PS2Net

        model = PS2Net()

        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            print("✓ 模型加载成功")
        else:
            print("⚠ 使用未训练的模型")

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✓ 模型运行在: {device}")

        return model, device

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None, None

def detect_with_segmentation(model, device, image_path):
    """使用分割方法检测停车位"""
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
        seg_map = model(img_tensor)

    # 分割图: [B, C, H, W]
    # C=3: 背景、车位线、车位区域
    seg_map = seg_map.squeeze(0).cpu().numpy()
    seg_map = np.argmax(seg_map, axis=0)  # [H, W]

    return {
        'image': img,
        'segmentation': seg_map,
        'size': (w, h)
    }

def extract_slots_from_segmentation(seg_map):
    """从分割图提取车位框"""
    # 提取车位线
    line_mask = (seg_map == 1).astype(np.uint8) * 255

    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

    # 检测轮廓
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    slots = []
    for contour in contours:
        # 拟合最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 过滤太小的区域
        area = cv2.contourArea(box)
        if area > 1000:
            slots.append(box)

    return slots

def visualize_segmentation(result, output_path):
    """可视化分割结果"""
    img = result['image'].copy()
    seg_map = result['segmentation']

    # 调整分割图到原始尺寸
    h, w = result['size']
    seg_map_resized = cv2.resize(seg_map.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST)

    # 提取车位框
    slots = extract_slots_from_segmentation(seg_map_resized)

    if len(slots) == 0:
        print("⚠ 未检测到停车位")
    else:
        print(f"✓ 检测到 {len(slots)} 个停车位")

    # 绘制分割图（半透明叠加）
    seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
    seg_colored[seg_map_resized == 1] = [0, 255, 0]  # 车位线 - 绿色
    seg_colored[seg_map_resized == 2] = [255, 0, 0]  # 车位区域 - 蓝色

    img_overlay = cv2.addWeighted(img, 0.7, seg_colored, 0.3, 0)

    # 绘制车位框
    for slot in slots:
        cv2.polylines(img_overlay, [slot], True, (0, 255, 255), 3)

    # 保存结果
    cv2.imwrite(str(output_path), img_overlay)
    print(f"✓ 结果已保存: {output_path}")

    # 显示
    cv2.imshow("PS2.0 Segmentation", cv2.resize(img_overlay, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='测试 PS2.0 停车位检测')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--output', default='ps2_result.jpg', help='输出图片路径')
    args = parser.parse_args()

    # 设置模型
    model_path = setup_ps2()

    # 加载模型
    model, device = load_ps2_model(model_path)
    if model is None:
        print("\n✗ 模型加载失败")
        return

    # 检测
    result = detect_with_segmentation(model, device, args.image)
    if result is None:
        return

    # 可视化
    visualize_segmentation(result, args.output)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
