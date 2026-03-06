"""
深度学习版本的车位标记点检测器 - 基于 DMPR-PS

参考: https://github.com/Teoge/DMPR-PS
使用预训练的深度学习模型检测车位标记点，比传统 CV 方法更准确。

使用方法:
1. 下载预训练模型: https://drive.google.com/open?id=1OuyF8bGttA11-CKJ4Mj3dYAl5q4NL5IT
2. 保存为 weights/dmpr_ps.pth
3. 在 ParkingGeometry 中启用: use_dl_marking_points=True
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional
from collections import namedtuple
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.transforms import ToTensor

# 数据结构
MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction', 'shape'])
Slot = namedtuple('Slot', ['x1', 'y1', 'x2', 'y2'])

# 配置常量
INPUT_IMAGE_SIZE = 512
FEATURE_MAP_SIZE = 16
BOUNDARY_THRESH = 0.05
VSLOT_MIN_DIST = 0.044771278151623496
VSLOT_MAX_DIST = 0.1099427457599304
HSLOT_MIN_DIST = 0.15057789144568634
HSLOT_MAX_DIST = 0.44449496544202816
SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = 0.8


class YetAnotherDarknet(nn.Module):
    """Darknet-19 风格的骨干网络."""

    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []

        # 初始卷积
        layers += [nn.Conv2d(input_channel_size, depth_factor, kernel_size=3,
                             stride=1, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(depth_factor)]
        layers += [nn.LeakyReLU(0.1)]

        # 5 个下采样阶段
        for i in range(5):
            # 下采样
            layers += [nn.Conv2d(depth_factor, 2*depth_factor, kernel_size=4,
                                stride=2, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(2*depth_factor)]
            layers += [nn.LeakyReLU(0.1)]
            depth_factor *= 2

            # 检测块 - 阶段3和4有2个块，其他阶段1个块
            if i == 2 or i == 3:
                num_blocks = 2
            else:
                num_blocks = 1

            for _ in range(num_blocks):
                # 压缩
                layers += [nn.Conv2d(depth_factor, depth_factor//2, kernel_size=1,
                                    stride=1, padding=0, bias=False)]
                layers += [nn.BatchNorm2d(depth_factor//2)]
                layers += [nn.LeakyReLU(0.1)]
                # 扩展
                layers += [nn.Conv2d(depth_factor//2, depth_factor, kernel_size=3,
                                    stride=1, padding=1, bias=False)]
                layers += [nn.BatchNorm2d(depth_factor)]
                layers += [nn.LeakyReLU(0.1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DirectionalPointDetector(nn.Module):
    """带方向的标记点检测器."""

    def __init__(self, input_channel_size=3, depth_factor=32, output_channel_size=6):
        super(DirectionalPointDetector, self).__init__()
        self.extract_feature = YetAnotherDarknet(input_channel_size, depth_factor)

        layers = []
        # 两个检测块
        for _ in range(2):
            layers += [nn.Conv2d(32*depth_factor, 16*depth_factor, kernel_size=1,
                                stride=1, padding=0, bias=False)]
            layers += [nn.BatchNorm2d(16*depth_factor)]
            layers += [nn.LeakyReLU(0.1)]
            layers += [nn.Conv2d(16*depth_factor, 32*depth_factor, kernel_size=3,
                                stride=1, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(32*depth_factor)]
            layers += [nn.LeakyReLU(0.1)]

        # 最终预测层
        layers += [nn.Conv2d(32*depth_factor, output_channel_size,
                            kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)

    def forward(self, x):
        features = self.extract_feature(x)
        prediction = self.predict(features)

        # 分割为点预测和角度预测
        point_pred, angle_pred = torch.split(prediction, [4, 2], dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)

        return torch.cat((point_pred, angle_pred), dim=1)


class DLMarkingPointDetector:
    """深度学习版本的标记点检测器."""

    def __init__(
        self,
        weights_path: str = "weights/dmpr_ps.pth",
        conf_threshold: float = 0.5,
        use_cuda: bool = True,
    ):
        """
        Args:
            weights_path: 预训练模型权重路径
            conf_threshold: 置信度阈值
            use_cuda: 是否使用 GPU
        """
        self.conf_threshold = conf_threshold

        # 设置设备
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"  [DMPR-PS] 使用设备: {self.device}")

        # 加载模型
        self.detector = DirectionalPointDetector(
            input_channel_size=3,
            depth_factor=32,
            output_channel_size=6  # 预训练模型使用 6 通道
        ).to(self.device)

        # 加载预训练权重
        try:
            self.detector.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            self.detector.eval()
            print(f"  [DMPR-PS] 成功加载模型: {weights_path}")
        except FileNotFoundError:
            print(f"  [DMPR-PS] ⚠ 未找到模型文件: {weights_path}")
            print(f"  [DMPR-PS] 请从 https://drive.google.com/open?id=1OuyF8bGttA11-CKJ4Mj3dYAl5q4NL5IT 下载")
            raise

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像."""
        if image.shape[0] != INPUT_IMAGE_SIZE or image.shape[1] != INPUT_IMAGE_SIZE:
            image = cv2.resize(image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        return torch.unsqueeze(ToTensor()(image), 0)

    def non_maximum_suppression(self, pred_points: List) -> List:
        """非极大值抑制，去除重复检测."""
        suppressed = [False] * len(pred_points)
        for i in range(len(pred_points) - 1):
            for j in range(i + 1, len(pred_points)):
                i_x, i_y = pred_points[i][1].x, pred_points[i][1].y
                j_x, j_y = pred_points[j][1].x, pred_points[j][1].y
                # 0.0625 = 1/16 (一个网格单元)
                if abs(j_x - i_x) < 0.0625 and abs(j_y - i_y) < 0.0625:
                    idx = i if pred_points[i][0] < pred_points[j][0] else j
                    suppressed[idx] = True

        return [p for i, p in enumerate(pred_points) if not suppressed[i]]

    def get_predicted_points(self, prediction: torch.Tensor) -> List[Tuple[float, MarkingPoint]]:
        """从预测张量提取标记点."""
        prediction = prediction.detach().cpu().numpy()
        predicted_points = []

        for i in range(prediction.shape[1]):  # 高度
            for j in range(prediction.shape[2]):  # 宽度
                confidence = prediction[0, i, j]
                if confidence >= self.conf_threshold:
                    # 计算归一化坐标
                    xval = (j + prediction[2, i, j]) / prediction.shape[2]
                    yval = (i + prediction[3, i, j]) / prediction.shape[1]

                    # 过滤边界点
                    if not (BOUNDARY_THRESH <= xval <= 1-BOUNDARY_THRESH and
                            BOUNDARY_THRESH <= yval <= 1-BOUNDARY_THRESH):
                        continue

                    # 从 cos/sin 计算方向
                    cos_value = prediction[4, i, j]
                    sin_value = prediction[5, i, j]
                    direction = math.atan2(sin_value, cos_value)

                    # 创建标记点
                    marking_point = MarkingPoint(
                        xval, yval, direction, prediction[1, i, j])
                    predicted_points.append((confidence, marking_point))

        return self.non_maximum_suppression(predicted_points)

    def detect_marking_points(self, image: np.ndarray) -> List:
        """
        检测车位标记点.

        Args:
            image: RGB 图像 [H, W, 3]

        Returns:
            标记点列表 [(confidence, MarkingPoint), ...]
        """
        original_height, original_width = image.shape[:2]

        # 预处理
        input_tensor = self.preprocess_image(image).to(self.device)

        # 推理
        with torch.no_grad():
            prediction = self.detector(input_tensor)

        # 提取标记点
        pred_points = self.get_predicted_points(prediction[0])

        # 转换回原始图像坐标
        marking_points = []
        for confidence, point in pred_points:
            # 创建兼容的标记点对象
            class MP:
                def __init__(self, x, y, direction, confidence):
                    self.x = x * original_width
                    self.y = y * original_height
                    self.direction = direction
                    self.confidence = confidence

            marking_points.append(MP(point.x, point.y, point.direction, confidence))

        return marking_points

    def calc_point_square_dist(self, point_a, point_b, img_shape):
        """计算两点之间的归一化距离平方."""
        h, w = img_shape[:2]
        distx = (point_a.x / w) - (point_b.x / w)
        disty = (point_a.y / h) - (point_b.y / h)
        return distx ** 2 + disty ** 2

    def pass_through_third_point(self, marking_points, i, j, img_shape):
        """检查两点连线是否穿过第三个点."""
        h, w = img_shape[:2]
        x_1, y_1 = marking_points[i].x / w, marking_points[i].y / h
        x_2, y_2 = marking_points[j].x / w, marking_points[j].y / h

        for point_idx, point in enumerate(marking_points):
            if point_idx == i or point_idx == j:
                continue

            x_0, y_0 = point.x / w, point.y / h
            vec1 = np.array([x_0 - x_1, y_0 - y_1])
            vec2 = np.array([x_2 - x_0, y_2 - y_0])

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                vec1 = vec1 / norm1
                vec2 = vec2 / norm2

                if np.dot(vec1, vec2) > SLOT_SUPPRESSION_DOT_PRODUCT_THRESH:
                    return True
        return False

    def find_parking_slot_from_points(
        self,
        marking_points: List,
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        从标记点重建车位边界.

        Args:
            marking_points: 标记点列表
            image_shape: (height, width)

        Returns:
            车位四边形角点 [4, 2] 或 None
        """
        if len(marking_points) < 4:
            return None

        h, w = image_shape

        # 过滤：只保留图像下半部分的标记点
        lower_points = [p for p in marking_points if p.y > h * 0.3]

        if len(lower_points) < 4:
            lower_points = marking_points  # 回退到所有点

        # 配对标记点形成车位
        slots = []
        for i in range(len(lower_points) - 1):
            for j in range(i + 1, len(lower_points)):
                point_i = lower_points[i]
                point_j = lower_points[j]

                # 距离过滤
                distance = self.calc_point_square_dist(point_i, point_j, (h, w))
                if not (VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST or
                        HSLOT_MIN_DIST <= distance <= HSLOT_MAX_DIST):
                    continue

                # 穿透过滤
                if self.pass_through_third_point(lower_points, i, j, (h, w)):
                    continue

                slots.append((i, j))

        if not slots:
            # 回退：选择最接近中心的 4 个点
            center_x, center_y = w / 2, h * 0.6
            distances = []
            for p in lower_points:
                dist = np.sqrt((p.x - center_x)**2 + (p.y - center_y)**2)
                distances.append((dist, p))

            distances.sort(key=lambda x: x[0])
            selected_points = [p for _, p in distances[:4]]
        else:
            # 使用第一个检测到的车位
            i, j = slots[0]
            # 找到与这两个点相关的其他点
            selected_points = [lower_points[i], lower_points[j]]

            # 简化：添加最近的两个点
            remaining = [p for idx, p in enumerate(lower_points) if idx not in (i, j)]
            if len(remaining) >= 2:
                dists = [(np.sqrt((p.x - selected_points[0].x)**2 +
                                 (p.y - selected_points[0].y)**2), p)
                        for p in remaining]
                dists.sort(key=lambda x: x[0])
                selected_points.extend([p for _, p in dists[:2]])

        if len(selected_points) < 4:
            return None

        # 按 y 坐标排序
        selected_points.sort(key=lambda p: p.y)
        top_points = selected_points[:2]
        bottom_points = selected_points[2:]

        # 按 x 坐标排序
        top_points.sort(key=lambda p: p.x)
        bottom_points.sort(key=lambda p: p.x)

        # 构建四边形
        corners = np.array([
            [top_points[0].x, top_points[0].y],
            [top_points[1].x, top_points[1].y],
            [bottom_points[1].x, bottom_points[1].y],
            [bottom_points[0].x, bottom_points[0].y],
        ], dtype=np.float32)

        return corners
