"""
车位标记点检测模块 - 基于传统 CV 方法

参考 DMPR-PS 的思路，但使用传统计算机视觉方法：
1. 检测图像中的角点（车位标记点）
2. 根据几何关系连接标记点形成车位边界
3. 输出车位四边形坐标

优势：
- 不需要深度学习模型
- 可以直接检测实际车位线
- 比启发式估计更准确
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2


class MarkingPoint:
    """车位标记点."""

    def __init__(self, x: float, y: float, direction: float, confidence: float = 1.0):
        self.x = x
        self.y = y
        self.direction = direction  # 标记点方向（弧度）
        self.confidence = confidence

    def __repr__(self):
        return f"MarkingPoint({self.x:.1f}, {self.y:.1f}, dir={np.degrees(self.direction):.1f}°)"


class MarkingPointDetector:
    """车位标记点检测器."""

    def __init__(
        self,
        corner_quality: float = 0.01,
        min_distance: int = 20,
        block_size: int = 7,
    ):
        """
        Args:
            corner_quality: 角点检测质量阈值
            min_distance: 角点之间的最小距离
            block_size: 角点检测窗口大小
        """
        self.corner_quality = corner_quality
        self.min_distance = min_distance
        self.block_size = block_size

    def detect_corners(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        检测图像中的角点（潜在的车位标记点）.

        Args:
            image: RGB 图像 [H, W, 3]

        Returns:
            角点列表 [(x, y), ...]
        """
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Shi-Tomasi 角点检测
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=self.corner_quality,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

        if corners is None:
            return []

        return [(float(x), float(y)) for x, y in corners[:, 0]]

    def detect_lines_around_corner(
        self,
        image: np.ndarray,
        corner: Tuple[float, float],
        roi_size: int = 30,
    ) -> List[Tuple[float, float, float, float]]:
        """
        在角点周围检测直线，用于确定标记点方向.

        Args:
            image: 灰度图像
            corner: 角点坐标 (x, y)
            roi_size: ROI 大小

        Returns:
            直线列表 [(x1, y1, x2, y2), ...]
        """
        h, w = image.shape[:2]
        cx, cy = int(corner[0]), int(corner[1])

        # 提取 ROI
        x1 = max(0, cx - roi_size)
        y1 = max(0, cy - roi_size)
        x2 = min(w, cx + roi_size)
        y2 = min(h, cy + roi_size)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return []

        # 边缘检测
        edges = cv2.Canny(roi, 50, 150)

        # Hough 直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=10,
            maxLineGap=5,
        )

        if lines is None:
            return []

        # 转换回全局坐标
        global_lines = []
        for x1_l, y1_l, x2_l, y2_l in lines[:, 0]:
            global_lines.append((
                float(x1_l + x1),
                float(y1_l + y1),
                float(x2_l + x1),
                float(y2_l + y1),
            ))

        return global_lines

    def estimate_direction(self, lines: List[Tuple[float, float, float, float]]) -> Optional[float]:
        """
        从直线估计标记点方向.

        Args:
            lines: 直线列表

        Returns:
            方向角（弧度），如果无法估计则返回 None
        """
        if not lines:
            return None

        # 计算每条直线的角度
        angles = []
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)

        # 使用中位数作为主方向
        return float(np.median(angles))

    def detect_marking_points(self, image: np.ndarray) -> List[MarkingPoint]:
        """
        检测车位标记点.

        Args:
            image: RGB 图像 [H, W, 3]

        Returns:
            标记点列表
        """
        # 1. 检测角点
        corners = self.detect_corners(image)
        print(f"  [标记点] 检测到 {len(corners)} 个角点")

        if not corners:
            return []

        # 2. 为每个角点估计方向
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        marking_points = []

        for corner in corners:
            lines = self.detect_lines_around_corner(gray, corner)
            direction = self.estimate_direction(lines)

            if direction is not None:
                marking_points.append(MarkingPoint(
                    x=corner[0],
                    y=corner[1],
                    direction=direction,
                    confidence=1.0,
                ))

        print(f"  [标记点] 有效标记点: {len(marking_points)} 个")
        return marking_points

    def find_parking_slot_from_points(
        self,
        marking_points: List[MarkingPoint],
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        从标记点重建车位边界.

        策略：
        1. 在图像下半部分寻找标记点
        2. 根据空间位置聚类成车位
        3. 选择最接近图像中心的车位

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
            return None

        # 简化版本：找到最接近图像中心的 4 个点
        center_x, center_y = w / 2, h * 0.6

        # 计算每个点到中心的距离
        distances = []
        for p in lower_points:
            dist = np.sqrt((p.x - center_x)**2 + (p.y - center_y)**2)
            distances.append((dist, p))

        # 排序并选择最近的 4 个点
        distances.sort(key=lambda x: x[0])
        selected_points = [p for _, p in distances[:4]]

        # 按 y 坐标排序（上面 2 个，下面 2 个）
        selected_points.sort(key=lambda p: p.y)
        top_points = selected_points[:2]
        bottom_points = selected_points[2:]

        # 按 x 坐标排序（左右）
        top_points.sort(key=lambda p: p.x)
        bottom_points.sort(key=lambda p: p.x)

        # 构建四边形：左上、右上、右下、左下
        corners = np.array([
            [top_points[0].x, top_points[0].y],      # 左上
            [top_points[1].x, top_points[1].y],      # 右上
            [bottom_points[1].x, bottom_points[1].y],  # 右下
            [bottom_points[0].x, bottom_points[0].y],  # 左下
        ], dtype=np.float32)

        return corners

    def visualize_marking_points(
        self,
        image: np.ndarray,
        marking_points: List[MarkingPoint],
    ) -> np.ndarray:
        """
        可视化标记点.

        Args:
            image: RGB 图像
            marking_points: 标记点列表

        Returns:
            可视化图像
        """
        vis = image.copy()

        for p in marking_points:
            # 绘制标记点（黄色圆圈）
            cv2.circle(vis, (int(p.x), int(p.y)), 5, (255, 255, 0), -1)

            # 绘制方向箭头
            length = 20
            dx = length * np.cos(p.direction)
            dy = length * np.sin(p.direction)
            end_x = int(p.x + dx)
            end_y = int(p.y + dy)
            cv2.arrowedLine(vis, (int(p.x), int(p.y)), (end_x, end_y),
                           (255, 255, 0), 2, tipLength=0.3)

        return vis
