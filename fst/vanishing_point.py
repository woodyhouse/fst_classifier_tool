"""
消失点检测模块 - 自动分析图片透视关系

基于论文: "Vanishing Point Detection in Urban Scenes Using Point and Line Features"
参考实现: https://github.com/rayryeng/XiaohuLuVPDetection

功能:
1. 检测图片中的消失点
2. 计算透视角度和方向
3. 自动调整 9 宫格透视参数
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2


class VanishingPointDetector:
    """
    消失点检测器.

    使用 Hough 直线检测 + RANSAC 拟合消失点.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 40,  # 降低阈值以检测更多直线
        min_line_length: int = 40,  # 降低最小长度
        max_line_gap: int = 15,     # 增加允许的间隙
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图片中的直线.

        Args:
            image: RGB 图片 [H, W, 3]

        Returns:
            直线列表 [(x1, y1, x2, y2), ...]
        """
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 增强对比度（帮助检测弱边缘）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 边缘检测
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # 形态学操作连接断裂的边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Hough 直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines[:, 0]]

    def compute_vanishing_point(
        self,
        lines: List[Tuple[int, int, int, int]],
        image_shape: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        """
        从直线计算消失点.

        使用 RANSAC 算法拟合消失点.

        Args:
            lines: 直线列表 [(x1, y1, x2, y2), ...]
            image_shape: (height, width)

        Returns:
            消失点坐标 (vx, vy) 或 None
        """
        if len(lines) < 2:
            return None

        h, w = image_shape

        # 过滤接近水平的直线（保留垂直和斜线）
        # 同时计算直线长度，优先使用长直线
        filtered_lines = []
        for x1, y1, x2, y2 in lines:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))

            # 放宽角度限制：10-170度（排除接近水平的线）
            # 车位线通常是垂直或斜向的
            if 10 * np.pi / 180 < angle < 170 * np.pi / 180:
                filtered_lines.append((x1, y1, x2, y2, length))

        if len(filtered_lines) < 2:
            return None

        # 按长度排序，优先使用长直线（更可靠）
        filtered_lines.sort(key=lambda x: x[4], reverse=True)
        # 只保留前50%的长直线
        filtered_lines = filtered_lines[:max(2, len(filtered_lines) // 2)]

        # 计算所有直线对的交点
        intersections = []
        for i in range(len(filtered_lines)):
            for j in range(i + 1, len(filtered_lines)):
                # 提取坐标（忽略长度）
                line1 = filtered_lines[i][:4]
                line2 = filtered_lines[j][:4]
                pt = self._line_intersection(line1, line2)
                if pt is not None:
                    x, y = pt
                    # 收紧交点范围：消失点应该在图像附近
                    # 允许在图像上方（俯视角度）但不能太远
                    if -0.5 * w < x < 1.5 * w and -0.5 * h < y < 1.2 * h:
                        intersections.append(pt)

        if len(intersections) < 3:
            return None

        # 使用 RANSAC 风格的鲁棒估计
        intersections = np.array(intersections)

        # 计算所有交点的中位数作为初始估计
        vx_init = np.median(intersections[:, 0])
        vy_init = np.median(intersections[:, 1])

        # 计算到初始估计的距离
        distances = np.sqrt((intersections[:, 0] - vx_init)**2 +
                           (intersections[:, 1] - vy_init)**2)

        # 只保留距离在 75% 分位数内的点（去除离群点）
        threshold = np.percentile(distances, 75)
        inliers = intersections[distances <= threshold]

        if len(inliers) < 2:
            # 回退到中位数
            return (float(vx_init), float(vy_init))

        # 使用内点的平均值作为最终消失点
        vx = np.mean(inliers[:, 0])
        vy = np.mean(inliers[:, 1])

        return (float(vx), float(vy))

    def _line_intersection(
        self,
        line1: Tuple[int, int, int, int],
        line2: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float]]:
        """
        计算两条直线的交点.

        Args:
            line1: (x1, y1, x2, y2)
            line2: (x3, y3, x4, y4)

        Returns:
            交点 (x, y) 或 None（平行线）
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # 计算直线方程 Ax + By = C
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3

        # 计算行列式
        det = A1 * B2 - A2 * B1

        if abs(det) < 1e-6:  # 平行线
            return None

        # 计算交点
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det

        return (x, y)

    def estimate_camera_parameters(
        self,
        vanishing_point: Tuple[float, float],
        image_shape: Tuple[int, int],
    ) -> dict:
        """
        从消失点估计相机参数.

        Args:
            vanishing_point: (vx, vy)
            image_shape: (height, width)

        Returns:
            相机参数字典:
            - horizon_y: 地平线 y 坐标
            - tilt_angle: 俯仰角（度）
            - offset_x: 水平偏移
        """
        h, w = image_shape
        vx, vy = vanishing_point

        # 地平线位置（消失点的 y 坐标）
        horizon_y = vy

        # 俯仰角估计（基于地平线位置）- 使用连续函数而非分段
        # 地平线越高（y越小），俯视角度越大
        horizon_ratio = np.clip(horizon_y / h, 0.0, 1.0)

        # 线性插值：horizon_ratio=0 → 70度, horizon_ratio=0.6 → 25度
        if horizon_ratio < 0.6:
            tilt_angle = 70 - (horizon_ratio / 0.6) * 45  # 70度到25度
        else:
            tilt_angle = 25 - ((horizon_ratio - 0.6) / 0.4) * 10  # 25度到15度

        tilt_angle = np.clip(tilt_angle, 15, 70)

        # 水平偏移（消失点偏离中心的程度）
        offset_x = (vx - w / 2) / w

        return {
            "horizon_y": horizon_y,
            "tilt_angle": tilt_angle,
            "offset_x": offset_x,
            "vanishing_point": vanishing_point,
        }

    def detect(self, image: np.ndarray) -> Optional[dict]:
        """
        完整的消失点检测流程.

        Args:
            image: RGB 图片 [H, W, 3]

        Returns:
            相机参数字典或 None
        """
        h, w = image.shape[:2]

        # 1. 检测直线
        lines = self.detect_lines(image)
        if len(lines) < 2:
            return None

        # 2. 计算消失点
        vp = self.compute_vanishing_point(lines, (h, w))
        if vp is None:
            return None

        # 3. 估计相机参数
        params = self.estimate_camera_parameters(vp, (h, w))
        params["num_lines"] = len(lines)

        return params

    def visualize(
        self,
        image: np.ndarray,
        lines: List[Tuple[int, int, int, int]],
        vanishing_point: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        可视化检测结果.

        Args:
            image: 原始图片
            lines: 检测到的直线
            vanishing_point: 消失点坐标

        Returns:
            可视化图片
        """
        vis = image.copy()

        # 绘制直线（黄色）
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # 绘制消失点（红色大圆）
        if vanishing_point is not None:
            vx, vy = vanishing_point
            vx, vy = int(vx), int(vy)
            cv2.circle(vis, (vx, vy), 10, (255, 0, 0), -1)
            cv2.circle(vis, (vx, vy), 15, (255, 0, 0), 2)
            cv2.putText(
                vis, "Vanishing Point",
                (vx + 20, vy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2,
            )

            # 绘制地平线（红色虚线）
            h, w = image.shape[:2]
            cv2.line(vis, (0, vy), (w, vy), (255, 0, 0), 2, cv2.LINE_AA)

        return vis


if __name__ == "__main__":
    # 测试代码
    import cv2

    img = cv2.imread("test.jpg")
    if img is None:
        print("请提供测试图片 test.jpg")
        exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector = VanishingPointDetector()

    # 检测消失点
    params = detector.detect(img_rgb)

    if params:
        print("✓ 检测成功！")
        print(f"  消失点: {params['vanishing_point']}")
        print(f"  地平线 y: {params['horizon_y']:.1f}")
        print(f"  俯仰角: {params['tilt_angle']:.1f}°")
        print(f"  水平偏移: {params['offset_x']:.3f}")
        print(f"  检测到 {params['num_lines']} 条直线")

        # 可视化
        lines = detector.detect_lines(img_rgb)
        vis = detector.visualize(img_rgb, lines, params['vanishing_point'])

        cv2.imwrite("vanishing_point_result.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("✓ 可视化结果已保存: vanishing_point_result.jpg")
    else:
        print("✗ 未检测到消失点")
