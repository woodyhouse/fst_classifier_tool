"""
车位几何分析模块 - 基于透视变换的 9 宫格划分.

功能:
1. 检测车位线（传统 CV 方法）
2. 计算透视变换矩阵
3. 在地面坐标系划分 9 宫格
4. 将地面网格投影回图像坐标系
5. 将检测到的物体映射到对应格子
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
import cv2

from fst.vanishing_point import VanishingPointDetector
from fst.marking_point_detector import MarkingPointDetector

# 尝试导入深度学习版本（可选）
try:
    from fst.dl_marking_point_detector import DLMarkingPointDetector
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("  [警告] 深度学习模块不可用，将使用传统 CV 方法")


@dataclass
class ParkingSlot:
    """车位信息."""
    corners: np.ndarray  # [4, 2] 四个角点 (x, y) - 图像坐标系
    center: Tuple[float, float]  # 中心点 - 图像坐标系
    width: float  # 宽度（像素）
    height: float  # 高度（像素）
    perspective_matrix: Optional[np.ndarray] = None  # 透视变换矩阵 (3x3)
    inverse_matrix: Optional[np.ndarray] = None  # 逆透视变换矩阵


@dataclass
class GridCell:
    """9 宫格中的一个格子."""
    position: str  # "1"-"7", "P_LEFT", "P_RIGHT"
    polygon: np.ndarray  # [N, 2] 多边形顶点 - 图像坐标系
    center: Tuple[float, float]  # 中心点 - 图像坐标系
    ground_polygon: Optional[np.ndarray] = None  # 地面坐标系多边形


class ParkingGeometry:
    """
    车位几何分析器 - 支持透视变换.

    工作流程:
    1. 检测车位线（白色/黄色直线）
    2. 找到车位的四个角点
    3. 计算透视变换矩阵（图像 → 地面）
    4. 在地面坐标系划分 9 宫格
    5. 将地面网格投影回图像坐标系
    """

    def __init__(
        self,
        line_color_lower: Tuple[int, int, int] = (180, 180, 180),  # 白色阈值
        line_color_upper: Tuple[int, int, int] = (255, 255, 255),
        canny_low: int = 50,
        canny_high: int = 150,
        ground_slot_size: Tuple[float, float] = (2.5, 5.0),  # 标准车位尺寸（米）
        use_vanishing_point: bool = True,  # 是否使用消失点检测
        use_marking_points: bool = True,  # 是否使用标记点检测（传统 CV）
        use_dl_marking_points: bool = False,  # 是否使用深度学习标记点检测
        dl_weights_path: str = "weights/dmpr_ps.pth",  # 深度学习模型权重路径
    ):
        self.line_color_lower = np.array(line_color_lower)
        self.line_color_upper = np.array(line_color_upper)
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.ground_slot_size = ground_slot_size  # (宽, 长) 单位: 米
        self.use_vanishing_point = use_vanishing_point
        self.use_marking_points = use_marking_points
        self.use_dl_marking_points = use_dl_marking_points

        # 初始化消失点检测器
        if use_vanishing_point:
            self.vp_detector = VanishingPointDetector(
                canny_low=canny_low,
                canny_high=canny_high,
            )

        # 初始化标记点检测器（传统 CV）
        if use_marking_points and not use_dl_marking_points:
            self.mp_detector = MarkingPointDetector(
                corner_quality=0.01,
                min_distance=20,
            )

        # 初始化深度学习标记点检测器
        if use_dl_marking_points:
            if not DL_AVAILABLE:
                print("  [错误] 深度学习模块不可用，请安装 PyTorch")
                print("  [提示] pip install torch torchvision")
                self.use_dl_marking_points = False
                self.use_marking_points = True  # 回退到传统方法
                self.mp_detector = MarkingPointDetector(
                    corner_quality=0.01,
                    min_distance=20,
                )
            else:
                try:
                    self.dl_mp_detector = DLMarkingPointDetector(
                        weights_path=dl_weights_path,
                        conf_threshold=0.5,
                        use_cuda=True,
                    )
                    print("  [DMPR-PS] 深度学习模型加载成功")
                except Exception as e:
                    print(f"  [错误] 无法加载深度学习模型: {e}")
                    print("  [提示] 回退到传统 CV 方法")
                    self.use_dl_marking_points = False
                    self.use_marking_points = True
                    self.mp_detector = MarkingPointDetector(
                        corner_quality=0.01,
                        min_distance=20,
                    )

    def detect_parking_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测车位线.

        Args:
            image: RGB 图片 [H, W, 3]

        Returns:
            直线列表 [(x1, y1, x2, y2), ...]
        """
        # 颜色过滤（白色/黄色）
        mask_white = cv2.inRange(image, self.line_color_lower, self.line_color_upper)

        # 黄色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))

        # 蓝色检测（部分停车场用蓝色）
        mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))

        mask = cv2.bitwise_or(mask_white, mask_yellow)
        mask = cv2.bitwise_or(mask, mask_blue)

        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 边缘检测
        edges = cv2.Canny(mask, self.canny_low, self.canny_high)

        # Hough 直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10,
        )

        if lines is None:
            return []

        return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines[:, 0]]

    def find_parking_slot(
        self,
        image: np.ndarray,
        lines: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Optional[ParkingSlot]:
        """
        从检测到的直线中找到车位的四个角点，并计算透视变换矩阵.

        策略:
        1. 优先使用深度学习标记点检测（最准确）
        2. 使用传统 CV 标记点检测
        3. 如果检测到足够的直线，尝试拟合车位四边形
        4. 否则使用启发式方法（假设车位在图片中心偏下区域）
        """
        h, w = image.shape[:2]

        # 策略 1: 深度学习标记点检测
        if self.use_dl_marking_points:
            marking_points = self.dl_mp_detector.detect_marking_points(image)
            if marking_points:
                corners = self.dl_mp_detector.find_parking_slot_from_points(
                    marking_points, (h, w)
                )
                if corners is not None:
                    print(f"  [DMPR-PS] 成功从深度学习标记点重建车位边界")
                    return self._create_slot_from_corners(corners, w, h)

        # 策略 2: 传统 CV 标记点检测
        if self.use_marking_points and not self.use_dl_marking_points:
            marking_points = self.mp_detector.detect_marking_points(image)
            if marking_points:
                corners = self.mp_detector.find_parking_slot_from_points(
                    marking_points, (h, w)
                )
                if corners is not None:
                    print(f"  [标记点] 成功从标记点重建车位边界")
                    return self._create_slot_from_corners(corners, w, h)

        # 策略 3: 从直线拟合
        if lines and len(lines) >= 4:
            slot = self._fit_slot_from_lines(lines, w, h)
            if slot:
                return slot

        # 策略 3: 启发式估计（支持消失点自适应）
        print(f"  [启发式] 使用启发式方法估计车位")
        return self._estimate_slot_heuristic(w, h, image)

    def _fit_slot_from_lines(
        self,
        lines: List[Tuple[int, int, int, int]],
        w: int,
        h: int,
    ) -> Optional[ParkingSlot]:
        """
        从检测到的直线拟合车位四边形（简化版本）.

        实际应用中需要更复杂的算法:
        - 直线聚类（按角度和位置）
        - 平行线检测
        - 交点计算
        """
        # TODO: 实现完整的直线拟合算法
        # 这里暂时返回 None，使用启发式方法
        return None

    def _estimate_slot_heuristic(self, w: int, h: int, image: Optional[np.ndarray] = None) -> ParkingSlot:
        """
        启发式估计车位位置（考虑透视效果）.

        如果启用消失点检测，会自动分析图片透视关系并调整参数.

        假设:
        - 车位在图片下半部分
        - 相机俯视角度约 30-45 度
        - 车位呈梯形（近大远小）
        """
        # 默认参数（固定透视）
        y_far_ratio = 0.35
        y_near_ratio = 0.85
        w_far_ratio = 0.4
        w_near_ratio = 0.6
        center_offset = 0.0

        # 如果启用消失点检测，自动调整参数
        if self.use_vanishing_point and image is not None:
            vp_params = self.vp_detector.detect(image)
            if vp_params:
                horizon_y = vp_params['horizon_y']
                offset_x = vp_params['offset_x']
                tilt_angle = vp_params['tilt_angle']

                print(f"  [消失点] 地平线 y={horizon_y:.1f}, 偏移={offset_x:.3f}, 俯角={tilt_angle:.1f}°")

                # 根据地平线位置连续调整参数
                horizon_ratio = np.clip(horizon_y / h, 0.0, 1.0)

                # 远端 y 坐标（地平线附近）
                # horizon_ratio 越小（地平线越高），远端越靠上
                y_far_ratio = 0.15 + horizon_ratio * 0.35  # 0.15-0.50

                # 近端 y 坐标（图片下方）
                # 俯视角越大，近端越靠下
                y_near_ratio = 0.75 + (1.0 - horizon_ratio) * 0.15  # 0.75-0.90

                # 远端宽度（透视收缩）
                # horizon_ratio 越小，透视效果越强，远端越窄
                perspective_strength = 1.0 - horizon_ratio  # 0-1
                w_far_ratio = 0.25 + perspective_strength * 0.25  # 0.25-0.50

                # 近端宽度
                w_near_ratio = 0.50 + perspective_strength * 0.20  # 0.50-0.70

                # 根据水平偏移调整车位中心
                # offset_x > 0: 消失点在右侧，车位可能偏左
                # offset_x < 0: 消失点在左侧，车位可能偏右
                center_offset = -offset_x * 0.2  # 反向补偿，减小系数避免过度调整

                print(f"  [自适应] 远端y={y_far_ratio:.3f}, 近端y={y_near_ratio:.3f}")
                print(f"  [自适应] 远端w={w_far_ratio:.3f}, 近端w={w_near_ratio:.3f}, 中心偏移={center_offset:.3f}")

        # 远端 y 坐标（图片上方）
        y_far = int(h * y_far_ratio)
        # 近端 y 坐标（图片下方）
        y_near = int(h * y_near_ratio)

        # 远端宽度（较窄）
        w_far = int(w * w_far_ratio)
        # 近端宽度（较宽）
        w_near = int(w * w_near_ratio)

        # 中心 x 坐标（考虑偏移）
        center_x = w / 2 + center_offset * w

        # 四个角点（逆时针：左上、右上、右下、左下）
        corners = np.array([
            [center_x - w_far/2, y_far],      # 左上（远端左）
            [center_x + w_far/2, y_far],      # 右上（远端右）
            [center_x + w_near/2, y_near],    # 右下（近端右）
            [center_x - w_near/2, y_near],    # 左下（近端左）
        ], dtype=np.float32)

        # 计算中心点（梯形中心）
        cx = center_x
        cy = (y_far + y_near) / 2

        # 计算透视变换矩阵
        # 地面坐标系: 标准矩形车位 (单位: 米 × 100 = 厘米)
        slot_w, slot_h = self.ground_slot_size
        ground_w = slot_w * 100  # 转换为厘米
        ground_h = slot_h * 100

        # 地面四个角点（矩形）
        ground_corners = np.array([
            [0, 0],                    # 左上
            [ground_w, 0],             # 右上
            [ground_w, ground_h],      # 右下
            [0, ground_h],             # 左下
        ], dtype=np.float32)

        # 计算透视变换矩阵（地面 → 图像）
        perspective_matrix = cv2.getPerspectiveTransform(ground_corners, corners)
        # 逆矩阵（图像 → 地面）
        inverse_matrix = cv2.getPerspectiveTransform(corners, ground_corners)

        # 计算车位尺寸（图像坐标系）
        slot_w_img = np.linalg.norm(corners[1] - corners[0])
        slot_h_img = np.linalg.norm(corners[3] - corners[0])

        return ParkingSlot(
            corners=corners,
            center=(cx, cy),
            width=slot_w_img,
            height=slot_h_img,
            perspective_matrix=perspective_matrix,
            inverse_matrix=inverse_matrix,
        )

    def _create_slot_from_corners(self, corners: np.ndarray, w: int, h: int) -> ParkingSlot:
        """
        从四个角点创建车位对象.

        Args:
            corners: [4, 2] 四个角点 (x, y)
            w: 图像宽度
            h: 图像高度

        Returns:
            ParkingSlot 对象
        """
        # 计算中心点
        cx = np.mean(corners[:, 0])
        cy = np.mean(corners[:, 1])

        # 计算透视变换矩阵
        slot_w, slot_h = self.ground_slot_size
        ground_w = slot_w * 100  # 转换为厘米
        ground_h = slot_h * 100

        # 地面四个角点（矩形）
        ground_corners = np.array([
            [0, 0],                    # 左上
            [ground_w, 0],             # 右上
            [ground_w, ground_h],      # 右下
            [0, ground_h],             # 左下
        ], dtype=np.float32)

        # 计算透视变换矩阵（地面 → 图像）
        perspective_matrix = cv2.getPerspectiveTransform(ground_corners, corners)
        # 逆矩阵（图像 → 地面）
        inverse_matrix = cv2.getPerspectiveTransform(corners, ground_corners)

        # 计算车位尺寸（图像坐标系）
        slot_w_img = np.linalg.norm(corners[1] - corners[0])
        slot_h_img = np.linalg.norm(corners[3] - corners[0])

        return ParkingSlot(
            corners=corners,
            center=(cx, cy),
            width=slot_w_img,
            height=slot_h_img,
            perspective_matrix=perspective_matrix,
            inverse_matrix=inverse_matrix,
        )

    def create_9grid(self, slot: ParkingSlot) -> Dict[str, GridCell]:
        """
        以车位为中心创建 9 宫格（考虑透视变换）.

        布局（地面坐标系）:
        ```
        远端:  1   2   3
        中间:  4   5   6
        近端:  7  P_L P_R
        ```

        流程:
        1. 在地面坐标系划分 9 宫格（矩形）
        2. 使用透视变换投影回图像坐标系（梯形）
        """
        if slot.perspective_matrix is None:
            # 回退到简单矩形划分
            return self._create_9grid_simple(slot)

        # 地面坐标系尺寸（厘米）
        slot_w, slot_h = self.ground_slot_size
        ground_w = slot_w * 100
        ground_h = slot_h * 100

        # 扩展区域（车位周围各扩展 1 个车位宽度）
        extend_w = ground_w  # 左右各扩展
        extend_h = ground_h * 0.5  # 前方扩展

        # 总区域尺寸
        total_w = ground_w + 2 * extend_w
        total_h = ground_h + extend_h

        # 网格尺寸
        grid_w = total_w / 3
        grid_h = total_h / 3

        # 原点偏移（使车位在合适位置）
        offset_x = -extend_w
        offset_y = -extend_h

        cells: Dict[str, GridCell] = {}

        # 定义 9 宫格布局（地面坐标系）
        grid_layout = [
            ("1", 0, 0), ("2", 1, 0), ("3", 2, 0),  # 远端
            ("4", 0, 1), ("5", 1, 1), ("6", 2, 1),  # 中间
            ("7", 1, 2),                             # 近端中间
        ]

        for pos, gx, gy in grid_layout:
            # 地面坐标系的矩形角点
            x1 = offset_x + gx * grid_w
            y1 = offset_y + gy * grid_h
            x2 = x1 + grid_w
            y2 = y1 + grid_h

            ground_polygon = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)

            # 投影到图像坐标系
            image_polygon = self._transform_polygon(ground_polygon, slot.perspective_matrix)

            # 计算图像坐标系中心
            center_x = np.mean(image_polygon[:, 0])
            center_y = np.mean(image_polygon[:, 1])

            cells[pos] = GridCell(
                position=pos,
                polygon=image_polygon,
                center=(center_x, center_y),
                ground_polygon=ground_polygon,
            )

        # 左右两侧（P_LEFT, P_RIGHT）
        # P_LEFT: 车位左侧
        p_left_ground = np.array([
            [offset_x - grid_w, offset_y + grid_h],
            [offset_x, offset_y + grid_h],
            [offset_x, offset_y + 2 * grid_h],
            [offset_x - grid_w, offset_y + 2 * grid_h],
        ], dtype=np.float32)
        p_left_image = self._transform_polygon(p_left_ground, slot.perspective_matrix)
        cells["P_LEFT"] = GridCell(
            position="P_LEFT",
            polygon=p_left_image,
            center=(np.mean(p_left_image[:, 0]), np.mean(p_left_image[:, 1])),
            ground_polygon=p_left_ground,
        )

        # P_RIGHT: 车位右侧
        p_right_ground = np.array([
            [offset_x + ground_w, offset_y + grid_h],
            [offset_x + ground_w + grid_w, offset_y + grid_h],
            [offset_x + ground_w + grid_w, offset_y + 2 * grid_h],
            [offset_x + ground_w, offset_y + 2 * grid_h],
        ], dtype=np.float32)
        p_right_image = self._transform_polygon(p_right_ground, slot.perspective_matrix)
        cells["P_RIGHT"] = GridCell(
            position="P_RIGHT",
            polygon=p_right_image,
            center=(np.mean(p_right_image[:, 0]), np.mean(p_right_image[:, 1])),
            ground_polygon=p_right_ground,
        )

        return cells

    def _transform_polygon(self, polygon: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        使用透视变换矩阵变换多边形.

        Args:
            polygon: [N, 2] 多边形顶点
            matrix: [3, 3] 透视变换矩阵

        Returns:
            变换后的多边形 [N, 2]
        """
        # 添加齐次坐标
        ones = np.ones((polygon.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([polygon, ones])  # [N, 3]

        # 应用变换
        transformed = homogeneous @ matrix.T  # [N, 3]

        # 归一化（除以 w）
        transformed[:, 0] /= transformed[:, 2]
        transformed[:, 1] /= transformed[:, 2]

        return transformed[:, :2]

    def _create_9grid_simple(self, slot: ParkingSlot) -> Dict[str, GridCell]:
        """
        简单矩形 9 宫格（无透视变换）- 回退方案.
        """
        cx, cy = slot.center
        w, h = slot.width, slot.height

        grid_w = w / 3
        grid_h = h / 3

        cells: Dict[str, GridCell] = {}

        positions = [
            ("1", 0, 0), ("2", 1, 0), ("3", 2, 0),
            ("4", 0, 1), ("5", 1, 1), ("6", 2, 1),
            ("7", 1, 2),
        ]

        for pos, gx, gy in positions:
            x1 = cx - w/2 + gx * grid_w
            y1 = cy - h/2 + gy * grid_h
            x2 = x1 + grid_w
            y2 = y1 + grid_h

            polygon = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)

            cells[pos] = GridCell(
                position=pos,
                polygon=polygon,
                center=((x1 + x2) / 2, (y1 + y2) / 2),
            )

        # 左右两侧
        cells["P_LEFT"] = GridCell(
            position="P_LEFT",
            polygon=np.array([
                [cx - w/2 - grid_w, cy - h/2],
                [cx - w/2, cy - h/2],
                [cx - w/2, cy + h/2],
                [cx - w/2 - grid_w, cy + h/2],
            ], dtype=np.float32),
            center=(cx - w/2 - grid_w/2, cy),
        )

        cells["P_RIGHT"] = GridCell(
            position="P_RIGHT",
            polygon=np.array([
                [cx + w/2, cy - h/2],
                [cx + w/2 + grid_w, cy - h/2],
                [cx + w/2 + grid_w, cy + h/2],
                [cx + w/2, cy + h/2],
            ], dtype=np.float32),
            center=(cx + w/2 + grid_w/2, cy),
        )

        return cells

    def map_point_to_grid(
        self,
        point: Tuple[float, float],
        grid_cells: Dict[str, GridCell],
    ) -> Optional[str]:
        """
        将点映射到 9 宫格中的某个格子.

        Args:
            point: (x, y) 坐标 - 图像坐标系
            grid_cells: 9 宫格字典

        Returns:
            格子位置 ("1"-"7", "P_LEFT", "P_RIGHT") 或 None
        """
        px, py = point

        for pos, cell in grid_cells.items():
            # 确保多边形是正确的格式 [N, 2] 且类型为 float32
            polygon = cell.polygon.astype(np.float32)
            if polygon.ndim != 2 or polygon.shape[1] != 2:
                continue

            # cv2.pointPolygonTest 需要连续的数组
            polygon = np.ascontiguousarray(polygon)

            result = cv2.pointPolygonTest(polygon, (float(px), float(py)), False)
            if result >= 0:  # 点在多边形内或边上
                return pos

        return None

    def visualize_grid(
        self,
        image: np.ndarray,
        grid_cells: Dict[str, GridCell],
        slot: Optional[ParkingSlot] = None,
    ) -> np.ndarray:
        """
        可视化 9 宫格（支持透视变换）.

        Returns:
            带网格线的图片
        """
        vis = image.copy()

        # 绘制车位边界（红色）
        if slot is not None:
            pts = slot.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (255, 0, 0), 3)

        # 绘制 9 宫格（绿色）
        for pos, cell in grid_cells.items():
            pts = cell.polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

            # 标注位置
            cx, cy = cell.center
            cv2.putText(
                vis, pos,
                (int(cx) - 10, int(cy) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 2,
            )

        return vis


if __name__ == "__main__":
    # 测试代码
    import cv2

    img = cv2.imread("test.jpg")
    if img is None:
        print("请提供测试图片 test.jpg")
        exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    geo = ParkingGeometry()
    lines = geo.detect_parking_lines(img_rgb)
    print(f"检测到 {len(lines)} 条直线")

    slot = geo.find_parking_slot(img_rgb, lines)
    if slot:
        print(f"车位中心: {slot.center}")
        print(f"车位尺寸: {slot.width:.0f}x{slot.height:.0f} 像素")
        print(f"透视变换: {'已启用' if slot.perspective_matrix is not None else '未启用'}")

        grid = geo.create_9grid(slot)
        vis = geo.visualize_grid(img_rgb, grid, slot)

        cv2.imwrite("grid_visualization.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("✓ 可视化结果已保存: grid_visualization.jpg")
