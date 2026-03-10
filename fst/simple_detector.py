"""
简化版车位线检测 - 基于霍夫直线检测
不需要复杂的几何分析，直接检测车位线
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


class SimpleParkingLineDetector:
    """简化版车位线检测器"""

    def __init__(self):
        # 霍夫变换参数
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 20

    def detect(self, image: np.ndarray) -> dict:
        """
        检测车位线

        Returns:
            {
                'vertical_lines': List[line],  # 垂直线
                'horizontal_lines': List[line],  # 水平线
                'slot_type': str,  # 车位类型估计
                'has_lines': bool  # 是否检测到线
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # 霍夫直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None or len(lines) == 0:
            return {
                'vertical_lines': [],
                'horizontal_lines': [],
                'slot_type': 'UNKNOWN',
                'has_lines': False
            }

        # 分类垂直线和水平线
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算角度
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # 垂直线：80-100度
            if 80 < angle < 100:
                vertical_lines.append(line[0])
            # 水平线：0-10度 或 170-180度
            elif angle < 10 or angle > 170:
                horizontal_lines.append(line[0])

        # 估计车位类型
        slot_type = self._estimate_slot_type(vertical_lines, horizontal_lines)

        return {
            'vertical_lines': vertical_lines,
            'horizontal_lines': horizontal_lines,
            'slot_type': slot_type,
            'has_lines': len(vertical_lines) > 0 or len(horizontal_lines) > 0
        }

    def _estimate_slot_type(self, v_lines: List, h_lines: List) -> str:
        """根据线条数量估计车位类型"""
        if len(v_lines) >= 2:
            return 'PERPENDICULAR'  # 垂直车位
        elif len(h_lines) >= 2:
            return 'PARALLEL'  # 平行车位
        elif len(v_lines) > 0 and len(h_lines) > 0:
            return 'ANGLED'  # 斜列车位
        else:
            return 'UNKNOWN'

    def visualize(self, image: np.ndarray, result: dict) -> np.ndarray:
        """可视化检测结果"""
        vis_img = image.copy()

        # 绘制垂直线（绿色）
        for line in result['vertical_lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制水平线（蓝色）
        for line in result['horizontal_lines']:
            x1, y1, x2, y2 = line
            cv2.line(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 添加文字信息
        info = f"Type: {result['slot_type']} | V:{len(result['vertical_lines'])} H:{len(result['horizontal_lines'])}"
        cv2.putText(vis_img, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_img


def simple_grid_mapping(obstacles: dict, image_shape: Tuple[int, int]) -> dict:
    """
    简化版 9 宫格映射
    根据障碍物的位置（bbox）直接映射到 9 宫格

    Args:
        obstacles: YOLO 检测结果 {class_name: [bbox1, bbox2, ...]}
        image_shape: (height, width)

    Returns:
        9 宫格映射 {'1': ['EMPTY'], '2': ['VEHICLE'], ...}
        注意：值是列表，符合 FST Schema
    """
    h, w = image_shape

    # 初始化 9 宫格（值为列表）
    grid = {str(i): ['EMPTY'] for i in range(1, 8)}
    grid['P_LEFT'] = ['EMPTY']
    grid['P_RIGHT'] = ['EMPTY']

    # 定义 9 宫格区域（简化版）
    # 将图像分成 3x3 网格
    grid_regions = {
        '1': (0, 0, w//3, h//3),           # 左上
        '2': (w//3, 0, 2*w//3, h//3),      # 中上
        '3': (2*w//3, 0, w, h//3),         # 右上
        '4': (0, h//3, w//3, 2*h//3),      # 左中
        '5': (w//3, h//3, 2*w//3, 2*h//3), # 中中
        '6': (2*w//3, h//3, w, 2*h//3),    # 右中
        '7': (0, 2*h//3, w, h),            # 底部（合并）
        'P_LEFT': (0, h//3, w//3, h),      # 左侧车位
        'P_RIGHT': (2*w//3, h//3, w, h),   # 右侧车位
    }

    # 映射障碍物到网格
    for obj_class, bboxes in obstacles.items():
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 找到障碍物所在的网格
            for grid_id, (gx1, gy1, gx2, gy2) in grid_regions.items():
                if gx1 <= center_x < gx2 and gy1 <= center_y < gy2:
                    # 如果该格子是空的，替换；否则追加
                    if grid[grid_id] == ['EMPTY']:
                        grid[grid_id] = [obj_class]
                    else:
                        # 避免重复
                        if obj_class not in grid[grid_id]:
                            grid[grid_id].append(obj_class)
                    break

    return grid
