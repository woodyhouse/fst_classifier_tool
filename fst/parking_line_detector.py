"""
车位线检测算法
基于边缘检测和直线检测的方法
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


class ParkingLineDetector:
    """车位线检测器"""

    def __init__(self,
                 min_line_length: int = 50,
                 max_line_gap: int = 10,
                 parallel_angle_threshold: float = 5.0,
                 min_distance: int = 15,
                 max_distance: int = 150,
                 min_overlap_ratio: float = 0.3,
                 max_length_ratio: float = 3.0):
        """
        初始化检测器

        Args:
            min_line_length: 最小线段长度（像素）
            max_line_gap: 线段间最大间隙
            parallel_angle_threshold: 平行判定角度阈值（度）
            min_distance: 两条线最小距离（像素）
            max_distance: 两条线最大距离（像素）
            min_overlap_ratio: 最小重叠比例（防止错位的平行线）
            max_length_ratio: 最大长度比例（两条线长度不应相差太大）
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.parallel_angle_threshold = parallel_angle_threshold
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_overlap_ratio = min_overlap_ratio
        self.max_length_ratio = max_length_ratio

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：转换为边缘图"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 更强的模糊以减少噪声
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

        # 降低Canny阈值以检测更多边缘
        edges = cv2.Canny(blurred, 30, 100)
        return edges

    def detect_lines(self, edges: np.ndarray) -> List[np.ndarray]:
        """检测直线"""
        # 降低阈值，增加maxLineGap以连接断裂的线段
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap * 3)
        return lines if lines is not None else []

    def calculate_angle(self, line: np.ndarray) -> float:
        """计算线段角度"""
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle

    def calculate_length(self, line: np.ndarray) -> float:
        """计算线段长度"""
        x1, y1, x2, y2 = line[0]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def are_parallel(self, line1: np.ndarray, line2: np.ndarray) -> bool:
        """判断两条线是否平行"""
        angle1 = self.calculate_angle(line1)
        angle2 = self.calculate_angle(line2)

        # 归一化角度到0-180度
        angle1 = angle1 % 180
        angle2 = angle2 % 180

        angle_diff = abs(angle1 - angle2)
        # 考虑180度周期性
        angle_diff = min(angle_diff, 180 - angle_diff)

        return angle_diff < self.parallel_angle_threshold

    def calculate_distance(self, line1: np.ndarray, line2: np.ndarray) -> float:
        """计算两条平行线之间的距离"""
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # 计算线1的中点到线2的距离
        mid_x1, mid_y1 = (x1 + x2) / 2, (y1 + y2) / 2

        # 线2的方向向量
        dx = x4 - x3
        dy = y4 - y3
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return float('inf')

        # 点到直线距离
        distance = abs(dy * mid_x1 - dx * mid_y1 + x4 * y3 - y4 * x3) / length
        return distance

    def merge_collinear_lines(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """合并共线的线段"""
        if len(lines) == 0:
            return []

        merged = []
        used = [False] * len(lines)

        for i in range(len(lines)):
            if used[i]:
                continue

            x1, y1, x2, y2 = lines[i][0]
            angle_i = self.calculate_angle(lines[i])

            # 找到所有与当前线段共线的线段
            group = [(x1, y1), (x2, y2)]

            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue

                angle_j = self.calculate_angle(lines[j])
                angle_diff = abs((angle_i % 180) - (angle_j % 180))
                angle_diff = min(angle_diff, 180 - angle_diff)

                # 角度相近
                if angle_diff > 10:
                    continue

                x3, y3, x4, y4 = lines[j][0]

                # 计算点到直线的距离
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)

                if length == 0:
                    continue

                dist1 = abs(dy * x3 - dx * y3 + x2 * y1 - y2 * x1) / length
                dist2 = abs(dy * x4 - dx * y4 + x2 * y1 - y2 * x1) / length

                # 如果两个端点都接近当前直线，则认为共线
                if dist1 < 20 and dist2 < 20:
                    group.extend([(x3, y3), (x4, y4)])
                    used[j] = True

            # 找到最远的两个点作为合并后的线段
            if len(group) >= 2:
                points = np.array(group)
                # 计算主方向
                mean_point = np.mean(points, axis=0)
                centered = points - mean_point
                _, _, vt = np.linalg.svd(centered)
                direction = vt[0]

                # 投影到主方向
                projections = np.dot(centered, direction)
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)

                p1 = points[min_idx].astype(int)
                p2 = points[max_idx].astype(int)

                merged.append(np.array([[p1[0], p1[1], p2[0], p2[1]]]))

            used[i] = True

        return merged

    def check_color_uniformity(self, image: np.ndarray, line1: np.ndarray, line2: np.ndarray) -> Tuple[bool, np.ndarray]:
        """检查两条线之间的颜色是否统一（白色或黄橙色）"""
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # 创建掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # 提取区域颜色
        region = cv2.bitwise_and(image, image, mask=mask)
        pixels = region[mask > 0]

        if len(pixels) == 0:
            return False, np.array([0, 0, 0])

        # 计算平均颜色
        mean_color = np.mean(pixels, axis=0)

        # 转换到HSV空间判断
        hsv_color = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv_color

        # 放宽颜色判定条件
        # 白色判定：低饱和度，中高亮度
        is_white = s < 80 and v > 100

        # 黄橙色判定：色调在15-50之间
        is_yellow_orange = 15 <= h <= 50 and s > 30 and v > 80

        # 灰色地面也可能是车位（放宽条件）
        is_gray = s < 50 and v > 50

        return is_white or is_yellow_orange or is_gray, mean_color

    def is_in_ceiling_area(self, line: np.ndarray, image_height: int) -> bool:
        """判断线段是否主要在天花板区域（图片上方1/3）"""
        x1, y1, x2, y2 = line[0]
        ceiling_threshold = image_height / 3

        # 计算线段中点的y坐标
        mid_y = (y1 + y2) / 2

        # 如果中点在上方1/3区域，认为是天花板区域
        return mid_y < ceiling_threshold

    def calculate_overlap_ratio(self, line1: np.ndarray, line2: np.ndarray) -> float:
        """
        计算两条平行线的投影重叠比例
        用于判断两条线是否真正对应（而非错位的平行线）
        """
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # 计算线段的角度
        angle = self.calculate_angle(line1)
        angle_rad = np.radians(angle)

        # 将点投影到垂直于线段方向的轴上
        if abs(angle) < 45 or abs(angle) > 135:
            # 接近水平，使用x坐标
            proj1_min, proj1_max = min(x1, x2), max(x1, x2)
            proj2_min, proj2_max = min(x3, x4), max(x3, x4)
        else:
            # 接近垂直，使用y坐标
            proj1_min, proj1_max = min(y1, y2), max(y1, y2)
            proj2_min, proj2_max = min(y3, y4), max(y3, y4)

        # 计算重叠区间
        overlap_min = max(proj1_min, proj2_min)
        overlap_max = min(proj1_max, proj2_max)
        overlap = max(0, overlap_max - overlap_min)

        # 计算较短线段的长度
        len1 = proj1_max - proj1_min
        len2 = proj2_max - proj2_min
        min_len = min(len1, len2)

        if min_len == 0:
            return 0

        return overlap / min_len

    def check_length_similarity(self, line1: np.ndarray, line2: np.ndarray) -> bool:
        """检查两条线的长度是否相似（车位线两边长度应该接近）"""
        len1 = self.calculate_length(line1)
        len2 = self.calculate_length(line2)

        if len1 == 0 or len2 == 0:
            return False

        ratio = max(len1, len2) / min(len1, len2)
        return ratio <= self.max_length_ratio

    def is_reasonable_angle(self, line: np.ndarray) -> bool:
        """
        检查线段角度是否合理
        车位线通常不会是完全水平的（那可能是墙或天花板）
        """
        angle = abs(self.calculate_angle(line))
        # 归一化到0-90度
        if angle > 90:
            angle = 180 - angle

        # 排除接近水平的线（0-15度），这些可能是墙壁或天花板边缘
        # 排除接近垂直的线（75-90度），这些可能是柱子或墙角
        return 15 < angle < 75

    def check_line_straightness(self, line: np.ndarray, original_points: List[Tuple[int, int]]) -> float:
        """
        检查线段的直线度（用于验证合并后的线段质量）
        返回平均偏差距离
        """
        if len(original_points) < 3:
            return 0

        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return float('inf')

        total_deviation = 0
        for px, py in original_points:
            # 点到直线的距离
            deviation = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length
            total_deviation += deviation

        return total_deviation / len(original_points)

    def detect_parking_lines(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        检测车位线对

        Returns:
            List of (line1, line2, mean_color)
        """
        edges = self.preprocess(image)
        lines = self.detect_lines(edges)

        if len(lines) < 2:
            return []

        # 合并共线的线段
        merged_lines = self.merge_collinear_lines(lines)

        if len(merged_lines) < 2:
            return []

        parking_line_pairs = []
        h, w = image.shape[:2]
        min_length_ratio = 0.08  # 降低最小长度比例

        for i in range(len(merged_lines)):
            for j in range(i + 1, len(merged_lines)):
                line1, line2 = merged_lines[i], merged_lines[j]

                # 规则1: 检查是否在天花板区域，如果是则跳过
                if self.is_in_ceiling_area(line1, h) and self.is_in_ceiling_area(line2, h):
                    continue

                # 规则2: 检查长度
                len1 = self.calculate_length(line1)
                len2 = self.calculate_length(line2)
                if len1 < max(h, w) * min_length_ratio or len2 < max(h, w) * min_length_ratio:
                    continue

                # 规则3: 检查长度相似性（车位线两边应该长度接近）
                if not self.check_length_similarity(line1, line2):
                    continue

                # 规则4: 检查是否平行
                if not self.are_parallel(line1, line2):
                    continue

                # 规则5: 检查投影重叠比例（防止错位的平行线）
                overlap_ratio = self.calculate_overlap_ratio(line1, line2)
                if overlap_ratio < self.min_overlap_ratio:
                    continue

                # 规则6: 检查距离（车位宽度应该在合理范围内）
                distance = self.calculate_distance(line1, line2)
                if distance < self.min_distance or distance > self.max_distance:
                    continue

                # 规则7: 检查颜色（车位内部应该是地面颜色）
                is_valid_color, mean_color = self.check_color_uniformity(image, line1, line2)
                if not is_valid_color:
                    continue

                parking_line_pairs.append((line1, line2, mean_color))

        return parking_line_pairs

    def draw_results(self, image: np.ndarray, parking_line_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """绘制检测结果"""
        result = image.copy()

        for idx, (line1, line2, mean_color) in enumerate(parking_line_pairs):
            # 绘制线条
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]

            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(result, (x3, y3), (x4, y4), (0, 255, 0), 2)

            # 绘制连接线
            cv2.line(result, (x1, y1), (x3, y3), (255, 0, 0), 1)
            cv2.line(result, (x2, y2), (x4, y4), (255, 0, 0), 1)

            # 标注编号
            mid_x = int((x1 + x2 + x3 + x4) / 4)
            mid_y = int((y1 + y2 + y3 + y4) / 4)
            cv2.putText(result, f"#{idx+1}", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return result

    def get_debug_images(self, image: np.ndarray) -> dict:
        """获取调试图像"""
        debug_images = {}

        # 边缘检测结果
        edges = self.preprocess(image)
        debug_images['edges'] = edges

        # 检测到的所有线段
        lines = self.detect_lines(edges)
        all_lines_img = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(all_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        debug_images['all_lines'] = all_lines_img

        # 合并后的线段
        merged_lines = self.merge_collinear_lines(lines)
        merged_lines_img = image.copy()
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(merged_lines_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        debug_images['merged_lines'] = merged_lines_img

        return debug_images
