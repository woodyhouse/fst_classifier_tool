"""
YOLO 物体检测模块 - 用于识别车位周围的障碍物.

使用预训练的 YOLOv8 模型检测:
- 车辆 (car, truck, bus, motorcycle)
- 人 (person)
- 其他常见物体

依赖: ultralytics (pip install ultralytics)
"""
from __future__ import annotations

import importlib
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from fst.paths import MODELS_DIR


@dataclass
class Detection:
    """单个检测结果."""
    class_name: str      # COCO 类别名
    confidence: float    # 置信度 [0, 1]
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) 像素坐标
    center: Tuple[float, float]  # (cx, cy) 中心点


# COCO 类别到 FST ObstacleType 的映射
COCO_TO_FST = {
    "car": "VEHICLE",
    "truck": "VEHICLE",
    "bus": "VEHICLE",
    "motorcycle": "VEHICLE",
    "bicycle": "VEHICLE",
    "person": "EMPTY",  # 人通常是临时的，不算障碍物
    "traffic light": "LAMP",
    "fire hydrant": "FIRE_BOX_SUSPENDED",
    "stop sign": "CONE",
    "parking meter": "THIN_POLE",
    "bench": "EMPTY",
    "potted plant": "BUSH",
}


class YOLODetector:
    """
    YOLO 物体检测器封装.

    Args:
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        conf_threshold: 置信度阈值
        device: 'cpu' 或 'cuda'
    """

    def __init__(
        self,
        model_size: str = "n",  # nano 最快
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:
            raise ImportError("ultralytics not available. Run: pip install ultralytics") from exc

        # Prefer local weights to avoid runtime download in restricted environments.
        candidates = [
            MODELS_DIR / f"yolov8{model_size}.pt",
            Path(f"models/yolov8{model_size}.pt"),
            Path(f"yolov8{model_size}.pt"),
        ]
        weight_path = next((p for p in candidates if p.exists()), Path(f"yolov8{model_size}.pt"))
        self.model = YOLO(str(weight_path))
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        检测图片中的物体.

        Args:
            image: RGB 图片 [H, W, 3], uint8

        Returns:
            检测结果列表
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]

                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    center=(float(cx), float(cy)),
                ))

        return detections

    def map_to_fst_type(self, coco_class: str) -> str:
        """将 COCO 类别映射到 FST ObstacleType."""
        return COCO_TO_FST.get(coco_class, "UNKNOWN")

    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: float = 0.3,
        allowed_classes: Optional[List[str]] = None,
        min_size: Optional[Tuple[float, float]] = None,
    ) -> List[Detection]:
        """
        过滤检测结果.

        Args:
            detections: 原始检测结果
            min_confidence: 最小置信度
            allowed_classes: 允许的类别列表（None 表示全部允许）
            min_size: 最小尺寸 (width, height)，过滤掉太小的检测框

        Returns:
            过滤后的检测结果
        """
        filtered = []

        for det in detections:
            # 置信度过滤
            if det.confidence < min_confidence:
                continue

            # 类别过滤
            if allowed_classes and det.class_name not in allowed_classes:
                continue

            # 尺寸过滤
            if min_size:
                x1, y1, x2, y2 = det.bbox
                w = x2 - x1
                h = y2 - y1
                if w < min_size[0] or h < min_size[1]:
                    continue

            filtered.append(det)

        return filtered

    def print_detections(self, detections: List[Detection]):
        """打印检测结果（调试用）."""
        print(f"\n检测到 {len(detections)} 个物体:")
        print("-" * 80)
        for i, det in enumerate(detections, 1):
            fst_type = self.map_to_fst_type(det.class_name)
            print(f"{i:2d}. {det.class_name:15s} → {fst_type:15s} "
                  f"(conf: {det.confidence:.3f}, center: {det.center[0]:.0f}, {det.center[1]:.0f})")
        print("-" * 80)

    def visualize(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_fst_type: bool = True,
    ) -> np.ndarray:
        """
        可视化检测结果.

        Args:
            image: 原始图片
            detections: 检测结果
            show_fst_type: 是否显示 FST 类型映射

        Returns:
            可视化图片
        """
        import cv2
        vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 标签文本
            if show_fst_type:
                fst_type = self.map_to_fst_type(det.class_name)
                label = f"{det.class_name} → {fst_type} ({det.confidence:.2f})"
            else:
                label = f"{det.class_name} ({det.confidence:.2f})"

            # 绘制标签背景
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis,
                (x1, y1 - text_h - 4),
                (x1 + text_w, y1),
                color, -1,
            )

            # 绘制标签文字
            cv2.putText(
                vis, label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1,
            )

            # 绘制中心点
            cx, cy = det.center
            cv2.circle(vis, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        return vis


if __name__ == "__main__":
    # 测试代码
    import cv2

    detector = YOLODetector(model_size="n")

    # 读取测试图片
    img = cv2.imread("test.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detector.detect(img_rgb)

    print(f"检测到 {len(detections)} 个物体:")
    for det in detections:
        print(f"  {det.class_name} ({det.confidence:.2f}) at {det.center}")
