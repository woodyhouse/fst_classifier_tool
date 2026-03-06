"""
混合推理引擎 - 整合 CNN 分类器 + YOLO 检测 + 几何分析.

工作流程:
1. CNN 分类器 → slot_type, maneuver, marking, special_scene
2. YOLO 检测 → 物体位置和类别
3. 几何分析 → 车位线检测 + 9 宫格划分
4. 物体映射 → 将检测到的物体映射到 9 宫格
5. 输出 FST JSON
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image

from fst.models import (
    FSTOutput, Slot, Marking, SpecialScene, Obstacles,
    SlotType, Maneuver, LineColor, LineVisibility, LineStyle,
    SLOT_TYPE_CLASSES, MANEUVER_CLASSES, LINE_COLOR_CLASSES,
    LINE_VIS_CLASSES, LINE_STYLE_CLASSES, SPECIAL_SCENE_CLASSES,
)
from fst.parking_geometry import ParkingGeometry, GridCell
from fst.yolo_detector import YOLODetector, Detection


class HybridInferenceEngine:
    """
    混合推理引擎.

    Args:
        cnn_model_path: CNN 分类器模型路径 (ONNX 或 PyTorch)
        yolo_model_size: YOLO 模型大小 ('n', 's', 'm', 'l', 'x')
        device: 'cpu' 或 'cuda'
        img_size: CNN 输入图片尺寸
    """

    def __init__(
        self,
        cnn_model_path: str,
        yolo_model_size: str = "n",
        device: str = "cpu",
        img_size: int = 384,
    ):
        self.device = device
        self.img_size = img_size

        # 加载 CNN 分类器
        self.cnn_model = self._load_cnn_model(cnn_model_path)

        # 初始化 YOLO 检测器
        self.yolo_detector = YOLODetector(
            model_size=yolo_model_size,
            conf_threshold=0.25,
            device=device,
        )

        # 初始化几何分析器
        self.geometry = ParkingGeometry()

        # 图像预处理参数（ImageNet 标准）
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_cnn_model(self, model_path: str):
        """加载 CNN 模型（支持 ONNX 和 PyTorch）."""
        path = Path(model_path)

        if path.suffix == ".onnx":
            import onnxruntime as ort
            return ort.InferenceSession(str(path))
        elif path.suffix == ".pth":
            from fst.network import FSTClassifier
            model = FSTClassifier(pretrained=False)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        else:
            raise ValueError(f"Unsupported model format: {path.suffix}")

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图片用于 CNN."""
        # Resize
        img = cv2.resize(image, (self.img_size, self.img_size))

        # 归一化
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img).float()

    def _run_cnn_inference(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """运行 CNN 推理."""
        if isinstance(self.cnn_model, torch.nn.Module):
            # PyTorch 模型
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.cnn_model(image_tensor)
                return {k: v.cpu().numpy() for k, v in outputs.items()}
        else:
            # ONNX 模型
            input_name = self.cnn_model.get_inputs()[0].name
            outputs = self.cnn_model.run(None, {input_name: image_tensor.numpy()})
            # 需要根据 ONNX 输出顺序映射
            return {
                "slot_type": outputs[0],
                "maneuver": outputs[1],
                "special_scene": outputs[2],
                "line_color": outputs[3],
                "line_vis": outputs[4],
                "line_style": outputs[5],
            }

    def _parse_cnn_outputs(self, outputs: Dict[str, np.ndarray]) -> Dict:
        """解析 CNN 输出."""
        result = {}

        # slot_type
        slot_idx = int(np.argmax(outputs["slot_type"][0]))
        result["slot_type"] = SLOT_TYPE_CLASSES[slot_idx]

        # maneuver
        maneuver_idx = int(np.argmax(outputs["maneuver"][0]))
        result["maneuver"] = MANEUVER_CLASSES[maneuver_idx]

        # marking
        color_idx = int(np.argmax(outputs["line_color"][0]))
        vis_idx = int(np.argmax(outputs["line_vis"][0]))
        style_idx = int(np.argmax(outputs["line_style"][0]))

        result["marking"] = {
            "line_color": LINE_COLOR_CLASSES[color_idx],
            "line_visibility": LINE_VIS_CLASSES[vis_idx],
            "line_style": LINE_STYLE_CLASSES[style_idx],
        }

        # special_scene (multi-label)
        scene_logits = outputs["special_scene"][0]
        scene_probs = 1 / (1 + np.exp(-scene_logits))  # sigmoid
        scene_indices = np.where(scene_probs > 0.5)[0]

        p0_scenes = []
        p1_scenes = []
        for idx in scene_indices:
            scene_name = SPECIAL_SCENE_CLASSES[idx]
            if scene_name in ["DEAD_END", "NARROW_LANE"]:
                p0_scenes.append(scene_name)
            else:
                p1_scenes.append(scene_name)

        result["special_scene"] = {"P0": p0_scenes, "P1": p1_scenes}

        return result

    def _map_detections_to_grid(
        self,
        detections: List[Detection],
        grid_cells: Dict[str, GridCell],
    ) -> Dict[str, List[str]]:
        """将 YOLO 检测结果映射到 9 宫格."""
        pos_map: Dict[str, List[str]] = {
            pos: [] for pos in grid_cells.keys()
        }

        for det in detections:
            # 使用物体中心点映射
            grid_pos = self.geometry.map_point_to_grid(det.center, grid_cells)

            if grid_pos:
                fst_type = self.yolo_detector.map_to_fst_type(det.class_name)
                if fst_type not in pos_map[grid_pos]:
                    pos_map[grid_pos].append(fst_type)

        # 填充空位置
        for pos in pos_map:
            if not pos_map[pos]:
                pos_map[pos] = ["EMPTY"]

        return pos_map

    def infer(self, image: np.ndarray, image_id: Optional[str] = None) -> FSTOutput:
        """
        完整推理流程.

        Args:
            image: RGB 图片 [H, W, 3], uint8
            image_id: 图片 ID（可选）

        Returns:
            FSTOutput 结构化输出
        """
        # 1. CNN 分类推理
        image_tensor = self._preprocess_image(image)
        cnn_outputs = self._run_cnn_inference(image_tensor)
        cnn_results = self._parse_cnn_outputs(cnn_outputs)

        # 2. YOLO 物体检测
        detections = self.yolo_detector.detect(image)

        # 3. 几何分析 - 检测车位并划分 9 宫格
        parking_slot = self.geometry.find_parking_slot(image)
        if parking_slot is None:
            # 如果检测失败，使用默认布局
            h, w = image.shape[:2]
            from fst.parking_geometry import ParkingSlot
            parking_slot = ParkingSlot(
                corners=np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32),
                center=(w/2, h/2),
                width=w,
                height=h,
            )

        grid_cells = self.geometry.create_9grid(parking_slot)

        # 4. 物体映射到 9 宫格
        pos_map = self._map_detections_to_grid(detections, grid_cells)

        # 5. 构建 FSTOutput
        output = FSTOutput(
            schema_version="fst.v1",
            image_id=image_id,
            fst_level=3,
            search_direction="UNKNOWN",
            slot=Slot(
                slot_type=SlotType(cnn_results["slot_type"]),
                marking=Marking(
                    line_color=LineColor(cnn_results["marking"]["line_color"]),
                    line_visibility=LineVisibility(cnn_results["marking"]["line_visibility"]),
                    line_style=LineStyle(cnn_results["marking"]["line_style"]),
                ),
            ),
            maneuver=cnn_results["maneuver"],
            special_scene=SpecialScene(
                P0=cnn_results["special_scene"]["P0"],
                P1=cnn_results["special_scene"]["P1"],
            ),
            obstacles=Obstacles(pos_map=pos_map),
        )

        return output

    def visualize(
        self,
        image: np.ndarray,
        output: FSTOutput,
        detections: Optional[List[Detection]] = None,
    ) -> np.ndarray:
        """
        可视化推理结果（调试用）.

        Args:
            image: 原始图片
            output: 推理结果
            detections: YOLO 检测结果（可选）

        Returns:
            可视化图片
        """
        vis = image.copy()

        # 绘制 9 宫格
        parking_slot = self.geometry.find_parking_slot(image)
        if parking_slot:
            grid_cells = self.geometry.create_9grid(parking_slot)
            vis = self.geometry.visualize_grid(vis, grid_cells)

        # 绘制 YOLO 检测框
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(
                    vis,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0), 2,
                )
                cv2.putText(
                    vis,
                    f"{det.class_name} {det.confidence:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2,
                )

        return vis


if __name__ == "__main__":
    # 测试代码
    engine = HybridInferenceEngine(
        cnn_model_path="checkpoints/best_model.pth",
        yolo_model_size="n",
        device="cpu",
    )

    # 读取测试图片
    img = cv2.imread("test.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 推理
    result = engine.infer(img_rgb, image_id="test_001")

    # 打印结果
    print(result.model_dump_json(indent=2))

    # 可视化
    detections = engine.yolo_detector.detect(img_rgb)
    vis = engine.visualize(img_rgb, result, detections)
    cv2.imwrite("result_visualization.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
