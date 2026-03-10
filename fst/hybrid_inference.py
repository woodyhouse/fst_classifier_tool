"""
Hybrid inference wrapper for simplified architecture.

This module keeps the historical API name `HybridInferenceEngine` so existing
CLI/examples continue to work, while internally using:
  - CNN heads for scene attributes
  - YOLO + rule-based grid mapping for obstacles
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from fst.inference import FSTInference
from fst.models import FSTOutput
from fst.paths import CHECKPOINTS_DIR, SAMPLE_GROUND_INPUT_PATH
from fst.yolo_detector import Detection


class HybridInferenceEngine:
    """
    Compatibility wrapper used by examples/CLI.

    Args:
        cnn_model_path: path to .onnx or .pth model
        yolo_model_size: YOLO model size n/s/m/l/x
        device: cpu/cuda
        img_size: CNN input size
    """

    def __init__(
        self,
        cnn_model_path: str,
        yolo_model_size: str = "n",
        device: str = "cpu",
        img_size: int = 384,
        use_yolo: bool = True,
    ):
        self.engine = FSTInference(
            model_path=cnn_model_path,
            img_size=img_size,
            device=device,
            use_yolo=use_yolo,
            yolo_model_size=yolo_model_size,
        )
        self.yolo_detector = self.engine.yolo_detector

    def infer(self, image: np.ndarray, image_id: Optional[str] = None) -> FSTOutput:
        """
        Run inference on RGB image.

        Args:
            image: RGB image as uint8 array [H, W, 3]
            image_id: optional ID
        """
        pil = Image.fromarray(image.astype(np.uint8))
        return self.engine.predict(pil, image_id=image_id)

    def visualize(
        self,
        image: np.ndarray,
        output: FSTOutput,
        detections: Optional[List[Detection]] = None,
    ) -> np.ndarray:
        """
        Visualize detections + simple 3x3 grid with mapped obstacle text.
        """
        import cv2
        vis = image.copy()
        h, w = vis.shape[:2]

        # Draw YOLO boxes.
        if detections is None and self.yolo_detector is not None:
            detections = self.yolo_detector.detect(image)
        if detections:
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                fst_type = self.yolo_detector.map_to_fst_type(det.class_name) if self.yolo_detector else "UNKNOWN"
                cv2.putText(
                    vis,
                    f"{det.class_name}->{fst_type} {det.confidence:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                )

        # Draw simple 3x3 grid.
        grid_h = h // 3
        grid_w = w // 3
        for i in (1, 2):
            cv2.line(vis, (0, i * grid_h), (w, i * grid_h), (0, 255, 0), 1)
            cv2.line(vis, (i * grid_w, 0), (i * grid_w, h), (0, 255, 0), 1)

        pos_text_anchor = {
            "1": (grid_w // 2, grid_h // 2),
            "2": (grid_w + grid_w // 2, grid_h // 2),
            "3": (2 * grid_w + grid_w // 2, grid_h // 2),
            "4": (grid_w // 2, grid_h + grid_h // 2),
            "5": (grid_w + grid_w // 2, grid_h + grid_h // 2),
            "6": (2 * grid_w + grid_w // 2, grid_h + grid_h // 2),
            "7": (w // 2, 2 * grid_h + grid_h // 2),
            "P_LEFT": (grid_w // 3, int(h * 0.78)),
            "P_RIGHT": (w - grid_w // 2, int(h * 0.78)),
        }
        for pos, anchor in pos_text_anchor.items():
            items = output.obstacles.pos_map.get(pos, ["UNKNOWN"])
            text = items[0] if items else "UNKNOWN"
            cv2.putText(
                vis,
                f"{pos}:{text}",
                (int(anchor[0]) - 45, int(anchor[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        # Draw key summary.
        cv2.putText(
            vis,
            f"slot={output.slot.slot_type.value}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        maneuver = output.maneuver if isinstance(output.maneuver, str) else output.maneuver.value
        cv2.putText(
            vis,
            f"maneuver={maneuver}",
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return vis


if __name__ == "__main__":
    import cv2
    engine = HybridInferenceEngine(
        cnn_model_path=str(CHECKPOINTS_DIR / "best_model.pth"),
        yolo_model_size="n",
        device="cpu",
    )
    test = cv2.imread(str(SAMPLE_GROUND_INPUT_PATH))
    rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    result = engine.infer(rgb, image_id="demo")
    print(result.model_dump_json(indent=2))
