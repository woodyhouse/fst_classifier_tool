"""
Unified inference engine for simplified architecture:
CNN (6 heads) + YOLO obstacle mapping.
"""
from __future__ import annotations

import json
import time
import importlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from fst.models import (
    SLOT_TYPE_CLASSES, MANEUVER_CLASSES, OBSTACLE_CLASSES,
    LINE_COLOR_CLASSES, LINE_VIS_CLASSES, LINE_STYLE_CLASSES,
    SPECIAL_SCENE_CLASSES, POSITION_KEYS,
    FSTOutput, Slot, Marking, Occupancy, SpecialScene, Obstacles, Token, Confidence,
    build_fst_text,
)
from fst.yolo_detector import YOLODetector


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class FSTInference:
    """
    Supports ONNX/PyTorch CNN heads and uses YOLO for obstacle mapping by default.

    Args:
        model_path: .onnx or .pth model path
        img_size: CNN input image size
        device: "cpu" or "cuda"
        scene_threshold: threshold for special_scene multi-label
        use_yolo: whether to use YOLO mapping when obstacle heads are absent
        yolo_model_size: YOLO model variant, one of n/s/m/l/x
        yolo_conf_threshold: YOLO confidence threshold
    """

    def __init__(
        self,
        model_path: str | Path,
        img_size: int = 384,
        device: str = "cpu",
        scene_threshold: float = 0.5,
        use_yolo: bool = True,
        yolo_model_size: str = "n",
        yolo_conf_threshold: float = 0.25,
    ):
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.scene_threshold = scene_threshold
        self.device = device
        self.runtime: str = ""

        self.session = None
        self.model = None
        self.output_names: List[str] = []
        self.has_obstacle_heads = False

        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            try:
                ort = importlib.import_module("onnxruntime")
            except Exception as exc:
                raise ImportError(f"onnxruntime is required for .onnx inference: {exc}") from exc
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.runtime = "onnx"
            print(f"Loaded ONNX model: {self.model_path}")
            print(f"  Outputs: {self.output_names}")
            print(f"  Provider: {self.session.get_providers()}")
        elif suffix == ".pth":
            if not TORCH_AVAILABLE:
                raise ImportError("torch is required for .pth inference.")
            from fst.network import FSTClassifier
            self.model = FSTClassifier(pretrained=False)
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
            load_result = self.model.load_state_dict(state_dict, strict=False)
            if load_result.unexpected_keys:
                print(f"Warning: ignored {len(load_result.unexpected_keys)} unexpected keys.")
            if load_result.missing_keys:
                print(f"Warning: missing {len(load_result.missing_keys)} keys when loading checkpoint.")
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            self.device = target_device
            self.model.to(self.device)
            self.model.eval()
            self.output_names = ["slot_type", "maneuver", "special_scene", "line_color", "line_vis", "line_style"]
            self.runtime = "torch"
            print(f"Loaded PyTorch model: {self.model_path}")
            print(f"  Device: {self.device}")
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

        self.has_obstacle_heads = all(f"obs_{i}" in self.output_names for i in range(len(POSITION_KEYS)))

        self.yolo_detector = None
        if use_yolo:
            try:
                self.yolo_detector = YOLODetector(
                    model_size=yolo_model_size,
                    conf_threshold=yolo_conf_threshold,
                    device=self.device,
                )
            except Exception as exc:
                print(f"Warning: YOLO disabled because initialization failed: {exc}")

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """PIL image to normalized NCHW float32 array."""
        img = img.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = arr[np.newaxis, ...]
        arr = (arr - MEAN) / STD
        return arr

    def predict_raw(self, img: Image.Image) -> Dict[str, np.ndarray]:
        """Run CNN inference and return raw logits by head name."""
        inp = self.preprocess(img)
        if self.runtime == "onnx":
            assert self.session is not None
            results = self.session.run(self.output_names, {"image": inp})
            return {name: result for name, result in zip(self.output_names, results)}

        assert self.model is not None and TORCH_AVAILABLE
        with torch.no_grad():
            tensor = torch.from_numpy(inp).to(self.device)
            outputs = self.model(tensor)
        return {k: v.detach().cpu().numpy() for k, v in outputs.items()}

    def predict(self, img: Image.Image, image_id: Optional[str] = None) -> FSTOutput:
        """Run full inference and return structured FST output."""
        t0 = time.time()
        raw = self.predict_raw(img)
        inference_ms = (time.time() - t0) * 1000

        slot_logits = raw["slot_type"][0]
        slot_idx = int(np.argmax(slot_logits))
        slot_conf = float(_softmax(slot_logits)[slot_idx])

        mv_logits = raw["maneuver"][0]
        mv_idx = int(np.argmax(mv_logits))
        mv_conf = float(_softmax(mv_logits)[mv_idx])

        ss_logits = raw["special_scene"][0]
        ss_probs = _sigmoid(ss_logits)
        active_scenes = [SPECIAL_SCENE_CLASSES[i] for i, p in enumerate(ss_probs) if p > self.scene_threshold]
        p0_tags = [s for s in active_scenes if s in ("DEAD_END", "NARROW_LANE")]
        p1_tags = [s for s in active_scenes if s not in ("DEAD_END", "NARROW_LANE")]

        lc_logits = raw["line_color"][0]
        lv_logits = raw["line_vis"][0]
        ls_logits = raw["line_style"][0]
        lc_idx = int(np.argmax(lc_logits))
        lv_idx = int(np.argmax(lv_logits))
        ls_idx = int(np.argmax(ls_logits))
        lc_conf = float(_softmax(lc_logits)[lc_idx])
        lv_conf = float(_softmax(lv_logits)[lv_idx])
        ls_conf = float(_softmax(ls_logits)[ls_idx])
        marking_conf = float(np.mean([lc_conf, lv_conf, ls_conf]))

        if self.has_obstacle_heads and all(f"obs_{i}" in raw for i in range(len(POSITION_KEYS))):
            pos_map, obs_conf = self._decode_obstacles_from_heads(raw)
            obstacle_source = "cnn_obs_heads"
        else:
            pos_map, obs_conf = self._decode_obstacles_with_yolo(np.asarray(img.convert("RGB")))
            obstacle_source = "yolo_grid_mapping"

        tokens = _build_tokens(pos_map)

        pos7_obs = pos_map.get("7", ["UNKNOWN"])[0]
        if pos7_obs in ("VEHICLE", "CONE", "WATER_BARRIER"):
            occ_status = "OCCUPIED"
            occ_by = [pos7_obs]
        elif pos7_obs == "EMPTY":
            occ_status = "FREE"
            occ_by = []
        else:
            occ_status = "UNKNOWN"
            occ_by = []

        has_scene = bool(p0_tags or p1_tags)
        has_obs_signal = any(pos_map[p][0] not in ("UNKNOWN",) for p in POSITION_KEYS)
        fst_level = 3 if has_obs_signal else (2 if has_scene else 1)

        overall_conf = float(np.mean([slot_conf, mv_conf, marking_conf, obs_conf]))

        output = FSTOutput(
            image_id=image_id,
            raw_description=f"[auto] source={obstacle_source}; inference_time={inference_ms:.1f}ms",
            fst_level=fst_level,
            search_direction="UNKNOWN",
            slot=Slot(
                slot_type=SLOT_TYPE_CLASSES[slot_idx],
                marking=Marking(
                    line_color=LINE_COLOR_CLASSES[lc_idx],
                    line_visibility=LINE_VIS_CLASSES[lv_idx],
                    line_style=LINE_STYLE_CLASSES[ls_idx],
                ),
                occupancy=Occupancy(status=occ_status, occupied_by=occ_by),
            ),
            maneuver=MANEUVER_CLASSES[mv_idx],
            special_scene=SpecialScene(P0=p0_tags, P1=p1_tags),
            obstacles=Obstacles(pos_map=pos_map, tokens=tokens),
            confidence=Confidence(
                overall=round(overall_conf, 3),
                slot_type=round(slot_conf, 3),
                marking=round(marking_conf, 3),
                occupancy=round(obs_conf, 3),
                obstacles=round(obs_conf, 3),
            ),
        )
        return output

    def predict_with_text(self, img: Image.Image, image_id: Optional[str] = None) -> Dict:
        """Return dict output with generated fst_text."""
        output = self.predict(img, image_id=image_id)
        result = output.model_dump(mode="json")
        result["fst_text"] = build_fst_text(output)
        return result

    def _decode_obstacles_from_heads(self, raw: Dict[str, np.ndarray]) -> tuple[Dict[str, List[str]], float]:
        pos_map: Dict[str, List[str]] = {}
        confs: List[float] = []
        for i, pos_key in enumerate(POSITION_KEYS):
            logits = raw[f"obs_{i}"][0]
            idx = int(np.argmax(logits))
            conf = float(_softmax(logits)[idx])
            pos_map[pos_key] = [OBSTACLE_CLASSES[idx]]
            confs.append(conf)
        return pos_map, float(np.mean(confs)) if confs else 0.5

    def _decode_obstacles_with_yolo(self, image_rgb: np.ndarray) -> tuple[Dict[str, List[str]], float]:
        if self.yolo_detector is None:
            empty = {pos: ["UNKNOWN"] for pos in POSITION_KEYS}
            return empty, 0.5

        detections = self.yolo_detector.detect(image_rgb)
        by_type: Dict[str, List[tuple[float, float, float, float]]] = {}
        confs: List[float] = []

        for det in detections:
            fst_type = self.yolo_detector.map_to_fst_type(det.class_name)
            if fst_type in ("EMPTY", "UNKNOWN"):
                continue
            by_type.setdefault(fst_type, []).append(det.bbox)
            confs.append(det.confidence)

        h, w = image_rgb.shape[:2]
        pos_map = _simple_grid_mapping(by_type, (h, w))
        # Ensure full position coverage.
        for pos in POSITION_KEYS:
            pos_map.setdefault(pos, ["EMPTY"])
        return pos_map, float(np.mean(confs)) if confs else 0.5


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_tokens(pos_map: Dict[str, List[str]]) -> List[Token]:
    by_type: Dict[str, List[str]] = {}
    for pos, items in pos_map.items():
        t = items[0] if items else "UNKNOWN"
        by_type.setdefault(t, []).append(pos)
    return [Token(positions=positions, type=t) for t, positions in sorted(by_type.items())]


def _simple_grid_mapping(
    obstacles: Dict[str, List[tuple[float, float, float, float]]],
    image_shape: tuple[int, int],
) -> Dict[str, List[str]]:
    """Map obstacle bboxes to simplified 9-grid positions."""
    h, w = image_shape
    grid = {str(i): ["EMPTY"] for i in range(1, 8)}
    grid["P_LEFT"] = ["EMPTY"]
    grid["P_RIGHT"] = ["EMPTY"]

    grid_regions = {
        "1": (0, 0, w // 3, h // 3),
        "2": (w // 3, 0, 2 * w // 3, h // 3),
        "3": (2 * w // 3, 0, w, h // 3),
        "4": (0, h // 3, w // 3, 2 * h // 3),
        "5": (w // 3, h // 3, 2 * w // 3, 2 * h // 3),
        "6": (2 * w // 3, h // 3, w, 2 * h // 3),
        "7": (0, 2 * h // 3, w, h),
        "P_LEFT": (0, h // 3, w // 3, h),
        "P_RIGHT": (2 * w // 3, h // 3, w, h),
    }

    for obj_type, bboxes in obstacles.items():
        for x1, y1, x2, y2 in bboxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            for grid_id, (gx1, gy1, gx2, gy2) in grid_regions.items():
                if gx1 <= cx < gx2 and gy1 <= cy < gy2:
                    if grid[grid_id] == ["EMPTY"]:
                        grid[grid_id] = [obj_type]
                    elif obj_type not in grid[grid_id]:
                        grid[grid_id].append(obj_type)
                    break
    return grid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FST unified inference")
    parser.add_argument("model", help="Path to .onnx or .pth model")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--yolo", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO obstacle mapping")
    args = parser.parse_args()

    engine = FSTInference(
        args.model,
        img_size=args.img_size,
        device=args.device,
        use_yolo=not args.no_yolo,
        yolo_model_size=args.yolo,
    )
    image = Image.open(args.image)
    output = engine.predict_with_text(image, image_id=Path(args.image).stem)
    print(json.dumps(output, indent=2, ensure_ascii=False))
