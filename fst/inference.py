"""
ONNX 推理引擎: 图片 → FST 结构化 JSON.

支持 CPU / GPU，单张或批量推理。
不依赖 PyTorch，仅需 onnxruntime + Pillow + numpy。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Please install onnxruntime: pip install onnxruntime")

from fst.models import (
    SLOT_TYPE_CLASSES, MANEUVER_CLASSES, OBSTACLE_CLASSES,
    LINE_COLOR_CLASSES, LINE_VIS_CLASSES, LINE_STYLE_CLASSES,
    SPECIAL_SCENE_CLASSES, POSITION_KEYS,
    FSTOutput, Slot, Marking, Occupancy, SpecialScene, Obstacles, Token, Confidence,
    build_fst_text,
)


# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class FSTInference:
    """
    ONNX 推理引擎.

    Args:
        model_path: ONNX 模型文件路径
        img_size: 输入图片尺寸 (默认 384)
        device: "cpu" 或 "cuda" (需 onnxruntime-gpu)
        scene_threshold: special_scene 多标签阈值
    """

    def __init__(
        self,
        model_path: str | Path,
        img_size: int = 384,
        device: str = "cpu",
        scene_threshold: float = 0.5,
    ):
        self.img_size = img_size
        self.scene_threshold = scene_threshold

        # ONNX Runtime session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4

        self.session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)

        # 获取输出名
        self.output_names = [o.name for o in self.session.get_outputs()]
        print(f"Loaded ONNX model: {model_path}")
        print(f"  Outputs: {self.output_names}")
        print(f"  Provider: {self.session.get_providers()}")

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """PIL Image → [1, 3, H, W] float32 normalized array."""
        img = img.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
        arr = arr.transpose(2, 0, 1)  # [3, H, W]
        arr = arr[np.newaxis, ...]  # [1, 3, H, W]
        arr = (arr - MEAN) / STD
        return arr

    def predict_raw(self, img: Image.Image) -> Dict[str, np.ndarray]:
        """单张推理，返回原始 logits."""
        inp = self.preprocess(img)
        results = self.session.run(self.output_names, {"image": inp})
        return {name: result for name, result in zip(self.output_names, results)}

    def predict(self, img: Image.Image, image_id: Optional[str] = None) -> FSTOutput:
        """单张推理，返回 FSTOutput 对象."""
        t0 = time.time()
        raw = self.predict_raw(img)
        inference_ms = (time.time() - t0) * 1000

        # ── 解码各 head ──

        # slot_type
        slot_logits = raw["slot_type"][0]
        slot_idx = int(np.argmax(slot_logits))
        slot_conf = float(_softmax(slot_logits)[slot_idx])
        slot_type_str = SLOT_TYPE_CLASSES[slot_idx]

        # maneuver
        mv_logits = raw["maneuver"][0]
        mv_idx = int(np.argmax(mv_logits))
        mv_conf = float(_softmax(mv_logits)[mv_idx])
        maneuver_str = MANEUVER_CLASSES[mv_idx]

        # special_scene (multi-label)
        ss_logits = raw["special_scene"][0]
        ss_probs = _sigmoid(ss_logits)
        active_scenes = [SPECIAL_SCENE_CLASSES[i] for i, p in enumerate(ss_probs) if p > self.scene_threshold]
        p0_tags = [s for s in active_scenes if s in ("DEAD_END", "NARROW_LANE")]
        p1_tags = [s for s in active_scenes if s not in ("DEAD_END", "NARROW_LANE")]

        # obstacles (9 positions)
        pos_map: Dict[str, List[str]] = {}
        obs_confs: List[float] = []
        for i, pos_key in enumerate(POSITION_KEYS):
            obs_logits = raw[f"obs_{i}"][0]
            obs_idx = int(np.argmax(obs_logits))
            obs_conf = float(_softmax(obs_logits)[obs_idx])
            pos_map[pos_key] = [OBSTACLE_CLASSES[obs_idx]]
            obs_confs.append(obs_conf)

        # tokens (合并同类)
        tokens = _build_tokens(pos_map)

        # marking
        lc_logits = raw["line_color"][0]
        lc_idx = int(np.argmax(lc_logits))
        lc_conf = float(_softmax(lc_logits)[lc_idx])

        lv_logits = raw["line_vis"][0]
        lv_idx = int(np.argmax(lv_logits))

        ls_logits = raw["line_style"][0]
        ls_idx = int(np.argmax(ls_logits))

        # occupancy 推断（从 pos_7 / 整体判断）
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

        # fst_level 推断
        has_obs_info = any(pos_map[p][0] not in ("UNKNOWN",) for p in POSITION_KEYS)
        has_scene = len(p0_tags) > 0 or len(p1_tags) > 0
        if has_obs_info:
            fst_level = 3
        elif has_scene:
            fst_level = 2
        else:
            fst_level = 1

        # 置信度
        overall_conf = np.mean([slot_conf, mv_conf, lc_conf, np.mean(obs_confs)])

        output = FSTOutput(
            image_id=image_id,
            raw_description=f"[auto] inference_time={inference_ms:.1f}ms",
            fst_level=fst_level,
            search_direction="UNKNOWN",  # 需要额外 meta 信息
            slot=Slot(
                slot_type=slot_type_str,
                marking=Marking(
                    line_color=LINE_COLOR_CLASSES[lc_idx],
                    line_visibility=LINE_VIS_CLASSES[lv_idx],
                    line_style=LINE_STYLE_CLASSES[ls_idx],
                ),
                occupancy=Occupancy(status=occ_status, occupied_by=occ_by),
            ),
            maneuver=maneuver_str,
            special_scene=SpecialScene(P0=p0_tags, P1=p1_tags),
            obstacles=Obstacles(pos_map=pos_map, tokens=tokens),
            confidence=Confidence(
                overall=round(float(overall_conf), 3),
                slot_type=round(slot_conf, 3),
                marking=round(lc_conf, 3),
                occupancy=round(float(np.mean(obs_confs)), 3),
                obstacles=round(float(np.mean(obs_confs)), 3),
            ),
        )

        return output

    def predict_with_text(self, img: Image.Image, image_id: Optional[str] = None) -> Dict:
        """返回 JSON dict + fst_text DSL."""
        output = self.predict(img, image_id)
        d = output.model_dump()
        d["fst_text"] = build_fst_text(output)
        return d


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_tokens(pos_map: Dict[str, List[str]]) -> List[Token]:
    """合并同类障碍物为 tokens."""
    by_type: Dict[str, List[str]] = {}
    for pos, items in pos_map.items():
        t = items[0] if items else "UNKNOWN"
        by_type.setdefault(t, []).append(pos)
    tokens = []
    for t, positions in sorted(by_type.items()):
        tokens.append(Token(positions=positions, type=t))
    return tokens


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m fst.inference <model.onnx> <image.jpg>")
        sys.exit(1)

    engine = FSTInference(sys.argv[1])
    img = Image.open(sys.argv[2])
    result = engine.predict_with_text(img, image_id=Path(sys.argv[2]).stem)
    print(json.dumps(result, indent=2, ensure_ascii=False))
