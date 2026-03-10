import math
import sys
from collections import namedtuple
from enum import Enum
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import torch

from fst.paths import EXTERNAL_DIR, PROJECT_ROOT, TEST_RESULTS_TEMPLATE_DIR

DMPR_DIR = EXTERNAL_DIR / "DMPR-PS"
sys.path.insert(0, str(DMPR_DIR))

import config  # noqa: E402
from model.detector import DirectionalPointDetector  # noqa: E402


MarkingPoint = namedtuple("MarkingPoint", ["x", "y", "direction", "shape"])


class PointShape(Enum):
    none = 0
    l_down = 1
    t_down = 2
    t_middle = 3
    t_up = 4
    l_up = 5


def direction_diff(direction_a, direction_b):
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2 * math.pi - diff


def detemine_point_shape(point, vector):
    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    if point.shape < 0.5:
        if direction_diff(vec_direct, point.direction) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.t_middle
        if direction_diff(vec_direct_up, point.direction) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.t_up
        if direction_diff(vec_direct_down, point.direction) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.t_down
    else:
        if direction_diff(vec_direct, point.direction) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.l_down
        if direction_diff(vec_direct_up, point.direction) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.l_up
    return PointShape.none


def non_maximum_suppression(pred_points):
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            i_x, i_y = pred_points[i][1].x, pred_points[i][1].y
            j_x, j_y = pred_points[j][1].x, pred_points[j][1].y
            if abs(j_x - i_x) < 0.0625 and abs(j_y - i_y) < 0.0625:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        return [pred_points[i] for i, sup in enumerate(suppressed) if not sup]
    return pred_points


def get_predicted_points(prediction, thresh):
    pred_points = []
    pred = prediction.detach().cpu().numpy()
    for i in range(pred.shape[1]):
        for j in range(pred.shape[2]):
            if pred[0, i, j] < thresh:
                continue
            xval = (j + pred[2, i, j]) / pred.shape[2]
            yval = (i + pred[3, i, j]) / pred.shape[1]
            if not (
                config.BOUNDARY_THRESH <= xval <= 1 - config.BOUNDARY_THRESH
                and config.BOUNDARY_THRESH <= yval <= 1 - config.BOUNDARY_THRESH
            ):
                continue
            direction = math.atan2(pred[5, i, j], pred[4, i, j])
            pred_points.append((pred[0, i, j], MarkingPoint(xval, yval, direction, pred[1, i, j])))
    return non_maximum_suppression(pred_points)


def calc_point_squre_dist(point_a, point_b):
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    return distx**2 + disty**2


def pass_through_third_point(marking_points, i, j):
    x1, y1 = marking_points[i].x, marking_points[i].y
    x2, y2 = marking_points[j].x, marking_points[j].y
    for idx, point in enumerate(marking_points):
        if idx in (i, j):
            continue
        x0, y0 = point.x, point.y
        vec1 = np.array([x0 - x1, y0 - y1], dtype=np.float32)
        vec2 = np.array([x2 - x0, y2 - y0], dtype=np.float32)
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        vec1 /= n1
        vec2 /= n2
        if float(np.dot(vec1, vec2)) > config.SLOT_SUPPRESSION_DOT_PRODUCT_THRESH:
            return True
    return False


def pair_marking_points(point_a, point_b):
    vector_ab = np.array([point_b.x - point_a.x, point_b.y - point_a.y], dtype=np.float32)
    n = np.linalg.norm(vector_ab)
    if n < 1e-6:
        return 0
    vector_ab /= n
    shape_a = detemine_point_shape(point_a, vector_ab)
    shape_b = detemine_point_shape(point_b, -vector_ab)
    if shape_a.value == 0 or shape_b.value == 0:
        return 0
    if shape_a.value == 3 and shape_b.value == 3:
        return 0
    if shape_a.value > 3 and shape_b.value > 3:
        return 0
    if shape_a.value < 3 and shape_b.value < 3:
        return 0
    if shape_a.value != 3:
        return 1 if shape_a.value > 3 else -1
    if shape_b.value < 3:
        return 1
    if shape_b.value > 3:
        return -1
    return 0


class FastDMPR:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model = DirectionalPointDetector(input_channel_size=3, depth_factor=32, output_channel_size=6)
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image_bgr, thresh=0.35):
        h, w = image_bgr.shape[:2]
        img512 = cv2.resize(image_bgr, (512, 512))
        x = torch.from_numpy(img512.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x)
        pred_points = get_predicted_points(pred[0], thresh)
        marking = [point for _, point in pred_points]

        slots = []
        for i in range(len(marking) - 1):
            for j in range(i + 1, len(marking)):
                pa, pb = marking[i], marking[j]
                dist = calc_point_squre_dist(pa, pb)
                if not (
                    config.VSLOT_MIN_DIST <= dist <= config.VSLOT_MAX_DIST
                    or config.HSLOT_MIN_DIST <= dist <= config.HSLOT_MAX_DIST
                ):
                    continue
                if pass_through_third_point(marking, i, j):
                    continue
                paired = pair_marking_points(pa, pb)
                if paired == 1:
                    slots.append((i, j))
                elif paired == -1:
                    slots.append((j, i))

        quads = []
        for i, j in slots:
            pa, pb = marking[i], marking[j]
            p0 = np.array([pa.x * w, pa.y * h], dtype=np.float32)
            p1 = np.array([pb.x * w, pb.y * h], dtype=np.float32)
            v = p1 - p0
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            v /= nv

            dist = calc_point_squre_dist(pa, pb)
            if config.VSLOT_MIN_DIST <= dist <= config.VSLOT_MAX_DIST:
                sep = config.LONG_SEPARATOR_LENGTH
            else:
                sep = config.SHORT_SEPARATOR_LENGTH

            p2 = np.array([p0[0] + h * sep * v[1], p0[1] - w * sep * v[0]], dtype=np.float32)
            p3 = np.array([p1[0] + h * sep * v[1], p1[1] - w * sep * v[0]], dtype=np.float32)
            quads.append(np.array([p0, p1, p3, p2], dtype=np.float32))

        points = [(p.x * w, p.y * h) for p in marking]
        return points, quads


def safe_imread(path: Path):
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def main():
    detector = FastDMPR(str(PROJECT_ROOT / "weights" / "dmpr_ps.pth"), device="cpu")
    out_dir = TEST_RESULTS_TEMPLATE_DIR / "fast_dmpr_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "image136.jpeg",
        "image353.jpeg",
        "image48.jpeg",
        "image135.jpeg",
        "image454.jpeg",
        "image430.jpeg",
        "image16.jpeg",
        "corrected_20260215_221708_7f45f0.jpg",
    ]
    for name in names:
        img_path = PROJECT_ROOT / "dataset/images quality" / name
        if not img_path.exists():
            img_path = PROJECT_ROOT / "dataset/images" / name
        img = safe_imread(img_path)
        if img is None:
            print("read fail:", name)
            continue
        points, quads = detector.detect(img, thresh=0.35)
        print(name, "points=", len(points), "quads=", len(quads))

        vis = img.copy()
        for x, y in points:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 255), -1)
        for quad in quads[:3]:
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / name), vis)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
