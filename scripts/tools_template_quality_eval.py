from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from fst.parking_line_detector import ParkingLineDetector
from fst.paths import DEFAULT_TEMPLATE_PATH, PROJECT_ROOT, TEST_RESULTS_TEMPLATE_DIR
from fst.template_parking_detector import ParkingMatch, TemplateParkingDetector


@dataclass
class EvalRow:
    file_name: str
    status: str
    score: float
    area_ratio: float
    top_y_ratio: float
    bottom_y_ratio: float
    side_left_angle: float
    side_right_angle: float
    side_angle_gap: float
    side_same_sign: int
    merged_lines: int


def safe_imread(path: Path) -> Optional[np.ndarray]:
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


def normalize_lines(detector: ParkingLineDetector, lines: List[np.ndarray]) -> List[np.ndarray]:
    if hasattr(detector, "merge_collinear_lines"):
        merged = detector.merge_collinear_lines(lines)
    elif hasattr(detector, "merge_lines"):
        merged = detector.merge_lines(lines)
    else:
        merged = lines if lines is not None else []

    out: List[np.ndarray] = []
    for line in merged:
        arr = np.asarray(line)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.size >= 4:
            out.append(arr.astype(np.float32).reshape(-1)[:4])
    return out


def _line_angle_signed(p0: np.ndarray, p1: np.ndarray) -> float:
    ang = float(np.rad2deg(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
    while ang <= -90.0:
        ang += 180.0
    while ang > 90.0:
        ang -= 180.0
    return ang


def quad_metrics(match: ParkingMatch, image_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float, float, float, int]:
    h, w = image_shape[:2]
    q = match.outer_quad.astype(np.float32)
    area_ratio = abs(float(cv2.contourArea(q))) / max(float(h * w), 1.0)
    top_y_ratio = float(((q[2, 1] + q[3, 1]) * 0.5) / max(float(h), 1.0))
    bottom_y_ratio = float(((q[0, 1] + q[1, 1]) * 0.5) / max(float(h), 1.0))

    left_ang = _line_angle_signed(q[0], q[3])
    right_ang = _line_angle_signed(q[1], q[2])
    side_gap = abs(left_ang - right_ang)
    same_sign = int(np.sign(left_ang) == np.sign(right_ang) and left_ang != 0.0 and right_ang != 0.0)
    return area_ratio, top_y_ratio, bottom_y_ratio, left_ang, right_ang, side_gap, same_sign


def eval_one(
    image_path: Path,
    line_detector: ParkingLineDetector,
    template_detector: TemplateParkingDetector,
    score_threshold: float,
    vis_dir: Path,
) -> EvalRow:
    image = safe_imread(image_path)
    if image is None:
        return EvalRow(
            file_name=image_path.name,
            status="READ_FAIL",
            score=0.0,
            area_ratio=0.0,
            top_y_ratio=0.0,
            bottom_y_ratio=0.0,
            side_left_angle=0.0,
            side_right_angle=0.0,
            side_angle_gap=0.0,
            side_same_sign=0,
            merged_lines=0,
        )

    edges = line_detector.preprocess(image)
    lines = line_detector.detect_lines(edges)
    merged = normalize_lines(line_detector, lines)

    matches = template_detector.detect(image, merged, score_threshold=score_threshold)
    out_path = vis_dir / image_path.name
    vis = image.copy()
    for m in matches:
        outer = m.outer_quad.astype(np.int32)
        inner = m.inner_quad.astype(np.int32)
        cv2.polylines(vis, [outer], True, (0, 255, 0), 2)
        cv2.polylines(vis, [inner], True, (255, 0, 0), 2)
        center = np.mean(m.outer_quad, axis=0).astype(np.int32)
        cv2.putText(
            vis,
            f"{m.score:.2f}",
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    cv2.imwrite(str(out_path), vis)

    if len(matches) == 0:
        return EvalRow(
            file_name=image_path.name,
            status="NO_MATCH",
            score=0.0,
            area_ratio=0.0,
            top_y_ratio=0.0,
            bottom_y_ratio=0.0,
            side_left_angle=0.0,
            side_right_angle=0.0,
            side_angle_gap=0.0,
            side_same_sign=0,
            merged_lines=len(merged),
        )

    m = matches[0]
    area_ratio, top_y_ratio, bottom_y_ratio, l_ang, r_ang, side_gap, same_sign = quad_metrics(m, image.shape)
    return EvalRow(
        file_name=image_path.name,
        status="OK",
        score=float(m.score),
        area_ratio=area_ratio,
        top_y_ratio=top_y_ratio,
        bottom_y_ratio=bottom_y_ratio,
        side_left_angle=l_ang,
        side_right_angle=r_ang,
        side_angle_gap=side_gap,
        side_same_sign=same_sign,
        merged_lines=len(merged),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "dataset/images quality")
    parser.add_argument("--output-dir", type=Path, default=TEST_RESULTS_TEMPLATE_DIR / "quality_eval_latest")
    parser.add_argument("--template-path", type=str, default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument("--score-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector = ParkingLineDetector(
        min_line_length=50,
        max_line_gap=10,
        parallel_angle_threshold=5.0,
        min_distance=5,
        max_distance=50,
    )
    template_detector = TemplateParkingDetector(template_path=args.template_path)

    images = sorted(
        [p for p in args.input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    rows: List[EvalRow] = []
    for idx, img_path in enumerate(images, start=1):
        row = eval_one(
            image_path=img_path,
            line_detector=detector,
            template_detector=template_detector,
            score_threshold=args.score_threshold,
            vis_dir=args.output_dir,
        )
        rows.append(row)
        print(f"[{idx}/{len(images)}] {img_path.name}: {row.status} score={row.score:.3f} lines={row.merged_lines}")

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_name",
                "status",
                "score",
                "area_ratio",
                "top_y_ratio",
                "bottom_y_ratio",
                "side_left_angle",
                "side_right_angle",
                "side_angle_gap",
                "side_same_sign",
                "merged_lines",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.file_name,
                    r.status,
                    f"{r.score:.6f}",
                    f"{r.area_ratio:.6f}",
                    f"{r.top_y_ratio:.6f}",
                    f"{r.bottom_y_ratio:.6f}",
                    f"{r.side_left_angle:.4f}",
                    f"{r.side_right_angle:.4f}",
                    f"{r.side_angle_gap:.4f}",
                    r.side_same_sign,
                    r.merged_lines,
                ]
            )

    total = len(rows)
    ok = sum(1 for r in rows if r.status == "OK")
    no_match = sum(1 for r in rows if r.status == "NO_MATCH")
    read_fail = sum(1 for r in rows if r.status == "READ_FAIL")
    print("\nSummary")
    print(f"total={total} ok={ok} no_match={no_match} read_fail={read_fail}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
