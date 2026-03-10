"""Manual template-detector checks."""
from __future__ import annotations

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from fst.parking_line_detector import ParkingLineDetector
from fst.paths import DEFAULT_TEMPLATE_PATH, PROJECT_ROOT, TEST_RESULTS_TEMPLATE_DIR
from fst.template_parking_detector import TemplateParkingDetector


def _merge_lines(detector: ParkingLineDetector, image: np.ndarray):
    edges = detector.preprocess(image)
    lines = detector.detect_lines(edges)
    if hasattr(detector, "merge_collinear_lines"):
        return detector.merge_collinear_lines(lines)
    if hasattr(detector, "merge_lines"):
        return detector.merge_lines(lines)
    return lines if lines is not None else []


def test_template_detector() -> None:
    template_detector = TemplateParkingDetector(template_path=str(DEFAULT_TEMPLATE_PATH))
    line_detector = ParkingLineDetector()
    output_dir = TEST_RESULTS_TEMPLATE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = [
        PROJECT_ROOT / "test_images" / "parking1.jpg",
        PROJECT_ROOT / "test_images" / "parking2.jpg",
        PROJECT_ROOT / "test_images" / "parking3.jpg",
    ]

    for image_path in test_images:
        if not image_path.exists():
            print(f"missing: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"read failed: {image_path}")
            continue

        merged_lines = _merge_lines(line_detector, image)
        matches = template_detector.detect(image, merged_lines, score_threshold=0.4)
        template_detector.visualize(image, matches, str(output_dir / image_path.name))
        print(f"{image_path.name}: matches={len(matches)} lines={len(merged_lines)}")


def test_with_simple_lines() -> None:
    image = np.ones((600, 800, 3), dtype=np.uint8) * 128
    merged_lines = [
        np.array([200, 500, 200, 200], dtype=np.float32),
        np.array([300, 500, 280, 200], dtype=np.float32),
    ]
    for line in merged_lines:
        x1, y1, x2, y2 = line.astype(int)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    template_detector = TemplateParkingDetector(template_path=str(DEFAULT_TEMPLATE_PATH))
    matches = template_detector.detect(image, merged_lines, score_threshold=0.3)
    output_dir = TEST_RESULTS_TEMPLATE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    template_detector.visualize(image, matches, str(output_dir / "simple_test.jpg"))
    print(f"simple_test matches={len(matches)}")


if __name__ == "__main__":
    test_with_simple_lines()
