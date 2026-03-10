"""End-to-end template matching checks."""
from __future__ import annotations

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from fst.paths import DEFAULT_TEMPLATE_PATH, PROJECT_ROOT, TEST_RESULTS_TEMPLATE_DIR
from fst.template_parking_detector import TemplateParkingDetector


def detect_lines_simple(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10,
    )
    if lines is None:
        return []
    return [np.array(line[0], dtype=np.float32) for line in lines]


def test_full_pipeline() -> None:
    test_images = list((PROJECT_ROOT / "test_images").glob("*.jpg"))
    test_images.extend((PROJECT_ROOT / "test_images").glob("*.png"))
    if not test_images:
        test_with_synthetic_data()
        return

    template_detector = TemplateParkingDetector(template_path=str(DEFAULT_TEMPLATE_PATH))
    output_dir = TEST_RESULTS_TEMPLATE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in test_images[:3]:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"read failed: {image_path}")
            continue

        lines = detect_lines_simple(image)
        if not lines:
            print(f"no lines: {image_path.name}")
            continue

        matches = template_detector.detect(image, lines, score_threshold=0.35)
        template_detector.visualize(image, matches, str(output_dir / f"result_{image_path.name}"))
        print(f"{image_path.name}: matches={len(matches)} lines={len(lines)}")


def test_with_synthetic_data() -> None:
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 100
    parking_spots = [
        ([150, 700, 150, 300], [250, 700, 240, 300]),
        ([300, 700, 300, 300], [400, 700, 390, 300]),
        ([450, 700, 450, 300], [550, 700, 540, 300]),
    ]

    lines = []
    for left, right in parking_spots:
        cv2.line(image, (left[0], left[1]), (left[2], left[3]), (255, 255, 255), 8)
        cv2.line(image, (right[0], right[1]), (right[2], right[3]), (255, 255, 255), 8)
        lines.append(np.array(left, dtype=np.float32))
        lines.append(np.array(right, dtype=np.float32))

    TEST_RESULTS_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TEST_RESULTS_TEMPLATE_DIR / "synthetic_input.jpg"), image)

    template_detector = TemplateParkingDetector(template_path=str(DEFAULT_TEMPLATE_PATH))
    matches = template_detector.detect(image, lines, score_threshold=0.3)
    template_detector.visualize(image, matches, str(TEST_RESULTS_TEMPLATE_DIR / "synthetic_result.jpg"))
    print(f"synthetic matches={len(matches)}")


if __name__ == "__main__":
    test_full_pipeline()
