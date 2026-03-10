"""Quick smoke test for the ground-view template detector."""
from __future__ import annotations

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from fst.paths import DEFAULT_GROUND_TEMPLATE_PATH, TEST_RESULTS_TEMPLATE_DIR
from fst.template_parking_detector import TemplateParkingDetector


def quick_test() -> None:
    output_dir = TEST_RESULTS_TEMPLATE_DIR / "ground_template_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "test_ground_input.jpg"
    result_path = output_dir / "test_ground_result.jpg"

    image = np.ones((600, 800, 3), dtype=np.uint8) * 100
    cv2.line(image, (200, 550), (220, 200), (255, 255, 255), 10)
    cv2.line(image, (280, 550), (260, 200), (255, 255, 255), 10)
    cv2.line(image, (350, 550), (370, 200), (255, 255, 255), 10)
    cv2.line(image, (430, 550), (410, 200), (255, 255, 255), 10)
    cv2.imwrite(str(input_path), image)

    detector = TemplateParkingDetector(template_path=str(DEFAULT_GROUND_TEMPLATE_PATH))
    lines = [
        np.array([200, 550, 220, 200], dtype=np.float32),
        np.array([280, 550, 260, 200], dtype=np.float32),
        np.array([350, 550, 370, 200], dtype=np.float32),
        np.array([430, 550, 410, 200], dtype=np.float32),
    ]

    matches = detector.detect(image, lines, score_threshold=0.3)
    detector.visualize(image, matches, str(result_path))

    print(f"matches={len(matches)}")
    print(f"input={input_path}")
    print(f"output={result_path}")


if __name__ == "__main__":
    quick_test()
