"""Visualize a subset of generated parking templates."""
from __future__ import annotations

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from fst.paths import DEFAULT_TEMPLATE_PATH, TEST_RESULTS_TEMPLATE_DIR
from fst.template_parking_detector import TemplateGenerator


def visualize_templates() -> None:
    generator = TemplateGenerator()
    generator.load_templates(str(DEFAULT_TEMPLATE_PATH))

    selected_indices = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    canvas = np.ones((3 * 300, 4 * 300, 3), dtype=np.uint8) * 200

    for idx, template_idx in enumerate(selected_indices):
        if template_idx >= len(generator.templates):
            continue

        template = generator.templates[template_idx]
        row = idx // 4
        col = idx % 4
        offset_x = col * 300 + 150
        offset_y = row * 300 + 150

        scale = 0.5
        outer_scaled = template.outer_quad * scale + np.array([offset_x, offset_y])
        inner_scaled = template.inner_quad * scale + np.array([offset_x, offset_y])

        cv2.polylines(canvas, [outer_scaled.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(canvas, [inner_scaled.astype(np.int32)], True, (255, 0, 0), 2)
        cv2.putText(
            canvas,
            f"P={template.perspective:.1f}",
            (offset_x - 40, offset_y + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"R={int(template.rotation)}",
            (offset_x - 40, offset_y + 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    output_path = TEST_RESULTS_TEMPLATE_DIR / "template_visualization.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Template visualization saved to: {output_path}")


if __name__ == "__main__":
    visualize_templates()
