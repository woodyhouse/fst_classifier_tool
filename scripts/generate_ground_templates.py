"""Generate the ground-view optimized template library."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from fst.paths import DEFAULT_GROUND_TEMPLATE_PATH
from fst.template_parking_detector import TemplateGenerator


def main() -> None:
    print("Generating ground-view optimized template library...")

    generator = TemplateGenerator()
    templates = generator.generate_template_library(
        width_range=[60, 80, 100, 120, 150],
        length_range=[150, 200, 250, 300],
        perspective_range=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        rotation_range=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
        line_width_range=[8, 12, 16],
    )

    DEFAULT_GROUND_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    generator.save_templates(str(DEFAULT_GROUND_TEMPLATE_PATH))

    print(f"Saved to: {DEFAULT_GROUND_TEMPLATE_PATH}")
    print(f"Total templates: {len(templates)}")
    for idx in [0, 1000, 2000, 3000, 4000, 5000, 6000]:
        if idx < len(templates):
            template = templates[idx]
            print(
                f"template[{idx}] width={template.width} length={template.length} "
                f"perspective={template.perspective:.2f} rotation={template.rotation}"
            )


if __name__ == "__main__":
    main()
