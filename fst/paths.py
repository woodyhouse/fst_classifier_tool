"""Common project paths shared by the app, scripts, and tests."""
from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

SCHEMA_DIR = PROJECT_ROOT / "schema"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESOURCES_DIR = PROJECT_ROOT / "resources"
TEMPLATES_DIR = RESOURCES_DIR / "templates"
SAMPLES_DIR = RESOURCES_DIR / "samples"

DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
DATASET_LABELS_DIR = DATASET_DIR / "labels"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
EXTERNAL_DIR = PROJECT_ROOT / "external"

TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"
TEST_RESULTS_BEV_DIR = PROJECT_ROOT / "test_results_bev"
TEST_RESULTS_SIMPLIFIED_DIR = PROJECT_ROOT / "test_results_simplified"
TEST_RESULTS_TEMPLATE_DIR = PROJECT_ROOT / "test_results_template"

YOLO_MAPPING_CONFIG_PATH = CONFIGS_DIR / "yolo_mapping_config.yaml"
DEFAULT_TEMPLATE_PATH = TEMPLATES_DIR / "parking_templates.pkl"
DEFAULT_GROUND_TEMPLATE_PATH = TEMPLATES_DIR / "parking_templates_ground.pkl"
SAMPLE_GROUND_INPUT_PATH = SAMPLES_DIR / "test_ground_input.jpg"
SAMPLE_GROUND_RESULT_PATH = SAMPLES_DIR / "test_ground_result.jpg"

