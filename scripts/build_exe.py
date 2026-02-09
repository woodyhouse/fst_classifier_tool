"""
PyInstaller 打包脚本: 将 Gradio 推理 App 打包为单文件可执行程序.

用法:
    python scripts/build_exe.py --model fst_classifier.onnx

产出:
    dist/fst_app.exe  (Windows)
    dist/fst_app      (Linux/Mac)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build(model_path: str, output_name: str = "fst_app"):
    model = Path(model_path)
    if not model.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # 确保 schema 文件存在
    schema_path = Path("schema/fst.v1.schema.json")
    if not schema_path.exists():
        print(f"Warning: Schema file not found at {schema_path}")

    # PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", output_name,
        # 添加数据文件
        "--add-data", f"{model}:.",
        "--add-data", f"schema:schema",
        # 隐藏导入
        "--hidden-import", "gradio",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "PIL",
        "--hidden-import", "fst",
        "--hidden-import", "fst.models",
        "--hidden-import", "fst.inference",
        "--hidden-import", "fst.app",
        # 入口
        "fst/app.py",
    ]

    print("Running PyInstaller...")
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"\n✅ Build complete!")
    print(f"   Output: dist/{output_name}")
    print(f"\n   Usage: ./dist/{output_name} --model fst_classifier.onnx")


def main():
    parser = argparse.ArgumentParser(description="Build standalone FST classifier app")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--name", default="fst_app", help="Output executable name")
    args = parser.parse_args()
    build(args.model, args.name)


if __name__ == "__main__":
    main()
