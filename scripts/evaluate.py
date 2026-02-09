"""
评估脚本: 在测试集上计算各 head 的准确率、混淆矩阵.

用法:
    python scripts/evaluate.py --model fst_classifier.onnx --data ./dataset_test
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from fst.inference import FSTInference
from fst.models import (
    SLOT_TYPE_CLASSES, MANEUVER_CLASSES, OBSTACLE_CLASSES,
    LINE_COLOR_CLASSES, POSITION_KEYS,
)


def load_labels(label_dir: Path) -> Dict[str, dict]:
    """加载所有 JSON 标注."""
    labels = {}
    for p in sorted(label_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            labels[p.stem] = json.load(f)
    return labels


def evaluate(model_path: str, data_root: str, img_size: int = 384):
    root = Path(data_root)
    img_dir = root / "images"
    lbl_dir = root / "labels"

    engine = FSTInference(model_path, img_size=img_size)
    labels = load_labels(lbl_dir)

    # 收集预测 vs 真值
    results = {
        "slot_type": {"y_true": [], "y_pred": []},
        "maneuver": {"y_true": [], "y_pred": []},
        "line_color": {"y_true": [], "y_pred": []},
    }
    obs_results = {pos: {"y_true": [], "y_pred": []} for pos in POSITION_KEYS}

    n_total = 0
    n_skipped = 0

    for stem, label in labels.items():
        # 查找对应图片
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            n_skipped += 1
            continue

        img = Image.open(img_path)
        pred = engine.predict_with_text(img, image_id=stem)

        # slot_type
        results["slot_type"]["y_true"].append(label.get("slot_type", "UNKNOWN"))
        results["slot_type"]["y_pred"].append(pred.get("slot", {}).get("slot_type", "UNKNOWN"))

        # maneuver
        results["maneuver"]["y_true"].append(label.get("maneuver", "UNKNOWN"))
        results["maneuver"]["y_pred"].append(pred.get("maneuver", "UNKNOWN"))

        # line_color
        results["line_color"]["y_true"].append(label.get("marking", {}).get("line_color", "UNKNOWN"))
        results["line_color"]["y_pred"].append(pred.get("slot", {}).get("marking", {}).get("line_color", "UNKNOWN"))

        # obstacles per position
        gt_obs = label.get("obstacles", {})
        pred_obs = pred.get("obstacles", {}).get("pos_map", {})
        for pos in POSITION_KEYS:
            gt_val = gt_obs.get(pos, "UNKNOWN")
            if isinstance(gt_val, list):
                gt_val = gt_val[0] if gt_val else "UNKNOWN"
            pred_val = pred_obs.get(pos, ["UNKNOWN"])
            if isinstance(pred_val, list):
                pred_val = pred_val[0] if pred_val else "UNKNOWN"
            obs_results[pos]["y_true"].append(gt_val)
            obs_results[pos]["y_pred"].append(pred_val)

        n_total += 1

    print(f"\n{'='*60}")
    print(f"Evaluation: {n_total} images ({n_skipped} skipped)")
    print(f"{'='*60}")

    # 各 head 报告
    for head_name, data in results.items():
        print(f"\n--- {head_name} ---")
        print(classification_report(data["y_true"], data["y_pred"], zero_division=0))

    # 障碍物整体
    all_obs_true = []
    all_obs_pred = []
    for pos in POSITION_KEYS:
        all_obs_true.extend(obs_results[pos]["y_true"])
        all_obs_pred.extend(obs_results[pos]["y_pred"])

    print(f"\n--- obstacles (all positions) ---")
    print(classification_report(all_obs_true, all_obs_pred, zero_division=0))

    # 逐位置准确率
    print(f"\n--- per-position accuracy ---")
    for pos in POSITION_KEYS:
        y_t = obs_results[pos]["y_true"]
        y_p = obs_results[pos]["y_pred"]
        acc = sum(1 for a, b in zip(y_t, y_p) if a == b) / max(len(y_t), 1)
        print(f"  {pos:>8s}: {acc:.1%}  (n={len(y_t)})")

    # 保存详细结果
    output = {
        "n_total": n_total,
        "n_skipped": n_skipped,
        "heads": {},
    }
    for head_name, data in results.items():
        acc = sum(1 for a, b in zip(data["y_true"], data["y_pred"]) if a == b) / max(len(data["y_true"]), 1)
        output["heads"][head_name] = {"accuracy": round(acc, 4)}

    report_path = Path("eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FST classifier")
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--data", required=True, help="Test dataset root")
    parser.add_argument("--img-size", type=int, default=384)
    args = parser.parse_args()
    evaluate(args.model, args.data, args.img_size)


if __name__ == "__main__":
    main()
