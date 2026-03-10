"""
FST 半自动标注工具 - 集成 YOLO + 透视分析 + 9宫格可视化

工作流程:
1. YOLO 自动检测物体（绘制检测框）
2. 车位线检测 + 透视分析（绘制透视关系）
3. 车位类型识别（垂直/水平/斜列）
4. 9宫格自动划分（绘制地面九宫格）
5. 物体自动映射到格子
6. 展示可视化结果 + 人工校验场景属性

用法:
    python -m fst.auto_label_tool --images ./dataset/images --labels ./dataset/labels
"""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2
from PIL import Image, ImageTk

from fst.models import (
    SLOT_TYPE_CLASSES, MANEUVER_CLASSES,
    LINE_COLOR_CLASSES, LINE_VIS_CLASSES, LINE_STYLE_CLASSES,
    SPECIAL_SCENE_CLASSES, POSITION_KEYS,
)
from fst.yolo_detector import YOLODetector, Detection
from fst.simple_detector import SimpleParkingLineDetector, simple_grid_mapping

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 中文显示名映射
_ZH = {
    "PERPENDICULAR": "垂直", "PARALLEL": "水平/侧方", "ANGLED": "斜列/鱼骨", "UNKNOWN": "未知",
    "PARK_IN": "泊入", "PARK_OUT": "泊出", "HEAD_IN": "车头泊入", "TAIL_OUT": "车尾泊出",
    "WHITE": "白色", "YELLOW": "黄色", "BLUE": "蓝色", "MIXED": "混合", "NONE": "无",
    "CLEAR": "清晰", "FAINT": "模糊", "MISSING": "缺失",
    "SOLID": "实线", "DASHED": "虚线",
}


def _display(val: str) -> str:
    zh = _ZH.get(val, "")
    return f"{val} ({zh})" if zh else val


class AutoLabelTool:
    """半自动标注工具."""

    def __init__(self, img_dir: Path, lbl_dir: Path, yolo_size: str = "n", use_dl: bool = False):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        # 收集图片
        self.images: List[Path] = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT]
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {img_dir}")

        self.current_idx = 0

        # 初始化 YOLO 和几何分析器
        print(f"正在加载 YOLO 模型 (yolov8{yolo_size})...")
        self.yolo = YOLODetector(model_size=yolo_size, conf_threshold=0.25)

        if use_dl:
            print("提示: 简化架构中已禁用深度学习标记点检测，自动回退到简单线检测。")
        self.line_detector = SimpleParkingLineDetector()

        # 缓存当前图片的分析结果
        self.current_detections: List[Detection] = []
        self.current_line_result: Dict = {"vertical_lines": [], "horizontal_lines": [], "slot_type": "UNKNOWN", "has_lines": False}
        self.current_obstacles: Dict[str, str] = {pos: "EMPTY" for pos in POSITION_KEYS}
        self.current_slot_type: str = "UNKNOWN"

        # ── 主窗口 ──
        self.root = tk.Tk()
        self.root.title("FST 半自动标注工具")
        self.root.geometry("1400x900")

        # ── 左侧: 可视化图片 ──
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.img_label = tk.Label(left_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True)

        # 导航栏
        nav_frame = tk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        tk.Button(nav_frame, text="◀ 上一张 (A)", command=self.prev_image, width=15).pack(side=tk.LEFT, padx=5)
        self.progress_label = tk.Label(nav_frame, text="1/1")
        self.progress_label.pack(side=tk.LEFT, expand=True)
        tk.Button(nav_frame, text="下一张 (D) ▶", command=self.next_image, width=15).pack(side=tk.RIGHT, padx=5)

        # 重新分析按钮
        tk.Button(left_frame, text="🔄 重新分析 (R)", command=self.reanalyze,
                  bg="#2196F3", fg="white", font=("", 10, "bold")).pack(pady=5)

        # ── 右侧: 标注面板 ──
        right_frame = tk.Frame(self.root, width=420)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)

        canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        self.panel = tk.Frame(canvas)
        self.panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0

        # 自动识别结果提示
        tk.Label(self.panel, text="🤖 自动识别结果", font=("", 12, "bold"), fg="#4CAF50").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 10))
        row += 1

        self.auto_result_label = tk.Label(self.panel, text="等待分析...", font=("", 9), fg="#666", wraplength=380, justify="left")
        self.auto_result_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        tk.Label(self.panel, text="━" * 50, font=("", 8)).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        # 车位类型（自动识别 + 可修改）
        tk.Label(self.panel, text="车位类型 (slot_type)", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 2))
        row += 1
        self.slot_type_var = tk.StringVar(value="UNKNOWN")
        self.slot_type_combo = ttk.Combobox(
            self.panel, textvariable=self.slot_type_var,
            values=[_display(v) for v in SLOT_TYPE_CLASSES], width=30, state="readonly")
        self.slot_type_combo.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 泊车动作
        tk.Label(self.panel, text="泊车动作 (maneuver)", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 2))
        row += 1
        self.maneuver_var = tk.StringVar(value="UNKNOWN")
        ttk.Combobox(self.panel, textvariable=self.maneuver_var,
                     values=[_display(v) for v in MANEUVER_CLASSES], width=30, state="readonly").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 标线颜色
        tk.Label(self.panel, text="标线颜色", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 2))
        row += 1
        self.line_color_var = tk.StringVar(value="UNKNOWN")
        ttk.Combobox(self.panel, textvariable=self.line_color_var,
                     values=[_display(v) for v in LINE_COLOR_CLASSES], width=30, state="readonly").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 标线可见度
        tk.Label(self.panel, text="标线可见度", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 2))
        row += 1
        self.line_vis_var = tk.StringVar(value="UNKNOWN")
        ttk.Combobox(self.panel, textvariable=self.line_vis_var,
                     values=[_display(v) for v in LINE_VIS_CLASSES], width=30, state="readonly").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 标线样式
        tk.Label(self.panel, text="标线样式", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 2))
        row += 1
        self.line_style_var = tk.StringVar(value="UNKNOWN")
        ttk.Combobox(self.panel, textvariable=self.line_style_var,
                     values=[_display(v) for v in LINE_STYLE_CLASSES], width=30, state="readonly").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 特殊场景 (多选)
        tk.Label(self.panel, text="特殊场景 (可多选)", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2))
        row += 1
        self.scene_vars: Dict[str, tk.BooleanVar] = {}
        for sc in SPECIAL_SCENE_CLASSES:
            var = tk.BooleanVar(value=False)
            self.scene_vars[sc] = var
            tk.Checkbutton(self.panel, text=_display(sc), variable=var).grid(
                row=row, column=0, columnspan=2, sticky="w")
            row += 1

        # 保存按钮
        row += 1
        tk.Button(self.panel, text="💾 保存标注 (S)", command=self.save_label,
                  bg="#4CAF50", fg="white", font=("", 12, "bold"), width=30, height=2).grid(
            row=row, column=0, columnspan=2, pady=10)

        # 键盘快捷键
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<s>", lambda e: self.save_label())
        self.root.bind("<r>", lambda e: self.reanalyze())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

        # 加载第一张
        self.load_image()

    def _extract_enum(self, display_str: str) -> str:
        """从 'PERPENDICULAR (垂直)' 提取 'PERPENDICULAR'."""
        return display_str.split(" ")[0].strip()

    def analyze_image(self, img_rgb: np.ndarray):
        """自动分析图片: YOLO + 简化线检测 + 规则映射."""
        print(f"\n🔍 开始自动分析...")

        # 1. YOLO 检测
        self.current_detections = self.yolo.detect(img_rgb)
        print(f"✓ YOLO 检测到 {len(self.current_detections)} 个物体")

        # 2. 简化线检测（用于车位类型估计）
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.current_line_result = self.line_detector.detect(img_bgr)
        n_v = len(self.current_line_result.get("vertical_lines", []))
        n_h = len(self.current_line_result.get("horizontal_lines", []))
        print(f"✓ 检测到线段: vertical={n_v}, horizontal={n_h}")

        # 3. 车位类型识别
        self.current_slot_type = self._infer_slot_type(self.current_line_result)
        print(f"✓ 推断车位类型: {self.current_slot_type}")

        # 4. 9宫格障碍物映射
        h, w = img_rgb.shape[:2]
        self.current_obstacles = self._map_obstacles(h, w)
        print(f"✓ 物体映射完成")

        # 更新自动识别结果显示
        result_text = "检测方法: YOLO + 简化线检测\n"
        result_text += f"车位类型: {_display(self.current_slot_type)}\n"
        result_text += f"检测物体: {len(self.current_detections)} 个\n"
        result_text += f"垂直线: {n_v} 条\n"
        result_text += f"水平线: {n_h} 条\n"
        result_text += f"9宫格障碍物: {sum(1 for v in self.current_obstacles.values() if v != 'EMPTY')} 个"
        self.auto_result_label.config(text=result_text)

        # 自动填充车位类型
        self.slot_type_var.set(_display(self.current_slot_type))

    def _infer_slot_type(self, line_result: Dict) -> str:
        """根据简化线检测结果推断车位类型."""
        if not line_result.get("has_lines", False):
            return "UNKNOWN"
        return line_result.get("slot_type", "UNKNOWN")

    def _map_obstacles(self, h: int, w: int) -> Dict[str, str]:
        """将YOLO检测结果映射到简化9宫格."""
        by_type: Dict[str, List] = {}
        for det in self.current_detections:
            fst_type = self.yolo.map_to_fst_type(det.class_name)
            if fst_type in ("EMPTY", "UNKNOWN"):
                continue
            by_type.setdefault(fst_type, []).append(det.bbox)

        grid = simple_grid_mapping(by_type, (h, w))
        obstacles: Dict[str, str] = {}
        for pos in POSITION_KEYS:
            val = grid.get(pos, ["EMPTY"])
            if isinstance(val, list):
                obstacles[pos] = val[0] if val else "UNKNOWN"
            else:
                obstacles[pos] = val
        return obstacles

    def visualize_analysis(self, img_rgb: np.ndarray) -> np.ndarray:
        """可视化分析结果: 检测框 + 线检测 + 简化9宫格."""
        vis = img_rgb.copy()

        # 1. YOLO检测框
        for det in self.current_detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 120, 255), 2)
            fst_type = self.yolo.map_to_fst_type(det.class_name)
            label = f"{det.class_name}→{fst_type}"
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 255), 1)
            cx, cy = det.center
            cv2.circle(vis, (int(cx), int(cy)), 4, (255, 0, 0), -1)

        # 2. 简化线检测可视化
        for line in self.current_line_result.get("vertical_lines", []):
            x1, y1, x2, y2 = line
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        for line in self.current_line_result.get("horizontal_lines", []):
            x1, y1, x2, y2 = line
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        # 3. 简化3x3网格 + 位置文本
        h, w = vis.shape[:2]
        grid_h = h // 3
        grid_w = w // 3
        for i in (1, 2):
            cv2.line(vis, (0, i * grid_h), (w, i * grid_h), (0, 255, 0), 1)
            cv2.line(vis, (i * grid_w, 0), (i * grid_w, h), (0, 255, 0), 1)

        text_anchor = {
            "1": (grid_w // 2, grid_h // 2),
            "2": (grid_w + grid_w // 2, grid_h // 2),
            "3": (2 * grid_w + grid_w // 2, grid_h // 2),
            "4": (grid_w // 2, grid_h + grid_h // 2),
            "5": (grid_w + grid_w // 2, grid_h + grid_h // 2),
            "6": (2 * grid_w + grid_w // 2, grid_h + grid_h // 2),
            "7": (w // 2, 2 * grid_h + grid_h // 2),
            "P_LEFT": (grid_w // 3, int(h * 0.78)),
            "P_RIGHT": (w - grid_w // 2, int(h * 0.78)),
        }
        for pos, (cx, cy) in text_anchor.items():
            obstacle = self.current_obstacles.get(pos, "EMPTY")
            cv2.putText(vis, f"{pos}:{obstacle}", (int(cx) - 45, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 4. 图例
        legend_y = 30
        cv2.putText(vis, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, "Blue: YOLO Detection", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
        cv2.putText(vis, "Red: Detected Line", (10, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(vis, "Green: 3x3 Grid", (10, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return vis

    def load_image(self):
        """加载当前图片并自动分析."""
        img_path = self.images[self.current_idx]
        self.progress_label.config(text=f"{self.current_idx + 1}/{len(self.images)}  |  {img_path.name}")

        # 读取图片
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 自动分析
        self.analyze_image(img_rgb)

        # 可视化
        vis = self.visualize_analysis(img_rgb)

        # 缩放显示
        max_w, max_h = 900, 800
        h, w = vis.shape[:2]
        ratio = min(max_w / w, max_h / h)
        if ratio < 1:
            vis = cv2.resize(vis, (int(w * ratio), int(h * ratio)))

        # 转换为 PIL 显示
        vis_pil = Image.fromarray(vis)
        self.tk_img = ImageTk.PhotoImage(vis_pil)
        self.img_label.config(image=self.tk_img)

        # 加载已有标注（如果存在）
        lbl_path = self.lbl_dir / (img_path.stem + ".json")
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                label = json.load(f)
            self._set_from_label(label)

    def reanalyze(self):
        """重新分析当前图片."""
        self.load_image()

    def _set_from_label(self, label: Dict):
        """从已有标注填充字段."""
        self.slot_type_var.set(_display(label.get("slot_type", "UNKNOWN")))
        self.maneuver_var.set(_display(label.get("maneuver", "UNKNOWN")))

        mk = label.get("marking", {})
        self.line_color_var.set(_display(mk.get("line_color", "UNKNOWN")))
        self.line_vis_var.set(_display(mk.get("line_visibility", "UNKNOWN")))
        self.line_style_var.set(_display(mk.get("line_style", "UNKNOWN")))

        # 特殊场景
        ss = label.get("special_scene", {})
        active = set(ss.get("P0", []) + ss.get("P1", []))
        for sc, var in self.scene_vars.items():
            var.set(sc in active)

    def save_label(self):
        """保存标注."""
        img_path = self.images[self.current_idx]

        # 收集特殊场景
        p0_list = [sc for sc in ["DEAD_END", "NARROW_LANE"] if self.scene_vars.get(sc, tk.BooleanVar()).get()]
        p1_list = [sc for sc in SPECIAL_SCENE_CLASSES if sc not in ("DEAD_END", "NARROW_LANE") and self.scene_vars.get(sc, tk.BooleanVar()).get()]

        label = {
            "image_id": img_path.stem,
            "slot_type": self._extract_enum(self.slot_type_var.get()),
            "maneuver": self._extract_enum(self.maneuver_var.get()),
            "marking": {
                "line_color": self._extract_enum(self.line_color_var.get()),
                "line_visibility": self._extract_enum(self.line_vis_var.get()),
                "line_style": self._extract_enum(self.line_style_var.get()),
            },
            "special_scene": {
                "P0": p0_list,
                "P1": p1_list,
            },
            "obstacles": self.current_obstacles,  # 使用自动识别的结果
        }

        lbl_path = self.lbl_dir / (img_path.stem + ".json")
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(label, f, indent=2, ensure_ascii=False)

        self.root.title(f"FST 半自动标注工具 — 已保存: {lbl_path.name}")
        print(f"✓ 已保存: {lbl_path}")

    def next_image(self):
        self.save_label()
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_image()

    def prev_image(self):
        self.save_label()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="FST Auto Labeling Tool")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", required=True, help="Label output directory")
    parser.add_argument("--yolo", default="n", choices=["n", "s", "m", "l", "x"], help="YOLO model size")
    parser.add_argument("--use-dl", action="store_true", help="Use deep learning (DMPR-PS) for marking point detection")
    args = parser.parse_args()

    tool = AutoLabelTool(Path(args.images), Path(args.labels), yolo_size=args.yolo, use_dl=args.use_dl)
    tool.run()


if __name__ == "__main__":
    main()
