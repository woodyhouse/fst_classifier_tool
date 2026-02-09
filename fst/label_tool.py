"""
FST 标注工具: 基于 tkinter 的轻量标注 GUI.

功能:
  - 加载图片文件夹，逐张标注
  - 下拉菜单选择车位类型、泊车动作、标线属性等
  - 9 宫格按钮选择各方位障碍物
  - 自动保存 JSON 到 labels/ 目录
  - 支持前进/后退浏览
  - 自动加载已有标注

用法:
    python -m fst.label_tool --images ./dataset/images --labels ./dataset/labels
"""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageTk

from fst.models import (
    SLOT_TYPE_CLASSES, MANEUVER_CLASSES,
    LINE_COLOR_CLASSES, LINE_VIS_CLASSES, LINE_STYLE_CLASSES,
    OBSTACLE_CLASSES, POSITION_KEYS,
    SPECIAL_SCENE_CLASSES,
)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 中文显示名映射
_ZH = {
    "PERPENDICULAR": "垂直", "PARALLEL": "水平/侧方", "ANGLED": "斜列/鱼骨", "UNKNOWN": "未知",
    "PARK_IN": "泊入", "PARK_OUT": "泊出", "HEAD_IN": "车头泊入", "TAIL_OUT": "车尾泊出",
    "WHITE": "白色", "YELLOW": "黄色", "BLUE": "蓝色", "MIXED": "混合", "NONE": "无", 
    "CLEAR": "清晰", "FAINT": "模糊", "MISSING": "缺失",
    "SOLID": "实线", "DASHED": "虚线",
    "EMPTY": "空", "VEHICLE": "车", "CURB": "路沿", "WALL": "墙", "PILLAR": "柱",
    "CONE": "锥桶", "WATER_BARRIER": "水马", "FENCE": "栅栏", "LAMP": "路灯",
    "FIRE_BOX_SUSPENDED": "悬空消防箱", "BUSH": "灌木丛",
    "DEAD_END": "断头路", "NARROW_LANE": "窄通道",
    "SLOPE": "坡道", "SPLIT_SLOPE": "分体坡道", "SPACE_UNMARKED": "空间(未画线)",
    "COLOR_BLOCK": "色块", "BRICK_GRASS": "砖草", "STEP_BRICK_GRASS": "台阶砖草",
    "WLC": "WLC(无线充电)", "MICRO": "微型", "BRICK_STONE": "砖石",
    "NARROW_SLOT": "窄车位", "EXTREME_NARROW_SLOT": "极窄车位", "MECHANICAL": "机械车位",
}


def _display(val: str) -> str:
    zh = _ZH.get(val, "")
    return f"{val} ({zh})" if zh else val


class LabelTool:
    def __init__(self, img_dir: Path, lbl_dir: Path):
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

        # ── 主窗口 ──
        self.root = tk.Tk()
        self.root.title("FST 车位标注工具")
        self.root.geometry("1280x900")

        # ── 左侧: 图片显示 ──
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.img_label = tk.Label(left_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True)

        nav_frame = tk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        tk.Button(nav_frame, text="◀ 上一张 (A)", command=self.prev_image, width=15).pack(side=tk.LEFT, padx=5)
        self.progress_label = tk.Label(nav_frame, text="1/1")
        self.progress_label.pack(side=tk.LEFT, expand=True)
        tk.Button(nav_frame, text="下一张 (D) ▶", command=self.next_image, width=15).pack(side=tk.RIGHT, padx=5)

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

        # 车位类型
        tk.Label(self.panel, text="车位类型 (slot_type)", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5,2))
        row += 1
        self.slot_type_var = tk.StringVar(value="UNKNOWN")
        self.slot_type_combo = ttk.Combobox(self.panel, textvariable=self.slot_type_var,
                                             values=[_display(v) for v in SLOT_TYPE_CLASSES], width=30, state="readonly")
        self.slot_type_combo.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 泊车动作
        tk.Label(self.panel, text="泊车动作 (maneuver)", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5,2))
        row += 1
        self.maneuver_var = tk.StringVar(value="UNKNOWN")
        self.maneuver_combo = ttk.Combobox(self.panel, textvariable=self.maneuver_var,
                                            values=[_display(v) for v in MANEUVER_CLASSES], width=30, state="readonly")
        self.maneuver_combo.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 标线颜色
        tk.Label(self.panel, text="标线颜色", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5,2))
        row += 1
        self.line_color_var = tk.StringVar(value="UNKNOWN")
        ttk.Combobox(self.panel, textvariable=self.line_color_var,
                     values=[_display(v) for v in LINE_COLOR_CLASSES], width=30, state="readonly").grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 标线可见度
        tk.Label(self.panel, text="标线可见度", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5,2))
        row += 1
        self.line_vis_var = tk.StringVar(value="UNKNOWN")
        ttk.Combobox(self.panel, textvariable=self.line_vis_var,
                     values=[_display(v) for v in LINE_VIS_CLASSES], width=30, state="readonly").grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 标线样式
        tk.Label(self.panel, text="标线样式", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5,2))
        row += 1
        self.line_style_var = tk.StringVar(value="UNKNOWN")
        ttk.Combobox(self.panel, textvariable=self.line_style_var,
                     values=[_display(v) for v in LINE_STYLE_CLASSES], width=30, state="readonly").grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # 特殊场景 (多选)
        tk.Label(self.panel, text="特殊场景 (可多选)", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10,2))
        row += 1
        self.scene_vars: Dict[str, tk.BooleanVar] = {}
        for sc in SPECIAL_SCENE_CLASSES:
            var = tk.BooleanVar(value=False)
            self.scene_vars[sc] = var
            tk.Checkbutton(self.panel, text=_display(sc), variable=var).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1

        # 障碍物方位 (9 个下拉)
        tk.Label(self.panel, text="障碍物方位 (1-7 + P左右)", font=("", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10,2))
        row += 1

        # 位置布局示意
        # 远端:  1  2  3
        #       P_L 7 P_R
        # 近端:  4  5  6
        pos_layout = [
            ["1", "2", "3"],
            ["P_LEFT", "7", "P_RIGHT"],
            ["4", "5", "6"],
        ]
        pos_labels = {
            "1": "①远左", "2": "②远中", "3": "③远右",
            "4": "④近左", "5": "⑤近中", "6": "⑥近右",
            "7": "⑦车位内", "P_LEFT": "P左", "P_RIGHT": "P右",
        }

        self.obs_vars: Dict[str, tk.StringVar] = {}
        obstacle_display = [_display(v) for v in OBSTACLE_CLASSES]

        for grid_row in pos_layout:
            frame = tk.Frame(self.panel)
            frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=1)
            for pos in grid_row:
                sub = tk.Frame(frame)
                sub.pack(side=tk.LEFT, padx=3)
                tk.Label(sub, text=pos_labels[pos], font=("", 8)).pack()
                var = tk.StringVar(value=_display("UNKNOWN"))
                self.obs_vars[pos] = var
                ttk.Combobox(sub, textvariable=var, values=obstacle_display, width=14, state="readonly").pack()
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
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

        # 加载第一张
        self.load_image()

    def _extract_enum(self, display_str: str) -> str:
        """从 'PERPENDICULAR (垂直)' 提取 'PERPENDICULAR'."""
        return display_str.split(" ")[0].strip()

    def load_image(self):
        """加载当前索引的图片和已有标注."""
        img_path = self.images[self.current_idx]
        self.progress_label.config(text=f"{self.current_idx + 1}/{len(self.images)}  |  {img_path.name}")

        # 显示图片
        img = Image.open(img_path)
        # 缩放以适配显示区域
        max_w, max_h = 700, 700
        ratio = min(max_w / img.width, max_h / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)

        # 加载已有标注
        lbl_path = self.lbl_dir / (img_path.stem + ".json")
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                label = json.load(f)
            self._set_from_label(label)
        else:
            self._reset_fields()

    def _reset_fields(self):
        """重置所有字段为默认值."""
        self.slot_type_var.set(_display("UNKNOWN"))
        self.maneuver_var.set(_display("UNKNOWN"))
        self.line_color_var.set(_display("UNKNOWN"))
        self.line_vis_var.set(_display("UNKNOWN"))
        self.line_style_var.set(_display("UNKNOWN"))
        for var in self.scene_vars.values():
            var.set(False)
        for var in self.obs_vars.values():
            var.set(_display("UNKNOWN"))

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

        # 障碍物
        obs = label.get("obstacles", {})
        for pos, var in self.obs_vars.items():
            val = obs.get(pos, "UNKNOWN")
            if isinstance(val, list):
                val = val[0] if val else "UNKNOWN"
            var.set(_display(val))

    def save_label(self):
        """保存当前标注为 JSON."""
        img_path = self.images[self.current_idx]

        # 收集特殊场景
        p0_list = [sc for sc in ["DEAD_END", "NARROW_LANE"] if self.scene_vars.get(sc, tk.BooleanVar()).get()]
        p1_list = [sc for sc in SPECIAL_SCENE_CLASSES if sc not in ("DEAD_END", "NARROW_LANE") and self.scene_vars.get(sc, tk.BooleanVar()).get()]

        # 收集障碍物
        obstacles = {}
        for pos, var in self.obs_vars.items():
            obstacles[pos] = self._extract_enum(var.get())

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
            "obstacles": obstacles,
        }

        lbl_path = self.lbl_dir / (img_path.stem + ".json")
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(label, f, indent=2, ensure_ascii=False)

        self.root.title(f"FST 标注工具 — 已保存: {lbl_path.name}")

    def next_image(self):
        self.save_label()  # 自动保存
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_image()

    def prev_image(self):
        self.save_label()  # 自动保存
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="FST Parking Slot Labeling Tool")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", required=True, help="Label output directory")
    args = parser.parse_args()

    tool = LabelTool(Path(args.images), Path(args.labels))
    tool.run()


if __name__ == "__main__":
    main()
