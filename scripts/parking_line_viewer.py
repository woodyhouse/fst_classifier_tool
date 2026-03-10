"""
车位线检测交互式查看器
支持图片缩放、拖动、参数调整
"""
import _bootstrap  # noqa: F401
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from fst.parking_line_detector import ParkingLineDetector
from fst.template_parking_detector import TemplateParkingDetector
from fst.paths import DEFAULT_GROUND_TEMPLATE_PATH


class ParkingLineViewer:
    """交互式车位线检测查看器"""

    def __init__(self, root):
        self.root = root
        self.root.title("车位线检测工具")
        self.root.geometry("1400x900")

        # 图像相关
        self.original_image = None
        self.display_image = None
        self.result_image = None
        self.debug_images = {}
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None

        # 检测器
        self.detector = None
        self.template_detector = None
        self.parking_line_pairs = []
        self.use_template_matching = tk.BooleanVar(value=False)

        self.setup_ui()
        self.update_detector()

    def setup_ui(self):
        """设置UI"""
        # 左侧控制面板
        control_frame = ttk.Frame(self.root, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 文件操作
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding=10)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_frame, text="打开图片", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="保存结果", command=self.save_result).pack(fill=tk.X, pady=2)

        # 参数设置
        param_frame = ttk.LabelFrame(control_frame, text="检测参数", padding=10)
        param_frame.pack(fill=tk.X, pady=5)

        # 最小线段长度
        ttk.Label(param_frame, text="最小线段长度:").pack(anchor=tk.W)
        self.min_length_var = tk.IntVar(value=50)
        ttk.Scale(param_frame, from_=20, to=200, variable=self.min_length_var,
                 orient=tk.HORIZONTAL, command=self.on_param_change).pack(fill=tk.X)
        self.min_length_label = ttk.Label(param_frame, text="50")
        self.min_length_label.pack(anchor=tk.W)

        # 平行角度阈值
        ttk.Label(param_frame, text="平行角度阈值:").pack(anchor=tk.W, pady=(10, 0))
        self.angle_threshold_var = tk.DoubleVar(value=5.0)
        ttk.Scale(param_frame, from_=1.0, to=15.0, variable=self.angle_threshold_var,
                 orient=tk.HORIZONTAL, command=self.on_param_change).pack(fill=tk.X)
        self.angle_label = ttk.Label(param_frame, text="5.0°")
        self.angle_label.pack(anchor=tk.W)

        # 最小距离
        ttk.Label(param_frame, text="最小距离:").pack(anchor=tk.W, pady=(10, 0))
        self.min_dist_var = tk.IntVar(value=15)
        ttk.Scale(param_frame, from_=5, to=100, variable=self.min_dist_var,
                 orient=tk.HORIZONTAL, command=self.on_param_change).pack(fill=tk.X)
        self.min_dist_label = ttk.Label(param_frame, text="15")
        self.min_dist_label.pack(anchor=tk.W)

        # 最大距离
        ttk.Label(param_frame, text="最大距离:").pack(anchor=tk.W, pady=(10, 0))
        self.max_dist_var = tk.IntVar(value=150)
        ttk.Scale(param_frame, from_=50, to=300, variable=self.max_dist_var,
                 orient=tk.HORIZONTAL, command=self.on_param_change).pack(fill=tk.X)
        self.max_dist_label = ttk.Label(param_frame, text="150")
        self.max_dist_label.pack(anchor=tk.W)

        # 检测按钮
        ttk.Button(param_frame, text="重新检测", command=self.detect).pack(fill=tk.X, pady=(10, 0))

        # 模板匹配选项
        ttk.Checkbutton(
            param_frame,
            text="使用模板匹配",
            variable=self.use_template_matching,
            command=self.on_template_toggle
        ).pack(fill=tk.X, pady=(10, 0))

        # 视图控制
        view_frame = ttk.LabelFrame(control_frame, text="视图控制", padding=10)
        view_frame.pack(fill=tk.X, pady=5)

        ttk.Button(view_frame, text="放大 (+)", command=self.zoom_in).pack(fill=tk.X, pady=2)
        ttk.Button(view_frame, text="缩小 (-)", command=self.zoom_out).pack(fill=tk.X, pady=2)
        ttk.Button(view_frame, text="重置视图", command=self.reset_view).pack(fill=tk.X, pady=2)

        # 调试视图选择
        ttk.Label(view_frame, text="显示模式:").pack(anchor=tk.W, pady=(10, 0))
        self.view_mode = tk.StringVar(value="result")
        ttk.Radiobutton(view_frame, text="检测结果", variable=self.view_mode,
                       value="result", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="边缘检测", variable=self.view_mode,
                       value="edges", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="所有线段", variable=self.view_mode,
                       value="all_lines", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="合并线段", variable=self.view_mode,
                       value="merged_lines", command=self.update_display).pack(anchor=tk.W)

        # 检测结果
        result_frame = ttk.LabelFrame(control_frame, text="检测结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 右侧画布
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定事件
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

    def on_param_change(self, event=None):
        """参数变化回调"""
        self.min_length_label.config(text=str(self.min_length_var.get()))
        self.angle_label.config(text=f"{self.angle_threshold_var.get():.1f}°")
        self.min_dist_label.config(text=str(self.min_dist_var.get()))
        self.max_dist_label.config(text=str(self.max_dist_var.get()))

    def on_template_toggle(self):
        """模板匹配开关切换"""
        if self.use_template_matching.get() and self.template_detector is None:
            self.result_text.insert(tk.END, "正在初始化模板检测器...\n")
            self.root.update()
            try:
                self.template_detector = TemplateParkingDetector(
                    template_path=str(DEFAULT_GROUND_TEMPLATE_PATH)
                )
                self.result_text.insert(tk.END, "模板检测器初始化完成\n")
            except Exception as e:
                self.result_text.insert(tk.END, f"初始化失败: {e}\n")
                self.use_template_matching.set(False)

    def update_detector(self):
        """更新检测器参数"""
        self.detector = ParkingLineDetector(
            min_line_length=self.min_length_var.get(),
            max_line_gap=10,
            parallel_angle_threshold=self.angle_threshold_var.get(),
            min_distance=self.min_dist_var.get(),
            max_distance=self.max_dist_var.get()
        )

    def load_image(self):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        self.original_image = self._safe_imread(file_path)
        if self.original_image is None:
            messagebox.showerror("错误", "无法加载图片")
            return

        self.reset_view()
        self.detect()

    @staticmethod
    def _safe_imread(path):
        """
        Read image robustly, including Unicode filenames on Windows.
        """
        img = cv2.imread(path)
        if img is not None:
            return img
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size == 0:
                return None
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def detect(self):
        """执行检测"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return

        self.update_detector()

        # 获取调试图像
        self.debug_images = self.detector.get_debug_images(self.original_image)

        # 执行检测
        if self.use_template_matching.get() and self.template_detector is not None:
            # 使用模板匹配
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "使用模板匹配检测...\n")

            # 先检测线段（需要先预处理得到边缘图）
            edges = self.detector.preprocess(self.original_image)
            lines = self.detector.detect_lines(edges)
            merged_lines = self._merge_lines_for_template(lines)

            self.result_text.insert(tk.END, f"检测到 {len(merged_lines)} 条合并线段\n")

            # 模板匹配
            matches = self.template_detector.detect(
                self.original_image,
                merged_lines,
                score_threshold=0.70
            )

            # 可视化模板匹配结果
            self.result_image = self.template_detector.visualize(self.original_image, matches)

            # 更新结果文本
            self.result_text.insert(tk.END, f"\n检测到 {len(matches)} 个车位\n\n")
            for i, match in enumerate(matches):
                self.result_text.insert(tk.END, f"车位 #{i+1}:\n")
                self.result_text.insert(tk.END, f"  匹配分数: {match.score:.3f}\n")
                self.result_text.insert(tk.END, f"  模板ID: {match.template_id}\n\n")
        else:
            # 使用原有的平行线检测
            self.parking_line_pairs = self.detector.detect_parking_lines(self.original_image)
            self.result_image = self.detector.draw_results(self.original_image, self.parking_line_pairs)

            # 更新结果文本
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"检测到 {len(self.parking_line_pairs)} 对车位线\n\n")

            for idx, (line1, line2, mean_color) in enumerate(self.parking_line_pairs):
                distance = self.detector.calculate_distance(line1, line2)
                len1 = self.detector.calculate_length(line1)
                len2 = self.detector.calculate_length(line2)
                angle1 = self.detector.calculate_angle(line1)
                angle2 = self.detector.calculate_angle(line2)

                self.result_text.insert(tk.END, f"车位线 #{idx+1}:\n")
                self.result_text.insert(tk.END, f"  距离: {distance:.1f} 像素\n")
                self.result_text.insert(tk.END, f"  长度: {len1:.1f}, {len2:.1f} 像素\n")
                self.result_text.insert(tk.END, f"  角度: {angle1:.1f}°, {angle2:.1f}°\n")
                self.result_text.insert(tk.END, f"  颜色: BGR{tuple(int(c) for c in mean_color)}\n\n")

        self.update_display()

    def _merge_lines_for_template(self, lines):
        """
        兼容旧/新检测器接口，并将线段统一为 [x1, y1, x2, y2] 格式。
        """
        # 新版接口
        if hasattr(self.detector, "merge_collinear_lines"):
            merged = self.detector.merge_collinear_lines(lines)
        # 旧版接口
        elif hasattr(self.detector, "merge_lines"):
            merged = self.detector.merge_lines(lines)
        else:
            merged = lines if lines is not None else []

        normalized = []
        for line in merged:
            arr = np.asarray(line)
            # 处理 [[x1,y1,x2,y2]] -> [x1,y1,x2,y2]
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.size >= 4:
                normalized.append(arr.astype(np.float32).reshape(-1)[:4])
        return normalized

    def update_display(self):
        """更新显示"""
        # 根据视图模式选择要显示的图像
        view_mode = self.view_mode.get()

        if view_mode == "result":
            display_source = self.result_image
        elif view_mode == "edges":
            display_source = self.debug_images.get('edges')
        elif view_mode == "all_lines":
            display_source = self.debug_images.get('all_lines')
        elif view_mode == "merged_lines":
            display_source = self.debug_images.get('merged_lines')
        else:
            display_source = self.result_image

        if display_source is None:
            return

        # 如果是灰度图，转换为BGR
        if len(display_source.shape) == 2:
            display_source = cv2.cvtColor(display_source, cv2.COLOR_GRAY2BGR)

        h, w = display_source.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # 计算缩放后的尺寸
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)

        # 缩放图像
        resized = cv2.resize(display_source, (scaled_w, scaled_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self.display_image = ImageTk.PhotoImage(pil_image)

        # 清空画布并显示
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.display_image)

    def zoom_in(self):
        """放大"""
        self.scale *= 1.2
        self.update_display()

    def zoom_out(self):
        """缩小"""
        self.scale /= 1.2
        self.update_display()

    def reset_view(self):
        """重置视图"""
        if self.original_image is None:
            return

        h, w = self.original_image.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # 计算适应画布的缩放比例
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        self.scale = min(scale_w, scale_h, 1.0)

        self.offset_x = 0
        self.offset_y = 0
        self.update_display()

    def on_drag_start(self, event):
        """开始拖动"""
        self.drag_start = (event.x, event.y)

    def on_drag_motion(self, event):
        """拖动中"""
        if self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (event.x, event.y)
            self.update_display()

    def on_mousewheel(self, event):
        """鼠标滚轮缩放"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def save_result(self):
        """保存结果"""
        if self.result_image is None:
            messagebox.showwarning("警告", "没有可保存的结果")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("所有文件", "*.*")]
        )

        if file_path:
            cv2.imwrite(file_path, self.result_image)
            messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")


def main():
    root = tk.Tk()
    app = ParkingLineViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
