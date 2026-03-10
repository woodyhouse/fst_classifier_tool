# FST 车位分类器 — 技术架构文档

## 1. 目标概述

构建一个**纯视觉分类模型**，输入地面视角停车位照片，输出符合 FST Schema v1 的结构化 JSON。

**核心约束：**
- ❌ 不依赖 LLM / VLM / 任何云端 API
- ✅ 用自有标注数据训练轻量 CNN 多头分类器
- ✅ 导出 ONNX，打包为桌面应用，任意电脑本地运行
- ✅ 输出完全对齐 FST v1 Schema（slot_type / maneuver / special_scene / obstacles 1-7+P）

---

## 2. 模型架构：Multi-Head CNN Classifier

```
┌─────────────────────────────────────────────────┐
│              Input Image (384×384 RGB)           │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         Backbone: EfficientNet-B0 / B2          │
│         (ImageNet pretrained, frozen first)      │
│         Output: 1280-d feature vector            │
└─────────────────┬───────────────────────────────┘
                  │
          ┌───────┴───────┐
          │  Shared Neck   │  (FC 1280→512, BN, ReLU, Dropout 0.3)
          └───────┬───────┘
                  │
    ┌─────────────┼─────────────┬──────────────┬──────────────┐
    │             │             │              │              │
┌───▼───┐   ┌───▼───┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
│Head A │   │Head B │   │ Head C  │   │ Head D  │   │ Head E  │
│SlotTyp│   │Maneuvr│   │SpecScen │   │Obstacle │   │Marking  │
│ 4-cls │   │ 5-cls │   │ 14-cls  │   │9×12-cls │   │ 3 sub   │
│softmax│   │softmax│   │ sigmoid │   │per-pos  │   │ heads   │
└───────┘   └───────┘   └─────────┘   │softmax  │   └─────────┘
                                       └─────────┘

Head A: slot_type        → 4类 (PERPENDICULAR/PARALLEL/ANGLED/UNKNOWN)
Head B: maneuver         → 5类 (PARK_IN/PARK_OUT/HEAD_IN/TAIL_OUT/UNKNOWN)
Head C: special_scene    → 14标签多标签 (P0×2 + P1×12), sigmoid
Head D: obstacles        → 9个位置 × 12类障碍物, 每位置独立 softmax
Head E: marking          → 3个子头 (color×6, visibility×4, style×3)
```

### 为什么选 EfficientNet-B0？
- 参数量仅 5.3M，ONNX 推理 <50ms（CPU）
- ImageNet pretrained 提供强大低层特征（边缘/纹理/颜色）
- 足够捕捉车位线、地面材质、障碍物形状
- 备选：MobileNetV3-Large（更快）、EfficientNet-B2（更准但稍慢）

---

## 3. 数据标注方案

### 3.1 标注文件格式

每张图片对应一个 JSON 标注文件（与 FST Schema 完全对齐）：

```
dataset/
  images/
    0001.jpg
    0002.jpg
    ...
  labels/
    0001.json    ← 标注 JSON
    0002.json
    ...
```

**标注 JSON 示例（精简版，只保留训练所需字段）：**

```json
{
  "image_id": "0001",
  "slot_type": "PERPENDICULAR",
  "maneuver": "PARK_IN",
  "search_direction": "RIGHT",
  "marking": {
    "line_color": "WHITE",
    "line_visibility": "CLEAR",
    "line_style": "SOLID"
  },
  "special_scene": {
    "P0": [],
    "P1": ["BRICK_GRASS"]
  },
  "obstacles": {
    "1": "EMPTY",
    "2": "EMPTY",
    "3": "LAMP",
    "4": "EMPTY",
    "5": "CURB",
    "6": "EMPTY",
    "7": "EMPTY",
    "P_LEFT": "VEHICLE",
    "P_RIGHT": "EMPTY"
  }
}
```

### 3.2 标注工具（内置简易 GUI）

项目自带 `label_tool.py`，基于 tkinter，支持：
- 加载图片文件夹
- 下拉菜单选择各字段枚举值
- 9 宫格按钮选择各方位障碍物
- 自动保存 JSON 到 labels/ 目录
- 支持快捷键快速标注

### 3.3 数据量建议

| 阶段 | 图片数 | 预期效果 |
|------|--------|----------|
| MVP-0 验证 | 200-500 | slot_type + marking 基本可用 |
| MVP-1 可用 | 1000-3000 | 主要场景覆盖，special_scene 可用 |
| MVP-2 生产 | 5000-10000 | 全量 FST-3 级输出稳定 |

**数据增强策略（内置）：** 随机裁剪、水平翻转（镜像时同步交换 1↔3/4↔6/P_LEFT↔P_RIGHT）、颜色抖动、亮度/对比度变换、高斯模糊。

---

## 4. 训练流水线

### 4.1 两阶段训练策略

**Stage 1: Backbone Frozen（5-10 epochs）**
- 冻结 EfficientNet backbone
- 只训练 Shared Neck + 所有 Head
- 学习率 1e-3，AdamW

**Stage 2: Fine-tune All（20-50 epochs）**
- 解冻 backbone 最后 2 个 block
- 学习率 1e-4，cosine annealing
- Early stopping (patience=10)

### 4.2 损失函数

```
Total Loss = λ_A * CE(slot_type)
           + λ_B * CE(maneuver)
           + λ_C * BCE(special_scene)        # 多标签
           + λ_D * Σ_{pos} CE(obstacle_pos)  # 9个位置各自CE
           + λ_E * [CE(color) + CE(vis) + CE(style)]

默认权重: λ_A=1.0, λ_B=1.0, λ_C=0.8, λ_D=0.5, λ_E=0.6
```

对类别不均衡的 head（如 special_scene），使用 Focal Loss 或类别权重。

---

## 5. 推理与导出

### 5.1 ONNX 导出

```python
torch.onnx.export(model, dummy_input, "fst_classifier.onnx",
                  input_names=["image"],
                  output_names=["slot_type", "maneuver", "special_scene",
                               "obs_1","obs_2","obs_3","obs_4","obs_5",
                               "obs_6","obs_7","obs_pl","obs_pr",
                               "line_color","line_vis","line_style"],
                  dynamic_axes={"image": {0: "batch"}})
```

### 5.2 推理流程

```
图片 → Resize(384) → Normalize → ONNX Runtime → 各 Head logits
  → argmax/sigmoid → 枚举映射 → FST JSON + fst_text DSL
```

推理速度目标：
- CPU (i5/Ryzen5): < 100ms/张
- GPU (GTX1060+): < 20ms/张

---

## 6. 打包分发方案

### 6.1 架构

```
fst_classifier_app/
  ├── fst_classifier.onnx          # 模型文件 (~20MB)
  ├── app.py                       # Gradio Web UI
  ├── inference.py                 # ONNX 推理引擎
  ├── models.py                    # Pydantic FST 数据模型
  ├── schema/fst.v1.schema.json    # JSON Schema
  └── requirements.txt
```

### 6.2 三种分发方式

| 方式 | 适合人群 | 大小 | 说明 |
|------|---------|------|------|
| **Python + pip** | 开发者 | ~25MB | `pip install -e .` + `fst serve` |
| **PyInstaller 单文件** | 普通用户 | ~150MB | 双击即用，内嵌 Python + ONNX Runtime |
| **Docker 镜像** | 服务端部署 | ~300MB | `docker run -p 7860:7860 fst-tool` |

推荐方案：**Gradio + PyInstaller**
- Gradio 提供浏览器 UI（拖拽图片 → 显示结果）
- PyInstaller 打包成 .exe（Windows）/ .app（Mac）
- 无需安装 Python，双击启动浏览器界面

---

## 7. 项目文件清单

```
fst_classifier_tool/
├── ARCHITECTURE.md              # 本文档
├── pyproject.toml               # 项目配置
├── requirements.txt             # 依赖
│
├── schema/
│   └── fst.v1.schema.json       # JSON Schema（来自工程骨架）
│
├── fst/
│   ├── __init__.py
│   ├── models.py                # Pydantic 数据模型（复用工程骨架）
│   ├── schema_validate.py       # Schema 校验
│   ├── dataset.py               # PyTorch Dataset + 数据增强
│   ├── network.py               # 多头 CNN 模型定义
│   ├── train.py                 # 训练脚本
│   ├── export_onnx.py           # ONNX 导出
│   ├── inference.py             # ONNX 推理引擎
│   ├── label_tool.py            # 标注工具 GUI
│   └── app.py                   # Gradio 推理 UI
│
├── scripts/
│   ├── build_exe.py             # PyInstaller 打包脚本
│   └── evaluate.py              # 评估脚本（accuracy / confusion matrix）
│
└── dataset/                     # 数据目录（不入版本控制）
    ├── images/
    └── labels/
```
