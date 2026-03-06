# 项目结构说明

## 目录结构

```
fst_classifier_tool/
├── cli.py                      # 统一命令行入口
├── README.md                   # 项目说明
├── ARCHITECTURE.md             # 技术架构文档
├── AUTO_LABEL_GUIDE.md         # 半自动标注工具使用指南
├── MIGRATION_GUIDE.md          # v1 迁移指南
├── requirements.txt            # Python 依赖
├── pyproject.toml              # 项目配置
├── .gitignore                  # Git 忽略规则
├── yolo_mapping_config.yaml    # YOLO 类别映射配置
├── yolov8n.pt                  # YOLO 预训练权重
│
├── schema/                     # JSON Schema 定义
│   └── fst.v1.schema.json
│
├── fst/                        # 核心代码包
│   ├── __init__.py
│   ├── models.py               # Pydantic 数据模型 + FST 枚举
│   ├── network.py              # 多头 CNN 网络架构
│   ├── dataset.py              # PyTorch Dataset + 数据增强
│   ├── train.py                # 两阶段训练脚本
│   ├── export_onnx.py          # ONNX 模型导出
│   ├── inference.py            # ONNX 推理引擎
│   ├── hybrid_inference.py     # 混合推理（CNN + YOLO）
│   ├── label_tool.py           # 传统手动标注工具
│   ├── auto_label_tool.py      # 半自动标注工具（推荐）
│   ├── app.py                  # Gradio Web 界面
│   ├── yolo_detector.py        # YOLO 物体检测封装
│   ├── vanishing_point.py      # 消失点检测算法
│   ├── marking_point_detector.py    # 车位线检测（传统 CV）
│   ├── dl_marking_point_detector.py # 车位线检测（深度学习）
│   ├── parking_geometry.py     # 几何分析 + 9宫格映射
│   └── schema_validate.py      # JSON Schema 校验
│
├── scripts/                    # 辅助脚本
│   ├── build_exe.py            # PyInstaller 打包
│   └── evaluate.py             # 模型评估
│
├── examples/                   # 使用示例
│   ├── hybrid_inference_demo.py    # 单张图片推理示例
│   └── batch_inference.py          # 批量推理示例
│
└── dataset/                    # 数据目录（不入版本控制）
    ├── images/                 # 原始图片
    └── labels/                 # 标注 JSON 文件
```

## 核心模块说明

### 1. 数据模型层 (`models.py`)
- FST 枚举定义（车位类型、泊车动作、障碍物等）
- Pydantic 数据模型（类型安全 + 自动校验）
- FST 文本生成（结构化 JSON → DSL 文本）

### 2. 网络架构层 (`network.py`)
- 多头 CNN 分类器
- 支持多种 backbone（EfficientNet、MobileNet 等）
- 4 个分类头：slot_type、maneuver、marking、special_scene

### 3. 训练层 (`train.py`, `dataset.py`)
- 两阶段训练策略（冻结 → 微调）
- FST 感知数据增强（保持车位几何特征）
- 多任务损失函数

### 4. 推理层 (`inference.py`, `hybrid_inference.py`)
- ONNX 推理引擎（跨平台部署）
- 混合推理：CNN 分类 + YOLO 检测 + 几何分析
- 9 宫格障碍物映射

### 5. 标注工具层 (`label_tool.py`, `auto_label_tool.py`)
- 传统手动标注：完全人工标注所有字段
- 半自动标注（推荐）：
  - YOLO 自动检测物体
  - 车位线自动检测
  - 透视分析 + 9宫格自动划分
  - 人工校验场景属性

### 6. 几何分析层
- `vanishing_point.py`: 消失点检测（透视分析）
- `marking_point_detector.py`: 车位线检测（传统 CV）
- `dl_marking_point_detector.py`: 车位线检测（深度学习）
- `parking_geometry.py`: 9宫格划分 + 物体映射

### 7. 物体检测层 (`yolo_detector.py`)
- YOLO 模型封装
- 类别映射（COCO → FST 障碍物）
- 检测结果过滤

## 使用流程

### 开发流程
1. 采集停车位照片 → `dataset/images/`
2. 运行半自动标注 → `python cli.py label --auto`
3. 训练模型 → `python cli.py train`
4. 导出 ONNX → `python cli.py export`
5. 推理测试 → `python cli.py infer --image test.jpg`

### 部署流程
1. 导出 ONNX 模型
2. 打包可执行文件 → `python scripts/build_exe.py`
3. 分发 `dist/fst_app.exe`（包含模型权重）

## 配置文件

### `yolo_mapping_config.yaml`
YOLO 类别到 FST 障碍物的映射规则：
```yaml
coco_to_fst:
  2: VEHICLE      # car
  3: VEHICLE      # motorcycle
  5: VEHICLE      # bus
  ...
```

### `requirements.txt`
Python 依赖包列表

### `pyproject.toml`
项目元数据 + 打包配置

## 数据目录（不入版本控制）

```
dataset/
├── images/
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
└── labels/
    ├── IMG_0001.json
    ├── IMG_0002.json
    └── ...
```

每张图片对应一个同名 JSON 标注文件。

## 输出目录（不入版本控制）

```
checkpoints/          # 训练检查点
├── best_model.pth
├── best_stage1.pth
└── history.json

models/               # 导出的 ONNX 模型
└── fst_classifier.onnx

output/               # 推理结果
├── result_001.json
├── result_001_vis.jpg
└── ...
```
