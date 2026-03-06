# FST 车位分类器工具 v2

混合架构：CNN 分类器 + YOLO 物体检测 + 几何分析，输入地面视角停车位照片，输出符合 FST Schema v1 的结构化 JSON。

**核心特性：**
- ✅ 混合架构：CNN (场景分类) + YOLO (物体检测) + 几何算法 (空间推理)
- ✅ 数据需求降低 60%：无需标注 9 宫格障碍物
- ✅ 泛化能力强：YOLO 预训练模型识别常见物体
- ✅ 可解释性好：可视化检测框和 9 宫格
- ✅ 本地推理：不依赖 LLM / 云端 API
- ✅ 内置标注工具 + Gradio 推理界面

---

## 快速开始

### 1. 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -e .
pip install opencv-python ultralytics
```

### 2. 使用 CLI

项目提供统一的命令行入口：

```bash
# 查看帮助
python cli.py --help

# 半自动标注（推荐）
python cli.py label --auto

# 传统手动标注
python cli.py label

# 训练模型
python cli.py train --batch 32 --epochs1 10 --epochs2 30

# 导出 ONNX
python cli.py export

# 单张推理
python cli.py infer --image test.jpg --visualize

# 批量推理
python cli.py infer --images ./test_images --visualize

# 启动 Web 界面
python cli.py app
```

### 3. 半自动标注工具

将采集的停车位照片放入 `dataset/images/`，然后：

```bash
python cli.py label --auto
```

**特性：**
- ✅ YOLO 自动检测物体（蓝色检测框）
- ✅ 车位线检测 + 透视分析（红色车位中心）
- ✅ 车位类型自动识别（垂直/水平/斜列）
- ✅ 9宫格自动划分（绿色网格）
- ✅ 物体自动映射到格子
- ⚠️ 人工校验场景属性

**快捷键：**
- **A / ←** 上一张，**D / →** 下一张
- **S** 保存，**R** 重新分析

详细说明见 [AUTO_LABEL_GUIDE.md](AUTO_LABEL_GUIDE.md)

---

## 项目结构

```
fst_classifier_tool/
├── cli.py                   # 统一命令行入口
├── README.md
├── ARCHITECTURE.md          # 技术架构文档
├── AUTO_LABEL_GUIDE.md      # 标注工具说明
├── MIGRATION_GUIDE.md       # v1 迁移指南
├── requirements.txt
├── pyproject.toml
├── schema/
│   └── fst.v1.schema.json
├── fst/                     # 核心代码
│   ├── models.py            # 数据模型
│   ├── network.py           # CNN 网络
│   ├── dataset.py           # 数据集
│   ├── train.py             # 训练脚本
│   ├── export_onnx.py       # ONNX 导出
│   ├── inference.py         # 推理引擎
│   ├── label_tool.py        # 手动标注工具
│   ├── auto_label_tool.py   # 半自动标注工具
│   ├── app.py               # Gradio 界面
│   ├── yolo_detector.py     # YOLO 检测器
│   ├── vanishing_point.py   # 消失点检测
│   ├── marking_point_detector.py  # 车位线检测
│   ├── parking_geometry.py  # 几何分析
│   └── schema_validate.py   # Schema 校验
├── scripts/
│   ├── build_exe.py         # 打包脚本
│   └── evaluate.py          # 评估脚本
├── examples/
│   ├── hybrid_inference_demo.py  # 单张推理示例
│   └── batch_inference.py        # 批量推理示例
└── dataset/                 # 数据目录（不入版本控制）
    ├── images/
    └── labels/
```

---

## 标注文件格式

每张图片对应一个同名 `.json` 文件：

```json
{
  "image_id": "IMG_0001",
  "slot_type": "PERPENDICULAR",
  "maneuver": "PARK_IN",
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
    "1": "EMPTY", "2": "EMPTY", "3": "LAMP",
    "4": "EMPTY", "5": "CURB",  "6": "EMPTY",
    "7": "EMPTY",
    "P_LEFT": "VEHICLE", "P_RIGHT": "EMPTY"
  }
}
```

---

## 数据量建议

| 阶段 | 图片数 | 预期效果 |
|------|--------|----------|
| 验证可行性 | 100-300 | slot_type + marking 基本可用 |
| 日常可用 | 500-1500 | 主要场景覆盖 |
| 生产级 | 2000+ | 全量 FST Level-3 稳定输出 |

**注：** 相比 v1 纯 CNN 方案，数据需求降低约 60%

---

## FST 输出示例

输入一张垂直车位照片，模型输出：

**FST 文本 (DSL):** `砖草3路灯5路沿车空垂直泊入`

**结构化 JSON:**
```json
{
  "schema_version": "fst.v1",
  "fst_level": 3,
  "slot": {
    "slot_type": "PERPENDICULAR",
    "marking": {"line_color": "WHITE", "line_visibility": "CLEAR", "line_style": "SOLID"}
  },
  "maneuver": "PARK_IN",
  "special_scene": {"P0": [], "P1": ["BRICK_GRASS"]},
  "obstacles": {
    "pos_map": {
      "1": ["EMPTY"], "2": ["EMPTY"], "3": ["LAMP"],
      "4": ["EMPTY"], "5": ["CURB"],  "6": ["EMPTY"],
      "7": ["EMPTY"], "P_LEFT": ["VEHICLE"], "P_RIGHT": ["EMPTY"]
    }
  }
}
```

---

## 模型选择参考

### CNN Backbone

| Backbone | 参数量 | 大小 | CPU 推理 | 适用场景 |
|----------|--------|------|----------|----------|
| efficientnet_b0 | 5.3M | ~15MB | ~30ms | **推荐默认** |
| mobilenetv3_large | 5.4M | ~13MB | ~20ms | 需要更快速度 |
| efficientnet_b2 | 9.1M | ~25MB | ~50ms | 需要更高精度 |

### YOLO 模型

| 模型 | 大小 | CPU 推理 | 精度 | 适用场景 |
|------|------|----------|------|----------|
| yolov8n | 6MB | ~50ms | 中等 | **推荐默认** |
| yolov8s | 22MB | ~80ms | 高 | 需要更高精度 |
| yolov8m | 52MB | ~150ms | 很高 | GPU 环境 |

---

## 性能对比

| 指标 | v1 (纯 CNN) | v2 (混合架构) |
|------|-------------|---------------|
| 总推理时间 | ~50ms | ~80ms |
| 模型大小 | ~20MB | ~21MB (15MB CNN + 6MB YOLO) |
| 训练数据需求 | 5000+ 张 | 2000+ 张 |
| 泛化能力 | 中等 | 强 |
| 可解释性 | 低 | 高 |

---

## 从 v1 迁移

如果你正在使用旧版本（纯 CNN），请参考 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) 进行迁移。
