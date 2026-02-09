# FST 车位分类器工具

纯视觉 CNN 多头分类器，输入地面视角停车位照片，输出符合 FST Schema v1 的结构化 JSON。

**核心特性：**
- ❌ 不依赖 LLM / 云端 API
- ✅ 用自有标注数据训练
- ✅ ONNX 导出，任意电脑本地 CPU 推理 (<100ms/张)
- ✅ 内置标注工具 + Gradio 推理界面
- ✅ 可 PyInstaller 打包为独立 .exe 分发

---

## 快速开始

### 1. 安装

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -e .
```

### 2. 标注数据

将采集的停车位照片放入 `dataset/images/`，然后启动标注工具：

```bash
python -m fst.label_tool --images dataset/images --labels dataset/labels
```

标注工具操作：
- **A / ←** 上一张，**D / →** 下一张
- **S** 保存当前标注（切换图片时也会自动保存）
- 下拉菜单选择车位类型、泊车动作等
- 9 宫格选择各方位障碍物

### 3. 训练模型

```bash
python -m fst.train \
    --data ./dataset \
    --output ./checkpoints \
    --backbone efficientnet_b0 \
    --batch 32 \
    --epochs1 10 \
    --epochs2 30
```

训练策略：
- **Stage 1** (10 epochs): 冻结 backbone，只训练分类头，快速收敛
- **Stage 2** (30 epochs): 解冻 backbone 最后 2 层，精细微调

### 4. 导出 ONNX

```bash
python -m fst.export_onnx \
    --checkpoint checkpoints/best_model.pth \
    --output fst_classifier.onnx
```

### 5. 运行推理界面

```bash
python -m fst.app --model fst_classifier.onnx --port 7860
```

浏览器会自动打开，拖入照片即可查看分类结果。

### 6. 打包分发 (可选)

```bash
pip install pyinstaller
python scripts/build_exe.py --model fst_classifier.onnx
# 产出: dist/fst_app.exe (Windows) 或 dist/fst_app (Linux/Mac)
```

---

## 项目结构

```
fst_classifier_tool/
├── ARCHITECTURE.md          # 详细技术架构文档
├── pyproject.toml
├── requirements.txt
├── schema/
│   └── fst.v1.schema.json   # JSON Schema (draft-07)
├── fst/
│   ├── models.py            # Pydantic 数据模型 + 枚举 + FST Text 生成
│   ├── network.py           # 多头 CNN 模型 (EfficientNet backbone)
│   ├── dataset.py           # PyTorch Dataset + FST 感知数据增强
│   ├── train.py             # 两阶段训练脚本
│   ├── export_onnx.py       # ONNX 导出
│   ├── inference.py         # ONNX 推理引擎
│   ├── label_tool.py        # tkinter 标注工具
│   ├── app.py               # Gradio 推理界面
│   └── schema_validate.py   # JSON Schema 校验
├── scripts/
│   ├── build_exe.py         # PyInstaller 打包
│   └── evaluate.py          # 评估脚本 (accuracy + confusion matrix)
└── dataset/                 # 数据目录 (不入版本控制)
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
| 验证可行性 | 200-500 | slot_type + marking 基本可用 |
| 日常可用 | 1000-3000 | 主要场景覆盖 |
| 生产级 | 5000+ | 全量 FST Level-3 稳定输出 |

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

## 评估

```bash
python scripts/evaluate.py --model fst_classifier.onnx --data ./dataset_test
```

输出各 head 的 precision/recall/F1 和逐方位准确率。

---

## 模型选择参考

| Backbone | 参数量 | ONNX 大小 | CPU 推理 | 适用场景 |
|----------|--------|-----------|----------|----------|
| efficientnet_b0 | 5.3M | ~20MB | ~50ms | **推荐默认** |
| mobilenetv3_large | 5.4M | ~18MB | ~30ms | 需要更快速度 |
| efficientnet_b2 | 9.1M | ~35MB | ~80ms | 需要更高精度 |

通过 `--backbone` 参数切换：
```bash
python -m fst.train --data ./dataset --backbone mobilenetv3_large_100
```
