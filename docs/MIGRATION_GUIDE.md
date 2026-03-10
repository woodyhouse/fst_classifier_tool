# 迁移指南：从纯 CNN 到混合架构

## 架构变更概述

### 旧架构（纯 CNN）
```
输入图片 → CNN 多头分类器 → FST JSON
              ├─ slot_type
              ├─ maneuver
              ├─ special_scene
              ├─ obstacles (9个头)  ← 已移除
              └─ marking
```

### 新架构（混合方案）
```
输入图片
  ↓
┌─────────────────┬──────────────────┐
│  CNN 分类器      │   YOLO 检测      │
│  (4个头)        │   (物体位置)     │
├─────────────────┼──────────────────┤
│ • slot_type     │ • 车辆           │
│ • maneuver      │ • 路沿           │
│ • marking       │ • 柱子/墙        │
│ • special_scene │ • 锥桶等         │
└─────────────────┴──────────────────┘
          ↓
    车位线检测 + 透视分析
          ↓
    物体映射到 9 宫格
          ↓
      FST JSON 输出
```

## 主要变更

### 1. 模型架构简化

**移除的部分：**
- 9 个 obstacle 分类头（`obs_0` ~ `obs_8`）
- 相关的损失函数和准确率计算

**保留的部分：**
- `slot_type` (车位类型)
- `maneuver` (泊车动作)
- `marking` (车位线特征)
- `special_scene` (特殊场景)

### 2. 新增模块

**`fst/yolo_detector.py`**
- 使用预训练 YOLOv8 检测物体
- 支持 COCO 80 类物体
- 自动映射到 FST ObstacleType

**`fst/parking_geometry.py`**
- 车位线检测（Canny + Hough）
- 9 宫格划分算法
- 物体-格子映射

**`fst/hybrid_inference.py`**
- 整合 CNN + YOLO + 几何分析
- 统一推理接口

## 迁移步骤

### 步骤 1: 安装新依赖

```bash
pip install opencv-python>=4.8 ultralytics>=8.0
```

### 步骤 2: 重新训练模型

由于模型架构变更，需要重新训练：

```bash
# 使用现有标注数据（只需 slot_type/maneuver/marking/special_scene）
python -m fst.train \
    --data ./dataset \
    --output ./checkpoints_v2 \
    --backbone efficientnet_b0 \
    --batch 32 \
    --epochs1 10 \
    --epochs2 30
```

**注意：**
- 旧的标注文件中的 `obstacles` 字段会被忽略
- 训练速度会更快（参数量减少约 40%）
- 模型文件更小（~15MB vs ~20MB）

### 步骤 3: 使用新的推理引擎

**旧方式（已废弃）：**
```python
from fst.inference import ONNXInferenceEngine

engine = ONNXInferenceEngine("model.onnx")
result = engine.infer(image)
```

**新方式：**
```python
from fst.hybrid_inference import HybridInferenceEngine

engine = HybridInferenceEngine(
    cnn_model_path="checkpoints_v2/best_model.pth",
    yolo_model_size="n",  # nano 最快
    device="cpu"
)
result = engine.infer(image, image_id="test_001")
```

### 步骤 4: 验证结果

使用旧标注数据验证新方案的准确性：

```python
from fst.hybrid_inference import HybridInferenceEngine
import json

engine = HybridInferenceEngine(...)

# 读取旧标注（包含 obstacles ground truth）
with open("dataset/labels/IMG_0001.json") as f:
    ground_truth = json.load(f)

# 新方案推理
result = engine.infer(image)

# 对比 obstacles 准确率
gt_obstacles = ground_truth["obstacles"]
pred_obstacles = result.obstacles.pos_map

# 计算准确率...
```

## 性能对比

| 指标 | 旧架构 (纯 CNN) | 新架构 (混合) |
|------|----------------|--------------|
| 模型大小 | ~20MB | ~15MB (CNN) + ~6MB (YOLO-n) |
| 推理速度 | ~50ms | ~80ms (CNN 30ms + YOLO 50ms) |
| 参数量 | ~8M | ~5M (CNN only) |
| 训练数据需求 | 5000+ 张 | 1000-2000 张 |
| 泛化能力 | 中等 | 强（YOLO 预训练） |
| 可解释性 | 低 | 高（可视化检测框） |

## 优势

1. **数据需求降低 60%**：不需要标注 9 宫格障碍物
2. **泛化能力提升**：YOLO 预训练模型见过的物体都能识别
3. **可调试性强**：可以可视化检测框和 9 宫格
4. **更易扩展**：新增障碍物类型无需重新训练

## 注意事项

### YOLO 模型选择

| 模型 | 大小 | 速度 | 精度 | 推荐场景 |
|------|------|------|------|----------|
| yolov8n | 6MB | 最快 | 中等 | **默认推荐** |
| yolov8s | 22MB | 快 | 高 | 需要更高精度 |
| yolov8m | 52MB | 中等 | 很高 | GPU 环境 |

### 首次运行

首次运行时 YOLO 会自动下载预训练模型（~6MB），请确保网络连接。

### 离线部署

如需离线部署，提前下载模型：

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 常见问题

**Q: 旧模型还能用吗？**
A: 旧模型（包含 obstacle 头）仍可加载，但推荐重新训练以获得更好性能。

**Q: 标注工具需要更新吗？**
A: 标注工具仍可使用，但 obstacles 字段仅用于验证，不参与训练。

**Q: 如何调试几何分析？**
A: 使用 `engine.visualize()` 方法可视化 9 宫格和检测框。

**Q: YOLO 检测不准怎么办？**
A: 可以调整 `conf_threshold` 参数或使用更大的模型（yolov8s/m）。

## 下一步

1. 重新训练模型
2. 在测试集上验证准确率
3. 调整几何分析参数（如需要）
4. 部署到生产环境

如有问题，请参考 `examples/hybrid_inference_demo.py`。
