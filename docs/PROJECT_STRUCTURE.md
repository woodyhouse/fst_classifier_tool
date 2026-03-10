# 项目结构说明

当前仓库已经按更接近常见 Python 后端/应用项目的方式整理为“代码、脚本、测试、文档、资源、运行数据”分层。

## 顶层目录

```text
fst_classifier_tool/
├── cli.py
├── README.md
├── pyproject.toml
├── requirements.txt
├── fst/
├── schema/
├── examples/
├── scripts/
├── tests/
├── docs/
├── configs/
├── resources/
├── dataset/
├── checkpoints/
├── models/
├── output/
├── weights/
└── external/
```

## 目录职责

- `fst/`
  核心应用代码。包含训练、推理、Web 界面、YOLO 封装、模板检测、数据模型以及统一路径定义。

- `schema/`
  FST 的 JSON Schema 定义文件。

- `examples/`
  对外演示用的推理示例。

- `scripts/`
  辅助脚本与开发工具，例如打包、评估、模板生成、模板可视化、DMPR 相关工具。  
  这里包含 `sitecustomize.py`，保证直接运行 `python scripts/...` 时也能导入项目包。

- `tests/`
  回归、实验和手工验证脚本。  
  同样包含 `sitecustomize.py`，避免脚本迁移到子目录后出现 `ModuleNotFoundError: fst`。

- `docs/`
  架构、迁移、标注指南、模型测试说明等文档。

- `configs/`
  项目配置文件，例如 `yolo_mapping_config.yaml`。

- `resources/`
  静态资源和本地素材。
  - `resources/templates/`: 预生成模板库
  - `resources/samples/`: 示例图片

## 运行期目录

这些目录主要承载本地数据、模型或生成结果，属于“运行数据层”，保留在根目录便于直接查看和管理：

- `dataset/`: 标注数据集
- `checkpoints/`: 训练检查点
- `models/`: 导出模型与 YOLO 权重
- `output/`: 推理输出
- `weights/`: 其他本地权重
- `external/`: 外部依赖仓库

## 路径约定

统一路径常量位于 `fst/paths.py`。  
以后新增脚本或测试时，优先从这里引用项目目录，而不是在文件里硬编码 `dataset/...`、`parking_templates.pkl`、`external/DMPR-PS` 这类相对路径。
