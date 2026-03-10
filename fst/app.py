"""
Gradio 推理 UI + 在线纠正 + 回流训练.

功能:
  1. 上传图片 → 模型预测 → 显示结果
  2. 预测结果可在页面上直接修正
  3. 点击"保存纠正"→ 图片+标注自动写入 dataset/ 用于下次训练

用法:
    python -m fst.app --model fst_classifier.onnx --port 7860
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image

from fst.inference import FSTInference
from fst.models import (
    SLOT_TYPE_CLASSES, MANEUVER_CLASSES, OBSTACLE_CLASSES,
    LINE_COLOR_CLASSES, LINE_VIS_CLASSES, LINE_STYLE_CLASSES,
    SPECIAL_SCENE_CLASSES, POSITION_KEYS,
    build_fst_text,
)
from fst.paths import DATASET_DIR, DATASET_IMAGES_DIR, DATASET_LABELS_DIR

# 全局推理引擎
engine: FSTInference | None = None

# 数据集保存路径
IMAGES_DIR = DATASET_IMAGES_DIR
LABELS_DIR = DATASET_LABELS_DIR

# 中文映射
_SLOT_ZH = {"PERPENDICULAR": "垂直", "PARALLEL": "水平", "ANGLED": "鱼骨", "UNKNOWN": "未知"}
_MV_ZH = {"PARK_IN": "泊入", "PARK_OUT": "泊出", "HEAD_IN": "车头泊入", "TAIL_OUT": "车尾泊出", "UNKNOWN": "未知"}
_OBS_ZH = {
    "EMPTY": "空", "VEHICLE": "车", "CURB": "路沿", "WALL": "墙", "PILLAR": "柱",
    "CONE": "锥桶", "WATER_BARRIER": "水马", "FENCE": "栅栏", "LAMP": "路灯",
    "FIRE_BOX_SUSPENDED": "悬空消防箱", "BUSH": "灌木丛",
    "CIVIL_AIR_DEFENSE_DOOR": "人防门", "STAIRS": "阶梯", "TREE": "树",
    "THIN_POLE": "细杆", "PARKING_LOCK": "地锁", "PROTRUDING_WALL": "凸出墙体",
    "UNKNOWN": "未知",
}
_POS_ZH = {
    "1": "①远左", "2": "②远中", "3": "③远右",
    "4": "④近左", "5": "⑤近中", "6": "⑥近右",
    "7": "⑦车位内", "P_LEFT": "P左侧", "P_RIGHT": "P右侧",
}

# 下拉选项（显示: "PERPENDICULAR (垂直)"）
def _opts(classes, zh_map=None):
    if zh_map:
        return [f"{c} ({zh_map.get(c, '')})" for c in classes]
    return list(classes)

def _extract(display_str):
    """从 'PERPENDICULAR (垂直)' 提取 'PERPENDICULAR'."""
    return display_str.split(" ")[0].strip() if display_str else "UNKNOWN"

SLOT_OPTS = _opts(SLOT_TYPE_CLASSES, _SLOT_ZH)
MV_OPTS = _opts(MANEUVER_CLASSES, _MV_ZH)
OBS_OPTS = _opts(OBSTACLE_CLASSES, _OBS_ZH)
LC_OPTS = _opts(LINE_COLOR_CLASSES)
LV_OPTS = _opts(LINE_VIS_CLASSES)
LS_OPTS = _opts(LINE_STYLE_CLASSES)
SS_OPTS = [f"{c}" for c in SPECIAL_SCENE_CLASSES]


def _find_display(value, opts):
    """根据枚举值找到对应的显示字符串."""
    for o in opts:
        if o.startswith(value):
            return o
    return opts[-1] if opts else value


def predict(image):
    """推理回调: 返回预测结果填充到各个编辑框."""
    if engine is None:
        return ["⚠️ 模型未加载"] + ["UNKNOWN"] * 14 + [[], "{}"]
    if image is None:
        return ["请上传图片"] + ["UNKNOWN"] * 14 + [[], "{}"]

    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, str):
        img = Image.open(image)
    else:
        img = Image.fromarray(image)

    try:
        result = engine.predict_with_text(img)
    except Exception as e:
        return [f"❌ 推理出错: {str(e)}"] + ["UNKNOWN"] * 14 + [[], "{}"]

    slot = result.get("slot", {})
    mk = slot.get("marking", {})
    obs = result.get("obstacles", {}).get("pos_map", {})
    ss = result.get("special_scene", {})
    conf = result.get("confidence", {})

    # 提取各字段值
    slot_type = slot.get("slot_type", "UNKNOWN")
    if hasattr(slot_type, 'value'):
        slot_type = slot_type.value
    maneuver = result.get("maneuver", "UNKNOWN")
    if hasattr(maneuver, 'value'):
        maneuver = maneuver.value

    lc = mk.get("line_color", "UNKNOWN")
    if hasattr(lc, 'value'):
        lc = lc.value
    lv = mk.get("line_visibility", "UNKNOWN")
    if hasattr(lv, 'value'):
        lv = lv.value
    ls = mk.get("line_style", "UNKNOWN")
    if hasattr(ls, 'value'):
        ls = ls.value

    # 障碍物
    obs_values = []
    for pos in POSITION_KEYS:
        items = obs.get(pos, ["UNKNOWN"])
        val = items[0] if items else "UNKNOWN"
        if hasattr(val, 'value'):
            val = val.value
        obs_values.append(_find_display(val, OBS_OPTS))

    # 特殊场景
    active_ss = ss.get("P0", []) + ss.get("P1", [])

    # 状态信息
    fst_text = result.get("fst_text", "")
    overall_conf = conf.get("overall", 0)
    status = f"✅ 预测完成 | 置信度 {overall_conf:.1%} | FST: {fst_text}"

    json_str = json.dumps(result, indent=2, ensure_ascii=False, default=str)

    return [
        status,
        _find_display(slot_type, SLOT_OPTS),
        _find_display(maneuver, MV_OPTS),
        _find_display(lc, LC_OPTS),
        _find_display(lv, LV_OPTS),
        _find_display(ls, LS_OPTS),
        *obs_values,  # 9 个障碍物
        active_ss,
        json_str,
    ]


def save_correction(
    image,
    slot_type, maneuver,
    line_color, line_vis, line_style,
    obs_1, obs_2, obs_3, obs_4, obs_5, obs_6, obs_7, obs_pl, obs_pr,
    special_scenes,
):
    """保存纠正后的标注到 dataset/ 目录."""
    if image is None:
        return "⚠️ 请先上传图片"

    # 确保目录存在
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 生成唯一文件名
    ts = time.strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    image_id = f"corrected_{ts}_{uid}"

    # 保存图片
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, str):
        img = Image.open(image)
    else:
        img = Image.fromarray(image)

    img_path = IMAGES_DIR / f"{image_id}.jpg"
    img.convert("RGB").save(img_path, "JPEG", quality=95)

    # 收集特殊场景
    ss_list = special_scenes if special_scenes else []
    p0_list = [s for s in ss_list if s in ("DEAD_END", "NARROW_LANE")]
    p1_list = [s for s in ss_list if s not in ("DEAD_END", "NARROW_LANE")]

    # 构建标注 JSON
    label = {
        "image_id": image_id,
        "slot_type": _extract(slot_type),
        "maneuver": _extract(maneuver),
        "marking": {
            "line_color": _extract(line_color),
            "line_visibility": _extract(line_vis),
            "line_style": _extract(line_style),
        },
        "special_scene": {
            "P0": p0_list,
            "P1": p1_list,
        },
        "obstacles": {
            "1": _extract(obs_1),
            "2": _extract(obs_2),
            "3": _extract(obs_3),
            "4": _extract(obs_4),
            "5": _extract(obs_5),
            "6": _extract(obs_6),
            "7": _extract(obs_7),
            "P_LEFT": _extract(obs_pl),
            "P_RIGHT": _extract(obs_pr),
        },
    }

    lbl_path = LABELS_DIR / f"{image_id}.json"
    with open(lbl_path, "w", encoding="utf-8") as f:
        json.dump(label, f, indent=2, ensure_ascii=False)

    # 统计当前数据集大小
    n_images = len(list(IMAGES_DIR.glob("*")))
    n_labels = len(list(LABELS_DIR.glob("*.json")))

    return f"✅ 已保存! 图片: {img_path.name} | 标注: {lbl_path.name} | 数据集: {n_images}张图/{n_labels}个标注"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FST 车位分类器") as demo:
        gr.Markdown("# 🅿️ FST 车位分类器\n上传地面视角停车位照片，自动输出 FST 结构化描述。**预测有误可直接修正并保存回训练集。**")

        with gr.Row():
            # ── 左列: 图片 + 分析按钮 ──
            with gr.Column(scale=1):
                input_image = gr.Image(label="上传停车位照片", type="pil")
                btn_predict = gr.Button("🔍 开始分析", variant="primary", size="lg")
                status_box = gr.Textbox(label="状态", interactive=False)

            # ── 右列: 可编辑的预测结果 ──
            with gr.Column(scale=1):
                gr.Markdown("### 📝 预测结果（可修正）")

                with gr.Row():
                    dd_slot = gr.Dropdown(choices=SLOT_OPTS, label="车位类型", value=SLOT_OPTS[-1], interactive=True)
                    dd_maneuver = gr.Dropdown(choices=MV_OPTS, label="泊车动作", value=MV_OPTS[-1], interactive=True)

                with gr.Row():
                    dd_lc = gr.Dropdown(choices=LC_OPTS, label="标线颜色", value=LC_OPTS[-1], interactive=True)
                    dd_lv = gr.Dropdown(choices=LV_OPTS, label="标线可见度", value=LV_OPTS[-1], interactive=True)
                    dd_ls = gr.Dropdown(choices=LS_OPTS, label="标线样式", value=LS_OPTS[-1], interactive=True)

                gr.Markdown("**障碍物方位** (远端 1-2-3 / 中间 P左-7-P右 / 近端 4-5-6)")
                with gr.Row():
                    dd_obs1 = gr.Dropdown(choices=OBS_OPTS, label="①远左", value="UNKNOWN (未知)", interactive=True)
                    dd_obs2 = gr.Dropdown(choices=OBS_OPTS, label="②远中", value="UNKNOWN (未知)", interactive=True)
                    dd_obs3 = gr.Dropdown(choices=OBS_OPTS, label="③远右", value="UNKNOWN (未知)", interactive=True)
                with gr.Row():
                    dd_obs_pl = gr.Dropdown(choices=OBS_OPTS, label="P左侧", value="UNKNOWN (未知)", interactive=True)
                    dd_obs7 = gr.Dropdown(choices=OBS_OPTS, label="⑦车位内", value="UNKNOWN (未知)", interactive=True)
                    dd_obs_pr = gr.Dropdown(choices=OBS_OPTS, label="P右侧", value="UNKNOWN (未知)", interactive=True)
                with gr.Row():
                    dd_obs4 = gr.Dropdown(choices=OBS_OPTS, label="④近左", value="UNKNOWN (未知)", interactive=True)
                    dd_obs5 = gr.Dropdown(choices=OBS_OPTS, label="⑤近中", value="UNKNOWN (未知)", interactive=True)
                    dd_obs6 = gr.Dropdown(choices=OBS_OPTS, label="⑥近右", value="UNKNOWN (未知)", interactive=True)

                dd_ss = gr.CheckboxGroup(choices=SS_OPTS, label="特殊场景 (可多选)", value=[], interactive=True)

                with gr.Row():
                    btn_save = gr.Button("💾 保存纠正到训练集", variant="secondary", size="lg")
                    save_status = gr.Textbox(label="保存状态", interactive=False)

                with gr.Accordion("完整 JSON 输出", open=False):
                    json_output = gr.Code(label="JSON", language="json")

        # ── 按钮绑定 ──
        predict_outputs = [
            status_box,
            dd_slot, dd_maneuver,
            dd_lc, dd_lv, dd_ls,
            dd_obs1, dd_obs2, dd_obs3,
            dd_obs4, dd_obs5, dd_obs6,
            dd_obs7, dd_obs_pl, dd_obs_pr,
            dd_ss,
            json_output,
        ]

        btn_predict.click(fn=predict, inputs=[input_image], outputs=predict_outputs)

        save_inputs = [
            input_image,
            dd_slot, dd_maneuver,
            dd_lc, dd_lv, dd_ls,
            dd_obs1, dd_obs2, dd_obs3,
            dd_obs4, dd_obs5, dd_obs6,
            dd_obs7, dd_obs_pl, dd_obs_pr,
            dd_ss,
        ]

        btn_save.click(fn=save_correction, inputs=save_inputs, outputs=[save_status])

        gr.Markdown("""
        ---
        **工作流:** 上传图片 → 点击分析 → 检查/修正结果 → 保存纠正 → 积累足够数据后重新训练
        ```
        python -m fst.train --data ./dataset --output ./checkpoints
        python -m fst.export_onnx --checkpoint checkpoints/best_model.pth --output fst_classifier.onnx
        ```
        """)

    return demo


def main():
    global engine

    parser = argparse.ArgumentParser(description="FST Classifier Gradio App")
    parser.add_argument("--model", required=True, help="Path to .onnx or .pth model")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--yolo", default="n", choices=["n", "s", "m", "l", "x"], help="YOLO model size")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    engine = FSTInference(
        args.model,
        img_size=args.img_size,
        device=args.device,
        use_yolo=True,
        yolo_model_size=args.yolo,
        yolo_conf_threshold=args.yolo_conf,
    )

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share, inbrowser=True, theme=gr.themes.Soft(), server_name="0.0.0.0")


if __name__ == "__main__":
    main()
