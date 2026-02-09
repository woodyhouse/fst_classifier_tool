"""
Gradio 推理 UI: 拖入图片 → 显示 FST 分类结果.

这是最终分发给用户使用的界面。
用法:
    python -m fst.app --model fst_classifier.onnx --port 7860
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from fst.inference import FSTInference
from fst.models import build_fst_text, POSITION_KEYS


# 全局推理引擎
engine: FSTInference | None = None

# 中文映射
_SLOT_ZH = {"PERPENDICULAR": "垂直", "PARALLEL": "水平", "ANGLED": "鱼骨", "UNKNOWN": "未知"}
_MV_ZH = {"PARK_IN": "泊入", "PARK_OUT": "泊出", "HEAD_IN": "车头泊入", "TAIL_OUT": "车尾泊出", "UNKNOWN": "未知"}
_OBS_ZH = {
    "EMPTY": "空", "VEHICLE": "车", "CURB": "路沿", "WALL": "墙", "PILLAR": "柱",
    "CONE": "锥桶", "WATER_BARRIER": "水马", "FENCE": "栅栏", "LAMP": "路灯",
    "FIRE_BOX_SUSPENDED": "悬空消防箱", "BUSH": "灌木丛", "UNKNOWN": "未知",
}
_POS_ZH = {
    "1": "①远左", "2": "②远中", "3": "③远右",
    "4": "④近左", "5": "⑤近中", "6": "⑥近右",
    "7": "⑦车位内", "P_LEFT": "P左侧", "P_RIGHT": "P右侧",
}


def format_result_html(result: dict) -> str:
    """将推理结果格式化为 HTML 表格."""
    slot = result.get("slot", {})
    mk = slot.get("marking", {})
    ss = result.get("special_scene", {})
    obs = result.get("obstacles", {}).get("pos_map", {})
    conf = result.get("confidence", {})

    slot_type = slot.get("slot_type", "UNKNOWN")
    maneuver = result.get("maneuver", "UNKNOWN")

    html = f"""
    <div style="font-family: sans-serif; max-width: 600px;">
      <h3 style="color: #1a73e8;">🅿️ FST 分类结果 (Level {result.get('fst_level', '?')})</h3>

      <table style="border-collapse: collapse; width: 100%; margin-bottom: 16px;">
        <tr style="background: #e8f0fe;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">车位类型</td>
          <td style="padding: 8px; border: 1px solid #ddd;">{slot_type} ({_SLOT_ZH.get(slot_type, '')})</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #888;">置信度 {conf.get('slot_type', 0):.1%}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">泊车动作</td>
          <td style="padding: 8px; border: 1px solid #ddd;">{maneuver} ({_MV_ZH.get(maneuver, '')})</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #888;">-</td>
        </tr>
        <tr style="background: #f8f9fa;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">标线颜色</td>
          <td style="padding: 8px; border: 1px solid #ddd;">{mk.get('line_color', '?')}</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #888;">置信度 {conf.get('marking', 0):.1%}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">标线可见度</td>
          <td style="padding: 8px; border: 1px solid #ddd;">{mk.get('line_visibility', '?')}</td>
          <td style="padding: 8px; border: 1px solid #ddd;"></td>
        </tr>
        <tr style="background: #f8f9fa;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">标线样式</td>
          <td style="padding: 8px; border: 1px solid #ddd;">{mk.get('line_style', '?')}</td>
          <td style="padding: 8px; border: 1px solid #ddd;"></td>
        </tr>
      </table>

      <h4>特殊场景</h4>
      <p>{', '.join(ss.get('P0', []) + ss.get('P1', [])) or '无'}</p>

      <h4>障碍物方位</h4>
      <table style="border-collapse: collapse; width: 100%; text-align: center;">
    """

    # 3×3 方位网格
    layout = [
        ["1", "2", "3"],
        ["P_LEFT", "7", "P_RIGHT"],
        ["4", "5", "6"],
    ]
    for grid_row in layout:
        html += "<tr>"
        for pos in grid_row:
            items = obs.get(pos, ["UNKNOWN"])
            val = items[0] if items else "UNKNOWN"
            bg = "#e8f5e9" if val == "EMPTY" else "#fff3e0" if val != "UNKNOWN" else "#f5f5f5"
            html += f'<td style="padding: 10px; border: 1px solid #ddd; background: {bg};">'
            html += f'<small style="color: #888;">{_POS_ZH.get(pos, pos)}</small><br>'
            html += f'<b>{_OBS_ZH.get(val, val)}</b>'
            html += "</td>"
        html += "</tr>"

    html += """
      </table>
      <p style="margin-top: 12px; color: #888; font-size: 12px;">
        总体置信度: {overall:.1%}
      </p>
    </div>
    """.format(overall=conf.get("overall", 0))

    return html


def predict(image):
    """Gradio 回调函数."""
    if engine is None:
        return "⚠️ 模型未加载", "{}"
    if image is None:
        return "请上传图片", "{}"

    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = Image.fromarray(image)

    result = engine.predict_with_text(img)

    html = format_result_html(result)
    fst_text = result.get("fst_text", "")
    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    return html, fst_text, json_str


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FST 车位分类器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🅿️ FST 车位分类器
        上传地面视角停车位照片，自动输出 FST 结构化描述。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="上传停车位照片", type="pil")
                btn = gr.Button("🔍 开始分析", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_html = gr.HTML(label="分类结果")
                fst_text_output = gr.Textbox(label="FST 文本 (DSL)", lines=2)
                json_output = gr.Code(label="完整 JSON 输出", language="json")

        btn.click(fn=predict, inputs=[input_image], outputs=[result_html, fst_text_output, json_output])
        input_image.change(fn=predict, inputs=[input_image], outputs=[result_html, fst_text_output, json_output])

        gr.Markdown("""
        ---
        **使用说明:**
        - 支持 JPG / PNG / BMP / WebP 格式
        - 推理完全在本地执行，不上传任何数据
        - FST 文本格式示例: `断头路空间3路灯5路沿7锥桶空车垂直泊入`
        """)

    return demo


def main():
    global engine

    parser = argparse.ArgumentParser(description="FST Classifier Gradio App")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    engine = FSTInference(args.model, img_size=args.img_size, device=args.device)

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
