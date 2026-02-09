"""
FST v1 数据模型 + 分类标签索引映射.

所有枚举值 ↔ 整数索引的映射集中在此文件，训练和推理共用。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── 枚举定义 ─────────────────────────────────────────────

class SlotType(str, Enum):
    PERPENDICULAR = "PERPENDICULAR"
    PARALLEL = "PARALLEL"
    ANGLED = "ANGLED"
    UNKNOWN = "UNKNOWN"


class Maneuver(str, Enum):
    PARK_IN = "PARK_IN"
    PARK_OUT = "PARK_OUT"
    HEAD_IN = "HEAD_IN"
    TAIL_OUT = "TAIL_OUT"
    UNKNOWN = "UNKNOWN"


class SearchDirection(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


class LineColor(str, Enum):
    WHITE = "WHITE"
    YELLOW = "YELLOW"
    BLUE = "BLUE"
    MIXED = "MIXED"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class LineVisibility(str, Enum):
    CLEAR = "CLEAR"
    FAINT = "FAINT"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"


class LineStyle(str, Enum):
    SOLID = "SOLID"
    DASHED = "DASHED"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class ObstacleType(str, Enum):
    EMPTY = "EMPTY"
    VEHICLE = "VEHICLE"
    CURB = "CURB"
    WALL = "WALL"
    PILLAR = "PILLAR"
    CONE = "CONE"
    WATER_BARRIER = "WATER_BARRIER"
    FENCE = "FENCE"
    LAMP = "LAMP"
    FIRE_BOX_SUSPENDED = "FIRE_BOX_SUSPENDED"
    BUSH = "BUSH"
    UNKNOWN = "UNKNOWN"


class P0Scene(str, Enum):
    DEAD_END = "DEAD_END"
    NARROW_LANE = "NARROW_LANE"


class P1Scene(str, Enum):
    SLOPE = "SLOPE"
    SPLIT_SLOPE = "SPLIT_SLOPE"
    SPACE_UNMARKED = "SPACE_UNMARKED"
    COLOR_BLOCK = "COLOR_BLOCK"
    BRICK_GRASS = "BRICK_GRASS"
    STEP_BRICK_GRASS = "STEP_BRICK_GRASS"
    WLC = "WLC"
    MICRO = "MICRO"
    BRICK_STONE = "BRICK_STONE"
    NARROW_SLOT = "NARROW_SLOT"
    EXTREME_NARROW_SLOT = "EXTREME_NARROW_SLOT"
    MECHANICAL = "MECHANICAL"


# ── 标签索引映射（训练 / 推理共用）───────────────────────

SLOT_TYPE_CLASSES: List[str] = [e.value for e in SlotType]
MANEUVER_CLASSES: List[str] = [e.value for e in Maneuver]
LINE_COLOR_CLASSES: List[str] = [e.value for e in LineColor]
LINE_VIS_CLASSES: List[str] = [e.value for e in LineVisibility]
LINE_STYLE_CLASSES: List[str] = [e.value for e in LineStyle]
OBSTACLE_CLASSES: List[str] = [e.value for e in ObstacleType]

# 特殊场景（多标签）：合并 P0 + P1 为一个 14 维向量
SPECIAL_SCENE_CLASSES: List[str] = (
    [e.value for e in P0Scene] + [e.value for e in P1Scene]
)

POSITION_KEYS: List[str] = ["1", "2", "3", "4", "5", "6", "7", "P_LEFT", "P_RIGHT"]

# 镜像映射：水平翻转时交换的位置
MIRROR_POS_MAP: Dict[str, str] = {
    "1": "3", "3": "1",
    "4": "6", "6": "4",
    "2": "2", "5": "5", "7": "7",
    "P_LEFT": "P_RIGHT", "P_RIGHT": "P_LEFT",
}


def _enum_to_idx(values: List[str]) -> Dict[str, int]:
    return {v: i for i, v in enumerate(values)}


SLOT_TYPE_TO_IDX = _enum_to_idx(SLOT_TYPE_CLASSES)
MANEUVER_TO_IDX = _enum_to_idx(MANEUVER_CLASSES)
LINE_COLOR_TO_IDX = _enum_to_idx(LINE_COLOR_CLASSES)
LINE_VIS_TO_IDX = _enum_to_idx(LINE_VIS_CLASSES)
LINE_STYLE_TO_IDX = _enum_to_idx(LINE_STYLE_CLASSES)
OBSTACLE_TO_IDX = _enum_to_idx(OBSTACLE_CLASSES)
SPECIAL_SCENE_TO_IDX = _enum_to_idx(SPECIAL_SCENE_CLASSES)

# ── Head 输出维度汇总 ────────────────────────────────────

NUM_SLOT_TYPE = len(SLOT_TYPE_CLASSES)        # 4
NUM_MANEUVER = len(MANEUVER_CLASSES)          # 5
NUM_SPECIAL_SCENE = len(SPECIAL_SCENE_CLASSES) # 14
NUM_OBSTACLE = len(OBSTACLE_CLASSES)          # 12
NUM_LINE_COLOR = len(LINE_COLOR_CLASSES)      # 6
NUM_LINE_VIS = len(LINE_VIS_CLASSES)          # 4
NUM_LINE_STYLE = len(LINE_STYLE_CLASSES)      # 3
NUM_POSITIONS = len(POSITION_KEYS)            # 9


# ── Pydantic 输出模型（推理结果）─────────────────────────

Position = Literal["1", "2", "3", "4", "5", "6", "7", "P_LEFT", "P_RIGHT"]


class Marking(BaseModel):
    line_color: LineColor = LineColor.UNKNOWN
    line_visibility: LineVisibility = LineVisibility.UNKNOWN
    line_style: LineStyle = LineStyle.UNKNOWN


class Occupancy(BaseModel):
    status: str = "UNKNOWN"
    occupied_by: List[str] = Field(default_factory=list)


class Slot(BaseModel):
    slot_type: SlotType = SlotType.UNKNOWN
    angle_deg: Optional[float] = None
    marking: Marking = Field(default_factory=Marking)
    occupancy: Occupancy = Field(default_factory=Occupancy)


class SpecialScene(BaseModel):
    P0: List[str] = Field(default_factory=list)
    P1: List[str] = Field(default_factory=list)


class Token(BaseModel):
    positions: List[str]
    type: str


class Obstacles(BaseModel):
    pos_map: Dict[str, List[str]]
    tokens: List[Token] = Field(default_factory=list)


class Confidence(BaseModel):
    overall: float = 0.5
    slot_type: float = 0.5
    marking: float = 0.5
    occupancy: float = 0.5
    obstacles: float = 0.5


class FSTOutput(BaseModel):
    schema_version: str = "fst.v1"
    image_id: Optional[str] = None
    raw_description: str = ""
    fst_level: int = 1
    search_direction: str = "UNKNOWN"
    slot: Slot = Field(default_factory=Slot)
    maneuver: str = "UNKNOWN"
    special_scene: SpecialScene = Field(default_factory=SpecialScene)
    obstacles: Obstacles
    confidence: Confidence = Field(default_factory=Confidence)


# ── FST Text DSL 生成 ────────────────────────────────────

# 中文映射表（用于生成 fst_text）
_SLOT_TYPE_ZH = {
    "PERPENDICULAR": "垂直", "PARALLEL": "水平",
    "ANGLED": "鱼骨", "UNKNOWN": "未知",
}
_MANEUVER_ZH = {
    "PARK_IN": "泊入", "PARK_OUT": "泊出",
    "HEAD_IN": "车头泊入", "TAIL_OUT": "车尾泊出", "UNKNOWN": "",
}
_OBSTACLE_ZH = {
    "EMPTY": "空", "VEHICLE": "车", "CURB": "路沿", "WALL": "墙",
    "PILLAR": "柱", "CONE": "锥桶", "WATER_BARRIER": "水马",
    "FENCE": "栅栏", "LAMP": "路灯", "FIRE_BOX_SUSPENDED": "悬空消防箱",
    "BUSH": "灌木丛", "UNKNOWN": "未知",
}
_P0_ZH = {"DEAD_END": "断头路", "NARROW_LANE": "窄通道"}
_P1_ZH = {
    "SLOPE": "坡道", "SPLIT_SLOPE": "分体坡道", "SPACE_UNMARKED": "空间",
    "COLOR_BLOCK": "色块", "BRICK_GRASS": "砖草", "STEP_BRICK_GRASS": "台阶砖草",
    "WLC": "WLC", "MICRO": "微型", "BRICK_STONE": "砖石",
    "NARROW_SLOT": "窄车位", "EXTREME_NARROW_SLOT": "极窄车位",
    "MECHANICAL": "机械车位",
}


def build_fst_text(output: FSTOutput) -> str:
    """
    从 FSTOutput 生成 FST 文本 DSL，如:
      '断头路 空间 3路灯5路沿7锥桶空车 垂直泊入'
    """
    parts: List[str] = []

    # P0 + P1 特殊场景
    for tag in output.special_scene.P0:
        parts.append(_P0_ZH.get(tag, tag))
    for tag in output.special_scene.P1:
        parts.append(_P1_ZH.get(tag, tag))

    # 障碍物 1-7（合并同类）
    obs_map = output.obstacles.pos_map
    # 按位置输出（紧凑格式）
    obs_parts: List[str] = []
    for pos in ["1", "2", "3", "4", "5", "6", "7"]:
        items = obs_map.get(pos, ["UNKNOWN"])
        item = items[0] if items else "UNKNOWN"
        if item not in ("EMPTY", "UNKNOWN"):
            obs_parts.append(f"{pos}{_OBSTACLE_ZH.get(item, item)}")

    # 5 号位置单独标注
    pos5_items = obs_map.get("5", ["UNKNOWN"])
    pos5 = pos5_items[0] if pos5_items else "UNKNOWN"
    if pos5 not in ("EMPTY", "UNKNOWN") and f"5{_OBSTACLE_ZH.get(pos5, pos5)}" not in obs_parts:
        obs_parts.append(f"5{_OBSTACLE_ZH.get(pos5, pos5)}")

    if obs_parts:
        parts.append("".join(obs_parts))

    # P_LEFT / P_RIGHT
    pl = obs_map.get("P_LEFT", ["UNKNOWN"])[0]
    pr = obs_map.get("P_RIGHT", ["UNKNOWN"])[0]
    parts.append(_OBSTACLE_ZH.get(pl, pl))
    parts.append(_OBSTACLE_ZH.get(pr, pr))

    # 车位类型 + 泊车动作
    parts.append(_SLOT_TYPE_ZH.get(output.slot.slot_type.value
                                    if isinstance(output.slot.slot_type, Enum)
                                    else output.slot.slot_type, ""))
    mv = (output.maneuver.value if isinstance(output.maneuver, Enum)
          else output.maneuver)
    parts.append(_MANEUVER_ZH.get(mv, ""))

    return "".join(p for p in parts if p)
