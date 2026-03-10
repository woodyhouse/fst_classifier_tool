"""
鍩轰簬鍙屽眰鍥涜竟褰㈡ā鏉垮尮閰嶇殑杞︿綅绾挎娴嬪櫒
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path

from fst.paths import DEFAULT_TEMPLATE_PATH


@dataclass
class ParkingTemplate:
    """杞︿綅妯℃澘"""
    outer_quad: np.ndarray  # 澶栧眰鍥涜竟褰?(4, 2)
    inner_quad: np.ndarray  # 鍐呭眰鍥涜竟褰?(4, 2)
    width: int              # 杞︿綅瀹藉害
    length: int             # 杞︿綅闀垮害
    perspective: float      # 閫忚绯绘暟
    rotation: float         # 鏃嬭浆瑙掑害
    line_width: int         # 绾垮


@dataclass
class ParkingMatch:
    """Template match result."""
    outer_quad: np.ndarray
    inner_quad: np.ndarray
    score: float
    template_id: int


class TemplateGenerator:
    """鍙屽眰鍥涜竟褰㈡ā鏉跨敓鎴愬櫒"""

    def __init__(self):
        self.templates: List[ParkingTemplate] = []

    def generate_single_template(
        self,
        width: int,
        length: int,
        perspective: float,
        rotation: float,
        line_width: int
    ) -> ParkingTemplate:
        """
        鐢熸垚鍗曚釜鍙屽眰鍥涜竟褰㈡ā鏉?
        Args:
            width: 杞︿綅瀹藉害锛堝儚绱狅級
            length: 杞︿綅闀垮害锛堝儚绱狅級
            perspective: 閫忚绯绘暟 0.3-1.0锛堣繙绔?杩戠瀹藉害姣旓級
            rotation: 鏃嬭浆瑙掑害 0-180搴?            line_width: 杞︿綅绾垮搴︼紙鍍忕礌锛?        """
        near_width = width
        far_width = width * perspective

        # 澶栧眰鍥涜竟褰紙鍚溅浣嶇嚎锛?        # 椤哄簭锛氬乏涓?-> 鍙充笅 -> 鍙充笂 -> 宸︿笂
        outer_quad = np.array([
            [-near_width/2, 0],           # 宸︿笅锛堣繎绔級
            [near_width/2, 0],            # 鍙充笅锛堣繎绔級
            [far_width/2, -length],       # 鍙充笂锛堣繙绔級
            [-far_width/2, -length]       # 宸︿笂锛堣繙绔級
        ], dtype=np.float32)

        # 鍐呭眰鍥涜竟褰紙绾溅浣嶅尯鍩燂級
        inner_near_width = near_width - 2 * line_width
        inner_far_width = far_width - 2 * line_width

        inner_quad = np.array([
            [-inner_near_width/2, -line_width],
            [inner_near_width/2, -line_width],
            [inner_far_width/2, -length + line_width],
            [-inner_far_width/2, -length + line_width]
        ], dtype=np.float32)

        # 搴旂敤鏃嬭浆
        if rotation != 0:
            angle_rad = np.deg2rad(rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            outer_quad = outer_quad @ rotation_matrix.T
            inner_quad = inner_quad @ rotation_matrix.T

        return ParkingTemplate(
            outer_quad=outer_quad,
            inner_quad=inner_quad,
            width=width,
            length=length,
            perspective=perspective,
            rotation=rotation,
            line_width=line_width
        )

    def generate_template_library(
        self,
        width_range: List[int] = [60, 80, 100, 120, 150],
        length_range: List[int] = [150, 200, 250, 300],
        perspective_range: List[float] = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        rotation_range: List[float] = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
        line_width_range: List[int] = [8, 12, 16]
    ) -> List[ParkingTemplate]:
        """
        鐢熸垚澶ц妯℃ā鏉垮簱锛堥拡瀵瑰湴闈㈣瑙掍紭鍖栵級

        鍦伴潰瑙嗚鐗圭偣锛?        - 閫忚鍙樺舰鏋佸己锛堣繙绔彲鑳藉彧鏈夎繎绔殑10-20%瀹藉害锛?        - 瑙掑害鍙樺寲澶э紙鎷嶆憚瑙掑害涓嶅浐瀹氾級
        - 杞︿綅绾垮湪鍥惧儚涓殑灏哄鍙樺寲鑼冨洿骞?
        榛樿鐢熸垚: 5 脳 4 脳 10 脳 12 脳 3 = 7200 涓ā鏉?        """
        self.templates = []
        template_id = 0

        for width in width_range:
            for length in length_range:
                for perspective in perspective_range:
                    for rotation in rotation_range:
                        for line_width in line_width_range:
                            template = self.generate_single_template(
                                width, length, perspective, rotation, line_width
                            )
                            self.templates.append(template)
                            template_id += 1

        print(f"Generated {len(self.templates)} templates (ground-view optimized)")
        return self.templates

    def save_templates(self, filepath: str):
        """Save template library."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.templates, f)
        print(f"Template library saved to: {filepath}")

    def load_templates(self, filepath: str):
        """Load template library."""
        with open(filepath, 'rb') as f:
            self.templates = pickle.load(f)
        print(f"Loaded {len(self.templates)} templates")
        return self.templates


class TemplateMatcher:
    """Template matcher."""

    def __init__(self, templates: List[ParkingTemplate]):
        self.templates = templates
        self.coarse_templates = []
        self._build_coarse_templates()

    def _build_coarse_templates(self):
        """鏋勫缓绮楀尮閰嶆ā鏉块泦锛堟瘡闅擭涓彇涓€涓級"""
        step = max(1, len(self.templates) // 100); self.coarse_templates = [i for i in range(0, len(self.templates), step)]
        print(f"Coarse matching uses {len(self.coarse_templates)} representative templates")

    def compute_geometric_similarity(
        self,
        candidate_quad: np.ndarray,
        template_quad: np.ndarray
    ) -> float:
        """
        璁＄畻鍑犱綍鐩镐技搴?
        Args:
            candidate_quad: 鍊欓€夊洓杈瑰舰 (4, 2)
            template_quad: 妯℃澘鍥涜竟褰?(4, 2)

        Returns:
            鐩镐技搴﹀垎鏁?0-1
        """
        # 璁＄畻杈归暱
        def edge_lengths(quad):
            lengths = []
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i + 1) % 4]
                lengths.append(np.linalg.norm(p2 - p1))
            return np.array(lengths)

        cand_lengths = edge_lengths(candidate_quad)
        temp_lengths = edge_lengths(template_quad)

        cand_lengths_norm = cand_lengths / (np.sum(cand_lengths) + 1e-6)
        temp_lengths_norm = temp_lengths / (np.sum(temp_lengths) + 1e-6)

        length_diff = np.abs(cand_lengths_norm - temp_lengths_norm)
        length_score = 1.0 - np.mean(length_diff)

        # 璁＄畻瑙掑害
        def compute_angles(quad):
            angles = []
            for i in range(4):
                p1 = quad[(i - 1) % 4]
                p2 = quad[i]
                p3 = quad[(i + 1) % 4]

                v1 = p1 - p2
                v2 = p3 - p2

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
            return np.array(angles)

        cand_angles = compute_angles(candidate_quad)
        temp_angles = compute_angles(template_quad)

        angle_diff = np.abs(cand_angles - temp_angles)
        angle_score = 1.0 - np.mean(angle_diff) / np.pi

        # 缁煎悎鍒嗘暟
        geometric_score = 0.6 * length_score + 0.4 * angle_score
        return max(0, geometric_score)

    def compute_perspective_score(
        self,
        quad: np.ndarray
    ) -> float:
        """
        璁＄畻閫忚鍚堢悊鎬у垎鏁?        妫€鏌ユ槸鍚︾鍚?涓婄獎涓嬪"鐨勯€忚瑙勫緥
        """
        # 璁＄畻涓婅竟鍜屼笅杈圭殑闀垮害
        bottom_length = np.linalg.norm(quad[1] - quad[0])  # 杩戠
        top_length = np.linalg.norm(quad[2] - quad[3])     # 杩滅

        perspective_ratio = top_length / (bottom_length + 1e-6)

        # 鍚堢悊鑼冨洿: 0.3 - 1.0
        if 0.3 <= perspective_ratio <= 1.0:
            return 1.0
        elif perspective_ratio > 1.0:
            # 杩濆弽閫忚瑙勫緥锛堜笂瀹戒笅绐勶級
            return 0.0
        else:
            # 閫忚杩囧己
            return max(0, perspective_ratio / 0.3)

    def match_template(
        self,
        candidate_outer: np.ndarray,
        candidate_inner: np.ndarray,
        top_k: int = 5,
        use_coarse: bool = True
    ) -> List[ParkingMatch]:
        """
        鍖归厤妯℃澘锛堟敮鎸佸垎灞傚尮閰嶅姞閫燂級

        Args:
            candidate_outer: 鍊欓€夊灞傚洓杈瑰舰
            candidate_inner: 鍊欓€夊唴灞傚洓杈瑰舰
            top_k: 杩斿洖鍓峩涓渶浣冲尮閰?            use_coarse: 鏄惁浣跨敤绮楀尮閰嶅姞閫?
        Returns:
            鍖归厤缁撴灉鍒楄〃
        """
        if use_coarse and len(self.coarse_templates) > 0:
            coarse_matches = []
            for template_id in self.coarse_templates:
                template = self.templates[template_id]

                outer_sim = self.compute_geometric_similarity(
                    candidate_outer, template.outer_quad
                )
                inner_sim = self.compute_geometric_similarity(
                    candidate_inner, template.inner_quad
                )
                perspective_score = self.compute_perspective_score(candidate_outer)

                total_score = (
                    0.4 * outer_sim +
                    0.3 * inner_sim +
                    0.3 * perspective_score
                )

                coarse_matches.append((template_id, total_score))

            # 鎵惧埌鏈€浣崇矖鍖归厤
            coarse_matches.sort(key=lambda x: x[1], reverse=True)
            best_coarse_id = coarse_matches[0][0]

            # 绗簩灞傦細鍦ㄦ渶浣崇矖鍖归厤闄勮繎绮剧粏鎼滅储
            search_range = 50  # 鍦ㄦ渶浣虫ā鏉垮墠鍚?0涓ā鏉垮唴鎼滅储
            start_id = max(0, best_coarse_id - search_range)
            end_id = min(len(self.templates), best_coarse_id + search_range)
            search_ids = range(start_id, end_id)
        else:
            search_ids = range(len(self.templates))

        # 绮剧粏鍖归厤
        matches = []
        for template_id in search_ids:
            template = self.templates[template_id]

            outer_sim = self.compute_geometric_similarity(
                candidate_outer, template.outer_quad
            )

            inner_sim = self.compute_geometric_similarity(
                candidate_inner, template.inner_quad
            )

            perspective_score = self.compute_perspective_score(candidate_outer)

            # 缁煎悎鍒嗘暟
            total_score = (
                0.4 * outer_sim +
                0.3 * inner_sim +
                0.3 * perspective_score
            )

            matches.append(ParkingMatch(
                outer_quad=candidate_outer.copy(),
                inner_quad=candidate_inner.copy(),
                score=total_score,
                template_id=template_id
            ))

        # 鎺掑簭骞惰繑鍥瀟op_k
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]


class TemplateParkingDetector:
    """鍩轰簬妯℃澘鍖归厤鐨勮溅浣嶇嚎妫€娴嬪櫒"""

    def __init__(self, template_path: Optional[str] = None):
        self.generator = TemplateGenerator()

        resolved_template_path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH

        if resolved_template_path.exists():
            self.generator.load_templates(str(resolved_template_path))
        else:
            print("鐢熸垚妯℃澘搴?..")
            self.generator.generate_template_library()
            resolved_template_path.parent.mkdir(parents=True, exist_ok=True)
            self.generator.save_templates(str(resolved_template_path))

        self.matcher = TemplateMatcher(self.generator.templates)

    @staticmethod
    def _line_length(line: np.ndarray) -> float:
        return float(np.linalg.norm(np.array([line[2] - line[0], line[3] - line[1]], dtype=np.float32)))

    @staticmethod
    def _line_angle_abs(line: np.ndarray) -> float:
        """Return acute line angle in degrees [0, 90]."""
        ang = abs(np.rad2deg(np.arctan2(line[3] - line[1], line[2] - line[0]))) % 180.0
        if ang > 90.0:
            ang = 180.0 - ang
        return float(ang)

    @staticmethod
    def _line_angle_signed(line: np.ndarray) -> float:
        """Return signed line angle in degrees within [-90, 90]."""
        ang = float(np.rad2deg(np.arctan2(line[3] - line[1], line[2] - line[0])))
        while ang <= -90.0:
            ang += 180.0
        while ang > 90.0:
            ang -= 180.0
        return ang

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        """Smallest absolute angular difference in degrees, periodic at 180."""
        d = abs(float(a) - float(b)) % 180.0
        if d > 90.0:
            d = 180.0 - d
        return float(d)

    @staticmethod
    def _line_overlap_ratio(line1: np.ndarray, line2: np.ndarray) -> float:
        """Projected overlap ratio along line1 direction."""
        p1 = np.array([[line1[0], line1[1]], [line1[2], line1[3]]], dtype=np.float32)
        p2 = np.array([[line2[0], line2[1]], [line2[2], line2[3]]], dtype=np.float32)

        v = p1[1] - p1[0]
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return 0.0
        u = v / n

        t1 = p1 @ u
        t2 = p2 @ u
        a1, b1 = float(np.min(t1)), float(np.max(t1))
        a2, b2 = float(np.min(t2)), float(np.max(t2))

        inter = max(0.0, min(b1, b2) - max(a1, a2))
        denom = max(min(b1 - a1, b2 - a2), 1e-6)
        return float(inter / denom)

    def _filter_lines_for_template(
        self,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int],
        min_angle: float = 20.0,
        max_angle: float = 75.0,
        min_near_y_ratio: float = 0.52,
        min_mid_y_ratio: float = 0.30,
        min_length_ratio: float = 0.08,
    ) -> List[np.ndarray]:
        """Suppress clutter before line pairing."""
        h, _ = image_shape
        filtered = []
        for line in lines:
            line = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
            length = self._line_length(line)
            if length < max(40.0, h * min_length_ratio):
                continue

            angle = self._line_angle_abs(line)
            # Keep oblique side-line candidates; suppress near-horizontal/vertical clutter.
            if angle < min_angle or angle > max_angle:
                continue

            near_pt, _ = self._line_near_far(line)
            mid_y = float((line[1] + line[3]) * 0.5)
            # Parking-side lines should reach lower area.
            if near_pt[1] < h * min_near_y_ratio or mid_y < h * min_mid_y_ratio:
                continue

            filtered.append(line)
        return filtered

    def _filter_lines_multistage(
        self,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        Adaptive line prefilter:
        start strict for precision, then progressively relax when candidates are scarce.
        """
        h, w = image_shape
        if len(lines) == 0:
            return []

        # For tiny inputs, use slightly relaxed defaults.
        if min(h, w) < 350:
            stages = [
                (16.0, 82.0, 0.42, 0.24, 0.06),
                (10.0, 86.0, 0.34, 0.20, 0.05),
                (6.0, 89.0, 0.28, 0.16, 0.04),
            ]
        else:
            stages = [
                (20.0, 78.0, 0.52, 0.30, 0.08),
                (12.0, 86.0, 0.44, 0.25, 0.07),
                (6.0, 89.0, 0.34, 0.21, 0.06),
            ]

        best = []
        for min_a, max_a, near_r, mid_r, len_r in stages:
            candidate = self._filter_lines_for_template(
                lines,
                image_shape=image_shape,
                min_angle=min_a,
                max_angle=max_a,
                min_near_y_ratio=near_r,
                min_mid_y_ratio=mid_r,
                min_length_ratio=len_r,
            )
            if len(candidate) > len(best):
                best = candidate

        # Hard fallback: keep strong oblique long lines in lower half.
        fallback = []
        for line in lines:
            line = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
            if self._line_length(line) < max(28.0, h * 0.045):
                continue
            angle = self._line_angle_abs(line)
            if angle < 5.0 or angle > 89.5:
                continue
            near_pt, _ = self._line_near_far(line)
            if near_pt[1] < h * 0.25:
                continue
            fallback.append(line)

        if len(best) >= 3:
            return best
        if len(fallback) >= 2:
            return fallback
        return best

    def _trim_line_for_ground_perspective(
        self,
        line: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Trim overly long lines that shoot into ceiling/upper wall, which often causes
        oversized fake quads. Keeps near endpoint and clips far endpoint by y.
        """
        h, _ = image_shape
        ln = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
        near_pt, far_pt = self._line_near_far(ln)

        if near_pt[1] > h * 0.45 and far_pt[1] < h * 0.20:
            target_y = h * 0.24
            dy = float(far_pt[1] - near_pt[1])
            if abs(dy) > 1e-6:
                t = float((target_y - near_pt[1]) / dy)
                t = float(np.clip(t, 0.0, 1.0))
                clipped_far = near_pt + t * (far_pt - near_pt)
                return np.array([near_pt[0], near_pt[1], clipped_far[0], clipped_far[1]], dtype=np.float32)

        return ln

    @staticmethod
    def _line_near_far(line: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (near_point, far_point), where near has larger y."""
        p1 = np.array([line[0], line[1]], dtype=np.float32)
        p2 = np.array([line[2], line[3]], dtype=np.float32)
        return (p1, p2) if p1[1] >= p2[1] else (p2, p1)

    @staticmethod
    def _sample_line_points(p0: np.ndarray, p1: np.ndarray, n: int = 40) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        return p0[None, :] * (1.0 - t) + p1[None, :] * t

    def _dedupe_line_pairs(
        self,
        line_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        uniq: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[np.ndarray, np.ndarray]] = {}
        for l1, l2 in line_pairs:
            a = tuple(np.round(np.asarray(l1, dtype=np.float32).reshape(-1)[:4]).astype(np.int32).tolist())
            b = tuple(np.round(np.asarray(l2, dtype=np.float32).reshape(-1)[:4]).astype(np.int32).tolist())
            key = (a, b) if a <= b else (b, a)
            if key not in uniq:
                uniq[key] = (l1, l2)
        return list(uniq.values())

    def _estimate_floor_axis_angle(
        self,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> Optional[float]:
        """
        Estimate dominant floor cross-line angle (near-horizontal lines in lower region).
        Returns signed angle in [-90, 90], or None if unavailable.
        """
        h, _ = image_shape
        if len(lines) == 0:
            return None

        angles = []
        weights = []
        for line in lines:
            ln = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
            length = self._line_length(ln)
            if length < max(36.0, h * 0.06):
                continue
            mid_y = float((ln[1] + ln[3]) * 0.5)
            if mid_y < h * 0.40:
                continue

            ang = self._line_angle_signed(ln)
            if abs(ang) > 35.0:
                continue
            angles.append(ang)
            weights.append(length)

        if len(angles) < 2:
            return None

        # Weighted mean on doubled angles to handle 180-degree periodicity.
        a = np.deg2rad(np.asarray(angles, dtype=np.float32) * 2.0)
        w = np.asarray(weights, dtype=np.float32)
        c = float(np.sum(np.cos(a) * w))
        s = float(np.sum(np.sin(a) * w))
        if abs(c) < 1e-6 and abs(s) < 1e-6:
            return None
        mean = 0.5 * np.rad2deg(np.arctan2(s, c))
        while mean <= -90.0:
            mean += 180.0
        while mean > 90.0:
            mean -= 180.0
        return float(mean)

    @staticmethod
    def _line_intersection_infinite(
        line1: np.ndarray,
        line2: np.ndarray
    ) -> Optional[np.ndarray]:
        """Intersection of two infinite lines in x1,y1,x2,y2 format."""
        x1, y1, x2, y2 = [float(v) for v in np.asarray(line1, dtype=np.float32).reshape(-1)[:4]]
        x3, y3, x4, y4 = [float(v) for v in np.asarray(line2, dtype=np.float32).reshape(-1)[:4]]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / den
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / den
        if not np.isfinite(px) or not np.isfinite(py):
            return None
        return np.array([px, py], dtype=np.float32)

    def _estimate_vanishing_point(
        self,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Estimate perspective vanishing point from merged lines.
        The point may be outside image bounds; we mainly expect it in upper region.
        Returns (point, confidence in [0, 1]).
        """
        h, w = image_shape
        if len(lines) < 2:
            return None, 0.0

        valid: List[Tuple[np.ndarray, float, float]] = []
        min_len = max(22.0, h * 0.05)
        for line in lines:
            ln = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
            length = self._line_length(ln)
            if length < min_len:
                continue
            ang = self._line_angle_signed(ln)
            abs_ang = abs(ang)
            if abs_ang < 7.0 or abs_ang > 89.0:
                continue
            near_pt, _ = self._line_near_far(ln)
            if near_pt[1] < h * 0.24:
                continue
            valid.append((ln, ang, length))

        if len(valid) < 2:
            return None, 0.0

        pts: List[np.ndarray] = []
        weights: List[float] = []
        for i in range(len(valid)):
            l1, a1, len1 = valid[i]
            for j in range(i + 1, len(valid)):
                l2, a2, len2 = valid[j]
                if np.sign(a1) == np.sign(a2) and self._angle_diff_deg(a1, a2) < 28.0:
                    continue

                inter = self._line_intersection_infinite(l1, l2)
                if inter is None:
                    continue
                x, y = float(inter[0]), float(inter[1])

                # VP can be outside image, but keep physically plausible range.
                if y > h * 0.72 or y < -3.0 * h:
                    continue
                if x < -3.5 * w or x > 4.5 * w:
                    continue

                ang_sep = max(8.0, self._angle_diff_deg(a1, a2))
                weight = 0.5 * (len1 + len2) * min(2.0, ang_sep / 28.0)
                if y < h * 0.5:
                    weight *= 1.10
                pts.append(inter)
                weights.append(float(weight))

        if len(pts) == 0:
            return None, 0.0

        pts_arr = np.asarray(pts, dtype=np.float32)
        w_arr = np.asarray(weights, dtype=np.float32)
        center = np.average(pts_arr, axis=0, weights=w_arr)

        if len(pts_arr) >= 6:
            d = np.linalg.norm(pts_arr - center[None, :], axis=1)
            cut = float(np.percentile(d, 72))
            keep = d <= max(cut, 1e-6)
            if np.sum(keep) >= 3:
                pts_arr = pts_arr[keep]
                w_arr = w_arr[keep]
                center = np.average(pts_arr, axis=0, weights=w_arr)

        diag = max(float(np.hypot(w, h)), 1e-6)
        spread = float(np.median(np.linalg.norm(pts_arr - center[None, :], axis=1)))
        spread_score = np.clip(1.0 - spread / (0.55 * diag), 0.0, 1.0)
        count_score = np.clip(len(pts_arr) / 8.0, 0.0, 1.0)
        conf = float(0.55 * spread_score + 0.45 * count_score)
        if float(center[1]) > h * 0.58:
            conf *= 0.55
        if float(center[1]) > h * 0.70:
            return None, 0.0
        return center.astype(np.float32), conf

    @staticmethod
    def _point_line_distance(
        point: np.ndarray,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Distance from point to infinite line through a-b."""
        p = np.asarray(point, dtype=np.float32)
        p0 = np.asarray(a, dtype=np.float32)
        p1 = np.asarray(b, dtype=np.float32)
        v = p1 - p0
        denom = float(np.linalg.norm(v))
        if denom < 1e-6:
            return 1e6
        return float(abs(v[0] * (p0[1] - p[1]) - v[1] * (p0[0] - p[0])) / denom)

    def _quad_vanishing_convergence_score(
        self,
        quad: np.ndarray,
        vanishing_point: Optional[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> float:
        """Score whether left/right edges converge to a shared vanishing point."""
        if vanishing_point is None:
            return 0.55

        h, w = image_shape
        pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        vp = np.asarray(vanishing_point, dtype=np.float32).reshape(2)

        left_n, left_f = pts[0], pts[3]
        right_n, right_f = pts[1], pts[2]

        d_left = self._point_line_distance(vp, left_n, left_f)
        d_right = self._point_line_distance(vp, right_n, right_f)
        diag = max(float(np.hypot(w, h)), 1e-6)
        line_fit = np.exp(-0.5 * (d_left + d_right) / (0.12 * diag + 1e-6))

        def edge_dir_score(pn: np.ndarray, pf: np.ndarray) -> float:
            e = pf - pn
            v = vp - pn
            en = float(np.linalg.norm(e))
            vn = float(np.linalg.norm(v))
            if en < 1e-6 or vn < 1e-6:
                return 0.0
            cos_sim = float(np.dot(e, v) / (en * vn))
            return float(np.clip((cos_sim + 1.0) * 0.5, 0.0, 1.0))

        dir_score = 0.5 * (edge_dir_score(left_n, left_f) + edge_dir_score(right_n, right_f))

        left_line = np.array([left_n[0], left_n[1], left_f[0], left_f[1]], dtype=np.float32)
        right_line = np.array([right_n[0], right_n[1], right_f[0], right_f[1]], dtype=np.float32)
        inter = self._line_intersection_infinite(left_line, right_line)
        if inter is None:
            inter_score = 0.0
        else:
            di = float(np.linalg.norm(inter - vp))
            inter_score = float(np.exp(-di / (0.18 * diag + 1e-6)))

        near_y = 0.5 * float(pts[0, 1] + pts[1, 1])
        vp_y = float(vp[1])
        if vp_y <= near_y + h * 0.10:
            top_gate = 1.0
        else:
            top_gate = float(np.clip(1.0 - (vp_y - near_y) / max(h * 0.55, 1.0), 0.0, 1.0))

        score = 0.45 * line_fit + 0.35 * dir_score + 0.20 * inter_score
        return float(np.clip(score * top_gate, 0.0, 1.0))

    def _quad_top_bottom_parallel_score(self, quad: np.ndarray) -> float:
        """Score top/bottom edge parallelism in [0,1]."""
        pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        bottom_ang = self._line_angle_signed(np.array([pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]], dtype=np.float32))
        top_ang = self._line_angle_signed(np.array([pts[3, 0], pts[3, 1], pts[2, 0], pts[2, 1]], dtype=np.float32))
        diff = self._angle_diff_deg(bottom_ang, top_ang)
        return float(max(0.0, 1.0 - diff / 26.0))

    @staticmethod
    def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _segment_intersection_point(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Return (intersection_point, is_collinear)."""
        p1 = np.asarray(p1, dtype=np.float32)
        p2 = np.asarray(p2, dtype=np.float32)
        q1 = np.asarray(q1, dtype=np.float32)
        q2 = np.asarray(q2, dtype=np.float32)
        r = p2 - p1
        s = q2 - q1
        rxs = self._cross2d(r, s)
        qmp = q1 - p1
        qmpxr = self._cross2d(qmp, r)

        if abs(rxs) < eps:
            return None, abs(qmpxr) < eps

        t = self._cross2d(qmp, s) / rxs
        u = self._cross2d(qmp, r) / rxs
        if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
            return p1 + t * r, False
        return None, False

    def _count_quad_line_intersections(
        self,
        quad: np.ndarray,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> int:
        """
        Count non-collinear crossings between quad boundary and merged lines.
        Lower is preferred for final slot selection.
        """
        h, w = image_shape
        pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        min_len = max(12.0, min(h, w) * 0.02)

        total = 0
        for line in lines:
            ln = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
            if self._line_length(ln) < min_len:
                continue
            p0 = np.array([ln[0], ln[1]], dtype=np.float32)
            p1 = np.array([ln[2], ln[3]], dtype=np.float32)
            line_ang = self._line_angle_signed(ln)

            intersections: List[np.ndarray] = []
            for e0, e1 in edges:
                edge_ln = np.array([e0[0], e0[1], e1[0], e1[1]], dtype=np.float32)
                if self._angle_diff_deg(line_ang, self._line_angle_signed(edge_ln)) < 8.0:
                    continue
                ip, collinear = self._segment_intersection_point(p0, p1, e0, e1)
                if collinear or ip is None:
                    continue
                if any(np.linalg.norm(ip - v) <= 2.5 for v in pts):
                    continue
                duplicated = False
                for prev in intersections:
                    if float(np.linalg.norm(ip - prev)) <= 2.5:
                        duplicated = True
                        break
                if not duplicated:
                    intersections.append(ip)

            total += len(intersections)
        return int(total)

    def _compute_edge_support(self, edges: np.ndarray, quad: np.ndarray) -> float:
        """Estimate quad edge support from image edges, returns [0, 1]."""
        h, w = edges.shape[:2]
        hit = 0
        total = 0
        for i in range(4):
            p0 = quad[i]
            p1 = quad[(i + 1) % 4]
            pts = self._sample_line_points(p0, p1, n=40)
            xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
            vals = edges[ys, xs] > 0
            hit += int(np.sum(vals))
            total += len(vals)
        return hit / max(total, 1)

    def _compute_segment_support(
        self,
        mask: Optional[np.ndarray],
        p0: np.ndarray,
        p1: np.ndarray,
        n: int = 40
    ) -> float:
        """Sample binary-mask support on a segment, returns [0, 1]."""
        if mask is None:
            return 0.0
        h, w = mask.shape[:2]
        pts = self._sample_line_points(np.asarray(p0, dtype=np.float32), np.asarray(p1, dtype=np.float32), n=n)
        xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
        return float(np.mean(mask[ys, xs] > 0))

    def _estimate_floor_split(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate a likely wall-floor split row ratio and a confidence score.
        Returns (ratio, confidence), where confidence is peak / p95.
        """
        h, _ = image.shape[:2]
        if h < 80:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        row_energy = np.mean(np.abs(gy), axis=1).astype(np.float32)
        row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 31), 0).reshape(-1)

        lo = int(h * 0.25)
        hi = int(h * 0.85)
        if hi - lo < 10:
            return None, None

        segment = row_energy[lo:hi]
        peak_idx = int(np.argmax(segment)) + lo
        peak = float(row_energy[peak_idx])
        p95 = float(np.percentile(segment, 95))
        conf = peak / max(p95, 1e-6)
        return float(peak_idx / max(float(h), 1.0)), conf

    def _candidate_prior_score(
        self,
        quad: np.ndarray,
        image_shape: Tuple[int, int],
        edges: np.ndarray,
        line_map: Optional[np.ndarray] = None,
        floor_axis_angle: Optional[float] = None,
        paint_mask: Optional[np.ndarray] = None,
        floor_split_ratio: Optional[float] = None,
        floor_split_conf: Optional[float] = None,
    ) -> float:
        """
        鍊欓€夊嚑浣曞厛楠?+ 杈圭紭鏀寔璇勫垎锛岃寖鍥?-1銆?        杩囦簬绂昏氨鐨勫€欓€夌洿鎺ヨ繑鍥?銆?        """
        h, w = image_shape
        pts = quad.astype(np.float32)

        margin = 8.0
        if np.any(pts[:, 0] < -margin) or np.any(pts[:, 0] > w - 1 + margin):
            return 0.0
        if np.any(pts[:, 1] < -margin) or np.any(pts[:, 1] > h - 1 + margin):
            return 0.0

        # 闈㈢Н绾︽潫
        area = abs(cv2.contourArea(pts.astype(np.float32)))
        area_ratio = area / float(max(h * w, 1))
        min_area_ratio = 0.010 if h < 400 else 0.050
        if area_ratio < min_area_ratio or area_ratio > 0.65:
            return 0.0

        bottom_len = float(np.linalg.norm(pts[1] - pts[0]))
        top_len = float(np.linalg.norm(pts[2] - pts[3]))
        if bottom_len < 18 or top_len < 10:
            return 0.0
        if top_len > bottom_len * 1.25:
            return 0.0

        left_len = float(np.linalg.norm(pts[3] - pts[0]))
        right_len = float(np.linalg.norm(pts[2] - pts[1]))
        depth_len = 0.5 * (left_len + right_len)
        width_len = max(bottom_len, top_len, 1e-6)
        depth_ratio = depth_len / width_len
        bottom_y_ratio_hint = float((pts[0, 1] + pts[1, 1]) * 0.5) / max(float(h), 1.0)
        if h < 300:
            min_depth_ratio = 0.14
        else:
            # Allow shallower near-bottom slots in frontal views.
            min_depth_ratio = 0.16 if bottom_y_ratio_hint > 0.62 else 0.24
        if depth_ratio < min_depth_ratio:
            return 0.0

        perspective = top_len / max(bottom_len, 1e-6)
        if perspective < 0.12 or perspective > 1.20:
            return 0.0

        bottom_y = float((pts[0, 1] + pts[1, 1]) / 2.0)
        top_y = float((pts[2, 1] + pts[3, 1]) / 2.0)
        top_y_ratio = top_y / max(float(h), 1.0)
        # In upper image zones, near-rectangle candidates are often wall bands.
        if top_y_ratio < 0.33 and perspective > 0.82:
            return 0.0
        if bottom_y <= top_y:
            return 0.0
        if top_y < h * 0.12:
            return 0.0
        if bottom_y < h * 0.30:
            return 0.0
        min_depth_span = h * (0.06 if h < 300 else 0.10)
        if (bottom_y - top_y) < min_depth_span:
            return 0.0

        if (
            floor_split_ratio is not None
            and floor_split_conf is not None
            and floor_split_conf > 1.55
            and floor_split_ratio > 0.55
            and top_y_ratio < (floor_split_ratio - 0.12)
        ):
            return 0.0

        cx = float(np.mean(pts[:, 0]))
        score_center = max(0.0, 1.0 - abs(cx - w / 2.0) / (w / 2.0 + 1e-6))
        score_bottom = np.clip((bottom_y - h * 0.35) / (h * 0.65 + 1e-6), 0.0, 1.0)
        score_depth = np.clip((bottom_y - top_y - min_depth_span) / (h * 0.50 + 1e-6), 0.0, 1.0)
        score_perspective = max(0.0, 1.0 - abs(perspective - 0.55) / 0.65)
        score_area = max(0.0, 1.0 - abs(area_ratio - 0.12) / 0.16)
        score_edge = self._compute_edge_support(edges, pts)
        score_line = self._compute_edge_support(line_map, pts) if line_map is not None else 0.0
        score_bottom_paint = self._compute_segment_support(paint_mask, pts[0], pts[1], n=52)
        score_top_paint = self._compute_segment_support(paint_mask, pts[3], pts[2], n=52)
        score_left_paint = self._compute_segment_support(paint_mask, pts[0], pts[3], n=36)
        score_right_paint = self._compute_segment_support(paint_mask, pts[1], pts[2], n=36)
        score_paint = (
            0.44 * score_bottom_paint +
            0.30 * score_top_paint +
            0.13 * score_left_paint +
            0.13 * score_right_paint
        )

        if score_edge < 0.08 and score_line < 0.08 and score_paint < 0.08:
            return 0.0
        if score_paint < 0.05 and score_line < 0.10:
            return 0.0
        if score_top_paint < 0.04 and score_bottom_paint < 0.08 and score_line < 0.12:
            return 0.0

        # Encourage top/bottom edges to be parallel and aligned with floor-axis direction.
        bottom_ang = self._line_angle_signed(np.array([pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1]], dtype=np.float32))
        top_ang = self._line_angle_signed(np.array([pts[3, 0], pts[3, 1], pts[2, 0], pts[2, 1]], dtype=np.float32))
        bt_diff = self._angle_diff_deg(bottom_ang, top_ang)
        if bt_diff > 50.0:
            return 0.0
        score_bt_parallel = max(0.0, 1.0 - bt_diff / 34.0)

        if floor_axis_angle is None:
            score_floor_align = 0.60
        else:
            diff_b = self._angle_diff_deg(bottom_ang, floor_axis_angle)
            diff_t = self._angle_diff_deg(top_ang, floor_axis_angle)
            mean_diff = 0.5 * (diff_b + diff_t)
            score_floor_align = max(0.0, 1.0 - mean_diff / 40.0)

        # Down-weight tiny edge-side candidates near image borders.
        width_ratio = bottom_len / max(float(w), 1.0)
        border_closeness = min(cx / max(float(w), 1.0), 1.0 - cx / max(float(w), 1.0))
        border_penalty = 1.0
        if width_ratio < 0.16 and border_closeness < 0.14:
            border_penalty = 0.55
        elif width_ratio < 0.11:
            border_penalty = 0.70

        return float(border_penalty * (
            0.20 * score_paint +
            0.18 * score_edge +
            0.14 * score_line +
            0.14 * score_perspective +
            0.12 * score_bottom +
            0.10 * score_depth +
            0.06 * score_area +
            0.03 * score_center +
            0.02 * score_bt_parallel +
            0.01 * score_floor_align
        ))

    def _wedge_false_positive_penalty(
        self,
        quad: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Penalize wedge-like wall/column false positives that often appear as
        side edges with the same slope direction and large angle mismatch.
        """
        h, w = image_shape
        pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)

        left_ang = self._line_angle_signed(
            np.array([pts[0, 0], pts[0, 1], pts[3, 0], pts[3, 1]], dtype=np.float32)
        )
        right_ang = self._line_angle_signed(
            np.array([pts[1, 0], pts[1, 1], pts[2, 0], pts[2, 1]], dtype=np.float32)
        )

        same_sign = (
            np.sign(left_ang) == np.sign(right_ang)
            and abs(left_ang) > 1e-3
            and abs(right_ang) > 1e-3
        )
        if not same_sign:
            return 1.0

        side_gap = abs(left_ang - right_ang)
        top_y = float((pts[2, 1] + pts[3, 1]) * 0.5) / max(float(h), 1.0)
        bottom_y = float((pts[0, 1] + pts[1, 1]) * 0.5) / max(float(h), 1.0)
        area_ratio = abs(float(cv2.contourArea(pts))) / max(float(h * w), 1.0)

        if side_gap > 58.0 and bottom_y < 0.78 and top_y < 0.50:
            return 0.35 if area_ratio < 0.060 else 0.45
        if side_gap > 70.0 and bottom_y < 0.72:
            return 0.72
        if (
            abs(left_ang) > 78.0
            and abs(right_ang) > 78.0
            and area_ratio < 0.080
            and top_y < 0.50
        ):
            return 0.62
        return 1.0

    def find_parallel_line_pairs(
        self,
        lines: List[np.ndarray],
        angle_threshold: float = 10.0,
        distance_range: Tuple[float, float] = (50, 200)
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        瀵绘壘骞宠绾垮

        Args:
            lines: 绾挎鍒楄〃锛屾瘡涓嚎娈典负 [x1, y1, x2, y2]
            angle_threshold: 瑙掑害宸紓闃堝€硷紙搴︼級
            distance_range: 骞宠绾块棿璺濊寖鍥达紙鍍忕礌锛?
        Returns:
            骞宠绾垮鍒楄〃
        """
        parallel_pairs = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i]
                line2 = lines[j]

                # 璁＄畻绾挎瑙掑害
                angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
                angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])

                # 瑙掑害宸紓
                angle_diff = np.abs(np.rad2deg(angle1 - angle2))
                angle_diff = min(angle_diff, 180 - angle_diff)

                if angle_diff > angle_threshold:
                    continue

                # 闀垮害杩囨护锛氳繃鐭嚎娈典笉鍙備笌妯℃澘鍖归厤
                len1 = self._line_length(line1)
                len2 = self._line_length(line2)
                if len1 < 40 or len2 < 40:
                    continue

                # 璁＄畻骞冲潎璺濈
                mid1 = np.array([(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2])
                mid2 = np.array([(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2])
                distance = np.linalg.norm(mid1 - mid2)

                if not (distance_range[0] <= distance <= distance_range[1]):
                    continue

                length_ratio = max(len1, len2) / max(min(len1, len2), 1e-6)
                if length_ratio > 2.4:
                    continue

                overlap_ratio = self._line_overlap_ratio(line1, line2)
                # Perspective side-lines can have low projected overlap when far apart.
                # Keep overlap gating mainly for close-range pairs.
                if distance < distance_range[0] * 1.6 and overlap_ratio < 0.20:
                    continue

                parallel_pairs.append((line1, line2))

        return parallel_pairs

    def find_converging_line_pairs(
        self,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int],
        near_span_range: Tuple[float, float]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Find perspective-converging side-line pairs (more suitable than strict parallel
        for ground-view parking slots).
        """
        h, w = image_shape
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        min_span, max_span = near_span_range

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i]
                line2 = lines[j]

                len1 = self._line_length(line1)
                len2 = self._line_length(line2)
                if len1 < 40 or len2 < 40:
                    continue

                near1, far1 = self._line_near_far(line1)
                near2, far2 = self._line_near_far(line2)

                near_span = abs(float(near1[0] - near2[0]))
                far_span = abs(float(far1[0] - far2[0]))
                if near_span < min_span or near_span > max_span:
                    continue

                # Converging trapezoid: top span should be meaningfully narrower than bottom span.
                if far_span >= near_span * 0.93:
                    continue

                # Top points should be meaningfully above bottom points.
                if max(float(far1[1]), float(far2[1])) >= min(float(near1[1]), float(near2[1])) - h * 0.06:
                    continue

                length_ratio = max(len1, len2) / max(min(len1, len2), 1e-6)
                if length_ratio > 2.8:
                    continue

                pairs.append((line1, line2))

        return pairs

    def infer_short_edges(
        self,
        line1: np.ndarray,
        line2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        浠庝袱鏉￠暱杈规帹鏂煭杈?
        Args:
            line1: 绗竴鏉￠暱杈?[x1, y1, x2, y2]
            line2: 绗簩鏉￠暱杈?[x1, y1, x2, y2]

        Returns:
            (杩戠鐭竟, 杩滅鐭竟)
        """
        line1_pts = [(line1[0], line1[1]), (line1[2], line1[3])]
        line2_pts = [(line2[0], line2[1]), (line2[2], line2[3])]

        line1_pts.sort(key=lambda p: p[1], reverse=True)
        line2_pts.sort(key=lambda p: p[1], reverse=True)

        # 杩戠鐭竟
        near_edge = np.array([
            line1_pts[0][0], line1_pts[0][1],
            line2_pts[0][0], line2_pts[0][1]
        ])

        # 杩滅鐭竟
        far_edge = np.array([
            line1_pts[1][0], line1_pts[1][1],
            line2_pts[1][0], line2_pts[1][1]
        ])

        return near_edge, far_edge

    def build_candidate_quads(
        self,
        parallel_pairs: List[Tuple[np.ndarray, np.ndarray]],
        line_width: int = 12
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        浠庡钩琛岀嚎瀵规瀯寤哄€欓€夊弻灞傚洓杈瑰舰

        Args:
            parallel_pairs: 骞宠绾垮鍒楄〃
            line_width: 杞︿綅绾垮搴?
        Returns:
            (澶栧眰鍥涜竟褰? 鍐呭眰鍥涜竟褰? 鍒楄〃
        """
        candidates = []

        for line1, line2 in parallel_pairs:
            near1, far1 = self._line_near_far(line1)
            near2, far2 = self._line_near_far(line2)

            # 鏍规嵁杩戠x纭畾宸﹀彸杈癸紝缁熶竴鐐瑰簭锛氬乏涓?>鍙充笅->鍙充笂->宸︿笂
            if near1[0] <= near2[0]:
                near_left, far_left = near1, far1
                near_right, far_right = near2, far2
            else:
                near_left, far_left = near2, far2
                near_right, far_right = near1, far1

            outer_quad = np.array([
                near_left,   # 宸︿笅
                near_right,  # 鍙充笅
                far_right,   # 鍙充笂
                far_left,    # 宸︿笂
            ], dtype=np.float32)

            # 鎺掗櫎鑷氦/閫€鍖栧洓杈瑰舰
            if abs(cv2.contourArea(outer_quad)) < 50:
                continue
            if not cv2.isContourConvex(outer_quad.astype(np.int32).reshape(-1, 1, 2)):
                continue

            def shrink_quad(quad, width):
                center = np.mean(quad, axis=0)
                shrunk = []
                for pt in quad:
                    direction = pt - center
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        new_pt = pt - direction * width
                        shrunk.append(new_pt)
                    else:
                        shrunk.append(pt)
                return np.array(shrunk, dtype=np.float32)

            inner_quad = shrink_quad(outer_quad, line_width)

            candidates.append((outer_quad, inner_quad))

        return candidates

    def build_horizontal_band_candidates(
        self,
        lines: List[np.ndarray],
        image_shape: Tuple[int, int],
        line_width: int = 12
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Build extra candidates from near-horizontal floor lines.
        This helps frontal views where slot side lines are weak.
        """
        h, w = image_shape

        horizontal_lines: List[np.ndarray] = []
        for line in lines:
            ln = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
            if self._line_length(ln) < max(36.0, w * 0.10):
                continue
            if self._line_angle_abs(ln) > 18.0:
                continue
            y_mean = 0.5 * (float(ln[1]) + float(ln[3]))
            if y_mean < h * 0.28:
                continue
            horizontal_lines.append(ln)

        horizontal_lines = sorted(horizontal_lines, key=self._line_length, reverse=True)[:22]
        if len(horizontal_lines) < 2:
            return []

        def sort_endpoints_x(ln: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            p0 = np.array([ln[0], ln[1]], dtype=np.float32)
            p1 = np.array([ln[2], ln[3]], dtype=np.float32)
            if p0[0] <= p1[0]:
                return p0, p1
            return p1, p0

        def overlap_ratio_x(a_l: float, a_r: float, b_l: float, b_r: float) -> float:
            inter = max(0.0, min(a_r, b_r) - max(a_l, b_l))
            min_len = max(min(a_r - a_l, b_r - b_l), 1e-6)
            return float(inter / min_len)

        candidates: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(horizontal_lines)):
            for j in range(i + 1, len(horizontal_lines)):
                l1 = horizontal_lines[i]
                l2 = horizontal_lines[j]

                l1_l, l1_r = sort_endpoints_x(l1)
                l2_l, l2_r = sort_endpoints_x(l2)

                y1 = 0.5 * (float(l1_l[1]) + float(l1_r[1]))
                y2 = 0.5 * (float(l2_l[1]) + float(l2_r[1]))
                if y1 >= y2:
                    near_l, near_r, far_l, far_r = l1_l, l1_r, l2_l, l2_r
                else:
                    near_l, near_r, far_l, far_r = l2_l, l2_r, l1_l, l1_r

                near_y = 0.5 * (float(near_l[1]) + float(near_r[1]))
                far_y = 0.5 * (float(far_l[1]) + float(far_r[1]))
                if near_y <= far_y:
                    continue
                if near_y < h * 0.42 or far_y < h * 0.24:
                    continue

                depth = near_y - far_y
                if depth < h * 0.06 or depth > h * 0.62:
                    continue

                near_w = float(np.linalg.norm(near_r - near_l))
                far_w = float(np.linalg.norm(far_r - far_l))
                if near_w < w * 0.12 or far_w < w * 0.08:
                    continue

                persp = far_w / max(near_w, 1e-6)
                if persp < 0.20 or persp > 1.20:
                    continue

                ox = overlap_ratio_x(float(near_l[0]), float(near_r[0]), float(far_l[0]), float(far_r[0]))
                if ox < 0.35:
                    continue

                outer_quad = np.array(
                    [near_l, near_r, far_r, far_l],
                    dtype=np.float32
                )
                if abs(cv2.contourArea(outer_quad)) < 50:
                    continue
                if not cv2.isContourConvex(outer_quad.astype(np.int32).reshape(-1, 1, 2)):
                    continue

                center = np.mean(outer_quad, axis=0)
                inner_pts = []
                for pt in outer_quad:
                    v = pt - center
                    n = float(np.linalg.norm(v))
                    if n > 1e-6:
                        inner_pts.append(pt - v / n * float(line_width))
                    else:
                        inner_pts.append(pt.copy())
                inner_quad = np.asarray(inner_pts, dtype=np.float32)
                candidates.append((outer_quad, inner_quad))

        return candidates

    def non_maximum_suppression(
        self,
        matches: List[ParkingMatch],
        iou_threshold: float = 0.5
    ) -> List[ParkingMatch]:
        """
        闈炴瀬澶у€兼姂鍒?
        Args:
            matches: 鍖归厤缁撴灉鍒楄〃
            iou_threshold: IoU闃堝€?
        Returns:
            杩囨护鍚庣殑鍖归厤缁撴灉
        """
        if len(matches) == 0:
            return []

        matches = sorted(matches, key=lambda x: x.score, reverse=True)

        def compute_iou(quad1, quad2):
            """璁＄畻涓や釜鍥涜竟褰㈢殑IoU锛堢畝鍖栫増鏈級"""
            # 浣跨敤澶栨帴鐭╁舰杩戜技
            def quad_to_rect(quad):
                x_min = np.min(quad[:, 0])
                y_min = np.min(quad[:, 1])
                x_max = np.max(quad[:, 0])
                y_max = np.max(quad[:, 1])
                return x_min, y_min, x_max, y_max

            x1_min, y1_min, x1_max, y1_max = quad_to_rect(quad1)
            x2_min, y2_min, x2_max, y2_max = quad_to_rect(quad2)

            # 璁＄畻浜ら泦
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0

            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

            # 璁＄畻骞堕泦
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area

            return inter_area / (union_area + 1e-6)

        keep = []
        while len(matches) > 0:
            current = matches[0]
            keep.append(current)
            matches = matches[1:]

            # 杩囨护涓庡綋鍓嶅尮閰嶉噸鍙犵殑缁撴灉
            filtered = []
            for match in matches:
                iou = compute_iou(current.outer_quad, match.outer_quad)
                if iou < iou_threshold:
                    filtered.append(match)
            matches = filtered

        return keep

    def detect(
        self,
        image: np.ndarray,
        merged_lines: List[np.ndarray],
        score_threshold: float = 0.5
    ) -> List[ParkingMatch]:
        """Detect parking-slot quads from merged line segments."""
        h, w = image.shape[:2]

        # Normalize and clip line coordinates, because upstream line-merging may
        # extrapolate endpoints outside the image.
        normalized_lines: List[np.ndarray] = []
        for line in merged_lines:
            ln = np.asarray(line, dtype=np.float32).reshape(-1)[:4]
            if not np.all(np.isfinite(ln)):
                continue
            ln[0] = float(np.clip(ln[0], 0.0, w - 1.0))
            ln[2] = float(np.clip(ln[2], 0.0, w - 1.0))
            ln[1] = float(np.clip(ln[1], 0.0, h - 1.0))
            ln[3] = float(np.clip(ln[3], 0.0, h - 1.0))
            if self._line_length(ln) < 8.0:
                continue
            normalized_lines.append(ln)

        # 1) Adaptive prefilter to balance precision/recall across scenes.
        lines = self._filter_lines_multistage(normalized_lines, (h, w))
        lines = [self._trim_line_for_ground_perspective(ln, (h, w)) for ln in lines]
        lines = sorted(
            lines,
            key=lambda ln: self._line_length(np.asarray(ln, dtype=np.float32).reshape(-1)[:4]),
            reverse=True
        )[:24]
        print(f"Template line prefilter: {len(merged_lines)} -> {len(lines)}")

        if len(lines) < 2:
            return []

        floor_axis_angle = self._estimate_floor_axis_angle(normalized_lines, (h, w))
        floor_split_ratio, floor_split_conf = self._estimate_floor_split(image)
        vp_source = lines if len(lines) >= 2 else normalized_lines
        vanishing_point, vp_conf = self._estimate_vanishing_point(vp_source, (h, w))
        if vanishing_point is not None:
            print(
                f"Estimated VP: ({float(vanishing_point[0]):.1f}, "
                f"{float(vanishing_point[1]):.1f}), conf={vp_conf:.2f}"
            )
        else:
            print("Estimated VP: None")

        # Use both converging and parallel side-line hypotheses.
        min_dist = max(24.0, w * 0.06)
        max_dist = max(140.0, w * 0.92)
        converging_pairs = self.find_converging_line_pairs(
            lines,
            image_shape=(h, w),
            near_span_range=(min_dist, max_dist),
        )

        parallel_pairs = self.find_parallel_line_pairs(
            lines,
            angle_threshold=8.0,
            distance_range=(min_dist, max_dist),
        )

        line_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        line_pairs.extend(converging_pairs)
        line_pairs.extend(parallel_pairs)

        # Relaxed retry for sparse scenes.
        if len(line_pairs) == 0:
            converging_pairs = self.find_converging_line_pairs(
                lines,
                image_shape=(h, w),
                near_span_range=(max(16.0, w * 0.03), max(80.0, w * 0.98)),
            )
            parallel_pairs = self.find_parallel_line_pairs(
                lines,
                angle_threshold=12.0,
                distance_range=(max(12.0, w * 0.02), max(80.0, w * 0.98)),
            )
            line_pairs.extend(converging_pairs)
            line_pairs.extend(parallel_pairs)
        print(
            f"Found {len(converging_pairs)} converging pairs, "
            f"{len(parallel_pairs)} parallel pairs, using {len(line_pairs)}"
        )
        line_pairs = self._dedupe_line_pairs(line_pairs)

        if len(line_pairs) == 0 and len(lines) >= 2:
            # Last-resort combinational pairing; downstream priors/templates will prune.
            max_seed = min(len(lines), 10)
            line_pairs = []
            for i in range(max_seed):
                for j in range(i + 1, max_seed):
                    line_pairs.append((lines[i], lines[j]))

        # Rank and cap pair pool to keep speed and precision stable.
        if len(line_pairs) > 240:
            def pair_quality(pair: Tuple[np.ndarray, np.ndarray]) -> float:
                l1, l2 = pair
                near1, far1 = self._line_near_far(l1)
                near2, far2 = self._line_near_far(l2)

                near_span = abs(float(near1[0] - near2[0]))
                far_span = abs(float(far1[0] - far2[0]))
                depth = max(0.0, (float(near1[1] + near2[1]) - float(far1[1] + far2[1])) * 0.5)
                len_score = np.clip(
                    (self._line_length(l1) + self._line_length(l2)) / max(float(w) * 1.5, 1.0),
                    0.0,
                    1.0,
                )
                span_score = np.clip((near_span - min_dist) / max(max_dist - min_dist, 1.0), 0.0, 1.0)
                depth_score = np.clip(depth / max(float(h) * 0.45, 1.0), 0.0, 1.0)
                perspective = far_span / max(near_span, 1e-6)
                persp_score = max(0.0, 1.0 - abs(perspective - 0.60) / 0.75)
                return float(0.40 * len_score + 0.20 * span_score + 0.20 * depth_score + 0.20 * persp_score)

            line_pairs = sorted(line_pairs, key=pair_quality, reverse=True)[:240]

        # Drop pairs whose far endpoints sit too high (ceiling/wall clutter).
        if len(line_pairs) > 0:
            def pair_is_ground(pair: Tuple[np.ndarray, np.ndarray]) -> bool:
                l1, l2 = pair
                near1, far1 = self._line_near_far(l1)
                near2, far2 = self._line_near_far(l2)

                # Side lines should not be almost horizontal.
                a1 = self._line_angle_abs(l1)
                a2 = self._line_angle_abs(l2)
                if max(a1, a2) < 10.0:
                    return False

                # In-ground slots usually have far endpoints above near endpoints,
                # but not near the image ceiling.
                far_min_y = min(float(far1[1]), float(far2[1]))
                near_max_y = max(float(near1[1]), float(near2[1]))
                if far_min_y < h * 0.18 and near_max_y > h * 0.55:
                    return False
                return True

            line_pairs = [p for p in line_pairs if pair_is_ground(p)]

        # If pairing pool is too small, enrich with broad combinations.
        if len(line_pairs) > 0 and len(line_pairs) < 8 and len(lines) >= 3:
            max_seed = min(len(lines), 12)
            extra_pairs = []
            for i in range(max_seed):
                for j in range(i + 1, max_seed):
                    extra_pairs.append((lines[i], lines[j]))
            line_pairs.extend(extra_pairs)
            line_pairs = self._dedupe_line_pairs(line_pairs)

        if len(line_pairs) == 0:
            return []

        def edge_pair_to_candidate(l1: np.ndarray, l2: np.ndarray):
            p = np.asarray(l1, dtype=np.float32).reshape(-1)[:4]
            q = np.asarray(l2, dtype=np.float32).reshape(-1)[:4]

            p_pts = [np.array([p[0], p[1]], dtype=np.float32), np.array([p[2], p[3]], dtype=np.float32)]
            q_pts = [np.array([q[0], q[1]], dtype=np.float32), np.array([q[2], q[3]], dtype=np.float32)]
            p_pts.sort(key=lambda t: t[0])  # left -> right
            q_pts.sort(key=lambda t: t[0])

            p_y = 0.5 * (p_pts[0][1] + p_pts[1][1])
            q_y = 0.5 * (q_pts[0][1] + q_pts[1][1])
            if p_y >= q_y:
                near_l, near_r = p_pts
                far_l, far_r = q_pts
            else:
                near_l, near_r = q_pts
                far_l, far_r = p_pts

            # Trim both edges to their shared x-overlap to avoid over-wide spans.
            near_x_l, near_x_r = float(near_l[0]), float(near_r[0])
            far_x_l, far_x_r = float(far_l[0]), float(far_r[0])
            overlap_l = max(near_x_l, far_x_l)
            overlap_r = min(near_x_r, far_x_r)
            near_w = max(near_x_r - near_x_l, 1e-6)
            far_w = max(far_x_r - far_x_l, 1e-6)
            overlap = overlap_r - overlap_l
            if overlap < max(18.0, w * 0.03):
                return None
            overlap_ratio = overlap / max(min(near_w, far_w), 1e-6)
            if overlap_ratio < 0.32:
                return None

            def interp_by_x(a: np.ndarray, b: np.ndarray, x: float) -> np.ndarray:
                dx = float(b[0] - a[0])
                if abs(dx) < 1e-6:
                    return np.array([x, 0.5 * float(a[1] + b[1])], dtype=np.float32)
                t = float((x - float(a[0])) / dx)
                y = float(a[1] + t * float(b[1] - a[1]))
                return np.array([x, y], dtype=np.float32)

            near_a = near_l.copy()
            near_b = near_r.copy()
            far_a = far_l.copy()
            far_b = far_r.copy()
            near_l = interp_by_x(near_a, near_b, overlap_l)
            near_r = interp_by_x(near_a, near_b, overlap_r)
            far_l = interp_by_x(far_a, far_b, overlap_l)
            far_r = interp_by_x(far_a, far_b, overlap_r)

            outer = np.array([near_l, near_r, far_r, far_l], dtype=np.float32)
            if abs(cv2.contourArea(outer)) < 40:
                return None

            center = np.mean(outer, axis=0)
            inner = []
            for pt in outer:
                v = pt - center
                n = np.linalg.norm(v)
                if n > 1e-6:
                    inner.append(pt - v / n * 12.0)
                else:
                    inner.append(pt.copy())
            inner = np.array(inner, dtype=np.float32)
            return outer, inner

        parking_pair_line_pairs = []
        parking_pair_candidates: List[Tuple[np.ndarray, np.ndarray]] = []
        try:
            from fst.parking_line_detector import ParkingLineDetector

            pd = ParkingLineDetector(
                min_line_length=max(35, int(min(h, w) * 0.06)),
                parallel_angle_threshold=7.0,
                min_distance=max(4, int(w * 0.02)),
                max_distance=max(60, int(w * 0.28)),
            )
            parking_pair_line_pairs = pd.detect_parking_lines(image)
            for l1, l2, _ in parking_pair_line_pairs[:24]:
                cand = edge_pair_to_candidate(l1[0], l2[0])
                if cand is not None:
                    parking_pair_candidates.append(cand)
        except Exception:
            parking_pair_line_pairs = []
            parking_pair_candidates = []

        # 2) Build candidate trapezoids.
        candidates = self.build_candidate_quads(line_pairs)
        if len(parking_pair_candidates) > 0:
            candidates.extend(parking_pair_candidates)
        print(
            f"Built {len(candidates)} candidate quads "
            f"(side={len(candidates) - len(parking_pair_candidates)}, pld={len(parking_pair_candidates)})"
        )

        # 3) Prior ranking + template matching.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 125), (180, 92, 255))
        yellow_mask = cv2.inRange(hsv, (12, 45, 90), (45, 255, 255))
        paint_mask = cv2.bitwise_or(white_mask, yellow_mask)
        paint_mask = cv2.morphologyEx(
            paint_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        line_map = np.zeros((h, w), dtype=np.uint8)
        for ln in normalized_lines:
            x1, y1, x2, y2 = [int(round(v)) for v in ln[:4]]
            cv2.line(line_map, (x1, y1), (x2, y2), 255, 2, lineType=cv2.LINE_AA)

        candidate_with_prior = []
        for outer_quad, inner_quad in candidates:
            prior = self._candidate_prior_score(
                outer_quad,
                (h, w),
                edges,
                line_map=line_map,
                floor_axis_angle=floor_axis_angle,
                paint_mask=paint_mask,
                floor_split_ratio=floor_split_ratio,
                floor_split_conf=floor_split_conf,
            )
            vp_score = self._quad_vanishing_convergence_score(outer_quad, vanishing_point, (h, w))
            parallel_score = self._quad_top_bottom_parallel_score(outer_quad)
            intersection_count = self._count_quad_line_intersections(outer_quad, normalized_lines, (h, w))
            intersection_score = 1.0 / (1.0 + 0.22 * float(intersection_count))

            # Unified geometric rule score requested by user:
            # (1) side convergence to VP, (2) top/bottom near parallel,
            # (3) lower merged-line crossings preferred.
            geom_rule_score = 0.50 * vp_score + 0.28 * parallel_score + 0.22 * intersection_score
            combined_prior = 0.72 * prior + 0.28 * geom_rule_score
            if vanishing_point is not None and vp_conf >= 0.30 and vp_score < 0.18:
                combined_prior *= 0.60

            candidate_with_prior.append(
                (
                    combined_prior,
                    prior,
                    vp_score,
                    parallel_score,
                    intersection_count,
                    outer_quad,
                    inner_quad,
                )
            )

        candidate_with_prior.sort(key=lambda x: (x[0], -x[4]), reverse=True)
        candidate_with_prior = candidate_with_prior[:60]

        scored_matches = []
        for (
            combined_prior,
            prior,
            vp_score,
            parallel_score,
            intersection_count,
            outer_quad,
            inner_quad,
        ) in candidate_with_prior:
            matches = self.matcher.match_template(outer_quad, inner_quad, top_k=1)
            if len(matches) == 0:
                continue

            best = matches[0]
            geom_bonus = 0.56 * vp_score + 0.44 * parallel_score
            fused = 0.52 * best.score + 0.30 * prior + 0.18 * geom_bonus
            wedge_penalty = self._wedge_false_positive_penalty(best.outer_quad, (h, w))
            intersection_penalty = 1.0 / (1.0 + 0.12 * max(0.0, float(intersection_count) - 2.0))
            if prior <= 1e-6:
                fused *= 0.70
            elif prior < 0.03:
                fused *= 0.84
            if vanishing_point is not None and vp_conf >= 0.45 and vp_score < 0.15:
                fused *= 0.70

            best.score = fused * wedge_penalty * intersection_penalty
            scored_matches.append((best, prior, intersection_count, vp_score, combined_prior))

        strict_prior_floor = 0.03
        relaxed_prior_floor = 0.015

        all_records = [
            r for r in scored_matches
            if r[0].score >= score_threshold and r[1] >= strict_prior_floor
        ]
        if len(all_records) == 0:
            # Adaptive threshold fallback for difficult images.
            relaxed_cut = max(0.40, score_threshold - 0.28)
            all_records = [
                r for r in scored_matches
                if r[0].score >= relaxed_cut and r[1] >= relaxed_prior_floor
            ]
            if len(all_records) == 0 and len(scored_matches) > 0:
                # Final safety net: prefer prior-supported top candidate.
                prior_supported = [r for r in scored_matches if r[1] >= 0.01]
                if len(prior_supported) > 0:
                    top_r = max(prior_supported, key=lambda x: x[0].score)
                    if top_r[0].score >= 0.46:
                        all_records = [top_r]
                else:
                    top_r = max(scored_matches, key=lambda x: x[0].score)
                    if top_r[0].score >= 0.55:
                        all_records = [top_r]

        all_matches = [r[0] for r in all_records]

        if len(all_matches) == 0:
            # Fallback path: reuse robust parallel-line pairs from ParkingLineDetector.
            try:
                from fst.parking_line_detector import ParkingLineDetector
                line_pairs = parking_pair_line_pairs
                if len(line_pairs) == 0:
                    pd = ParkingLineDetector(
                        min_line_length=max(35, int(min(h, w) * 0.06)),
                        parallel_angle_threshold=7.0,
                        min_distance=max(4, int(w * 0.02)),
                        max_distance=max(60, int(w * 0.28)),
                    )
                    line_pairs = pd.detect_parking_lines(image)

                fallback_scored = []
                best_prior_only = None
                best_prior_only_score = 0.0
                best_template_only = None
                best_template_only_score = 0.0
                for l1, l2, _ in line_pairs[:20]:
                    cand = edge_pair_to_candidate(l1[0], l2[0])
                    if cand is None:
                        continue
                    outer_quad, inner_quad = cand
                    prior = self._candidate_prior_score(
                        outer_quad,
                        (h, w),
                        edges,
                        line_map=line_map,
                        floor_axis_angle=floor_axis_angle,
                        paint_mask=paint_mask,
                        floor_split_ratio=floor_split_ratio,
                        floor_split_conf=floor_split_conf,
                    )
                    ms = self.matcher.match_template(outer_quad, inner_quad, top_k=1)
                    if len(ms) == 0:
                        continue
                    m = ms[0]
                    vp_score = self._quad_vanishing_convergence_score(outer_quad, vanishing_point, (h, w))
                    parallel_score = self._quad_top_bottom_parallel_score(outer_quad)
                    intersection_count = self._count_quad_line_intersections(outer_quad, normalized_lines, (h, w))
                    geom_bonus = 0.56 * vp_score + 0.44 * parallel_score
                    intersection_penalty = 1.0 / (1.0 + 0.12 * max(0.0, float(intersection_count) - 2.0))

                    m.score = (0.38 * m.score + 0.38 * prior + 0.24 * geom_bonus) * intersection_penalty
                    if m.score >= 0.48 and prior >= 0.02:
                        fallback_scored.append(m)

                    prior_rank = float(prior * intersection_penalty * (0.62 + 0.38 * geom_bonus))
                    if prior_rank > best_prior_only_score:
                        best_prior_only_score = prior_rank
                        best_prior_only = (outer_quad.copy(), inner_quad.copy())

                    # Template-only backup for scenes where geometric priors are too strict.
                    tpl = float(ms[0].score)
                    area_ratio = abs(float(cv2.contourArea(outer_quad))) / max(float(h * w), 1.0)
                    top_y_ratio = float((outer_quad[2, 1] + outer_quad[3, 1]) * 0.5) / max(float(h), 1.0)
                    bottom_y_ratio = float((outer_quad[0, 1] + outer_quad[1, 1]) * 0.5) / max(float(h), 1.0)
                    vp_score_tpl = self._quad_vanishing_convergence_score(outer_quad, vanishing_point, (h, w))
                    parallel_score_tpl = self._quad_top_bottom_parallel_score(outer_quad)
                    inter_tpl = self._count_quad_line_intersections(outer_quad, normalized_lines, (h, w))
                    tpl_adjust = (
                        tpl
                        * (0.64 + 0.20 * vp_score_tpl + 0.16 * parallel_score_tpl)
                        * (1.0 / (1.0 + 0.10 * max(0.0, float(inter_tpl) - 2.0)))
                    )
                    if (
                        tpl_adjust > best_template_only_score
                        and 0.035 <= area_ratio <= 0.60
                        and bottom_y_ratio >= 0.40
                        and top_y_ratio >= 0.10
                        and vp_score_tpl >= 0.20
                    ):
                        best_template_only_score = float(tpl_adjust)
                        m_tpl = ParkingMatch(
                            outer_quad=ms[0].outer_quad.copy(),
                            inner_quad=ms[0].inner_quad.copy(),
                            score=float(tpl_adjust),
                            template_id=ms[0].template_id,
                        )
                        best_template_only = m_tpl

                if len(fallback_scored) > 0:
                    all_matches = fallback_scored
                elif best_prior_only is not None and best_prior_only_score >= 0.16:
                    outer_quad, inner_quad = best_prior_only
                    prior_only_match = ParkingMatch(
                        outer_quad=outer_quad,
                        inner_quad=inner_quad,
                        score=float(0.42 + 0.50 * min(best_prior_only_score, 1.0)),
                        template_id=-1,
                    )
                    all_matches = [prior_only_match]
                elif best_template_only is not None and best_template_only_score >= 0.52:
                    best_template_only.score = float(max(0.50, min(0.86, best_template_only_score)))
                    all_matches = [best_template_only]
            except Exception:
                pass

        if len(all_matches) == 0:
            # Final rectangle fallback from paint-mask percentiles.
            ys, xs = np.where((line_map > 0) & (np.indices((h, w))[0] >= int(h * 0.28)))
            if len(xs) >= 140:
                x1 = float(np.percentile(xs, 25))
                x2 = float(np.percentile(xs, 75))
                y1 = float(np.percentile(ys, 20))
                y2 = float(np.percentile(ys, 88))
                if (x2 - x1) > w * 0.14 and (y2 - y1) > h * 0.08:
                    outer_quad = np.array(
                        [[x1, y2], [x2, y2], [x2, y1], [x1, y1]],
                        dtype=np.float32,
                    )
                    center = np.mean(outer_quad, axis=0)
                    inner_pts = []
                    for pt in outer_quad:
                        v = pt - center
                        n = np.linalg.norm(v)
                        if n > 1e-6:
                            inner_pts.append(pt - v / n * 12.0)
                        else:
                            inner_pts.append(pt.copy())
                    inner_quad = np.asarray(inner_pts, dtype=np.float32)
                    all_matches = [
                        ParkingMatch(
                            outer_quad=outer_quad,
                            inner_quad=inner_quad,
                            score=0.50,
                            template_id=-2,
                        )
                    ]

        # User rule: among valid quads, prefer the one with fewest crossings
        # against merged yellow lines in "merged_lines" mode.
        if len(all_matches) > 1:
            ranked = []
            for m in all_matches:
                intersections = self._count_quad_line_intersections(m.outer_quad, normalized_lines, (h, w))
                vp_score = self._quad_vanishing_convergence_score(m.outer_quad, vanishing_point, (h, w))
                para_score = self._quad_top_bottom_parallel_score(m.outer_quad)
                ranked.append((m, intersections, vp_score, para_score))

            min_intersections = min(r[1] for r in ranked)
            filtered = [r for r in ranked if r[1] == min_intersections]
            filtered.sort(
                key=lambda r: (r[0].score + 0.10 * r[2] + 0.06 * r[3]),
                reverse=True,
            )
            all_matches = [r[0] for r in filtered]
            print(f"Min-crossing filter: min={min_intersections}, kept={len(all_matches)}")

        print(f"Matched {len(all_matches)} candidates")

        # 4) NMS + top-k output.
        final_matches = self.non_maximum_suppression(all_matches, iou_threshold=0.25)
        if len(final_matches) > 1:
            ranked = []
            for m in final_matches:
                intersections = self._count_quad_line_intersections(m.outer_quad, normalized_lines, (h, w))
                ranked.append((m, intersections))
            min_intersections = min(r[1] for r in ranked)
            final_matches = [r[0] for r in ranked if r[1] == min_intersections]
        final_matches = sorted(final_matches, key=lambda m: m.score, reverse=True)[:1]
        if len(final_matches) > 0:
            final_intersections = self._count_quad_line_intersections(final_matches[0].outer_quad, normalized_lines, (h, w))
            print(f"Final quad crossings={final_intersections}")
        print(f"Kept {len(final_matches)} after NMS")
        return final_matches

    def visualize(
        self,
        image: np.ndarray,
        matches: List[ParkingMatch],
        output_path: str | None = None
    ) -> np.ndarray:
        """Visualize detection results."""
        vis_img = image.copy()

        for match in matches:
            outer_pts = match.outer_quad.astype(np.int32)
            cv2.polylines(vis_img, [outer_pts], True, (0, 255, 0), 2)

            inner_pts = match.inner_quad.astype(np.int32)
            cv2.polylines(vis_img, [inner_pts], True, (255, 0, 0), 2)

            # 鏄剧ず鍒嗘暟
            center = np.mean(match.outer_quad, axis=0).astype(np.int32)
            cv2.putText(
                vis_img,
                f"{match.score:.2f}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"Visualization saved: {output_path}")

        return vis_img


if __name__ == "__main__":
    # 娴嬭瘯妯℃澘鐢熸垚
    generator = TemplateGenerator()
    templates = generator.generate_template_library()

    print(f"\n妯℃澘绀轰緥:")
    print(f"绗?涓ā鏉? width={templates[0].width}, length={templates[0].length}, "
          f"perspective={templates[0].perspective}, rotation={templates[0].rotation}")
    print(f"澶栧眰鍥涜竟褰?\n{templates[0].outer_quad}")
    print(f"鍐呭眰鍥涜竟褰?\n{templates[0].inner_quad}")
